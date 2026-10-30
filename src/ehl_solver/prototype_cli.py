"""Reproducible CPU point-contact prototype: python -m ehl_solver.prototype_cli."""

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import scipy

from ehl_solver.property_coupled import PropertyCoupledSolution
from ehl_solver.property_load_newton import PropertyLoadNewtonError, _solve_property_load_state
from ehl_solver.property_warm import WarmPropertyRoot
from ehl_solver.prototype import (
    PointContactCase,
    audit_history,
    audit_root,
    hertz_guess,
    transfer_guess,
)


def clean_json(value):
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return {"nonfinite_diagnostic": repr(value)}
    return value


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_run(directory):
    """Re-audit an accepted prototype result before using its fields as a seed."""
    directory = Path(directory)
    report = json.loads((directory / "summary.json").read_text())
    if report["status"] != "accepted_numerical_root":
        raise ValueError("Warm start requires an accepted numerical root")
    case = PointContactCase(**report["case"])
    path = directory / "solution.npz"
    if digest(path) != report["state_sha256"]:
        raise ValueError("Seed state checksum does not match report")
    with np.load(path, allow_pickle=False) as a:
        problem = case.problem()
        np.testing.assert_array_equal(a["x_m"], problem.x_m)
        np.testing.assert_array_equal(a["y_m"], problem.y_m)
        solution = PropertyCoupledSolution(
            a["pressure_pa"],
            a["theta"],
            a["film_m"],
            a["displacement_m"],
            problem.load_n(a["pressure_pa"]),
            {},
            (),
            1.0,
        )
        root = WarmPropertyRoot(float(a["offset_m"]), solution)
    audit = audit_root(case, root)
    if not audit["numerically_accepted"]:
        raise ValueError("Saved seed failed independent numerical audit")
    return case, root, report


def save_state(path, system, state, root):
    s = root.solution
    np.savez_compressed(
        path,
        state_dimensionless=state,
        x_m=system.problem.x_m,
        y_m=system.problem.y_m,
        pressure_pa=s.pressure_pa,
        theta=s.fractional_film_content,
        film_m=s.film_thickness_m,
        displacement_m=s.elastic_displacement_m,
        offset_m=root.central_film_offset_m,
    )


def plot_result(case, root, iterations, directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = root.solution
    p = case.problem()
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    mx = abs(p.x_m) <= 1.5 * case.hertz.radius_m
    my = abs(p.y_m) <= 1.5 * case.hertz.radius_m
    extent = [p.x_m[mx][0] * 1e3, p.x_m[mx][-1] * 1e3, p.y_m[my][0] * 1e3, p.y_m[my][-1] * 1e3]
    cap_nm = 2 * case.correlation().central_film_m * 1e9
    for ax, field, scale, title in [
        (axes[0, 0], s.pressure_pa, 1e-9, "Pressure [GPa], unfiltered"),
        (axes[0, 1], s.film_thickness_m, 1e9, f"Film [nm], colour capped at {cap_nm:.0f}"),
    ]:
        artist = ax.imshow(
            field[np.ix_(my, mx)] * scale,
            extent=extent,
            origin="lower",
            interpolation="nearest",
            vmax=cap_nm if scale == 1e9 else None,
        )
        ax.set(xlabel="x [mm], entrainment +x", ylabel="y [mm]", title=title)
        fig.colorbar(artist, ax=ax)
    axes[1, 0].plot(p.x_m * 1e3, s.pressure_pa[case.ny // 2] / 1e9)
    axes[1, 0].set(
        xlabel="x [mm]",
        ylabel="Centreline pressure [GPa]",
        xlim=(-1.5 * case.hertz.radius_m * 1e3, 1.5 * case.hertz.radius_m * 1e3),
    )
    for key, label in [
        ("mass_maximum", "Scaled mass residual"),
        ("relative_global_balance", "Global mass balance"),
    ]:
        axes[1, 1].semilogy(
            [a["iteration"] for a in iterations],
            [max(a["audit"][key], 1e-18) for a in iterations],
            ".-",
            label=label,
        )
    axes[1, 1].set(xlabel="Newton correction / state", ylabel="Residual measure")
    axes[1, 1].legend(fontsize=8)
    fig.suptitle(f"{case.load_n:g} N, {case.ny}x{case.nx}, {case.transport}: numerical prototype")
    fig.savefig(directory / "fields.png", dpi=160)
    plt.close(fig)


def write_markdown(directory, report):
    case = report["case"]
    audit = report["independent_audit"]
    hd = audit["hamrock_dowson"]
    rows = []
    for label, field, delta in (
        ("Minimum", "minimum_film_m", "minimum_relative_correlation_difference"),
        ("Central", "central_film_m", "central_relative_correlation_difference"),
    ):
        solved, reference = audit[field] * 1e9, hd[field] * 1e9
        rows.append(f"| {label} | {solved:.5f} | {reference:.5f} | {audit[delta]:+.3%} |")
    table_rows = "\n".join(rows)
    local = audit["mass"]["relative_local_balance"]
    global_balance = audit["mass"]["relative_global_balance"]
    text = f"""# Circular point-contact prototype result

Status: **{report["status"]}**. Load {case["load_n"]:g} N;
mesh (ny,nx)=({case["ny"]},{case["nx"]}); transport {case["transport"]}.

| Film quantity | Solved [nm] | Hamrock–Dowson [nm] | Relative difference |
| --- | ---: | ---: | ---: |
{table_rows}

Force: {audit["force_n"]:.15g} N; relative force error {audit["force_relative_error"]:.3e}.
Scaled maximum mass residual: {audit["mass"]["mass_maximum"]:.3e}.
Relative local/global mass balance: {local:.3e} / {global_balance:.3e}.

This is a steady, isothermal, fully flooded, smooth circular Newtonian-lubricant
model with declared pressure-dependent fluid laws. Temperature, roughness,
thermal effects and shear thinning are not modeled. No particular real
lubricant is experimentally validated.

A single converged run does not establish mesh/domain independence or a
physical error bound. Correlation agreement is reported separately from
equation/force/feasibility acceptance. Inspect the unfiltered pressure and
perform grid/domain studies before engineering use. Donor capacity has
first-order numerical diffusion; arithmetic capacity remains a reference.

Inputs: `case.json`; exact states and SI fields: `solution.npz`;
full histories/audits/source hashes: `summary.json`;
package sources: `source/ehl_solver/`.
"""
    (directory / "report.md").write_text(text)


def run_case(case, directory, *, seed_directory=None, maximum_iterations=30, plots=True):
    if type(maximum_iterations) is not int or not 0 <= maximum_iterations <= 30:
        raise ValueError("Correction budget must be in 0..30")
    directory = Path(directory)
    system = case.system()
    if seed_directory is None:
        state, initialization = hertz_guess(case, system)
    else:
        old_case, old_root, _old_report = load_run(seed_directory)
        state, initialization = transfer_guess(old_case, case, old_root, system)
        initialization.update(
            source=str(seed_directory), source_sha256=digest(Path(seed_directory) / "solution.npz")
        )
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "case.json").write_text(json.dumps(asdict(case), indent=2))
    try:
        revision = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
            ).stdout.strip()
            or None
        )
    except OSError:
        revision = None
    report = dict(
        status="running",
        case=asdict(case),
        initialization=initialization,
        coordinates="(ny,nx), x follows mean entrainment; fields SI",
        nondimensionalization=dict(
            length_m=system.problem.length_scale_m,
            film_m=system.problem.film_scale_m,
            pressure_pa=system.problem.pressure_scale_pa,
            reynolds_parameter=system.problem.reynolds_parameter,
            unknowns="P=p/p_ref,D=1-theta,h0/h_ref; C-order interior",
        ),
        runtime=dict(
            python=sys.version,
            numpy=np.__version__,
            scipy=scipy.__version__,
            platform=platform.platform(),
            git_revision=revision,
        ),
        maximum_corrections=maximum_iterations,
        maximum_krylov_cycles=10,
        iterations=[],
        physical_validation_complete=False,
        spatial_validation="not established by a single run",
    )
    package = Path(__file__).parent
    source = directory / "source" / "ehl_solver"
    source.mkdir(parents=True)
    report["source_sha256"] = {}
    for path in sorted(package.glob("*.py")):
        shutil.copyfile(path, source / path.name)
        report["source_sha256"][path.name] = digest(path)

    def write():
        (directory / "summary.json").write_text(
            json.dumps(clean_json(report), indent=2, allow_nan=False)
        )

    latest = {}

    def observe(iteration, raw, root):
        save_state(directory / f"state_{iteration:03d}.npz", system, raw, root)
        latest["state"] = raw.copy()
        report["iterations"].append(
            dict(
                iteration=iteration,
                load_n=root.solution.load_n,
                audit=root.solution.audit,
                offset_m=root.central_film_offset_m,
            )
        )
        write()

    write()
    try:
        root = _solve_property_load_state(
            system,
            state,
            maximum_iterations,
            10,
            1e-10,
            True,
            active_lookahead=True,
            content_bound_pivots=True,
            state_observer=observe,
        )
        report["status"] = "accepted_numerical_root"
        history = root.solution.history
    except PropertyLoadNewtonError as exc:
        root, history = exc.last_root, exc.history
        report.update(
            status="rejected", failure_reason=str(exc), failure_diagnostics=exc.diagnostics
        )
    if root is None:
        write()
        return report
    report["history"] = history
    report["independent_audit"] = audit_root(case, root)
    if report["independent_audit"]["numerically_accepted"] != (
        report["status"] == "accepted_numerical_root"
    ):
        raise RuntimeError("Independent final acceptance disagrees with nonlinear solver")
    report["history_audit"] = audit_history(history)
    save_state(directory / "solution.npz", system, latest["state"], root)
    report["state_sha256"] = digest(directory / "solution.npz")
    write()
    write_markdown(directory, report)
    if plots:
        plot_result(case, root, report["iterations"], directory)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, help="JSON configuration with SI physical quantities")
    parser.add_argument("--output", required=True, type=Path, help="New result directory")
    parser.add_argument(
        "--seed", type=Path, help="Accepted prototype output for spatial warm start"
    )
    parser.add_argument("--max-iterations", type=int, default=30)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    try:
        case = (
            PointContactCase(**json.loads(args.case.read_text()))
            if args.case
            else PointContactCase()
        )
        report = run_case(
            case,
            args.output,
            seed_directory=args.seed,
            maximum_iterations=args.max_iterations,
            plots=not args.no_plots,
        )
    except (ValueError, TypeError, FileExistsError, FileNotFoundError) as exc:
        parser.error(str(exc))
    print(f"{report['status']}: {case.load_n:g} N, (ny,nx)=({case.ny},{case.nx}), {case.transport}")
    if "independent_audit" in report:
        audit = report["independent_audit"]
        print(
            f"Film min/central: {audit['minimum_film_m'] * 1e9:.3f} / "
            f"{audit['central_film_m'] * 1e9:.3f} nm"
        )
        print(
            f"Independent force: {audit['force_n']:.12g} N; "
            f"mass: {audit['mass']['mass_maximum']:.3e}"
        )
    print(f"Report: {args.output / 'report.md'}")
    return 0 if report["status"] == "accepted_numerical_root" else 1


if __name__ == "__main__":
    raise SystemExit(main())
