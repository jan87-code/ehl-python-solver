"""Reproducible mesh/domain/boundary/elasticity error budget, CPU only.

Constant eta=0.04 Pa s, Ue=0.001 m/s, h0=0.1 um, Rx=10 mm, Ry=20 mm.
L=sqrt(2 Rx h0), Ly=sqrt(2 Ry h0), p_ref=12 eta Ue L/h0^2, Lambda=1.
Graded axes retain an exactly uniform core and fixed geometric outer cells.
Requested extents snap outward to existing nodes for nested domain studies.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid

from ehl_solver.cavitating_coupled import CavitatingElasticityProblem, solve_cavitating_elasticity
from ehl_solver.cavitating_newton import CavitatingNewtonError, solve_cavitating_newton
from ehl_solver.elasticity_audit import direct_midpoint_displacement
from ehl_solver.rectilinear_cavitation import flux_audit, solve_rectilinear_cavitation
from examples.run_scientific_validation import provenance
from examples.scientific_figures import extrema, figure, save

H0, RX, RY, ETA, SPEED = 1e-7, 0.01, 0.02, 0.04, 0.001
L, LY = np.sqrt(2 * RX * H0), np.sqrt(2 * RY * H0)
P_REF = 12 * ETA * SPEED * L / H0**2
CRITERIA = {
    "relative_load_film_pressure": 0.01,
    "mass_maximum": 1e-8,
    "relative_local_balance": 1e-10,
    "relative_global_balance": 1e-10,
    "complementarity": 1e-10,
    "front_tolerance_core_cells": 2,
    "minimum_consecutive_domain_changes": 2,
}


def graded_axis(upstream, downstream, step, *, snap=True, growth=1.1):
    """Dimensionless X-like coordinates; core [-2,2], fixed outer increments."""
    if (
        not all(np.isfinite(v) and v > 0 for v in (upstream, downstream, step))
        or min(upstream, downstream) < 2
        or not 1 < growth <= 1.2
        or not np.isclose(2 / step, round(2 / step))
        or 2 / step > 512
    ):
        raise ValueError("invalid bounded graded-axis parameters")

    def half(extent):
        nodes = list(np.arange(round(2 / step) + 1) * step)
        increment = step
        while nodes[-1] < extent:
            if len(nodes) >= 512:
                raise ValueError("graded axis exceeds node budget")
            new = nodes[-1] + increment
            nodes.append(new if snap or new <= extent else extent)
            increment *= growth
        return np.array(nodes)

    west, east = half(upstream), half(downstream)
    return np.r_[-west[:0:-1], east]


def fronts(x, theta):
    """Centerline front brackets, not subcell front estimates; threshold explicit."""
    dry = theta[theta.shape[0] // 2] < 1 - 1e-8
    return [
        {
            "kind": "reformation" if dry[i] else "rupture",
            "bracket_m": [float(x[i]), float(x[i + 1])],
        }
        for i in np.flatnonzero(np.diff(dry))
    ]


def accepted(audit, fields):
    return bool(
        all(
            audit[key] <= CRITERIA[key]
            for key in ("mass_maximum", "relative_local_balance", "relative_global_balance")
        )
        and audit["maximum_complementarity"] <= CRITERIA["complementarity"]
        and fields["pressure_pa"].min() >= -1e-10 * P_REF
        and fields["theta"].min() >= -1e-10
        and fields["theta"].max() <= 1 + 1e-10
        and np.max(abs(fields["pressure_pa"][[0, -1]])) == 0
        and np.max(abs(fields["pressure_pa"][:, [0, -1]])) == 0
        and np.max(abs(fields["theta"][1:-1, 0] - 1)) == 0
    )


def solve_case(name, extents, step, *, family, uniform=False, snap=True):
    up, down, side = extents
    if uniform:
        x = np.linspace(-up, down, round((up + down) / step) + 1) * L
        y = np.linspace(-side, side, round(2 * side / step) + 1) * LY
    else:
        growth = 1.05 if family == "outermesh" else 1.1
        x = graded_axis(up, down, step, snap=snap, growth=growth) * L
        y = graded_axis(side, side, step, snap=snap, growth=growth) * LY
    if len(x) * len(y) > 70000:
        raise ValueError("case exceeds 70000-node budget")
    h = H0 + x[None, :] ** 2 / (2 * RX) + y[:, None] ** 2 / (2 * RY)
    if family == "coupled":
        problem = CavitatingElasticityProblem(x, y, H0, RX, RY, 1e9, L, H0, P_REF, 1)
        try:
            result = solve_cavitating_newton(problem)
            cold_failure, reference_history = None, []
        except CavitatingNewtonError as error:
            cold_failure = error
            # The study's relative flux criterion is stricter than the
            # reference defaults. Resolve the reference more accurately,
            # keeping the original bounded iteration counts.
            reference = solve_cavitating_elasticity(problem, mass_tolerance=1e-10)
            reference_history = [asdict(row) for row in reference.history]
            result = solve_cavitating_newton(
                problem, initial_solution=reference, mass_tolerance=1e-10
            )
        solution = result.solution
        p, theta, h = (
            solution.pressure_pa,
            solution.fractional_film_content,
            solution.film_thickness_m,
        )
        w = solution.elastic_displacement_m
        iterations = len(result.history)
    else:
        result = solve_rectilinear_cavitation(
            x, y, h, ETA, SPEED, length_scale_m=L, film_scale_m=H0, pressure_scale_pa=P_REF
        )
        p, theta = result.pressure_pa, result.fractional_film_content
        w, iterations = np.zeros_like(h), result.iterations
    fields = {
        "x_m": x,
        "y_m": y,
        "pressure_pa": p,
        "theta": theta,
        "film_m": h,
        "displacement_m": w,
    }
    audit = flux_audit(
        x, y, h, p, theta, ETA, SPEED, length_scale_m=L, film_scale_m=H0, pressure_scale_pa=P_REF
    )
    j, i = np.unravel_index(p.argmax(), p.shape)
    targets = [[0.0, 0.0], [float(x[i]), float(y[j])]]
    direct = direct_midpoint_displacement(x, y, p, 1e9, targets)
    entry = {
        "name": name,
        "family": family,
        "requested_extents": extents,
        "core_step": step,
        "shape": list(p.shape),
        "uniform": uniform,
        "x_limits_m": [float(x[0]), float(x[-1])],
        "y_limits_m": [float(y[0]), float(y[-1])],
        "minimum_dx_m": float(np.diff(x).min()),
        "maximum_dx_m": float(np.diff(x).max()),
        "minimum_dy_m": float(np.diff(y).min()),
        "maximum_dy_m": float(np.diff(y).max()),
        "load_n": float(trapezoid(trapezoid(p, x, axis=1), y)),
        "central_film_m": float(h[len(y) // 2, np.argmin(abs(x))]),
        "theta_min": float(theta.min()),
        "theta_max": float(theta.max()),
        "extrema": extrema(fields),
        "fronts": fronts(x, theta),
        "iterations": iterations,
        "audit": audit,
        "accepted": accepted(audit, fields),
        "direct_displacement_targets_m": targets,
        "diagnostic_e_star_pa": 1e9,
        "direct_displacement_m": direct.tolist(),
        "maximum_rigid_slope": float(np.hypot(max(abs(x)) / RX, max(abs(y)) / RY)),
    }
    if uniform:
        entry["direct_fft_displacement_error_m"] = float(
            max(abs(direct - [w[len(y) // 2, np.argmin(abs(x))], w[j, i]]))
        )
        entry["reference_history"] = reference_history
        entry["newton_history"] = [asdict(row) for row in result.history]
        if cold_failure is not None:
            entry["cold_start_failure"] = {
                "reason": str(cold_failure),
                "diagnostics": cold_failure.diagnostics,
                "history": [asdict(row) for row in cold_failure.history],
            }
            failed = cold_failure.last_solution
            if failed is not None:
                fields.update(
                    {
                        "failed_pressure_pa": failed.pressure_pa,
                        "failed_theta": failed.fractional_film_content,
                        "failed_film_m": failed.film_thickness_m,
                        "failed_displacement_m": failed.elastic_displacement_m,
                    }
                )
    # Quantify omitted source pressure on this same solved field, separately
    # from re-solving a smaller domain. These are different error mechanisms.
    crop = (abs(y[:, None]) <= 16 * LY) & (abs(x[None, :]) <= 16 * L)
    cropped_p = np.where(crop, p, 0.0)
    cropped_w = direct_midpoint_displacement(x, y, cropped_p, 1e9, [[0.0, 0.0]])[0]
    entry["omitted_pressure_outside_16"] = {
        "load_n": entry["load_n"] - float(trapezoid(trapezoid(cropped_p, x, axis=1), y)),
        "center_displacement_m": float(direct[0] - cropped_w),
        "center_displacement_fraction": float((direct[0] - cropped_w) / direct[0]),
    }
    return entry, fields


def plots(directory, cases):
    context = (
        "Isoviscous incompressible JFO; eta=.04 Pa s; Ue=.001 m/s; h0=.1 um\n"
        "Rx=.01 m, Ry=.02 m; west theta=1, all pressure edges=0; entrainment +x\n"
        "Rigid unless explicitly coupled (E*=1 GPa); raw states/flow audits in summary.json\n"
        "Formal parabolic far-field study; slopes and failed residual checks retained"
    )
    domains = [c for c in cases if c.get("family") == "all"]
    specifications = (
        ("load_domain", lambda c: c["load_n"], "Load [N]"),
        (
            "minimum_film_domain",
            lambda c: c["extrema"]["minimum_film"]["value_si"] * 1e6,
            "Sampled minimum film [um]",
        ),
        (
            "peak_pressure_domain",
            lambda c: c["extrema"]["maximum_pressure"]["value_si"] / 1e6,
            "Peak pressure [MPa]",
        ),
    )
    for name, value, label in specifications:
        fig, (ax,) = figure(name.replace("_", " "), context)
        for family, title in (
            ("all", "Rigid core dX=.125"),
            ("coarse_all", "Rigid core dX=.25"),
            ("coupled", "Coupled uniform dX=.25"),
        ):
            selected = [
                c
                for c in cases
                if c["family"] == family
                and (
                    family != "coupled"
                    or c["name"] == "coupled_mesh_0.25"
                    or c["name"].startswith("coupled_extent_")
                )
            ]
            ax.plot(
                [abs(c["x_limits_m"][0]) / L for c in selected],
                [value(c) for c in selected],
                "o-",
                label=title,
            )
        rejected = [c for c in domains if not c["accepted"]]
        ax.plot(
            [abs(c["x_limits_m"][0]) / L for c in rejected],
            [value(c) for c in rejected],
            "rx",
            markersize=10,
            label="Residual rejected",
        )
        ax.set(xlabel="Actual upstream extent / L", ylabel=label)
        ax.legend()
        save(fig, directory, name)
    fig, (ax,) = figure("Boundary outward volume flow; negative west denotes inlet", context)
    for k, side in enumerate(("west", "east", "south", "north")):
        ax.plot(
            [abs(c["x_limits_m"][0]) / L for c in domains],
            [c["audit"]["outward_volume_flow_m3_s"][k] for c in domains],
            "o-",
            label=side,
        )
    ax.set(xlabel="Actual upstream extent / L", ylabel="Outward volume flow [m^3/s]")
    ax.legend()
    save(fig, directory, "mass_flow_components")
    fig, axes = figure("Fixed physical domain mesh refinement", context, rows=2)
    mesh = [c for c in cases if c.get("family") == "mesh"]
    axes[0].plot([c["core_step"] for c in mesh], [c["load_n"] for c in mesh], "o-")
    axes[0].set(xlabel="Core dx/L", ylabel="Load [N]")
    axes[1].plot(
        [c["core_step"] for c in mesh],
        [c["extrema"]["maximum_pressure"]["value_si"] / 1e6 for c in mesh],
        "o-",
    )
    axes[1].set(xlabel="Core dx/L", ylabel="Peak pressure [MPa]")
    save(fig, directory, "mesh_refinement")
    for key, factor, label in (
        ("pressure_pa", 1e-6, "Pressure [MPa]"),
        ("film_m", 1e6, "Film thickness [um]"),
        ("theta", 1.0, "Fractional film content"),
        ("displacement_m", 1e6, "Coupled displacement [um]"),
    ):
        fig, (ax,) = figure("Common-coordinate centerline overlays", context)
        for c in cases:
            if c.get("family") not in ("all", "coupled"):
                continue
            with np.load(directory / f"{c['name']}.npz") as f:
                ax.plot(
                    f["x_m"] * 1e3,
                    f[key][len(f["y_m"]) // 2] * factor,
                    label=c["name"] + (" [residual failed]" if not c["accepted"] else ""),
                )
        ax.set(xlim=(-2 * L * 1e3, 2 * L * 1e3), xlabel="x [mm], inlet to outlet", ylabel=label)
        ax.legend(fontsize=7)
        save(fig, directory, f"centerline_{key}")
    fig, axes = figure("Conservation and source-truncation error budget", context, rows=2)
    axes[0].semilogy(range(len(cases)), [c["audit"]["mass_maximum"] for c in cases], "o-")
    axes[0].axhline(1e-8, color="red", linestyle=":")
    axes[0].set(xlabel="Case index (summary order)", ylabel="Original max continuity defect")
    axes[1].plot(
        [abs(c["x_limits_m"][0]) / L for c in domains],
        [c["omitted_pressure_outside_16"]["center_displacement_fraction"] for c in domains],
        "o-",
    )
    axes[1].set(
        xlabel="Actual upstream extent / L", ylabel="Omitted w(0) fraction beyond 16 lengths"
    )
    save(fig, directory, "error_budget")
    fig, (ax,) = figure("Centerline rupture brackets; theta < 1-1e-8", context)
    for k, case in enumerate(domains):
        for front in case["fronts"]:
            lo, hi = np.array(front["bracket_m"]) * 1e6
            ax.errorbar(
                (lo + hi) / 2,
                k,
                xerr=(hi - lo) / 2,
                fmt="o",
                color="tab:blue" if case["accepted"] else "red",
            )
    ax.set_yticks(range(len(domains)), [c["name"] for c in domains])
    ax.set(xlabel="Front bracket [um], inlet to outlet (+x)", ylabel="Domain case")
    save(fig, directory, "rupture_brackets")


def main(directory):
    directory.mkdir(parents=True, exist_ok=False)
    report = {
        "criteria_declared_before_run": CRITERIA,
        "runtime": provenance(),
        "inputs": {
            "h0_m": H0,
            "Rx_m": RX,
            "Ry_m": RY,
            "eta_pa_s": ETA,
            "Ue_m_s": SPEED,
            "L_m": L,
            "Ly_m": LY,
            "p_ref_pa": P_REF,
        },
        "scope": "rigid graded far field + unchanged uniform coupled references",
        "cases": [],
        "failures": [],
        "domain_independence_established": False,
    }
    configurations = [
        (f"mesh_{d}", (32, 32, 32), d, "mesh", False, False) for d in (0.5, 0.25, 0.125, 0.0625)
    ]
    configurations += [
        (f"all_{a}", (a, a, a), 0.125, "all", False, True) for a in (8, 16, 32, 64, 128, 192, 256)
    ]
    configurations += [
        (f"coarse_all_{a}", (a, a, a), 0.25, "coarse_all", False, True) for a in (64, 128, 192, 256)
    ]
    configurations += [("outermesh_fine", (32, 32, 32), 0.0625, "outermesh", False, False)]
    for side, index in (("inlet", 0), ("outlet", 1), ("transverse", 2)):
        for a in (64, 128):
            extents = [32, 32, 32]
            extents[index] = a
            configurations.append((f"{side}_{a}", extents, 0.125, side, False, True))
    configurations += [
        (f"coupled_mesh_{d}", (4, 4, 4), d, "coupled", True, False) for d in (0.5, 0.25, 0.125)
    ]
    configurations += [
        (f"coupled_extent_{a}", (a, a, a), 0.25, "coupled", True, False) for a in (8, 16)
    ]
    for name, extents, step, family, uniform, snap in configurations:
        try:
            entry, fields = solve_case(
                name, extents, step, family=family, uniform=uniform, snap=snap
            )
            np.savez_compressed(directory / f"{name}.npz", **fields)
            report["cases"].append(entry)
            print(
                name,
                entry["shape"],
                entry["load_n"],
                entry["audit"]["mass_maximum"],
                "accepted" if entry["accepted"] else "residual rejected",
                flush=True,
            )
        except (RuntimeError, ValueError) as error:
            failure = {"name": name, "reason": str(error)}
            if isinstance(error, CavitatingNewtonError):
                failure["diagnostics"] = error.diagnostics
            report["failures"].append(failure)
            print(name, "failed:", error, flush=True)
        (directory / "summary.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
    plots(directory, report["cases"])
    for family in ("all", "coarse_all", "mesh"):
        cases = [c for c in report["cases"] if c["family"] == family]
        changes = []
        for previous, current in zip(cases, cases[1:], strict=False):
            changes.append(
                {
                    "from": previous["name"],
                    "to": current["name"],
                    "both_accepted": previous["accepted"] and current["accepted"],
                    "relative_load": abs(current["load_n"] / previous["load_n"] - 1),
                    "relative_minimum_film": abs(
                        current["extrema"]["minimum_film"]["value_si"]
                        / previous["extrema"]["minimum_film"]["value_si"]
                        - 1
                    ),
                    "relative_peak_pressure": abs(
                        current["extrema"]["maximum_pressure"]["value_si"]
                        / previous["extrema"]["maximum_pressure"]["value_si"]
                        - 1
                    ),
                }
            )
        report[family + "_changes"] = changes
    report["status"] = "error-budget study completed; coupled domain independence remains open"
    report["figures"] = [
        p.relative_to(directory).as_posix() for p in sorted(directory.glob("*.png"))
    ]
    (directory / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=Path("results/farfield_budget"))
    main(parser.parse_args().output_directory)
