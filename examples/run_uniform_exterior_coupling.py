"""Bounded nonlinear uniform exterior/Picard verification at a finite reservoir.

Recomputes independent full monolithic references; no reference fields seed
the exterior solves. No nonuniform FFT, new rheology or GPU execution.
"""

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ehl_solver.cavitating_coupled import CavitatingElasticityProblem
from ehl_solver.exterior import ExteriorSolution, retained_source_displacements
from ehl_solver.exterior_coupled import ExteriorCouplingError, solve_uniform_exterior_coupling
from examples.run_exterior_response import fields
from examples.run_farfield_budget import ETA, H0, LY, P_REF, RX, RY, SPEED, L, solve_case
from examples.run_scientific_validation import provenance, save_json
from examples.scientific_figures import figure, save


def main(directory, *, include_fine=False):
    directory.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "incomplete",
        "runtime": provenance(),
        "references": [],
        "cases": [],
        "include_fine": include_fine,
        "scope": "Uniform finite-reservoir prescribed-offset exterior/Picard experiment",
        "criteria": {
            "relative_pressure_load_film_reference": 1e-8,
            "original_mass": 1e-8,
            "relative_local_global_constraints": 1e-10,
            "absolute_source_error_m": 1e-18,
        },
    }

    def checkpoint():
        save_json(directory / "summary.json", report)

    checkpoint()
    try:
        configurations = [(0.5, (1, 2)), (0.25, (2,))]
        if include_fine:
            configurations.append((0.125, (1,)))
        for step, cuts in configurations:
            reference, rf = solve_case(
                f"monolithic_{step}", (8, 8, 8), step, family="coupled", uniform=True, snap=False
            )
            report["references"].append(reference)
            np.savez_compressed(directory / f"reference_{step}.npz", **rf)
            x, y = rf["x_m"], rf["y_m"]
            problem = CavitatingElasticityProblem(x, y, H0, RX, RY, 1e9, L, H0, P_REF, 1)
            if step == 0.5:
                try:
                    solve_uniform_exterior_coupling(
                        problem,
                        ETA,
                        SPEED,
                        core_bounds_m=(-L, L, -LY, LY),
                        maximum_iterations_per_stage=1,
                    )
                except ExteriorCouplingError as exc:
                    report["bounded_failure"] = {
                        "reason": str(exc),
                        "history": [asdict(h) for h in exc.history],
                        "exterior_history": exc.exterior_history,
                    }
                    last = exc.last_solution
                    np.savez_compressed(
                        directory / "failed_budget_one.npz",
                        **fields(
                            x,
                            y,
                            last.film_thickness_m,
                            last.pressure_pa,
                            last.fractional_film_content,
                        ),
                        displacement_m=last.elastic_displacement_m,
                    )
                    checkpoint()
                else:
                    raise AssertionError("budget-one probe unexpectedly converged")
            for cut in cuts:
                result = solve_uniform_exterior_coupling(
                    problem, ETA, SPEED, core_bounds_m=(-cut * L, cut * L, -cut * LY, cut * LY)
                )
                s = result.solution
                displacement = retained_source_displacements(
                    x,
                    y,
                    ExteriorSolution(
                        s.pressure_pa,
                        s.fractional_film_content,
                        result.retained_mask,
                        result.audit,
                        (),
                    ),
                    1e9,
                )
                source_error = float(
                    abs(
                        displacement["total_displacement_m"]
                        - s.elastic_displacement_m[result.retained_mask]
                    ).max()
                )
                entry = {
                    "step": step,
                    "cut": cut,
                    "shape": list(s.pressure_pa.shape),
                    "load_n": s.load_n,
                    "minimum_film_m": float(s.film_thickness_m.min()),
                    "maximum_pressure_pa": float(s.pressure_pa.max()),
                    "relative_pressure_error": float(
                        np.linalg.norm(s.pressure_pa - rf["pressure_pa"])
                        / np.linalg.norm(rf["pressure_pa"])
                    ),
                    "relative_load_error": abs(s.load_n / reference["load_n"] - 1),
                    "relative_film_error": float(
                        np.linalg.norm(s.film_thickness_m - rf["film_m"])
                        / np.linalg.norm(rf["film_m"])
                    ),
                    "maximum_content_error": float(
                        abs(s.fractional_film_content - rf["theta"]).max()
                    ),
                    "all_inner_direct_fft_error_m": source_error,
                    "inner_targets": len(displacement["target_points_m"]),
                    "audit": result.audit,
                    "history": [asdict(h) for h in s.history],
                    "exterior_history": result.exterior_history,
                    "state_file": f"coupled_{step}_{cut}.npz",
                }
                entry["accepted"] = bool(
                    reference["accepted"]
                    and max(
                        entry["relative_pressure_error"],
                        entry["relative_load_error"],
                        entry["relative_film_error"],
                    )
                    <= 1e-8
                    and source_error <= 1e-18
                )
                np.savez_compressed(
                    directory / entry["state_file"],
                    **fields(x, y, s.film_thickness_m, s.pressure_pa, s.fractional_film_content),
                    displacement_m=s.elastic_displacement_m,
                    retained_mask=result.retained_mask,
                    **displacement,
                )
                report["cases"].append(entry)
                checkpoint()
                print(
                    step,
                    cut,
                    len(s.history),
                    entry["relative_pressure_error"],
                    result.audit["mass_maximum"],
                    flush=True,
                )
        render(directory, report)
        report["figures"] = [p.name for p in sorted(directory.glob("*.png"))]
        report["status"] = (
            "verified restricted uniform exterior coupling"
            if all(c["accepted"] for c in report["cases"])
            else "failed acceptance"
        )
        checkpoint()
    except Exception as exc:
        report["status"] = "failed"
        report["failure"] = repr(exc)
        checkpoint()
        raise
    return report


def render(directory, report):
    context = (
        "Uniform physical reservoir +/-8L,+/-8Ly; ambient edge p, west theta=1\n"
        "h0=.1 um,Rx=10 mm,Ry=20 mm,E*=1 GPa,eta=.04 Pa s,Ue=.001 m/s\n"
        "Separate damped Picard/exterior response; full-source uniform FFT; prescribed h0"
    )
    fig, axes = figure(
        "Original physical continuity and film defects through compliance stages", context, rows=2
    )
    for c in report["cases"]:
        label = f"dX={c['step']}, cut={c['cut']}"
        axes[0].semilogy(
            [h["exact_film_audit"]["mass_maximum"] for h in c["exterior_history"]], label=label
        )
        axes[1].plot([h["film_defect_relative"] for h in c["history"]], label=label)
    axes[0].axhline(1e-8, color="black", linestyle=":")
    axes[0].set_ylabel("Original max continuity")
    axes[0].legend(fontsize=8)
    axes[1].set(
        xlabel="Outer iteration (five compliance stages)", ylabel="Relative exact-film defect"
    )
    axes[1].set_yscale("symlog", linthresh=1e-16)
    axes[1].legend(fontsize=8)
    save(fig, directory, "coupled_histories")
    fig, axes = figure("Independent monolithic versus exterior-coupled fields", context, rows=2)
    c = report["cases"][-1]
    f = np.load(directory / c["state_file"])
    r = np.load(directory / f"reference_{c['step']}.npz")
    j = len(f["y_m"]) // 2
    for a, label in ((r, "Independent full monolithic"), (f, "Exterior/Picard")):
        axes[0].plot(a["x_m"] * 1e3, a["pressure_pa"][j] / 1e6, label=label)
        axes[1].plot(a["x_m"] * 1e3, a["film_m"][j] * 1e6, label=label)
    axes[0].set_ylabel("Pressure [MPa]")
    axes[0].legend()
    axes[1].set(xlabel="x [mm], entrainment +x", ylabel="Film [um]")
    axes[1].legend()
    save(fig, directory, "coupled_fields")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory", type=Path, default=Path("results/uniform_exterior_coupling")
    )
    parser.add_argument(
        "--include-fine", action="store_true", help="Add bounded 129x129 reference/response case"
    )
    args = parser.parse_args()
    main(args.output_directory, include_fine=args.include_fine)
