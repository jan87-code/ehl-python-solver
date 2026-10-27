"""Bounded readiness probes for the user's 200 N circular EHL case.

A failed probe stays failed. No field/pressure rescaling to reach the force.
This runner tests the current positive-offset branch before signed continuation.
"""

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ehl_solver.benchmark_200n import Circular200N
from ehl_solver.property_coupled import PropertyCouplingError, solve_property_coupling
from ehl_solver.property_load import PropertyLoadError, solve_property_at_load
from examples.run_property_load_verification import save_state
from examples.run_scientific_validation import provenance, save_json


def retain_failure(directory, exc, problem):
    """Save every available exception-chain state under a distinct level."""
    levels = []
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        level = {"exception": type(exc).__name__, "message": str(exc)}
        history = getattr(exc, "history", None)
        if history is not None:
            level["history"] = history
        state = getattr(exc, "last_solution", None)
        if state is not None:
            arrays = {}
            for name in (
                "pressure_pa",
                "fractional_film_content",
                "film_thickness_m",
                "elastic_displacement_m",
            ):
                if hasattr(state, name):
                    arrays[name] = getattr(state, name)
            if arrays:
                filename = f"failed_level_{len(levels)}.npz"
                np.savez_compressed(
                    directory / filename, x_m=problem.x_m, y_m=problem.y_m, **arrays
                )
                level["state_file"] = filename
            level["audit"] = getattr(state, "audit", None)
            level["history"] = getattr(state, "history", history)
        levels.append(level)
        exc = getattr(exc, "inner_failure", None)
    save_json(directory / "failure.json", levels)
    return levels


def main(directory, *, ny=17, nx=25):
    c = Circular200N()
    c.problem(5e-6, ny=ny, nx=nx)  # Validate before creating output.
    directory.mkdir(parents=True, exist_ok=False)
    fluid = c.fluid()
    report = {
        "status": "incomplete",
        "runtime": provenance(),
        "case": c.parameters(),
        "shape_ny_nx": [ny, nx],
        "bounds_in_hertz_radii": [-3, 3, -3, 3],
        "criteria": {
            "mass_maximum": 1e-8,
            "relative_local_global": 1e-10,
            "complementarity": 1e-10,
            "relative_force": 1e-8,
            "mesh_domain_relative_change": 0.01,
        },
        "probes": [],
        "target_achieved": False,
        "mesh_verified": False,
        "domain_verified": False,
    }

    def checkpoint():
        save_json(directory / "summary.json", report)

    checkpoint()
    for offset in (5e-6, 2e-6, 0.5e-6, 0.1e-6):
        path = directory / f"offset_{offset * 1e6:g}um"
        path.mkdir()
        problem = c.problem(offset, ny=ny, nx=nx)
        record = {"offset_m": offset, "accepted": False}
        try:
            solution = solve_property_coupling(problem, fluid, c.entrainment_m_s)
            save_state(path / "solution.npz", problem, solution, fluid)
            save_json(path / "history.json", solution.history)
            record.update(
                accepted=True,
                load_n=solution.load_n,
                audit=solution.audit,
                minimum_film_m=float(solution.film_thickness_m.min()),
                peak_pressure_pa=float(solution.pressure_pa.max()),
                central_film_m=float(solution.film_thickness_m[ny // 2, nx // 2]),
            )
        except (PropertyCouplingError, ValueError, FloatingPointError) as exc:
            record["failure"] = retain_failure(path, exc, problem)
        report["probes"].append(record)
        checkpoint()
        print(
            f"offset={offset:.6g} m accepted={record['accepted']} Q={record.get('load_n')}",
            flush=True,
        )
    accepted = sorted((p for p in report["probes"] if p["accepted"]), key=lambda p: p["offset_m"])
    if len(accepted) >= 2 and accepted[0]["load_n"] >= c.load_n >= accepted[-1]["load_n"]:
        path = directory / "load_control"
        path.mkdir()
        problem = c.problem(accepted[-1]["offset_m"], ny=ny, nx=nx)

        def callback(e, s):
            save_state(path / f"probe_{e.central_film_offset_m:.15g}.npz", problem, s, fluid)
            save_json(path / f"probe_{e.central_film_offset_m:.15g}.json", asdict(e))

        try:
            solution = solve_property_at_load(
                problem,
                fluid,
                c.entrainment_m_s,
                c.load_n,
                h0_bracket_m=(accepted[0]["offset_m"], accepted[-1]["offset_m"]),
                evaluation_callback=callback,
            )
            save_state(path / "solution.npz", problem, solution.coupled_solution, fluid)
            report["target_achieved"] = True
            report["load_control"] = {
                "h0_m": solution.central_film_offset_m,
                "relative_force_error": solution.relative_load_error,
            }
        except PropertyLoadError as exc:
            report["load_failure"] = retain_failure(path, exc, problem)
    else:
        report["load_control_not_attempted"] = (
            "No independently audited positive-offset bracket for 200 N"
        )
    report["status"] = "readiness assessment complete; 200 N physical validation open"
    checkpoint()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--ny", type=int, default=17)
    parser.add_argument("--nx", type=int, default=25)
    args = parser.parse_args()
    main(args.output_directory, ny=args.ny, nx=args.nx)
