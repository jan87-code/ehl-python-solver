"""Bounded warm signed-offset branch for the registered circular 200 N case.

Reaches measured-load roots first. Film-height comparison is withheld unless
200 N force closure and later mesh/domain accuracy are established.
"""

import argparse
from pathlib import Path

import numpy as np

from ehl_solver.benchmark_200n import Circular200N
from ehl_solver.elasticity_audit import direct_midpoint_displacement
from ehl_solver.property_coupled import solve_property_coupling
from ehl_solver.property_warm import WarmContinuationError, continue_warm_property_offset
from examples.run_property_load_verification import save_state
from examples.run_scientific_validation import provenance, save_json
from examples.scientific_figures import figure, save


def main(directory, target_offset_m=-5e-6):
    directory.mkdir(parents=True, exist_ok=False)
    case = Circular200N()
    fluid = case.fluid()
    problem = case.problem(0.5e-6)
    report = {
        "status": "incomplete",
        "runtime": provenance(),
        "case": case.parameters(),
        "shape_ny_nx": list(problem.shape),
        "user_confirmed_viscosity_law": "Roelands",
        "maximum_continuation_attempts": 24,
        "target_offset_m": target_offset_m,
        "target_load_n": 200.0,
        "target_force_closed": False,
        "mesh_verified": False,
        "domain_verified": False,
        "attempts": [],
        "accepted": [],
    }

    def checkpoint():
        save_json(directory / "summary.json", report)

    checkpoint()
    seed = solve_property_coupling(problem, fluid, 2.0)
    save_state(directory / "seed.npz", problem, seed, fluid)
    save_json(directory / "seed_history.json", seed.history)
    report["seed_load_n"] = seed.load_n
    checkpoint()

    def callback(record, result):
        index = len(report["attempts"])
        path = directory / f"attempt_{index:02d}"
        path.mkdir()
        entry = dict(record)
        if record["accepted"]:
            s = result.solution
            save_state(path / "solution.npz", problem, s, fluid)
            save_json(path / "newton_history.json", s.history)
            peak = np.unravel_index(np.argmax(s.pressure_pa), s.pressure_pa.shape)
            direct = direct_midpoint_displacement(
                problem.x_m,
                problem.y_m,
                s.pressure_pa,
                case.contact_modulus_pa,
                [[0.0, 0.0], [problem.x_m[peak[1]], problem.y_m[peak[0]]]],
            )
            stored = np.array([s.elastic_displacement_m[8, 12], s.elastic_displacement_m[peak]])
            entry.update(
                peak_pressure_pa=float(s.pressure_pa.max()),
                central_film_m=float(s.film_thickness_m[8, 12]),
                direct_sample_error_m=float(abs(direct - stored).max()),
                state_file=f"attempt_{index:02d}/solution.npz",
            )
            if entry["direct_sample_error_m"] > 1e-18:
                raise RuntimeError("direct elastic source audit failed")
            report["accepted"].append(entry)
        else:
            save_json(path / "failure_history.json", result.history)
            if result.last_solution is not None:
                save_state(path / "failed_partial.npz", problem, result.last_solution, fluid)
            entry["failure_type"] = type(result).__name__
        report["attempts"].append(entry)
        checkpoint()
        print(
            index,
            record["trial_offset_m"],
            record["accepted"],
            record.get("load_n"),
            record.get("reason", ""),
            flush=True,
        )

    try:
        root, ledger = continue_warm_property_offset(
            problem,
            fluid,
            2.0,
            target_offset_m,
            seed=seed,
            seed_offset_m=0.5e-6,
            initial_step_m=0.1e-6,
            maximum_attempts=24,
            callback=callback,
        )
        report["offset_target_reached"] = True
    except WarmContinuationError as exc:
        root = exc.last_root
        report["offset_target_reached"] = False
        report["continuation_limit"] = str(exc)
    report["last_offset_m"] = root.central_film_offset_m
    report["last_load_n"] = root.solution.load_n
    report["status"] = "bounded signed-offset assessment complete; 200 N physical validation open"
    fig, axes = figure(
        "Signed-offset continuation; measured force, not imposed force",
        "R=15 mm, E_contact=113 GPa, eta0=.04 Pa s, Ue=2 m/s; (ny,nx)=(17,25)\n"
        "Roelands/Dowson-Higginson; flooded finite +/-3a reservoir",
        rows=2,
    )
    states = report["accepted"]
    axes[0].plot([s["trial_offset_m"] * 1e6 for s in states], [s["load_n"] for s in states], "o-")
    axes[0].axhline(200, color="black", linestyle=":", label="requested 200 N")
    axes[0].set_ylabel("Measured load [N]")
    axes[0].legend()
    axes[1].plot(
        [s["trial_offset_m"] * 1e6 for s in states],
        [s["minimum_film_m"] * 1e6 for s in states],
        "o-",
    )
    axes[1].set_ylabel("Sampled minimum total film [um]")
    for ax in axes:
        ax.set_xlabel("Signed rigid offset h0 [um]")
    save(fig, directory, "signed_branch")
    report["figure_manifest"] = ["signed_branch.png"]
    checkpoint()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--target-offset-um", type=float, default=-5.0)
    args = parser.parse_args()
    main(args.output_directory, args.target_offset_um * 1e-6)
