"""Checkpointed coarse/refined matrix-free force continuation from preserved roots."""

import argparse
import hashlib
from pathlib import Path

import numpy as np

from ehl_solver.benchmark_200n import Circular200N
from ehl_solver.elasticity_audit import direct_midpoint_displacement
from ehl_solver.property_coupled import PropertyCoupledSolution
from ehl_solver.property_load_newton import PropertyLoadPathError, continue_property_load
from examples.run_property_load_verification import save_state
from examples.run_scientific_validation import provenance, save_json
from examples.scientific_figures import figure, save


def finite_json(value):
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return {"nonfinite_diagnostic": repr(value)}
    return value


def main(directory, coarse_source, refined_source):
    case = Circular200N()
    fluid = case.fluid()
    directory.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "incomplete",
        "runtime": provenance(),
        "case": case.parameters(),
        "grids": {},
        "matched_loads": [],
        "force_target_n": 200.0,
        "mesh_verified": False,
        "physical_domain_verified": False,
    }

    def checkpoint():
        save_json(directory / "summary.json", finite_json(report))

    checkpoint()
    specifications = [
        (
            "coarse",
            17,
            25,
            coarse_source,
            -3.014203906597183e-7,
            [10.0, 20.0, 40.0, 80.0, 120.0, 160.0, 200.0],
        ),
        (
            "refined",
            33,
            49,
            refined_source,
            0.5e-6,
            [2.0, 4.0, 8.0, 10.0, 20.0, 40.0, 80.0, 120.0, 160.0, 200.0],
        ),
    ]
    for name, ny, nx, source, offset, targets in specifications:
        problem = case.problem(0.5e-6, ny=ny, nx=nx)
        path = directory / name
        path.mkdir()
        with np.load(source) as a:
            np.testing.assert_array_equal(a["x_m"], problem.x_m)
            np.testing.assert_array_equal(a["y_m"], problem.y_m)
            seed = PropertyCoupledSolution(
                a["pressure_pa"],
                a["theta"],
                a["film_m"],
                a["displacement_m"],
                problem.load_n(a["pressure_pa"]),
                {},
                (),
                1.0,
            )
        grid = {
            "shape_ny_nx": [ny, nx],
            "input_npz": str(source),
            "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "attempts": [],
            "closed_loads": [],
            "force_200n_closed": False,
        }
        report["grids"][name] = grid
        checkpoint()

        def callback(
            record, result, *, grid=grid, path=path, problem=problem, ny=ny, nx=nx, name=name
        ):
            index = len(grid["attempts"])
            step_path = path / f"attempt_{index:03d}"
            step_path.mkdir()
            entry = dict(record)
            if record["accepted"]:
                state = result.solution
                save_state(step_path / "solution.npz", problem, state, fluid)
                save_json(step_path / "history.json", finite_json(state.history))
                peak = np.unravel_index(np.argmax(state.pressure_pa), problem.shape)
                direct = direct_midpoint_displacement(
                    problem.x_m,
                    problem.y_m,
                    state.pressure_pa,
                    case.contact_modulus_pa,
                    [[0.0, 0.0], [problem.x_m[peak[1]], problem.y_m[peak[0]]]],
                )
                stored = [
                    state.elastic_displacement_m[ny // 2, nx // 2],
                    state.elastic_displacement_m[peak],
                ]
                error = float(abs(direct - stored).max())
                if error > 1e-18:
                    raise RuntimeError("direct source check failed")
                entry.update(
                    minimum_film_m=float(state.film_thickness_m.min()),
                    central_film_m=float(state.film_thickness_m[ny // 2, nx // 2]),
                    peak_pressure_pa=float(state.pressure_pa.max()),
                    direct_error_m=error,
                    state_file=str((step_path / "solution.npz").relative_to(directory)),
                )
            else:
                save_json(
                    step_path / "failure.json",
                    finite_json({"history": result.history, "diagnostics": result.diagnostics}),
                )
                if result.last_root is not None:
                    save_state(
                        step_path / "failed_partial.npz", problem, result.last_root.solution, fluid
                    )
                    entry["failed_offset_m"] = result.last_root.central_film_offset_m
            grid["attempts"].append(entry)
            checkpoint()
            print(
                name,
                index,
                record["target_load_n"],
                record["accepted"],
                record.get("reason", ""),
                flush=True,
            )

        for target in targets:
            try:
                result = continue_property_load(
                    problem,
                    fluid,
                    2.0,
                    target,
                    seed=seed,
                    seed_offset_m=offset,
                    initial_increment_n=min(2.0, max(0.25, seed.load_n * 0.2)),
                    maximum_attempts=12,
                    callback=callback,
                )
            except PropertyLoadPathError as exc:
                grid["stall"] = {
                    "requested_load_n": target,
                    "reason": str(exc),
                    "last_load_n": exc.last_root.solution.load_n,
                    "last_offset_m": exc.last_root.central_film_offset_m,
                }
                checkpoint()
                break
            seed = result.root.solution
            offset = result.root.central_film_offset_m
            force_error = abs(seed.load_n / target - 1)
            if force_error > 1e-8:
                raise RuntimeError("independent matched-load gate failed")
            grid["closed_loads"].append(
                {
                    "target_n": target,
                    "calculated_n": seed.load_n,
                    "relative_force_error": force_error,
                    "offset_m": offset,
                    "minimum_film_m": float(seed.film_thickness_m.min()),
                    "central_film_m": float(seed.film_thickness_m[ny // 2, nx // 2]),
                    "peak_pressure_pa": float(seed.pressure_pa.max()),
                    "audit": seed.audit,
                }
            )
            grid["force_200n_closed"] = target == 200.0
            checkpoint()
    coarse = {p["target_n"]: p for p in report["grids"]["coarse"]["closed_loads"]}
    fine = {p["target_n"]: p for p in report["grids"]["refined"]["closed_loads"]}
    for target in sorted(coarse.keys() & fine.keys()):
        a, b = coarse[target], fine[target]
        report["matched_loads"].append(
            {
                "target_n": target,
                "coarse": a,
                "refined": b,
                "relative_minimum_film_change": b["minimum_film_m"] / a["minimum_film_m"] - 1,
                "relative_central_film_change": b["central_film_m"] / a["central_film_m"] - 1,
                "relative_peak_pressure_change": b["peak_pressure_pa"] / a["peak_pressure_pa"] - 1,
            }
        )
    fig, axes = figure(
        "Load-directed material JFO: accepted physical roots only",
        "R=15 mm, E_contact=113 GPa, Ue=2 m/s; flooded finite +/-3a reservoir\n"
        "Roelands / Dowson-Higginson; mesh and domain accuracy remain separate",
        rows=2,
    )
    for name, grid in report["grids"].items():
        a = [e for e in grid["attempts"] if e["accepted"]]
        axes[0].plot(
            [e["load_n"] for e in a], [e["minimum_film_m"] * 1e6 for e in a], "o-", label=name
        )
        axes[1].plot([e["load_n"] for e in a], [e["offset_m"] * 1e6 for e in a], "o-", label=name)
    axes[0].set_ylabel("Sampled minimum film [um]")
    axes[1].set_ylabel("Signed offset [um]")
    for ax in axes:
        ax.set_xlabel("Calculated load [N]")
        ax.legend()
    save(fig, directory, "load_directed")
    report["figure_manifest"] = ["load_directed.png"]
    report["status"] = "bounded force-continuation assessment complete; physical validation open"
    checkpoint()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--coarse-source", type=Path, required=True)
    parser.add_argument("--refined-source", type=Path, required=True)
    args = parser.parse_args()
    main(args.output_directory, args.coarse_source, args.refined_source)
