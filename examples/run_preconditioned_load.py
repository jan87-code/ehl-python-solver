"""Resume a preserved accepted load root with an explicitly selected preconditioner."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ehl_solver.benchmark_200n import Circular200N
from ehl_solver.elasticity_audit import direct_midpoint_displacement
from ehl_solver.property_coupled import PropertyCoupledSolution
from ehl_solver.property_load_newton import PropertyLoadPathError, continue_property_load
from examples.run_load_directed_200n import finite_json
from examples.run_property_load_verification import save_state
from examples.run_scientific_validation import provenance, save_json
from examples.scientific_figures import figure, save


def main(source, directory, mode):
    original = json.loads((source / "summary.json").read_text())
    c = Circular200N()
    fluid = c.fluid()
    if original["case"] != c.parameters():
        raise ValueError("registered circular case required")
    old = original["grids"]["refined"]
    ny, nx = old["shape_ny_nx"]
    problem = c.problem(0.5e-6, ny=ny, nx=nx)
    entry = [e for e in old["attempts"] if e["accepted"]][-1]
    input_path = source / entry["state_file"]
    with np.load(input_path) as a:
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
    offset = entry["offset_m"]
    directory.mkdir(parents=True, exist_ok=False)
    grid = {
        "shape_ny_nx": [ny, nx],
        "input_npz": str(input_path),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "attempts": [],
        "closed_loads": [],
        "force_200n_closed": False,
    }
    report = {
        "status": "incomplete",
        "runtime": provenance(),
        "case": c.parameters(),
        "preconditioner": mode,
        "grids": {"refined": grid},
        "force_target_n": 200.0,
        "mesh_verified": False,
        "physical_domain_verified": False,
        "matched_loads": [],
    }

    def checkpoint():
        save_json(directory / "summary.json", finite_json(report))

    checkpoint()

    def callback(record, result):
        index = len(grid["attempts"])
        path = directory / f"attempt_{index:03d}"
        path.mkdir()
        item = dict(record)
        if record["accepted"]:
            s = result.solution
            save_state(path / "solution.npz", problem, s, fluid)
            save_json(path / "history.json", finite_json(s.history))
            peak = np.unravel_index(np.argmax(s.pressure_pa), problem.shape)
            direct = direct_midpoint_displacement(
                problem.x_m,
                problem.y_m,
                s.pressure_pa,
                c.contact_modulus_pa,
                [[0.0, 0.0], [problem.x_m[peak[1]], problem.y_m[peak[0]]]],
            )
            stored = [s.elastic_displacement_m[ny // 2, nx // 2], s.elastic_displacement_m[peak]]
            error = float(abs(direct - stored).max())
            if error > 1e-18:
                raise RuntimeError("direct source audit failed")
            item.update(
                minimum_film_m=float(s.film_thickness_m.min()),
                central_film_m=float(s.film_thickness_m[ny // 2, nx // 2]),
                peak_pressure_pa=float(s.pressure_pa.max()),
                direct_error_m=error,
                state_file=f"attempt_{index:03d}/solution.npz",
            )
        else:
            save_json(
                path / "failure.json",
                finite_json({"history": result.history, "diagnostics": result.diagnostics}),
            )
            if result.last_root is not None:
                save_state(path / "failed_partial.npz", problem, result.last_root.solution, fluid)
                item["failed_offset_m"] = result.last_root.central_film_offset_m
        grid["attempts"].append(item)
        checkpoint()
        print(
            index, record["target_load_n"], record["accepted"], record.get("reason", ""), flush=True
        )

    for target in (120.0, 160.0, 200.0):
        if target <= seed.load_n:
            continue
        try:
            path = continue_property_load(
                problem,
                fluid,
                2.0,
                target,
                seed=seed,
                seed_offset_m=offset,
                initial_increment_n=5.0,
                maximum_attempts=16,
                callback=callback,
                preconditioner=mode,
            )
        except PropertyLoadPathError as exc:
            grid["stall"] = {
                "requested_load_n": target,
                "reason": str(exc),
                "last_load_n": exc.last_root.solution.load_n,
                "last_offset_m": exc.last_root.central_film_offset_m,
            }
            break
        seed = path.root.solution
        offset = path.root.central_film_offset_m
        error = abs(seed.load_n / target - 1)
        if error > 1e-8:
            raise RuntimeError("independent force gate failed")
        grid["closed_loads"].append(
            {
                "target_n": target,
                "calculated_n": seed.load_n,
                "relative_force_error": error,
                "offset_m": offset,
                "minimum_film_m": float(seed.film_thickness_m.min()),
                "central_film_m": float(seed.film_thickness_m[ny // 2, nx // 2]),
                "peak_pressure_pa": float(seed.pressure_pa.max()),
                "audit": seed.audit,
            }
        )
        grid["force_200n_closed"] = target == 200.0
        checkpoint()
    report["status"] = "bounded preconditioned restart complete; mesh/physical accuracy open"
    a = [e for e in grid["attempts"] if e["accepted"]]
    fig, axes = figure(
        "Preconditioned force restart; original equations and acceptance",
        f"(ny,nx)=({ny},{nx}); preconditioner={mode}; preserved 100 N-class seed\n"
        "R=15 mm, E_contact=113 GPa, Ue=2 m/s; flooded finite +/-3a reservoir",
        rows=2,
    )
    axes[0].plot([e["load_n"] for e in a], [e["minimum_film_m"] * 1e6 for e in a], "o-")
    axes[0].set_ylabel("Sampled minimum film [um]")
    axes[1].plot([e["load_n"] for e in a], [e["offset_m"] * 1e6 for e in a], "o-")
    axes[1].set_ylabel("Signed offset [um]")
    for ax in axes:
        ax.set_xlabel("Calculated load [N]")
    save(fig, directory, "restart")
    report["figure_manifest"] = ["restart.png"]
    checkpoint()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument(
        "--preconditioner", choices=["frozen", "material", "local"], default="local"
    )
    args = parser.parse_args()
    main(args.input_directory, args.output_directory, args.preconditioner)
