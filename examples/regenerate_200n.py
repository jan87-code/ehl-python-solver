"""Checkpointed recovery of the registered 33x49 200 N source from cold physics.

No saved film values are fitted. Original equations, tolerances, 30 Newton
corrections, eight active models and twelve load attempts per target apply.
Resume only an independently audited accepted root into a NEW directory.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from ehl_solver.benchmark_200n import Circular200N
from ehl_solver.elasticity_audit import direct_midpoint_displacement
from ehl_solver.property_coupled import (
    PropertyCoupledSolution,
    PropertyCouplingError,
    solve_property_coupling,
)
from ehl_solver.property_load_newton import (
    PropertyLoadPathError,
    PropertyLoadSystem,
    continue_property_load,
    verify_seed,
)
from examples.run_load_directed_200n import finite_json
from examples.run_property_load_verification import save_state
from examples.run_scientific_validation import provenance, save_json


def main(directory, source=None):
    case = Circular200N()
    fluid = case.fluid()
    problem = case.problem(.5e-6, ny=33, nx=49)
    directory.mkdir(parents=True, exist_ok=False)
    grid = {"shape_ny_nx": [33, 49], "attempts": [], "closed_loads": [],
            "force_200n_closed": False}
    report = {"status": "incomplete", "stage": "seed", "case": case.parameters(),
              "runtime": provenance(), "preconditioner": "local",
              "grids": {"refined": grid}, "force_target_n": 200.,
              "mesh_verified": False, "physical_domain_verified": False}

    def checkpoint():
        save_json(directory / "summary.json", finite_json(report))

    checkpoint()
    if source is None:
        print("Cold full-material prescribed-offset seed: 33x49, h0=0.5 um", flush=True)
        try:
            seed = solve_property_coupling(problem, fluid, case.entrainment_m_s)
        except PropertyCouplingError as error:
            save_json(directory / "seed_failure.json", finite_json(error.history))
            if error.last_solution is not None:
                save_state(directory / "failed_seed_partial.npz", problem,
                           error.last_solution, fluid)
            report["status"] = "cold seed failed; no accepted load root"
            checkpoint()
            raise
        offset = .5e-6
        save_json(directory / "seed_history.json", finite_json(seed.history))
    else:
        old = json.loads((source / "summary.json").read_text())
        if old["case"] != case.parameters() or old["grids"]["refined"]["shape_ny_nx"] != [33, 49]:
            raise ValueError("registered 33x49 source required")
        accepted = [e for e in old["grids"]["refined"]["attempts"] if e["accepted"]]
        entry = accepted[-1] if accepted else old["seed"]
        path = source / entry["state_file"]
        with np.load(path) as arrays:
            np.testing.assert_array_equal(arrays["x_m"], problem.x_m)
            np.testing.assert_array_equal(arrays["y_m"], problem.y_m)
            p, t, h, w = [arrays[k].copy() for k in
                          ("pressure_pa", "theta", "film_m", "displacement_m")]
        seed = PropertyCoupledSolution(p, t, h, w, problem.load_n(p), {}, (), 1.)
        offset = entry["offset_m"]
        report["input"] = {"state_file": str(path),
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    verify_seed(PropertyLoadSystem(problem, fluid, case.entrainment_m_s, 200.), seed, offset)
    if source is not None and seed.load_n >= 200.*(1-1e-8):
        raise ValueError("source already closes 200 N; audit/use it directly rather than resume")
    save_state(directory / "seed.npz", problem, seed, fluid)
    report["seed"] = {"load_n": seed.load_n, "offset_m": offset, "state_file": "seed.npz"}
    checkpoint()

    def callback(record, result):
        index = len(grid["attempts"])
        path = directory / f"attempt_{index:03d}"
        path.mkdir()
        entry = dict(record)
        if record["accepted"]:
            s = result.solution
            save_state(path / "solution.npz", problem, s, fluid)
            save_json(path / "history.json", finite_json(s.history))
            peak = np.unravel_index(np.argmax(s.pressure_pa), problem.shape)
            direct = direct_midpoint_displacement(
                problem.x_m, problem.y_m, s.pressure_pa, case.contact_modulus_pa,
                [[0., 0.], [problem.x_m[peak[1]], problem.y_m[peak[0]]]],
            )
            error = float(np.max(abs(direct - [s.elastic_displacement_m[16, 24],
                                              s.elastic_displacement_m[peak]])))
            if error > 1e-18:
                raise RuntimeError("independent direct elastic source audit failed")
            entry.update(state_file=f"attempt_{index:03d}/solution.npz",
                         direct_error_m=error, minimum_film_m=float(s.film_thickness_m.min()),
                         central_film_m=float(s.film_thickness_m[16, 24]),
                         peak_pressure_pa=float(s.pressure_pa.max()))
        else:
            save_json(path / "failure.json", finite_json({
                "history": result.history, "diagnostics": result.diagnostics,
            }))
            if result.last_root is not None:
                save_state(path / "failed_partial.npz", problem, result.last_root.solution, fluid)
        grid["attempts"].append(entry)
        checkpoint()
        print(index, record["target_load_n"], record["accepted"],
              record.get("reason", ""), flush=True)

    for target in (2., 4., 8., 10., 20., 40., 80., 120., 160., 200.):
        if seed.load_n >= target*(1-1e-8):
            continue
        report["stage"] = f"load {target:g} N"
        checkpoint()
        try:
            path = continue_property_load(
                problem, fluid, case.entrainment_m_s, target, seed=seed, seed_offset_m=offset,
                initial_increment_n=min(2., max(.25, .2*seed.load_n)), maximum_attempts=12,
                preconditioner="local", callback=callback,
            )
        except PropertyLoadPathError as error:
            report["status"] = "bounded load continuation failed; partial states are not roots"
            report["failure"] = str(error)
            checkpoint()
            raise
        seed, offset = path.root.solution, path.root.central_film_offset_m
        if abs(seed.load_n/target-1) > 1e-8:
            raise RuntimeError("independent force gate failed")
        grid["closed_loads"].append({"target_n": target, "load_n": seed.load_n,
                                     "offset_m": offset})
        grid["force_200n_closed"] = target == 200.
        checkpoint()
    if abs(seed.load_n/200.-1) > 1e-8:
        raise RuntimeError("recovery did not close the registered 200 N target")
    grid["force_200n_closed"] = True
    report["status"] = "recovered accepted 200 N source; mesh/physical accuracy open"
    report["stage"] = "complete"
    # Preserve exact production sources with the recovery, not in Git outputs.
    snapshot = directory / "source_snapshot"
    snapshot.mkdir()
    root = Path(__file__).resolve().parents[1]
    for path in (root / "src/ehl_solver").glob("*.py"):
        shutil.copy2(path, snapshot / path.name)
    checkpoint()
    manifest = {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in directory.rglob("*") if p.is_file()}
    save_json(directory / "sha256.json", manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--input-directory", type=Path)
    args = parser.parse_args()
    main(args.output_directory, args.input_directory)
