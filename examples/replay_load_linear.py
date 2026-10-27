"""Replay one archived failed bordered correction; no physical root is accepted here."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from ehl_solver.benchmark_200n import Circular200N
from ehl_solver.cavitating_newton import CavitatingLinearError, solve_cavitating_correction
from ehl_solver.property_load_newton import PropertyLoadSystem
from examples.run_load_directed_200n import finite_json
from examples.run_scientific_validation import provenance, save_json


def main(source, directory):
    report = json.loads((source / "summary.json").read_text())
    entry = report["grids"]["refined"]["attempts"][30]
    state_path = source / "refined/attempt_030/failed_partial.npz"
    case = Circular200N()
    problem = case.problem(0.5e-6, ny=33, nx=49)
    if report["case"] != case.parameters() or entry["accepted"]:
        raise ValueError("expected original failed circular correction")
    with np.load(state_path) as a:
        seed = SimpleNamespace(pressure_pa=a["pressure_pa"], fractional_film_content=a["theta"])
        np.testing.assert_array_equal(a["x_m"], problem.x_m)
        np.testing.assert_array_equal(a["y_m"], problem.y_m)
    directory.mkdir(parents=True, exist_ok=False)
    output = {
        "status": "incomplete",
        "runtime": provenance(),
        "input_sha256": hashlib.sha256(state_path.read_bytes()).hexdigest(),
        "input_state": str(state_path),
        "scope": "unconverged linear-state replay; not a physical force root",
        "target_force_n": entry["target_load_n"],
        "results": [],
    }
    for mode in ("frozen", "material", "local"):
        system = PropertyLoadSystem(
            problem, case.fluid(), 2.0, entry["target_load_n"], preconditioner=mode
        )
        z = system.pack(seed, entry["failed_offset_m"])
        start = perf_counter()
        record = {"mode": mode, "accepted_linear_correction": False}
        try:
            correction = solve_cavitating_correction(
                system, z, maximum_krylov_cycles=10, gmres_restart=60
            )
            measured = float(
                np.linalg.norm(system.jacobian_vector(z, correction.direction) + system.residual(z))
                / np.linalg.norm(system.residual(z))
            )
            record.update(
                accepted_linear_correction=measured <= 5e-9,
                true_linear_residual=measured,
                linear_iterations=correction.linear_iterations,
                krylov_histories=correction.krylov_histories,
                used_bordered_fallback=correction.used_bordered_fallback,
            )
            np.savez_compressed(
                directory / f"{mode}_direction.npz", state=z, direction=correction.direction
            )
        except CavitatingLinearError as exc:
            record.update(reason=str(exc), diagnostics=exc.diagnostics)
        record["wall_s"] = perf_counter() - start
        output["results"].append(record)
        save_json(directory / "summary.json", finite_json(output))
        print(
            mode,
            record["accepted_linear_correction"],
            record.get("true_linear_residual"),
            record["wall_s"],
            flush=True,
        )
    output["status"] = "bounded linear replay complete; physical root not attempted"
    save_json(directory / "summary.json", finite_json(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    main(args.input_directory, args.output_directory)
