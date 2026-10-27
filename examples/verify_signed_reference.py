"""Re-audit a saved signed root against an independent numerical-Jacobian solve."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ehl_solver.benchmark_200n import Circular200N
from ehl_solver.property_coupled import PropertyCoupledSolution
from ehl_solver.property_reynolds import admissible, mass_audit
from examples.property_signed_reference import signed_reference
from examples.run_property_load_verification import save_state
from examples.run_scientific_validation import provenance, save_json


def main(source, directory):
    report = json.loads((source / "summary.json").read_text())
    case = Circular200N()
    if report["case"] != case.parameters():
        raise ValueError("registered physical case required")
    entry = next(e for e in report["accepted"] if e["trial_offset_m"] < -0.1e-6)
    problem = case.problem(0.5e-6)
    fluid = case.fluid()
    offset = entry["trial_offset_m"]
    path = source / entry["state_file"]
    with np.load(path) as a:
        p, t, h, w = [a[k] for k in ("pressure_pa", "theta", "film_m", "displacement_m")]
    scales = dict(
        length_scale_m=problem.length_scale_m,
        film_scale_m=problem.film_scale_m,
        pressure_scale_pa=problem.pressure_scale_pa,
    )
    audit = mass_audit(problem.x_m, problem.y_m, h, p, t, fluid, 2.0, **scales)
    if not admissible(audit, p, t, problem.pressure_scale_pa):
        raise RuntimeError("input root fails independent mass audit")
    seed = PropertyCoupledSolution(p, t, h, w, problem.load_n(p), audit, (), 1.0)
    status, (rp, rt, rh, rw) = signed_reference(problem, fluid, 2.0, offset, seed)
    reference_audit = mass_audit(problem.x_m, problem.y_m, rh, rp, rt, fluid, 2.0, **scales)
    relative_pressure = float(np.linalg.norm(rp - p) / np.linalg.norm(p))
    relative_film = float(np.linalg.norm(rh - h) / np.linalg.norm(h))
    if (
        not admissible(reference_audit, rp, rt, problem.pressure_scale_pa)
        or max(relative_pressure, relative_film) > 1e-7
    ):
        raise RuntimeError("independent signed reference comparison failed")
    directory.mkdir(parents=True, exist_ok=False)
    reference = PropertyCoupledSolution(
        rp, rt, rh, rw, problem.load_n(rp), reference_audit, (), 1.0
    )
    save_state(directory / "reference.npz", problem, reference, fluid)
    save_json(
        directory / "summary.json",
        {
            "status": "verified local signed-root algorithm agreement",
            "runtime": provenance(),
            "input_npz": str(path),
            "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "offset_m": offset,
            "reference_solver": status,
            "audit": reference_audit,
            "relative_pressure_error": relative_pressure,
            "relative_film_error": relative_film,
            "maximum_content_error": float(abs(rt - t).max()),
            "relative_force_error": abs(reference.load_n / seed.load_n - 1),
            "scope": "local numerical verification; not branch uniqueness or 200 N validation",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    main(args.input_directory, args.output_directory)
