"""Independent field audit of a rejected mesh trial; never promotes it to a root."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid

from ehl_solver.benchmark_200n import Circular200N
from ehl_solver.property_reynolds import admissible, mass_audit
from examples.run_scientific_validation import provenance, save_json


def main(source, directory):
    report = json.loads((source / "summary.json").read_text())
    grid = report["grids"]["refined"]
    case = Circular200N()
    if (report["case"] != case.parameters() or "failed" not in report["status"]
            or any(a["accepted"] for a in grid["attempts"])):
        raise ValueError("rejected registered trial with no accepted target root required")
    directory.mkdir(parents=True, exist_ok=False)
    ny, nx = grid["shape_ny_nx"]
    p = case.problem(.5e-6, ny=ny, nx=nx,
                     maximum_physical_nodes=report.get("maximum_physical_nodes", 20000))
    path = source / "failed_partial.npz"
    with np.load(path) as a:
        np.testing.assert_array_equal(a["x_m"], p.x_m)
        np.testing.assert_array_equal(a["y_m"], p.y_m)
        pressure, theta, film, displacement = [a[k].copy() for k in
                                              ("pressure_pa", "theta", "film_m", "displacement_m")]
    h, w = p.fields(pressure, central_film_offset_m=report["failed_offset_m"])
    audit = mass_audit(p.x_m, p.y_m, film, pressure, theta, case.fluid(), 2.,
                       length_scale_m=p.length_scale_m, film_scale_m=p.film_scale_m,
                       pressure_scale_pa=p.pressure_scale_pa)
    force = float(trapezoid(trapezoid(pressure, p.x_m, axis=1), p.y_m))
    original_pass = admissible(audit, pressure, theta, p.pressure_scale_pa)
    if original_pass and abs(force/report["target_n"]-1) <= 1e-8:
        raise RuntimeError("rejected partial unexpectedly passes original gates; investigate")
    save_json(directory / "summary.json", {
        "status": "rejected partial audited; NOT a converged mesh solution",
        "runtime": provenance(), "state_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "original_field_mass_gates_pass": original_pass, "force_n": force,
        "relative_force_error": abs(force/report["target_n"]-1), "audit": audit,
        "film_reference_error_m": float(abs(h-film).max()),
        "displacement_reference_error_m": float(abs(w-displacement).max()),
        "reason": grid["stall"]["reason"],
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    main(args.input_directory, args.output_directory)
