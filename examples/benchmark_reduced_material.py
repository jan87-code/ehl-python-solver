"""Full/reduced field-LU parity in actual CPU material/JFO force Newton repairs.

All modes use cached elasticity, original gates and a +0.01 Href offset guess
at the registered accepted 200 N roots. Each worker warms once, measures three
repairs by default, and independently audits every returned state/history.
Timings are complete repair solves; OS RSS is worker-lifetime, not incremental.
This does not test cold continuation, unresolved mesh accuracy or new physics.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from ehl_solver.benchmark_200n import Circular200N
from examples.benchmark_cached_material import worker
from examples.run_load_directed_200n import finite_json
from examples.run_scientific_validation import provenance, save_json


def compare_pair(directory, index, rows):
    """Compare only independently audited roots from identical saved sources."""
    if any(r["status"] != "reference-audited nonlinear repair passed" for r in rows.values()):
        raise ValueError("two independently audited nonlinear repairs required")
    a, b = rows["full"], rows["reduced"]
    if any(a["source"][k] != b["source"][k] for k in ("source_sha256", "summary_sha256")):
        raise ValueError("full/reduced workers used different source snapshots")
    for mode, row in rows.items():
        if row["field_factorization"] != mode or row["elasticity_mode"] != "cached":
            raise ValueError("worker did not use requested factorization/cached elasticity")
        histories = row["measurement_histories"]
        if any(m["field_factorization"] != mode for history in histories
               for h in history for m in h["models"]):
            raise ValueError("Newton or active recovery lost factorization selection")
    with np.load(directory / f"grid_{index}_full/solution.npz") as full, np.load(
            directory / f"grid_{index}_reduced/solution.npz") as reduced:
        for mode, saved in (("full", full), ("reduced", reduced)):
            for name in ("pressure_pa", "theta", "film_m"):
                actual = saved[name]
                expected = rows[mode]["measurement_field_sha256"][-1][name]
                if not np.all(np.isfinite(actual)) or hashlib.sha256(
                        actual.tobytes()).hexdigest() != expected:
                    raise ValueError("saved output differs from audited measurement")
        norm = float(np.linalg.norm(full["pressure_pa"]))
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("finite nonzero pressure norm required")
        pressure = float(np.linalg.norm(full["pressure_pa"]-reduced["pressure_pa"])/norm)
        film = float(abs(full["film_m"]-reduced["film_m"]).max())
        content = float(abs(full["theta"]-reduced["theta"]).max())
    offset = abs(a["audit"]["offset_m"]-b["audit"]["offset_m"])
    tolerance_m = 1e-10*Circular200N().hertz_approach_m
    if not np.all(np.isfinite([pressure, film, content, offset])):
        raise ValueError("finite root comparison required")
    if pressure > 1e-7 or content > 1e-7 or film > tolerance_m or offset > tolerance_m:
        raise RuntimeError("full/reduced matched-force roots differ")
    return {"pressure_relative_l2_difference": pressure, "film_maximum_difference_m": film,
            "content_maximum_difference": content, "offset_difference_m": offset,
            "film_offset_tolerance_m": tolerance_m,
            "full_over_reduced_repair_runtime": a["elapsed_s"]/b["elapsed_s"]}


def main(sources, directory, repeats):
    directory.mkdir(parents=True, exist_ok=False)
    report = {"status": "incomplete", "runtime": provenance(), "cases": [],
              "scope": "actual optional field-LU Newton repairs; no cold or mesh validation"}
    save_json(directory / "summary.json", report)
    failed = False
    for index, source in enumerate(sources):
        row = {"source_directory": str(source), "modes": {}}
        report["cases"].append(row)
        for mode in ("full", "reduced"):
            target = directory / f"grid_{index}_{mode}"
            result = subprocess.run([
                sys.executable, "-m", "examples.benchmark_reduced_material",
                "--input-directories", str(source), "--output-directory", str(target),
                "--worker-mode", mode, "--repeats", str(repeats),
            ], check=False)
            measured = json.loads((target / "summary.json").read_text())
            measured["worker_returncode"] = result.returncode
            row["modes"][mode] = measured
            failed |= result.returncode != 0
            save_json(directory / "summary.json", finite_json(report))
            print(index, mode, measured["status"], flush=True)
        if all(r["worker_returncode"] == 0 for r in row["modes"].values()):
            row["comparison"] = compare_pair(directory, index, row["modes"])
            save_json(directory / "summary.json", finite_json(report))
    report["status"] = ("worker failures; incomplete" if failed
                        else "full/reduced nonlinear repairs independently audited")
    save_json(directory / "summary.json", finite_json(report))
    if failed:
        raise RuntimeError("nonlinear worker failed; original failure/partial retained")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directories", nargs="+", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--worker-mode", choices=("full", "reduced"))
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 5:
        parser.error("repeats must lie in 1..5")
    if args.worker_mode:
        if len(args.input_directories) != 1:
            parser.error("worker requires one source")
        worker(args.input_directories[0], args.output_directory, "cached", args.repeats,
               field_factorization=args.worker_mode)
    else:
        main(args.input_directories, args.output_directory, args.repeats)
