"""Isolated-process CPU reference/cached material Newton parity at 200 N.

Timings include system construction and the complete original repair solve.
Each worker warms once, then measures three times by default; every result is independently
audited through reference convolution, original mass/force gates and history.
OS peak resident memory covers worker startup, audits, warm-up and measured
solve; it is not a device allocation measurement or incremental solver peak.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from ehl_solver.property_load_newton import PropertyLoadNewtonError
from ehl_solver.property_transfer import PropertyLoadGuess, repair_property_load_guess
from examples.audit_transport_wavelengths import load_root
from examples.run_load_directed_200n import finite_json
from examples.run_material_reservoir import audit_fields
from examples.run_property_load_verification import save_state
from examples.run_scientific_validation import provenance, save_json


def peak_memory():
    """OS worker-lifetime high-water mark, with explicit platform units."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "peak_rss", "rss", "peak_paged_pool", "paged_pool", "peak_nonpaged_pool",
                    "nonpaged_pool", "pagefile", "peak_pagefile",
                )]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        query = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
        query.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        query.restype = wintypes.BOOL
        c = Counters()
        c.cb = ctypes.sizeof(c)
        if not query(kernel.GetCurrentProcess(), ctypes.byref(c), c.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return {"peak_resident_bytes": c.peak_rss, "peak_pagefile_bytes": c.peak_pagefile,
                "method": "Windows GetProcessMemoryInfo worker-lifetime high-water marks"}
    import resource

    units = 1 if sys.platform == "darwin" else 1024
    return {"peak_resident_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*units),
            "method": "getrusage worker-lifetime resident high-water mark"}


def worker(source, directory, mode, repeats, *, field_factorization="full"):
    directory.mkdir(parents=True, exist_ok=False)
    report = {"status": "incomplete", "runtime": provenance(), "elasticity_mode": mode,
              "field_factorization": field_factorization}
    save_json(directory / "summary.json", report)
    system, state, seed, info = load_root(source)
    p, f = system.problem, system.base.fluid
    initial_offset = state[-1]*p.film_scale_m+.01*p.film_scale_m
    guess = PropertyLoadGuess(seed.pressure_pa.copy(), seed.fractional_film_content.copy(),
                              initial_offset, {"accepted": False})
    report.update(source=info, initial_offset_m=initial_offset,
                  initialization="unchanged pressure/content; +0.01 href offset; unaccepted")

    def solve():
        return repair_property_load_guess(p, f, 2., 200., guess, elasticity_mode=mode,
                                          field_factorization=field_factorization)

    try:
        warm = solve()
        audit_fields(p, warm.solution, warm.central_film_offset_m, warm.solution.history)
        del warm
        samples, audits, histories, hashes = [], [], [], []
        for _ in range(repeats):
            start = time.perf_counter()
            root = solve()
            samples.append(time.perf_counter()-start)
            audits.append(audit_fields(p, root.solution, root.central_film_offset_m,
                                       root.solution.history))
            histories.append(finite_json(root.solution.history))
            hashes.append({name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in (
                ("pressure_pa", root.solution.pressure_pa),
                ("theta", root.solution.fractional_film_content),
                ("film_m", root.solution.film_thickness_m),
            )})
            report.update(elapsed_samples_s=samples, measurement_audits=audits,
                          measurement_histories=histories, measurement_field_sha256=hashes)
            save_json(directory / "summary.json", finite_json(report))
    except PropertyLoadNewtonError as exc:
        report.update(status="rejected nonlinear solve", reason=str(exc),
                      diagnostics=finite_json(exc.diagnostics), history=finite_json(exc.history))
        if exc.last_root is not None:
            save_state(directory / "rejected_partial.npz", p, exc.last_root.solution, f)
        save_json(directory / "summary.json", report)
        raise
    report.update(elapsed_s=float(np.median(samples)), elapsed_samples_s=samples,
                  measurement_audits=audits, audit=audits[-1],
                  history=finite_json(root.solution.history))
    if (hashlib.sha256(Path(info["source"]).read_bytes()).hexdigest() != info["source_sha256"]
            or hashlib.sha256((source / "summary.json").read_bytes()).hexdigest()
            != info["summary_sha256"]):
        raise RuntimeError("source modified")
    save_state(directory / "solution.npz", p, root.solution, f)
    report.update(memory=peak_memory(), status="reference-audited nonlinear repair passed")
    save_json(directory / "summary.json", report)


def main(sources, directory, repeats):
    directory.mkdir(parents=True, exist_ok=False)
    report = {"status": "incomplete", "runtime": provenance(), "cases": [],
              "scope": "CPU full-material 200 N local repair; no cold recovery or cap change"}
    save_json(directory / "summary.json", report)
    for index, source in enumerate(sources):
        rows = {}
        for mode in ("reference", "cached"):
            target = directory / f"grid_{index}_{mode}"
            subprocess.run([sys.executable, "-m", "examples.benchmark_cached_material",
                            "--input-directories", str(source), "--output-directory", str(target),
                            "--worker-mode", mode, "--repeats", str(repeats)], check=True)
            rows[mode] = json.loads((target / "summary.json").read_text())
        with np.load(directory / f"grid_{index}_reference/solution.npz") as a, np.load(
                directory / f"grid_{index}_cached/solution.npz") as b:
            pressure_error = float(np.linalg.norm(a["pressure_pa"]-b["pressure_pa"])
                                   /np.linalg.norm(a["pressure_pa"]))
            film_error = float(abs(a["film_m"]-b["film_m"]).max())
        offset_error = abs(rows["reference"]["audit"]["offset_m"]
                           - rows["cached"]["audit"]["offset_m"])
        if pressure_error > 1e-7 or film_error > 1e-12 or offset_error > 1e-12:
            raise RuntimeError("reference/cached nonlinear roots differ")
        report["cases"].append({
            "source": str(source), "reference": rows["reference"], "cached": rows["cached"],
            "pressure_relative_l2_difference": pressure_error,
            "film_maximum_difference_m": film_error, "offset_difference_m": offset_error,
            "reference_over_cached_runtime": rows["reference"]["elapsed_s"]
                                             /rows["cached"]["elapsed_s"],
        })
        save_json(directory / "summary.json", report)
    report["status"] = "CPU reference/cached nonlinear repairs independently audited"
    save_json(directory / "summary.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directories", nargs="+", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--worker-mode", choices=("reference", "cached"))
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 5:
        parser.error("repeats must lie in 1..5")
    if args.worker_mode:
        if len(args.input_directories) != 1:
            parser.error("worker requires exactly one source")
        worker(args.input_directories[0], args.output_directory, args.worker_mode, args.repeats)
    else:
        main(args.input_directories, args.output_directory, args.repeats)
