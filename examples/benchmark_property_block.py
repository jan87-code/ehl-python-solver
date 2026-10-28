"""Bounded isolated CPU full-bordered material/JFO preconditioner comparison.

No nonlinear steps or root acceptance. Accepted/rejected source classifications
are independently checked, and each source/mode has a separate worker process.
Cached CPU elasticity is used uniformly. Setup, one inverse action and GMRES
wall time are separate; memory is OS worker-lifetime RSS, not incremental peak.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from scipy.integrate import trapezoid
from scipy.sparse.linalg import LinearOperator, gmres, splu

from ehl_solver.benchmark_200n import Circular200N
from ehl_solver.property_block_multigrid import PropertyBlockMultigrid
from ehl_solver.property_load_newton import PropertyLoadSystem
from ehl_solver.property_reynolds import admissible, mass_audit
from ehl_solver.property_tangent_multigrid import PropertyTangentLU, PropertyTangentMultigrid
from examples.audit_transport_wavelengths import load_root
from examples.benchmark_cached_material import peak_memory
from examples.run_load_directed_200n import finite_json
from examples.run_scientific_validation import provenance, save_json

MODES = ("frozen", "material", "local", "multigrid", "material-mg", "local-mg",
         "material-reduced", "local-reduced")
DEFAULT_MODES = ("frozen", "material", "local", "multigrid")


def preconditioner_mode(mode):
    if mode not in MODES:
        raise ValueError("unknown diagnostic inverse mode")
    return {"multigrid": "frozen", "material-mg": "material", "local-mg": "local",
            "material-reduced": "material", "local-reduced": "local"}.get(mode, mode)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_source(source):
    """Audit a registered saved root or failed partial; never promote a partial."""
    summary_path = source / "summary.json"
    summary_hash = digest(summary_path)
    summary = json.loads(summary_path.read_text())
    case = Circular200N()
    if summary.get("case") != case.parameters() or summary.get("status") == "incomplete":
        raise ValueError("complete registered 200 N source required")
    grid = summary["grids"]["refined"]
    entries = [e for e in grid["attempts"] if e.get("accepted")
               and abs(e.get("load_n", 0)/200-1) <= 1e-8]
    if entries:
        state_path = source / entries[-1]["state_file"]
        state_hash = digest(state_path)
        system, state, _, info = load_root(source)
        if info["source_sha256"] != state_hash:
            raise RuntimeError("accepted input changed during audit")
        info["classification"] = "accepted 200 N source root"
    else:
        if "failed" not in summary["status"] or summary.get("target_n") != 200.:
            raise ValueError("accepted 200 N root or explicitly failed 200 N partial required")
        path = source / "failed_partial.npz"
        state_hash = digest(path)
        ny, nx = grid["shape_ny_nx"]
        problem = case.problem(.5e-6, ny=ny, nx=nx)
        with np.load(path, allow_pickle=False) as a:
            np.testing.assert_array_equal(a["x_m"], problem.x_m)
            np.testing.assert_array_equal(a["y_m"], problem.y_m)
            pressure, theta, film, displacement = [a[k].copy() for k in
                ("pressure_pa", "theta", "film_m", "displacement_m")]
        offset = summary["failed_offset_m"]
        h, w = problem.fields(pressure, central_film_offset_m=offset)
        for actual, expected in ((h, film), (w, displacement)):
            np.testing.assert_allclose(actual, expected, rtol=0,
                                       atol=1e-10*problem.film_scale_m)
        audit = mass_audit(problem.x_m, problem.y_m, film, pressure, theta, case.fluid(), 2.,
                           length_scale_m=problem.length_scale_m,
                           film_scale_m=problem.film_scale_m,
                           pressure_scale_pa=problem.pressure_scale_pa)
        force = float(trapezoid(trapezoid(pressure, problem.x_m, axis=1), problem.y_m))
        if admissible(audit, pressure, theta, problem.pressure_scale_pa) and abs(force/200-1)<=1e-8:
            raise ValueError("failed partial unexpectedly passes original gates; investigate")
        system = PropertyLoadSystem(problem, case.fluid(), 2., 200.)
        seed = SimpleNamespace(pressure_pa=pressure, fractional_film_content=theta)
        state = system.pack(seed, offset)
        info = {"source": str(path), "source_sha256": state_hash,
                "summary_sha256": summary_hash, "classification": "rejected partial; NOT a root",
                "force_n": force, "original_mass_audit": audit}
    if (digest(Path(info["source"])) != info["source_sha256"]
            or digest(summary_path) != summary_hash):
        raise RuntimeError("input changed during audit")
    info.update(summary_sha256=summary_hash, shape_ny_nx=list(system.problem.shape))
    return system, state, info


def sparse_bytes(matrix):
    return matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes


def build_inverse(system, state, mode):
    """Full bordered inverse surrogate; no fallback on MG or border failure.

    material-mg/local-mg retain the respective sparse pressure tangent and all
    mass coupling in a nonsymmetric reduced hierarchy. multigrid retains the
    earlier frozen SPD pressure/content prototype. Sparse modes factor the
    full existing frozen/material/local border. MG
    uses one fixed pressure/content action plus its approximate force Schur
    complement. These timings do not reproduce production block-Schur/retry work.
    """
    if system.preconditioner != preconditioner_mode(mode):
        raise ValueError("mode and system preconditioner must match")
    if mode in ("frozen", "material", "local"):
        lu = splu(system.frozen_matrix(state))
        payload = sparse_bytes(lu.L)+sparse_bytes(lu.U)+lu.perm_r.nbytes+lu.perm_c.nbytes
        return LinearOperator((system.size, system.size), matvec=lu.solve, dtype=np.float64), {
            "factor_payload_bytes": payload,
        }
    if mode == "multigrid":
        block = PropertyBlockMultigrid(system, state)
    elif mode.endswith("-reduced"):
        block = PropertyTangentLU(system, state)
    else:
        block = PropertyTangentMultigrid(system, state)
    column_direction = np.zeros(system.size)
    column_direction[-1] = 1.
    column = system.jacobian_vector(state, column_direction)[:-1]
    row = np.r_[np.full(system.n, system.problem.pressure_scale_pa*system.problem.dx_m
                       *system.problem.dy_m/system.target_load_n), np.zeros(system.n)]
    response = block.apply(column)
    schur = -float(row@response)
    indicator = abs(schur)/max(np.linalg.norm(row)*np.linalg.norm(response), np.finfo(float).tiny)
    if not np.isfinite(indicator) or indicator <= 1e-10:
        raise ValueError("unsafe approximate MG force Schur complement")

    def apply(rhs):
        fields = block.apply(rhs[:-1])
        offset = (rhs[-1]-row@fields)/schur
        return np.r_[fields-response*offset, offset]

    payload = sparse_bytes(block.pressure)+sparse_bytes(block.content)
    if mode == "multigrid":
        payload += sparse_bytes(block.transport)
    payload += block.full.nbytes+block.cavitated.nbytes+response.nbytes+row.nbytes
    if mode.endswith("-reduced"):
        factor = block.factor
        factors = sum(sparse_bytes(a) for a in (factor.L, factor.U))
        factors += factor.perm_r.nbytes+factor.perm_c.nbytes
        payload += sparse_bytes(block.reduced)+factors
        return LinearOperator((system.size, system.size), matvec=apply, dtype=np.float64), {
            "array_payload_bytes": payload, "factor_payload_bytes": factors,
            "approximate_schur_indicator": indicator, "reduced_unknowns": system.n,
            "field_solver": "explicit exact reduced sparse LU control; no MG fallback",
        }
    if mode != "multigrid":
        payload += sparse_bytes(block.reduced)+sparse_bytes(block.multigrid.matrix)
        payload += block.multigrid.row_scale.nbytes+block.multigrid.column_scale.nbytes
        payload += block.multigrid.diagnostics["factor_payload_bytes"]
    for level in block.multigrid.levels:
        payload += sparse_bytes(level["matrix"])
        if "diagonal" in level:
            payload += level["diagonal"].nbytes
        if "prolongation" in level:
            payload += sparse_bytes(level["prolongation"])
        if "cholesky" in level:
            payload += level["cholesky"][0].nbytes
    return LinearOperator((system.size, system.size), matvec=apply, dtype=np.float64), {
        "array_payload_bytes": payload, "approximate_schur_indicator": indicator,
        "hierarchy": block.multigrid.diagnostics,
        "payload_scope": "sum stored array bytes; may share storage; excludes workspaces",
    }


def measure_solve(system, state, inverse, rhs, *, restart, cycles):
    """Retain exhausted GMRES traces and independently measure the true residual."""
    calls, trace = [], []

    def action(v):
        calls.append(1)
        return system.jacobian_vector(state, v)

    operator = LinearOperator((system.size, system.size), matvec=action, dtype=np.float64)
    start = perf_counter()
    answer, info = gmres(operator, rhs, M=inverse, rtol=1e-9, atol=0,
                         restart=restart, maxiter=cycles, callback=trace.append,
                         callback_type="pr_norm")
    elapsed = perf_counter()-start
    error = float(np.linalg.norm(system.jacobian_vector(state, answer)-rhs))
    norm = float(np.linalg.norm(rhs))
    measured = error/norm if norm else error
    return answer, {"verified_linear_result": bool(info == 0 and np.isfinite(measured)
                                                  and measured <= 5e-9),
                    "gmres_info": int(info), "true_relative_residual": measured,
                    "iterations": len(trace), "jvp_calls": len(calls),
                    "preconditioned_residual_history": [float(v) for v in trace],
                    "gmres_wall_s": elapsed}


def worker(source, directory, mode, repeats, restart, cycles):
    directory.mkdir(parents=True, exist_ok=False)
    report = {"status": "incomplete", "runtime": provenance(), "mode": mode}
    save_json(directory / "summary.json", report)
    original, state, info = load_source(source)
    report.update(source=info, shape_ny_nx=list(original.problem.shape),
                  elasticity_mode="cached CPU for all modes", repeats=repeats,
                  restart=restart, maximum_cycles=cycles, krylov_rtol=1e-9,
                  scope="isolated full-bordered linear diagnostic; no nonlinear step")
    report["memory_after_input_audit"] = peak_memory()
    report["thread_environment"] = {k: os.environ.get(k) for k in
                                    ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}
    report["state_sha256"] = hashlib.sha256(state.tobytes()).hexdigest()
    save_json(directory / "summary.json", report)
    snapshot = state.copy()
    known = np.random.default_rng(2981).normal(size=original.size)
    known /= np.linalg.norm(known)
    reference_rhs = original.jacobian_vector(state, known)
    reference_residual = original.residual(state)

    def setup():
        system = PropertyLoadSystem(original.problem, original.base.fluid, original.base.speed,
                                    original.target_load_n,
                                    preconditioner=preconditioner_mode(mode),
                                    elasticity_mode="cached")
        inverse, metadata = build_inverse(system, state, mode)
        return system, inverse, metadata

    # Warm-up is separate, and is included in lifetime RSS, not setup samples.
    try:
        warm_system, warm_inverse, _ = setup()
        warm_inverse@reference_rhs
        del warm_system, warm_inverse
        setup_samples = []
        system, inverse = None, None
        for repeat in range(repeats):
            if repeat:
                del system, inverse
            start = perf_counter()
            system, inverse, metadata = setup()
            setup_samples.append(perf_counter()-start)
    except (ValueError, RuntimeError, FloatingPointError) as exc:
        report.update(status="inverse setup rejected", reason=str(exc), memory=peak_memory())
        np.testing.assert_array_equal(state, snapshot)
        if (digest(Path(info["source"])) != info["source_sha256"]
                or digest(source / "summary.json") != info["summary_sha256"]):
            raise RuntimeError("source modified during rejected setup") from exc
        save_json(directory / "summary.json", finite_json(report))
        return
    cache_error = float(np.linalg.norm(system.jacobian_vector(state, known)-reference_rhs)
                        /np.linalg.norm(reference_rhs))
    residual_error = float(abs(system.residual(state)-reference_residual).max())
    if cache_error > 1e-11 or residual_error > 1e-10:
        raise RuntimeError("cached complete operator disagrees with original reference")
    inverse@reference_rhs  # excluded action warm-up
    applications = []
    for _ in range(repeats):
        start = perf_counter()
        inverse@reference_rhs
        applications.append(perf_counter()-start)
    report.update(setup_samples_s=setup_samples, setup_median_s=float(np.median(setup_samples)),
                  application_samples_s=applications,
                  application_median_s=float(np.median(applications)), storage=metadata,
                  reference_jvp_relative_error=cache_error,
                  reference_residual_maximum_difference=residual_error, solves={})
    arrays = {"state": state, "known_direction": known}
    for name, rhs in (("newton", -reference_residual), ("known_vector", reference_rhs)):
        samples = []
        for _ in range(repeats):
            answer, measured = measure_solve(system, state, inverse, rhs,
                                              restart=restart, cycles=cycles)
            if name == "known_vector":
                measured["known_direction_relative_error"] = float(np.linalg.norm(answer-known))
            samples.append(measured)
        report["solves"][name] = samples
        arrays[name+"_rhs"] = rhs
        arrays[name+"_direction"] = answer
        save_json(directory / "summary.json", finite_json(report))
    np.testing.assert_array_equal(state, snapshot)
    if digest(Path(info["source"])) != info["source_sha256"] or digest(
            source / "summary.json") != info["summary_sha256"]:
        raise RuntimeError("source modified")
    np.savez_compressed(directory / "linear_vectors.npz", **arrays)
    report.update(status="bounded linear assessment complete; no new root", memory=peak_memory())
    save_json(directory / "summary.json", finite_json(report))


def compare_saved_directions(row, directory, index):
    """Compare only verified directions from workers using identical source bytes."""
    modes = row["modes"]
    identities = {(m["source"]["source_sha256"], m["source"]["summary_sha256"], m["state_sha256"])
                  for m in modes.values()}
    if len(identities) != 1:
        raise RuntimeError("workers used different source snapshots")
    comparisons = {}
    for name in ("newton", "known_vector"):
        reference = modes["local"].get("solves", {}).get(name, [])
        if not reference or not all(r["verified_linear_result"] for r in reference):
            continue
        with np.load(directory / f"source_{index}_local/linear_vectors.npz") as a:
            local = a[name+"_direction"].copy()
        comparisons[name] = {}
        for mode, report in modes.items():
            samples = report.get("solves", {}).get(name, [])
            if samples and all(r["verified_linear_result"] for r in samples):
                with np.load(directory / f"source_{index}_{mode}/linear_vectors.npz") as a:
                    error = float(np.linalg.norm(a[name+"_direction"]-local))
                comparisons[name][mode] = error/max(float(np.linalg.norm(local)), 1e-300)
    row["verified_direction_relative_differences_to_local"] = comparisons


def main(sources, directory, repeats, restart, cycles, modes=DEFAULT_MODES):
    if (not modes or "local" not in modes or len(set(modes)) != len(modes)
            or any(mode not in MODES for mode in modes)):
        raise ValueError("unique supported modes including local reference required")
    directory.mkdir(parents=True, exist_ok=False)
    report = {"status": "incomplete", "runtime": provenance(), "cases": [],
              "scope": "isolated CPU full-bordered solves; not production Newton timings"}
    errors = False
    save_json(directory / "summary.json", report)
    for index, source in enumerate(sources):
        row = {"source_directory": str(source), "modes": {}}
        report["cases"].append(row)
        for mode in modes:
            target = directory / f"source_{index}_{mode}"
            command = [sys.executable, "-m", "examples.benchmark_property_block",
                       "--input-directories", str(source), "--output-directory", str(target),
                       "--worker-mode", mode, "--repeats", str(repeats),
                       "--restart", str(restart), "--cycles", str(cycles)]
            result = subprocess.run(command, check=False)
            summary = json.loads((target / "summary.json").read_text())
            summary["worker_returncode"] = result.returncode
            row["modes"][mode] = summary
            errors |= result.returncode != 0
            save_json(directory / "summary.json", finite_json(report))
            print(index, mode, summary["status"], flush=True)
        if all(m["worker_returncode"] == 0 for m in row["modes"].values()):
            compare_saved_directions(row, directory, index)
            save_json(directory / "summary.json", finite_json(report))
    report["status"] = ("worker errors; incomplete" if errors
                        else "bounded linear comparison complete")
    save_json(directory / "summary.json", finite_json(report))
    if errors:
        raise RuntimeError("worker error; see retained summaries")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directories", nargs="+", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--worker-mode", choices=MODES)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=DEFAULT_MODES)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--restart", type=int, default=60)
    parser.add_argument("--cycles", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 5 or not 2 <= args.restart <= 80 or not 1 <= args.cycles <= 10:
        parser.error("repeats 1..5, restart 2..80 and cycles 1..10 required")
    if args.worker_mode:
        if len(args.input_directories) != 1:
            parser.error("worker requires one source")
        worker(args.input_directories[0], args.output_directory, args.worker_mode,
               args.repeats, args.restart, args.cycles)
    else:
        main(args.input_directories, args.output_directory, args.repeats, args.restart,
             args.cycles, args.modes)
