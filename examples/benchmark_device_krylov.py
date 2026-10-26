"""Synchronized signed full-film CPU/CUDA Krylov benchmark (new output paths)."""

import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

from ehl_solver.backend import make_backend
from ehl_solver.device import DeviceFullFilmEvaluator
from ehl_solver.device_bordered import solve_device_bordered
from ehl_solver.device_krylov import device_gmres
from ehl_solver.device_newton import solve_device_full_film_load
from ehl_solver.device_preconditioner import DeviceFrozenFilmPreconditioner
from ehl_solver.monolithic import (
    BorderedLinearSolveError,
    BorderedReynoldsElasticityProblem,
    MonolithicNewtonError,
    build_frozen_film_preconditioner,
    solve_bordered_correction,
    solve_monolithic_full_film_load,
)
from ehl_solver.reynolds import solve_reynolds_pressure


def case(n, compliance=1., e_star=1e8):
    x = np.linspace(-.5e-3, .75e-3, n)
    y = np.linspace(-.5e-3, .5e-3, n)
    xx, yy = np.meshgrid(x, y)
    geometry = xx**2 / .02 + yy**2 / .04
    dx, dy = (x[1]-x[0]) / 5e-4, (y[1]-y[0]) / 5e-4

    def pressure(offset):
        return solve_reynolds_pressure(
            (offset + geometry) / 20e-6, dx, dy, 12., rtol=1e-12, atol=1e-14
        ).pressure

    target = float(pressure(20e-6).sum() * 21250. * (x[1]-x[0]) * (y[1]-y[0]))
    problem = BorderedReynoldsElasticityProblem(
        x, y, .01, .02, e_star, 5e-4, 20e-6, 21250., 12., target,
        compliance_factor=compliance, reynolds_residual_scale=100.,
        krylov_rtol=1e-8, krylov_max_iterations=300, gmres_restart=80,
    )
    initial = pressure(18e-6)
    return problem, initial, np.r_[initial[1:-1, 1:-1].ravel(), .9]


def measured(call, synchronize=lambda: None, repeats=1, warmup=0):
    for _ in range(warmup):
        call()
    samples = []
    for _ in range(repeats):
        synchronize()
        start = time.perf_counter()
        result = call()
        synchronize()
        samples.append((time.perf_counter()-start)*1000)
    return result, statistics.median(samples)


def compare(n, repeats, max_steps, newton_max_size, output_directory, skip_cpu_linear=False,
            matched_torch_cpu=False):
    backend = make_backend("torch", "cuda")
    torch = backend.xp
    sync = torch.cuda.synchronize
    problem, initial, state = case(n)
    restart = 40
    # CPU SciPy maxiter counts cycles; device GMRES counts Arnoldi steps.
    problem.gmres_restart = restart
    problem.krylov_max_iterations = max_steps // restart
    state_d, transfer_ms = measured(
        lambda: torch.as_tensor(state, dtype=torch.float64, device="cuda"), sync,
        repeats=repeats, warmup=1,
    )
    evaluator, setup_ms = measured(lambda: DeviceFullFilmEvaluator(problem, backend), sync)
    if skip_cpu_linear:
        frozen_cpu, cpu_pre_setup = None, None
    else:
        frozen_cpu, cpu_pre_setup = measured(
            lambda: build_frozen_film_preconditioner(problem, state)
        )
    frozen_gpu, gpu_pre_setup = measured(
        lambda: DeviceFrozenFilmPreconditioner(evaluator, state_d), sync
    )
    residual = problem.residual(state)
    rhs = -residual[:-1]
    rhs_d = -evaluator.residual(state_d)[:-1]
    cpu_apply = None
    if frozen_cpu is not None:
        _, cpu_apply = measured(lambda: frozen_cpu.operator @ rhs, repeats=repeats, warmup=1)
    _, gpu_apply = measured(lambda: frozen_gpu(rhs_d), sync, repeats=repeats, warmup=1)

    def pressure_cpu(v):
        return problem.jacobian_vector(state, np.r_[v, 0.])[:-1]

    linearized = evaluator.linearize(state_d)

    def pressure_gpu(v):
        return linearized(torch.cat((v, v.new_zeros(1))))[:-1]

    cpu_operator = LinearOperator((rhs.size, rhs.size), matvec=pressure_cpu, dtype=np.float64)

    def solve_cpu():
        counts = []
        solution, info = gmres(
            cpu_operator, rhs, M=frozen_cpu.operator, rtol=1e-8, atol=0., restart=restart,
            maxiter=max_steps//restart, callback=counts.append, callback_type="pr_norm",
        )
        return solution, info, len(counts)

    cpu_linear, cpu_ms = (None, None, None), None
    if not skip_cpu_linear:
        cpu_linear, cpu_ms = measured(solve_cpu, repeats=repeats, warmup=1)
    torch.cuda.reset_peak_memory_stats()
    gpu_linear, gpu_ms = measured(
        lambda: device_gmres(pressure_gpu, rhs_d, preconditioner=frozen_gpu, rtol=1e-8,
                             restart=restart, max_iterations=max_steps),
        sync, repeats=repeats, warmup=1,
    )
    cpu_true = None if skip_cpu_linear else float(
        np.linalg.norm(pressure_cpu(cpu_linear[0])-rhs)/np.linalg.norm(rhs)
    )
    gpu_true = float(torch.linalg.vector_norm(pressure_gpu(gpu_linear.solution)-rhs_d)
                     / torch.linalg.vector_norm(rhs_d))
    entry = {
        "n": n, "restart": restart, "max_arnoldi_steps": max_steps,
        "target_load_n": problem.target_load_n,
        "load_residual_scale_n": problem.load_residual_scale_n,
        "cpu_gmres_ms": cpu_ms, "gpu_gmres_ms": gpu_ms,
        "cpu_iterations": cpu_linear[2], "gpu_iterations": gpu_linear.iterations,
        "cpu_info": cpu_linear[1], "gpu_reason": gpu_linear.reason,
        "cpu_true_relative_residual": cpu_true, "gpu_true_relative_residual": gpu_true,
        "linear_speedup": cpu_ms/gpu_ms if cpu_linear[1] == 0 and gpu_linear.converged else None,
        "cpu_preconditioner": "scipy_ilu", "gpu_preconditioner": "scaled_laplacian_dst",
        "cpu_preconditioner_setup_ms": cpu_pre_setup, "gpu_preconditioner_setup_ms": gpu_pre_setup,
        "cpu_preconditioner_apply_ms": cpu_apply, "gpu_preconditioner_apply_ms": gpu_apply,
        "initial_h2d_ms": transfer_ms, "evaluator_setup_ms": setup_ms,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    if cpu_linear[1] == 0 and gpu_linear.converged:
        difference = gpu_linear.solution.cpu().numpy() - cpu_linear[0]
        entry["linear_solution_relative_l2"] = float(np.linalg.norm(difference)
                                                     / np.linalg.norm(cpu_linear[0]))
    print(json.dumps({"linear_complete": entry}), flush=True)
    checkpoint = output_directory / f"grid_{n}.json"
    checkpoint.write_text(json.dumps(entry, indent=2))
    options = {"reynolds_tolerance": 1e-10, "load_tolerance": 1e-10}
    if matched_torch_cpu:
        matching = DeviceFullFilmEvaluator(problem, make_backend("torch", "cpu"))
        matching_state = torch.as_tensor(state, dtype=torch.float64)
        matching_jvp = matching.linearize(matching_state)
        matching_pre = DeviceFrozenFilmPreconditioner(matching, matching_state)

        def matching_operator(v):
            return matching_jvp(torch.cat((v, v.new_zeros(1))))[:-1]

        matching_linear, entry["matched_cpu_gmres_ms"] = measured(
            lambda: device_gmres(matching_operator, torch.as_tensor(rhs, dtype=torch.float64),
                                 preconditioner=matching_pre, rtol=1e-8, restart=restart,
                                 max_iterations=max_steps), repeats=repeats, warmup=1,
        )
        entry["matched_cpu_linear_iterations"] = matching_linear.iterations
        entry["matched_cpu_linear_true_relative_residual"] = matching_linear.true_relative_residual
        assert matching_linear.converged, matching_linear.reason
        matching_solution, entry["matched_cpu_solver_ms"] = measured(
            lambda: solve_device_full_film_load(matching, matching_state, restart=restart,
                                               max_krylov_iterations=max_steps, **options)
        )
    cpu_solution = None
    if n <= newton_max_size:
        try:
            cpu_correction, entry["cpu_bordered_ms"] = measured(
                lambda: solve_bordered_correction(problem, state)
            )
            cpu_delta = np.r_[cpu_correction.pressure_correction.ravel(),
                              cpu_correction.h0_correction_m/20e-6]
        except BorderedLinearSolveError as error:
            entry["cpu_bordered_failure"] = str(error)
            cpu_delta = None
        gpu_correction, entry["gpu_bordered_ms"] = measured(
            lambda: solve_device_bordered(evaluator, state_d, restart=restart,
                                          max_iterations=max_steps), sync
        )
        if cpu_delta is not None:
            entry["bordered_correction_relative_l2"] = float(np.linalg.norm(
                gpu_correction.direction.cpu().numpy()-cpu_delta)/np.linalg.norm(cpu_delta))
        try:
            cpu_solution, entry["cpu_solver_ms"] = measured(
                lambda: solve_monolithic_full_film_load(problem, initial, 18e-6, **options)
            )
            entry["cpu_newton_iterations"] = len(cpu_solution.history)
            entry["cpu_nonlinear_history"] = [r.reynolds_residual_scaled
                                               for r in cpu_solution.history]
            entry["cpu_newton_krylov_iterations"] = sum(
                r.krylov_iterations for r in cpu_solution.history
            )
        except MonolithicNewtonError as error:
            entry["cpu_solver_failure"] = str(error)
            entry["cpu_solver_ms"] = None
    else:
        entry["cpu_solver_ms"] = None
        entry["cpu_solver_limit"] = "full CPU Newton limited by --newton-max-size"
    torch.cuda.reset_peak_memory_stats()
    checkpoint.write_text(json.dumps(entry, indent=2))
    gpu_solution, entry["gpu_solver_ms"] = measured(
        lambda: solve_device_full_film_load(evaluator, state_d, restart=restart,
                                           max_krylov_iterations=max_steps, **options), sync
    )
    entry["gpu_newton_iterations"] = len(gpu_solution.history)
    entry["gpu_nonlinear_history"] = [r.reynolds_residual_scaled for r in gpu_solution.history]
    entry["gpu_newton_krylov_iterations"] = sum(
        r.iterations for h in gpu_solution.history for r in h.krylov_diagnostics
    )
    entry["gpu_mean_newton_ms"] = entry["gpu_solver_ms"]/max(len(gpu_solution.history), 1)
    entry["peak_allocated_bytes"] = max(entry["peak_allocated_bytes"],
                                         torch.cuda.max_memory_allocated())
    entry["peak_reserved_bytes"] = max(entry["peak_reserved_bytes"],
                                        torch.cuda.max_memory_reserved())
    final, entry["final_d2h_ms"] = measured(lambda: gpu_solution.state.cpu().numpy(), sync)
    entry["gpu_end_to_end_ms"] = (transfer_ms + setup_ms + entry["gpu_solver_ms"]
                                   + entry["final_d2h_ms"])
    # Independent original CPU equations and geometry/FFT reconstruction.
    audit = problem.residual(final)
    entry["final_cpu_reynolds_rms"] = float(
        np.linalg.norm(audit[:-1])/np.sqrt(problem.pressure_size)
    )
    entry["final_cpu_scaled_load_error"] = float(audit[-1])
    entry["final_h0_m"] = gpu_solution.central_film_offset_m
    entry["minimum_film_m"] = float(gpu_solution.film_thickness_m.min())
    entry["final_load_error_n"] = gpu_solution.calculated_load_n - problem.target_load_n
    assert entry["final_cpu_reynolds_rms"] <= options["reynolds_tolerance"]
    assert abs(entry["final_cpu_scaled_load_error"]) <= options["load_tolerance"]
    if cpu_solution is not None:
        entry["solver_speedup"] = entry["cpu_solver_ms"]/entry["gpu_solver_ms"]
        entry["end_to_end_speedup"] = entry["cpu_solver_ms"]/entry["gpu_end_to_end_ms"]
        gpu_pressure = gpu_solution.pressure_dimensionless.cpu().numpy()
        entry["final_pressure_relative_l2"] = float(np.linalg.norm(
            gpu_pressure-cpu_solution.pressure_dimensionless)/np.linalg.norm(cpu_solution.pressure_dimensionless))
        entry["final_h0_difference_m"] = abs(gpu_solution.central_film_offset_m
                                             - cpu_solution.central_film_offset_m)
    if matched_torch_cpu:
        entry["matched_linear_speedup"] = entry["matched_cpu_gmres_ms"]/entry["gpu_gmres_ms"]
        entry["matched_solver_speedup"] = entry["matched_cpu_solver_ms"]/entry["gpu_solver_ms"]
        entry["matched_pressure_relative_l2"] = float(np.linalg.norm(
            matching_solution.pressure_dimensionless.numpy()
            - gpu_solution.pressure_dimensionless.cpu().numpy()
        ) / np.linalg.norm(matching_solution.pressure_dimensionless.numpy()))
    fields = {"state": final, "x_m": problem.x, "y_m": problem.y,
              "pressure_dimensionless": gpu_solution.pressure_dimensionless.cpu().numpy(),
              "film_m": gpu_solution.film_thickness_m.cpu().numpy(),
              "displacement_m": gpu_solution.elastic_displacement_m.cpu().numpy()}
    if cpu_solution is not None:
        fields["cpu_pressure_dimensionless"] = cpu_solution.pressure_dimensionless
    np.savez_compressed(output_directory / f"fields_{n}.npz", **fields)
    checkpoint.write_text(json.dumps(entry, indent=2))
    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[33])
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-krylov-steps", type=int, default=160)
    parser.add_argument("--newton-max-size", type=int, default=129)
    parser.add_argument("--skip-cpu-linear", action="store_true",
                        help="GPU timings only, following a separately retained CPU budget attempt")
    parser.add_argument("--matched-torch-cpu", action="store_true",
                        help="Also measure identical cached solver/preconditioning on torch CPU")
    args = parser.parse_args()
    if (args.repeats < 1 or args.max_krylov_steps < 40 or args.max_krylov_steps % 40
            or any(n < 3 for n in args.sizes)):
        parser.error("repeats >=1 and max-krylov-steps a positive multiple of 40 required")
    args.output_directory.mkdir(parents=True, exist_ok=False)
    report = {"python": platform.python_version(), "numpy": np.__version__,
              "threads": {k: os.environ.get(k) for k in
                          ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")}, "cases": []}
    if args.compare:
        backend = make_backend("torch", "cuda")
        report.update(torch=backend.xp.__version__, cuda=backend.xp.version.cuda,
                      device=backend.xp.cuda.get_device_name(), os=platform.platform(),
                      linear_warmups=1, linear_repeats=args.repeats,
                      nonlinear_repeats=1, nonlinear_tolerance=1e-10,
                      total_vram_bytes=backend.xp.cuda.mem_get_info()[1])
    for n in args.sizes:
        if args.compare:
            entry = compare(n, args.repeats, args.max_krylov_steps, args.newton_max_size,
                            args.output_directory, args.skip_cpu_linear, args.matched_torch_cpu)
            report["cases"].append(entry)
            (args.output_directory / "summary.json").write_text(json.dumps(report, indent=2))
            print(json.dumps(entry), flush=True)
            continue
        problem, initial, state = case(n)
        start = time.perf_counter()
        correction = solve_bordered_correction(problem, state)
        linear_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        solution = solve_monolithic_full_film_load(
            problem, initial, 18e-6, reynolds_tolerance=1e-7, load_tolerance=1e-7
        )
        wall_ms = (time.perf_counter() - start) * 1000
        entry = {"n": n, "cpu_correction_ms": linear_ms, "cpu_solver_ms": wall_ms,
                 "newton_iterations": len(solution.history),
                 "krylov_iterations": correction.pressure_krylov_iterations,
                 "bordered_residual": correction.bordered_residual_relative,
                 "h0_m": solution.central_film_offset_m,
                 "load_error_n": solution.load_residual_n}
        report["cases"].append(entry)
        (args.output_directory / "summary.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(entry), flush=True)


if __name__ == "__main__":
    main()
