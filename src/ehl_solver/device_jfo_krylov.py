"""GPU-resident bordered GMRES correction for the donor/JFO Newton model."""

import math
import time
from dataclasses import dataclass

from ehl_solver.device_jfo_preconditioner import (
    ExperimentalDeviceDonorJFOPreconditioner,
)
from ehl_solver.device_krylov import DeviceGMRESResult, device_gmres


@dataclass(frozen=True)
class DeviceJFOCorrection:
    direction: object
    linear_residual_relative: float
    operator_applications: int
    setup_seconds: float
    solve_seconds: float
    preconditioner: ExperimentalDeviceDonorJFOPreconditioner
    krylov: DeviceGMRESResult


class DeviceJFOKrylovError(ArithmeticError):
    def __init__(
        self, message, krylov, *, operator_applications, setup_seconds, solve_seconds,
        preconditioner,
    ):
        super().__init__(message)
        self.krylov = krylov
        self.operator_applications = operator_applications
        self.setup_seconds = setup_seconds
        self.solve_seconds = solve_seconds
        self.preconditioner = preconditioner


def solve_device_jfo_correction(
    evaluator,
    state,
    active_full,
    *,
    guide=None,
    rtol=1e-9,
    restart=60,
    max_iterations=600,
    memory_limit_bytes=2**30,
):
    """Test the selected bordered Newton system with an experimental preconditioner."""
    import torch

    if not math.isfinite(rtol) or not 0 < rtol <= 1:
        raise ValueError("rtol must be finite and in (0,1]")
    z = evaluator._vector(state)
    if guide is None:
        guide = torch.zeros_like(z)
    else:
        guide = evaluator._vector(guide)
    residual = evaluator.linear_residual(z, active_full)
    operator = evaluator.linearize(z, guide, active_full=active_full)

    def synchronize():
        if z.is_cuda:
            torch.cuda.synchronize(z.device)

    synchronize()
    setup_start = time.perf_counter()
    preconditioner = ExperimentalDeviceDonorJFOPreconditioner(
        evaluator, z, active_full, memory_limit_bytes=memory_limit_bytes
    )
    synchronize()
    setup_seconds = time.perf_counter() - setup_start
    applications = 0

    def counted_operator(vector):
        nonlocal applications
        applications += 1
        return operator(vector)

    synchronize()
    solve_start = time.perf_counter()
    result = device_gmres(
        counted_operator,
        -residual,
        preconditioner=preconditioner,
        rtol=rtol,
        restart=restart,
        max_iterations=max_iterations,
        memory_limit_bytes=memory_limit_bytes,
    )
    synchronize()
    solve_seconds = time.perf_counter() - solve_start
    correction = result.solution
    true_relative = float(
        torch.linalg.vector_norm(counted_operator(correction) + residual)
        / torch.linalg.vector_norm(residual).clamp_min(torch.finfo(z.dtype).tiny)
    )
    if not result.converged or not math.isfinite(true_relative) or true_relative > 5 * rtol:
        raise DeviceJFOKrylovError(
            f"device JFO GMRES failed true-residual gate ({true_relative:.3e}; "
            f"{result.reason})",
            result,
            operator_applications=applications,
            setup_seconds=setup_seconds,
            solve_seconds=solve_seconds,
            preconditioner=preconditioner,
        )
    return DeviceJFOCorrection(
        correction,
        true_relative,
        applications,
        setup_seconds,
        solve_seconds,
        preconditioner,
        result,
    )
