"""GPU-resident Schur correction with verified full-bordered GMRES recovery."""

import math
from dataclasses import dataclass

from ehl_solver.device_krylov import device_gmres
from ehl_solver.device_preconditioner import DeviceFrozenFilmPreconditioner


@dataclass(frozen=True)
class DeviceBorderedCorrection:
    direction: object  # interior dimensionless P followed by h0/h_ref
    bordered_residual_relative: float
    schur_denominator: float
    schur_condition_indicator: float
    used_fallback: bool
    diagnostics: tuple  # (label, DeviceGMRESResult), including rejected attempts


class DeviceBorderedError(RuntimeError):
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = tuple(diagnostics)


def solve_device_bordered(
    evaluator, state, *, rtol=1e-8, restart=40, max_iterations=400,
    preconditioner="laplacian", schur_tolerance=1e-10, bordered_tolerance=None,
    memory_limit_bytes=2**30,
):
    """Mirror CPU Schur equations and accuracy ceilings; all corrections resident.

    Failed Schur solves/unsafe denominator/inaccurate reconstruction trigger an
    explicitly recorded full-bordered solve, never a CPU solve. Runtime device
    failures propagate. Tolerances use original, unpreconditioned equations.
    """
    import torch

    if (not math.isfinite(rtol) or not 0 < rtol <= 1
            or not math.isfinite(schur_tolerance) or schur_tolerance <= 0
            or (bordered_tolerance is not None and
                (not math.isfinite(bordered_tolerance) or bordered_tolerance <= 0))):
        raise ValueError("invalid Krylov/Schur/bordered tolerance")
    effective_rtol = min(rtol, 1e-2)
    tolerance = min(1e-2, max(1e-7, 2*effective_rtol)
                    if bordered_tolerance is None else bordered_tolerance)
    residual = evaluator.residual(state)
    jvp = evaluator.linearize(state)
    frozen = DeviceFrozenFilmPreconditioner(evaluator, state, strategy=preconditioner)
    h_direction = torch.zeros_like(state)
    h_direction[-1] = 1.
    column = jvp(h_direction)[:-1]
    gradient = evaluator.load_gradient
    records = []

    def pressure_operator(vector):
        direction = torch.cat((vector, vector.new_zeros(1)))
        return jvp(direction)[:-1]

    def solve(operator, rhs, inverse, label):
        result = device_gmres(
            operator, rhs, preconditioner=inverse, rtol=effective_rtol,
            restart=restart, max_iterations=max_iterations, memory_limit_bytes=memory_limit_bytes,
        )
        records.append((label, result))
        if not result.converged:
            raise ArithmeticError(f"{label}: {result.reason}, true residual "
                                  f"{result.true_relative_residual:.3e}")
        return result.solution

    def bordered_error(direction):
        return float(torch.linalg.vector_norm(jvp(direction) + residual)
                     / torch.linalg.vector_norm(residual).clamp_min(torch.finfo(state.dtype).tiny))

    try:
        x = solve(pressure_operator, -residual[:-1], frozen, "pressure_residual")
        y = solve(pressure_operator, column, frozen, "h0_response")
        schur = -torch.dot(gradient, y)
        denominator = float(schur)
        scale = max(float(torch.linalg.vector_norm(gradient)*torch.linalg.vector_norm(y)),
                    torch.finfo(state.dtype).tiny)
        indicator = abs(denominator) / scale
        if not math.isfinite(denominator) or indicator <= schur_tolerance:
            raise ArithmeticError("unsafe Schur denominator")
        dh = (-residual[-1] - torch.dot(gradient, x)) / schur
        direction = torch.cat((x - y*dh, dh.reshape(1)))
        relative = bordered_error(direction)
        if not math.isfinite(relative) or relative > tolerance:
            raise ArithmeticError("inaccurate reconstructed bordered correction")
        return DeviceBorderedCorrection(direction, relative, denominator, indicator,
                                         False, tuple(records))
    except ArithmeticError as error:
        schur_error = str(error)

    def block_inverse(rhs):
        return torch.cat((frozen(rhs[:-1]), rhs[-1:]))

    try:
        direction = solve(jvp, -residual, block_inverse, "full_bordered_fallback")
    except ArithmeticError as error:
        raise DeviceBorderedError(f"{error}; Schur attempt: {schur_error}", records) from None
    relative = bordered_error(direction)
    if not math.isfinite(relative) or relative > tolerance:
        raise DeviceBorderedError("full bordered correction failed true-residual check", records)
    return DeviceBorderedCorrection(direction, relative, math.nan, 0., True, tuple(records))
