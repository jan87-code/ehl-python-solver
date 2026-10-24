"""Mathematical signed-full-film load control through the central gap offset."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import trapezoid
from scipy.optimize import brentq

from ehl_solver.coupled import CoupledReynoldsSolution, solve_coupled_reynolds_elasticity


@dataclass(frozen=True)
class LoadEvaluation:
    """One converged coupled-state evaluation in the load root search."""

    central_film_offset_m: float
    calculated_signed_load_n: float
    load_residual_n: float
    relative_load_error: float
    nonlinear_iterations: int


@dataclass(frozen=True)
class FullFilmLoadSolution:
    """Mathematical full-film root and its converged coupled state."""

    central_film_offset_m: float
    target_load_n: float
    calculated_signed_load_n: float
    load_residual_n: float
    relative_load_error: float
    coupled_solution: CoupledReynoldsSolution
    evaluations: tuple[LoadEvaluation, ...]


def _signed_load_n(
    x_m: NDArray[np.float64],
    y_m: NDArray[np.float64],
    pressure_pa: NDArray[np.float64],
) -> float:
    along_x = trapezoid(pressure_pa, x_m, axis=1)
    return float(trapezoid(along_x, y_m, axis=0))


def solve_full_film_at_load(
    x_m: ArrayLike,
    y_m: ArrayLike,
    target_load_n: float,
    radius_x_m: float,
    radius_y_m: float,
    e_star_pa: float,
    length_scale_m: float,
    film_scale_m: float,
    pressure_scale_pa: float,
    reynolds_parameter: float,
    *,
    h0_bracket_m: tuple[float, float],
    load_tolerance: float = 1e-5,
    h0_tolerance_m: float = 1e-11,
    monotonicity_samples: int = 5,
    max_root_iterations: int = 50,
    coupling_options: dict[str, Any] | None = None,
) -> FullFilmLoadSolution:
    """Find ``h0`` for a target signed pressure integral in the full-film model.

    This is explicitly a mathematical verification tool, not physical
    EHL-load control: the unconstrained Reynolds-elasticity model retains
    negative pressure, so ``Q(h0)=integral(p dx dy)`` may include tensile
    contributions. Every root-function evaluation calls the existing
    prescribed-offset solver and uses its result only after that solver
    converges. A failed nonlinear state propagates as an error; it is never
    passed to Brent's method as a successful load evaluation.

    Before root finding, the signed load is sampled at the configured number
    of equally spaced offsets. The sampled response must be strictly
    monotone and bracket the target; otherwise this function raises
    ``ValueError``. The monotonicity check is evidence only over the selected
    bracket, not a global proof. ``h0_bracket_m`` is a dimensional SI interval
    and all other geometric and material inputs follow
    :func:`solve_coupled_reynolds_elasticity`.

    Coupled solves are currently independent (no warm-start state transfer);
    callers should use modest grids and a bracket narrow enough that each
    converged state is tractable. Pressure is not clipped and no cavitation,
    load stepping, or physical pressure admissibility is implied.
    """
    x = np.asarray(x_m, dtype=np.float64)
    y = np.asarray(y_m, dtype=np.float64)
    if (
        x.ndim != 1
        or y.ndim != 1
        or min(x.size, y.size) < 3
        or not np.all(np.isfinite(x))
        or not np.all(np.isfinite(y))
        or np.any(np.diff(x) <= 0.0)
        or np.any(np.diff(y) <= 0.0)
    ):
        raise ValueError("x_m and y_m must be finite, ascending 1-D coordinate vectors")
    if not math.isfinite(target_load_n) or target_load_n == 0.0:
        raise ValueError("target_load_n must be finite and nonzero")
    if (
        len(h0_bracket_m) != 2
        or not all(math.isfinite(value) and value > 0.0 for value in h0_bracket_m)
        or h0_bracket_m[0] >= h0_bracket_m[1]
    ):
        raise ValueError("h0_bracket_m must be a positive increasing SI interval")
    if not math.isfinite(load_tolerance) or load_tolerance <= 0.0:
        raise ValueError("load_tolerance must be finite and positive")
    if not math.isfinite(h0_tolerance_m) or h0_tolerance_m <= 0.0:
        raise ValueError("h0_tolerance_m must be finite and positive")
    if (
        not isinstance(monotonicity_samples, int)
        or isinstance(monotonicity_samples, bool)
        or monotonicity_samples < 3
    ):
        raise ValueError("monotonicity_samples must be an integer of at least three")
    if (
        not isinstance(max_root_iterations, int)
        or isinstance(max_root_iterations, bool)
        or max_root_iterations < 1
    ):
        raise ValueError("max_root_iterations must be a positive integer")

    options = dict(coupling_options or {})
    evaluations: dict[float, LoadEvaluation] = {}
    solutions: dict[float, CoupledReynoldsSolution] = {}

    def evaluate(h0: float) -> float:
        if h0 not in evaluations:
            solution = solve_coupled_reynolds_elasticity(
                x,
                y,
                h0,
                radius_x_m,
                radius_y_m,
                e_star_pa,
                length_scale_m,
                film_scale_m,
                pressure_scale_pa,
                reynolds_parameter,
                **options,
            )
            signed_load = _signed_load_n(x, y, solution.pressure_pa)
            residual = signed_load - target_load_n
            evaluations[h0] = LoadEvaluation(
                central_film_offset_m=h0,
                calculated_signed_load_n=signed_load,
                load_residual_n=residual,
                relative_load_error=abs(residual) / abs(target_load_n),
                nonlinear_iterations=len(solution.history),
            )
            solutions[h0] = solution
        return evaluations[h0].load_residual_n

    sample_offsets = np.linspace(
        h0_bracket_m[0], h0_bracket_m[1], monotonicity_samples
    )
    sample_loads = np.array(
        [evaluate(float(offset)) + target_load_n for offset in sample_offsets],
        dtype=np.float64,
    )
    load_differences = np.diff(sample_loads)
    if not (np.all(load_differences > 0.0) or np.all(load_differences < 0.0)):
        raise ValueError(
            "sampled signed load is not strictly monotone over h0_bracket_m"
        )
    bracket_residuals = np.array(
        [
            evaluations[h0_bracket_m[0]].load_residual_n,
            evaluations[h0_bracket_m[1]].load_residual_n,
        ]
    )
    if bracket_residuals[0] * bracket_residuals[1] > 0.0:
        raise ValueError(
            "h0_bracket_m does not bracket target_load_n using converged states"
        )

    root = float(
        brentq(
            evaluate,
            h0_bracket_m[0],
            h0_bracket_m[1],
            xtol=h0_tolerance_m,
            rtol=4.0 * np.finfo(np.float64).eps,
            maxiter=max_root_iterations,
        )
    )
    final_residual = evaluate(root)
    final_evaluation = evaluations[root]
    if final_evaluation.relative_load_error > load_tolerance:
        raise RuntimeError(
            "full-film load root met h0 tolerance but not load tolerance: "
            f"relative_load_error={final_evaluation.relative_load_error:.3e}, "
            f"target={load_tolerance:.3e}"
        )
    ordered_evaluations = tuple(evaluations.values())
    return FullFilmLoadSolution(
        central_film_offset_m=root,
        target_load_n=target_load_n,
        calculated_signed_load_n=final_evaluation.calculated_signed_load_n,
        load_residual_n=final_residual,
        relative_load_error=final_evaluation.relative_load_error,
        coupled_solution=solutions[root],
        evaluations=ordered_evaluations,
    )
