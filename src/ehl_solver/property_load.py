"""Bounded positive-load control for the pressure-dependent JFO/elastic CPU model.

The uniform SI reservoir and all scales stay fixed while h0 changes. Every
force probe is a separately converged, independently audited coupled root.
This initial nested reference supports positive offset brackets, not signed
approach/strong-load branch continuation or a globally unique solution claim.
"""

from dataclasses import dataclass, replace
from time import perf_counter

import numpy as np
from scipy.integrate import trapezoid
from scipy.optimize import brentq

from ehl_solver.property_coupled import PropertyCoupledSolution, solve_property_coupling
from ehl_solver.property_reynolds import admissible, mass_audit


@dataclass(frozen=True)
class PropertyLoadEvaluation:
    central_film_offset_m: float
    calculated_load_n: float
    load_residual_n: float
    relative_load_error: float
    minimum_film_m: float
    coupled_iterations: int
    maximum_film_error_m: float
    maximum_displacement_error_m: float
    audit: dict
    elapsed_s: float


@dataclass(frozen=True)
class PropertyLoadSolution:
    central_film_offset_m: float
    target_load_n: float
    calculated_load_n: float
    relative_load_error: float
    coupled_solution: PropertyCoupledSolution
    evaluations: tuple
    h0_bracket_m: tuple
    root_iterations: int


class PropertyLoadError(RuntimeError):
    """Failure with accepted probes, last valid root and any failed inner state.

    `last_solution` is the last audited force evaluation. `inner_failure`
    can carry a different last, unconverged coupled/inner candidate, at
    `failed_offset_m`; such a candidate is never a successful force probe.
    """

    def __init__(
        self,
        message,
        evaluations,
        last_solution,
        *,
        failed_offset_m=None,
        inner_failure=None,
        failed_solution=None,
    ):
        super().__init__(message)
        self.evaluations = tuple(evaluations)
        self.last_solution = last_solution
        self.failed_offset_m = failed_offset_m
        self.inner_failure = inner_failure
        self.failed_solution = failed_solution


def solve_property_at_load(
    problem,
    fluid,
    speed_m_s,
    target_load_n,
    *,
    h0_bracket_m,
    relative_load_tolerance=1e-8,
    offset_tolerance_m=1e-16,
    monotonicity_samples=3,
    maximum_evaluations=24,
    maximum_root_iterations=24,
    constitutive_strength=1.0,
    maximum_iterations_per_stage=100,
    evaluation_callback=None,
):
    """Find a positive h0 [m] for Q=integral p dx dy [N] on a fixed SI domain.

    `problem` is an existing uniform CavitatingElasticityProblem; its offset
    is replaced, with coordinates, E*, fluid, inlet, kernel conventions and
    nondimensional scales held fixed. No domain/natural-length regeneration.
    Only converged force values enter bracketed Brent iteration. Require a
    sampled decreasing force branch and a target bracket; no automatic
    expansion, pressure rescaling, clipping or tolerance relaxation.

    Up to 24 force probes; nonlinear/active/node limits remain inherited.
    Cache only scalar probe records and retain the best/last full solution
    to bound memory. Optional callback(evaluation, solution) streams complete
    fields/histories for each newly accepted probe; it is not called for cache
    hits. Callback failure stops with diagnostics. The returned force is
    independently re-integrated and checked even if the offset root converges.
    Positive offsets reflect the current cold-start coupling limitation;
    physical film feasibility is checked separately, not inferred from h0.
    """
    if not np.isfinite(target_load_n) or target_load_n <= 0:
        raise ValueError("target_load_n must be positive finite SI force")
    bounds = np.asarray(h0_bracket_m, dtype=float)
    if bounds.shape != (2,) or not np.all(np.isfinite(bounds)) or not 0 < bounds[0] < bounds[1]:
        raise ValueError("h0_bracket_m must be a positive increasing SI interval")
    if (
        not np.isfinite(relative_load_tolerance)
        or not 0 < relative_load_tolerance <= 1e-8
        or not np.isfinite(offset_tolerance_m)
        or offset_tolerance_m <= 0
    ):
        raise ValueError("invalid tolerances; relative force tolerance cannot exceed 1e-8")
    for name, value, lower, upper in (
        ("monotonicity_samples", monotonicity_samples, 3, 7),
        ("maximum_evaluations", maximum_evaluations, 3, 24),
        ("maximum_root_iterations", maximum_root_iterations, 1, 24),
        ("maximum_iterations_per_stage", maximum_iterations_per_stage, 1, 100),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or not lower <= value <= upper:
            raise ValueError(f"{name} must be an integer in {lower}..{upper}")
    if maximum_evaluations < monotonicity_samples:
        raise ValueError("evaluation budget is smaller than monotonicity sampling")
    if (
        not np.isfinite(speed_m_s)
        or speed_m_s <= 0
        or not np.isfinite(constitutive_strength)
        or not 0 <= constitutive_strength <= 1
    ):
        raise ValueError("invalid speed or constitutive strength")
    if evaluation_callback is not None and not callable(evaluation_callback):
        raise ValueError("evaluation_callback must be callable")
    cache = {}
    evaluations = []
    last = None
    best = None
    best_record = None

    def fail(message, **kwargs):
        return PropertyLoadError(message, evaluations, last, **kwargs)

    def evaluate(offset):
        nonlocal last, best, best_record
        offset = float(offset)
        if offset in cache:
            return cache[offset].load_residual_n
        if len(evaluations) >= maximum_evaluations:
            raise fail("force evaluation budget exhausted", failed_offset_m=offset)
        start = perf_counter()
        candidate = None
        try:
            trial = replace(problem, central_film_offset_m=offset)
            candidate = solve_property_coupling(
                trial,
                fluid,
                speed_m_s,
                constitutive_strength=constitutive_strength,
                maximum_iterations_per_stage=maximum_iterations_per_stage,
            )
            exact, w = trial.fields(candidate.pressure_pa)
            audit = mass_audit(
                trial.x_m,
                trial.y_m,
                candidate.film_thickness_m,
                candidate.pressure_pa,
                candidate.fractional_film_content,
                fluid,
                speed_m_s,
                length_scale_m=trial.length_scale_m,
                film_scale_m=trial.film_scale_m,
                pressure_scale_pa=trial.pressure_scale_pa,
                strength=constitutive_strength,
            )
            film_error = float(abs(exact - candidate.film_thickness_m).max())
            displacement_error = float(abs(w - candidate.elastic_displacement_m).max())
            if (
                candidate.constitutive_strength != constitutive_strength
                or not admissible(
                    audit,
                    candidate.pressure_pa,
                    candidate.fractional_film_content,
                    trial.pressure_scale_pa,
                )
                or candidate.film_thickness_m.min() <= trial.film_floor_m
                or max(film_error, displacement_error) > 1e-10 * trial.film_scale_m
            ):
                raise ValueError("coupled force candidate failed independent physical/film audit")
            calculated = float(
                trapezoid(trapezoid(candidate.pressure_pa, trial.x_m, axis=1), trial.y_m)
            )
            if (
                not np.isfinite(calculated)
                or calculated <= 0
                or abs(calculated - candidate.load_n)
                > 1e-10 * max(abs(calculated), np.finfo(float).tiny)
            ):
                raise ValueError("coupled force candidate failed independent SI force integration")
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            failed = candidate if candidate is not None else getattr(exc, "last_solution", None)
            raise fail(
                str(exc), failed_offset_m=offset, inner_failure=exc, failed_solution=failed
            ) from exc
        residual = calculated - target_load_n
        record = PropertyLoadEvaluation(
            offset,
            calculated,
            residual,
            abs(residual) / target_load_n,
            float(candidate.film_thickness_m.min()),
            len(candidate.history),
            film_error,
            displacement_error,
            audit,
            perf_counter() - start,
        )
        cache[offset] = record
        evaluations.append(record)
        last = candidate
        if best_record is None or record.relative_load_error < best_record.relative_load_error:
            best_record, best = record, candidate
        if evaluation_callback is not None:
            try:
                evaluation_callback(record, candidate)
            except Exception as exc:
                raise fail(
                    "force-evaluation callback failed", failed_offset_m=offset, inner_failure=exc
                ) from exc
        return residual

    samples = np.linspace(bounds[0], bounds[1], monotonicity_samples)
    for offset in samples:
        evaluate(offset)
    forces = np.array([cache[float(h)].calculated_load_n for h in samples])
    if not np.all(np.diff(forces) < 0):
        raise fail("sampled force is not strictly decreasing on the selected bracket")
    if best_record.relative_load_error > relative_load_tolerance and not (
        forces[0] >= target_load_n >= forces[-1]
    ):
        raise fail("selected positive offset interval does not bracket the target force")
    root_iterations = 0
    if best_record.relative_load_error > relative_load_tolerance:
        try:
            root, info = brentq(
                evaluate,
                *bounds,
                xtol=offset_tolerance_m,
                rtol=4 * np.finfo(float).eps,
                maxiter=maximum_root_iterations,
                full_output=True,
            )
            root_iterations = int(info.iterations)
            evaluate(root)
        except PropertyLoadError:
            raise
        except (RuntimeError, ValueError) as exc:
            raise fail(f"offset root failed: {exc}", inner_failure=exc) from exc
    if best_record.relative_load_error > relative_load_tolerance:
        raise fail("offset root met its stopping rule but failed the independent force tolerance")
    # Return the closest audited force state, not an interpolated/rescaled pressure field.
    return PropertyLoadSolution(
        best_record.central_film_offset_m,
        target_load_n,
        best_record.calculated_load_n,
        best_record.relative_load_error,
        best,
        tuple(evaluations),
        tuple(float(v) for v in bounds),
        root_iterations,
    )
