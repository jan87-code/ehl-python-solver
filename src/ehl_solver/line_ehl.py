"""Stationary load-controlled plane-strain line-contact EHL/JFO solver."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import trapezoid
from scipy.optimize import brentq
from scipy.signal import find_peaks, peak_widths

from ehl_solver.backend import Backend, make_backend
from ehl_solver.constitutive import IsothermalFluid
from ehl_solver.geometry import LineContactGeometry, contact_geometry_from_config
from ehl_solver.line_cavitation import (
    VariablePropertyLineReynoldsSolution,
    audit_variable_property_line_flux,
    solve_variable_property_line_reynolds,
)
from ehl_solver.line_contact import PlaneStrainLineElasticityOperator


@dataclass(frozen=True)
class LineContactEHLCase:
    """Stationary line-contact inputs; load is exclusively a line load [N/m]."""

    x_m: NDArray[np.float64]
    geometry: LineContactGeometry
    line_load_n_m: float
    e_star_pa: float
    fluid: IsothermalFluid
    entrainment_speed_m_s: float
    length_scale_m: float
    film_scale_m: float
    pressure_scale_pa: float
    elastic_reference_length_m: float
    inlet_fractional_film_content: float = 1.0

    def __post_init__(self):
        x = np.array(self.x_m, dtype=np.float64, copy=True)
        if (
            x.ndim != 1
            or x.size < 5
            or not np.all(np.isfinite(x))
            or np.any(np.diff(x) <= 0.0)
            or not x[0] < 0.0 < x[-1]
        ):
            raise ValueError("x_m must contain at least five finite ascending coordinates [m]")
        spacing = float(x[1] - x[0])
        spacing_tolerance = 32 * np.finfo(np.float64).eps * max(
            abs(spacing), float(np.max(np.abs(x)))
        )
        if not np.allclose(np.diff(x), spacing, rtol=1e-10, atol=spacing_tolerance):
            raise ValueError("x_m must be uniformly spaced [m]")
        if not isinstance(self.geometry, LineContactGeometry):
            raise ValueError("line EHL requires LineContactGeometry, not point-contact geometry")
        if not isinstance(self.fluid, IsothermalFluid):
            raise ValueError("fluid must be an IsothermalFluid instance")
        for value, name in (
            (self.line_load_n_m, "line_load_n_m [N/m]"),
            (self.e_star_pa, "e_star_pa [Pa]"),
            (self.entrainment_speed_m_s, "entrainment_speed_m_s [m/s]"),
            (self.length_scale_m, "length_scale_m [m]"),
            (self.film_scale_m, "film_scale_m [m]"),
            (self.pressure_scale_pa, "pressure_scale_pa [Pa]"),
            (self.elastic_reference_length_m, "elastic_reference_length_m [m]"),
        ):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive finite")
        if (
            isinstance(self.inlet_fractional_film_content, bool)
            or not math.isfinite(self.inlet_fractional_film_content)
            or not 0.0 <= self.inlet_fractional_film_content <= 1.0
        ):
            raise ValueError("inlet_fractional_film_content must lie in [0,1]")
        x.setflags(write=False)
        object.__setattr__(self, "x_m", x)

    @property
    def dx_m(self) -> float:
        return float(self.x_m[1] - self.x_m[0])

    @property
    def geometry_m(self) -> NDArray[np.float64]:
        return self.geometry.h_geom_x(self.x_m)


def line_contact_ehl_case_from_config(
    config: Mapping[str, object],
) -> LineContactEHLCase:
    """Create line EHL inputs while rejecting point-contact total-load fields.

    The required selector is ``classical_line_ehl`` and the only supported
    load key is ``line_load_n_m``. In particular, ``load_n`` and ``force_n``
    are rejected rather than interpreted as per-unit-length loads.
    """
    if not isinstance(config, Mapping):
        raise ValueError("line EHL configuration must be a mapping")
    values = dict(config)
    if "load_n" in values or "force_n" in values or "target_load_n" in values:
        raise ValueError(
            "finite-contact total loads [N] are not valid here; use line_load_n_m [N/m]"
        )
    allowed = {
        "type",
        "x_m",
        "geometry",
        "line_load_n_m",
        "e_star_pa",
        "fluid",
        "entrainment_speed_m_s",
        "length_scale_m",
        "film_scale_m",
        "pressure_scale_pa",
        "elastic_reference_length_m",
        "inlet_fractional_film_content",
    }
    unknown = values.keys() - allowed
    if unknown:
        raise ValueError(f"unsupported line EHL fields: {sorted(unknown)}")
    if values.pop("type", None) != "classical_line_ehl":
        raise ValueError("line EHL configuration type must be 'classical_line_ehl'")
    geometry_config = values.pop("geometry", None)
    if not isinstance(geometry_config, Mapping):
        raise ValueError("geometry must select classical_line geometry")
    geometry = contact_geometry_from_config(geometry_config)
    if not isinstance(geometry, LineContactGeometry):
        raise ValueError("line EHL requires geometry.type='classical_line'")

    fluid_config = values.pop("fluid", None)
    if not isinstance(fluid_config, Mapping):
        raise ValueError("fluid must contain IsothermalFluid parameters")
    fluid_values = dict(fluid_config)
    fluid_fields = {
        "viscosity_pa_s",
        "density_kg_m3",
        "viscosity_index",
        "viscosity_reference_pa_s",
        "roelands_pressure_pa",
        "density_a_per_pa",
        "density_b_per_pa",
    }
    unknown_fluid = fluid_values.keys() - fluid_fields
    if unknown_fluid:
        raise ValueError(f"unsupported fluid fields: {sorted(unknown_fluid)}")
    required_fluid = {"viscosity_pa_s", "density_kg_m3"}
    if required_fluid - fluid_values.keys():
        raise ValueError(
            f"fluid requires {sorted(required_fluid - fluid_values.keys())}"
        )
    fluid_kwargs: dict[str, float] = {}
    for key, value in fluid_values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"fluid.{key} must be numeric")
        fluid_kwargs[key] = float(value)
    fluid = IsothermalFluid(**fluid_kwargs)

    required = {
        "x_m",
        "line_load_n_m",
        "e_star_pa",
        "entrainment_speed_m_s",
        "length_scale_m",
        "film_scale_m",
        "pressure_scale_pa",
        "elastic_reference_length_m",
    }
    if required - values.keys():
        raise ValueError(f"line EHL configuration requires {sorted(required - values.keys())}")
    try:
        return LineContactEHLCase(
            x_m=np.asarray(values["x_m"], dtype=np.float64),
            geometry=geometry,
            line_load_n_m=_numeric_config_value(values, "line_load_n_m"),
            e_star_pa=_numeric_config_value(values, "e_star_pa"),
            fluid=fluid,
            entrainment_speed_m_s=_numeric_config_value(
                values, "entrainment_speed_m_s"
            ),
            length_scale_m=_numeric_config_value(values, "length_scale_m"),
            film_scale_m=_numeric_config_value(values, "film_scale_m"),
            pressure_scale_pa=_numeric_config_value(values, "pressure_scale_pa"),
            elastic_reference_length_m=_numeric_config_value(
                values, "elastic_reference_length_m"
            ),
            inlet_fractional_film_content=_numeric_config_value(
                values, "inlet_fractional_film_content", default=1.0
            ),
        )
    except (TypeError, OverflowError) as error:
        raise ValueError("invalid numeric line EHL configuration") from error


def _numeric_config_value(
    values: Mapping[str, object], key: str, *, default: float | None = None
) -> float:
    value = values.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    return float(value)


@dataclass(frozen=True)
class LineEHLIteration:
    compliance_factor: float
    iteration: int
    pressure_update_relative: float
    film_defect_scaled: float
    minimum_film_m: float
    mass_maximum: float
    relative_local_balance: float
    relative_global_balance: float
    maximum_complementarity: float
    property_iterations: int


@dataclass(frozen=True)
class LineEHLSolution:
    pressure_pa: NDArray[np.float64]
    fractional_film_content: NDArray[np.float64]
    film_thickness_m: NDArray[np.float64]
    elastic_displacement_m: NDArray[np.float64]
    central_film_offset_m: float
    line_load_n_m: float
    calculated_line_load_n_m: float
    relative_load_error: float
    audit: dict[str, float]
    history: tuple[LineEHLIteration, ...]
    backend: str


class LineEHLConvergenceError(RuntimeError):
    """Retain the latest coupled line-EHL state and nonlinear history."""

    def __init__(self, message, history, last_solution):
        super().__init__(message)
        self.history = tuple(history)
        self.last_solution = last_solution


@dataclass(frozen=True)
class LineEHLLoadTrial:
    central_film_offset_m: float
    calculated_line_load_n_m: float
    relative_load_residual: float
    nonlinear_iterations: int


@dataclass(frozen=True)
class LineEHLLoadSolution:
    central_film_offset_m: float
    target_line_load_n_m: float
    calculated_line_load_n_m: float
    relative_load_error: float
    coupled_solution: LineEHLSolution
    trials: tuple[LineEHLLoadTrial, ...]


class LineEHLLoadConvergenceError(RuntimeError):
    """Preserve converged h0 trials when the requested load root is rejected."""

    def __init__(self, message, trials, last_solution):
        super().__init__(message)
        self.trials = tuple(trials)
        self.last_solution = last_solution


def _numpy_field(values, backend: Backend) -> NDArray[np.float64]:
    if backend.name == "torch":
        return np.asarray(values.detach().cpu().numpy(), dtype=np.float64)
    return np.asarray(values, dtype=np.float64)


def _line_load_n_m(pressure_pa: NDArray[np.float64], dx_m: float) -> float:
    return float(np.sum(pressure_pa, dtype=np.float64) * dx_m)


def _relative_pressure_update(
    candidate: NDArray[np.float64],
    previous: NDArray[np.float64],
    scale: float,
) -> float:
    return float(
        np.linalg.norm(candidate - previous)
        / max(np.linalg.norm(candidate), scale * 1e-12)
    )


def solve_line_ehl_at_offset(
    case: LineContactEHLCase,
    central_film_offset_m: float,
    *,
    backend: Backend | None = None,
    compliance_factors: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    film_relaxation: float = 0.35,
    pressure_tolerance: float = 1e-8,
    film_tolerance: float = 1e-8,
    mass_tolerance: float = 1e-8,
    maximum_iterations_per_stage: int = 150,
    maximum_property_iterations: int = 100,
    initial_pressure_pa: ArrayLike | None = None,
) -> LineEHLSolution:
    """Converge Reynolds/JFO and plane-strain elasticity at prescribed ``h0``."""
    if not isinstance(case, LineContactEHLCase):
        raise ValueError("case must be a LineContactEHLCase")
    if (
        isinstance(central_film_offset_m, bool)
        or not math.isfinite(central_film_offset_m)
        or central_film_offset_m <= 0.0
    ):
        raise ValueError("central_film_offset_m must be positive finite [m]")
    if (
        not compliance_factors
        or compliance_factors[0] != 0.0
        or compliance_factors[-1] != 1.0
        or any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in compliance_factors)
        or any(
            right <= left
            for left, right in zip(compliance_factors, compliance_factors[1:], strict=False)
        )
    ):
        raise ValueError("compliance_factors must increase strictly from 0 to 1")
    if not math.isfinite(film_relaxation) or not 0.0 < film_relaxation <= 1.0:
        raise ValueError("film_relaxation must lie in (0,1]")
    if not all(
        math.isfinite(value) and value > 0.0
        for value in (pressure_tolerance, film_tolerance, mass_tolerance)
    ):
        raise ValueError("nonlinear tolerances must be positive finite")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 1
        for value in (maximum_iterations_per_stage, maximum_property_iterations)
    ):
        raise ValueError("iteration budgets must be positive integers")

    selected = make_backend() if backend is None else backend
    geometry_m = case.geometry_m
    operator = PlaneStrainLineElasticityOperator(
        case.x_m,
        case.e_star_pa,
        case.elastic_reference_length_m,
        selected,
    )
    pressure = np.zeros_like(case.x_m)
    if initial_pressure_pa is not None:
        pressure = np.array(initial_pressure_pa, dtype=np.float64, copy=True)
        if (
            pressure.shape != case.x_m.shape
            or not np.all(np.isfinite(pressure))
            or np.any(pressure < 0.0)
        ):
            raise ValueError("initial_pressure_pa must be a finite nonnegative line field")
        pressure[[0, -1]] = 0.0
    use_property_continuation = initial_pressure_pa is None
    film = central_film_offset_m + geometry_m
    if not np.all(np.isfinite(film)) or np.any(film <= 0.0):
        raise ValueError("initial film must be finite and strictly positive")
    history: list[LineEHLIteration] = []
    last_solution: LineEHLSolution | None = None

    for gamma in compliance_factors:
        for iteration in range(1, maximum_iterations_per_stage + 1):
            try:
                flow: VariablePropertyLineReynoldsSolution = (
                    solve_variable_property_line_reynolds(
                        case.x_m,
                        film,
                        case.fluid,
                        case.entrainment_speed_m_s,
                        length_scale_m=case.length_scale_m,
                        film_scale_m=case.film_scale_m,
                        pressure_scale_pa=case.pressure_scale_pa,
                        inlet_fractional_film_content=case.inlet_fractional_film_content,
                        strengths=(
                            (0.0, 0.25, 0.5, 0.75, 1.0)
                            if use_property_continuation
                            else (1.0,)
                        ),
                        pressure_tolerance=min(pressure_tolerance, 1e-9),
                        mass_tolerance=mass_tolerance,
                        max_iterations_per_strength=maximum_property_iterations,
                        initial_pressure_pa=pressure,
                    )
                )
                use_property_continuation = False
                candidate_pressure = flow.pressure_pa
                displacement = _numpy_field(operator(candidate_pressure), selected)
                exact_film = central_film_offset_m + geometry_m + gamma * displacement
                if not np.all(np.isfinite(exact_film)) or np.any(exact_film <= 0.0):
                    raise ValueError("coupled film is nonfinite or not strictly positive")
                audit = audit_variable_property_line_flux(
                    case.x_m,
                    exact_film,
                    candidate_pressure,
                    flow.fractional_film_content,
                    case.fluid,
                    case.entrainment_speed_m_s,
                    length_scale_m=case.length_scale_m,
                    film_scale_m=case.film_scale_m,
                    pressure_scale_pa=case.pressure_scale_pa,
                )
            except (RuntimeError, ValueError, FloatingPointError) as error:
                raise LineEHLConvergenceError(str(error), history, last_solution) from error

            pressure_update = _relative_pressure_update(
                candidate_pressure, pressure, case.pressure_scale_pa
            )
            film_defect = float(
                np.max(np.abs(exact_film - film)) / case.film_scale_m
            )
            calculated_load = _line_load_n_m(candidate_pressure, case.dx_m)
            history.append(
                LineEHLIteration(
                    float(gamma),
                    iteration,
                    pressure_update,
                    film_defect,
                    float(np.min(exact_film)),
                    audit["mass_maximum"],
                    audit["relative_local_balance"],
                    audit["relative_global_balance"],
                    audit["maximum_complementarity"],
                    len(flow.history),
                )
            )
            last_solution = LineEHLSolution(
                candidate_pressure.copy(),
                flow.fractional_film_content.copy(),
                exact_film.copy(),
                displacement.copy(),
                float(central_film_offset_m),
                case.line_load_n_m,
                calculated_load,
                abs(calculated_load / case.line_load_n_m - 1.0),
                audit,
                tuple(history),
                f"{selected.platform}/{selected.device}",
            )
            admissible = (
                audit["mass_maximum"] <= mass_tolerance
                and audit["relative_local_balance"] <= 1e-10
                and audit["relative_global_balance"] <= 1e-10
                and audit["maximum_complementarity"] <= 1e-10
                and audit["minimum_pressure_pa"] >= -case.pressure_scale_pa * 1e-10
                and audit["minimum_fractional_film_content"] >= -1e-10
                and audit["maximum_fractional_film_content"] <= 1.0 + 1e-10
                and audit["boundary_pressure_maximum_pa"] == 0.0
            )
            if (
                pressure_update <= pressure_tolerance
                and film_defect <= film_tolerance
                and admissible
            ):
                pressure = candidate_pressure.copy()
                film = exact_film
                break

            pressure = candidate_pressure.copy()
            film += film_relaxation * (exact_film - film)
        else:
            raise LineEHLConvergenceError(
                f"line EHL compliance stage {gamma} exhausted after "
                f"{maximum_iterations_per_stage} iterations; latest={history[-1]}",
                history,
                last_solution,
            )

    if last_solution is None:
        raise RuntimeError("line EHL coupling produced no candidate")
    return last_solution


def solve_line_ehl_at_load(
    case: LineContactEHLCase,
    h0_bracket_m: tuple[float, float],
    *,
    backend: Backend | None = None,
    load_tolerance: float = 1e-5,
    h0_tolerance_m: float = 1e-12,
    monotonicity_samples: int = 3,
    maximum_root_iterations: int = 50,
    coupling_options: dict[str, Any] | None = None,
) -> LineEHLLoadSolution:
    """Close the target line load ``integral(p dx)`` in N/m through ``h0``."""
    if (
        len(h0_bracket_m) != 2
        or any(
            isinstance(value, bool) or not math.isfinite(value) or value <= 0.0
            for value in h0_bracket_m
        )
        or h0_bracket_m[0] >= h0_bracket_m[1]
    ):
        raise ValueError("h0_bracket_m must be a positive increasing SI interval")
    if not math.isfinite(load_tolerance) or load_tolerance <= 0.0:
        raise ValueError("load_tolerance must be positive finite")
    if not math.isfinite(h0_tolerance_m) or h0_tolerance_m <= 0.0:
        raise ValueError("h0_tolerance_m must be positive finite")
    if (
        not isinstance(monotonicity_samples, int)
        or isinstance(monotonicity_samples, bool)
        or monotonicity_samples < 3
    ):
        raise ValueError("monotonicity_samples must be an integer of at least three")
    if (
        not isinstance(maximum_root_iterations, int)
        or isinstance(maximum_root_iterations, bool)
        or maximum_root_iterations < 1
    ):
        raise ValueError("maximum_root_iterations must be a positive integer")
    options = dict(coupling_options or {})
    trials: dict[float, LineEHLLoadTrial] = {}
    roots: dict[float, LineEHLSolution] = {}

    def evaluate(h0: float) -> float:
        offset = float(h0)
        if offset not in trials:
            trial_options = dict(options)
            if "initial_pressure_pa" not in trial_options and roots:
                nearest = min(roots, key=lambda known: abs(known - offset))
                trial_options["initial_pressure_pa"] = roots[nearest].pressure_pa
            try:
                solution = solve_line_ehl_at_offset(
                    case,
                    offset,
                    backend=backend,
                    **trial_options,
                )
            except LineEHLConvergenceError as error:
                raise LineEHLLoadConvergenceError(
                    f"line EHL coupling failed at h0={offset:.9e} m: {error}",
                    trials.values(),
                    error.last_solution,
                ) from error
            residual = solution.calculated_line_load_n_m - case.line_load_n_m
            trials[offset] = LineEHLLoadTrial(
                offset,
                solution.calculated_line_load_n_m,
                residual / case.line_load_n_m,
                len(solution.history),
            )
            roots[offset] = solution
        return trials[offset].relative_load_residual * case.line_load_n_m

    sample_offsets = np.linspace(
        h0_bracket_m[0], h0_bracket_m[1], monotonicity_samples
    )
    sample_loads = np.array(
        [evaluate(float(offset)) + case.line_load_n_m for offset in sample_offsets]
    )
    if not np.all(np.diff(sample_loads) < 0.0):
        raise LineEHLLoadConvergenceError(
            "sampled line load must decrease strictly as h0 increases",
            trials.values(),
            roots[float(sample_offsets[-1])],
        )
    if sample_loads[0] < case.line_load_n_m or sample_loads[-1] > case.line_load_n_m:
        raise LineEHLLoadConvergenceError(
            "h0_bracket_m does not bracket the target line_load_n_m [N/m]",
            trials.values(),
            roots[float(sample_offsets[-1])],
        )

    try:
        root = float(
            brentq(
                evaluate,
                h0_bracket_m[0],
                h0_bracket_m[1],
                xtol=h0_tolerance_m,
                rtol=4.0 * np.finfo(np.float64).eps,
                maxiter=maximum_root_iterations,
            )
        )
        evaluate(root)
    except (RuntimeError, ValueError) as error:
        if isinstance(error, LineEHLLoadConvergenceError):
            raise
        last_solution = list(roots.values())[-1] if roots else None
        raise LineEHLLoadConvergenceError(
            f"line-load root solver failed: {error}",
            trials.values(),
            last_solution,
        ) from error
    root_trial = trials[root]
    if abs(root_trial.relative_load_residual) > load_tolerance:
        raise LineEHLLoadConvergenceError(
            "line-load root met h0 tolerance but failed the relative N/m load gate: "
            f"{root_trial.relative_load_residual:.3e} > {load_tolerance:.3e}",
            trials.values(),
            roots[root],
        )
    return LineEHLLoadSolution(
        root,
        case.line_load_n_m,
        root_trial.calculated_line_load_n_m,
        abs(root_trial.relative_load_residual),
        roots[root],
        tuple(trials.values()),
    )


def audit_line_elasticity_direct(
    pressure_pa: ArrayLike,
    operator: PlaneStrainLineElasticityOperator,
) -> NDArray[np.float64]:
    """Compute the same linear convolution by direct summation, not an FFT."""
    pressure = np.asarray(pressure_pa, dtype=np.float64)
    if (
        pressure.shape != (operator.size,)
        or not np.all(np.isfinite(pressure))
        or np.any(pressure < 0.0)
    ):
        raise ValueError("pressure_pa must be a finite nonnegative line field")
    kernel = _numpy_field(operator.influence_kernel, operator.backend)
    return np.convolve(pressure, kernel, mode="full")[
        operator.size - 1 : 2 * operator.size - 1
    ]


@dataclass(frozen=True)
class LineEHLFeatures:
    central_film_m: float
    minimum_film_m: float
    minimum_film_location_m: float
    peak_pressure_pa: float
    peak_pressure_location_m: float
    outlet_constriction_film_m: float
    outlet_constriction_location_m: float
    petrusevich_spike_location_m: float | None
    petrusevich_spike_width_m: float | None
    petrusevich_spike_integrated_pressure_pa_m: float | None


def analyze_line_ehl_features(
    x_m: ArrayLike,
    pressure_pa: ArrayLike,
    film_m: ArrayLike,
    hertz_half_width_m: float,
) -> LineEHLFeatures:
    """Report bulk, outlet-constriction and resolved outlet-spike quantities.

    The reported spike is the most prominent local maximum at ``x>b`` whose
    prominence is at least 0.1% of the global pressure maximum. Its width is
    the half-prominence width; its pressure integral uses the interpolated
    half-prominence interval. This is a grid diagnostic, not a feature
    acceptance criterion.
    """
    x = np.asarray(x_m, dtype=np.float64)
    pressure = np.asarray(pressure_pa, dtype=np.float64)
    film = np.asarray(film_m, dtype=np.float64)
    if (
        x.ndim != 1
        or x.size < 5
        or pressure.shape != x.shape
        or film.shape != x.shape
        or not all(np.all(np.isfinite(field)) for field in (x, pressure, film))
        or np.any(np.diff(x) <= 0.0)
        or np.any(pressure < 0.0)
        or np.any(film <= 0.0)
        or not x[0] < 0.0 < x[-1]
        or not math.isfinite(hertz_half_width_m)
        or hertz_half_width_m <= 0.0
    ):
        raise ValueError("invalid line EHL feature fields or Hertz width")
    center = int(np.argmin(np.abs(x)))
    minimum = int(np.argmin(film))
    outlet = np.flatnonzero(x >= 0.5 * hertz_half_width_m)
    if outlet.size == 0:
        raise ValueError("x_m must include points in the line-contact outlet region")
    outlet_minimum = outlet[int(np.argmin(film[outlet]))]
    peak = int(np.argmax(pressure))
    peaks, properties = find_peaks(
        pressure,
        prominence=max(float(np.max(pressure)) * 1e-3, np.finfo(np.float64).tiny),
    )
    prominence_by_peak = dict(zip(peaks, properties["prominences"], strict=True))
    candidates = [index for index in peaks if x[index] > hertz_half_width_m]
    spike_location = spike_width = spike_integral = None
    if candidates:
        spike = max(
            candidates,
            key=prominence_by_peak.__getitem__,
        )
        widths, _, left_ips, right_ips = peak_widths(
            pressure, [spike], rel_height=0.5
        )
        spike_width = float(widths[0] * (x[1] - x[0]))
        left_x = float(np.interp(left_ips[0], np.arange(x.size), x))
        right_x = float(np.interp(right_ips[0], np.arange(x.size), x))
        if right_x > left_x:
            interior = (x > left_x) & (x < right_x)
            spike_x = np.r_[left_x, x[interior], right_x]
            spike_pressure = np.r_[
                np.interp(left_x, x, pressure),
                pressure[interior],
                np.interp(right_x, x, pressure),
            ]
            spike_integral = float(trapezoid(spike_pressure, spike_x))
        spike_location = float(x[spike])
    return LineEHLFeatures(
        float(film[center]),
        float(film[minimum]),
        float(x[minimum]),
        float(pressure[peak]),
        float(x[peak]),
        float(film[outlet_minimum]),
        float(x[outlet_minimum]),
        spike_location,
        spike_width,
        spike_integral,
    )
