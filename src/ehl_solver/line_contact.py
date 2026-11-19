"""Dry classical line contact with plane-strain logarithmic elasticity."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.fft import next_fast_len

from ehl_solver.backend import Backend, as_float64, make_backend
from ehl_solver.contact_numerics import positive_part, project_nonnegative_fixed_sum
from ehl_solver.geometry import LineContactGeometry


@dataclass(frozen=True)
class LineContactLoads:
    """Integrated line load [N/m] and first moment per length [N]."""

    line_load_n_m: float
    moment_about_y_n: float


@dataclass(frozen=True)
class DryLineContactResult:
    pressure_pa: object
    elastic_displacement_m: object
    gap_m: object
    approach_m: float
    loads: LineContactLoads
    relative_line_load_error: float
    iterations: int
    relative_kkt_residual: float
    maximum_penetration_m: float


class DryLineContactConvergenceError(RuntimeError):
    """Line contact failed its KKT tolerance; the latest iterate is retained."""

    def __init__(self, message: str, last_result: DryLineContactResult):
        super().__init__(message)
        self.last_result = last_result


def _uniform_axis(values: ArrayLike) -> np.ndarray:
    axis = np.asarray(values, dtype=np.float64)
    if (
        axis.ndim != 1
        or axis.size < 5
        or not np.all(np.isfinite(axis))
        or np.any(np.diff(axis) <= 0)
    ):
        raise ValueError("x_m must have at least five finite ascending coordinates [m]")
    spacing = float(axis[1] - axis[0])
    tolerance = 32 * np.finfo(np.float64).eps * max(
        abs(spacing), float(np.max(np.abs(axis)))
    )
    if not np.allclose(np.diff(axis), spacing, rtol=1e-10, atol=tolerance):
        raise ValueError("x_m must be uniformly spaced [m]")
    return axis


def _log_primitive(value: float, reference_length_m: float) -> float:
    if value == 0.0:
        return 0.0
    return value * (math.log(reference_length_m / abs(value)) + 1.0)


def _line_influence_kernel(
    size: int, dx_m: float, e_star_pa: float, reference_length_m: float
) -> np.ndarray:
    """Cell-integrate ``2/(pi E*) * log(L_ref/abs(x))`` at every lag."""
    lags = np.arange(-(size - 1), size, dtype=np.float64) * dx_m
    upper = lags + 0.5 * dx_m
    lower = lags - 0.5 * dx_m
    primitive = np.vectorize(_log_primitive, otypes=[np.float64])
    integrated_log = primitive(upper, reference_length_m) - primitive(
        lower, reference_length_m
    )
    return 2.0 * integrated_log / (math.pi * e_star_pa)


class PlaneStrainLineElasticityOperator:
    """1-D cell-integrated plane-strain operator from pressure [Pa] to gap [m].

    The logarithmic datum is explicit:
    ``w(x)=2/(pi E*) integral(p(s)*log(L_ref/abs(x-s)) ds)``.
    FFTs are zero-padded to the full linear-convolution length; the reference
    length changes only the additive displacement datum.
    """

    def __init__(
        self,
        x_m: ArrayLike,
        e_star_pa: float,
        reference_length_m: float,
        backend: Backend | None = None,
        *,
        memory_limit_bytes: int = 2**28,
    ):
        if isinstance(e_star_pa, bool) or not math.isfinite(e_star_pa) or e_star_pa <= 0:
            raise ValueError("e_star_pa must be positive finite Pa")
        if (
            isinstance(reference_length_m, bool)
            or not math.isfinite(reference_length_m)
            or reference_length_m <= 0
        ):
            raise ValueError("reference_length_m must be positive finite m")
        if type(memory_limit_bytes) is not int or memory_limit_bytes < 1:
            raise ValueError("memory_limit_bytes must be a positive integer")
        self.x_m = _uniform_axis(x_m)
        self.size = self.x_m.size
        self.dx_m = float(self.x_m[1] - self.x_m[0])
        self.e_star_pa = float(e_star_pa)
        self.reference_length_m = float(reference_length_m)
        self.backend = make_backend() if backend is None else backend
        self.fft_size = next_fast_len(3 * self.size - 2)
        self.estimated_workspace_bytes = 64 * self.fft_size
        budget = memory_limit_bytes
        if self.backend.name == "torch" and self.backend.device == "cuda":
            free, _ = self.backend.xp.cuda.mem_get_info()
            budget = min(budget, free // 2)
        if self.estimated_workspace_bytes > budget:
            raise MemoryError("Line-contact FFT workspace estimate exceeds memory budget")

        kernel = _line_influence_kernel(
            self.size, self.dx_m, self.e_star_pa, self.reference_length_m
        )
        self.influence_kernel = as_float64(kernel, self.backend)
        self.kernel_spectrum = self.backend.xp.fft.rfft(
            self.influence_kernel, n=self.fft_size
        )

    def __call__(self, pressure_pa: ArrayLike):
        xp = self.backend.xp
        pressure = as_float64(pressure_pa, self.backend)
        size = int(pressure.numel()) if self.backend.name == "torch" else pressure.size
        if pressure.ndim != 1 or size != self.size:
            raise ValueError(f"pressure_pa must have shape ({self.size},)")
        if not bool(xp.all(xp.isfinite(pressure)).item()):
            raise ValueError("pressure_pa must be finite")
        spectrum = xp.fft.rfft(pressure, n=self.fft_size)
        full = xp.fft.irfft(spectrum * self.kernel_spectrum, n=self.fft_size)
        return full[self.size - 1 : 2 * self.size - 1]


def integrate_line_pressure(
    pressure_pa: ArrayLike,
    x_m: ArrayLike,
    backend: Backend | None = None,
) -> LineContactLoads:
    """Integrate 1-D pressure to line load [N/m], never total force [N]."""
    selected = make_backend() if backend is None else backend
    xp = selected.xp
    pressure = as_float64(pressure_pa, selected)
    x = as_float64(x_m, selected)
    x_size = int(x.numel()) if selected.name == "torch" else x.size
    pressure_size = int(pressure.numel()) if selected.name == "torch" else pressure.size
    if pressure.ndim != 1 or x.ndim != 1 or pressure_size != x_size or x_size < 2:
        raise ValueError("pressure_pa and x_m must be matching 1-D arrays")
    differences = xp.diff(x)
    dx = float(differences[0].item())
    dx_reference = xp.full_like(differences, dx) if selected.name == "torch" else dx
    is_uniform = xp.allclose(differences, dx_reference, rtol=1e-10, atol=0)
    if (
        not math.isfinite(dx)
        or dx <= 0
        or not bool(is_uniform.item() if hasattr(is_uniform, "item") else is_uniform)
    ):
        raise ValueError("x_m must be uniformly spaced and ascending")
    if not bool(xp.all(xp.isfinite(pressure)).item()) or bool(xp.any(pressure < 0).item()):
        raise ValueError("pressure_pa must be finite and nonnegative")
    line_load = xp.sum(pressure) * dx
    moment_about_y = xp.sum(pressure * x) * dx
    return LineContactLoads(float(line_load.item()), float(moment_about_y.item()))


def solve_dry_line_contact(
    x_m: ArrayLike,
    geometry: LineContactGeometry,
    line_load_n_m: float,
    e_star_pa: float,
    *,
    reference_length_m: float,
    backend: Backend | None = None,
    tolerance: float = 2e-6,
    maximum_iterations: int = 20000,
    memory_limit_bytes: int = 2**28,
) -> DryLineContactResult:
    """Solve dry load-controlled contact using pressure per unit axial length.

    The fixed-load constraint is ``sum(p_i)*dx = line_load_n_m``. The gap is
    ``h_geom_x + w - approach`` with the plane-strain logarithmic datum given
    by ``reference_length_m``. This solver is distinct from finite 2-D contact
    and from the existing 3-D point-contact elasticity operator.
    """
    if (
        isinstance(line_load_n_m, bool)
        or not math.isfinite(line_load_n_m)
        or line_load_n_m <= 0
    ):
        raise ValueError("line_load_n_m must be positive finite N/m")
    if isinstance(tolerance, bool) or not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be positive finite")
    if type(maximum_iterations) is not int or maximum_iterations < 1:
        raise ValueError("maximum_iterations must be a positive integer")

    x = _uniform_axis(x_m)
    geometry_function = getattr(geometry, "h_geom_x", None)
    if not callable(geometry_function):
        raise ValueError("geometry must provide h_geom_x(x_m)")
    geometry_m = np.asarray(geometry_function(x), dtype=np.float64)
    if geometry_m.shape != x.shape or not np.all(np.isfinite(geometry_m)):
        raise ValueError(f"geometry.h_geom_x must return a finite field with shape {x.shape}")

    selected = make_backend() if backend is None else backend
    xp = selected.xp
    dx = float(x[1] - x[0])
    operator = PlaneStrainLineElasticityOperator(
        x,
        e_star_pa,
        reference_length_m,
        selected,
        memory_limit_bytes=memory_limit_bytes,
    )
    geometry_device = as_float64(geometry_m, selected)
    load = float(line_load_n_m)
    size = int(x.size)
    cell_line_load = load / size
    if selected.name == "torch":
        q = xp.full((size,), cell_line_load, dtype=xp.float64, device=selected.device)
    else:
        q = np.full(size, cell_line_load, dtype=np.float64)
    accelerated = q.clone() if selected.name == "torch" else q.copy()
    lipschitz = float(xp.max(xp.abs(operator.kernel_spectrum)).item()) / dx
    if not math.isfinite(lipschitz) or lipschitz <= 0:
        raise RuntimeError("Line-elasticity operator produced an invalid contact step bound")
    step = 1.0 / lipschitz
    scale = max(float(np.max(np.abs(geometry_m))), np.finfo(np.float64).tiny)
    active_threshold = load * 1e-12 / size
    momentum = 1.0
    last_result = None

    for iteration in range(1, maximum_iterations + 1):
        pressure_at_accelerated = accelerated / dx
        displacement_at_accelerated = operator(pressure_at_accelerated)
        gradient = displacement_at_accelerated + geometry_device
        updated = project_nonnegative_fixed_sum(
            accelerated - step * gradient, load, selected
        )
        next_momentum = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * momentum**2))
        accelerated = updated + ((momentum - 1.0) / next_momentum) * (updated - q)
        q, momentum = updated, next_momentum

        if iteration % 10 != 0 and iteration != maximum_iterations:
            continue
        pressure = q / dx
        displacement = operator(pressure)
        stationarity = geometry_device + displacement
        approach = float((xp.sum(q * stationarity) / load).item())
        gap = stationarity - approach
        active = q > active_threshold
        active_error = (
            float(xp.max(xp.abs(gap[active])).item()) if bool(xp.any(active).item()) else math.inf
        )
        inactive = ~active
        penetration = (
            float(xp.max(positive_part(-gap[inactive], selected)).item())
            if bool(xp.any(inactive).item())
            else 0.0
        )
        residual = max(active_error, penetration) / max(scale, abs(approach))
        loads = integrate_line_pressure(pressure, x, selected)
        last_result = DryLineContactResult(
            pressure,
            displacement,
            gap,
            approach,
            loads,
            loads.line_load_n_m / load - 1.0,
            iteration,
            residual,
            float(xp.max(positive_part(-gap, selected)).item()),
        )
        if residual <= tolerance:
            return last_result

    raise DryLineContactConvergenceError(
        f"Dry line contact did not reach KKT tolerance {tolerance:.3e} in "
        f"{maximum_iterations} iterations; residual="
        f"{last_result.relative_kkt_residual:.3e}",
        last_result,
    )
