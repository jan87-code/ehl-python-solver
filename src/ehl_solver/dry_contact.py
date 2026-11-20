"""Load-controlled frictionless dry contact using zero-padded FFT elasticity."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ehl_solver.backend import Backend, as_float64, make_backend
from ehl_solver.contact import ContactLoadMoments, integrate_contact_pressure
from ehl_solver.contact_numerics import positive_part, project_nonnegative_fixed_sum
from ehl_solver.elasticity import ElasticityOperator
from ehl_solver.geometry import ContactGeometry


@dataclass(frozen=True)
class DryContactResult:
    pressure_pa: object
    elastic_displacement_m: object
    gap_m: object
    approach_m: float
    loads: ContactLoadMoments
    iterations: int
    relative_kkt_residual: float
    maximum_penetration_m: float


class DryContactConvergenceError(RuntimeError):
    """Dry contact failed its KKT tolerance; the latest iterate is retained."""

    def __init__(self, message: str, last_result: DryContactResult):
        super().__init__(message)
        self.last_result = last_result


def _uniform_axis(values: ArrayLike, name: str) -> np.ndarray:
    axis = np.asarray(values, dtype=np.float64)
    if (
        axis.ndim != 1
        or axis.size < 5
        or not np.all(np.isfinite(axis))
        or np.any(np.diff(axis) <= 0)
    ):
        raise ValueError(f"{name} must have at least five finite ascending coordinates [m]")
    spacing = float(axis[1] - axis[0])
    tolerance = 32 * np.finfo(np.float64).eps * max(
        abs(spacing), float(np.max(np.abs(axis)))
    )
    if not np.allclose(np.diff(axis), spacing, rtol=1e-10, atol=tolerance):
        raise ValueError(f"{name} must be uniformly spaced [m]")
    return axis


def solve_dry_contact(
    x_m: ArrayLike,
    y_m: ArrayLike,
    geometry: ContactGeometry,
    load_n: float,
    e_star_pa: float,
    *,
    backend: Backend | None = None,
    tolerance: float = 2e-6,
    maximum_iterations: int = 20000,
    memory_limit_bytes: int = 2**30,
) -> DryContactResult:
    """Solve frictionless non-adhesive contact for a prescribed normal load.

    Coordinates are cell centers and arrays use ``(ny,nx)``. Pressure is
    constrained nonnegative with ``sum(p)*dx*dy = load_n``. Positive elastic
    displacement opens the gap, so ``gap = h_geom + w - approach``. The
    projected accelerated-gradient solve enforces nonnegative gap and pressure
    complementarity; this is a dry-contact verification tool, not an EHL/JFO
    solve.
    """
    if isinstance(load_n, bool) or not math.isfinite(load_n) or load_n <= 0:
        raise ValueError("load_n must be positive finite N")
    if isinstance(e_star_pa, bool) or not math.isfinite(e_star_pa) or e_star_pa <= 0:
        raise ValueError("e_star_pa must be positive finite Pa")
    if isinstance(tolerance, bool) or not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be positive finite")
    if (
        type(maximum_iterations) is not int
        or maximum_iterations < 1
        or type(memory_limit_bytes) is not int
        or memory_limit_bytes < 1
    ):
        raise ValueError("maximum_iterations and memory_limit_bytes must be positive integers")

    x, y = _uniform_axis(x_m, "x_m"), _uniform_axis(y_m, "y_m")
    geometry_function = getattr(geometry, "h_geom", None)
    if not callable(geometry_function):
        raise ValueError("geometry must provide h_geom(x_m, y_m)")
    geometry_m = np.asarray(geometry_function(x, y), dtype=np.float64)
    shape = (y.size, x.size)
    if geometry_m.shape != shape or not np.all(np.isfinite(geometry_m)):
        raise ValueError(f"geometry.h_geom must return a finite field with shape (ny,nx)={shape}")

    selected = make_backend() if backend is None else backend
    xp = selected.xp
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
    cell_area = dx * dy
    operator = ElasticityOperator(
        shape,
        dx,
        dy,
        e_star_pa,
        selected,
        memory_limit_bytes=memory_limit_bytes,
    )
    geometry_device = as_float64(geometry_m, selected)
    load = float(load_n)
    pressure_cells = int(np.prod(shape))
    cell_load = load / pressure_cells
    q = xp.full((pressure_cells,), cell_load, dtype=xp.float64, device=selected.device) \
        if selected.name == "torch" else np.full(pressure_cells, cell_load, dtype=np.float64)
    accelerated = q.clone() if selected.name == "torch" else q.copy()
    momentum = 1.0
    lipschitz = float(xp.max(xp.abs(operator.kernel_spectrum)).item()) / cell_area
    if not math.isfinite(lipschitz) or lipschitz <= 0:
        raise RuntimeError("Elasticity operator produced an invalid contact step bound")
    step = 1.0 / lipschitz
    scale = max(float(np.max(np.abs(geometry_m))), np.finfo(np.float64).tiny)
    active_threshold = load * 1e-12 / pressure_cells
    last_result = None

    for iteration in range(1, maximum_iterations + 1):
        displacement_at_accelerated = operator(
            (accelerated / cell_area).reshape(shape)
        )
        gradient = (displacement_at_accelerated + geometry_device).reshape(-1)
        updated = project_nonnegative_fixed_sum(accelerated - step * gradient, load, selected)
        next_momentum = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * momentum**2))
        accelerated = updated + ((momentum - 1.0) / next_momentum) * (updated - q)
        q, momentum = updated, next_momentum

        if iteration % 10 != 0 and iteration != maximum_iterations:
            continue
        pressure = (q / cell_area).reshape(shape)
        displacement = operator(pressure)
        stationarity = geometry_device + displacement
        approach = float((xp.sum(q * stationarity.reshape(-1)) / load).item())
        gap = stationarity - approach
        active = (q > active_threshold).reshape(shape)
        active_error = (
            float(xp.max(xp.abs(gap[active])).item()) if bool(xp.any(active).item()) else math.inf
        )
        inactive = ~active
        penetration = (
            float(xp.max(positive_part(-gap[inactive], selected)).item())
            if bool(xp.any(inactive).item())
            else 0.0
        )
        kkt_residual = max(active_error, penetration) / max(scale, abs(approach))
        loads = integrate_contact_pressure(pressure, x, y, selected)
        last_result = DryContactResult(
            pressure,
            displacement,
            gap,
            approach,
            loads,
            iteration,
            kkt_residual,
            float(xp.max(positive_part(-gap, selected)).item()),
        )
        if kkt_residual <= tolerance:
            return last_result

    raise DryContactConvergenceError(
        f"Dry contact did not reach KKT tolerance {tolerance:.3e} in "
        f"{maximum_iterations} iterations; residual="
        f"{last_result.relative_kkt_residual:.3e}",
        last_result,
    )


def solve_finite_roller_dry_contact(
    x_m: ArrayLike,
    y_m: ArrayLike,
    geometry: ContactGeometry,
    roller_length_m: float,
    load_n: float,
    e_star_pa: float,
    *,
    backend: Backend | None = None,
    tolerance: float = 2e-6,
    maximum_iterations: int = 20000,
    memory_limit_bytes: int = 2**30,
) -> DryContactResult:
    """Solve dry contact on the finite roller's full cell-centred axial span.

    The outer axial cell edges must coincide with ``+/-roller_length_m/2``.
    No pressure exists beyond the modelled roller end. Elasticity remains a
    two-dimensional half-space approximation rather than a finite-body model.
    """
    if (
        isinstance(roller_length_m, bool)
        or not math.isfinite(roller_length_m)
        or roller_length_m <= 0.0
    ):
        raise ValueError("roller_length_m must be positive finite metres")
    y = _uniform_axis(y_m, "y_m")
    dy = float(y[1] - y[0])
    edges = np.array([y[0] - dy / 2.0, y[-1] + dy / 2.0])
    expected_edges = np.array([-roller_length_m / 2.0, roller_length_m / 2.0])
    if not np.allclose(edges, expected_edges, rtol=1e-10, atol=0.0):
        raise ValueError("axial cell edges must coincide with the finite roller ends")
    result = solve_dry_contact(
        x_m,
        y,
        geometry,
        load_n,
        e_star_pa,
        backend=backend,
        tolerance=tolerance,
        maximum_iterations=maximum_iterations,
        memory_limit_bytes=memory_limit_bytes,
    )
    pressure = result.pressure_pa
    if bool((pressure[:, 0] > 0).any().item()) or bool((pressure[:, -1] > 0).any().item()):
        raise RuntimeError("contact reaches a rolling-direction computational boundary")
    return result
