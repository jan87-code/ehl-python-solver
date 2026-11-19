"""Force and moment integration hooks for cell-centered normal-pressure fields."""

from __future__ import annotations

import math
from dataclasses import dataclass

from numpy.typing import ArrayLike

from ehl_solver.backend import Backend, as_float64, make_backend


@dataclass(frozen=True)
class ContactLoadMoments:
    """Integrated normal load and right-handed moments about the contact origin."""

    force_n: float
    moment_x_n_m: float
    moment_y_n_m: float


def integrate_contact_pressure(
    pressure_pa: ArrayLike,
    x_m: ArrayLike,
    y_m: ArrayLike,
    backend: Backend | None = None,
) -> ContactLoadMoments:
    """Integrate cell-centered pressure over a uniform ``(ny,nx)`` grid.

    Positive pressure acts in the negative z direction. Thus
    ``Mx=-integral(y*p dA)`` and ``My=integral(x*p dA)``. Coordinates are cell
    centers and all pressure cells have area ``dx*dy``.
    """
    selected = make_backend() if backend is None else backend
    xp = selected.xp
    pressure = as_float64(pressure_pa, selected)
    x = as_float64(x_m, selected)
    y = as_float64(y_m, selected)
    if pressure.ndim != 2 or x.ndim != 1 or y.ndim != 1:
        raise ValueError("pressure_pa must be (ny,nx), with 1-D x_m and y_m axes")
    nx = int(x.numel()) if selected.name == "torch" else int(x.size)
    ny = int(y.numel()) if selected.name == "torch" else int(y.size)
    if tuple(pressure.shape) != (ny, nx):
        raise ValueError("pressure_pa shape must match coordinate axes in (ny,nx) order")
    if nx < 2 or ny < 2:
        raise ValueError("At least two pressure-cell centers are required on each axis")
    dx_values, dy_values = xp.diff(x), xp.diff(y)
    dx, dy = float(dx_values[0].item()), float(dy_values[0].item())
    dx_reference = xp.full_like(dx_values, dx) if selected.name == "torch" else dx
    dy_reference = xp.full_like(dy_values, dy) if selected.name == "torch" else dy
    x_uniform = xp.allclose(dx_values, dx_reference, rtol=1e-10, atol=0)
    y_uniform = xp.allclose(dy_values, dy_reference, rtol=1e-10, atol=0)
    if (
        not math.isfinite(dx)
        or not math.isfinite(dy)
        or dx <= 0
        or dy <= 0
        or not bool(x_uniform.item() if hasattr(x_uniform, "item") else x_uniform)
        or not bool(y_uniform.item() if hasattr(y_uniform, "item") else y_uniform)
    ):
        raise ValueError("Contact load integration requires uniformly spaced ascending axes")
    if not bool(xp.all(xp.isfinite(pressure)).item()) or bool(xp.any(pressure < 0).item()):
        raise ValueError("pressure_pa must be finite and nonnegative")

    cell_area = dx * dy
    force = xp.sum(pressure) * cell_area
    moment_x = -xp.sum(pressure * y[:, None]) * cell_area
    moment_y = xp.sum(pressure * x[None, :]) * cell_area
    return ContactLoadMoments(
        float(force.item()), float(moment_x.item()), float(moment_y.item())
    )
