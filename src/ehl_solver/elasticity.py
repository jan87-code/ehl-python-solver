"""Cell-integrated elastic half-space deformation for normal pressure fields."""
import math

from ehl_solver.backend import Backend, as_float64, make_backend
from ehl_solver.convolution import fft_convolve_same


def _rectangle_primitive(x, y, backend: Backend):
    xp = backend.xp
    abs_x = xp.abs(x)
    abs_y = xp.abs(y)
    safe_x = xp.where(abs_x > 0, abs_x, xp.ones_like(abs_x))
    safe_y = xp.where(abs_y > 0, abs_y, xp.ones_like(abs_y))
    quadrant = abs_x * xp.arcsinh(abs_y / safe_x) + abs_y * xp.arcsinh(abs_x / safe_y)
    return xp.sign(x) * xp.sign(y) * quadrant


def cell_integrated_kernel(
    grid_shape: tuple[int, int],
    dx: float,
    dy: float,
    e_star_pa: float,
    backend: Backend | None = None,
):
    """Build the displacement-per-pressure kernel on an ``(ny, nx)`` grid.

    Grid values denote cell centers; x is the entrainment axis and y is
    transverse. The returned odd-sized kernel covers every source/target
    cell-center offset and is ordered ``(ny_lag, nx_lag)``. Its coefficient
    at each lag is the integral of ``1 / r`` over a rectangular source cell,
    divided by ``pi * e_star_pa``. Thus pressure in Pa convolves to
    displacement in m. ``dx`` and ``dy`` are cell widths in m, and
    ``e_star_pa`` is the combined effective modulus in Pa following the
    Hertz convention ``1/E* = sum((1 - nu_i**2) / E_i)``.
    """
    if backend is None:
        backend = make_backend()
    if len(grid_shape) != 2 or any(
        not isinstance(size, int) or isinstance(size, bool) or size < 1 for size in grid_shape
    ):
        raise ValueError("grid_shape must contain two positive integers (ny, nx)")
    if not math.isfinite(dx) or dx <= 0 or not math.isfinite(dy) or dy <= 0:
        raise ValueError("dx and dy must be finite and positive")
    if not math.isfinite(e_star_pa) or e_star_pa <= 0:
        raise ValueError("e_star_pa must be finite and positive")

    ny, nx = grid_shape
    xp = backend.xp
    if backend.name == "numpy":
        x_lags = xp.arange(-(nx - 1), nx, dtype=xp.float64) * dx
        y_lags = xp.arange(-(ny - 1), ny, dtype=xp.float64) * dy
    else:
        x_lags = xp.arange(-(nx - 1), nx, dtype=xp.float64, device=backend.device) * dx
        y_lags = xp.arange(-(ny - 1), ny, dtype=xp.float64, device=backend.device) * dy
    x_lo = (x_lags - dx / 2)[None, :]
    x_hi = (x_lags + dx / 2)[None, :]
    y_lo = (y_lags - dy / 2)[:, None]
    y_hi = (y_lags + dy / 2)[:, None]

    integral = (
        _rectangle_primitive(x_hi, y_hi, backend)
        - _rectangle_primitive(x_lo, y_hi, backend)
        - _rectangle_primitive(x_hi, y_lo, backend)
        + _rectangle_primitive(x_lo, y_lo, backend)
    )
    return integral / (math.pi * e_star_pa)


def elastic_deformation(
    pressure_pa,
    dx: float,
    dy: float,
    e_star_pa: float,
    backend: Backend | None = None,
):
    """Convolve cell-centered pressure [Pa] into displacement [m].

    ``dx`` and ``dy`` are cell widths [m]; ``e_star_pa`` is the combined
    effective modulus [Pa] using the Hertz convention.
    """
    if backend is None:
        backend = make_backend()
    pressure = as_float64(pressure_pa, backend)
    if pressure.ndim != 2:
        raise ValueError("pressure_pa must be a 2-D array with shape (ny, nx)")
    kernel = cell_integrated_kernel(pressure.shape, dx, dy, e_star_pa, backend)
    return fft_convolve_same(pressure, kernel, backend)
