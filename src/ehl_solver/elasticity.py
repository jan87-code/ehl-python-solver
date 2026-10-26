"""Cell-integrated elastic half-space deformation for normal pressure fields."""
import math

from scipy.fft import next_fast_len

from ehl_solver.backend import Backend, as_float64, make_backend
from ehl_solver.convolution import fft_convolve_same


class ElasticityOperator:
    """Reusable float64 linear DC-FFT operator: pressure [Pa] -> gap increase [m].

    The analytically integrated NumPy kernel is constructed once and uploaded
    once for torch. No global device cache is retained: releasing this object
    releases its spectrum. Inputs already on the selected device stay there.
    Padding is at least (3*ny-2, 3*nx-2), rounded up to FFT-friendly lengths;
    backward FFT normalization and extraction at (ny-1, nx-1) match the reference.

    CUDA allocations are guarded by a conservative workspace estimate, a 1 GiB
    per-operator budget and half of currently free memory. This is a guard, not
    a guarantee about vendor FFT workspace or concurrent device allocations.
    CUDA errors propagate; there is no fallback to CPU.
    """

    def __init__(
        self, grid_shape, dx, dy, e_star_pa, backend=None, *, memory_limit_bytes=2**30
    ):
        self.backend = make_backend() if backend is None else backend
        # Validate physical parameters and shape before allocating a kernel.
        cell_integrated_kernel((1, 1), dx, dy, e_star_pa)
        if len(grid_shape) != 2 or any(
            not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in grid_shape
        ):
            raise ValueError("grid_shape must contain two positive integers (ny, nx)")
        if not isinstance(memory_limit_bytes, int) or memory_limit_bytes <= 0:
            raise ValueError("memory_limit_bytes must be a positive integer")
        self.shape = tuple(grid_shape)
        self.fft_shape = tuple(next_fast_len(3 * n - 2) for n in self.shape)
        # Real buffers, complex spectra, input/output and kernel construction.
        self.estimated_workspace_bytes = 128 * math.prod(self.fft_shape)
        budget = memory_limit_bytes
        if self.backend.name == "torch" and self.backend.device == "cuda":
            free, _ = self.backend.xp.cuda.mem_get_info()
            budget = min(budget, free // 2)
        if self.estimated_workspace_bytes > budget:
            raise MemoryError("DC-FFT workspace estimate exceeds memory budget")
        kernel = cell_integrated_kernel(self.shape, dx, dy, e_star_pa)
        kernel = as_float64(kernel, self.backend)
        self.kernel_spectrum = self.backend.xp.fft.rfft2(kernel, s=self.fft_shape)

    def __call__(self, pressure_pa):
        pressure = as_float64(pressure_pa, self.backend)
        if tuple(pressure.shape) != self.shape:
            raise ValueError("pressure_pa must match operator shape (ny, nx)")
        xp = self.backend.xp
        spectrum = xp.fft.rfft2(pressure, s=self.fft_shape)
        full = xp.fft.irfft2(spectrum * self.kernel_spectrum, s=self.fft_shape)
        ny, nx = self.shape
        return full[ny - 1:2 * ny - 1, nx - 1:2 * nx - 1]


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
