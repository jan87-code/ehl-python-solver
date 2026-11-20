"""Pressure-to-displacement compliance operators for load-controlled dry contact."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from ehl_solver.backend import Backend, as_float64, make_backend
from ehl_solver.elasticity import ElasticityOperator, cell_integrated_kernel


class ContactCompliance(Protocol):
    """Linear, self-adjoint positive compliance on a cell-centred grid.

    The bound is an upper bound on the pressure-to-displacement L2 operator
    norm in metres per pascal. A custom implementation must uphold this
    contract; the solver does not certify external FE-derived operators.
    """

    shape: tuple[int, int]
    backend: Backend
    dx_m: float
    dy_m: float
    @property
    def pressure_norm_bound_m_per_pa(self) -> float:
        """Bound the pressure-to-displacement norm [m/Pa]."""

    def __call__(self, pressure_pa: Any) -> Any:
        """Return displacement in metres on the same backend and grid."""


class HalfSpaceFFTCompliance(ElasticityOperator, ContactCompliance):
    """Existing cell-integrated, zero-padded half-space FFT compliance."""

    def __init__(
        self, grid_shape, dx, dy, e_star_pa, backend=None, *, memory_limit_bytes=2**30
    ):
        super().__init__(
            grid_shape, dx, dy, e_star_pa, backend, memory_limit_bytes=memory_limit_bytes
        )
        self.dx_m = float(dx)
        self.dy_m = float(dy)

    @property
    def pressure_norm_bound_m_per_pa(self) -> float:
        return float(
            self.backend.xp.max(self.backend.xp.abs(self.kernel_spectrum)).item()
        )


class DirectHalfSpaceCompliance(ContactCompliance):
    """Dense direct half-space influence coefficients, for small reference grids."""

    def __init__(
        self,
        grid_shape: tuple[int, int],
        dx: float,
        dy: float,
        e_star_pa: float,
        backend: Backend | None = None,
        *,
        memory_limit_bytes: int = 2**26,
    ):
        if type(memory_limit_bytes) is not int or memory_limit_bytes <= 0:
            raise ValueError("memory_limit_bytes must be a positive integer")
        cell_integrated_kernel((1, 1), dx, dy, e_star_pa)
        if len(grid_shape) != 2 or any(
            type(size) is not int or size < 1 for size in grid_shape
        ):
            raise ValueError("grid_shape must contain two positive integers (ny, nx)")
        self.shape = grid_shape
        self.backend = make_backend() if backend is None else backend
        self.dx_m = float(dx)
        self.dy_m = float(dy)
        ny, nx = grid_shape
        count = ny * nx
        if count > 1024 or 32 * count**2 > memory_limit_bytes:
            raise MemoryError("direct contact influence matrix exceeds its reference budget")
        kernel = cell_integrated_kernel(grid_shape, dx, dy, e_star_pa)
        rows, cols = np.indices(grid_shape)
        rows, cols = rows.reshape(-1), cols.reshape(-1)
        matrix = kernel[
            rows[:, None] - rows[None, :] + ny - 1,
            cols[:, None] - cols[None, :] + nx - 1,
        ]
        absolute = np.abs(matrix)
        self._pressure_norm_bound_m_per_pa = float(
            np.sqrt(np.max(np.sum(absolute, axis=0)) * np.max(np.sum(absolute, axis=1)))
        )
        self._matrix = as_float64(matrix, self.backend)

    @property
    def pressure_norm_bound_m_per_pa(self) -> float:
        return self._pressure_norm_bound_m_per_pa

    def __call__(self, pressure_pa):
        pressure = as_float64(pressure_pa, self.backend)
        if tuple(pressure.shape) != self.shape:
            raise ValueError("pressure_pa must match compliance shape (ny,nx)")
        return (self._matrix @ pressure.reshape(-1)).reshape(self.shape)
