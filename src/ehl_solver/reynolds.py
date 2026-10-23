"""Dimensionless residual for steady isoviscous-incompressible Reynolds flow."""

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray


def reynolds_residual(
    pressure: ArrayLike,
    film: ArrayLike,
    dx: float,
    dy: float,
    reynolds_parameter: float,
    source: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Return the conservative Reynolds-equation residual on interior cells.

    ``pressure`` and ``film`` are dimensionless arrays with shape ``(ny, nx)``;
    x is the entrainment direction and y is transverse. ``dx`` and ``dy`` are
    dimensionless grid spacings. All input cells, including boundary cells,
    contribute to the stencil, but residuals are returned only for the
    ``(ny - 2, nx - 2)`` interior cells. Boundary conditions are not imposed.

    The source-free equation is

        div(H**3 grad(P)) - Lambda * dH/dX = 0.

    ``reynolds_parameter`` is Lambda = 12 * eta0 * Ue * L /
    (h_ref**2 * p_ref), where Ue = (u1 + u2) / 2 is the mean surface
    velocity in the x direction. ``source`` optionally supplies the
    dimensionless manufactured source S, so the residual is
    div(H**3 grad(P)) - Lambda * dH/dX - S.

    Face values of H**3 use arithmetic interpolation, pressure gradients use
    adjacent cell centers, and the divergence is formed from opposing face
    fluxes. The returned residual is dimensionless.
    """
    p = np.asarray(pressure, dtype=np.float64)
    h = np.asarray(film, dtype=np.float64)
    if p.ndim != 2 or h.ndim != 2:
        raise ValueError("pressure and film must be 2-D arrays with shape (ny, nx)")
    if p.shape != h.shape:
        raise ValueError("pressure and film must have the same shape")
    if min(p.shape) < 3:
        raise ValueError("pressure and film must each have at least 3 cells per axis")
    if not np.all(np.isfinite(p)):
        raise ValueError("pressure must contain only finite values")
    if not np.all(np.isfinite(h)) or np.any(h <= 0):
        raise ValueError("film must contain only finite, positive values")
    if not math.isfinite(dx) or dx <= 0 or not math.isfinite(dy) or dy <= 0:
        raise ValueError("dx and dy must be finite and positive")
    if not math.isfinite(reynolds_parameter) or reynolds_parameter < 0:
        raise ValueError("reynolds_parameter must be finite and nonnegative")

    interior = (slice(1, -1), slice(1, -1))
    film_cubed = h**3
    pressure_center = p[interior]
    film_center = film_cubed[interior]

    flux_east = (
        0.5 * (film_center + film_cubed[1:-1, 2:])
        * (p[1:-1, 2:] - pressure_center)
        / dx
    )
    flux_west = (
        0.5 * (film_center + film_cubed[1:-1, :-2])
        * (pressure_center - p[1:-1, :-2])
        / dx
    )
    flux_north = (
        0.5 * (film_center + film_cubed[2:, 1:-1])
        * (p[2:, 1:-1] - pressure_center)
        / dy
    )
    flux_south = (
        0.5 * (film_center + film_cubed[:-2, 1:-1])
        * (pressure_center - p[:-2, 1:-1])
        / dy
    )
    residual = (flux_east - flux_west) / dx + (flux_north - flux_south) / dy
    residual -= reynolds_parameter * (h[1:-1, 2:] - h[1:-1, :-2]) / (2 * dx)

    if source is not None:
        manufactured_source = np.asarray(source, dtype=np.float64)
        if manufactured_source.shape != p.shape:
            raise ValueError("source must have the same shape as pressure and film")
        if not np.all(np.isfinite(manufactured_source)):
            raise ValueError("source must contain only finite values")
        residual -= manufactured_source[interior]

    return residual
