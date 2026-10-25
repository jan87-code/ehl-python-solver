"""Analytical rigid, infinite-width parabolic JFO inlet/rupture reference."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq


def _i2(x):
    return 0.5 * (np.arctan(x) + x / (1 + x * x))


def _i3(x):
    return x / (4 * (1 + x * x) ** 2) + 0.75 * _i2(x)


@dataclass(frozen=True)
class ParabolicJFOReference:
    """SI 1-D fields; force is per transverse width [N/m], flux [m²/s]."""

    x_m: NDArray[np.float64]
    film_thickness_m: NDArray[np.float64]
    pressure_pa: NDArray[np.float64]
    fractional_film_content: NDArray[np.float64]
    pressure_gradient_pa_per_m: NDArray[np.float64]
    rupture_location_m: float
    volume_flux_per_width_m2_s: float
    force_per_width_n_m: float
    length_scale_m: float
    pressure_scale_pa: float


def parabolic_jfo_reference(
    x_m: ArrayLike,
    minimum_film_m: float,
    radius_m: float,
    viscosity_pa_s: float,
    entrainment_speed_m_s: float,
) -> ParabolicJFOReference:
    """Solve the rigid infinite-width, finite-inlet, smooth-rupture limit.

    x [m] must ascend from a negative flooded zero-pressure inlet through
    x=0 to an outlet downstream of rupture. h=h_min+x²/(2R), constant eta,
    Ue=(u1+u2)/2 >0. No elastic deformation, transverse pressure gradient,
    starvation, reformation or pressure-dependent properties are included.

    L=sqrt(2R*h_min), X=x/L, H=1+X² and p_ref=12*eta*Ue*L/h_min²
    give Lambda=1. Integrals I2'=H^-2 and I3'=H^-3 determine Xc from
    I2(Xc)-I2(Xin)=(1+Xc²)*(I3(Xc)-I3(Xin)). Upstream P=I2-I2_in
    -(1+Xc²)*(I3-I3_in); downstream P=0 and theta=(1+Xc²)/H.
    Thus p(xc)=p'(xc)=0 and q=Ue*h(xc) on both sides. The analytic force
    follows integration by parts, independently of the supplied sample grid.
    This is a benchmark solution, not a 2-D point-contact EHL model.
    """
    x = np.array(x_m, dtype=np.float64, copy=True)
    if (
        x.ndim != 1
        or x.size < 3
        or not np.all(np.isfinite(x))
        or np.any(np.diff(x) <= 0)
        or x[0] >= 0
        or x[-1] <= 0
    ):
        raise ValueError(
            "x_m must be finite ascending 1-D coordinates from negative inlet to positive outlet"
        )
    if not all(
        math.isfinite(v) and v > 0
        for v in (minimum_film_m, radius_m, viscosity_pa_s, entrainment_speed_m_s)
    ):
        raise ValueError(
            "film, radius, viscosity and mean entrainment speed must be positive SI values"
        )
    length = math.sqrt(2 * radius_m * minimum_film_m)
    denominator = minimum_film_m * minimum_film_m
    if not math.isfinite(length) or length <= 0 or denominator == 0:
        raise ValueError("derived reference scales are outside float64 range")
    pressure_scale = 12 * viscosity_pa_s * entrainment_speed_m_s * length / denominator
    if not math.isfinite(pressure_scale) or pressure_scale <= 0:
        raise ValueError("derived pressure scale is outside float64 range")
    X = x / length
    inlet = X[0]

    def condition(c):
        return _i2(c) - _i2(inlet) - (1 + c * c) * (_i3(c) - _i3(inlet))

    c = brentq(condition, 0.0, -inlet, xtol=1e-14, rtol=1e-14)
    if X[-1] <= c:
        raise ValueError("outlet must be downstream of the computed rupture location")
    hc = 1 + c * c
    h = minimum_film_m * (1 + X * X)
    full = X < c
    p = np.zeros(x.shape, dtype=np.float64)
    p[full] = pressure_scale * (_i2(X[full]) - _i2(inlet) - hc * (_i3(X[full]) - _i3(inlet)))
    theta = np.ones(x.shape, dtype=np.float64)
    theta[~full] = hc / (1 + X[~full] ** 2)
    gradient = np.zeros(x.shape, dtype=np.float64)
    gradient[full] = (pressure_scale / length) * ((1 + X[full] ** 2) - hc) / (1 + X[full] ** 2) ** 3
    hin = 1 + inlet * inlet
    force = pressure_scale * length * (hin - hc) ** 2 / (4 * hc * hin**2)
    return ParabolicJFOReference(
        x,
        h,
        p,
        theta,
        gradient,
        c * length,
        entrainment_speed_m_s * minimum_film_m * hc,
        force,
        length,
        pressure_scale,
    )
