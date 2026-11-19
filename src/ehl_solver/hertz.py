"""Analytical non-adhesive, frictionless Hertz point-contact references.

E_star: contact effective (plane-strain combined) modulus [Pa], defined by
1/E_star = (1-nu1**2)/E1 + (1-nu2**2)/E2.
Do NOT insert the alternative E_prime = 2 E_star here without conversion.
"""
from dataclasses import dataclass
from math import isfinite, pi, sqrt

from scipy.optimize import brentq
from scipy.special import ellipe, ellipk


@dataclass(frozen=True)
class HertzResult:
    radius_m: float
    peak_pressure_pa: float
    approach_m: float


@dataclass(frozen=True)
class EllipticalHertzResult:
    semi_axis_x_m: float
    semi_axis_y_m: float
    peak_pressure_pa: float
    approach_m: float
    axis_ratio_y_over_x: float
    elliptic_parameter: float


def circular_point_contact(load_n: float, radius_m: float, e_star_pa: float) -> HertzResult:
    if min(load_n, radius_m, e_star_pa) <= 0:
        raise ValueError("Load, effective radius and effective modulus must be positive")
    a = (3.0 * load_n * radius_m / (4.0 * e_star_pa)) ** (1.0 / 3.0)
    return HertzResult(a, 3.0 * load_n / (2.0 * pi * a * a), a * a / radius_m)


def elliptical_point_contact(
    load_n: float, radius_x_m: float, radius_y_m: float, e_star_pa: float
) -> EllipticalHertzResult:
    """Hertz elliptical-contact reference; radii and load are SI.

    The Hertz modulus convention is ``1/E* = sum((1-nu_i**2)/E_i)``.
    Ellipticity follows the complete-elliptic-integral relation described by
    Hokada et al., arXiv:2508.06967, Eqs. (5), (8), and (11). The leading
    near-circular series avoids cancellation when the curvature ratio is
    within ``1e-5`` of unity.
    """
    values = (load_n, radius_x_m, radius_y_m, e_star_pa)
    if not all(isfinite(value) and value > 0 for value in values):
        raise ValueError("Load, radii and effective modulus must be positive finite SI values")

    curvature_x, curvature_y = 1.0 / radius_x_m, 1.0 / radius_y_m
    major_curvature = max(curvature_x, curvature_y)
    minor_curvature = min(curvature_x, curvature_y)
    curvature_ratio = major_curvature / minor_curvature

    curvature_difference = curvature_ratio - 1.0
    if curvature_difference <= 1e-5:
        axis_ratio = 1.0 + (2.0 / 3.0) * curvature_difference
        parameter = 1.0 - 1.0 / axis_ratio**2
    else:
        def residual(ratio):
            parameter = 1.0 - 1.0 / ratio**2
            first = ellipk(parameter)
            second = ellipe(parameter)
            return (ratio**2 * second - first) / (first - second) - curvature_ratio

        upper = 2.0
        while residual(upper) < 0:
            upper *= 2.0
            if upper > 1e6:
                raise ValueError("Curvature ratio is outside the supported Hertz ellipticity range")
        axis_ratio = brentq(residual, 1.0 + 1e-8, upper)
        parameter = 1.0 - 1.0 / axis_ratio**2

    first, second = ellipk(parameter), ellipe(parameter)
    curvature_sum = curvature_x + curvature_y
    short_axis = (
        3.0 * second * load_n / (pi * axis_ratio * e_star_pa * curvature_sum)
    ) ** (1.0 / 3.0)
    long_axis = axis_ratio * short_axis
    if curvature_x >= curvature_y:
        semi_axis_x, semi_axis_y = short_axis, long_axis
    else:
        semi_axis_x, semi_axis_y = long_axis, short_axis
    approach = first / 2.0 * (
        9.0 * curvature_sum / second * (load_n / (pi * axis_ratio * e_star_pa)) ** 2
    ) ** (1.0 / 3.0)
    peak_pressure = 3.0 * load_n / (2.0 * pi * semi_axis_x * semi_axis_y)
    return EllipticalHertzResult(
        semi_axis_x,
        semi_axis_y,
        peak_pressure,
        approach,
        semi_axis_y / semi_axis_x,
        parameter,
    )


def pressure_radial(r_m: float, hertz: HertzResult) -> float:
    if r_m < 0:
        raise ValueError("r_m must be >= 0")
    return hertz.peak_pressure_pa * sqrt(max(0.0, 1.0 - (r_m / hertz.radius_m) ** 2))


def pressure_elliptical(x_m: float, y_m: float, hertz: EllipticalHertzResult) -> float:
    """Evaluate the analytical Hertz pressure inside an elliptical contact."""
    if not isfinite(x_m) or not isfinite(y_m):
        raise ValueError("Coordinates must be finite SI values")
    radius_squared = (x_m / hertz.semi_axis_x_m) ** 2 + (
        y_m / hertz.semi_axis_y_m
    ) ** 2
    return hertz.peak_pressure_pa * sqrt(max(0.0, 1.0 - radius_squared))
