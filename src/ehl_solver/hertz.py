"""Analytical non-adhesive, frictionless Hertz *circular* point contact.

E_star: contact effective (plane-strain combined) modulus [Pa], defined by
1/E_star = (1-nu1**2)/E1 + (1-nu2**2)/E2.
Do NOT insert the alternative E_prime = 2 E_star here without conversion.
"""
from dataclasses import dataclass
from math import pi, sqrt


@dataclass(frozen=True)
class HertzResult:
    radius_m: float
    peak_pressure_pa: float
    approach_m: float


def circular_point_contact(load_n: float, radius_m: float, e_star_pa: float) -> HertzResult:
    if min(load_n, radius_m, e_star_pa) <= 0:
        raise ValueError("Load, effective radius and effective modulus must be positive")
    a = (3.0 * load_n * radius_m / (4.0 * e_star_pa)) ** (1.0 / 3.0)
    return HertzResult(a, 3.0 * load_n / (2.0 * pi * a * a), a * a / radius_m)


def pressure_radial(r_m: float, hertz: HertzResult) -> float:
    if r_m < 0:
        raise ValueError("r_m must be >= 0")
    return hertz.peak_pressure_pa * sqrt(max(0.0, 1.0 - (r_m / hertz.radius_m) ** 2))
