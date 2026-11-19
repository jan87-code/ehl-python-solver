"""Hamrock–Dowson fully flooded isothermal viscous-elastic film correlations.

Reference: Hamrock & Dowson, J. Lubr. Technol. 99 (1977), 264–275,
DOI 10.1115/1.3453074; NASA TP-1342 (1978), equations 2–8 and 26.
Central fit: Hc=2.69 U^.67 G^.53 W^(-.067)(1-.61 exp(-.73k)).
These are correlation estimates, not Reynolds/load/conservation acceptance.
"""

from dataclasses import dataclass
from math import exp, expm1, isfinite


@dataclass(frozen=True)
class FilmCorrelation:
    minimum_film_m: float
    central_film_m: float
    elastic_modulus_prime_pa: float
    speed_parameter: float
    material_parameter: float
    load_parameter: float
    ellipticity: float


@dataclass(frozen=True)
class LineContactFilmCorrelation:
    minimum_film_m: float
    elastic_modulus_prime_pa: float
    speed_parameter: float
    material_parameter: float
    line_load_parameter: float


def dowson_higginson_line_minimum_film(
    *,
    line_load_n_m,
    radius_x_m,
    contact_modulus_pa,
    viscosity_pa_s,
    entrainment_m_s,
    alpha_per_pa,
):
    """Return the classical fully flooded isothermal line-contact h_min estimate.

    The approximate proximity equation is ``H_min=1.6*G**0.6*U**0.7*W**-0.13``,
    with ``U=eta0*u/(E_prime*R)``, ``G=alpha*E_prime``,
    ``W=w_line/(E_prime*R)`` and ``h_min=R*H_min``. It is an independent
    correlation, not an acceptance threshold or an exact solution of the
    Roelands/Dowson-Higginson fluid law used by the solver. The SI contact
    modulus follows the package Hertz convention and ``E_prime=2*E_star``.
    """
    values = (
        line_load_n_m,
        radius_x_m,
        contact_modulus_pa,
        viscosity_pa_s,
        entrainment_m_s,
        alpha_per_pa,
    )
    if not all(isfinite(value) and value > 0.0 for value in values):
        raise ValueError("line-correlation SI parameters must be positive finite")
    e_prime_pa = 2.0 * contact_modulus_pa
    speed_parameter = viscosity_pa_s * entrainment_m_s / (
        e_prime_pa * radius_x_m
    )
    material_parameter = alpha_per_pa * e_prime_pa
    line_load_parameter = line_load_n_m / (e_prime_pa * radius_x_m)
    minimum = (
        radius_x_m
        * 1.6
        * material_parameter**0.6
        * speed_parameter**0.7
        * line_load_parameter**-0.13
    )
    if not all(
        isfinite(value) and value > 0.0
        for value in (
            e_prime_pa,
            speed_parameter,
            material_parameter,
            line_load_parameter,
            minimum,
        )
    ):
        raise ValueError("line-correlation parameters overflowed or underflowed")
    return LineContactFilmCorrelation(
        minimum,
        e_prime_pa,
        speed_parameter,
        material_parameter,
        line_load_parameter,
    )


def hamrock_dowson_point_contact(
    *,
    load_n,
    radius_x_m,
    contact_modulus_pa,
    viscosity_pa_s,
    entrainment_m_s,
    alpha_per_pa,
    ellipticity=1.0,
):
    """Return SI film estimates with E'=2 E_contact and Ue=(u1+u2)/2.

    E_contact obeys 1/E_contact=sum((1-nu_i²)/E_i), matching Hertz/elasticity.
    U=eta0 Ue/(E' Rx), G=alpha E', W=Q/(E' Rx²); H=h/Rx.
    k is the actual transverse/entrainment Hertz semi-axis ratio, not Ry/Rx;
    k=1 for circular contact. Restrict to the original fitted k range [1,8]
    with entrainment along the minor axis. No inference of k from geometry,
    starvation correction, heating, roughness or shear thinning is included.
    Alpha is the declared pressure-viscosity coefficient used by the fit;
    differing fluid-law alpha definitions must be recorded by the caller.
    """
    values = (load_n, radius_x_m, contact_modulus_pa, viscosity_pa_s, entrainment_m_s, alpha_per_pa)
    if not all(isfinite(v) and v > 0 for v in values):
        raise ValueError("Hamrock-Dowson SI parameters must be positive finite")
    if not isfinite(ellipticity) or not 1 <= ellipticity <= 8:
        raise ValueError("Fitted ellipticity range is 1 <= k <= 8")
    ep = 2 * contact_modulus_pa
    u, g, w = (
        viscosity_pa_s * entrainment_m_s / (ep * radius_x_m),
        alpha_per_pa * ep,
        load_n / (ep * radius_x_m**2),
    )
    minimum = radius_x_m * 3.63 * u**0.68 * g**0.49 * w ** (-0.073) * (-expm1(-0.68 * ellipticity))
    central = (
        radius_x_m
        * 2.69
        * u**0.67
        * g**0.53
        * w ** (-0.067)
        * (1 - 0.61 * exp(-0.73 * ellipticity))
    )
    if not all(isfinite(v) and v > 0 for v in (ep, u, g, w, minimum, central)):
        raise ValueError("Correlation parameters overflowed or underflowed")
    return FilmCorrelation(minimum, central, ep, u, g, w, float(ellipticity))
