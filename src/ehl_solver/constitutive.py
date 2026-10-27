"""Isothermal SI Roelands viscosity and Dowson-Higginson density, float64.

Primary formulation: NASA RP-1255 (1991), equations 4-10 and 4-19.
Pressure is gauge relative to cavitation pc=0. These are empirical fluid
models, not experimental validation of a particular lubricant. No clipping.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IsothermalFluid:
    viscosity_pa_s: float
    density_kg_m3: float
    viscosity_index: float = 0.68
    viscosity_reference_pa_s: float = 6.31e-5
    roelands_pressure_pa: float = 1.96e8
    density_a_per_pa: float = 0.6e-9
    density_b_per_pa: float = 1.7e-9

    def __post_init__(self):
        positive = (
            self.viscosity_pa_s,
            self.density_kg_m3,
            self.viscosity_reference_pa_s,
            self.roelands_pressure_pa,
            self.density_b_per_pa,
        )
        if not all(np.isfinite(v) and v > 0 for v in positive):
            raise ValueError("fluid SI reference properties must be positive finite")
        if (
            not np.isfinite(self.viscosity_index)
            or not 0 <= self.viscosity_index <= 1
            or not np.isfinite(self.density_a_per_pa)
            or self.density_a_per_pa < 0
            or self.viscosity_pa_s <= self.viscosity_reference_pa_s
        ):
            raise ValueError("invalid fluid index/density coefficient or viscosity reference")

    def ratios(self, pressure_pa, *, strength=1.0):
        """Return rho/rho0, eta/eta0 and their derivatives [1/Pa].

        Strength in [0,1] multiplies density increment and log viscosity
        increment. Zero is the exact constant-property continuation limit.
        Small negative roundoff pressure is allowed only above -0.5 gamma
        and where density stays positive, to evaluate feasible Newton trials;
        no signed tensile-fluid solution is thereby physical. Overflow raises.
        """
        p = np.asarray(pressure_pa, dtype=np.float64)
        if (
            not np.all(np.isfinite(p))
            or np.any(p <= -0.5 * self.roelands_pressure_pa)
            or not np.isfinite(strength)
            or not 0 <= strength <= 1
        ):
            raise ValueError("invalid pressure domain or constitutive strength")
        a, b = self.density_a_per_pa, self.density_b_per_pa
        denominator = 1 + b * p
        if np.any(denominator <= 0):
            raise ValueError("density pole reached")
        rho = 1 + strength * a * p / denominator
        drho = strength * a / denominator**2
        logarithm = np.log(self.viscosity_pa_s / self.viscosity_reference_pa_s)
        t = np.log1p(p / self.roelands_pressure_pa)
        log_eta = strength * logarithm * np.expm1(self.viscosity_index * t)
        dlog_eta = (
            strength
            * logarithm
            * self.viscosity_index
            / self.roelands_pressure_pa
            * np.exp((self.viscosity_index - 1) * t)
        )
        with np.errstate(over="raise", invalid="raise"):
            eta = np.exp(log_eta)
            deta = eta * dlog_eta
        if np.any(rho <= 0) or not all(np.all(np.isfinite(v)) for v in (rho, drho, eta, deta)):
            raise ValueError("nonpositive or nonfinite constitutive state")
        return rho, eta, drho, deta

    def properties(self, pressure_pa, *, strength=1.0):
        """Return rho [kg/m^3], eta [Pa s], d rho/dp and d eta/dp in SI."""
        r, v, dr, dv = self.ratios(pressure_pa, strength=strength)
        return (
            self.density_kg_m3 * r,
            self.viscosity_pa_s * v,
            self.density_kg_m3 * dr,
            self.viscosity_pa_s * dv,
        )
