"""User-supplied circular 200 N case; explicit 113 GPa modulus convention.

Hamrock-Dowson film values are supplied correlation estimates, not exact
solutions. Roelands/Dowson-Higginson laws and rho0 are declared assumptions.
"""

from dataclasses import asdict, dataclass

import numpy as np

from ehl_solver.cavitating_coupled import CavitatingElasticityProblem
from ehl_solver.constitutive import IsothermalFluid


@dataclass(frozen=True)
class Circular200N:
    load_n: float = 200.0
    radius_m: float = 0.015
    contact_modulus_pa: float = 113e9
    entrainment_m_s: float = 2.0
    viscosity_pa_s: float = 0.04
    alpha_per_pa: float = 2.21e-8
    supplied_minimum_film_m: float = 0.278495e-6
    supplied_central_film_m: float = 0.477353e-6

    @property
    def hertz_radius_m(self):
        return (3 * self.load_n * self.radius_m / (4 * self.contact_modulus_pa)) ** (1 / 3)

    @property
    def hertz_pressure_pa(self):
        return 3 * self.load_n / (2 * np.pi * self.hertz_radius_m**2)

    @property
    def hertz_approach_m(self):
        return self.hertz_radius_m**2 / self.radius_m

    def fluid(self):
        # Match d(log eta)/dp at p=0, rather than identifying alpha with z.
        gamma, eta_star = 1.96e8, 6.31e-5
        z = self.alpha_per_pa * gamma / np.log(self.viscosity_pa_s / eta_star)
        return IsothermalFluid(self.viscosity_pa_s, 850.0, viscosity_index=z)

    def parameters(self):
        u = self.viscosity_pa_s * self.entrainment_m_s / (self.contact_modulus_pa * self.radius_m)
        w = self.load_n / (self.contact_modulus_pa * self.radius_m**2)
        g = self.alpha_per_pa * self.contact_modulus_pa
        return {
            **asdict(self),
            "modulus_definition": "1/E_contact=sum((1-nu_i**2)/E_i); E_prime=2*E_contact",
            "dimensionless_contact_convention": {"U": u, "W": w, "G": g},
            "dimensionless_E_prime_convention": {"U": u / 2, "W": w / 2, "G": 2 * g},
            "hertz_radius_m": self.hertz_radius_m,
            "hertz_pressure_pa": self.hertz_pressure_pa,
            "hertz_approach_m": self.hertz_approach_m,
            "fluid": asdict(self.fluid()),
            "assumptions": [
                "Roelands law matched to ambient logarithmic slope alpha; not Barus",
                "rho0=850 kg/m^3; Dowson-Higginson a=0.6e-9/Pa,b=1.7e-9/Pa",
                "isothermal; temperature unspecified; no thermal law used",
                (
                    "finite ambient-pressure rectangle; "
                    "flooded west is not proven far-field convergence"
                ),
                (
                    "supplied film values are correlation estimates; "
                    "source/formula not independently matched"
                ),
            ],
        }

    def problem(self, offset_m, *, ny=17, nx=25, extent=3.0):
        if (
            any(
                not isinstance(n, int) or isinstance(n, bool) or n < 5 or n % 2 == 0
                for n in (ny, nx)
            )
            or ny * nx > 20000
            or not np.isfinite(extent)
            or extent < 2
        ):
            raise ValueError("odd centered grids >=5, <=20000 nodes and extent >=2 required")
        a, h, p = self.hertz_radius_m, self.hertz_approach_m, self.hertz_pressure_pa
        lam = 12 * self.viscosity_pa_s * self.entrainment_m_s * a / (h**2 * p)
        return CavitatingElasticityProblem(
            np.linspace(-extent, extent, nx) * a,
            np.linspace(-extent, extent, ny) * a,
            offset_m,
            self.radius_m,
            self.radius_m,
            self.contact_modulus_pa,
            a,
            h,
            p,
            lam,
        )
