"""Independent continuum Kirchhoff reference, constant film, zero entrainment.

F(p)=integral rho/eta dp is linear in x; no JFO/sparse solver is called.
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq


def pressure_driven_channel(x_m, film_m, fluid, inlet_pa, outlet_pa):
    x = np.asarray(x_m, float)
    if (
        x.ndim != 1
        or len(x) < 3
        or np.any(np.diff(x) <= 0)
        or not 0 <= outlet_pa < inlet_pa
        or film_m <= 0
    ):
        raise ValueError("invalid SI pressure-driven channel")

    def mobility(p):
        rho, eta, _, _ = fluid.properties(p)
        return float(rho / eta)

    def potential(p):
        return quad(mobility, outlet_pa, p, epsabs=1e-5, epsrel=2e-12)[0]

    total = potential(inlet_pa)
    pressures = np.array(
        [
            brentq(lambda p, target=target: potential(p) - target, outlet_pa, inlet_pa, xtol=1e-7)
            for target in (1 - (x - x[0]) / (x[-1] - x[0])) * total
        ]
    )
    return pressures, film_m**3 * total / (12 * (x[-1] - x[0]))
