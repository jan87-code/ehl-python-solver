"""Independent finite-difference hybrid root for local signed-offset verification.

Small grids only; an audited warm root supplies a perturbed initial guess.
This compares algorithms locally, not experimental data or branch uniqueness.
"""

import numpy as np
from scipy.optimize import root

from ehl_solver.property_reynolds import mass_fluxes


def signed_reference(problem, fluid, speed, offset, initial):
    ny, nx = problem.shape
    n = (ny - 2) * (nx - 2)
    if n > 400:
        raise ValueError("independent dense reference limited to 400 interior nodes")
    z = np.r_[
        initial.pressure_pa[1:-1, 1:-1].ravel() / problem.pressure_scale_pa,
        1 - initial.fractional_film_content[1:-1, 1:-1].ravel(),
    ]
    z[:n] *= 1.0001
    z[n:] *= 0.9999

    def unpack(z):
        p, t = np.zeros(problem.shape), np.ones(problem.shape)
        p[1:-1, 1:-1] = z[:n].reshape(ny - 2, nx - 2) * problem.pressure_scale_pa
        t[1:-1, 1:-1] = 1 - z[n:].reshape(ny - 2, nx - 2)
        t[1:-1, 0] = problem.inlet_fractional_film_content
        t[1:-1, -1] = t[1:-1, -2]
        t[0], t[-1] = t[1], t[-2]
        h, w = problem.fields(p, central_film_offset_m=offset)
        return p, t, h, w

    def residual(z):
        p, t, h, _ = unpack(z)
        qx, qy = mass_fluxes(problem.x_m, problem.y_m, h, p, t, fluid, speed)
        div = np.diff(qx[1:-1], axis=1) / problem.dx_m + np.diff(qy[:, 1:-1], axis=0) / problem.dy_m
        ref = (
            fluid.density_kg_m3
            * problem.film_scale_m**3
            * problem.pressure_scale_pa
            / (12 * fluid.viscosity_pa_s * problem.length_scale_m)
        )
        return np.r_[div.ravel() * problem.length_scale_m / ref / 100, np.minimum(z[:n], z[n:])]

    result = root(residual, z, method="hybr", options={"xtol": 1e-10, "maxfev": 4000})
    return {
        "success": bool(result.success),
        "message": str(result.message),
        "evaluations": int(result.nfev),
        "maximum_scaled_residual": float(abs(residual(result.x)).max()),
    }, unpack(result.x)
