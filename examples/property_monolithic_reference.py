"""Small dense independent nonlinear algorithm for a prescribed-offset reference.

No production solver change: SciPy hybrid root solves complete pressure/content
and exact-film equations, warm only from the old independent isoviscous root.
"""

import numpy as np
from scipy.optimize import root

from ehl_solver.cavitating_newton import solve_cavitating_newton
from ehl_solver.property_reynolds import mass_fluxes


def small_monolithic_reference(problem, fluid, speed):
    ny, nx = problem.shape
    n = (ny - 2) * (nx - 2)
    if n > 200:
        raise ValueError("dense verification reference limited to 200 interior nodes")
    base = solve_cavitating_newton(problem).solution
    initial = np.r_[
        base.pressure_pa[1:-1, 1:-1].ravel() / problem.pressure_scale_pa,
        1 - base.fractional_film_content[1:-1, 1:-1].ravel(),
    ]

    def unpack(z):
        p = np.zeros(problem.shape)
        theta = np.ones(problem.shape)
        p[1:-1, 1:-1] = z[:n].reshape(ny - 2, nx - 2) * problem.pressure_scale_pa
        theta[1:-1, 1:-1] = 1 - z[n:].reshape(ny - 2, nx - 2)
        theta[1:-1, 0] = problem.inlet_fractional_film_content
        theta[1:-1, -1] = theta[1:-1, -2]
        theta[0], theta[-1] = theta[1], theta[-2]
        h, w = problem.fields(p)
        return p, theta, h, w

    def residual(z):
        p, t, h, _ = unpack(z)
        qx, qy = mass_fluxes(problem.x_m, problem.y_m, h, p, t, fluid, speed)
        div = np.diff(qx[1:-1], axis=1) / problem.dx_m + np.diff(qy[:, 1:-1], axis=0) / problem.dy_m
        qref = (
            fluid.density_kg_m3
            * problem.film_scale_m**3
            * problem.pressure_scale_pa
            / (12 * fluid.viscosity_pa_s * problem.length_scale_m)
        )
        return np.r_[div.ravel() * problem.length_scale_m / qref / 100, np.minimum(z[:n], z[n:])]

    r = root(residual, initial, method="hybr", options={"xtol": 1e-10, "maxfev": 4000})
    p, t, h, w = unpack(r.x)
    return {
        "success": bool(r.success),
        "message": str(r.message),
        "evaluations": int(r.nfev),
        "maximum_scaled_residual": float(abs(residual(r.x)).max()),
    }, (p, t, h, w)
