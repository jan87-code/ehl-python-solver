"""Experimental conservative donor-capacity transport; NOT a production stencil.

SI fields (ny,nx), +x entrainment only. Pressure diffusion retains arithmetic
face mobility. Advection uses U*(rho*h*theta)_left, instead of the reference
U*average(rho*h)*theta_left. First-order artificial diffusion is explicit;
no cavitation, boundary condition, elastic coupling or Newton solve is implied.
"""

import numpy as np


def _fields(x, y, h, p, theta, speed):
    x, y, h, p, theta = [np.asarray(a, dtype=np.float64) for a in (x, y, h, p, theta)]
    if (x.ndim != 1 or y.ndim != 1 or min(x.size, y.size) < 3
            or any(a.shape != (y.size, x.size) for a in (h, p, theta))
            or any(not np.all(np.isfinite(a)) for a in (x, y, h, p, theta))
            or np.any(np.diff(x) <= 0) or np.any(np.diff(y) <= 0)
            or np.any(h <= 0) or not np.isfinite(speed) or speed < 0):
        raise ValueError("finite matching SI fields, ascending axes, h>0 and +x speed required")
    return x, y, h, p, theta


def donor_fluxes(x, y, h, p, theta, fluid, speed, *, strength=1.):
    """Return x/y face mass fluxes [kg/(m s)] without clipping any field."""
    x, y, h, p, theta = _fields(x, y, h, p, theta, speed)
    rho, eta, _, _ = fluid.properties(p, strength=strength)
    mobility = rho*h**3/(12*eta)
    qx = -.5*(mobility[:, 1:]+mobility[:, :-1])*np.diff(p, axis=1)/np.diff(x)
    qx += speed*(rho*h*theta)[:, :-1]
    qy = -.5*(mobility[1:]+mobility[:-1])*np.diff(p, axis=0)/np.diff(y)[:, None]
    return qx, qy


def donor_flux_direction(x, y, h, p, theta, dh, dp, dt, fluid, speed, *, strength=1.):
    """Analytic smooth flux JVP per dimensionless direction amplitude.

    Film direction can be supplied by elasticity, but no elastic model or
    complementarity derivative is selected here. Directions include boundaries.
    """
    x, y, h, p, theta = _fields(x, y, h, p, theta, speed)
    dh, dp, dt = [np.asarray(a, dtype=np.float64) for a in (dh, dp, dt)]
    if any(a.shape != h.shape or not np.all(np.isfinite(a)) for a in (dh, dp, dt)):
        raise ValueError("finite same-shaped directions required")
    rho, eta, dr, de = fluid.properties(p, strength=strength)
    m = rho*h**3/(12*eta)
    dm = m*((dr/rho-de/eta)*dp+3*dh/h)
    dc = dr*dp*h+rho*dh
    qx = -.5*(dm[:, 1:]+dm[:, :-1])*np.diff(p, axis=1)/np.diff(x)
    qx -= .5*(m[:, 1:]+m[:, :-1])*np.diff(dp, axis=1)/np.diff(x)
    qx += speed*(dc*theta+rho*h*dt)[:, :-1]
    qy = -.5*(dm[1:]+dm[:-1])*np.diff(p, axis=0)/np.diff(y)[:, None]
    qy -= .5*(m[1:]+m[:-1])*np.diff(dp, axis=0)/np.diff(y)[:, None]
    return qx, qy


def face_divergence(x, y, qx, qy):
    """Interior midpoint-volume mass divergence [kg/(m^2 s)].

    Boundaries are supplied face fluxes, not changed boundary conditions.
    The weighted sum telescopes to the outward boundary mass flow [kg/s].
    """
    x, y, qx, qy = [np.asarray(a, dtype=np.float64) for a in (x, y, qx, qy)]
    if (x.ndim != 1 or y.ndim != 1 or min(x.size, y.size) < 3
            or qx.shape != (y.size, x.size-1) or qy.shape != (y.size-1, x.size)
            or any(not np.all(np.isfinite(a)) for a in (x, y, qx, qy))
            or np.any(np.diff(x) <= 0) or np.any(np.diff(y) <= 0)):
        raise ValueError("finite face fluxes matching ascending axes required")
    return (np.diff(qx[1:-1], axis=1)/(.5*(x[2:]-x[:-2]))
            + np.diff(qy[:, 1:-1], axis=0)/(.5*(y[2:]-y[:-2]))[:, None])
