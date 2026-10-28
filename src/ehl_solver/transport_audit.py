"""Independent SI directional mass-flux decomposition; no solver/stencil changes.

Fields are (ny,nx), entrainment +x, pressure Pa, film m, content dimensionless.
Returned interior divergence components have units kg/(m^2 s). This is a
static discrete response diagnostic, NOT a temporal stability criterion.
"""

import numpy as np


def mass_direction_components(x_m, y_m, film_m, pressure_pa, theta, fluid, speed_m_s,
                              *, pressure_direction_pa, film_direction_m,
                              content_direction):
    """Split the exact face-flux directional derivative into seven physical terms.

    Film direction may include nonlocal elasticity and offset. Density and
    viscosity contributions are held separate, without freezing the complete
    Jacobian. No direction or state is projected/clipped. Boundary directions
    are supplied explicitly by the caller, including content outflow copies.
    """
    x, y = np.asarray(x_m, dtype=np.float64), np.asarray(y_m, dtype=np.float64)
    h, p, t, dp, dh, dt = [np.asarray(v, dtype=np.float64) for v in (
        film_m, pressure_pa, theta, pressure_direction_pa, film_direction_m, content_direction
    )]
    if (x.ndim != 1 or y.ndim != 1 or min(x.size, y.size) < 3
            or any(v.shape != (y.size, x.size) for v in (h, p, t, dp, dh, dt))
            or any(not np.all(np.isfinite(v)) for v in (x, y, h, p, t, dp, dh, dt))
            or np.any(np.diff(x) <= 0) or np.any(np.diff(y) <= 0) or np.any(h <= 0)
            or not np.isfinite(speed_m_s) or speed_m_s < 0):
        raise ValueError("finite SI fields/directions, ascending axes and positive film required")
    rho, eta, rho_p, eta_p = fluid.properties(p)
    mobility = rho*h**3/(12*eta)
    capacity = rho*h
    changes = {
        "density_diffusion": rho_p*dp*h**3/(12*eta),
        "viscosity_diffusion": -mobility*eta_p*dp/eta,
        "film_diffusion": 3*mobility*dh/h,
    }
    dx, dy = np.diff(x), np.diff(y)[:, None]

    def diffusion(coefficient, pressure):
        return (-.5*(coefficient[:, 1:]+coefficient[:, :-1])*np.diff(pressure, axis=1)/dx,
                -.5*(coefficient[1:, :]+coefficient[:-1, :])*np.diff(pressure, axis=0)/dy)

    fluxes = {"pressure_diffusion": diffusion(mobility, dp)}
    fluxes.update({name: diffusion(value, p) for name, value in changes.items()})
    zero_y = np.zeros((y.size-1, x.size))
    for name, change, content in (
        ("density_advection", rho_p*dp*h, t),
        ("film_advection", rho*dh, t),
        ("content_advection", capacity, dt),
    ):
        fluxes[name] = (speed_m_s*.5*(change[:, 1:]+change[:, :-1])*content[:, :-1], zero_y)
    widths_x, widths_y = .5*(x[2:]-x[:-2]), .5*(y[2:]-y[:-2])[:, None]
    return {name: np.diff(qx[1:-1], axis=1)/widths_x
                  + np.diff(qy[:, 1:-1], axis=0)/widths_y
            for name, (qx, qy) in fluxes.items()}


def pressure_mode_directions(x_m, y_m, pressure_pa, contact_radius_m,
                             *, amplitude_pa=1e6, support_fraction=.1):
    """Zero-edge, zero-sum SI pressure probes in a pressure-supported inner region.

    Each direction has peak amplitude at most amplitude_pa and is corrected
    within its smooth window to give zero uniform-grid force derivative.
    Its pressure support avoids cavitated/tied complementarity branches;
    nonlocal elastic film changes still cover the entire physical domain.
    """
    x, y, p = np.asarray(x_m), np.asarray(y_m), np.asarray(pressure_pa)
    if (x.ndim != 1 or y.ndim != 1 or p.shape != (y.size, x.size)
            or min(p.shape) < 3 or not np.all(np.isfinite(p)) or p.max() <= 0
            or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y))
            or np.any(np.diff(x) <= 0) or np.any(np.diff(y) <= 0)
            or not np.allclose(np.diff(x), x[1]-x[0], rtol=1e-10, atol=0)
            or not np.allclose(np.diff(y), y[1]-y[0], rtol=1e-10, atol=0)
            or not np.isfinite(contact_radius_m) or contact_radius_m <= 0
            or not np.isfinite(amplitude_pa) or amplitude_pa <= 0
            or not 0 < support_fraction < 1):
        raise ValueError("positive finite SI probe radius/amplitude and loaded pressure required")
    xx, yy = np.meshgrid(x/contact_radius_m, y/contact_radius_m)
    # A diagnostic window, never a modification of a production pressure field.
    window = np.exp(-2*(xx**4 + yy**4)) * (p > support_fraction*p.max())
    window[[0, -1], :] = 0
    window[:, [0, -1]] = 0
    if np.count_nonzero(window) < 4 or window.sum() <= 0:
        raise ValueError("insufficient loaded interior support for pressure probes")
    j, i = np.indices(p.shape)
    patterns = {"alternating_x": (-1.)**i, "alternating_y": (-1.)**j,
                "alternating_xy": (-1.)**(i+j), "smooth_x": xx, "smooth_y": yy}
    output = {}
    for name, pattern in patterns.items():
        values = window*(pattern - np.sum(window*pattern)/window.sum())
        peak = abs(values).max()
        if peak == 0:
            raise ValueError("degenerate pressure probe")
        output[name] = amplitude_pa*values/peak
    return output


def nyquist_pressure_projection(x_m, y_m, pressure_pa, contact_radius_m):
    """Raw fixed-physical-window alternating projections, not an instability test.

    Also report nearest-axis-zero sampled centerlines to expose cancellation
    between opposite-phase rows/columns in the global projections.
    No smoothing, detrending or filtered production field is constructed.
    A sharp but physical profile can have nonzero projections; compare a
    sampled independent reference rather than declaring any value unphysical.
    """
    x, y, p = np.asarray(x_m), np.asarray(y_m), np.asarray(pressure_pa)
    if (x.ndim != 1 or y.ndim != 1 or min(x.size, y.size) < 1
            or p.shape != (y.size, x.size) or not np.all(np.isfinite(p))
            or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y))
            or not np.isfinite(contact_radius_m) or contact_radius_m <= 0):
        raise ValueError("finite pressure matching SI axes and positive radius required")
    xx, yy = np.meshgrid(x/contact_radius_m, y/contact_radius_m)
    weighted = np.exp(-2*(xx**4+yy**4))*p
    denominator = abs(weighted).sum()
    if denominator == 0:
        return {name: 0. for name in ("x", "y", "xy", "centerline_x", "centerline_y")}
    j, i = np.indices(p.shape)
    result = {name: float(abs(np.sum(weighted*sign))/denominator)
              for name, sign in (("x", (-1.)**i), ("y", (-1.)**j), ("xy", (-1.)**(i+j)))}
    for name, values, sign in (
        ("centerline_x", weighted[np.argmin(abs(y)), :], (-1.)**np.arange(x.size)),
        ("centerline_y", weighted[:, np.argmin(abs(x))], (-1.)**np.arange(y.size)),
    ):
        divisor = abs(values).sum()
        result[name] = float(abs(np.sum(values*sign))/divisor) if divisor else 0.
    return result
