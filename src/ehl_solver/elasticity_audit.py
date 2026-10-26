"""Independent cell-integrated spatial elasticity at bounded target points."""

import numpy as np


def direct_midpoint_displacement(x_m, y_m, pressure_pa, e_star_pa, target_points_m):
    """Free-space w [m] for pressure [Pa], axes/targets [m], E* [Pa].

    Interior source cells end at neighboring-node midpoints. Boundary
    pressures must be exactly zero. At most 32 targets and 70000 source
    nodes; no FFT or uniform-cell approximation is used. For uniform grids
    this equals the existing rectangular cell-integrated kernel convention.
    On graded grids this is a diagnostic of a prescribed pressure field,
    not a coupled nonuniform elasticity solver.
    """
    x, y = np.asarray(x_m, float), np.asarray(y_m, float)
    p, targets = np.asarray(pressure_pa, float), np.asarray(target_points_m, float)
    if (
        any(
            a.ndim != 1 or len(a) < 3 or not np.all(np.isfinite(a)) or np.any(np.diff(a) <= 0)
            for a in (x, y)
        )
        or p.shape != (len(y), len(x))
        or p.size > 70000
        or not np.all(np.isfinite(p))
        or targets.ndim != 2
        or targets.shape[1] != 2
        or not 1 <= len(targets) <= 32
        or not np.all(np.isfinite(targets))
        or not np.isfinite(e_star_pa)
        or e_star_pa <= 0
    ):
        raise ValueError("invalid bounded SI direct-displacement inputs")
    if np.any(np.r_[p[0], p[-1], p[:, 0], p[:, -1]] != 0):
        raise ValueError("boundary pressure must be zero for midpoint source support")
    ex, ey = 0.5 * (x[1:] + x[:-1]), 0.5 * (y[1:] + y[:-1])

    def primitive(a, b):
        aa, bb = np.abs(a), np.abs(b)
        return (
            np.sign(a)
            * np.sign(b)
            * (
                aa * np.arcsinh(bb / np.where(aa > 0, aa, 1))
                + bb * np.arcsinh(aa / np.where(bb > 0, bb, 1))
            )
        )

    result = []
    for tx, ty in targets:
        lo_x, hi_x = (tx - ex[1:])[None, :], (tx - ex[:-1])[None, :]
        lo_y, hi_y = (ty - ey[1:])[:, None], (ty - ey[:-1])[:, None]
        integral = (
            primitive(hi_x, hi_y)
            - primitive(lo_x, hi_y)
            - primitive(hi_x, lo_y)
            + primitive(lo_x, lo_y)
        )
        result.append(np.sum(p[1:-1, 1:-1] * integral) / (np.pi * e_star_pa))
    return np.asarray(result)
