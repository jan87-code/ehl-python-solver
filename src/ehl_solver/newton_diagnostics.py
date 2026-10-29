"""Read-only SI affine-feasibility diagnostics, not solver acceptance rules."""

import numpy as np


def affine_feasibility_limits(pressure_pa, content, film_m, pressure_direction_pa,
                              content_direction, film_direction_m, *, pressure_scale_pa,
                              film_floor_m, feasibility_tolerance=1e-10):
    """Report separate pressure/content/film bounds on z+alpha*d, without clipping.

    All fields have shape (ny,nx), +x entrainment. Pressure and its direction
    are Pa, film/floor/direction are m, content and alpha are dimensionless.
    Pressure/content constraints apply to interior unknowns; film applies to
    the whole physical grid. The existing feasibility tolerance gives
    p>=-tol*p_ref, -tol<=theta<=1+tol, and strict h>film_floor_m. Reported raw
    bounds are *uncapped*: equality at the film limit is infeasible. No safety
    factor, step choice, line search or physical acceptance is applied here.
    None means that no finite float64 bound is available for that category.
    Up to eight exact-minimum cells and the total tie count are reported;
    near ties are not rounded into equality. Cell coordinates are (iy,ix).
    """
    fields = [np.asarray(v, dtype=np.float64) for v in (
        pressure_pa, content, film_m, pressure_direction_pa, content_direction, film_direction_m)]
    p, theta, h, dp, dt, dh = fields
    if (p.ndim != 2 or min(p.shape) < 3 or any(v.shape != p.shape for v in fields)
            or any(not np.all(np.isfinite(v)) for v in fields)
            or not np.isfinite(pressure_scale_pa) or pressure_scale_pa <= 0
            or not np.isfinite(film_floor_m) or film_floor_m < 0
            or not np.isfinite(feasibility_tolerance) or feasibility_tolerance < 0):
        raise ValueError("finite SI fields, positive pressure scale and nonnegative floor/tol "
                         "required")
    tol = feasibility_tolerance
    if (np.any(p[1:-1, 1:-1] < -tol*pressure_scale_pa)
            or np.any(theta[1:-1, 1:-1] < -tol)
            or np.any(theta[1:-1, 1:-1] > 1+tol) or np.any(h <= film_floor_m)):
        raise ValueError("initial state violates the original feasibility bounds")

    def limit(margin, closing_speed, offset, strict):
        crossing = closing_speed > 0
        ratios = np.full(margin.shape, np.inf)
        with np.errstate(over="ignore", divide="ignore", invalid="raise"):
            # Overflow denotes an unbounded float64 ratio, never a clipped state.
            np.divide(margin, closing_speed, out=ratios, where=crossing)
        bound = float(np.min(ratios))
        if not np.isfinite(bound):
            return {"alpha_limit": None, "strict": strict, "cells_iy_ix": [], "tie_count": 0}
        cells = np.argwhere(ratios == bound)
        return {"alpha_limit": bound, "strict": strict,
                "cells_iy_ix": (cells[:8]+offset).tolist(), "tie_count": int(len(cells))}

    interior = np.s_[1:-1, 1:-1]
    result = {
        "pressure_lower": limit(p[interior]+tol*pressure_scale_pa, -dp[interior], 1, False),
        "content_lower": limit(theta[interior]+tol, -dt[interior], 1, False),
        "content_upper": limit(1+tol-theta[interior], dt[interior], 1, False),
        "film_lower": limit(h-film_floor_m, -dh, 0, True),
    }
    finite = [v["alpha_limit"] for v in result.values() if v["alpha_limit"] is not None]
    first = min(finite) if finite else None
    return {"limits": result, "first_alpha_limit": first,
            "first_constraints": [k for k, v in result.items()
                                  if first is not None and v["alpha_limit"] == first],
            "full_step_strictly_inside": first is None or first > 1}
