"""Bounded CPU prescribed-film JFO reference on rectilinear SI grids.

This verification path supports graded far fields, not coupled elasticity.
Never apply the translation-invariant DC-FFT kernel to these unequal cells.
Default pressure is zero on all edges, with uniform west content and +x
entrainment. Optional prescribed pressure edges are verified separately.
"""

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, diags, hstack, vstack
from scipy.sparse.linalg import MatrixRankWarning

from ehl_solver.cavitation import _solve_active_system


def axis(values):
    values = np.asarray(values, dtype=np.float64)
    if (
        values.ndim != 1
        or len(values) < 3
        or not np.all(np.isfinite(values))
        or np.any(np.diff(values) <= 0)
    ):
        raise ValueError("SI axes must have at least three finite ascending nodes")
    return values


def flux_audit(
    x_m,
    y_m,
    film_m,
    pressure_pa,
    theta,
    viscosity_pa_s,
    speed_m_s,
    *,
    length_scale_m,
    film_scale_m,
    pressure_scale_pa,
):
    """Independent SI complete-face fluxes; control widths are midpoint distances."""
    x, y = axis(x_m), axis(y_m)
    h, p = np.asarray(film_m), np.asarray(pressure_pa)
    qx = (
        -0.5
        * (h[:, 1:] ** 3 + h[:, :-1] ** 3)
        * np.diff(p, axis=1)
        / (12 * viscosity_pa_s * np.diff(x))
        + speed_m_s * 0.5 * (h[:, 1:] + h[:, :-1]) * theta[:, :-1]
    )
    qy = (
        -0.5
        * (h[1:] ** 3 + h[:-1] ** 3)
        * np.diff(p, axis=0)
        / (12 * viscosity_pa_s * np.diff(y)[:, None])
    )
    wx, wy = 0.5 * (x[2:] - x[:-2]), 0.5 * (y[2:] - y[:-2])
    residual = np.diff(qx[1:-1], axis=1) / wx + np.diff(qy[:, 1:-1], axis=0) / wy[:, None]
    scale = (abs(qx[1:-1, 1:]) + abs(qx[1:-1, :-1])) * wy[:, None] + (
        abs(qy[1:, 1:-1]) + abs(qy[:-1, 1:-1])
    ) * wx
    volumes = residual * wy[:, None] * wx
    outward = np.array([-qx[1:-1, 0] @ wy, qx[1:-1, -1] @ wy, -qy[0, 1:-1] @ wx, qy[-1, 1:-1] @ wx])
    q_ref = film_scale_m**3 * pressure_scale_pa / (12 * viscosity_pa_s * length_scale_m)
    return {
        "mass_maximum": float(np.max(abs(residual)) * length_scale_m / q_ref),
        "relative_local_balance": float(np.max(abs(volumes) / np.maximum(scale, 1e-300))),
        "outward_volume_flow_m3_s": outward.tolist(),
        "net_volume_flow_m3_s": float(outward.sum()),
        "relative_global_balance": float(
            abs(outward.sum()) / np.maximum(abs(outward).sum(), 1e-300)
        ),
        "summed_volume_balance_m3_s": float(volumes.sum()),
        "maximum_complementarity": float(np.max(abs(p / pressure_scale_pa * (1 - theta)))),
    }


@dataclass(frozen=True)
class RectilinearSolution:
    pressure_pa: np.ndarray
    fractional_film_content: np.ndarray
    iterations: int
    audit: dict
    refinement_history: tuple = ()


def solve_rectilinear_cavitation(
    x_m,
    y_m,
    film_m,
    viscosity_pa_s,
    speed_m_s,
    *,
    length_scale_m,
    film_scale_m,
    pressure_scale_pa,
    inlet_content=1.0,
    maximum_iterations=100,
    boundary_pressure_pa=None,
    flux_refinement_steps=0,
):
    """SI interfaces; H=h/h_ref, P=p/p_ref, X=x/L, Y=y/L internally.

    Face mobility and film are arithmetic means, theta is west-upwinded.
    Divide face-flow differences by each interior midpoint control width.
    The inherited active-set tolerance is 1e-10 for P and theta. No clipping.
    At most 70000 nodes; sparse solves use positive diagonal equilibration.
    Optional nonnegative SI Dirichlet data are used only on edge nodes;
    positive-pressure edges are fully filled. None preserves zero edges.
    Up to three opt-in physical-flux defect corrections may use a wider
    residual dtype, with float64 states/corrections. Availability is checked
    explicitly; original float64 SI audits must improve before acceptance.
    """
    x, y = axis(x_m), axis(y_m)
    h = np.asarray(film_m, dtype=np.float64)
    if (
        h.shape != (len(y), len(x))
        or not np.all(np.isfinite(h))
        or np.any(h <= 0)
        or h.size > 70000
    ):
        raise ValueError("positive finite film must match (ny,nx), at most 70000 nodes")
    if not all(
        np.isfinite(v) and v > 0
        for v in (viscosity_pa_s, speed_m_s, length_scale_m, film_scale_m, pressure_scale_pa)
    ):
        raise ValueError("viscosity, mean speed and SI scales must be positive finite")
    if not np.isfinite(inlet_content) or not 0 <= inlet_content <= 1:
        raise ValueError("inlet content must lie in [0,1]")
    if (
        not isinstance(maximum_iterations, int)
        or isinstance(maximum_iterations, bool)
        or maximum_iterations < 1
    ):
        raise ValueError("maximum_iterations must be a positive integer")
    if (
        not isinstance(flux_refinement_steps, int)
        or isinstance(flux_refinement_steps, bool)
        or not 0 <= flux_refinement_steps <= 3
    ):
        raise ValueError("flux_refinement_steps must be an integer from zero to three")
    if flux_refinement_steps and np.finfo(np.longdouble).nmant <= np.finfo(float).nmant:
        raise RuntimeError("wider physical-flux residual arithmetic is unavailable")
    boundary = np.zeros(h.shape)
    if boundary_pressure_pa is not None:
        supplied = np.asarray(boundary_pressure_pa, dtype=np.float64)
        if supplied.shape != h.shape or not np.all(np.isfinite(supplied)) or np.any(supplied < 0):
            raise ValueError(
                "boundary pressure must be nonnegative finite SI data of shape (ny,nx)"
            )
        boundary[[0, -1]] = supplied[[0, -1]]
        boundary[:, [0, -1]] = supplied[:, [0, -1]]
        if np.any(boundary[1:-1, 0] > 0) and inlet_content != 1:
            raise ValueError("a positive-pressure inlet must be fully filled")
    H = h / film_scale_m
    X, Y = x / length_scale_m, y / length_scale_m
    parameter = (
        12 * viscosity_pa_s * speed_m_s * length_scale_m / (film_scale_m**2 * pressure_scale_pa)
    )
    ny, nx = h.shape
    n = (ny - 2) * (nx - 2)
    rows, cols, values = [], [], []
    rhs_mass = np.zeros(n)
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            k = (j - 1) * (nx - 2) + i - 1
            wx, wy = (X[i + 1] - X[i - 1]) / 2, (Y[j + 1] - Y[j - 1]) / 2
            diagonal = 0.0
            for dj, di, width, distance in (
                (0, -1, wx, X[i] - X[i - 1]),
                (0, 1, wx, X[i + 1] - X[i]),
                (-1, 0, wy, Y[j] - Y[j - 1]),
                (1, 0, wy, Y[j + 1] - Y[j]),
            ):
                coefficient = 0.5 * (H[j, i] ** 3 + H[j + dj, i + di] ** 3) / (width * distance)
                diagonal += coefficient
                if 0 < j + dj < ny - 1 and 0 < i + di < nx - 1:
                    rows.append(k)
                    cols.append(k + dj * (nx - 2) + di)
                    values.append(-coefficient)
                else:
                    rhs_mass[k] += coefficient * boundary[j + dj, i + di] / pressure_scale_pa
            rows.extend([k, k])
            cols.extend([k, n + k])
            values.extend([diagonal, parameter * 0.5 * (H[j, i] + H[j, i + 1]) / wx])
            west = parameter * 0.5 * (H[j, i] + H[j, i - 1]) / wx
            if i == 1:
                rhs_mass[k] += west * inlet_content
            else:
                rows.append(k)
                cols.append(n + k - 1)
                values.append(-west)
    mass = csr_matrix((values, (rows, cols)), shape=(n, 2 * n))
    full = np.ones(n, dtype=bool)
    seen = set()
    for iteration in range(1, maximum_iterations + 1):
        key = np.packbits(full).tobytes()
        if key in seen:
            raise RuntimeError("rectilinear cavitation active set cycled")
        seen.add(key)
        constraints = hstack([diags((~full).astype(float)), diags(full.astype(float))])
        matrix = vstack([mass, constraints], format="csr")
        with warnings.catch_warnings():
            warnings.simplefilter("error", MatrixRankWarning)
            state = _solve_active_system(matrix, np.r_[rhs_mass, full.astype(float)], True)
        if not np.all(np.isfinite(state)):
            raise RuntimeError("nonfinite rectilinear active-set state")
        p, theta = state[:n], state[n:]
        if np.any((~full) & (theta < -1e-10)):
            raise RuntimeError("negative rectilinear content")
        switch_dry, switch_wet = full & (p < -1e-10), (~full) & (theta > 1 + 1e-10)
        if np.any(switch_dry | switch_wet):
            full[switch_dry], full[switch_wet] = False, True
            continue

        def fields_and_audit(z):
            pressure, content = boundary.copy(), np.ones(h.shape)
            pressure[1:-1, 1:-1] = z[:n].reshape(ny - 2, nx - 2) * pressure_scale_pa
            content[1:-1, 1:-1] = z[n:].reshape(ny - 2, nx - 2)
            content[1:-1, 0] = inlet_content
            content[1:-1, -1] = content[1:-1, -2]
            content[0], content[-1] = content[1], content[-2]
            content[boundary > 0] = 1  # Prescribed wet Dirichlet edges, not unknown clipping.
            audit = flux_audit(
                x,
                y,
                h,
                pressure,
                content,
                viscosity_pa_s,
                speed_m_s,
                length_scale_m=length_scale_m,
                film_scale_m=film_scale_m,
                pressure_scale_pa=pressure_scale_pa,
            )
            return pressure, content, audit

        pressure, content, audit = fields_and_audit(state)
        traces = []
        for correction in range(flux_refinement_steps):
            from ehl_solver.flux_diagnostics import face_components

            residual = face_components(
                x, y, h, pressure, content, viscosity_pa_s, speed_m_s, dtype=np.longdouble
            )[-1]
            q_ref = film_scale_m**3 * pressure_scale_pa / (12 * viscosity_pa_s * length_scale_m)
            rhs = np.r_[
                -np.asarray(residual.ravel() * length_scale_m / q_ref, dtype=float),
                np.where(full, 1 - state[n:], -state[:n]),
            ]
            delta = _solve_active_system(matrix, rhs, True)
            trial = state + delta
            tp, tt, ta = fields_and_audit(trial)
            feasible = (
                trial[:n].min() >= -1e-10
                and trial[n:].min() >= -1e-10
                and trial[n:].max() <= 1 + 1e-10
                and ta["maximum_complementarity"] <= 1e-10
            )
            improved = feasible and ta["mass_maximum"] < audit["mass_maximum"]
            traces.append(
                {
                    "step": correction + 1,
                    "before": audit["mass_maximum"],
                    "trial": ta["mass_maximum"],
                    "accepted": bool(improved),
                }
            )
            if not improved:
                break
            state, pressure, content, audit = trial, tp, tt, ta
        return RectilinearSolution(pressure, content, iteration, audit, tuple(traces))
    raise RuntimeError("rectilinear active-set iteration limit reached")
