"""Finite-reservoir prescribed-film JFO exterior response on CPU.

No far-field pressure trace is borrowed. Exterior pressure AND transported
content are eliminated for each trial active set, then reconstructed and
checked. This is a discrete response for a specified finite reservoir, not
an infinite-domain boundary law or a coupled nonuniform elastic solver.
"""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix, diags, hstack, vstack
from scipy.sparse.linalg import splu

from ehl_solver.cavitation import _solve_active_system
from ehl_solver.rectilinear_cavitation import axis, flux_audit


@dataclass(frozen=True)
class ExteriorSolution:
    pressure_pa: np.ndarray
    fractional_film_content: np.ndarray
    retained_mask: np.ndarray
    audit: dict
    history: tuple
    full_active_set: np.ndarray | None = None


@dataclass(frozen=True)
class ExteriorTangent:
    """Direction per unit dimensionless perturbation amplitude; SI pressure [Pa]."""

    pressure_direction_pa: np.ndarray
    content_direction: np.ndarray
    linear_info: dict


def _mass_operator(x, y, film, parameter, inlet):
    """Independent conservative dimensionless row assembly, zero pressure edges."""
    ny, nx = film.shape
    n = (ny - 2) * (nx - 2)
    rows, cols, values = [], [], []
    rhs = np.zeros(n)
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            k = (j - 1) * (nx - 2) + i - 1
            wx, wy = (x[i + 1] - x[i - 1]) / 2, (y[j + 1] - y[j - 1]) / 2
            diagonal = 0.0
            for dj, di, width, distance in (
                (0, -1, wx, x[i] - x[i - 1]),
                (0, 1, wx, x[i + 1] - x[i]),
                (-1, 0, wy, y[j] - y[j - 1]),
                (1, 0, wy, y[j + 1] - y[j]),
            ):
                coefficient = (film[j, i] ** 3 + film[j + dj, i + di] ** 3) / (2 * width * distance)
                diagonal += coefficient
                if 0 < j + dj < ny - 1 and 0 < i + di < nx - 1:
                    rows.append(k)
                    cols.append(k + dj * (nx - 2) + di)
                    values.append(-coefficient)
            rows.extend((k, k))
            cols.extend((k, n + k))
            values.extend((diagonal, parameter * (film[j, i] + film[j, i + 1]) / (2 * wx)))
            west = parameter * (film[j, i] + film[j, i - 1]) / (2 * wx)
            if i == 1:
                rhs[k] = west * inlet
            else:
                rows.append(k)
                cols.append(n + k - 1)
                values.append(-west)
    return coo_matrix((values, (rows, cols)), shape=(n, 2 * n)).tocsr(), rhs


def _exterior_factor(matrix):
    """Positive row/column equilibration; reuse one LU for all response columns."""
    row = abs(matrix).max(axis=1).toarray().ravel()
    if np.any(row <= 0) or not np.all(np.isfinite(row)):
        raise RuntimeError("invalid exterior row scale")
    row = 1 / row
    scaled = diags(row) @ matrix
    col = abs(scaled).max(axis=0).toarray().ravel()
    if np.any(col <= 0) or not np.all(np.isfinite(col)):
        raise RuntimeError("invalid exterior column scale")
    col = 1 / col
    factor = splu((scaled @ diags(col)).tocsc())

    def solve(rhs):
        if rhs.ndim == 1:
            return col * factor.solve(row * rhs)
        return col[:, None] * factor.solve(row[:, None] * rhs)

    return solve


def _condensed_solve(matrix, rhs, retained):
    """z_E = offset + response z_B; Schur contribution touches only B rows.

    The full pressure/content constraints remain in the eliminated system.
    B includes pressure and content columns communicating with the exterior;
    transported deficit/content is never silently reset at an inner inlet.
    """
    n = len(retained)
    ii = np.r_[np.flatnonzero(retained), n + np.flatnonzero(retained)]
    ee = np.r_[np.flatnonzero(~retained), n + np.flatnonzero(~retained)]
    aii, aie = matrix[ii][:, ii], matrix[ii][:, ee]
    aei, aee = matrix[ee][:, ii], matrix[ee][:, ee]
    incoming = np.unique(aei.nonzero()[1])
    outgoing = np.unique(aie.nonzero()[0])
    # Store the response, solving at most 16 columns together. Include temporary
    # scaled RHS/solution arrays and the small boundary block before allocation.
    batch = min(16, len(incoming))
    response_bytes = 8 * len(ee) * (len(incoming) + 3 * batch + 3) + 8 * len(incoming) * len(
        outgoing
    )
    if len(incoming) > 256 or response_bytes > 64 * 1024**2:
        raise ValueError("exterior response exceeds 256 columns or 64 MiB work budget")
    solve = _exterior_factor(aee)
    offset = solve(rhs[ee])
    response = np.empty((len(ee), len(incoming)))
    for start in range(0, len(incoming), batch):
        stop = start + batch
        response[:, start:stop] = -solve(aei[:, incoming[start:stop]].toarray())
    block = np.asarray(aie[outgoing] @ response)
    rr, cc = np.meshgrid(outgoing, incoming, indexing="ij")
    schur = aii + coo_matrix((block.ravel(), (rr.ravel(), cc.ravel())), shape=aii.shape).tocsr()
    inner = _solve_active_system(schur, rhs[ii] - aie @ offset, True)
    state = np.empty(2 * n)
    state[ii], state[ee] = inner, offset + response @ inner[incoming]
    residual = matrix @ state - rhs
    denominator = np.asarray(abs(matrix) @ abs(state)) + abs(rhs)
    backward = float(np.max(abs(residual) / np.maximum(denominator, 1e-300)))
    return state, {
        "retained_unknowns": len(ii),
        "exterior_unknowns": len(ee),
        "response_columns": len(incoming),
        "dense_work_estimate_bytes": response_bytes,
        "relative_row_backward_error": backward,
    }


def solve_exterior_response(
    x_m,
    y_m,
    film_m,
    viscosity_pa_s,
    speed_m_s,
    *,
    core_bounds_m,
    length_scale_m,
    film_scale_m,
    pressure_scale_pa,
    inlet_content=1.0,
    maximum_iterations=100,
):
    """Solve a finite ambient-pressure reservoir with a condensed exterior.

    SI public fields: x/y/h [m], eta [Pa s], Ue [m/s], p [Pa]. Arrays
    (ny,nx), +x entrainment. core_bounds_m=(xmin,xmax,ymin,ymax) selects
    retained interior nodes, not a new physical pressure/supply boundary.
    Internally X=x/L,Y=y/L,H=h/h_ref,P=p/p_ref and Lambda=12 eta Ue L /
    (h_ref^2 p_ref). At most 20000 nodes, 100 active sets, 256 response
    columns, 64 MiB estimated dense work. No clipping or regularization.
    Original independent SI audits are returned, never replaced by the
    Schur/backward error. This routine does not assert domain independence.
    """
    x, y = axis(x_m), axis(y_m)
    film = np.asarray(film_m, dtype=float)
    bounds = np.asarray(core_bounds_m, dtype=float)
    if (
        film.shape != (len(y), len(x))
        or film.size > 20000
        or not np.all(np.isfinite(film))
        or np.any(film <= 0)
        or bounds.shape != (4,)
        or not np.all(np.isfinite(bounds))
    ):
        raise ValueError("invalid prescribed film, core bounds or 20000-node budget")
    if not all(
        np.isfinite(v) and v > 0
        for v in (viscosity_pa_s, speed_m_s, length_scale_m, film_scale_m, pressure_scale_pa)
    ):
        raise ValueError("positive finite SI scales, viscosity and speed required")
    if not np.isfinite(inlet_content) or not 0 <= inlet_content <= 1:
        raise ValueError("inlet content must lie in [0,1]")
    if (
        not isinstance(maximum_iterations, int)
        or isinstance(maximum_iterations, bool)
        or not 1 <= maximum_iterations <= 100
    ):
        raise ValueError("active-set budget must lie in 1..100")
    if not (x[0] < bounds[0] < bounds[1] < x[-1] and y[0] < bounds[2] < bounds[3] < y[-1]):
        raise ValueError("core must be a strict interior rectangle")
    mask = (
        (x[None, :] >= bounds[0])
        & (x[None, :] <= bounds[1])
        & (y[:, None] >= bounds[2])
        & (y[:, None] <= bounds[3])
    )
    if np.count_nonzero(mask.any(axis=0)) < 3 or np.count_nonzero(mask.any(axis=1)) < 3:
        raise ValueError("core must retain at least three nodes in each direction")
    retained = mask[1:-1, 1:-1].ravel()
    mass, forcing = _mass_operator(
        x / length_scale_m,
        y / length_scale_m,
        film / film_scale_m,
        12 * viscosity_pa_s * speed_m_s * length_scale_m / (film_scale_m**2 * pressure_scale_pa),
        inlet_content,
    )
    n = len(retained)
    full, seen, history = np.ones(n, dtype=bool), set(), []
    for iteration in range(1, maximum_iterations + 1):
        key = np.packbits(full).tobytes()
        if key in seen:
            raise RuntimeError("exterior JFO active set cycled")
        seen.add(key)
        constraints = hstack([diags((~full).astype(float)), diags(full.astype(float))])
        system = vstack([mass, constraints], format="csr")
        state, info = _condensed_solve(system, np.r_[forcing, full.astype(float)], retained)
        if not np.all(np.isfinite(state)):
            raise RuntimeError("nonfinite exterior response")
        p, theta = state[:n], state[n:]
        if np.any((~full) & (theta < -1e-10)):
            raise RuntimeError("negative exterior content")
        dry, wet = full & (p < -1e-10), (~full) & (theta > 1 + 1e-10)
        info.update(
            iteration=iteration,
            switched_nodes=int(np.sum(dry | wet)),
            exterior_dry_nodes=int(np.sum((~full) & (~retained))),
        )
        history.append(info)
        if np.any(dry | wet):
            full[dry], full[wet] = False, True
            continue
        pressure, content = np.zeros_like(film), np.ones_like(film)
        pressure[1:-1, 1:-1] = p.reshape(film.shape[0] - 2, film.shape[1] - 2) * pressure_scale_pa
        content[1:-1, 1:-1] = theta.reshape(film.shape[0] - 2, film.shape[1] - 2)
        content[1:-1, 0], content[1:-1, -1] = inlet_content, content[1:-1, -2]
        content[0], content[-1] = content[1], content[-2]
        audit = flux_audit(
            x,
            y,
            film,
            pressure,
            content,
            viscosity_pa_s,
            speed_m_s,
            length_scale_m=length_scale_m,
            film_scale_m=film_scale_m,
            pressure_scale_pa=pressure_scale_pa,
        )
        return ExteriorSolution(pressure, content, mask, audit, tuple(history), full.copy())
    raise RuntimeError("exterior JFO active-set iteration limit reached")


def exterior_source_budget(x_m, y_m, solution, e_star_pa, target_points_m):
    """SI nodal source-cell force/displacement partition, not a coupled solve.

    Zero physical pressure edges imply midpoint source-cell areas equal the
    nonzero trapezoidal nodal weights. Cells belong to retained/exterior sets
    by their source node; interface cells straddle the inner node rectangle.
    This disjoint partition is not a sharp rectangle pressure integral.
    Selected-target displacement uses direct cell integration on either grid.
    """
    from ehl_solver.elasticity_audit import direct_midpoint_displacement

    x, y = axis(x_m), axis(y_m)
    p, mask = solution.pressure_pa, solution.retained_mask
    if p.shape != (len(y), len(x)) or mask.shape != p.shape:
        raise ValueError("source partition must match axes")
    weights = ((y[2:] - y[:-2]) / 2)[:, None] * ((x[2:] - x[:-2]) / 2)
    inner, exterior = np.where(mask, p, 0), np.where(mask, 0, p)
    targets = np.asarray(target_points_m, dtype=float)
    wi, we, total = [
        direct_midpoint_displacement(x, y, field, e_star_pa, targets)
        for field in (inner, exterior, p)
    ]
    qi, qe = [float(np.sum(field[1:-1, 1:-1] * weights)) for field in (inner, exterior)]
    return {
        "retained_source_load_n": qi,
        "exterior_source_load_n": qe,
        "total_load_n": qi + qe,
        "target_points_m": targets.tolist(),
        "e_star_pa": e_star_pa,
        "retained_displacement_m": wi.tolist(),
        "exterior_displacement_m": we.tolist(),
        "total_displacement_m": total.tolist(),
        "maximum_displacement_partition_error_m": float(abs(wi + we - total).max()),
    }


def differentiate_exterior_response(
    x_m,
    y_m,
    film_m,
    film_direction_m,
    solution,
    viscosity_pa_s,
    speed_m_s,
    *,
    length_scale_m,
    film_scale_m,
    pressure_scale_pa,
):
    """Conditional prescribed-film derivative, with the returned active set fixed.

    h(epsilon)=h+epsilon*film_direction_m. Differentiate original conservative
    fluxes, solve A dz=-d(div q)/dh dh with zero constraint directions, and
    condense/reconstruct exterior directions. Physical reservoir and supply,
    scales, core mask and active set stay fixed. This is NOT a derivative
    through a front change, nor a coupled-film Newton implementation.
    """
    x, y = axis(x_m), axis(y_m)
    h, dh = np.asarray(film_m, float), np.asarray(film_direction_m, float)
    if (
        h.shape != (len(y), len(x))
        or h.size > 20000
        or dh.shape != h.shape
        or not np.all(np.isfinite(h))
        or not np.all(np.isfinite(dh))
        or np.any(h <= 0)
        or solution.pressure_pa.shape != h.shape
        or solution.fractional_film_content.shape != h.shape
        or solution.retained_mask.shape != h.shape
        or solution.retained_mask.dtype != bool
        or not np.all(np.isfinite(solution.pressure_pa))
        or not np.all(np.isfinite(solution.fractional_film_content))
    ):
        raise ValueError("invalid prescribed-film tangent arrays or node budget")
    if not all(
        np.isfinite(v) and v > 0
        for v in (viscosity_pa_s, speed_m_s, length_scale_m, film_scale_m, pressure_scale_pa)
    ):
        raise ValueError("positive finite tangent scales, viscosity and speed required")
    n = (len(y) - 2) * (len(x) - 2)
    if (
        solution.full_active_set is None
        or solution.full_active_set.shape != (n,)
        or solution.full_active_set.dtype != bool
    ):
        raise ValueError("tangent requires the solver's recorded full active set")
    audit = flux_audit(
        x,
        y,
        h,
        solution.pressure_pa,
        solution.fractional_film_content,
        viscosity_pa_s,
        speed_m_s,
        length_scale_m=length_scale_m,
        film_scale_m=film_scale_m,
        pressure_scale_pa=pressure_scale_pa,
    )
    if (
        audit["mass_maximum"] > 1e-8
        or audit["maximum_complementarity"] > 1e-10
        or solution.pressure_pa.min() < -1e-10 * pressure_scale_pa
        or solution.fractional_film_content.min() < -1e-10
        or solution.fractional_film_content.max() > 1 + 1e-10
    ):
        raise ValueError("tangent requires a feasible converged root for the supplied film")
    X, Y, H, dH = x / length_scale_m, y / length_scale_m, h / film_scale_m, dh / film_scale_m
    parameter = (
        12 * viscosity_pa_s * speed_m_s * length_scale_m / (film_scale_m**2 * pressure_scale_pa)
    )
    mass, _ = _mass_operator(X, Y, H, parameter, 1)
    full = solution.full_active_set
    constraints = hstack([diags((~full).astype(float)), diags(full.astype(float))])
    matrix = vstack([mass, constraints], format="csr")
    p = solution.pressure_pa / pressure_scale_pa
    theta = solution.fractional_film_content
    d_mobility = 3 * H**2 * dH
    dqx = -(d_mobility[:, 1:] + d_mobility[:, :-1]) / 2 * np.diff(p, axis=1) / np.diff(X)
    dqx += parameter * (dH[:, 1:] + dH[:, :-1]) / 2 * theta[:, :-1]
    dqy = -(d_mobility[1:] + d_mobility[:-1]) / 2 * np.diff(p, axis=0) / np.diff(Y)[:, None]
    wx, wy = (X[2:] - X[:-2]) / 2, (Y[2:] - Y[:-2]) / 2
    residual = np.diff(dqx[1:-1], axis=1) / wx + np.diff(dqy[:, 1:-1], axis=0) / wy[:, None]
    z, info = _condensed_solve(
        matrix, np.r_[-residual.ravel(), np.zeros(n)], solution.retained_mask[1:-1, 1:-1].ravel()
    )
    if not np.all(np.isfinite(z)):
        raise RuntimeError("nonfinite prescribed-film exterior tangent")
    dp, dt = np.zeros_like(h), np.zeros_like(h)
    dp[1:-1, 1:-1] = z[:n].reshape(h.shape[0] - 2, h.shape[1] - 2) * pressure_scale_pa
    dt[1:-1, 1:-1] = z[n:].reshape(h.shape[0] - 2, h.shape[1] - 2)
    dt[1:-1, -1] = dt[1:-1, -2]
    dt[0], dt[-1] = dt[1], dt[-2]
    return ExteriorTangent(dp, dt, info)


def retained_source_displacements(x_m, y_m, solution, e_star_pa):
    """Direct free-space source accounting at EVERY retained target, bounded.

    Return SI target coordinates and retained/exterior/total displacement [m].
    At most 2048 targets and 20 million source-target pairs; existing direct
    source-cell evaluations are batched at 32 targets. No dense kernel is
    stored and no unequal-cell FFT is used. This evaluates a given pressure,
    not an elasticity-coupled pressure solve.
    """
    from ehl_solver.elasticity_audit import direct_midpoint_displacement

    x, y = axis(x_m), axis(y_m)
    p, mask = solution.pressure_pa, solution.retained_mask
    if p.shape != (len(y), len(x)) or mask.shape != p.shape or mask.dtype != bool:
        raise ValueError("invalid retained-target source fields")
    indices = np.argwhere(mask)
    if len(indices) > 2048 or len(indices) * p.size > 20_000_000:
        raise ValueError("retained displacement exceeds 2048 targets or 20 million pairs")
    if len(indices) == 0:
        raise ValueError("no retained targets")
    points = np.column_stack([x[indices[:, 1]], y[indices[:, 0]]])
    output = {"target_points_m": points}
    for name, field in (
        ("retained_displacement_m", np.where(mask, p, 0)),
        ("exterior_displacement_m", np.where(mask, 0, p)),
        ("total_displacement_m", p),
    ):
        output[name] = np.concatenate(
            [
                direct_midpoint_displacement(x, y, field, e_star_pa, points[k : k + 32])
                for k in range(0, len(points), 32)
            ]
        )
    return output
