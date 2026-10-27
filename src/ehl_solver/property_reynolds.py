"""CPU prescribed-film JFO with pressure-dependent SI density and viscosity.

Fields (ny,nx), +x entrainment. Physical mass flux, not volume flux, is
conserved. The old constant-property reference remains independent.
"""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix, diags, hstack, vstack

from ehl_solver.cavitation import _solve_active_system
from ehl_solver.rectilinear_cavitation import axis


@dataclass(frozen=True)
class PropertySolution:
    pressure_pa: np.ndarray
    fractional_film_content: np.ndarray
    audit: dict
    history: tuple


class PropertyConvergenceError(RuntimeError):
    def __init__(self, message, history, last_solution):
        super().__init__(message)
        self.history = tuple(history)
        self.last_solution = last_solution


def mass_fluxes(x, y, h, p, theta, fluid, speed, *, strength=1):
    """SI face fluxes [kg/(m s)]; density and viscosity use unrelaxed p."""
    rho, eta, _, _ = fluid.properties(p, strength=strength)
    mobility = rho * h**3 / (12 * eta)
    capacity = rho * h
    qx = -(mobility[:, 1:] + mobility[:, :-1]) / 2 * np.diff(p, axis=1) / np.diff(x)
    qx += speed * (capacity[:, 1:] + capacity[:, :-1]) / 2 * theta[:, :-1]
    qy = -(mobility[1:] + mobility[:-1]) / 2 * np.diff(p, axis=0) / np.diff(y)[:, None]
    return qx, qy


def mass_audit(
    x, y, h, p, theta, fluid, speed, *, length_scale_m, film_scale_m, pressure_scale_pa, strength=1
):
    """Independent SI midpoint face audit; original dimensional mass balance."""
    qx, qy = mass_fluxes(x, y, h, p, theta, fluid, speed, strength=strength)
    wx, wy = (x[2:] - x[:-2]) / 2, (y[2:] - y[:-2]) / 2
    residual = np.diff(qx[1:-1], axis=1) / wx + np.diff(qy[:, 1:-1], axis=0) / wy[:, None]
    volumes = residual * wx * wy[:, None]
    local = (abs(qx[1:-1, 1:]) + abs(qx[1:-1, :-1])) * wy[:, None] + (
        abs(qy[1:, 1:-1]) + abs(qy[:-1, 1:-1])
    ) * wx
    out = np.array([-qx[1:-1, 0] @ wy, qx[1:-1, -1] @ wy, -qy[0, 1:-1] @ wx, qy[-1, 1:-1] @ wx])
    ref = (
        fluid.density_kg_m3
        * film_scale_m**3
        * pressure_scale_pa
        / (12 * fluid.viscosity_pa_s * length_scale_m)
    )
    return {
        "mass_maximum": float(abs(residual).max() * length_scale_m / ref),
        "maximum_si_mass_divergence_kg_m2_s": float(abs(residual).max()),
        "relative_local_balance": float(np.max(abs(volumes) / np.maximum(local, 1e-300))),
        "relative_global_balance": float(abs(out.sum()) / max(abs(out).sum(), 1e-300)),
        "outward_mass_flow_kg_s": out.tolist(),
        "net_mass_flow_kg_s": float(out.sum()),
        "summed_cell_mass_defect_kg_s": float(volumes.sum()),
        "maximum_complementarity": float(np.max(abs(p / pressure_scale_pa * (1 - theta)))),
    }


def admissible(audit, p, theta, pressure_scale_pa):
    return bool(
        audit["mass_maximum"] <= 1e-8
        and audit["relative_local_balance"] <= 1e-10
        and audit["relative_global_balance"] <= 1e-10
        and audit["maximum_complementarity"] <= 1e-10
        and p.min() >= -1e-10 * pressure_scale_pa
        and theta.min() >= -1e-10
        and theta.max() <= 1 + 1e-10
    )


def _frozen_system(X, Y, mobility, capacity, parameter, boundary, inlet):
    ny, nx = mobility.shape
    n = (ny - 2) * (nx - 2)
    rows, cols, vals = [], [], []
    rhs = np.zeros(n)
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            k = (j - 1) * (nx - 2) + i - 1
            wx = (X[i + 1] - X[i - 1]) / 2
            wy = (Y[j + 1] - Y[j - 1]) / 2
            diagonal = 0.0
            for dj, di, width, distance in (
                (0, -1, wx, X[i] - X[i - 1]),
                (0, 1, wx, X[i + 1] - X[i]),
                (-1, 0, wy, Y[j] - Y[j - 1]),
                (1, 0, wy, Y[j + 1] - Y[j]),
            ):
                c = (mobility[j, i] + mobility[j + dj, i + di]) / (2 * width * distance)
                diagonal += c
                if 0 < j + dj < ny - 1 and 0 < i + di < nx - 1:
                    rows.append(k)
                    cols.append(k + dj * (nx - 2) + di)
                    vals.append(-c)
                else:
                    rhs[k] += c * boundary[j + dj, i + di]
            rows.extend((k, k))
            cols.extend((k, n + k))
            vals.extend((diagonal, parameter * (capacity[j, i] + capacity[j, i + 1]) / (2 * wx)))
            west = parameter * (capacity[j, i] + capacity[j, i - 1]) / (2 * wx)
            if i == 1:
                rhs[k] += west * inlet
            else:
                rows.append(k)
                cols.append(n + k - 1)
                vals.append(-west)
    return coo_matrix((vals, (rows, cols)), shape=(n, 2 * n)).tocsr(), rhs


def _active_solve(mass, rhs, full):
    n = len(full)
    seen = set()
    for iteration in range(1, 101):
        key = np.packbits(full).tobytes()
        if key in seen:
            raise RuntimeError("variable-property frozen active set cycled")
        seen.add(key)
        constraints = hstack([diags((~full).astype(float)), diags(full.astype(float))])
        state = _solve_active_system(
            vstack([mass, constraints], format="csr"), np.r_[rhs, full.astype(float)], True
        )
        if not np.all(np.isfinite(state)):
            raise RuntimeError("nonfinite frozen JFO root")
        p, theta = state[:n], state[n:]
        if np.any((~full) & (theta < -1e-10)):
            raise RuntimeError("negative frozen JFO content")
        dry, wet = full & (p < -1e-10), (~full) & (theta > 1 + 1e-10)
        if not np.any(dry | wet):
            return p, theta, full, iteration
        full = full.copy()
        full[dry] = False
        full[wet] = True
    raise RuntimeError("frozen JFO active-set budget exhausted")


def solve_property_reynolds(
    x_m,
    y_m,
    film_m,
    fluid,
    speed_m_s,
    *,
    length_scale_m,
    film_scale_m,
    pressure_scale_pa,
    inlet_content=1.0,
    boundary_pressure_pa=None,
    strengths=(0, 0.25, 0.5, 0.75, 1),
    maximum_iterations_per_stage=100,
    initial_pressure_pa=None,
):
    """Bounded coefficient Picard/active sets; SI input, no field clipping.

    X=x/L,Y=y/L,H=h/href,P=p/pref, R=rho/rho0,V=eta/eta0.
    qhat=-avg(R H^3/V) grad P + Lambda avg(R H) theta_west.
    Lambda=12 eta0 Ue L/(href^2 pref); physical mass-flux scale is
    rho0 href^3 pref/(12 eta0 L). Density is the liquid law, theta content.
    Only coefficients are relaxed (rho arithmetic, viscosity logarithmic).
    Each returned p/theta pair solves a frozen constrained problem; convergence
    MUST pass the original unrelaxed SI mass/constraint audit. Up to 20000
    nodes, 100 nonlinear iterations per strength and 100 active-set iterations.
    """
    x, y = axis(x_m), axis(y_m)
    h = np.asarray(film_m, float)
    if (
        h.shape != (len(y), len(x))
        or h.size > 20000
        or not np.all(np.isfinite(h))
        or np.any(h <= 0)
    ):
        raise ValueError("invalid positive film or 20000-node budget")
    if (
        not all(np.isfinite(v) and v > 0 for v in (length_scale_m, film_scale_m, pressure_scale_pa))
        or not np.isfinite(speed_m_s)
        or speed_m_s < 0
    ):
        raise ValueError("invalid SI scales or speed")
    if not np.isfinite(inlet_content) or not 0 <= inlet_content <= 1:
        raise ValueError("inlet content must lie in [0,1]")
    if (
        not strengths
        or any(not np.isfinite(s) or not 0 <= s <= 1 for s in strengths)
        or any(b <= a for a, b in zip(strengths, strengths[1:], strict=False))
    ):
        raise ValueError("strengths must increase in [0,1]")
    if (
        not isinstance(maximum_iterations_per_stage, int)
        or isinstance(maximum_iterations_per_stage, bool)
        or not 1 <= maximum_iterations_per_stage <= 100
    ):
        raise ValueError("nonlinear stage budget must lie in 1..100")
    boundary = np.zeros_like(h)
    if boundary_pressure_pa is not None:
        given = np.asarray(boundary_pressure_pa, float)
        if given.shape != h.shape or not np.all(np.isfinite(given)) or np.any(given < 0):
            raise ValueError("invalid nonnegative SI boundary pressure")
        boundary[[0, -1]] = given[[0, -1]]
        boundary[:, [0, -1]] = given[:, [0, -1]]
        if np.any(boundary[1:-1, 0] > 0) and inlet_content != 1:
            raise ValueError("positive inlet pressure requires flooded content")
    pressure = (
        boundary.copy()
        if initial_pressure_pa is None
        else np.array(initial_pressure_pa, float, copy=True)
    )
    if (
        pressure.shape != h.shape
        or not np.all(np.isfinite(pressure))
        or np.min(pressure) < -1e-10 * pressure_scale_pa
    ):
        raise ValueError("invalid initial SI pressure")
    pressure[[0, -1]] = boundary[[0, -1]]
    pressure[:, [0, -1]] = boundary[:, [0, -1]]
    H = h / film_scale_m
    X, Y = x / length_scale_m, y / length_scale_m
    parameter = (
        12
        * fluid.viscosity_pa_s
        * speed_m_s
        * length_scale_m
        / (film_scale_m**2 * pressure_scale_pa)
    )
    full = np.ones((len(y) - 2) * (len(x) - 2), dtype=bool)
    history = []
    last = None
    for strength in strengths:
        r, v, _, _ = fluid.ratios(pressure, strength=strength)
        for iteration in range(1, maximum_iterations_per_stage + 1):
            try:
                # Known edge properties must never be relaxed away from their Dirichlet values.
                br, bv, _, _ = fluid.ratios(boundary, strength=strength)
                r[[0, -1]] = br[[0, -1]]
                r[:, [0, -1]] = br[:, [0, -1]]
                v[[0, -1]] = bv[[0, -1]]
                v[:, [0, -1]] = bv[:, [0, -1]]
                matrix, rhs = _frozen_system(
                    X,
                    Y,
                    r * H**3 / v,
                    r * H,
                    parameter,
                    boundary / pressure_scale_pa,
                    inlet_content,
                )
                pi, ti, full, active = _active_solve(matrix, rhs, full)
                candidate = boundary.copy()
                theta = np.ones_like(h)
                candidate[1:-1, 1:-1] = pi.reshape(len(y) - 2, len(x) - 2) * pressure_scale_pa
                theta[1:-1, 1:-1] = ti.reshape(len(y) - 2, len(x) - 2)
                theta[1:-1, 0] = inlet_content
                theta[1:-1, -1] = theta[1:-1, -2]
                theta[0], theta[-1] = theta[1], theta[-2]
                theta[boundary > 0] = 1
                audit = mass_audit(
                    x,
                    y,
                    h,
                    candidate,
                    theta,
                    fluid,
                    speed_m_s,
                    length_scale_m=length_scale_m,
                    film_scale_m=film_scale_m,
                    pressure_scale_pa=pressure_scale_pa,
                    strength=strength,
                )
                change = float(
                    np.linalg.norm(candidate - pressure)
                    / max(np.linalg.norm(candidate), pressure_scale_pa * 1e-300)
                )
                history.append(
                    {
                        "strength": strength,
                        "iteration": iteration,
                        "pressure_update": change,
                        "audit": audit,
                        "active_iterations": active,
                    }
                )
                last = PropertySolution(candidate, theta, audit, tuple(history))
                if change <= 1e-9 and admissible(audit, candidate, theta, pressure_scale_pa):
                    pressure = candidate
                    break
                true_r, true_v, _, _ = fluid.ratios(candidate, strength=strength)
                r = 0.5 * (r + true_r)
                v = np.exp(0.5 * (np.log(v) + np.log(true_v)))
                pressure = candidate
            except (RuntimeError, ValueError, FloatingPointError) as exc:
                raise PropertyConvergenceError(str(exc), history, last) from exc
        else:
            raise PropertyConvergenceError(f"property stage {strength} exhausted", history, last)
    return last


def mass_flux_direction(x, y, h, p, theta, dh, dp, dtheta, fluid, speed, *, strength=1):
    """Analytic SI face-flux direction for independently supplied h,p,theta.

    Directions are per unit dimensionless amplitude. This smooth flux
    derivative does not differentiate the nonsmooth cavitation constraints.
    """
    rho, eta, drho, deta = fluid.properties(p, strength=strength)
    m = rho * h**3 / (12 * eta)
    c = rho * h
    dm = ((drho / eta - rho * deta / eta**2) * h**3 * dp + 3 * rho * h**2 / eta * dh) / 12
    dc = drho * h * dp + rho * dh
    qx = -(dm[:, 1:] + dm[:, :-1]) / 2 * np.diff(p, axis=1) / np.diff(x)
    qx -= (m[:, 1:] + m[:, :-1]) / 2 * np.diff(dp, axis=1) / np.diff(x)
    qx += speed * (
        (dc[:, 1:] + dc[:, :-1]) / 2 * theta[:, :-1] + (c[:, 1:] + c[:, :-1]) / 2 * dtheta[:, :-1]
    )
    qy = -(dm[1:] + dm[:-1]) / 2 * np.diff(p, axis=0) / np.diff(y)[:, None]
    qy -= (m[1:] + m[:-1]) / 2 * np.diff(dp, axis=0) / np.diff(y)[:, None]
    return qx, qy
