"""Isolated MC-limited capacity flux benchmark; not a coupled EHL solver.

SI fields (ny,nx), +x mean entrainment. Reconstruct A=rho*h*theta at x faces
using a bounded piecewise linear slope. Pressure mobility stays arithmetic.
The west face uses a bounded one-sided slope from the first three nodes.
No field projection, boundary change or global TVD/EHL accuracy claim is made.
"""

import numpy as np

from ehl_solver.transport_candidate import _fields, donor_flux_direction, donor_fluxes


def _reconstruction(x, capacity, direction=None):
    """Shared-face reconstruction and a selected piecewise-branch linearization.

    At limiter ties select centered, then backward, then forward. At extrema
    or zero slopes select zero derivative. The resulting JVP is linear in its
    direction, but is not a classical derivative at limiter switching points.
    """
    widths = np.diff(x)
    differences = np.diff(capacity, axis=1) / widths
    backward, forward = differences[:, :-1], differences[:, 1:]
    centered = (widths[1:] * backward + widths[:-1] * forward) / (widths[:-1] + widths[1:])
    candidates = np.stack([centered, 2 * backward, 2 * forward])
    selected = np.argmin(abs(candidates), axis=0)
    monotone = ((backward > 0) & (forward > 0)) | ((backward < 0) & (forward < 0))
    slope = np.where(monotone, np.take_along_axis(candidates, selected[None], axis=0)[0], 0.0)
    # One-sided quadratic derivative at the west node; bound by its next face.
    db, df = np.diff(capacity[:, :3], axis=1).T / widths[:2, None]
    west = ((2 * widths[0] + widths[1]) * db - widths[0] * df) / (widths[0] + widths[1])
    west_active = ((west > 0) & (db > 0)) | ((west < 0) & (db < 0))
    west_centered = abs(west) <= 2 * abs(db)
    west_slope = np.where(west_active, np.where(west_centered, west, 2 * db), 0.0)
    face = capacity[:, :-1].copy()
    face[:, 0] += 0.5 * widths[0] * west_slope
    face[:, 1:] += 0.5 * widths[1:] * slope
    if direction is None:
        return face
    differences = np.diff(direction, axis=1) / widths
    backward, forward = differences[:, :-1], differences[:, 1:]
    centered = (widths[1:] * backward + widths[:-1] * forward) / (widths[:-1] + widths[1:])
    candidates = np.stack([centered, 2 * backward, 2 * forward])
    dslope = np.where(monotone, np.take_along_axis(candidates, selected[None], axis=0)[0], 0.0)
    db, df = np.diff(direction[:, :3], axis=1).T / widths[:2, None]
    west = ((2 * widths[0] + widths[1]) * db - widths[0] * df) / (widths[0] + widths[1])
    dwest = np.where(west_active, np.where(west_centered, west, 2 * db), 0.0)
    dface = direction[:, :-1].copy()
    dface[:, 0] += 0.5 * widths[0] * dwest
    dface[:, 1:] += 0.5 * widths[1:] * dslope
    return face, dface


def limited_fluxes(x, y, h, p, theta, fluid, speed, *, strength=1.0):
    """Return shared x/y mass fluxes [kg/(m s)]; no nonlinear solve implied."""
    x, y, h, p, theta = _fields(x, y, h, p, theta, speed)
    rho, _, _, _ = fluid.properties(p, strength=strength)
    capacity = rho * h * theta
    qx, qy = donor_fluxes(x, y, h, p, theta, fluid, speed, strength=strength)
    qx += speed * (_reconstruction(x, capacity) - capacity[:, :-1])
    return qx, qy


def limited_flux_direction(x, y, h, p, theta, dh, dp, dt, fluid, speed, *, strength=1.0):
    """Analytic selected-branch flux JVP, including all h,p,theta directions.

    Finite differences must stay on a smooth limiter branch. At switches this
    is a documented generalized linearization, not a direction-dependent JVP.
    """
    x, y, h, p, theta = _fields(x, y, h, p, theta, speed)
    dh, dp, dt = (np.asarray(v, dtype=np.float64) for v in (dh, dp, dt))
    if any(v.shape != h.shape or not np.all(np.isfinite(v)) for v in (dh, dp, dt)):
        raise ValueError("finite same-shaped directions required")
    rho, _, dr, _ = fluid.properties(p, strength=strength)
    capacity = rho * h * theta
    direction = (dr * dp * h + rho * dh) * theta + rho * h * dt
    _, reconstructed = _reconstruction(x, capacity, direction)
    qx, qy = donor_flux_direction(x, y, h, p, theta, dh, dp, dt, fluid, speed, strength=strength)
    qx += speed * (reconstructed - direction[:, :-1])
    return qx, qy


def _minmod_branch(values, guide):
    """Select the branch reached along values + t*guide as t -> 0+.

    Comparisons are lexicographic in (value, guide). No near-zero tolerance,
    slope snapping or altered flux is introduced. -1 denotes an inactive slope.
    """
    signs = np.where(values != 0, np.sign(values), np.sign(guide))
    positive = np.all(signs > 0, axis=0)
    negative = np.all(signs < 0, axis=0)
    smallest = values == np.min(values, axis=0)
    largest = values == np.max(values, axis=0)
    increase = np.argmin(np.where(smallest, guide, np.inf), axis=0)
    decrease = np.argmax(np.where(largest, guide, -np.inf), axis=0)
    return np.where(positive, increase, np.where(negative, decrease, -1))


def _slope_candidates(x, capacity):
    widths = np.diff(x)
    differences = np.diff(capacity, axis=1) / widths
    backward, forward = differences[:, :-1], differences[:, 1:]
    centered = (widths[1:] * backward + widths[:-1] * forward) / (widths[1:] + widths[:-1])
    interior = np.stack([centered, 2 * backward, 2 * forward])
    b, f = differences[:, 0], differences[:, 1]
    west = ((2 * widths[0] + widths[1]) * b - widths[0] * f) / (widths[0] + widths[1])
    return np.stack([west, 2 * b]), interior


def _apply_branch(candidates, branch):
    selected = np.take_along_axis(candidates, np.maximum(branch, 0)[None], axis=0)[0]
    return np.where(branch >= 0, selected, 0.0)


def freeze_limited_flux_jvp(x, y, h, p, theta, dh, dp, dt, fluid, speed, *, strength=1.0):
    """Freeze a guide-selected flux linearization; return action(dh,dp,dt).

    Base fields are copied. The guide directions have SI units m, Pa and 1
    per unit dimensionless amplitude. Branches are selected along this guide
    once, then held fixed for every action: the returned map is linear and
    suitable for *testing* Krylov products. On its guide it equals the true
    one-sided flux derivative. It is not a complete coupled bordered Jacobian
    or a proof of membership in the global Clarke generalized Jacobian.
    """
    x, y, h, p, theta = [a.copy() for a in _fields(x, y, h, p, theta, speed)]
    rho, _, dr, _ = fluid.properties(p, strength=strength)

    def capacity_direction(dh, dp, dt):
        directions = [np.asarray(a, dtype=np.float64) for a in (dh, dp, dt)]
        if any(a.shape != h.shape or not np.all(np.isfinite(a)) for a in directions):
            raise ValueError("finite same-shaped directions required")
        dh, dp, dt = directions
        return (dr * dp * h + rho * dh) * theta + rho * h * dt

    capacity = rho * h * theta
    guide = capacity_direction(dh, dp, dt)
    base_candidates = _slope_candidates(x, capacity)
    guide_candidates = _slope_candidates(x, guide)
    branches = [
        _minmod_branch(v, g) for v, g in zip(base_candidates, guide_candidates, strict=True)
    ]
    widths = np.diff(x)

    def action(dh, dp, dt):
        direction = capacity_direction(dh, dp, dt)
        candidates = _slope_candidates(x, direction)
        west, interior = [_apply_branch(v, b) for v, b in zip(candidates, branches, strict=True)]
        correction = np.empty_like(direction[:, :-1])
        correction[:, 0] = 0.5 * widths[0] * west
        correction[:, 1:] = 0.5 * widths[1:] * interior
        qx, qy = donor_flux_direction(
            x, y, h, p, theta, dh, dp, dt, fluid, speed, strength=strength
        )
        qx += speed * correction
        return qx, qy

    return action


def limited_flux_directional_derivative(
    x, y, h, p, theta, dh, dp, dt, fluid, speed, *, strength=1.0
):
    """True one-sided flux derivative lim(t->0+) [q(z+t*d)-q(z)]/t.

    This map is positively homogeneous but generally NOT additive in the
    direction at limiter switches. Do not use it as a Krylov LinearOperator.
    No tolerance merges near ties; finite-step differences can cross nearby
    branches before reaching the mathematical derivative at the base field.
    """
    action = freeze_limited_flux_jvp(x, y, h, p, theta, dh, dp, dt, fluid, speed, strength=strength)
    return action(dh, dp, dt)
