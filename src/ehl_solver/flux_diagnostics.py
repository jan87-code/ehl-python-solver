"""Independent SI flux decomposition and nested-domain force accounting."""

import numpy as np
from scipy.integrate import trapezoid


def face_components(x_m, y_m, film_m, pressure_pa, theta, eta_pa_s, speed_m_s, *, dtype=np.float64):
    """Return Couette qx, Poiseuille qx/qy [m^2/s], original divergence [m/s].

    All interfaces are SI and arrays are (ny,nx), entrainment +x. A wider
    dtype is an explicit diagnostic arithmetic choice, never a state cast
    made silently by a solver. Face film/mobility are arithmetic averages.
    """
    x, y, h, p, content = [
        np.asarray(a, dtype=dtype) for a in (x_m, y_m, film_m, pressure_pa, theta)
    ]
    if (
        x.ndim != 1
        or y.ndim != 1
        or min(len(x), len(y)) < 3
        or np.any(np.diff(x) <= 0)
        or np.any(np.diff(y) <= 0)
        or any(a.shape != (len(y), len(x)) for a in (h, p, content))
        or any(not np.all(np.isfinite(a)) for a in (x, y, h, p, content))
        or np.any(h <= 0)
        or not np.isfinite(eta_pa_s)
        or eta_pa_s <= 0
        or not np.isfinite(speed_m_s)
        or speed_m_s < 0
    ):
        raise ValueError("invalid SI face-flux inputs")
    eta, speed = dtype(eta_pa_s), dtype(speed_m_s)
    couette = speed * (h[:, 1:] + h[:, :-1]) / 2 * content[:, :-1]
    px = -(h[:, 1:] ** 3 + h[:, :-1] ** 3) / 2 * np.diff(p, axis=1) / (12 * eta * np.diff(x))
    py = -(h[1:] ** 3 + h[:-1] ** 3) / 2 * np.diff(p, axis=0) / (12 * eta * np.diff(y)[:, None])
    qx = couette + px
    wx, wy = (x[2:] - x[:-2]) / 2, (y[2:] - y[:-2]) / 2
    residual = np.diff(qx[1:-1], axis=1) / wx + np.diff(py[:, 1:-1], axis=0) / wy[:, None]
    return couette, px, py, residual


def decompose_flux(
    x_m,
    y_m,
    film_m,
    pressure_pa,
    theta,
    eta_pa_s,
    speed_m_s,
    *,
    length_scale_m,
    film_scale_m,
    pressure_scale_pa,
):
    """JSON-safe flow/cancellation/roundoff diagnostics, retaining SI units.

    Inlet components are positive-inward. Boundary budgets are outward in
    west/east/south/north order. Cancellation condition = sum(abs(parts)) /
    abs(net), so one means no cancellation; exact nonzero opposing parts
    with zero net have condition None plus an explicit zero-net count.
    """
    if not all(np.isfinite(v) and v > 0 for v in (length_scale_m, film_scale_m, pressure_scale_pa)):
        raise ValueError("diagnostic SI scales must be positive finite")
    args = (x_m, y_m, film_m, pressure_pa, theta, eta_pa_s, speed_m_s)
    c, px, py, residual = face_components(*args)
    wc, wpx, wpy, wide = face_components(*args, dtype=np.longdouble)
    x, y = np.asarray(x_m), np.asarray(y_m)
    wx, wy = np.diff(x)[1:] / 2 + np.diff(x)[:-1] / 2, np.diff(y)[1:] / 2 + np.diff(y)[:-1] / 2
    qx = c + px
    inlet_c = float(c[1:-1, 0] @ wy)
    inlet_p = float(px[1:-1, 0] @ wy)
    inlet = inlet_c + inlet_p
    parts = np.abs(c) + np.abs(px)
    nonzero = np.abs(qx) > 0
    zero_net_opposing = int(np.sum((~nonzero) & (parts > 0)))
    factor = np.divide(parts, abs(qx), out=np.ones_like(parts), where=nonzero)
    j, i = np.unravel_index(np.argmax(abs(wide)), wide.shape)
    q_ref = film_scale_m**3 * pressure_scale_pa / (12 * eta_pa_s * length_scale_m)
    term = (abs(qx[1:-1, 1:]) + abs(qx[1:-1, :-1])) / wx + (
        abs(py[1:, 1:-1]) + abs(py[:-1, 1:-1])
    ) / wy[:, None]
    outward = [-inlet, float(qx[1:-1, -1] @ wy), -float(py[0, 1:-1] @ wx), float(py[-1, 1:-1] @ wx)]
    # Perturb the stored SI pressure by one ULP at the largest-defect node.
    # This is sensitivity, not a proof of the attainable global residual.
    changed = np.array(pressure_pa, copy=True)
    changed[j + 1, i + 1] = np.nextafter(changed[j + 1, i + 1], np.inf)
    perturbed = face_components(
        x_m, y_m, film_m, changed, theta, eta_pa_s, speed_m_s, dtype=np.longdouble
    )[-1]
    return {
        "inlet_couette_m3_s": inlet_c,
        "inlet_poiseuille_m3_s": inlet_p,
        "inlet_net_m3_s": inlet,
        "inlet_cancellation_condition": (abs(inlet_c) + abs(inlet_p)) / abs(inlet)
        if inlet
        else None,
        "maximum_face_cancellation_condition": None if zero_net_opposing else float(factor.max()),
        "nonzero_opposing_parts_with_zero_net": zero_net_opposing,
        "maximum_couette_face_m2_s": float(abs(c).max()),
        "maximum_poiseuille_x_face_m2_s": float(abs(px).max()),
        "maximum_poiseuille_y_face_m2_s": float(abs(py).max()),
        "outward_volume_flow_m3_s": outward,
        "net_volume_flow_m3_s": float(sum(outward)),
        "mass_maximum_float64": float(abs(residual).max() * length_scale_m / q_ref),
        "mass_maximum_wider": float(abs(wide).max() * length_scale_m / q_ref),
        "max_flux_divergence_evaluation_difference": float(
            abs(residual - wide).max() * length_scale_m / q_ref
        ),
        "maximum_flux_evaluation_difference_m2_s": float(
            max(abs(c + px - wc - wpx).max(), abs(py - wpy).max())
        ),
        "eps_face_balance_scale": float(np.finfo(float).eps * term.max() * length_scale_m / q_ref),
        "maximum_defect_xy_m": [float(x[i + 1]), float(y[j + 1])],
        "one_pressure_ulp_defect_change": float(
            abs(perturbed - wide).max() * length_scale_m / q_ref
        ),
        "mobility_dynamic_range": float((np.max(film_m) / np.min(film_m)) ** 3),
        "state_mantissa_bits": np.finfo(np.float64).nmant,
        "audit_mantissa_bits": np.finfo(np.longdouble).nmant,
    }


def nested_load_budget(old_x_m, old_y_m, old_pressure_pa, new_x_m, new_y_m, new_pressure_pa):
    """Exact discrete Q_new-Q_old = added-area load + shared-area readjustment.

    Both pressure integrals use SI trapezoidal quadrature. Old axes must be
    nested in the new ones, with equal physical coordinates. No interpolation
    or inference about continuum truncation error is performed.
    """

    def indices(old, new):
        old, new = np.asarray(old), np.asarray(new)
        indices = np.argmin(abs(new[:, None] - old), axis=0)
        if not np.allclose(new[indices], old, atol=1e-18, rtol=1e-13):
            raise ValueError("force budget requires nested physical coordinates")
        if np.any(np.diff(indices) != 1):
            raise ValueError("force budget requires unchanged cells within the shared domain")
        return indices

    ix, iy = indices(old_x_m, new_x_m), indices(old_y_m, new_y_m)

    def load(p, x, y):
        return float(trapezoid(trapezoid(p, x, axis=1), y))

    old = load(old_pressure_pa, old_x_m, old_y_m)
    new = load(new_pressure_pa, new_x_m, new_y_m)
    shared = load(np.asarray(new_pressure_pa)[np.ix_(iy, ix)], old_x_m, old_y_m)
    return {
        "old_load_n": old,
        "new_load_n": new,
        "change_n": new - old,
        "added_area_load_n": new - shared,
        "shared_area_readjustment_n": shared - old,
        "added_area_fraction_of_change": (new - shared) / (new - old) if new != old else None,
        "identity_defect_n": (new - old) - ((new - shared) + (shared - old)),
    }
