"""Independent SI limited-capacity flux/film/force reconstruction, CPU float64."""

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import convolve2d

from ehl_solver.elasticity import cell_integrated_kernel
from ehl_solver.property_reynolds import admissible


def limited_mass_audit(problem, fluid, speed, h, p, theta, *, transport_fraction=1.0):
    """No call to the candidate flux, reconstruction or JVP implementation."""
    if (
        isinstance(transport_fraction, bool)
        or not np.isfinite(transport_fraction)
        or not 0 <= transport_fraction <= 1
    ):
        raise ValueError("transport_fraction must be finite in [0,1]")
    x, y = problem.x_m, problem.y_m
    rho, eta, _, _ = fluid.properties(p)
    capacity = rho * h * theta
    slopes = np.diff(capacity, axis=1) / np.diff(x)
    b, f = slopes[:, :-1], slopes[:, 1:]
    center = (np.diff(x)[1:] * b + np.diff(x)[:-1] * f) / (x[2:] - x[:-2])
    slope = np.where(
        ((b > 0) & (f > 0)) | ((b < 0) & (f < 0)),
        np.sign(b) * np.minimum(np.minimum(2 * abs(b), abs(center)), 2 * abs(f)),
        0.0,
    )
    adv = capacity[:, :-1].copy()
    adv[:, 1:] += 0.5 * np.diff(x)[1:] * slope
    w0, w1 = x[1] - x[0], x[2] - x[1]
    first = ((2 * w0 + w1) * slopes[:, 0] - w0 * slopes[:, 1]) / (w0 + w1)
    first = np.where(
        ((first > 0) & (slopes[:, 0] > 0)) | ((first < 0) & (slopes[:, 0] < 0)),
        np.sign(first) * np.minimum(abs(first), 2 * abs(slopes[:, 0])),
        0.0,
    )
    adv[:, 0] += 0.5 * w0 * first
    adv = capacity[:, :-1] + transport_fraction * (adv - capacity[:, :-1])
    m = rho * h**3 / (12 * eta)
    qx = -0.5 * (m[:, 1:] + m[:, :-1]) * np.diff(p, axis=1) / np.diff(x) + speed * adv
    qy = -0.5 * (m[1:] + m[:-1]) * np.diff(p, axis=0) / np.diff(y)[:, None]
    wx, wy = 0.5 * (x[2:] - x[:-2]), 0.5 * (y[2:] - y[:-2])
    residual = np.diff(qx[1:-1], axis=1) / wx + np.diff(qy[:, 1:-1], axis=0) / wy[:, None]
    defect = residual * wx * wy[:, None]
    throughput = (abs(qx[1:-1, 1:]) + abs(qx[1:-1, :-1])) * wy[:, None] + (
        abs(qy[1:, 1:-1]) + abs(qy[:-1, 1:-1])
    ) * wx
    outward = np.array([-qx[1:-1, 0] @ wy, qx[1:-1, -1] @ wy, -qy[0, 1:-1] @ wx, qy[-1, 1:-1] @ wx])
    qref = (
        fluid.density_kg_m3
        * problem.film_scale_m**3
        * problem.pressure_scale_pa
        / (12 * fluid.viscosity_pa_s * problem.length_scale_m)
    )
    return dict(
        mass_maximum=float(abs(residual).max() * problem.length_scale_m / qref),
        maximum_si_mass_divergence_kg_m2_s=float(abs(residual).max()),
        relative_local_balance=float(np.max(abs(defect) / np.maximum(throughput, 1e-300))),
        relative_global_balance=float(abs(outward.sum()) / max(abs(outward).sum(), 1e-300)),
        outward_mass_flow_kg_s=outward.tolist(),
        net_mass_flow_kg_s=float(outward.sum()),
        summed_cell_mass_defect_kg_s=float(defect.sum()),
        maximum_complementarity=float(np.max(abs(p / problem.pressure_scale_pa * (1 - theta)))),
    )


def audit_limited_state(probe, state, *, direct=False):
    problem, case = probe.system.problem, probe.case
    p, theta, h = probe.fields(state)
    error = 0.0
    if direct:
        kernel = cell_integrated_kernel(
            problem.shape, problem.dx_m, problem.dy_m, case.contact_modulus_pa
        )
        reconstructed = (
            state[-1] * problem.film_scale_m
            + problem.rigid_geometry_m
            + convolve2d(p, kernel, mode="same")
        )
        error = float(abs(reconstructed - h).max())
        h = reconstructed
    mass = limited_mass_audit(
        problem,
        case.fluid(),
        case.entrainment_m_s,
        h,
        p,
        theta,
        transport_fraction=probe.transport_fraction,
    )
    force = float(trapezoid(trapezoid(p, x=problem.x_m, axis=1), x=problem.y_m))
    boundaries = np.r_[p[0], p[-1], p[:, 0], p[:, -1]]
    accepted = bool(
        admissible(mass, p, theta, problem.pressure_scale_pa)
        and abs(force / case.load_n - 1) <= 1e-8
        and h.min() > problem.film_floor_m
        and np.all(boundaries == 0)
        and np.all(theta[1:-1, 0] == problem.inlet_fractional_film_content)
        and np.array_equal(theta[1:-1, -1], theta[1:-1, -2])
        and np.array_equal(theta[0], theta[1])
        and np.array_equal(theta[-1], theta[-2])
        and error <= 1e-10 * problem.film_scale_m
    )
    return dict(
        numerically_accepted=accepted,
        mass=mass,
        force_n=force,
        force_relative_error=force / case.load_n - 1,
        minimum_film_m=float(h.min()),
        central_film_m=float(h[case.ny // 2, case.nx // 2]),
        film_reconstruction_error_m=error,
        direct_elasticity_checked=direct,
        transport_fraction=probe.transport_fraction,
        full_limited_equations=probe.transport_fraction == 1.0,
    )
