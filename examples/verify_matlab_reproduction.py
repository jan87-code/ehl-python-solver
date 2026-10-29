"""Independent SI direct-elasticity/face-flux audit of reference NPZ states.

No call to the reproduction residual, FFT or Newton routines. Array order is
(ny,nx), +x entrainment. This tests pressure-only reference acceptance, not JFO.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import convolve2d


def verify_state(path, *, alpha_fraction, load_n=50.0):
    if not np.isfinite(alpha_fraction) or not 0 <= alpha_fraction <= 1:
        raise ValueError("alpha_fraction must be finite and in [0,1]")
    if not np.isfinite(load_n) or load_n <= 0:
        raise ValueError("load_n must be positive and finite")
    with np.load(path, allow_pickle=False) as data:
        p, x, y = data["pressure_pa"], data["x_m"], data["y_m"]
        offset = float(data["offset_m"])
        film_stored, displacement_stored = data["film_m"], data["displacement_m"]
    if p.shape != (33, 33) or film_stored.shape != p.shape or displacement_stored.shape != p.shape:
        raise ValueError("N33 reference requires (ny,nx)=(33,33) fields")
    if not all(np.all(np.isfinite(v)) for v in (p, x, y, offset, film_stored, displacement_stored)):
        raise ValueError("Nonfinite saved state")
    radius, modulus, speed, eta0, alpha_p, p0 = 0.015, 1.13e11, 2.0, 0.04, 2.21e-8, 1.98e8
    a = (3 * load_n * radius / (4 * modulus)) ** (1 / 3)
    ph, href = 3 * load_n / (2 * np.pi * a * a), a * a / radius
    if (
        x.shape != (33,)
        or y.shape != (33,)
        or not np.allclose(x, np.linspace(-4, 4, 33) * a, rtol=1e-12, atol=1e-18)
        or not np.allclose(y, x, rtol=1e-12, atol=1e-18)
    ):
        raise ValueError("Saved coordinates do not match the reference domain/scaling")
    dx, dy = x[1] - x[0], y[1] - y[0]
    lx, ly = np.meshgrid(np.arange(-32, 33) * dx, np.arange(-32, 33) * dy)
    r = np.hypot(lx, ly)
    r[32, 32] = 1
    # SI sampled half-space influence; integrate its singular source cell only.
    kernel = dx * dy / (np.pi * modulus * r)
    kernel[32, 32] = (2 * dx * np.arcsinh(dy / dx) + 2 * dy * np.arcsinh(dx / dy)) / (
        np.pi * modulus
    )
    displacement = convolve2d(p, kernel, mode="same")
    xx, yy = np.meshgrid(x, y)
    film = offset + (xx * xx + yy * yy) / (2 * radius) + displacement
    if np.min(film) <= 0:
        raise ValueError("Independent film reconstruction is not positive")
    pressure_for_fluid = np.maximum(p, 0)
    density = 1 + 0.6e-9 * pressure_for_fluid / (1 + 1.7e-9 * pressure_for_fluid)
    z = alpha_fraction * alpha_p * p0 / (np.log(eta0) + 9.67)
    viscosity = eta0 * np.exp(
        np.minimum((np.log(eta0) + 9.67) * ((1 + pressure_for_fluid / p0) ** z - 1), 700)
    )
    mobility = density * film**3 / (12 * viscosity)
    face_x = 2 * mobility[:, :-1] * mobility[:, 1:] / (mobility[:, :-1] + mobility[:, 1:])
    face_y = 2 * mobility[:-1, :] * mobility[1:, :] / (mobility[:-1, :] + mobility[1:, :])
    qx = -face_x * np.diff(p, axis=1) / dx + speed * (density * film)[:, :-1]
    qy = -face_y * np.diff(p, axis=0) / dy
    divergence = np.diff(qx, axis=1)[1:-1, :] / dx + np.diff(qy, axis=0)[:, 1:-1] / dy
    diagonal = (face_x[1:-1, 1:] + face_x[1:-1, :-1]) / dx**2 + (
        face_y[1:, 1:-1] + face_y[:-1, 1:-1]
    ) / dy**2
    # Original gamma=1/max(diagA,eps) expressed in SI, including its floor.
    nd_to_si_diagonal = href**3 / (12 * eta0 * a * a)
    diagonal = np.maximum(diagonal, np.finfo(float).eps * nd_to_si_diagonal)
    scaled = divergence / diagonal / ph
    pi = p[1:-1, 1:-1] / ph
    fb = np.hypot(pi, scaled) - pi - scaled
    fb_relative = float(np.max(abs(fb)) / max(1, np.max(pi)))
    load = float(trapezoid(trapezoid(p, x=x, axis=1), x=y))
    load_error = load / load_n - 1
    boundary = np.concatenate(
        (qx[1:-1, -1] * dy, -qx[1:-1, 0] * dy, qy[-1, 1:-1] * dx, -qy[0, 1:-1] * dx)
    )
    net, absolute = float(np.sum(boundary)), float(np.sum(abs(boundary)))
    film_error = float(np.max(abs(film - film_stored)))
    displacement_error = float(np.max(abs(displacement - displacement_stored)))
    boundary_zero = bool(not np.any(p[[0, -1], :]) and not np.any(p[:, [0, -1]]))
    gates = (
        fb_relative <= 2e-5
        and abs(load_error) <= 1e-4
        and np.min(p) >= 0
        and np.min(film) > 1e-10 * href
        and boundary_zero
        and film_error < 1e-17
        and displacement_error < 1e-17
    )
    return {
        "alpha_fraction": alpha_fraction,
        "load_n": load,
        "load_relative_error": float(load_error),
        "fb_relative": fb_relative,
        "passes_reference_gates": bool(gates),
        "film_min_m": float(np.min(film)),
        "pressure_min_pa": float(np.min(p)),
        "boundary_pressure_zero": boundary_zero,
        "stored_film_max_error_m": film_error,
        "stored_displacement_max_error_m": displacement_error,
        "density_normalized_net_boundary_mass_flux_m3_per_s": net,
        "net_boundary_mass_flux_relative": abs(net) / max(absolute, np.finfo(float).tiny),
        "divergence_theorem_error_m3_per_s": float(abs(np.sum(divergence) * dx * dy - net)),
        "mass_conserving_jfo_validation": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", type=Path)
    parser.add_argument("--alpha", type=float, required=True)
    args = parser.parse_args()
    print(
        json.dumps(verify_state(args.state, alpha_fraction=args.alpha), indent=2, allow_nan=False)
    )
