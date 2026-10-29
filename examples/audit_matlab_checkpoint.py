"""Independent evaluation of the supplied N33 MATLAB pressure-only checkpoint.

This diagnostic transcribes its discrete operator, not its Newton algorithm.
Never imports/runs MATLAB code or promotes the checkpoint to a Python JFO root.
Arrays are (ny,nx), +x entrainment; reported physical values use SI units.
"""

import argparse
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import convolve2d

from ehl_solver.elasticity import ElasticityOperator


@lru_cache(maxsize=1)
def _elastic_grid():
    n, dx = 33, 0.25
    lag = np.arange(-n + 1, n) * dx
    xx, yy = np.meshgrid(lag, lag)
    radius = np.hypot(xx, yy)
    radius[n - 1, n - 1] = 1
    kernel = (2 / np.pi**2) / radius
    kernel[n - 1, n - 1] = (2 / np.pi**2) * 4 * dx * np.arcsinh(1) / dx**2
    spectrum = np.fft.fft2(np.fft.ifftshift(kernel))
    kernel.flags.writeable = spectrum.flags.writeable = False
    return kernel, spectrum


def reference_fields(pressure, offset, scale, stage, *, _signed_probe=False):
    """Evaluate the reference's dimensionless N33 equations, including scaling."""
    p = np.asarray(pressure, dtype=np.float64)
    if p.shape != (33, 33) or not np.all(np.isfinite(p)):
        raise ValueError("Reference requires finite (33,33) pressure")
    if not np.isfinite(offset) or (not _signed_probe and np.any(p < 0)):
        raise ValueError("Invalid reference pressure or offset")
    if np.any(p[[0, -1], :]) or np.any(p[:, [0, -1]]):
        raise ValueError("Reference requires zero boundary pressure")
    n = 33
    dx = 8 / (n - 1)
    # Square-cell integral of 1/r; all other reference cells use point samples.
    kernel, spectrum = _elastic_grid()
    padded = np.zeros((2 * n - 1, 2 * n - 1))
    padded[:n, :n] = p * dx**2
    deformation = np.fft.ifft2(np.fft.fft2(padded) * spectrum).real[:n, :n]
    coord = np.linspace(-4, 4, n)
    x, y = np.meshgrid(coord, coord)
    h = offset + (x * x + y * y) / 2 + deformation
    if np.min(h) <= 1e-10:
        raise ValueError("Reference film fails strict positive-film floor")
    # MATLAB permits signed FD probes: elasticity uses signed P, fluid uses max(P,0).
    physical_p = np.maximum(p, 0) * scale["pH"]
    rho = 1 + stage["rho_scale"] * 0.6e-9 * physical_p / (1 + 1.7e-9 * physical_p)
    z = stage["alpha_scale"] * 2.21e-8 * 1.98e8 / (np.log(0.04) + 9.67)
    eta = np.exp(np.minimum((np.log(0.04) + 9.67) * ((1 + physical_p / 1.98e8) ** z - 1), 700))
    k = rho * h**3 / eta
    kx = 2 * k[:, :-1] * k[:, 1:] / np.maximum(k[:, :-1] + k[:, 1:], np.finfo(float).tiny)
    ky = 2 * k[:-1, :] * k[1:, :] / np.maximum(k[:-1, :] + k[1:, :], np.finfo(float).tiny)
    e, w = kx[1:-1, 1:] / dx**2, kx[1:-1, :-1] / dx**2
    north, south = ky[1:, 1:-1] / dx**2, ky[:-1, 1:-1] / dx**2
    diagonal = e + w + north + south
    pi = p[1:-1, 1:-1]
    residual = (
        diagonal * pi
        - e * p[1:-1, 2:]
        - w * p[1:-1, :-2]
        - north * p[2:, 1:-1]
        - south * p[:-2, 1:-1]
        + scale["lambda"] * (rho * h)[1:-1, 1:-1] / dx
        - scale["lambda"] * (rho * h)[1:-1, :-2] / dx
    )
    scaled = residual / np.maximum(diagonal, np.finfo(float).eps)
    fb = np.hypot(pi, scaled) - pi - scaled
    return {
        "deformation": deformation,
        "film": h,
        "kernel": kernel,
        "residual": residual,
        "scaled_residual": scaled,
        "fb": fb,
        "dx": dx,
    }


def audit(path):
    data = loadmat(path, simplify_cells=True)
    p, scale, stage = data["P"], data["scale"], data["stage"]
    # These are the supplied source's constants, not inferred from arbitrary MAT files.
    a = (3 * scale["Q"] * 0.015 / (4 * 1.13e11)) ** (1 / 3)
    ph = 3 * scale["Q"] / (2 * np.pi * a * a)
    expected = {
        "a": a,
        "pH": ph,
        "H_scale": a * a / 0.015,
        "lambda": 12 * 0.04 * 2 * 0.015**2 / (ph * a**3),
    }
    if data["cfg"]["N"] != 33 or not all(
        np.isclose(scale[key], value, rtol=1e-10, atol=0) for key, value in expected.items()
    ):
        raise ValueError("Checkpoint scaling does not match the supplied N33 reference")
    fields = reference_fields(p, data["H0"], scale, stage)
    target = reference_fields(p, data["H0"], scale, {**stage, "alpha_scale": 1.0})
    fb_rel = float(np.max(np.abs(fields["fb"])) / max(1, np.max(p)))
    load_rel = float(np.sum(p) * fields["dx"] ** 2 / (2 * np.pi / 3) - 1)
    # Compare the FFT indexing to an independent full linear direct convolution.
    direct = convolve2d(p, fields["kernel"], mode="same") * fields["dx"] ** 2
    # A deliberate operator comparison, not a Python nonlinear solution.
    displacement = ElasticityOperator(
        p.shape, fields["dx"] * scale["a"], fields["dx"] * scale["a"], 1.13e11
    )(p * scale["pH"])
    python_film = (fields["film"] - fields["deformation"]) * scale["H_scale"] + displacement
    return {
        "checkpoint_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "shape_ny_nx": list(p.shape),
        "load_n": float(scale["Q"]),
        "viscosity_alpha_fraction": float(stage["alpha_scale"]),
        "fb_relative": fb_rel,
        "load_relative_error": load_rel,
        "same_state_full_viscosity_fb_relative": float(
            np.max(np.abs(target["fb"])) / max(1, np.max(p))
        ),
        "passes_saved_matlab_gates": bool(
            fb_rel <= data["cfg"]["tol_fb_inf"] and abs(load_rel) <= data["cfg"]["tol_load_rel"]
        ),
        "pressure_max_pa": float(np.max(p) * scale["pH"]),
        "reference_film_min_m": float(np.min(fields["film"]) * scale["H_scale"]),
        "reference_film_centre_m": float(fields["film"][16, 16] * scale["H_scale"]),
        "fft_direct_max_error_dimensionless": float(np.max(abs(direct - fields["deformation"]))),
        "integrated_kernel_displacement_relative_l2_change": float(
            np.linalg.norm(displacement / scale["H_scale"] - fields["deformation"])
            / np.linalg.norm(fields["deformation"])
        ),
        "integrated_kernel_same_offset_film_min_m": float(np.min(python_film)),
        "python_jfo_solution": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.checkpoint), indent=2, allow_nan=False))
