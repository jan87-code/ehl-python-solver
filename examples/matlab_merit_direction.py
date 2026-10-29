"""Exact one-sided merit slope for the isolated N33 reference projected path.

Dimensionless internal variables follow reproduce_matlab_newton: P=p/pH,
H=h/(a²/R), MATLAB F interior order. The diagnostic retains pressure projection;
this module does not change the production JFO solver or its convergence gates.
"""

import numpy as np

from examples.audit_matlab_checkpoint import _elastic_grid
from examples.reproduce_matlab_newton import QSTAR


def merit_direction(system, x, offset, dp, dh):
    """Return d/dlambda merit(max(x+lambda*dp,0),h0+lambda*dh) at 0+."""
    f, g, fields = system.evaluate(x, offset)
    direction = np.where(x == 0, np.maximum(dp, 0), dp)
    p, v = np.zeros((33, 33)), np.zeros((33, 33))
    p[1:-1, 1:-1], v[1:-1, 1:-1] = (
        x.reshape((31, 31), order="F"),
        direction.reshape((31, 31), order="F"),
    )
    padded = np.zeros((65, 65))
    padded[:33, :33] = v * 0.25**2
    _kernel, spectrum = _elastic_grid()
    delta_h = dh + np.fft.ifft2(np.fft.fft2(padded) * spectrum).real[:33, :33]
    h = fields["film"]
    pp, vp = p * system.scale["pH"], v * system.scale["pH"]
    rho = 1 + system.stage["rho_scale"] * 0.6e-9 * pp / (1 + 1.7e-9 * pp)
    drho = system.stage["rho_scale"] * 0.6e-9 * vp / (1 + 1.7e-9 * pp) ** 2
    a = np.log(0.04) + 9.67
    z = system.stage["alpha_scale"] * 2.21e-8 * 1.98e8 / a
    log_eta = a * ((1 + pp / 1.98e8) ** z - 1)
    dlog_eta = a * z / 1.98e8 * (1 + pp / 1.98e8) ** (z - 1) * vp
    # Reproduce the source's min(log_eta,700), including its one-sided kink.
    dlog_eta = np.where(
        log_eta < 700, dlog_eta, np.where(log_eta > 700, 0, np.minimum(dlog_eta, 0))
    )
    eta = np.exp(np.minimum(log_eta, 700))
    k = rho * h**3 / eta
    dk = drho * h**3 / eta + 3 * rho * h * h * delta_h / eta - k * dlog_eta

    def harmonic_pair(lo, hi, dlo, dhi):
        denominator = lo + hi
        if np.any(denominator <= np.finfo(float).tiny):
            raise ValueError("Directional audit does not support underflowed face mobility")
        value = 2 * lo * hi / denominator
        derivative = 2 * ((hi / denominator) ** 2 * dlo + (lo / denominator) ** 2 * dhi)
        return value, derivative

    kx, dkx = harmonic_pair(k[:, :-1], k[:, 1:], dk[:, :-1], dk[:, 1:])
    ky, dky = harmonic_pair(k[:-1, :], k[1:, :], dk[:-1, :], dk[1:, :])
    dx = 0.25
    mass_derivative = drho * h + rho * delta_h
    dqx = (
        -dkx * np.diff(p, axis=1) / dx
        - kx * np.diff(v, axis=1) / dx
        + system.scale["lambda"] * mass_derivative[:, :-1]
    )
    dqy = -dky * np.diff(p, axis=0) / dx - ky * np.diff(v, axis=0) / dx
    dr = (np.diff(dqx, axis=1)[1:-1, :] + np.diff(dqy, axis=0)[:, 1:-1]) / dx
    diagonal = (kx[1:-1, 1:] + kx[1:-1, :-1] + ky[1:, 1:-1] + ky[:-1, 1:-1]) / dx**2
    ddiagonal = (dkx[1:-1, 1:] + dkx[1:-1, :-1] + dky[1:, 1:-1] + dky[:-1, 1:-1]) / dx**2
    ddiagonal = np.where(
        diagonal > np.finfo(float).eps,
        ddiagonal,
        np.where(diagonal < np.finfo(float).eps, 0, np.maximum(ddiagonal, 0)),
    )
    rs = fields["scaled_residual"]
    drs = (dr - rs * ddiagonal) / np.maximum(diagonal, np.finfo(float).eps)
    rs, drs = rs.ravel(order="F"), drs.ravel(order="F")
    norm = np.hypot(x, rs)
    safe_norm = np.where(norm > 0, norm, 1)
    df = (x * direction + rs * drs) / safe_norm - direction - drs
    df = np.where(norm > 0, df, np.hypot(direction, drs) - direction - drs)
    slope = (
        np.mean(f * df)
        + system.cfg["merit_load_weight"] * g * (system.weights @ direction) / QSTAR**2
    )
    if not np.isfinite(slope) or not np.all(np.isfinite(df)):
        raise ValueError("Nonfinite projected merit derivative")
    return float(slope), df, direction
