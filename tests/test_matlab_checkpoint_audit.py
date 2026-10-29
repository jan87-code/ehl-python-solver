"""Reference diagnostic checks; no dependency on the user's private MAT file."""

import numpy as np
import pytest
from scipy.signal import convolve2d

from examples.audit_matlab_checkpoint import reference_fields


def test_reference_fft_is_linear_with_asymmetric_pressure():
    p = np.zeros((33, 33))
    p[2, 28], p[19, 7] = 0.3, 1.2
    fields = reference_fields(
        p, 2, {"pH": 8e8, "lambda": 0.05}, {"rho_scale": 1, "alpha_scale": 0.89}
    )
    direct = convolve2d(p, fields["kernel"], mode="same") * fields["dx"] ** 2
    np.testing.assert_allclose(fields["deformation"], direct, atol=2e-16, rtol=2e-14)
    # Separate source-cell self coefficient from the FFT/direct indexing check.
    expected_self = 8 / np.pi**2 * fields["dx"] * np.arcsinh(1)
    p.fill(0)
    p[16, 16] = 1
    impulse = reference_fields(
        p, 2, {"pH": 8e8, "lambda": 0.05}, {"rho_scale": 0, "alpha_scale": 0}
    )
    assert impulse["deformation"][16, 16] == pytest.approx(expected_self)


def test_reference_reynolds_is_flux_divergence():
    rng = np.random.default_rng(3)
    p = np.zeros((33, 33))
    p[1:-1, 1:-1] = rng.uniform(0, 0.5, (31, 31))
    fields = reference_fields(p, 2, {"pH": 8e8, "lambda": 0.05}, {"rho_scale": 0, "alpha_scale": 0})
    h, dx = fields["film"], fields["dx"]
    k = h**3
    xface = 2 * k[:, 1:] * k[:, :-1] / (k[:, 1:] + k[:, :-1])
    yface = 2 * k[1:, :] * k[:-1, :] / (k[1:, :] + k[:-1, :])
    qx = -xface * np.diff(p, axis=1) / dx + 0.05 * h[:, :-1]
    qy = -yface * np.diff(p, axis=0) / dx
    divergence = (np.diff(qx, axis=1)[1:-1, :] + np.diff(qy, axis=0)[:, 1:-1]) / dx
    np.testing.assert_allclose(fields["residual"], divergence, rtol=1e-13, atol=1e-10)


@pytest.mark.parametrize("failure", ["boundary", "negative", "film"])
def test_reference_rejects_invalid_states(failure):
    p = np.zeros((33, 33))
    if failure == "boundary":
        p[0, 2] = 1
    if failure == "negative":
        p[16, 16] = -1
    with pytest.raises(ValueError):
        reference_fields(
            p,
            -1 if failure == "film" else 2,
            {"pH": 8e8, "lambda": 0.05},
            {"rho_scale": 0, "alpha_scale": 0},
        )
