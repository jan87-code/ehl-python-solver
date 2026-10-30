import numpy as np
import pytest

from ehl_solver.constitutive import IsothermalFluid
from ehl_solver.limited_transport import limited_flux_direction, limited_fluxes
from ehl_solver.transport_candidate import donor_flux_direction, face_divergence
from examples.benchmark_limited_transport import extrema_errors, smooth_errors


def test_scalar_nonuniform_faces_bounded_capacity_and_conservation():
    rng = np.random.default_rng(712)
    x, y = np.array([0.0, 0.05, 0.15, 0.4, 0.5, 1.0]) * 1e-3, np.array([0.0, 0.1, 0.3, 0.8]) * 1e-3
    h = 1e-6 * (1 + rng.random((4, 6)))
    p, t = 1e8 * rng.random(h.shape), rng.random(h.shape)
    fluid = IsothermalFluid(0.04, 850)
    qx, qy = limited_fluxes(x, y, h, p, t, fluid, 2.0)
    rho, eta, _, _ = fluid.properties(p)
    capacity = rho * h * t
    for j in range(4):
        for i in range(5):
            forward = (capacity[j, i + 1] - capacity[j, i]) / (x[i + 1] - x[i])
            if i == 0:
                next_slope = (capacity[j, 2] - capacity[j, 1]) / (x[2] - x[1])
                w0, w1 = x[1] - x[0], x[2] - x[1]
                raw = ((2 * w0 + w1) * forward - w0 * next_slope) / (w0 + w1)
                slope = np.sign(raw) * min(abs(raw), 2 * abs(forward)) if raw * forward > 0 else 0
            else:
                backward = (capacity[j, i] - capacity[j, i - 1]) / (x[i] - x[i - 1])
                left, right = x[i] - x[i - 1], x[i + 1] - x[i]
                center = (right * backward + left * forward) / (left + right)
                slope = (
                    np.sign(forward) * min(abs(center), 2 * abs(backward), 2 * abs(forward))
                    if backward * forward > 0
                    else 0
                )
            face = capacity[j, i] + 0.5 * (x[i + 1] - x[i]) * slope
            assert (
                min(capacity[j, i : i + 2]) - 1e-18 <= face <= max(capacity[j, i : i + 2]) + 1e-18
            )
            expected = (
                -(
                    rho[j, i] * h[j, i] ** 3 / eta[j, i]
                    + rho[j, i + 1] * h[j, i + 1] ** 3 / eta[j, i + 1]
                )
                / 24
            )
            expected *= (p[j, i + 1] - p[j, i]) / (x[i + 1] - x[i])
            assert qx[j, i] == pytest.approx(expected + 2 * face, rel=3e-14, abs=1e-18)
    wx, wy = 0.5 * (x[2:] - x[:-2]), 0.5 * (y[2:] - y[:-2])
    integrated = np.sum(face_divergence(x, y, qx, qy) * wx * wy[:, None])
    boundary = (qx[1:-1, -1] - qx[1:-1, 0]) @ wy + (qy[-1, 1:-1] - qy[0, 1:-1]) @ wx
    assert integrated == pytest.approx(boundary, rel=1e-13, abs=1e-18)


@pytest.mark.parametrize("strength", [0.0, 1.0])
def test_analytic_flux_jvp_includes_material_film_and_content(strength):
    rng = np.random.default_rng(943)
    x, y = np.linspace(0, 1e-3, 13), np.linspace(0, 5e-4, 7)
    p = 1e8 * (1 + 0.1 * rng.random((7, 13)))
    h, t = 1e-6 * (1 + 0.2 * rng.random(p.shape)), 0.6 + 0.2 * rng.random(p.shape)
    dh, dp, dt = [scale * rng.normal(size=p.shape) for scale in (1e-8, 1e6, 0.01)]
    fluid = IsothermalFluid(0.04, 850)
    before = [a.copy() for a in (h, p, t)]
    analytic = limited_flux_direction(x, y, h, p, t, dh, dp, dt, fluid, 2.0, strength=strength)
    step = 1e-5
    plus = limited_fluxes(
        x, y, h + step * dh, p + step * dp, t + step * dt, fluid, 2.0, strength=strength
    )
    minus = limited_fluxes(
        x, y, h - step * dh, p - step * dp, t - step * dt, fluid, 2.0, strength=strength
    )
    for a, b, c in zip(analytic, plus, minus, strict=True):
        assert np.linalg.norm((b - c) / (2 * step) - a) / np.linalg.norm(a) < 2e-7
    for a, b in zip(before, (h, p, t), strict=True):
        np.testing.assert_array_equal(a, b)


def test_selected_generalized_derivative_is_linear_at_plateaus_and_nyquist_survives():
    rng = np.random.default_rng(23)
    x, y = np.arange(33) * 1e-5, np.arange(5) * 1e-5
    p, h, t = np.full((5, 33), 1e7), np.full((5, 33), 1e-6), np.ones((5, 33))
    fluid = IsothermalFluid(0.04, 850)
    a, b = rng.normal(size=p.shape) * 0.01, rng.normal(size=p.shape) * 0.01
    zero = np.zeros_like(p)

    def action(dt):
        return limited_flux_direction(x, y, h, p, t, zero, zero, dt, fluid, 2.0)

    for combined, one, two in zip(action(a + b), action(a), action(b), strict=True):
        np.testing.assert_allclose(combined, one + two, rtol=1e-13, atol=1e-18)
    dt = np.broadcast_to(0.01 * (-1.0) ** np.arange(33), p.shape)
    limited = action(dt)
    donor = donor_flux_direction(x, y, h, p, t, zero, zero, dt, fluid, 2.0)
    for a, b in zip(limited, donor, strict=True):
        np.testing.assert_array_equal(a, b)
    assert np.linalg.norm(face_divergence(x, y, *limited)) > 0


@pytest.mark.parametrize("downstream", [0.0, 0.2])
def test_sharp_front_capacity_is_bounded_without_modifying_fields(downstream):
    x, y = np.linspace(0, 1e-3, 65), np.linspace(0, 1e-4, 5)
    p, h = np.zeros((5, 65)), np.full((5, 65), 1e-6)
    t = np.broadcast_to(np.where(x < 0.5e-3, 1.0, downstream), p.shape).copy()
    old = t.copy()
    qx, qy = limited_fluxes(x, y, h, p, t, IsothermalFluid(0.04, 850), 2.0)
    assert qx.min() >= 2 * 850e-6 * downstream
    assert qx.max() <= 2 * 850e-6
    np.testing.assert_array_equal(qy, 0)
    np.testing.assert_array_equal(t, old)


def test_exact_constant_transported_capacity_is_stationary_for_variable_film():
    x, y = np.linspace(0, 1e-3, 65), np.linspace(0, 1e-4, 5)
    p = np.zeros((5, 65))
    h = np.broadcast_to(1e-6 * (1 + 0.2 * np.sin(2 * np.pi * x / 1e-3)), p.shape)
    theta = 0.7e-6 / h
    q = limited_fluxes(x, y, h, p, theta, IsothermalFluid(0.04, 850), 2.0)
    assert np.max(abs(face_divergence(x, y, *q))) < 5e-14


@pytest.mark.parametrize("varying_content", [False, True])
def test_monotone_manufactured_source_is_second_order_including_west_column(varying_content):
    errors = [smooth_errors(n, varying_content) for n in (33, 65, 129)]
    for old, new in zip(errors[:-1], errors[1:], strict=True):
        assert 0.95 < np.log2(old["donor"] / new["donor"]) < 1.05
        assert 1.9 < np.log2(old["limited"] / new["limited"]) < 2.1
    assert errors[-1]["limited"] < errors[-1]["donor"] / 100


def test_invalid_directions_and_physical_fields_raise_without_clamping():
    x, y, p = np.arange(5.0), np.arange(3.0), np.zeros((3, 5))
    fluid = IsothermalFluid(0.04, 850)
    with pytest.raises(ValueError):
        limited_fluxes(x, y, p, p, p + 1, fluid, 2.0)
    with pytest.raises(ValueError):
        limited_flux_direction(x, y, p + 1, p, p + 1, p, np.zeros(2), p, fluid, 2.0)


def test_limiter_extrema_errors_are_recorded_without_assuming_second_order():
    rows = [extrema_errors(n) for n in (33, 65, 129)]
    assert all(r["limited"] < r["donor"] for r in rows)
    assert rows[-1]["limited"] < rows[0]["limited"]
    # Record convergence without imposing a second-order expectation at switches.
    assert np.isfinite(np.log2(rows[1]["limited"] / rows[2]["limited"]))
