import numpy as np
import pytest

from ehl_solver.constitutive import IsothermalFluid
from ehl_solver.limited_transport import (
    _apply_branch,
    _minmod_branch,
    freeze_limited_flux_jvp,
    limited_flux_directional_derivative,
    limited_fluxes,
)
from ehl_solver.transport_candidate import face_divergence


@pytest.mark.parametrize(
    "base,guide,expected",
    [
        ([1, 2, 3], [0.4, -0.2, 1], 0.4),
        ([-1, -2, -3], [0.4, -0.2, 1], 0.4),
        ([1, 1, 3], [0.4, -0.2, 1], -0.2),
        ([-1, -1, -3], [0.4, -0.2, 1], 0.4),
        ([0, 1, 2], [0.3, -1, 1], 0.3),
        ([0, 1, 2], [-0.3, -1, 1], 0),
        ([0, 0, 2], [0.3, 0.2, 1], 0.2),
        ([0, 0, 2], [0.3, -0.2, 1], 0),
        ([0, -1, -2], [-0.3, 1, 1], -0.3),
        ([0, -1, -2], [0.3, 1, 1], 0),
        ([0, 0, 0], [1, 2, 3], 1),
        ([0, 0, 0], [-1, -2, -3], -1),
        ([0, 0, 0], [-1, 1, 0], 0),
        ([0, 0, 0], [0, 1, 2], 0),
        ([-1, 0, 1], [1, 2, 3], 0),
        ([1e-100, 2e-100, 3e-100], [-1, 0.3, 1], -1),
    ],
)
def test_minmod_one_sided_corner_table_without_near_tie_snapping(base, guide, expected):
    v, g = np.array(base, dtype=float)[:, None], np.array(guide, dtype=float)[:, None]
    result = _apply_branch(g, _minmod_branch(v, g))
    assert result[0] == pytest.approx(expected)


def fields():
    x, y = np.linspace(0, 1e-3, 13), np.linspace(0, 5e-4, 7)
    h, p, theta = (np.full((7, 13), value) for value in (1e-6, 1e8, 0.75))
    return x, y, h, p, theta, IsothermalFluid(0.04, 850)


@pytest.mark.parametrize("strength", [0.0, 1.0])
def test_exact_plateau_flux_derivative_matches_global_one_sided_differences(strength):
    x, y, h, p, t, fluid = fields()
    rng = np.random.default_rng(9421)
    dh, dp, dt = [rng.normal(size=h.shape) * s for s in (1e-8, 1e6, 0.01)]
    analytic = limited_flux_directional_derivative(
        x, y, h, p, t, dh, dp, dt, fluid, 2.0, strength=strength
    )
    base = limited_fluxes(x, y, h, p, t, fluid, 2.0, strength=strength)
    errors = []
    for step in (1e-2, 1e-3, 1e-4):
        plus = limited_fluxes(
            x, y, h + step * dh, p + step * dp, t + step * dt, fluid, 2.0, strength=strength
        )
        twice = limited_fluxes(
            x,
            y,
            h + 2 * step * dh,
            p + 2 * step * dp,
            t + 2 * step * dt,
            fluid,
            2.0,
            strength=strength,
        )
        error = []
        for a, b, c, d in zip(analytic, base, plus, twice, strict=True):
            fd = (4 * (c - b) - (d - b)) / (2 * step)
            error.append(np.linalg.norm(fd - a) / np.linalg.norm(a))
        errors.append(max(error))
    assert min(errors) < 2e-7


def test_frozen_guide_map_is_additive_and_agrees_with_true_derivative_on_guide():
    x, y, h, p, t, fluid = fields()
    rng = np.random.default_rng(234)
    a = [rng.normal(size=h.shape) * s for s in (1e-8, 1e6, 0.01)]
    b = [rng.normal(size=h.shape) * s for s in (1e-8, 1e6, 0.01)]
    action = freeze_limited_flux_jvp(x, y, h, p, t, *a, fluid, 2.0)
    for actual, expected in zip(
        action(*a), limited_flux_directional_derivative(x, y, h, p, t, *a, fluid, 2.0), strict=True
    ):
        np.testing.assert_array_equal(actual, expected)
    for total, one, two in zip(
        action(*[u + v for u, v in zip(a, b, strict=True)]), action(*a), action(*b), strict=True
    ):
        np.testing.assert_allclose(total, one + two, rtol=2e-12, atol=1e-18)
    saved = action(*a)
    h[:] *= 1.1
    p[:] *= 1.1
    t[:] *= 0.9
    for old, new in zip(saved, action(*a), strict=True):
        np.testing.assert_array_equal(old, new)


def test_true_directional_map_is_homogeneous_but_not_additive_at_switches():
    x, y, h, p, t, fluid = fields()
    zero = np.zeros_like(h)
    rng = np.random.default_rng(341)
    a, b = rng.normal(size=h.shape) * 0.01, rng.normal(size=h.shape) * 0.01

    def derivative(dt):
        return limited_flux_directional_derivative(x, y, h, p, t, zero, zero, dt, fluid, 2.0)[0]

    np.testing.assert_allclose(derivative(3 * a), 3 * derivative(a), rtol=1e-12, atol=1e-18)
    assert np.linalg.norm(derivative(a + b) - derivative(a) - derivative(b)) > 1e-6


def test_directional_merit_slope_and_armijo_on_exact_plateau_switches():
    x, y, h, p, t, fluid = fields()
    zero = np.zeros_like(h)
    direction = np.broadcast_to(np.linspace(-0.03, 0.03, len(x)), h.shape).copy()

    def residual(theta):
        # Prescribed-film mass residual with explicit manufactured source.
        q = limited_fluxes(x, y, h, p, theta, fluid, 2.0, strength=0.0)
        return face_divergence(x, y, *q).ravel() - 0.1

    base = residual(t)
    dq = limited_flux_directional_derivative(
        x, y, h, p, t, zero, zero, direction, fluid, 2.0, strength=0.0
    )
    dr = face_divergence(x, y, *dq).ravel()
    slope = base @ dr
    assert slope < 0
    merit = 0.5 * (base @ base)
    step = 1e-3
    fd = (
        0.5 * np.dot(residual(t + step * direction), residual(t + step * direction)) - merit
    ) / step
    assert fd == pytest.approx(slope, rel=0.01)
    assert 0.5 * np.dot(residual(t + step * direction), residual(t + step * direction)) <= (
        merit + 1e-4 * step * slope
    )
    opposite = -direction
    dq = limited_flux_directional_derivative(
        x, y, h, p, t, zero, zero, opposite, fluid, 2.0, strength=0.0
    )
    assert base @ face_divergence(x, y, *dq).ravel() > 0


def test_invalid_freeze_guide_or_action_direction_is_rejected():
    x, y, h, p, t, fluid = fields()
    zero = np.zeros_like(h)
    with pytest.raises(ValueError):
        freeze_limited_flux_jvp(x, y, h, p, t, zero, np.zeros(2), zero, fluid, 2.0)
    action = freeze_limited_flux_jvp(x, y, h, p, t, zero, zero, zero, fluid, 2.0)
    bad = zero.copy()
    bad[1, 1] = np.nan
    with pytest.raises(ValueError):
        action(zero, zero, bad)


def test_full_residual_probe_preserves_original_pressure_tolerance_and_strict_film():
    from ehl_solver.prototype import PointContactCase, hertz_guess
    from examples.benchmark_limiter_switches import ResidualProbe

    case = PointContactCase(ny=9, nx=13)
    probe = ResidualProbe(case)
    state, _ = hertz_guess(case, probe.system)
    state[0] = -1e-17  # Accepted roundoff-scale negative P; do not clamp it.
    assert probe.feasible(state)
    p, _, _ = probe.fields(state)
    assert p[1, 1] < 0
    state[0] = -2e-10
    assert not probe.feasible(state)
    state[0] = 0
    _, _, film = probe.fields(state)
    state[-1] -= 2 * film.min() / probe.system.problem.film_scale_m
    assert not probe.feasible(state)


def test_complete_probe_differentiates_copied_outlet_and_transverse_content():
    from ehl_solver.prototype import PointContactCase, hertz_guess
    from examples.benchmark_limiter_switches import ResidualProbe

    probe = ResidualProbe(PointContactCase(ny=9, nx=13))
    state, _ = hertz_guess(probe.case, probe.system)
    direction = np.zeros(probe.size)
    direction[probe.n : 2 * probe.n] = np.random.default_rng(71).normal(size=probe.n) * 1e-3
    _, _, dt = probe.directions(state, direction)
    step = 1e-3
    _, plus, _ = probe.fields(state + step * direction)
    _, minus, _ = probe.fields(state - step * direction)
    np.testing.assert_allclose((plus - minus) / (2 * step), dt, rtol=1e-7, atol=1e-12)
    np.testing.assert_array_equal(dt[1:-1, -1], dt[1:-1, -2])
    np.testing.assert_array_equal(dt[0], dt[1])
    np.testing.assert_array_equal(dt[-1], dt[-2])


def test_complete_smooth_residual_frozen_product_has_independent_finite_difference_check():
    from examples.benchmark_limiter_switches import smooth_complete_probe

    result = smooth_complete_probe()
    assert result["full_residual_difference_check_passed"]
