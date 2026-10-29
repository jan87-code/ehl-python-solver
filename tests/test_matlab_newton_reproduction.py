"""Independent numerical checks of the isolated MATLAB-equation reproducer."""

import numpy as np
import pytest

from examples.reproduce_matlab_newton import (
    QSTAR,
    ReferenceSystem,
    active_step,
    continuation,
    direct_step,
    line_search,
    lm_step,
    schur_step,
)


@pytest.fixture
def cfg():
    return dict(
        jac_fd_max=2e-5,
        jac_fd_min=1e-8,
        jac_fd_rel=7e-7,
        jac_fb_floor=1e-8,
        jac_regularization=[0, 1e-10, 1e-8, 1e-6, 1e-4],
        schur_den_floor=1e-10,
        load_scale=QSTAR,
        enable_bordered_direct=True,
        as_pressure_tol=1e-8,
        as_min_active_nodes=1,
        merit_load_weight=1,
        lm_mu_candidates=[1e-7, 1e-3],
        lm_max_active_sweeps=8,
        lm_zero_tol=1e-11,
        lm_trust_dp_inf=0.12,
        lm_trust_dH0=0.05,
        max_backtrack=20,
        min_lambda=2**-20,
        armijo_c=1e-4,
        alpha_step_initial=0.0025,
        alpha_step_max=0.0125,
        alpha_step_min=0.000625,
        alpha_step_grow=1.25,
        alpha_step_shrink=0.5,
        alpha_max_attempts=45,
        alpha_targets=[0.9, 1],
        tol_fb_inf=2e-5,
        tol_load_rel=1e-4,
    )


def test_complete_reference_jacobian_matches_joint_direction(cfg):
    system = ReferenceSystem(
        {"pH": 8e8, "lambda": 0.05, "H_scale": 2e-6}, {"rho_scale": 1, "alpha_scale": 0.8}, cfg
    )
    rng = np.random.default_rng(7)
    x = rng.uniform(0.01, 0.05, 961)
    f, g, fields = system.evaluate(x, 2)
    jp, jh, info = system.jacobian(x, 2, fields)
    direction, dh = rng.normal(size=961), 0.3
    epsilon = 1e-6
    fp, gp, _ = system.evaluate(x + epsilon * direction, 2 + epsilon * dh)
    fm, gm, _ = system.evaluate(x - epsilon * direction, 2 - epsilon * dh)
    np.testing.assert_allclose(
        jp @ direction + jh * dh, (fp - fm) / (2 * epsilon), atol=2e-7, rtol=2e-5
    )
    assert system.weights @ direction == pytest.approx((gp - gm) / (2 * epsilon), rel=1e-8)
    assert info["one_sided_columns"] == 0
    # Signed probes may cross zero; ordinary accepted states may not.
    signed = x.copy()
    signed[8] = -1e-7
    system.evaluate(signed, 2, signed_probe=True)
    with pytest.raises(ValueError):
        system.evaluate(signed, 2)


@pytest.mark.parametrize("method", [schur_step, direct_step])
def test_bordered_correction_against_independent_full_solve(method, cfg):
    j = np.array([[3.0, 1.0], [1.0, 2.0]])
    h, f, w = np.array([0.4, -0.7]), np.array([1.0, -2.0]), np.array([0.3, 0.2])
    g = 0.12
    dp, dh, info = method(j, h, f, g, w, cfg)
    border = np.block([[j, h[:, None]], [w[None, :], np.zeros((1, 1))]])
    np.testing.assert_allclose(
        np.append(dp, dh), np.linalg.solve(border, -np.append(f, g)), atol=1e-13
    )
    assert info["pressure_linear_relative"] < 1e-12
    assert info["load_linear_relative"] < 1e-12


def test_direct_border_recovers_singular_pressure_block(cfg):
    j = np.diag([1.0, 0.0])
    h, f, w = np.array([0.0, 1.0]), np.array([2.0, 3.0]), np.array([0.0, 1.0])
    local = {**cfg, "jac_regularization": [0]}
    assert schur_step(j, h, f, 0.1, w, local) is None
    dp, dh, _ = direct_step(j, h, f, 0.1, w, local)
    np.testing.assert_allclose(j @ dp + h * dh + f, 0, atol=1e-14)
    assert w @ dp == pytest.approx(-0.1)


def test_active_elimination_carries_pressure_and_load_contributions(cfg):
    j = np.array([[3.0, 0.2, 0.1], [0.5, 2.0, 0.3], [0.7, 0.1, 3.0]])
    h, f, w = np.array([0.3, 0.4, -0.8]), np.array([0.1, 0.2, -0.4]), np.ones(3)
    x, rs = np.array([1e-9, 0.1, 0.2]), np.ones(3)
    dp, dh, info = active_step(j, h, f, 0.02, w, x, rs, cfg)
    assert dp[0] == -x[0]
    np.testing.assert_allclose((j @ dp + h * dh + f)[1:], 0, atol=1e-13)
    assert w @ dp == pytest.approx(-0.02)
    assert info["active_count"] == 1


def test_bounded_lm_respects_zero_pressure_tangent_and_caps(cfg):
    dp, dh, info = lm_step(
        np.eye(3),
        np.ones(3),
        np.array([2.0, -1.0, 0.5]),
        0.02,
        np.ones(3),
        np.array([0.0, 0.1, 0.0]),
        cfg,
    )
    assert dp[0] >= 0 and dp[2] >= 0
    assert np.max(abs(dp)) <= cfg["lm_trust_dp_inf"]
    assert abs(dh) <= cfg["lm_trust_dH0"]
    assert info["slope"] < 0


def test_reference_linesearch_records_projection_and_invalid_film(cfg):
    class System:
        def __init__(self):
            self.cfg = cfg

        def evaluate(self, x, offset):
            if offset <= 0:
                raise ValueError("strict film floor")
            assert np.min(x) >= 0
            return x, 0.0, {}

    result, trials = line_search(
        System(), np.array([1.0, 0.0]), 1.0, np.array([-2.0, -1.0]), -1.5, 1.0
    )
    assert trials[0]["valid"] is False
    assert trials[1]["valid"] is True
    assert trials[1]["clipped_nodes"] == 1
    np.testing.assert_array_equal(result[0], [0.0, 0.0])
    assert result[1] > 0


def test_failed_continuation_never_reuses_rejected_state(monkeypatch, cfg):
    from examples import reproduce_matlab_newton as module

    system = ReferenceSystem({"H_scale": 1}, {"rho_scale": 1, "alpha_scale": 0.8978125}, cfg)
    monkeypatch.setattr(module, "diagnostics", lambda *a: dict(fb_relative=0.0, load_relative=0.0))
    calls = []

    def fake_solve(trial, x, h):
        calls.append((trial.stage["alpha_scale"], x.copy(), h))
        return x + 7, h + 9, dict(success=False, iterations=20)

    monkeypatch.setattr(module, "solve", fake_solve)
    x = np.array([0.2, 0.3])
    last_x, last_h, report = continuation(system, x, -0.6)
    assert len(calls) == 2
    assert calls[1][0] < calls[0][0]
    for _, used_x, used_h in calls:
        np.testing.assert_array_equal(used_x, x)
        assert used_h == -0.6
    np.testing.assert_array_equal(last_x, x)
    assert last_h == -0.6
    assert report["stop"] == "minimum_alpha_step"
    assert report["last_verified_alpha"] == 0.8978125
    assert not report["target_reached"]


def test_independent_si_direct_flux_audit_matches_reference(tmp_path, cfg):
    from examples.reproduce_matlab_newton import diagnostics, save_state
    from examples.verify_matlab_reproduction import verify_state

    a = (3 * 50 * 0.015 / (4 * 1.13e11)) ** (1 / 3)
    ph, href = 3 * 50 / (2 * np.pi * a * a), a * a / 0.015
    scale = dict(
        a=a, pH=ph, H_scale=href, Q=50, **{"lambda": 12 * 0.04 * 2 * 0.015**2 / (ph * a**3)}
    )
    system = ReferenceSystem(scale, {"rho_scale": 1, "alpha_scale": 0.8978125}, cfg)
    rng = np.random.default_rng(9)
    x = rng.uniform(0.001, 0.02, 961)
    path = tmp_path / "state.npz"
    save_state(path, system, x, 2)
    independently = verify_state(path, alpha_fraction=0.8978125)
    expected = diagnostics(system, x, 2)
    assert independently["fb_relative"] == pytest.approx(expected["fb_relative"], rel=1e-12)
    assert independently["load_relative_error"] == pytest.approx(
        expected["load_relative"], abs=2e-15
    )
    assert independently["stored_film_max_error_m"] < 1e-18
    assert independently["stored_displacement_max_error_m"] < 1e-18
    assert independently["divergence_theorem_error_m3_per_s"] < 1e-18
    assert not independently["passes_reference_gates"]
    assert not independently["mass_conserving_jfo_validation"]
    with pytest.raises(ValueError, match="alpha_fraction"):
        verify_state(path, alpha_fraction=float("nan"))
