from types import SimpleNamespace

import numpy as np
import pytest

from ehl_solver.cavitating_newton import CavitatingLinearError, solve_cavitating_correction
from ehl_solver.limited_system import LimitedLoadLinearization, LimitedLoadProbe
from ehl_solver.limited_transport import freeze_limited_flux_jvp
from ehl_solver.prototype import PointContactCase


def state_and_guide(*, load=200.0, content=0.65, ny=9, nx=13):
    case = PointContactCase(load_n=load, ny=ny, nx=nx)
    probe = LimitedLoadProbe(case)
    rng = np.random.default_rng(523)
    p = np.zeros((ny, nx))
    theta = np.ones_like(p)
    p[1:-1, 1:-1] = (
        0.02 * probe.system.problem.pressure_scale_pa * (1 + 0.1 * rng.random((ny - 2, nx - 2)))
    )
    theta[1:-1, 1:-1] = content
    z = probe.system.pack(SimpleNamespace(pressure_pa=p, fractional_film_content=theta), 2e-6)
    guide = rng.normal(size=probe.size) * 1e-3
    return case, z, guide


@pytest.mark.parametrize("mode", ["material", "local"])
def test_sparse_preconditioner_matches_independent_frozen_film_or_local_flux_action(mode):
    case, z, guide = state_and_guide()
    model = LimitedLoadLinearization(case, z, guide, preconditioner=mode)
    probe, problem = model.probe, model.problem
    p, theta, h = probe.fields(z)
    dhg, dpg, dtg = probe.directions(z, guide)
    flux_action = freeze_limited_flux_jvp(
        problem.x_m, problem.y_m, h, p, theta, dhg, dpg, dtg, case.fluid(), case.entrainment_m_s
    )
    v = np.random.default_rng(52).normal(size=model.size) * 1e-3
    _, dp, dt = probe.directions(z, v)
    compliance = problem.elastic_kernel[case.ny - 1, case.nx - 1] if mode == "local" else 0.0
    dh = compliance * dp + v[-1] * problem.film_scale_m
    expected = model.jacobian_vector(z, v)
    expected[: model.n] = probe.mass(flux_action(dh, dp, dt))
    matrix = model.frozen_matrix(z)
    measured = matrix @ v
    assert (
        np.linalg.norm((measured - expected)[: model.n]) / np.linalg.norm(expected[: model.n])
        < 1e-12
    )
    np.testing.assert_allclose(measured[model.n :], expected[model.n :], rtol=1e-13, atol=1e-15)
    # Sparse field preconditioning intentionally omits nonlocal elasticity.
    true = model.jacobian_vector(z, v)
    assert np.linalg.norm((true - measured)[: model.n]) > 1e-10
    assert matrix.nnz < 40 * model.size
    e = np.zeros(model.size)
    e[-1] = 1
    np.testing.assert_allclose(matrix @ e, model.jacobian_vector(z, e), rtol=1e-14, atol=1e-15)


def test_frozen_model_owns_base_and_rejects_another_state_or_invalid_direction():
    case, z, g = state_and_guide()
    model = LimitedLoadLinearization(case, z, g)
    before = model.jacobian_vector(z, g)
    matrix = model.frozen_matrix(z)
    matrix.data[:] = 0
    assert model.frozen_matrix(z).nnz > 0
    with pytest.raises(ValueError):
        model.guide[0] = 0
    with pytest.raises(ValueError):
        model.state[0] = 0
    original = z.copy()
    z[-1] += 0.1
    g[:] = 0
    np.testing.assert_array_equal(model.jacobian_vector(original, model.guide), before)
    with pytest.raises(ValueError, match="state is frozen"):
        model.jacobian_vector(z, model.guide)
    with pytest.raises(ValueError, match="direction"):
        model.jacobian_vector(original, np.zeros(3))
    with pytest.raises(ValueError):
        LimitedLoadLinearization(case, original, np.full(model.size, np.nan))


def test_complete_correction_true_residual_and_schur_reconstruction():
    case, z, g = state_and_guide(load=1.0, content=1.0, ny=5, nx=7)
    model = LimitedLoadLinearization(case, z, g, preconditioner="material")
    correction = solve_cavitating_correction(model, z, maximum_krylov_cycles=10, gmres_restart=60)
    residual = model.residual(z)
    check = model.jacobian_vector(z, correction.direction) + residual
    assert np.linalg.norm(check) / np.linalg.norm(residual) <= 5e-9
    assert correction.true_linear_residual <= 5e-9
    assert not correction.used_bordered_fallback
    # Independent tiny-grid dense solve is a verification oracle only.
    identity = np.eye(model.size)
    dense = np.column_stack([model.jacobian_vector(z, v) for v in identity])
    direct = np.linalg.solve(dense, -residual)
    np.testing.assert_allclose(correction.direction, direct, rtol=1e-6, atol=1e-9)
    expected_slope = residual @ model.probe.derivative(z, correction.direction) / model.size
    assert model.merit_slope(z, residual, correction.direction) == expected_slope


def test_forced_full_bordered_recovery_retains_true_linear_checks():
    case, z, g = state_and_guide(load=1.0, content=1.0, ny=5, nx=7)
    model = LimitedLoadLinearization(case, z, g, preconditioner="local")
    result = solve_cavitating_correction(
        model, z, maximum_krylov_cycles=10, gmres_restart=60, schur_tolerance=2.0
    )
    assert result.used_bordered_fallback
    assert result.true_linear_residual <= 5e-9
    assert all(v <= 5e-9 for v in result.linear_true_residuals)
    assert result.krylov_histories


def test_singular_preconditioner_failure_preserves_diagnostics(monkeypatch):
    case, z, g = state_and_guide(load=1.0, content=1.0, ny=5, nx=7)
    model = LimitedLoadLinearization(case, z, g)
    matrix = model.frozen_matrix(z)
    matrix.data[:] = 0
    monkeypatch.setattr(model, "frozen_matrix", lambda state: matrix)
    with pytest.raises(CavitatingLinearError) as exc:
        solve_cavitating_correction(model, z, maximum_krylov_cycles=10, gmres_restart=60)
    assert exc.value.diagnostics["field_setup_failures"]
    assert "linear_true_residuals" in exc.value.diagnostics


def test_exhausted_krylov_retains_independently_measured_failed_residuals():
    case, z, g = state_and_guide(load=1.0, content=1.0, ny=9, nx=13)
    model = LimitedLoadLinearization(case, z, g)
    with pytest.raises(CavitatingLinearError) as exc:
        solve_cavitating_correction(model, z, maximum_krylov_cycles=1, gmres_restart=1)
    data = exc.value.diagnostics
    assert data["linear_true_residuals"]
    assert data["linear_true_residuals"][-1] > 5e-9
    assert data["krylov_histories"]
    assert data["linear_iterations"] > 0
