"""Optional reduced field LU: complete Newton gates and explicit recovery."""

import numpy as np
import pytest
from scipy.integrate import trapezoid

from ehl_solver.cavitating_newton import CavitatingLinearError, solve_cavitating_correction
from ehl_solver.constitutive import IsothermalFluid
from ehl_solver.property_coupled import solve_property_coupling
from ehl_solver.property_load_newton import (
    PropertyLoadNewtonError,
    PropertyLoadSystem,
    continue_property_load,
    solve_property_load_newton,
)
from ehl_solver.property_reynolds import admissible, mass_audit
from ehl_solver.property_transfer import PropertyLoadGuess, repair_property_load_guess
from examples.run_property_load_verification import make_problem


@pytest.fixture(scope="module")
def case():
    problem, fluid = make_problem(7, 9), IsothermalFluid(.04, 850.)
    return problem, fluid, solve_property_coupling(problem, fluid, .1)


def system_state(case, **kwargs):
    p, f, seed = case
    system = PropertyLoadSystem(p, f, .1, seed.load_n*1.05, **kwargs)
    return system, system.pack(seed, p.central_film_offset_m)


def original_gates(problem, fluid, root, target):
    s = root.solution
    audit = mass_audit(problem.x_m, problem.y_m, s.film_thickness_m, s.pressure_pa,
                       s.fractional_film_content, fluid, .1,
                       length_scale_m=problem.length_scale_m, film_scale_m=problem.film_scale_m,
                       pressure_scale_pa=problem.pressure_scale_pa)
    assert admissible(audit, s.pressure_pa, s.fractional_film_content, problem.pressure_scale_pa)
    force = trapezoid(trapezoid(s.pressure_pa, problem.x_m, axis=1), problem.y_m)
    assert abs(force/target-1) <= 1e-8
    h, w = problem.fields(s.pressure_pa, central_film_offset_m=root.central_film_offset_m)
    np.testing.assert_allclose(s.film_thickness_m, h, atol=1e-18, rtol=0)
    np.testing.assert_allclose(s.elastic_displacement_m, w, atol=1e-18, rtol=0)
    assert h.min() > problem.film_floor_m
    for row in s.history:
        selected = [(m, t) for m in row["models"] for t in m["trials"] if t["accepted"]]
        assert len(selected) == 1
        m, t = selected[0]
        assert m["linear_true_residual"] <= 5e-9 and m["slope"] < 0
        assert t["merit"] <= row["merit"]+1e-4*t["alpha"]*m["slope"]


@pytest.mark.parametrize("preconditioner", ["frozen", "material", "local"])
@pytest.mark.parametrize("elasticity", ["reference", "cached"])
def test_complete_jvp_dense_schur_full_recovery_and_strict_film(case, preconditioner, elasticity):
    system, state = system_state(case, preconditioner=preconditioner, elasticity_mode=elasticity,
                                 field_factorization="reduced")
    original = state.copy()
    smooth = state.copy()
    smooth[:system.n], smooth[system.n:2*system.n] = .001, .02
    v = np.random.default_rng(349).normal(size=system.size)*.001
    step = 1e-5
    fd = (system.residual(smooth+step*v)-system.residual(smooth-step*v))/(2*step)
    np.testing.assert_allclose(system.jacobian_vector(smooth, v), fd, rtol=2e-6, atol=1e-9)
    for view in (system, system.correction_view(state, ~system.full(state))):
        dense = np.column_stack([view.jacobian_vector(state, e) for e in np.eye(system.size)])
        expected = np.linalg.solve(dense, -view.residual(state))
        for tolerance in (1e-10, 2.):
            correction = solve_cavitating_correction(view, state, schur_tolerance=tolerance)
            assert correction.used_bordered_fallback == (tolerance == 2.)
            assert correction.true_linear_residual <= 5e-9
            assert np.linalg.norm(correction.direction-expected)/np.linalg.norm(expected) < 1e-7
            assert not correction.field_setup_failures
    np.testing.assert_array_equal(state, original)
    view = system.correction_view(state, system.full(state))
    assert view.field_factorization == "reduced"
    bad = state.copy()
    bad[-1] = -1/system.problem.film_scale_m
    with pytest.raises(ValueError, match="film"):
        system.residual(bad)


def test_reuses_one_assembled_matrix_and_default_does_not_request_reduction(case, monkeypatch):
    for mode in ("full", "reduced"):
        system, state = system_state(case, field_factorization=mode)
        assemble, reduced = system.frozen_matrix, system.reduced_field_factor
        matrices, used = [], []

        def matrix(z, assemble=assemble, matrices=matrices):
            matrices.append(assemble(z))
            return matrices[-1]

        def factor(z, m, mode=mode, matrices=matrices, used=used, reduced=reduced):
            assert mode == "reduced"
            assert m is matrices[0]
            used.append(m)
            return reduced(z, m)

        monkeypatch.setattr(system, "frozen_matrix", matrix)
        monkeypatch.setattr(system, "reduced_field_factor", factor)
        assert solve_cavitating_correction(system, state).true_linear_residual <= 5e-9
        assert len(matrices) == 1
        assert len(used) == (mode == "reduced")
    with pytest.raises(ValueError, match="field_factorization"):
        system_state(case, field_factorization="multigrid")


@pytest.mark.parametrize("preconditioner", ["frozen", "material", "local"])
def test_nonlinear_root_matches_full_and_original_gates(case, preconditioner):
    p, f, seed = case
    roots = []
    for mode in ("full", "reduced"):
        root = solve_property_load_newton(p, f, .1, seed.load_n*1.05, seed=seed,
                    seed_offset_m=p.central_film_offset_m, preconditioner=preconditioner,
                    elasticity_mode="cached", field_factorization=mode)
        original_gates(p, f, root, seed.load_n*1.05)
        assert all(m["field_factorization"] == mode for row in root.solution.history
                   for m in row["models"])
        roots.append(root)
    a, b = roots
    assert np.linalg.norm(a.solution.pressure_pa-b.solution.pressure_pa)/np.linalg.norm(
        a.solution.pressure_pa) < 1e-7
    assert abs(a.central_film_offset_m-b.central_film_offset_m) < 1e-12*p.film_scale_m


def test_reduced_setup_failure_recovers_only_through_explicit_full_border(case, monkeypatch):
    import ehl_solver.cavitating_newton as linear

    system, state = system_state(case, field_factorization="reduced")
    original, shapes = linear.splu, []

    def fail(*args):
        raise RuntimeError("injected reduced factor failure")

    def full(matrix):
        shapes.append(matrix.shape)
        return original(matrix)

    monkeypatch.setattr(system, "reduced_field_factor", fail)
    monkeypatch.setattr(linear, "splu", full)
    correction = solve_cavitating_correction(system, state)
    assert correction.used_bordered_fallback and correction.true_linear_residual <= 5e-9
    assert shapes == [(system.size, system.size)]
    assert correction.field_setup_failures == ({"factorization": "reduced",
                                               "reason": "injected reduced factor failure"},)


def test_total_factor_failure_and_false_krylov_success_preserve_diagnostics(case, monkeypatch):
    import ehl_solver.cavitating_newton as linear

    system, state = system_state(case, field_factorization="reduced")

    def fail(*args):
        raise RuntimeError("injected factor failure")

    with monkeypatch.context() as m:
        m.setattr(system, "reduced_field_factor", fail)
        m.setattr(linear, "splu", fail)
        with pytest.raises(CavitatingLinearError, match="full bordered") as caught:
            solve_cavitating_correction(system, state)
        assert caught.value.diagnostics["field_setup_failures"]
        assert caught.value.diagnostics["field_factorization"] == "reduced"

    def false_success(op, rhs, **kwargs):
        kwargs["callback"](1.)
        return np.zeros_like(rhs), 0

    monkeypatch.setattr(linear, "gmres", false_success)
    with pytest.raises(CavitatingLinearError, match="true bordered") as caught:
        solve_cavitating_correction(system, state)
    assert caught.value.diagnostics["true_linear_residual"] == pytest.approx(1.)
    assert len(caught.value.diagnostics["krylov_histories"]) == 2


def test_block_krylov_failure_recovers_and_retains_failed_true_residual(case, monkeypatch):
    import ehl_solver.cavitating_newton as linear

    system, state = system_state(case, field_factorization="reduced")
    original, calls = linear.gmres, []

    def solve(op, rhs, **kwargs):
        calls.append(op.shape)
        if len(calls) == 1:
            kwargs["callback"](1.)
            return np.zeros_like(rhs), 1
        return original(op, rhs, **kwargs)

    monkeypatch.setattr(linear, "gmres", solve)
    result = solve_cavitating_correction(system, state)
    assert result.used_bordered_fallback and result.true_linear_residual <= 5e-9
    assert result.linear_true_residuals[0] == 1.
    assert calls == [(2*system.n, 2*system.n), (system.size, system.size)]


def test_continuation_retry_preserves_factorization_and_original_seed(case, monkeypatch):
    import ehl_solver.property_load_newton as module

    p, f, seed = case
    original, calls = module.solve_property_load_newton, []

    def solve(*args, **kwargs):
        calls.append((kwargs["seed_offset_m"], kwargs["field_factorization"],
                      kwargs["seed"].pressure_pa.copy()))
        if len(calls) == 1:
            raise PropertyLoadNewtonError("injected retry", (), None)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "solve_property_load_newton", solve)
    result = continue_property_load(p, f, .1, 1.05*seed.load_n, seed=seed,
                seed_offset_m=p.central_film_offset_m, initial_increment_n=.05*seed.load_n,
                elasticity_mode="cached", field_factorization="reduced")
    assert calls[0][:2] == calls[1][:2] == (p.central_film_offset_m, "reduced")
    np.testing.assert_array_equal(calls[0][2], calls[1][2])
    assert not result.attempts[0]["accepted"] and result.attempts[-1]["accepted"]
    original_gates(p, f, result.root, 1.05*seed.load_n)


def test_active_lookahead_rejected_trials_and_failed_selected_backup(case, monkeypatch):
    import ehl_solver.property_load_newton as module

    p, f, seed = case
    guess = PropertyLoadGuess(seed.pressure_pa, np.ones_like(seed.fractional_film_content),
                              p.central_film_offset_m, {"accepted": False})
    root = repair_property_load_guess(p, f, .1, seed.load_n, guess,
                                      elasticity_mode="cached", field_factorization="reduced")
    original_gates(p, f, root, seed.load_n)
    assert any(m["recovery"] > 0 for row in root.solution.history for m in row["models"])
    assert any(not t["accepted"] for row in root.solution.history
               for m in row["models"] for t in m["trials"])
    assert any(t["alpha"] < 1 for row in root.solution.history
               for m in row["models"] for t in m["trials"])
    original = module.solve_cavitating_correction

    def fail_selected(system, *args, **kwargs):
        assert system.field_factorization == "reduced"
        if system.linear_mode:
            raise CavitatingLinearError("injected selected failure", {"injected": True})
        return original(system, *args, **kwargs)

    monkeypatch.setattr(module, "solve_cavitating_correction", fail_selected)
    with pytest.raises(PropertyLoadNewtonError, match="budget") as caught:
        repair_property_load_guess(p, f, .1, seed.load_n, guess, maximum_iterations=1,
                            elasticity_mode="cached", field_factorization="reduced")
    row = caught.value.history[0]
    assert row["active_recovery_failures"][0]["diagnostics"]["injected"]
    accepted = [(m, t) for m in row["models"] for t in m["trials"] if t["accepted"]]
    assert len(accepted) == 1 and "fallback" in accepted[0][1]["reason"]
    model, trial = accepted[0]
    assert trial["alpha"] < .1
    assert trial["merit"] <= row["merit"]+1e-4*trial["alpha"]*model["slope"]
    assert caught.value.last_root.solution.film_thickness_m.min() > p.film_floor_m


def test_nonlinear_krylov_failure_keeps_last_root_and_zero_budget_does_not_accept(
        case, monkeypatch):
    import ehl_solver.cavitating_newton as linear

    p, f, seed = case

    def fail(op, rhs, **kwargs):
        kwargs["callback"](1.)
        return np.zeros_like(rhs), 1

    monkeypatch.setattr(linear, "gmres", fail)
    with pytest.raises(PropertyLoadNewtonError) as caught:
        solve_property_load_newton(p, f, .1, 1.05*seed.load_n, seed=seed,
              seed_offset_m=p.central_film_offset_m, field_factorization="reduced")
    assert caught.value.last_root is not None and caught.value.diagnostics["krylov_histories"]
    assert caught.value.diagnostics["field_factorization"] == "reduced"
    guess = PropertyLoadGuess(seed.pressure_pa, seed.fractional_film_content,
                              p.central_film_offset_m, {"accepted": False})
    with pytest.raises(PropertyLoadNewtonError, match="budget"):
        repair_property_load_guess(p, f, .1, 1.05*seed.load_n, guess, maximum_iterations=0,
                                    field_factorization="reduced")
