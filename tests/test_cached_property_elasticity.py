"""Reference/cached CPU parity with unchanged bordered and physical gates."""

import numpy as np
import pytest

from ehl_solver.cavitating_newton import solve_cavitating_correction
from ehl_solver.constitutive import IsothermalFluid
from ehl_solver.property_coupled import solve_property_coupling
from ehl_solver.property_load_newton import (
    PropertyLoadNewtonError,
    PropertyLoadSystem,
    continue_property_load,
    solve_property_load_newton,
)
from ehl_solver.property_reynolds import admissible
from ehl_solver.property_transfer import PropertyLoadGuess, repair_property_load_guess
from examples.run_property_load_verification import make_problem


@pytest.fixture(scope="module")
def case():
    p, f = make_problem(7, 9), IsothermalFluid(.04, 850.)
    return p, f, solve_property_coupling(p, f, .1)


def test_cached_complete_bordered_action_dense_correction_and_strict_film(case):
    p, f, seed = case
    ref = PropertyLoadSystem(p, f, .1, seed.load_n*1.05)
    fast = PropertyLoadSystem(p, f, .1, seed.load_n*1.05, elasticity_mode="cached")
    z = ref.pack(seed, p.central_film_offset_m)
    rng = np.random.default_rng(764)
    v = .001*rng.normal(size=ref.size)
    smooth = z.copy()
    smooth[:ref.n] = .001
    smooth[ref.n:2*ref.n] = .02
    step = 1e-5
    fd = (fast.residual(smooth+step*v)-fast.residual(smooth-step*v))/(2*step)
    np.testing.assert_allclose(fast.jacobian_vector(smooth, v), fd, rtol=2e-6, atol=1e-9)
    matrix = np.column_stack([ref.jacobian_vector(z, e) for e in np.eye(ref.size)])
    cached = np.column_stack([fast.jacobian_vector(z, e) for e in np.eye(ref.size)])
    np.testing.assert_allclose(cached, matrix, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(fast.residual(z), ref.residual(z), rtol=0, atol=1e-13)
    expected = np.linalg.solve(matrix, -ref.residual(z))
    for tolerance in (1e-10, 2.):
        correction = solve_cavitating_correction(fast, z, schur_tolerance=tolerance)
        assert np.linalg.norm(correction.direction-expected)/np.linalg.norm(expected) < 1e-7
        assert correction.true_linear_residual <= 5e-9
        assert correction.used_bordered_fallback == (tolerance == 2.)
    view = fast.correction_view(z, ~fast.full(z))
    assert view.base.elastic_operator is fast.base.elastic_operator
    rejected = z.copy()
    rejected[-1] = -1/p.film_scale_m
    with pytest.raises(ValueError, match="film"):
        fast.residual(rejected)
    with pytest.raises(ValueError, match="elasticity_mode"):
        PropertyLoadSystem(p, f, .1, seed.load_n, elasticity_mode="cuda")


def test_cached_nonlinear_root_and_original_armijo_checks(case):
    p, f, seed = case
    roots = []
    for mode in ("reference", "cached"):
        r = solve_property_load_newton(
            p, f, .1, seed.load_n*1.05, seed=seed, seed_offset_m=p.central_film_offset_m,
            elasticity_mode=mode,
        )
        roots.append(r)
        s = r.solution
        assert admissible(s.audit, s.pressure_pa, s.fractional_film_content, p.pressure_scale_pa)
        assert abs(s.load_n/(seed.load_n*1.05)-1) <= 1e-8
        # Independent original convolution and original SI mass audit are retained.
        h, w = p.fields(s.pressure_pa, central_film_offset_m=r.central_film_offset_m)
        np.testing.assert_allclose(s.film_thickness_m, h, rtol=0, atol=1e-18)
        np.testing.assert_allclose(s.elastic_displacement_m, w, rtol=0, atol=1e-18)
        for row in s.history:
            accepted = [(m, t) for m in row["models"] for t in m["trials"] if t["accepted"]]
            assert len(accepted) == 1
            m, t = accepted[0]
            assert m["linear_true_residual"] <= 5e-9
            assert m["slope"] < 0
            assert t["merit"] <= row["merit"]+1e-4*t["alpha"]*m["slope"]
    a, b = roots
    assert np.linalg.norm(a.solution.pressure_pa-b.solution.pressure_pa)/np.linalg.norm(
        a.solution.pressure_pa) < 1e-7
    assert abs(a.central_film_offset_m-b.central_film_offset_m) < 1e-12*p.film_scale_m


def test_cached_krylov_failure_keeps_diagnostics(case, monkeypatch):
    import ehl_solver.cavitating_newton as linear

    p, f, seed = case

    def fail(op, rhs, **kwargs):
        kwargs["callback"](1.)
        return np.zeros_like(rhs), 1

    monkeypatch.setattr(linear, "gmres", fail)
    with pytest.raises(PropertyLoadNewtonError) as caught:
        solve_property_load_newton(
            p, f, .1, 1.05*seed.load_n, seed=seed, seed_offset_m=p.central_film_offset_m,
            elasticity_mode="cached",
        )
    assert caught.value.diagnostics["true_linear_residual"] > 5e-9
    assert caught.value.diagnostics["krylov_histories"]
    assert caught.value.last_root is not None


def test_cached_continuation_retry_keeps_mode_and_original_seed(case, monkeypatch):
    import ehl_solver.property_load_newton as module

    p, f, seed = case
    original, calls = module.solve_property_load_newton, []

    def solve(*args, **kwargs):
        calls.append((kwargs["seed_offset_m"], kwargs["elasticity_mode"]))
        if len(calls) == 1:
            raise PropertyLoadNewtonError("injected retry", (), None)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "solve_property_load_newton", solve)
    result = continue_property_load(
        p, f, .1, 1.05*seed.load_n, seed=seed, seed_offset_m=p.central_film_offset_m,
        initial_increment_n=.05*seed.load_n, elasticity_mode="cached",
    )
    assert calls[0] == calls[1] == (p.central_film_offset_m, "cached")
    assert not result.attempts[0]["accepted"]
    assert result.attempts[-1]["accepted"]
    assert abs(result.root.solution.load_n/(1.05*seed.load_n)-1) <= 1e-8


def test_cached_active_lookahead_and_rejected_trials(case):
    p, f, seed = case
    guess = PropertyLoadGuess(seed.pressure_pa, np.ones_like(seed.fractional_film_content),
                              p.central_film_offset_m, {"accepted": False})
    result = repair_property_load_guess(p, f, .1, seed.load_n, guess, elasticity_mode="cached")
    history = result.solution.history
    assert any(m["recovery"] > 0 for row in history for m in row["models"])
    assert any(not t["accepted"] for row in history for m in row["models"] for t in m["trials"])
    assert admissible(result.solution.audit, result.solution.pressure_pa,
                      result.solution.fractional_film_content, p.pressure_scale_pa)
    assert np.linalg.norm(result.solution.pressure_pa-seed.pressure_pa)/np.linalg.norm(
        seed.pressure_pa) < 1e-7


def test_cached_failed_selected_model_retains_feasible_backup(case, monkeypatch):
    import ehl_solver.property_load_newton as module
    from ehl_solver.cavitating_newton import CavitatingLinearError

    p, f, seed = case
    guess = PropertyLoadGuess(seed.pressure_pa, np.ones_like(seed.fractional_film_content),
                              p.central_film_offset_m, {"accepted": False})
    original = module.solve_cavitating_correction

    def fail_selected(system, *args, **kwargs):
        if system.linear_mode:
            raise CavitatingLinearError("injected selected failure", {"injected": True})
        return original(system, *args, **kwargs)

    monkeypatch.setattr(module, "solve_cavitating_correction", fail_selected)
    with pytest.raises(PropertyLoadNewtonError, match="budget") as caught:
        repair_property_load_guess(p, f, .1, seed.load_n, guess, maximum_iterations=1,
                                   elasticity_mode="cached")
    row = caught.value.history[0]
    assert row["active_recovery_failures"][0]["diagnostics"]["injected"]
    accepted = [(m, t) for m in row["models"] for t in m["trials"] if t["accepted"]]
    assert len(accepted) == 1
    m, t = accepted[0]
    assert "fallback" in t["reason"]
    assert t["merit"] <= row["merit"]+1e-4*t["alpha"]*m["slope"]
