"""Diagnostic observers cannot mutate Newton state or promote unaccepted iterates."""

import numpy as np
import pytest

from ehl_solver.cavitating_newton import CavitatingLinearError
from ehl_solver.constitutive import IsothermalFluid
from ehl_solver.property_coupled import solve_property_coupling
from ehl_solver.property_load_newton import (
    PropertyLoadNewtonError,
    PropertyLoadSystem,
    PropertyStateObserverError,
    _solve_property_load_state,
)
from examples.run_property_load_verification import make_problem


@pytest.fixture(scope="module")
def case():
    p, f = make_problem(7, 9), IsothermalFluid(.04, 850.)
    seed = solve_property_coupling(p, f, .1)
    system = PropertyLoadSystem(p, f, .1, seed.load_n*1.05, preconditioner="local",
                                elasticity_mode="cached")
    return system, system.pack(seed, p.central_film_offset_m)


def solve(system, z, observer=None, iterations=30):
    return _solve_property_load_state(system, z, iterations, 10, 1e-10, True,
                                      state_observer=observer)


def test_snapshot_mutations_cannot_change_actual_state_audit_history_or_result(case):
    system, z = case
    original = z.copy()
    baseline = solve(system, z)
    observed = []

    def observer(iteration, snapshot, root):
        assert not snapshot.flags.writeable
        observed.append((iteration, snapshot.copy()))
        snapshot.setflags(write=True)
        snapshot[:] = 1e100
        root.solution.pressure_pa[:] = -1e100
        root.solution.audit["mass_maximum"] = np.inf
        if root.solution.history:
            root.solution.history[0]["models"].clear()

    candidate = solve(system, z, observer)
    np.testing.assert_array_equal(z, original)
    np.testing.assert_array_equal(candidate.solution.pressure_pa, baseline.solution.pressure_pa)
    np.testing.assert_array_equal(candidate.solution.fractional_film_content,
                                  baseline.solution.fractional_film_content)
    assert candidate.central_film_offset_m == baseline.central_film_offset_m
    assert [i for i, _ in observed] == list(range(len(candidate.solution.history)+1))
    np.testing.assert_array_equal(observed[0][1], original)
    assert candidate.solution.audit["mass_maximum"] <= 1e-8
    for row in candidate.solution.history:
        for model in row["models"]:
            assert len(model["active_override_indices"]) == model["active_changes"]


def test_observer_failure_is_separate_from_nonlinear_failure_and_keeps_original_cause(case):
    system, z = case

    def broken(*args):
        raise ValueError("diagnostic output failed")

    with pytest.raises(PropertyStateObserverError) as caught:
        solve(system, z, broken)
    assert isinstance(caught.value.__cause__, ValueError)
    assert not isinstance(caught.value, PropertyLoadNewtonError)
    with pytest.raises(ValueError, match="callable"):
        solve(system, z, "bad")


def test_zero_budget_state_is_observed_but_not_promoted_to_accepted_root(case):
    system, z = case
    seen = []
    with pytest.raises(PropertyLoadNewtonError, match="budget"):
        solve(system, z, lambda i, state, root: seen.append((i, state.copy())), iterations=0)
    assert len(seen) == 1 and seen[0][0] == 0
    np.testing.assert_array_equal(seen[0][1], z)


def test_failed_linear_model_retains_exact_snapshot_and_selected_rows(case, monkeypatch):
    import ehl_solver.property_load_newton as module

    system, z = case
    seen = []

    def fail(*args, **kwargs):
        raise CavitatingLinearError("injected linear failure", {"true_linear_residual": 1.})

    monkeypatch.setattr(module, "solve_cavitating_correction", fail)
    with pytest.raises(PropertyLoadNewtonError) as caught:
        solve(system, z, lambda i, state, root: seen.append(state.copy()))
    assert caught.value.diagnostics["iteration"] == 0
    assert caught.value.diagnostics["model"] == 0
    assert caught.value.diagnostics["active_override_indices"] == []
    np.testing.assert_array_equal(seen[0], z)
