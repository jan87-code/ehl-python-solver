"""Raw-state model replay and recorded-history classification, independent gates."""

import copy

import numpy as np
import pytest

from ehl_solver.constitutive import IsothermalFluid
from ehl_solver.property_coupled import solve_property_coupling
from ehl_solver.property_load_newton import (
    PropertyLoadNewtonError,
    PropertyLoadSystem,
    _solve_property_load_state,
)
from ehl_solver.property_transfer import PropertyLoadGuess
from examples.analyze_mesh_newton import replay_model, summarize_history
from examples.run_property_load_verification import make_problem


@pytest.fixture(scope="module")
def capture():
    problem, fluid = make_problem(7, 9), IsothermalFluid(.04, 850.)
    seed = solve_property_coupling(problem, fluid, .1)
    system = PropertyLoadSystem(problem, fluid, .1, seed.load_n, preconditioner="local",
                                elasticity_mode="cached")
    guess = PropertyLoadGuess(seed.pressure_pa, np.ones_like(seed.fractional_film_content),
                              problem.central_film_offset_m, {"accepted": False})
    z = system.pack(guess, guess.central_film_offset_m)
    snapshots = {}

    def observe(index, state, root):
        snapshots[index] = state.copy()

    try:
        root = _solve_property_load_state(system, z, 1, 10, 1e-10, True,
                    active_lookahead=True, content_bound_pivots=True, state_observer=observe)
        history = root.solution.history
    except PropertyLoadNewtonError as exc:
        history = exc.history
    assert history and 1 in snapshots
    return system, snapshots, history


def test_replay_selected_rows_true_correction_original_slope_bound_and_next_state(capture):
    system, snapshots, history = capture
    models = [m for m in history[0]["models"] if any(t["accepted"] for t in m["trials"])]
    assert len(models) == 1
    model = models[0]
    result = replay_model(system, snapshots[0], model, next_state=snapshots[1])
    assert result["complete_true_linear_residual"] <= 5e-9
    assert result["original_slope_relative_difference"] <= 1e-6
    assert result["original_armijo_pass"]
    assert result["sampled_trials"]
    assert result["next_state_relative_difference"] < 1e-10
    assert result["recomputed_fraction_to_boundary"] == pytest.approx(model["fraction_to_boundary"],
                                                                       rel=1e-7, abs=1e-15)


def test_replay_requires_exact_selected_row_metadata_and_next_snapshot(capture):
    system, snapshots, history = capture
    selected = next(m for m in history[0]["models"] if any(t["accepted"] for t in m["trials"]))
    old = copy.deepcopy(selected)
    del old["active_override_indices"]
    with pytest.raises(ValueError, match="metadata"):
        replay_model(system, snapshots[0], old)
    bad = copy.deepcopy(selected)
    bad["active_override_indices"] = [system.n]
    bad["active_changes"] = 1
    with pytest.raises(ValueError, match="indices"):
        replay_model(system, snapshots[0], bad)
    bad["active_override_indices"] = [1.5]
    with pytest.raises(ValueError, match="integer"):
        replay_model(system, snapshots[0], bad)
    with pytest.raises(ValueError, match="finite exact next"):
        replay_model(system, snapshots[0], selected, next_state=np.full_like(snapshots[1], np.nan))
    changed = snapshots[1].copy()
    changed[-1] += 1
    with pytest.raises(RuntimeError, match="next snapshot"):
        replay_model(system, snapshots[0], selected, next_state=changed)


def test_summary_separates_linear_checks_active_changes_and_nonaccepted_trial_reasons(capture):
    _, _, history = capture
    result = summarize_history(history)
    assert result["models_total"] == len(history[0]["models"])
    expected = sum(m["linear_iterations"] for m in history[0]["models"])
    assert result["linear_iterations_total"] == expected
    assert result["recorded_final_linear_gate_failures"] == []
    assert result["minimum_accepted_alpha"] <= 1
    assert result["nonaccepted_trial_reasons"]
    changed = copy.deepcopy(history)
    changed[0]["models"][0]["linear_true_residual"] = 1.
    assert summarize_history(changed)["recorded_final_linear_gate_failures"] == [[0, 0]]


def test_summary_rejects_ambiguous_acceptance_and_noncontiguous_history(capture):
    _, _, history = capture
    changed = copy.deepcopy(history)
    selected = next(t for m in changed[0]["models"] for t in m["trials"] if t["accepted"])
    changed[0]["models"][0]["trials"].append(dict(selected))
    with pytest.raises(ValueError, match="one accepted"):
        summarize_history(changed)
    changed = copy.deepcopy(history)
    changed[0]["iteration"] = 2
    with pytest.raises(ValueError, match="contiguous"):
        summarize_history(changed)
