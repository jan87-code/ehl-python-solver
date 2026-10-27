"""Analytic complementarity benchmark for a blocking theta=0 constraint."""

from types import SimpleNamespace

import numpy as np
import pytest

from ehl_solver.property_load_newton import (
    PropertyLoadNewtonError,
    _content_bound_blockers,
    _solve_property_load_state,
)


def test_only_earliest_near_empty_content_constraints_are_selected():
    # n=3, pressure P, deficit D=1-theta, then rigid offset.
    z = np.array([0.1, 0.1, 0.1, 1 + 1e-10, 1 - 1e-11, 0.5, 1.0])
    d = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0])
    np.testing.assert_array_equal(_content_bound_blockers(z, d, 3), [0])
    z[3] = z[4] = 1 + 1e-10
    np.testing.assert_array_equal(_content_bound_blockers(z, d, 3), [0, 1])
    np.testing.assert_array_equal(_content_bound_blockers(z, -d, 3), [])


def test_other_bounds_and_nonempty_content_do_not_trigger_content_pivots():
    z = np.array([-1e-10, 0.1, 1.0, 0.5, 1.0])
    d = np.array([-1.0, 0.0, 1.0, 1.0, 0.0])
    np.testing.assert_array_equal(_content_bound_blockers(z, d, 2), [])
    z = np.array([0.1, 0.1, 0.8, 0.5, 1.0])
    d = np.array([0.0, 0.0, 1e12, 0.0, 0.0])
    np.testing.assert_array_equal(_content_bound_blockers(z, d, 2), [])


class ContentSystem:
    """Mass P+D=2, min(P,D)=0 has feasible root P=2,D=0.

    At P=.1,D=1+tol the min-selected cavitated model requests D_new=2,
    so theta falls below its lower bound. Selecting the full-film constraint
    D_new=0 instead recovers the exact root without projection.
    """

    mass_pressure_coefficient = 1.0
    n, size, target_load_n = 1, 3, 1.0
    problem = SimpleNamespace(pressure_scale_pa=1.0, film_floor_m=0.01)
    base = SimpleNamespace(unpack=lambda z: (None, None, np.ones((2, 2))))

    def full(self, z):
        return z[:1] >= z[1:2]

    def sync(self, z):
        pass

    def film_direction(self, z, d):
        return np.zeros((2, 2))

    def residual(self, z):
        return np.array([self.mass_pressure_coefficient * z[0] + z[1] - 2, min(z[:2]), z[2] - 1])

    def merit_slope(self, z, r, d):
        dc = d[1] if z[0] >= z[1] else d[0]
        return float(r @ np.array([self.mass_pressure_coefficient * d[0] + d[1], dc, d[2]]) / 3)

    def correction_view(self, z, full):
        return SimpleNamespace(selected_full=full[0])

    def solution(self, z, history):
        defect = abs(self.mass_pressure_coefficient * z[0] + z[1] - 2)
        audit = {
            "mass_maximum": defect,
            "relative_local_balance": defect,
            "relative_global_balance": defect,
            "maximum_complementarity": abs(z[0] * z[1]),
        }
        return SimpleNamespace(
            central_film_offset_m=z[-1],
            solution=SimpleNamespace(
                audit=audit,
                pressure_pa=z[:1],
                fractional_film_content=1 - z[1:2],
                load_n=1.0,
                history=history,
            ),
        )


@pytest.mark.parametrize("enabled", [False, True])
def test_targeted_content_pivot_recovers_analytic_root_with_original_armijo(monkeypatch, enabled):
    import ehl_solver.property_load_newton as module

    system = ContentSystem()
    z = np.array([0.1, 1 + 1e-10, 1.0])
    original = z.copy()

    def solve(view, state, **kwargs):
        full = bool(system.full(state)[0]) if view is system else view.selected_full
        jac = np.array(
            [[1.0, 1.0, 0.0], [0.0, 1.0, 0.0] if full else [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        rhs = np.array([state[0] + state[1] - 2, state[1] if full else state[0], state[2] - 1])
        d = np.linalg.solve(jac, -rhs)
        assert np.linalg.norm(jac @ d + rhs) < 1e-14
        return SimpleNamespace(
            direction=d,
            true_linear_residual=0.0,
            linear_iterations=1,
            krylov_histories=[],
            used_bordered_fallback=False,
            schur_indicator=1.0,
        )

    monkeypatch.setattr(module, "solve_cavitating_correction", solve)
    if not enabled:
        with pytest.raises(PropertyLoadNewtonError, match="exhausted") as error:
            _solve_property_load_state(
                system, z, 1, 1, 1e-10, True, active_lookahead=True, content_bound_pivots=False
            )
        np.testing.assert_array_equal(error.value.last_root.solution.pressure_pa, z[:1])
    else:
        result = _solve_property_load_state(
            system, z, 1, 1, 1e-10, True, active_lookahead=True, content_bound_pivots=True
        )
        np.testing.assert_allclose(result.solution.pressure_pa, [2.0], atol=1e-14)
        np.testing.assert_allclose(result.solution.fractional_film_content, [1.0], atol=1e-14)
        row = result.solution.history[0]
        assert row["models"][0]["content_bound_pivots"] == [0]
        chosen = [(m, t) for m in row["models"] for t in m["trials"] if t["accepted"]]
        assert len(chosen) == 1
        m, t = chosen[0]
        assert m["recovery"] == 1
        assert t["merit"] <= row["merit"] + 1e-4 * t["alpha"] * m["slope"]
    np.testing.assert_array_equal(z, original)


def test_infeasible_complementarity_cycle_retains_failure_and_original_state(monkeypatch):
    import ehl_solver.property_load_newton as module

    system = ContentSystem()
    # D-P=2 has no P>=-tol, D<=1+tol root. Both exact active models
    # point out of the current feasible corner; a repeated model must stop.
    system.mass_pressure_coefficient = -1.0
    z = np.array([-1e-10, 1 + 1e-10, 1.0])
    original = z.copy()
    calls = []

    def solve(view, state, **kwargs):
        full = bool(system.full(state)[0]) if view is system else view.selected_full
        calls.append(full)
        jac = np.array(
            [[-1.0, 1.0, 0.0], [0.0, 1.0, 0.0] if full else [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        rhs = np.array([-state[0] + state[1] - 2, state[1] if full else state[0], state[2] - 1])
        d = np.linalg.solve(jac, -rhs)
        assert np.linalg.norm(jac @ d + rhs) < 1e-14
        return SimpleNamespace(
            direction=d,
            true_linear_residual=0.0,
            linear_iterations=1,
            krylov_histories=[],
            used_bordered_fallback=False,
            schur_indicator=1.0,
        )

    monkeypatch.setattr(module, "solve_cavitating_correction", solve)
    with pytest.raises(PropertyLoadNewtonError, match="exhausted") as error:
        _solve_property_load_state(
            system, z, 1, 1, 1e-10, True, active_lookahead=True, content_bound_pivots=True
        )
    assert calls == [False, True]
    assert not any(t["accepted"] for m in error.value.history[0]["models"] for t in m["trials"])
    np.testing.assert_array_equal(error.value.last_root.solution.pressure_pa, z[:1])
    np.testing.assert_array_equal(z, original)
