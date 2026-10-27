"""Globalization regressions isolate candidate choice from physical equations."""

from types import SimpleNamespace

import numpy as np
import pytest

from ehl_solver.property_load_newton import PropertyLoadNewtonError, _solve_property_load_state


class CandidateSystem:
    n, size, target_load_n = 2, 5, 1.0
    problem = SimpleNamespace(pressure_scale_pa=1.0, film_floor_m=0.01)
    base = SimpleNamespace(unpack=lambda z: (None, None, np.ones((2, 2))))

    def solution(self, z, history):
        solution = SimpleNamespace(
            audit={
                "mass_maximum": 1,
                "relative_local_balance": 1,
                "relative_global_balance": 1,
                "maximum_complementarity": 1,
            },
            pressure_pa=z[:2],
            fractional_film_content=1 - z[2:4],
            load_n=1,
        )
        return SimpleNamespace(solution=solution, central_film_offset_m=z[-1])

    def residual(self, z):
        return np.r_[z[:2], np.zeros(3)]

    def full(self, z):
        return np.ones(2, dtype=bool)

    def correction_view(self, z, full):
        return self

    def sync(self, z):
        pass

    def film_direction(self, z, d):
        return np.zeros((2, 2))

    def merit_slope(self, z, r, d):
        return float(r[:2] @ d[:2] / self.size)


def correction(direction):
    return SimpleNamespace(
        direction=np.array(direction),
        true_linear_residual=0.0,
        linear_iterations=1,
        krylov_histories=[],
        used_bordered_fallback=False,
        schur_indicator=1.0,
    )


@pytest.mark.parametrize("lookahead", [False, True])
def test_better_deferred_step_wins_over_worse_final_model(monkeypatch, lookahead):
    import ehl_solver.property_load_newton as module

    directions = iter(([-20.0, 0.0, 0.0, 0.0, 0.0], [-1e12, 1e11, 0.0, 0.0, 0.0]))
    monkeypatch.setattr(
        module, "solve_cavitating_correction", lambda *a, **k: correction(next(directions))
    )
    z = np.array([1.0, 1.0, 0.1, 0.1, 1.0])
    with pytest.raises(PropertyLoadNewtonError, match="budget") as error:
        _solve_property_load_state(
            CandidateSystem(), z, 1, 1, 1e-10, True, active_lookahead=lookahead
        )
    row = error.value.history[0]
    accepts = [(m, t) for m in row["models"] for t in m["trials"] if t["accepted"]]
    assert len(accepts) == 1
    m, chosen = accepts[0]
    assert m["recovery"] == 0
    assert chosen["merit"] < 0.101
    assert chosen["merit"] <= row["merit"] + 1e-4 * chosen["alpha"] * m["slope"]
    if lookahead:
        assert len(row["models"]) == 2
        assert any("superseded" in t["reason"] for t in row["models"][-1]["trials"])


@pytest.mark.parametrize("offset_step", [0.0, 1.0])
def test_lookahead_rejects_rounded_no_progress_step(monkeypatch, offset_step):
    import ehl_solver.property_load_newton as module

    monkeypatch.setattr(
        module,
        "solve_cavitating_correction",
        lambda *a, **k: correction([-1e-30, 0.0, 0.0, 0.0, offset_step]),
    )
    z = np.array([1.0, 1.0, 0.1, 0.1, 1.0])
    with pytest.raises(PropertyLoadNewtonError, match="exhausted") as error:
        _solve_property_load_state(CandidateSystem(), z, 1, 1, 1e-10, True, active_lookahead=True)
    assert not any(t["accepted"] for m in error.value.history[0]["models"] for t in m["trials"])
    np.testing.assert_array_equal(error.value.last_root.solution.pressure_pa, z[:2])
    assert any(
        "progress" in t["reason"] for m in error.value.history[0]["models"] for t in m["trials"]
    )


def test_legacy_selection_reproduces_worse_final_model(monkeypatch):
    import ehl_solver.property_load_newton as module

    directions = iter(([-20.0, 0.0, 0.0, 0.0, 0.0], [-1e12, 1e11, 0.0, 0.0, 0.0]))
    monkeypatch.setattr(
        module, "solve_cavitating_correction", lambda *a, **k: correction(next(directions))
    )
    z = np.array([1.0, 1.0, 0.1, 0.1, 1.0])
    with pytest.raises(PropertyLoadNewtonError, match="budget") as error:
        _solve_property_load_state(
            CandidateSystem(),
            z,
            1,
            1,
            1e-10,
            True,
            active_lookahead=True,
            lookahead_selection="legacy",
        )
    row = error.value.history[0]
    chosen = [(m, t) for m in row["models"] for t in m["trials"] if t["accepted"]]
    assert len(chosen) == 1
    assert chosen[0][0]["recovery"] == 1
    assert chosen[0][1]["merit"] > 0.12


def test_invalid_lookahead_selection_is_rejected():
    with pytest.raises(ValueError, match="lookahead_selection"):
        _solve_property_load_state(
            CandidateSystem(), np.ones(5), 1, 1, 1e-10, True, lookahead_selection="unrecognized"
        )
