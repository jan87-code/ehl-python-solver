from dataclasses import replace

import numpy as np
import pytest

from ehl_solver.constitutive import IsothermalFluid
from ehl_solver.property_coupled import solve_property_coupling
from ehl_solver.property_load_newton import PropertyLoadNewtonError, solve_property_load_newton
from ehl_solver.property_transfer import prolongate_property_guess, repair_property_load_guess
from examples.run_property_load_verification import make_problem


@pytest.fixture(scope="module")
def transfer():
    source = make_problem(7, 9)
    target = make_problem(13, 17)
    fluid = IsothermalFluid(0.04, 850)
    seed = solve_property_coupling(source, fluid, 0.1)
    guess = prolongate_property_guess(
        source, target, fluid, 0.1, seed=seed, seed_offset_m=source.central_film_offset_m
    )
    return source, target, fluid, seed, guess


def test_nested_force_preservation_and_explicit_nonroot(transfer):
    source, target, fluid, seed, guess = transfer
    assert guess.diagnostics["accepted"] is False
    assert guess.diagnostics["offset_adjustment_m"] == 0
    assert guess.diagnostics["relative_load_difference"] < 1e-12
    assert target.load_n(guess.pressure_pa) == pytest.approx(seed.load_n, rel=1e-12)
    np.testing.assert_allclose(guess.pressure_pa[::2, ::2], seed.pressure_pa, rtol=1e-12, atol=1e-9)
    original = seed.pressure_pa.copy()
    with pytest.raises(PropertyLoadNewtonError, match="budget") as error:
        repair_property_load_guess(target, fluid, 0.1, seed.load_n, guess, maximum_iterations=0)
    assert error.value.last_root is not None
    np.testing.assert_array_equal(seed.pressure_pa, original)


def test_repaired_force_matches_independent_cold_initialization(transfer):
    source, target, f, seed, guess = transfer
    result = repair_property_load_guess(target, f, 0.1, seed.load_n, guess)
    cold = solve_property_coupling(target, f, 0.1)
    reference = solve_property_load_newton(
        target, f, 0.1, seed.load_n, seed=cold, seed_offset_m=target.central_film_offset_m
    )
    assert abs(result.solution.load_n / seed.load_n - 1) <= 1e-8
    assert result.solution.audit["relative_local_balance"] <= 1e-10
    assert (
        np.linalg.norm(result.solution.pressure_pa - reference.solution.pressure_pa)
        / np.linalg.norm(reference.solution.pressure_pa)
        < 1e-7
    )
    assert abs(result.central_film_offset_m / reference.central_film_offset_m - 1) < 1e-7


def test_transfer_rejects_physical_changes_and_invalid_guesses(transfer):
    source, target, f, seed, guess = transfer
    with pytest.raises(ValueError, match="same geometry"):
        prolongate_property_guess(
            source,
            replace(target, e_star_pa=2 * target.e_star_pa),
            f,
            0.1,
            seed=seed,
            seed_offset_m=source.central_film_offset_m,
        )
    shifted = replace(target, x_m=target.x_m + 1e-6)
    with pytest.raises(ValueError, match="reservoir bounds"):
        prolongate_property_guess(
            source, shifted, f, 0.1, seed=seed, seed_offset_m=source.central_film_offset_m
        )
    p = guess.pressure_pa.copy()
    p[0, 0] = 1.0
    with pytest.raises(ValueError, match="boundary bounds"):
        repair_property_load_guess(target, f, 0.1, seed.load_n, replace(guess, pressure_pa=p))
    with pytest.raises(ValueError, match="film"):
        repair_property_load_guess(
            target, f, 0.1, seed.load_n, replace(guess, central_film_offset_m=-1.0)
        )


def test_active_lookahead_recovers_flooded_initial_content(transfer):
    from ehl_solver.property_transfer import PropertyLoadGuess

    p, _, fluid, seed, _ = transfer
    guess = PropertyLoadGuess(
        seed.pressure_pa.copy(),
        np.ones_like(seed.fractional_film_content),
        p.central_film_offset_m,
        {"accepted": False},
    )
    result = repair_property_load_guess(p, fluid, 0.1, seed.load_n, guess)
    assert any(m["recovery"] > 0 for row in result.solution.history for m in row["models"])
    assert (
        np.linalg.norm(result.solution.pressure_pa - seed.pressure_pa)
        / np.linalg.norm(seed.pressure_pa)
        < 1e-7
    )


def test_failed_lookahead_keeps_verified_feasible_backup(transfer, monkeypatch):
    import ehl_solver.property_load_newton as module
    from ehl_solver.cavitating_newton import CavitatingLinearError
    from ehl_solver.property_transfer import PropertyLoadGuess

    p, _, fluid, seed, _ = transfer
    guess = PropertyLoadGuess(
        seed.pressure_pa.copy(),
        np.ones_like(seed.fractional_film_content),
        p.central_film_offset_m,
        {"accepted": False},
    )
    real = module.solve_cavitating_correction

    def fail_selected(system, *args, **kwargs):
        if system.linear_mode:
            raise CavitatingLinearError("injected selected-model failure", {"test_failure": True})
        return real(system, *args, **kwargs)

    monkeypatch.setattr(module, "solve_cavitating_correction", fail_selected)
    with pytest.raises(PropertyLoadNewtonError, match="budget") as error:
        repair_property_load_guess(p, fluid, 0.1, seed.load_n, guess, maximum_iterations=1)
    row = error.value.history[0]
    assert row["active_recovery_failures"][0]["diagnostics"]["test_failure"]
    accepts = [(m, t) for m in row["models"] for t in m["trials"] if t["accepted"]]
    assert len(accepts) == 1
    model, t = accepts[0]
    assert "fallback" in t["reason"]
    assert t["merit"] <= row["merit"] + 1e-4 * t["alpha"] * model["slope"]
