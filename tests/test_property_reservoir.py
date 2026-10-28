"""Independent small-domain root parity and exact force/node preservation."""

from dataclasses import replace

import numpy as np
import pytest

from ehl_solver.constitutive import IsothermalFluid
from ehl_solver.property_coupled import solve_property_coupling
from ehl_solver.property_load_newton import PropertyLoadNewtonError, solve_property_load_newton
from ehl_solver.property_reservoir import extend_property_guess, extended_reservoir_problem
from ehl_solver.property_transfer import repair_property_load_guess
from examples.run_property_load_verification import make_problem


@pytest.fixture(scope="module")
def source():
    problem = make_problem(7, 9)
    fluid = IsothermalFluid(.04, 850.)
    seed = solve_property_coupling(problem, fluid, .1)
    return problem, fluid, seed


@pytest.mark.parametrize("counts", [dict(west=2), dict(east=2), dict(south=1, north=1)])
def test_same_spacing_force_and_exact_overlap_without_seed_mutation(source, counts):
    p, fluid, seed = source
    before = seed.pressure_pa.copy()
    target = extended_reservoir_problem(p, **counts)
    guess = extend_property_guess(p, target, fluid, .1, seed=seed,
                                  seed_offset_m=p.central_film_offset_m)
    assert not guess.diagnostics["accepted"]
    assert guess.diagnostics["relative_force_difference"] < 1e-12
    assert guess.diagnostics["overlap_film_error_m"] < 1e-18
    assert guess.diagnostics["offset_adjustment_m"] == 0.
    np.testing.assert_array_equal(seed.pressure_pa, before)
    assert target.dx_m == pytest.approx(p.dx_m, rel=1e-12)
    assert target.dy_m == pytest.approx(p.dy_m, rel=1e-12)
    if counts.get("west", 0):
        np.testing.assert_array_equal(guess.pressure_pa[:, :counts["west"]], 0)


def test_extended_repair_matches_independent_cold_root_at_same_force(source):
    p, fluid, seed = source
    target = extended_reservoir_problem(p, west=2)
    guess = extend_property_guess(p, target, fluid, .1, seed=seed,
                                  seed_offset_m=p.central_film_offset_m)
    with pytest.raises(PropertyLoadNewtonError):
        repair_property_load_guess(target, fluid, .1, seed.load_n, guess, maximum_iterations=0)
    repaired = repair_property_load_guess(target, fluid, .1, seed.load_n, guess)
    cold = solve_property_coupling(target, fluid, .1)
    independent = solve_property_load_newton(
        target, fluid, .1, seed.load_n, seed=cold, seed_offset_m=target.central_film_offset_m
    )
    np.testing.assert_allclose(repaired.solution.pressure_pa, independent.solution.pressure_pa,
                               rtol=2e-7, atol=1e-5)
    assert repaired.central_film_offset_m == pytest.approx(independent.central_film_offset_m,
                                                          rel=2e-7)
    assert abs(repaired.solution.load_n/seed.load_n-1) < 1e-8


def test_extension_rejects_physical_change_noninteger_nodes_and_cap(source):
    p, fluid, seed = source
    target = extended_reservoir_problem(p, west=2)
    with pytest.raises(ValueError, match="geometry"):
        extend_property_guess(p, replace(target, e_star_pa=p.e_star_pa*2), fluid, .1,
                              seed=seed, seed_offset_m=p.central_film_offset_m)
    with pytest.raises(ValueError, match="exact-node"):
        extend_property_guess(p, replace(target, x_m=target.x_m+1e-9), fluid, .1,
                              seed=seed, seed_offset_m=p.central_film_offset_m)
    for counts in (dict(west=True), dict(west=-1), dict(west=9), dict()):
        with pytest.raises(ValueError):
            extended_reservoir_problem(p, **counts)
    # Guard is checked before constructing a kernel or allocating a target field.
    from types import SimpleNamespace
    large = SimpleNamespace(x_m=np.arange(145), y_m=np.arange(137))
    with pytest.raises(ValueError, match="20000"):
        extended_reservoir_problem(large, west=8)
