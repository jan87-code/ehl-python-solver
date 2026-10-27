from dataclasses import replace

import numpy as np
import pytest

from ehl_solver.constitutive import IsothermalFluid
from ehl_solver.property_coupled import solve_property_coupling
from ehl_solver.property_warm import (
    PropertyWarmSystem,
    WarmPropertyError,
    solve_warm_property_offset,
)
from examples.property_monolithic_reference import small_monolithic_reference
from examples.run_property_load_verification import make_problem


@pytest.fixture(scope="module")
def seed_case():
    problem = make_problem(7, 9)
    fluid = IsothermalFluid(0.04, 850)
    seed = solve_property_coupling(problem, fluid, 0.1)
    return problem, fluid, seed


def test_full_material_elastic_jvp_and_true_tie_slope(seed_case):
    p, f, seed = seed_case
    system = PropertyWarmSystem(p, f, 0.1, p.central_film_offset_m)
    rng = np.random.default_rng(814)
    z = np.r_[rng.uniform(0.001, 0.002, system.n), rng.uniform(0.02, 0.05, system.n)]
    v = rng.normal(size=system.size) * 0.001
    eps = 1e-5
    difference = (system.residual(z + eps * v) - system.residual(z - eps * v)) / (2 * eps)
    np.testing.assert_allclose(system.jacobian_vector(z, v), difference, rtol=2e-6, atol=1e-9)
    z[: system.n] = z[system.n :] = 0
    residual = system.residual(z)
    # Give complementarity rows a nonzero weight to exercise directional ties.
    residual[system.n :] = 1
    action = system.jacobian_vector(z, v)
    action[system.n :] = np.minimum(v[: system.n], v[system.n :])
    assert system.merit_slope(z, residual, v) == pytest.approx(residual @ action / system.size)


def test_warm_positive_offset_matches_independent_reference(seed_case):
    p, f, seed = seed_case
    offset = p.central_film_offset_m * 0.9
    result = solve_warm_property_offset(
        p, f, 0.1, offset, seed=seed, seed_offset_m=p.central_film_offset_m
    )
    _, (pressure, theta, h, w) = small_monolithic_reference(
        replace(p, central_film_offset_m=offset), f, 0.1
    )
    np.testing.assert_allclose(result.solution.pressure_pa, pressure, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(result.solution.film_thickness_m, h, rtol=1e-8)
    np.testing.assert_allclose(result.solution.fractional_film_content, theta, atol=1e-8)
    for record in result.solution.history:
        assert record["true_linear_residual"] <= 5e-9
        assert record["slope"] < 0
        assert record["trials"][-1]["accepted"]


def test_seed_identity_and_invalid_seed_and_feasibility(seed_case):
    p, f, seed = seed_case
    result = solve_warm_property_offset(
        p, f, 0.1, p.central_film_offset_m, seed=seed, seed_offset_m=p.central_film_offset_m
    )
    assert not result.solution.history
    np.testing.assert_array_equal(result.solution.pressure_pa, seed.pressure_pa)
    with pytest.raises(ValueError, match="seed failed"):
        solve_warm_property_offset(
            p, f, 0.1, 0.0, seed=seed, seed_offset_m=2 * p.central_film_offset_m
        )
    with pytest.raises(WarmPropertyError) as error:
        solve_warm_property_offset(
            p, f, 0.1, -1.0, seed=seed, seed_offset_m=p.central_film_offset_m
        )
    assert error.value.last_solution is None
    assert "film" in str(error.value)
    with pytest.raises(WarmPropertyError, match="budget") as error:
        solve_warm_property_offset(
            p,
            f,
            0.1,
            p.central_film_offset_m * 0.9,
            seed=seed,
            seed_offset_m=p.central_film_offset_m,
            maximum_iterations=0,
        )
    assert error.value.last_solution is not None


def test_dense_resource_cap_and_invalid_speed(seed_case):
    p, f, _ = seed_case
    with pytest.raises(ValueError, match="400"):
        PropertyWarmSystem(make_problem(33, 33), f, 0.1, 0.0)
    with pytest.raises(ValueError, match="Lambda"):
        PropertyWarmSystem(p, f, 0.2, 0.0)


def test_continuation_rejects_trial_and_rolls_back(seed_case, monkeypatch):
    import ehl_solver.property_warm as module

    p, f, seed = seed_case
    real = module.solve_warm_property_offset
    calls = []

    def solve(*args, **kwargs):
        calls.append(kwargs["seed_offset_m"])
        if len(calls) == 2:
            raise WarmPropertyError("injected retry", (), None)
        return real(*args, **kwargs)

    monkeypatch.setattr(module, "solve_warm_property_offset", solve)
    observed = []
    result, history = module.continue_warm_property_offset(
        p,
        f,
        0.1,
        0.95 * p.central_film_offset_m,
        seed=seed,
        seed_offset_m=p.central_film_offset_m,
        initial_step_m=0.05 * p.central_film_offset_m,
        callback=lambda r, s: observed.append(r["accepted"]),
    )
    assert result.central_film_offset_m == 0.95 * p.central_film_offset_m
    assert observed[0] is False and observed[-1] is True
    assert calls[1] == calls[2] == p.central_film_offset_m
    assert len(history) == len(observed)
    with pytest.raises(module.WarmContinuationError) as error:
        module.continue_warm_property_offset(
            p,
            f,
            0.1,
            0.8 * p.central_film_offset_m,
            seed=seed,
            seed_offset_m=p.central_film_offset_m,
            initial_step_m=0.01 * p.central_film_offset_m,
            maximum_attempts=1,
        )
    assert error.value.last_root.solution.audit["mass_maximum"] <= 1e-8


def test_signed_root_matches_independent_numerical_jacobian_reference():
    from ehl_solver.benchmark_200n import Circular200N
    from ehl_solver.property_reynolds import admissible, mass_audit
    from ehl_solver.property_warm import continue_warm_property_offset
    from examples.property_signed_reference import signed_reference

    c = Circular200N()
    p = c.problem(0.5e-6, ny=17, nx=25)
    f = c.fluid()
    seed = solve_property_coupling(p, f, 2.0)
    result, attempts = continue_warm_property_offset(
        p, f, 2.0, -0.2e-6, seed=seed, seed_offset_m=0.5e-6, initial_step_m=0.1e-6
    )
    s = result.solution
    assert result.central_film_offset_m < 0 and s.film_thickness_m.min() > p.film_floor_m
    status, (pressure, theta, h, w) = signed_reference(p, f, 2.0, -0.2e-6, s)
    audit = mass_audit(
        p.x_m,
        p.y_m,
        h,
        pressure,
        theta,
        f,
        2.0,
        length_scale_m=p.length_scale_m,
        film_scale_m=p.film_scale_m,
        pressure_scale_pa=p.pressure_scale_pa,
    )
    assert admissible(audit, pressure, theta, p.pressure_scale_pa), status
    assert np.linalg.norm(pressure - s.pressure_pa) / np.linalg.norm(pressure) < 1e-7
    np.testing.assert_allclose(h, s.film_thickness_m, rtol=1e-7)
    assert p.central_film_offset_m == 0.5e-6


def test_corrupt_boundary_seed_is_rejected(seed_case):
    p, f, seed = seed_case
    pressure = seed.pressure_pa.copy()
    pressure[0, 0] = 1.0
    with pytest.raises(ValueError, match="seed failed"):
        solve_warm_property_offset(
            p,
            f,
            0.1,
            p.central_film_offset_m,
            seed=replace(seed, pressure_pa=pressure),
            seed_offset_m=p.central_film_offset_m,
        )
