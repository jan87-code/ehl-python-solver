import numpy as np
import pytest

from ehl_solver.cavitating_coupled import CavitatingElasticityProblem
from ehl_solver.cavitating_newton import solve_cavitating_newton
from ehl_solver.exterior_coupled import ExteriorCouplingError, solve_uniform_exterior_coupling
from examples.run_farfield_budget import ETA, H0, LY, P_REF, RX, RY, SPEED, L


def problem():
    u = np.linspace(-4, 4, 17)
    return CavitatingElasticityProblem(u * L, u * LY, H0, RX, RY, 1e9, L, H0, P_REF, 1)


def test_exterior_nonlinear_iteration_matches_independent_monolithic_root():
    case = problem()
    ref = solve_cavitating_newton(case).solution
    actual = solve_uniform_exterior_coupling(case, ETA, SPEED, core_bounds_m=(-L, L, -LY, LY))
    p = actual.solution
    assert p.compliance_factor == 1
    assert np.linalg.norm(p.pressure_pa - ref.pressure_pa) / np.linalg.norm(ref.pressure_pa) < 1e-8
    assert p.load_n == pytest.approx(ref.load_n, rel=1e-8)
    np.testing.assert_allclose(p.film_thickness_m, ref.film_thickness_m, rtol=1e-9)
    assert actual.audit["mass_maximum"] <= 1e-8
    assert actual.audit["relative_local_balance"] <= 1e-10
    assert actual.audit["relative_global_balance"] <= 1e-10
    assert actual.audit["maximum_complementarity"] <= 1e-10
    assert p.film_thickness_m.min() > case.film_floor_m
    assert max(h.iteration for h in p.history) <= 100


def test_exterior_coupling_retains_failure_and_rejects_changed_flow_model():
    case = problem()
    with pytest.raises(ExteriorCouplingError, match="exhausted") as failure:
        solve_uniform_exterior_coupling(
            case, ETA, SPEED, core_bounds_m=(-L, L, -LY, LY), maximum_iterations_per_stage=1
        )
    assert failure.value.history
    assert failure.value.exterior_history
    assert failure.value.last_solution is not None
    assert failure.value.last_solution.compliance_factor == 0
    with pytest.raises(ValueError, match="Reynolds parameter"):
        solve_uniform_exterior_coupling(case, ETA, 2 * SPEED, core_bounds_m=(-L, L, -LY, LY))
