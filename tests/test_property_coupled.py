import numpy as np
import pytest

from ehl_solver.cavitating_coupled import CavitatingElasticityProblem
from ehl_solver.cavitating_newton import solve_cavitating_newton
from ehl_solver.constitutive import IsothermalFluid
from ehl_solver.property_coupled import PropertyCouplingError, solve_property_coupling
from ehl_solver.property_reynolds import admissible, mass_audit
from examples.property_monolithic_reference import small_monolithic_reference
from examples.run_farfield_budget import H0, LY, P_REF, RX, RY, L


def case():
    x = np.linspace(-4, 6, 13) * L
    y = np.linspace(-4, 4, 9) * LY
    return CavitatingElasticityProblem(x, y, H0, RX, RY, 1e11, L, H0, P_REF * 100, 1)


def test_coupled_constant_property_limit_and_retained_failure():
    p = case()
    fluid = IsothermalFluid(0.04, 850)
    reference = solve_cavitating_newton(p).solution
    r = solve_property_coupling(p, fluid, 0.1, constitutive_strength=0)
    assert (
        np.linalg.norm(r.pressure_pa - reference.pressure_pa)
        / np.linalg.norm(reference.pressure_pa)
        < 1e-8
    )
    assert r.load_n == pytest.approx(reference.load_n, rel=1e-8)
    with pytest.raises(PropertyCouplingError) as error:
        solve_property_coupling(p, fluid, 0.1, maximum_iterations_per_stage=1)
    assert error.value.history and error.value.last_solution is not None


def test_pressure_dependent_coupling_matches_independent_complete_nonlinear_root():
    p = case()
    fluid = IsothermalFluid(0.04, 850)
    r = solve_property_coupling(p, fluid, 0.1)
    status, (pressure, theta, h, _) = small_monolithic_reference(p, fluid, 0.1)
    audit = mass_audit(
        p.x_m,
        p.y_m,
        h,
        pressure,
        theta,
        fluid,
        0.1,
        length_scale_m=L,
        film_scale_m=H0,
        pressure_scale_pa=P_REF * 100,
    )
    assert admissible(audit, pressure, theta, P_REF * 100), status
    assert np.linalg.norm(r.pressure_pa - pressure) / np.linalg.norm(pressure) < 1e-8
    assert r.load_n == pytest.approx(p.load_n(pressure), rel=1e-8)
    np.testing.assert_allclose(r.film_thickness_m, h, rtol=1e-9)
