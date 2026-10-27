import numpy as np
import pytest

from ehl_solver.cavitation import solve_cavitating_reynolds
from ehl_solver.constitutive import IsothermalFluid
from ehl_solver.property_reynolds import (
    PropertyConvergenceError,
    mass_flux_direction,
    mass_fluxes,
    solve_property_reynolds,
)
from examples.property_channel_reference import pressure_driven_channel
from examples.run_farfield_budget import H0, LY, P_REF, RX, RY, L


@pytest.mark.parametrize("supply", [1.0, 0.9])
def test_constant_property_limit_matches_independent_jfo(supply):
    u = np.linspace(-4, 4, 17)
    x, y = u * L, u * LY
    h = H0 + x[None, :] ** 2 / (2 * RX) + y[:, None] ** 2 / (2 * RY)
    ref = solve_cavitating_reynolds(
        h / H0, 0.5, 0.5 * LY / L, 1, inlet_fractional_film_content=supply
    )
    r = solve_property_reynolds(
        x,
        y,
        h,
        IsothermalFluid(0.04, 850),
        0.001,
        length_scale_m=L,
        film_scale_m=H0,
        pressure_scale_pa=P_REF,
        strengths=(0,),
        inlet_content=supply,
    )
    np.testing.assert_allclose(r.pressure_pa / P_REF, ref.pressure, atol=1e-12)
    np.testing.assert_allclose(r.fractional_film_content, ref.fractional_film_content, atol=1e-12)
    assert r.audit["mass_maximum"] < 1e-8
    assert r.audit["relative_global_balance"] < 1e-10


def test_pressure_dependent_channel_converges_to_independent_integral_reference():
    fluid = IsothermalFluid(0.04, 850)
    errors = []
    for nx in (17, 33, 65):
        x = np.linspace(0, 0.001, nx)
        y = np.linspace(-0.001, 0.001, 5)
        exact, flux = pressure_driven_channel(x, 20e-6, fluid, 5e6, 1e6)
        h = np.full((5, nx), 20e-6)
        bc = np.broadcast_to(exact, h.shape)
        r = solve_property_reynolds(
            x,
            y,
            h,
            fluid,
            0,
            length_scale_m=0.001,
            film_scale_m=20e-6,
            pressure_scale_pa=5e6,
            boundary_pressure_pa=bc,
        )
        errors.append(np.linalg.norm(r.pressure_pa[2] - exact) / np.linalg.norm(exact))
        qx, _ = mass_fluxes(x, y, h, r.pressure_pa, r.fractional_film_content, fluid, 0)
        assert abs(qx[2].mean() / flux - 1) < 0.001
        assert r.audit["mass_maximum"] < 1e-8
        assert r.audit["relative_local_balance"] < 1e-10
    assert errors[2] < errors[1] / 3 < errors[0] / 9


def test_variable_mass_flux_complete_smooth_direction_matches_finite_difference():
    rng = np.random.default_rng(31)
    x = np.linspace(-0.001, 0.001, 11)
    y = np.linspace(-0.0005, 0.0005, 9)
    h = np.full((9, 11), 20e-6)
    p = 5e6 + 1e6 * np.exp(-((x[None, :] / 0.001) ** 2) - (y[:, None] / 0.0005) ** 2)
    theta = np.full_like(h, 0.8)
    dh = rng.normal(size=h.shape) * 1e-6
    dp = rng.normal(size=h.shape) * 1e6
    dt = rng.normal(size=h.shape) * 0.1
    fluid = IsothermalFluid(0.04, 850)
    analytic = mass_flux_direction(x, y, h, p, theta, dh, dp, dt, fluid, 0.1)
    eps = 1e-5
    a = mass_fluxes(x, y, h + eps * dh, p + eps * dp, theta + eps * dt, fluid, 0.1)
    b = mass_fluxes(x, y, h - eps * dh, p - eps * dp, theta - eps * dt, fluid, 0.1)
    for actual, plus, minus in zip(analytic, a, b, strict=True):
        fd = (plus - minus) / (2 * eps)
        assert np.linalg.norm(actual - fd) / np.linalg.norm(actual) < 1e-8


def test_property_failure_retains_candidate_and_never_raises_iteration_cap():
    x = np.linspace(-L, L, 9)
    y = np.linspace(-LY, LY, 9)
    h = H0 + x[None, :] ** 2 / (2 * RX) + y[:, None] ** 2 / (2 * RY)
    with pytest.raises(PropertyConvergenceError) as exc:
        solve_property_reynolds(
            x,
            y,
            h,
            IsothermalFluid(0.04, 850),
            0.001,
            length_scale_m=L,
            film_scale_m=H0,
            pressure_scale_pa=P_REF,
            maximum_iterations_per_stage=1,
        )
    assert exc.value.last_solution is not None and exc.value.history


def test_constant_high_pressure_transport_uses_liquid_density_and_si_mass_units():
    fluid = IsothermalFluid(0.04, 850)
    x = np.linspace(0, 0.001, 7)
    y = np.linspace(-0.001, 0.001, 5)
    h = np.full((5, 7), 20e-6)
    p = np.full_like(h, 1e8)
    r = solve_property_reynolds(
        x,
        y,
        h,
        fluid,
        0.1,
        length_scale_m=0.001,
        film_scale_m=20e-6,
        pressure_scale_pa=1e8,
        boundary_pressure_pa=p,
        initial_pressure_pa=p,
    )
    np.testing.assert_allclose(r.pressure_pa, p, rtol=1e-13)
    rho = fluid.properties(1e8)[0]
    expected = rho * 0.1 * 20e-6 * (y[-1] + y[-2] - y[0] - y[1]) / 2
    assert r.audit["outward_mass_flow_kg_s"][0] == pytest.approx(-expected, rel=1e-12)
    assert r.audit["outward_mass_flow_kg_s"][1] == pytest.approx(expected, rel=1e-12)
