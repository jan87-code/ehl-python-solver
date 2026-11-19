import numpy as np
import pytest
from scipy.integrate import trapezoid

from ehl_solver.constitutive import IsothermalFluid
from ehl_solver.line_cavitation import (
    audit_variable_property_line_flux,
    compute_line_mass_flux_components,
    solve_cavitating_line_reynolds,
    solve_variable_property_line_reynolds,
)
from ehl_solver.line_contact import integrate_line_pressure


def _converging_diverging_film(size):
    x = np.linspace(0.0, 1.0, size)
    return x, 1.4 - 0.4 * x + 0.8 * x**2


def _independent_flux_residual(film, pressure, saturation, dx, reynolds_parameter):
    face_film = 0.5 * (film[:-1] + film[1:])
    face_film_cubed = 0.5 * (film[:-1] ** 3 + film[1:] ** 3)
    face_flux = (
        -face_film_cubed * np.diff(pressure) / dx
        + reynolds_parameter * face_film * saturation[:-1]
    )
    return np.diff(face_flux) / dx, face_flux


def test_full_film_line_solution_matches_independent_tridiagonal_system():
    x = np.linspace(0.0, 1.0, 41)
    film = 1.4 - 0.4 * x
    dx = x[1] - x[0]
    reynolds_parameter = 2.0
    interior = film[1:-1]
    interior_cubed = interior**3
    matrix = np.zeros((interior.size, interior.size))
    rhs = np.zeros(interior.size)

    for i in range(1, film.size - 1):
        row = i - 1
        h_east_cubed = 0.5 * (interior_cubed[row] + film[i + 1] ** 3)
        h_west_cubed = 0.5 * (film[i - 1] ** 3 + interior_cubed[row])
        matrix[row, row] = (h_east_cubed + h_west_cubed) / dx**2
        if row + 1 < interior.size:
            matrix[row, row + 1] = -h_east_cubed / dx**2
        if row > 0:
            matrix[row, row - 1] = -h_west_cubed / dx**2
        h_east = 0.5 * (film[i] + film[i + 1])
        h_west = 0.5 * (film[i - 1] + film[i])
        rhs[row] = -reynolds_parameter * (h_east - h_west) / dx

    solution = solve_cavitating_line_reynolds(
        film, dx, reynolds_parameter
    )

    np.testing.assert_allclose(solution.pressure[1:-1], np.linalg.solve(matrix, rhs))
    np.testing.assert_array_equal(solution.pressure[[0, -1]], 0.0)
    np.testing.assert_allclose(solution.fractional_film_content, 1.0)
    assert solution.active_set_iterations == 1
    assert solution.maximum_complementarity_violation == 0.0
    assert solution.maximum_mass_balance_residual < 1e-9
    assert abs(solution.net_boundary_flux) < 1e-10


def test_line_jfo_is_admissible_locally_conservative_and_load_auditable():
    x, film = _converging_diverging_film(41)
    dx = x[1] - x[0]
    solution = solve_cavitating_line_reynolds(film, dx, 2.0)
    residual, flux = _independent_flux_residual(
        film,
        solution.pressure,
        solution.fractional_film_content,
        dx,
        2.0,
    )

    assert solution.pressure.shape == film.shape
    assert solution.fractional_film_content.shape == film.shape
    np.testing.assert_array_equal(solution.pressure[[0, -1]], 0.0)
    assert solution.fractional_film_content[0] == 1.0
    assert 0.0 <= solution.pressure.min()
    assert 0.0 <= solution.fractional_film_content.min()
    assert solution.fractional_film_content.max() <= 1.0
    assert np.any(solution.fractional_film_content[1:-1] < 1.0 - 1e-8)
    assert np.all(
        solution.pressure[1:-1][solution.fractional_film_content[1:-1] < 1.0 - 1e-8]
        == 0.0
    )
    assert solution.maximum_complementarity_violation < 1e-10
    assert solution.maximum_mass_balance_residual < 1e-10
    np.testing.assert_allclose(residual, 0.0, atol=1e-10)
    assert flux[0] == pytest.approx(-solution.west_boundary_flux, abs=1e-14)
    assert flux[-1] == pytest.approx(solution.east_boundary_flux, abs=1e-14)
    assert abs(solution.net_boundary_flux) < 1e-10

    length_scale_m = 0.01
    pressure_scale_pa = 1e6
    x_m = length_scale_m * x
    pressure_pa = pressure_scale_pa * solution.pressure
    line_load_n_m = integrate_line_pressure(pressure_pa, x_m).line_load_n_m
    independent_line_load_n_m = trapezoid(pressure_pa, x_m)
    assert line_load_n_m == pytest.approx(independent_line_load_n_m, rel=2e-14)
    assert line_load_n_m == pytest.approx(
        pressure_scale_pa * length_scale_m * trapezoid(solution.pressure, x),
        rel=2e-14,
    )


def test_line_jfo_pressure_integral_refines_on_converging_diverging_film():
    pressure_integrals = []
    for size in (21, 41, 81):
        x, film = _converging_diverging_film(size)
        solution = solve_cavitating_line_reynolds(film, x[1] - x[0], 2.0)
        pressure_integrals.append(trapezoid(solution.pressure, x))
        assert solution.maximum_mass_balance_residual < 1e-9
        assert abs(solution.net_boundary_flux) < 1e-9

    changes = np.abs(np.diff(pressure_integrals))
    assert pressure_integrals[0] < pressure_integrals[1] < pressure_integrals[2]
    assert changes[1] < changes[0]


def test_line_jfo_inlet_starvation_reduces_pressure_and_conserves_flux():
    x, film = _converging_diverging_film(41)
    dx = x[1] - x[0]
    flooded = solve_cavitating_line_reynolds(film, dx, 2.0)
    starved = solve_cavitating_line_reynolds(
        film, dx, 2.0, inlet_fractional_film_content=0.99
    )

    assert starved.inlet_fractional_film_content == 0.99
    assert starved.fractional_film_content[0] == 0.99
    assert trapezoid(starved.pressure, x) < trapezoid(flooded.pressure, x)
    assert starved.pressure.max() < flooded.pressure.max()
    assert starved.maximum_mass_balance_residual < 1e-10
    assert abs(starved.net_boundary_flux) < 1e-10


def test_variable_property_line_jfo_recovers_constant_property_solution():
    x_hat, film_hat = _converging_diverging_film(41)
    length_scale_m = 0.01
    film_scale_m = 1e-4
    speed_m_s = 0.1
    viscosity_pa_s = 0.04
    reynolds_parameter = 2.0
    pressure_scale_pa = (
        12
        * viscosity_pa_s
        * speed_m_s
        * length_scale_m
        / (film_scale_m**2 * reynolds_parameter)
    )
    x_m = length_scale_m * x_hat
    film_m = film_scale_m * film_hat
    fluid = IsothermalFluid(viscosity_pa_s, 850.0)
    constant = solve_cavitating_line_reynolds(
        film_hat,
        x_hat[1] - x_hat[0],
        reynolds_parameter,
    )

    variable = solve_variable_property_line_reynolds(
        x_m,
        film_m,
        fluid,
        speed_m_s,
        length_scale_m=length_scale_m,
        film_scale_m=film_scale_m,
        pressure_scale_pa=pressure_scale_pa,
        strengths=(0.0,),
    )

    np.testing.assert_allclose(
        variable.pressure_pa / pressure_scale_pa,
        constant.pressure,
        rtol=2e-12,
        atol=2e-14,
    )
    np.testing.assert_allclose(
        variable.fractional_film_content,
        constant.fractional_film_content,
        rtol=0.0,
        atol=2e-12,
    )
    assert variable.audit["mass_maximum"] < 1e-10
    assert variable.audit["maximum_complementarity"] < 1e-12


def test_pressure_dependent_line_jfo_passes_independent_si_flux_audit():
    x_hat, film_hat = _converging_diverging_film(81)
    length_scale_m = 0.01
    film_scale_m = 1e-4
    pressure_scale_pa = 1e6
    speed_m_s = 0.1
    fluid = IsothermalFluid(
        viscosity_pa_s=0.04,
        density_kg_m3=850.0,
        viscosity_index=0.68,
    )
    x_m = length_scale_m * x_hat
    film_m = film_scale_m * film_hat
    result = solve_variable_property_line_reynolds(
        x_m,
        film_m,
        fluid,
        speed_m_s,
        length_scale_m=length_scale_m,
        film_scale_m=film_scale_m,
        pressure_scale_pa=pressure_scale_pa,
    )
    independent = audit_variable_property_line_flux(
        x_m,
        film_m,
        result.pressure_pa,
        result.fractional_film_content,
        fluid,
        speed_m_s,
        length_scale_m=length_scale_m,
        film_scale_m=film_scale_m,
        pressure_scale_pa=pressure_scale_pa,
    )

    assert np.all(result.pressure_pa >= 0.0)
    assert np.all((0.0 <= result.fractional_film_content) & (
        result.fractional_film_content <= 1.0
    ))
    np.testing.assert_array_equal(result.pressure_pa[[0, -1]], 0.0)
    assert independent["mass_maximum"] < 1e-8
    assert independent["relative_local_balance"] < 1e-10
    assert independent["relative_global_balance"] < 1e-10
    assert independent["maximum_complementarity"] < 1e-10
    assert result.history[-1]["strength"] == 1.0


def test_line_mass_flux_components_reconstruct_uniform_fully_flooded_flow():
    x = np.linspace(-1e-4, 1e-4, 17)
    film = np.full_like(x, 2e-6)
    pressure = np.zeros_like(x)
    theta = np.ones_like(x)
    fluid = IsothermalFluid(viscosity_pa_s=0.04, density_kg_m3=850.0)
    entrainment_speed_m_s = 0.1

    flux = compute_line_mass_flux_components(
        x,
        film,
        pressure,
        theta,
        fluid,
        entrainment_speed_m_s,
    )
    expected_flux = (
        fluid.density_kg_m3 * film[0] * entrainment_speed_m_s
    )

    np.testing.assert_array_equal(flux.pressure_driven_kg_m_s, 0.0)
    np.testing.assert_allclose(flux.entrainment_kg_m_s, expected_flux)
    np.testing.assert_allclose(flux.total_kg_m_s, expected_flux)
    np.testing.assert_allclose(flux.face_x_m, 0.5 * (x[:-1] + x[1:]))
    np.testing.assert_allclose(flux.density_kg_m3, fluid.density_kg_m3)
    np.testing.assert_allclose(flux.viscosity_pa_s, fluid.viscosity_pa_s)

    inlet_starved_theta = np.ones_like(x)
    inlet_starved_theta[0] = 0.5
    starved_flux = compute_line_mass_flux_components(
        x,
        film,
        pressure,
        inlet_starved_theta,
        fluid,
        entrainment_speed_m_s,
    )
    assert starved_flux.entrainment_kg_m_s[0] == pytest.approx(
        0.5 * expected_flux
    )
    assert starved_flux.entrainment_kg_m_s[1] == pytest.approx(expected_flux)

    increasing_pressure = np.linspace(0.0, 1e6, x.size)
    pressure_gradient_flux = compute_line_mass_flux_components(
        x,
        film,
        increasing_pressure,
        theta,
        fluid,
        entrainment_speed_m_s,
    )
    assert np.all(pressure_gradient_flux.pressure_driven_kg_m_s < 0.0)
    np.testing.assert_allclose(
        pressure_gradient_flux.total_kg_m_s,
        pressure_gradient_flux.pressure_driven_kg_m_s
        + pressure_gradient_flux.entrainment_kg_m_s,
    )


def test_line_mass_flux_diagnostics_allow_only_pressure_roundoff():
    x = np.linspace(-1e-4, 1e-4, 17)
    film = np.full_like(x, 2e-6)
    pressure = np.zeros_like(x)
    pressure[8] = -1e-19
    theta = np.ones_like(x)
    fluid = IsothermalFluid(viscosity_pa_s=0.04, density_kg_m3=850.0)

    flux = compute_line_mass_flux_components(
        x, film, pressure, theta, fluid, 0.1
    )
    assert np.all(np.isfinite(flux.total_kg_m_s))

    pressure[8] = -1e-10
    with pytest.raises(ValueError, match="finite admissible SI fields"):
        compute_line_mass_flux_components(
            x, film, pressure, theta, fluid, 0.1
        )


def test_uniform_line_film_has_zero_pressure_and_constant_fully_flooded_flux():
    x_m = np.linspace(-1e-4, 1e-4, 41)
    film_m = np.full_like(x_m, 2e-6)
    fluid = IsothermalFluid(viscosity_pa_s=0.04, density_kg_m3=850.0)

    solution = solve_variable_property_line_reynolds(
        x_m,
        film_m,
        fluid,
        0.1,
        length_scale_m=1e-4,
        film_scale_m=2e-6,
        pressure_scale_pa=1e6,
    )

    np.testing.assert_array_equal(solution.pressure_pa, 0.0)
    np.testing.assert_array_equal(solution.fractional_film_content, 1.0)
    assert solution.audit["relative_local_balance"] == 0.0
    assert solution.audit["relative_global_balance"] == 0.0


def test_variable_property_line_solution_is_invariant_to_pressure_reference_scale():
    x_hat, film_hat = _converging_diverging_film(41)
    length_scale_m = 0.01
    film_scale_m = 1e-4
    x_m = length_scale_m * x_hat
    film_m = film_scale_m * film_hat
    fluid = IsothermalFluid(viscosity_pa_s=0.04, density_kg_m3=850.0)
    solutions = [
        solve_variable_property_line_reynolds(
            x_m,
            film_m,
            fluid,
            0.1,
            length_scale_m=length_scale_m,
            film_scale_m=film_scale_m,
            pressure_scale_pa=pressure_scale_pa,
        )
        for pressure_scale_pa in (1e6, 5e6)
    ]

    np.testing.assert_allclose(
        solutions[0].pressure_pa,
        solutions[1].pressure_pa,
        rtol=2e-7,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        solutions[0].fractional_film_content,
        solutions[1].fractional_film_content,
        rtol=0.0,
        atol=2e-8,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dx", 0.0),
        ("reynolds_parameter", 0.0),
        ("inlet_fractional_film_content", -0.01),
        ("inlet_fractional_film_content", 1.01),
        ("pressure_tolerance", float("nan")),
        ("saturation_tolerance", -1.0),
        ("max_active_set_iterations", 0),
        ("equilibrate_linear_system", "false"),
    ],
)
def test_line_jfo_rejects_invalid_inputs(field, value):
    arguments = {
        "dx": 0.25,
        "reynolds_parameter": 1.0,
    }
    arguments[field] = value
    with pytest.raises(ValueError):
        solve_cavitating_line_reynolds(np.ones(9), **arguments)
