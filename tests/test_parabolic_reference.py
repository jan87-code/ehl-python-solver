import numpy as np
import pytest
from scipy.integrate import quad

from ehl_solver.parabolic_reference import parabolic_jfo_reference

H0, RADIUS, VISCOSITY, SPEED = 20e-6, 0.01, 0.04, 0.425
LENGTH = np.sqrt(2 * RADIUS * H0)


def reference(inlet=-3, count=65):
    return parabolic_jfo_reference(
        np.linspace(inlet * LENGTH, 3 * LENGTH, count), H0, RADIUS, VISCOSITY, SPEED
    )


def test_pressure_primitive_force_and_rupture_match_independent_quadrature():
    result = reference()
    hc = H0 + result.rupture_location_m**2 / (2 * RADIUS)

    def derivative(x):
        h = H0 + x * x / (2 * RADIUS)
        return 12 * VISCOSITY * SPEED * (h - hc) / h**3

    for x, p in zip(result.x_m, result.pressure_pa, strict=True):
        expected = (
            quad(derivative, result.x_m[0], x, epsabs=1e-7)[0]
            if x < result.rupture_location_m
            else 0
        )
        assert p == pytest.approx(expected, rel=1e-10, abs=1e-7)
    rupture_pressure = quad(derivative, result.x_m[0], result.rupture_location_m, epsabs=1e-7)[0]
    assert abs(rupture_pressure) < 1e-6
    force = quad(
        lambda x: quad(derivative, result.x_m[0], x, epsabs=1e-7)[0],
        result.x_m[0],
        result.rupture_location_m,
        epsabs=1e-7,
    )[0]
    assert result.force_per_width_n_m == pytest.approx(force, rel=1e-10)


def test_pressure_content_and_flux_are_admissible_and_mass_conserving():
    result = reference()
    flux = (
        -(result.film_thickness_m**3) / (12 * VISCOSITY) * result.pressure_gradient_pa_per_m
        + SPEED * result.film_thickness_m * result.fractional_film_content
    )
    np.testing.assert_allclose(flux, result.volume_flux_per_width_m2_s, rtol=2e-14)
    assert result.pressure_pa.min() >= -1e-8
    assert result.fractional_film_content.min() > 0
    assert result.fractional_film_content.max() == 1
    np.testing.assert_allclose(result.pressure_pa * (1 - result.fractional_film_content), 0)
    assert result.pressure_pa[0] == result.pressure_pa[-1] == 0


def test_finite_inlet_influence_decreases_as_inlet_moves_upstream():
    values = [reference(inlet) for inlet in (-0.5, -1, -2, -4, -8, -16)]
    forces = np.array([r.force_per_width_n_m for r in values])
    assert np.all(np.diff(forces) > 0)
    assert forces[-1] - forces[-2] < forces[-2] - forces[-3]
    assert 0.46 < values[-1].rupture_location_m / LENGTH < 0.49


@pytest.mark.parametrize(
    "changes",
    [
        {"minimum_film_m": 0},
        {"entrainment_speed_m_s": -1},
        {"x_m": np.array([0.0, LENGTH, 2 * LENGTH])},
        {"x_m": np.array([-LENGTH, 0, 1e-4 * LENGTH])},
    ],
)
def test_invalid_reference_inputs_are_rejected(changes):
    arguments = dict(
        x_m=np.linspace(-3 * LENGTH, 3 * LENGTH, 65),
        minimum_film_m=H0,
        radius_m=RADIUS,
        viscosity_pa_s=VISCOSITY,
        entrainment_speed_m_s=SPEED,
    )
    with pytest.raises(ValueError):
        parabolic_jfo_reference(**(arguments | changes))


def test_wide_two_dimensional_jfo_centerline_converges_to_independent_reference():
    from examples.run_parabolic_cavitation_reference import compare_grid

    cases = [compare_grid(nx) for nx in (33, 65, 129)]
    assert (
        cases[2]["pressure_relative_l2"]
        < cases[1]["pressure_relative_l2"]
        < cases[0]["pressure_relative_l2"]
    )
    assert (
        cases[2]["relative_force_error"]
        < cases[1]["relative_force_error"]
        < cases[0]["relative_force_error"]
    )
    assert cases[-1]["pressure_relative_l2"] < 0.02
    assert cases[-1]["relative_force_error"] < 0.02
    assert cases[-1]["theta_maximum_absolute_error"] < 0.03
    for case in cases:
        assert case["mass_maximum"] < 1e-8
        assert case["complementarity_maximum"] < 1e-10
        spacing = (case["x_limits_m"][1] - case["x_limits_m"][0]) / (case["shape"][1] - 1)
        assert abs(case["first_cavitated_node_m"] - case["analytical_rupture_m"]) < 2 * spacing
