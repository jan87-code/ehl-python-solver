import numpy as np
import pytest

from examples.run_line_ehl_physical_diagnostics import _venner_case


def test_venner_1990_case_mapping_reproduces_moes_groups_and_scales():
    case, hertz, groups, inputs, h0_bracket = _venner_case(33)

    assert groups["M"] == pytest.approx(20.0, rel=2e-14)
    assert groups["L"] == pytest.approx(10.0, rel=2e-14)
    assert inputs["line_load_n_m"] == pytest.approx(
        groups["W"] * groups["E_prime_pa"] * inputs["radius_m"]
    )
    assert inputs["sum_surface_speed_m_s"] == pytest.approx(
        2.0 * case.entrainment_speed_m_s
    )
    assert case.length_scale_m == pytest.approx(hertz.half_width_m)
    assert case.film_scale_m == pytest.approx(
        hertz.half_width_m**2 / inputs["radius_m"]
    )
    assert case.elastic_reference_length_m == pytest.approx(
        0.1 * hertz.half_width_m
    )
    assert inputs["elastic_reference_length_over_b"] == pytest.approx(0.1)
    assert case.pressure_scale_pa == pytest.approx(hertz.peak_pressure_pa)
    assert case.x_m[[0, -1]] / hertz.half_width_m == pytest.approx([-4.0, 1.5])
    assert h0_bracket == pytest.approx(
        [0.1 * case.film_scale_m, 2.0 * case.film_scale_m]
    )


def test_venner_1990_case_matches_roelands_and_dowson_higginson_laws():
    case, hertz, _, inputs, _ = _venner_case(33)
    pressure_pa = hertz.peak_pressure_pa * np.array([0.0, 0.583, 0.875, 1.0])
    density, viscosity, _, _ = case.fluid.properties(pressure_pa)
    pressure_reference_pa = inputs["roelands_pressure_pa"]
    viscosity_index = inputs["viscosity_index"]
    alpha_per_pa = inputs["alpha_per_pa"]
    expected_viscosity_ratio = np.exp(
        alpha_per_pa
        * pressure_reference_pa
        / viscosity_index
        * ((1.0 + pressure_pa / pressure_reference_pa) ** viscosity_index - 1.0)
    )
    expected_density_ratio = 1.0 + (
        0.6e-9 * pressure_pa / (1.0 + 1.7e-9 * pressure_pa)
    )

    np.testing.assert_allclose(
        viscosity / case.fluid.viscosity_pa_s,
        expected_viscosity_ratio,
        rtol=2e-14,
        atol=0.0,
    )
    np.testing.assert_allclose(
        density / case.fluid.density_kg_m3,
        expected_density_ratio,
        rtol=2e-14,
        atol=0.0,
    )
