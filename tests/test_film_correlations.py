import pytest

from ehl_solver.film_correlations import hamrock_dowson_point_contact

BASE = dict(
    load_n=50.0,
    radius_x_m=0.015,
    contact_modulus_pa=113e9,
    viscosity_pa_s=0.04,
    entrainment_m_s=2.0,
    alpha_per_pa=2.21e-8,
)


def test_si_film_reference_uses_hd_prime_modulus():
    result = hamrock_dowson_point_contact(**BASE)
    assert result.elastic_modulus_prime_pa == 226e9
    assert result.minimum_film_m == pytest.approx(2.8414972979824044e-7, rel=2e-14)
    assert result.central_film_m == pytest.approx(4.97969177492642e-7, rel=2e-14)
    assert result.speed_parameter == pytest.approx(0.08 / (226e9 * 0.015))
    # Historical supplied values used E_contact in place of E' in U,G,W.
    old_convention = hamrock_dowson_point_contact(**{**BASE, "contact_modulus_pa": 56.5e9})
    assert result.minimum_film_m / old_convention.minimum_film_m == pytest.approx(2 ** (-0.117))
    assert result.central_film_m / old_convention.central_film_m == pytest.approx(2 ** (-0.073))


@pytest.mark.parametrize(
    "key,factor,minimum_power,central_power",
    [
        ("load_n", 4.0, -0.073, -0.067),
        ("entrainment_m_s", 2.0, 0.68, 0.67),
        ("viscosity_pa_s", 3.0, 0.68, 0.67),
        ("alpha_per_pa", 0.8978125, 0.49, 0.53),
        ("radius_x_m", 2.0, 0.466, 0.464),
    ],
)
def test_independent_dimensional_power_laws(key, factor, minimum_power, central_power):
    first = hamrock_dowson_point_contact(**BASE)
    second = hamrock_dowson_point_contact(**{**BASE, key: BASE[key] * factor})
    assert second.minimum_film_m / first.minimum_film_m == pytest.approx(factor**minimum_power)
    assert second.central_film_m / first.central_film_m == pytest.approx(factor**central_power)


@pytest.mark.parametrize(
    "key,value",
    [
        ("load_n", 0),
        ("contact_modulus_pa", -1),
        ("entrainment_m_s", float("nan")),
        ("alpha_per_pa", float("inf")),
        ("ellipticity", 0.9),
        ("ellipticity", 8.1),
    ],
)
def test_invalid_or_outside_fit_inputs_raise(key, value):
    with pytest.raises(ValueError):
        hamrock_dowson_point_contact(**{**BASE, key: value})


@pytest.mark.parametrize("fraction", [0.8978125, 1.0])
def test_matched_fluid_is_checkpoint_roelands_not_joint_property_strength(fraction):
    import numpy as np

    from examples.compare_conservative_50n import matched_fluid

    p = np.array([0.0, 1e8, 1e9])
    fluid = matched_fluid(fraction)
    rho, eta, _, _ = fluid.ratios(p)
    z = fraction * 2.21e-8 * 1.98e8 / (np.log(0.04) + 9.67)
    expected_eta = np.exp((np.log(0.04) + 9.67) * ((1 + p / 1.98e8) ** z - 1))
    np.testing.assert_allclose(eta, expected_eta, rtol=1e-13)
    np.testing.assert_allclose(rho, 1 + 0.6e-9 * p / (1 + 1.7e-9 * p), rtol=1e-15)
    ambient_alpha = (
        fluid.viscosity_index
        * np.log(0.04 / fluid.viscosity_reference_pa_s)
        / fluid.roelands_pressure_pa
    )
    assert ambient_alpha == pytest.approx(fraction * 2.21e-8)


def test_film_comparison_uses_physical_centre_on_refined_grid():
    import numpy as np

    from examples.compare_conservative_50n import film_comparison

    film = np.full((65, 65), 500e-9)
    film[32, 32] = 600e-9
    result = film_comparison(film, 1.0)
    assert result["sampled_central_m"] == 600e-9
    assert result["sampled_minimum_m"] == 500e-9
    assert result["central_relative_difference"] == pytest.approx(
        600e-9 / result["central_film_m"] - 1
    )
