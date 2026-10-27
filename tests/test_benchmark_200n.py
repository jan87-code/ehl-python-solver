import numpy as np
import pytest

from ehl_solver.benchmark_200n import Circular200N


def test_supplied_hertz_and_modulus_conventions():
    c = Circular200N()
    assert c.hertz_radius_m == pytest.approx(0.271041e-3, rel=2e-6)
    assert c.hertz_approach_m == pytest.approx(4.897541e-6, rel=2e-6)
    assert c.hertz_pressure_pa == pytest.approx(1299e6, rel=0.001)
    d = c.parameters()
    assert d["dimensionless_contact_convention"] == pytest.approx(
        dict(U=4.7198e-11, W=7.8663e-6, G=2497.3), rel=1e-5
    )
    e = d["dimensionless_E_prime_convention"]
    assert e["G"] == pytest.approx(4994.6)
    assert e["W"] == pytest.approx(3.9331366765e-6)


def test_alpha_is_ambient_slope_and_not_default_roelands_index():
    c = Circular200N()
    f = c.fluid()
    rho, eta, _, derivative = f.properties(0.0)
    assert eta == pytest.approx(0.04)
    assert rho == pytest.approx(850)
    assert derivative / eta == pytest.approx(c.alpha_per_pa, rel=1e-14)
    # Independent central difference, without using the analytic derivative.
    v_plus = f.properties(100.0)[1]
    v_minus = f.properties(-100.0)[1]
    assert (np.log(v_plus) - np.log(v_minus)) / 200 == pytest.approx(c.alpha_per_pa, rel=1e-8)
    assert f.viscosity_index != 0.68


def test_fixed_physical_grid_and_scales_when_offset_changes():
    c = Circular200N()
    p, q = c.problem(5e-6), c.problem(0.5e-6)
    np.testing.assert_array_equal(p.x_m, q.x_m)
    np.testing.assert_array_equal(p.y_m, q.y_m)
    assert p.x_m[len(p.x_m) // 2] == 0
    assert p.y_m[len(p.y_m) // 2] == 0
    assert p.shape == (17, 25)
    assert p.reynolds_parameter == q.reynolds_parameter
    assert p.e_star_pa == 113e9
    assert p.length_scale_m == q.length_scale_m == c.hertz_radius_m
    assert p.film_scale_m == q.film_scale_m == c.hertz_approach_m


@pytest.mark.parametrize("kwargs", [dict(nx=8), dict(ny=1), dict(extent=1), dict(nx=201, ny=201)])
def test_invalid_grid_rejected_before_allocation(kwargs):
    with pytest.raises(ValueError):
        Circular200N().problem(1e-6, **kwargs)
