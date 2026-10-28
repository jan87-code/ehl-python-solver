"""Independent symbols and nonsmooth/boundary checks for diagnostic probes."""

from types import SimpleNamespace

import numpy as np
import pytest

from ehl_solver.cavitating_newton import CavitatingNewtonSystem
from ehl_solver.constitutive import IsothermalFluid
from ehl_solver.transport_audit import mass_direction_components
from ehl_solver.transport_probes import (
    active_front_probes,
    copy_content_boundary_direction,
    minimum_directional,
    resolved_pressure_probes,
)


def test_true_minimum_directional_and_exact_not_rounded_ties():
    a, b = np.array([0., 2., 1., 1.]), np.array([0., 1., 2., 1.])
    da, db = np.array([1., 4., -2., 3.]), np.array([2., -3., 5., 1.])
    expected = np.array([1., -3., -2., 1.])
    np.testing.assert_array_equal(minimum_directional(a, b, da, db), expected)
    step = 1e-6
    np.testing.assert_allclose((np.minimum(a+step*da, b+step*db)-np.minimum(a, b))/step,
                               expected, rtol=1e-9)
    assert minimum_directional(np.nextafter(1., 2.), 1., -20., 7.) == 7.
    with pytest.raises(ValueError):
        minimum_directional(np.nan, 0., 1., 2.)


def test_existing_merit_slope_uses_directional_tie_not_selected_model():
    # This checks the unchanged production rule on a non-root trial tie.
    system = SimpleNamespace(n=1, jacobian_vector=lambda z, v: np.array([0., v[1]]))
    state, direction, residual = np.array([.2, .2]), np.array([0., 1.]), np.array([0., .2])
    selected = system.jacobian_vector(state, direction)
    assert selected[1] == 1.
    assert minimum_directional(state[0], state[1], direction[0], direction[1]) == 0.
    assert CavitatingNewtonSystem.merit_slope(system, state, residual, direction) == 0.


@pytest.mark.parametrize("cells", [4, 8, 16])
def test_resolved_uniform_discrete_symbols_and_continuum_attenuation(cells):
    x, y = np.linspace(0., .001, 65), np.linspace(0., .0005, 9)
    h, p = np.full((9, 65), 2e-6), np.full((9, 65), 1e7)
    dx = x[1]-x[0]
    k = 2*np.pi/(cells*dx)
    dp = np.broadcast_to(1e6*np.cos(k*x), p.shape)
    fluid = IsothermalFluid(.04, 850.)
    terms = mass_direction_components(x, y, h, p, np.ones_like(p), fluid, .1,
                                      pressure_direction_pa=dp, film_direction_m=p*0,
                                      content_direction=p*0)
    rho, eta, dr, _ = fluid.properties(1e7)
    expected_diff = 4*rho*(2e-6)**3/(12*eta)*np.sin(k*dx/2)**2/dx**2*dp[1:-1, 1:-1]
    expected_adv = -.1*2e-6*dr*1e6*np.sin(k*x[1:-1])*np.sin(k*dx)/dx
    np.testing.assert_allclose(terms["pressure_diffusion"], expected_diff,
                               rtol=2e-12, atol=1e-11)
    np.testing.assert_allclose(terms["density_advection"],
                               np.broadcast_to(expected_adv, expected_diff.shape),
                               rtol=2e-12, atol=1e-11)
    assert 0 < np.sin(k*dx)/(k*dx) < 1
    assert 0 < 4*np.sin(k*dx/2)**2/(k*dx)**2 < 1


def test_physical_wavelength_sampling_force_edges_and_explicit_omissions():
    a = .00027
    x, y = np.linspace(-3*a, 3*a, 25), np.linspace(-3*a, 3*a, 17)
    xx, yy = np.meshgrid(x, y)
    p = 1e9*np.exp(-(xx**2+yy**2)/a**2)
    probes, omitted = resolved_pressure_probes(x, y, p, a)
    assert "physical_x_1a_cos" in probes
    assert "physical_y_1a_cos" not in probes
    assert len(omitted) == 5
    assert not probes["nyquist_x"]["metadata"]["resolved_carrier"]
    for item in probes.values():
        v = item["pressure_pa"]
        assert abs(v.sum()) < 2e-7
        np.testing.assert_array_equal(np.r_[v[0], v[-1], v[:, 0], v[:, -1]], 0.)
        if item["metadata"]["resolved_carrier"]:
            assert item["metadata"]["cells_per_wavelength"] >= 4*(1-1e-12)
    with pytest.raises(ValueError):
        resolved_pressure_probes(x, y, p, a, wavelength_factors=(1., 1.))


def front_fields():
    p, theta = np.zeros((7, 9)), np.ones((7, 9))
    p[1:-1, 1:5] = 1e7
    theta[1:-1, 5:] = .6
    return p, theta


def test_front_directions_are_one_sided_bound_feasible_without_mutation():
    p, theta = front_fields()
    original_p, original_t = p.copy(), theta.copy()
    probes, metadata = active_front_probes(p, theta, 1e9)
    assert len(probes) == 3
    assert metadata["missing_faces"] == ["reformation"]
    assert metadata["exact_interior_ties"] == 0
    assert metadata["selected_faces"]["rupture"]["full_yx"] == (3, 4)
    for item in probes.values():
        assert (p+item["pressure_pa"]).min() >= 0
        assert (theta+item["content"]).min() >= 0
        assert (theta+item["content"]).max() <= 1
        assert abs(item["pressure_pa"].sum()) < 1e-9
    np.testing.assert_array_equal(p, original_p)
    np.testing.assert_array_equal(theta, original_t)
    reverse, reversed_metadata = active_front_probes(p[:, ::-1], theta[:, ::-1], 1e9)
    assert reversed_metadata["missing_faces"] == ["rupture"]
    assert len(reverse) == 3
    for item in reverse.values():
        assert (p[:, ::-1]+item["pressure_pa"]).min() >= 0
        assert (theta[:, ::-1]+item["content"]).min() >= 0
        assert (theta[:, ::-1]+item["content"]).max() <= 1


def test_copied_content_boundaries_and_exact_tie_probe():
    dt = np.zeros((7, 9))
    dt[1, -2] = -.01
    changed = copy_content_boundary_direction(dt)
    assert changed[0, -1] == changed[1, -1] == changed[1, -2] == -.01
    np.testing.assert_array_equal(changed[:, 0], 0)
    assert dt[0, -1] == 0
    p, theta = front_fields()
    p[3, 3] = 0.
    probes, metadata = active_front_probes(p, theta, 1e9)
    assert metadata["exact_interior_ties"] == 1
    assert "exact_tie_content_loss" in probes


def test_uniform_no_front_and_invalid_content_are_explicit():
    p = np.full((7, 9), 1e7)
    probes, metadata = active_front_probes(p, np.ones_like(p), 1e9)
    assert not probes
    assert set(metadata["missing_faces"]) == {"rupture", "reformation"}
    with pytest.raises(ValueError):
        active_front_probes(p, np.full_like(p, -1.), 1e9)
    integer_p, integer_t = np.full((7, 9), 10_000_000), np.ones((7, 9), dtype=int)
    integer_p[:, 5:], integer_t[:, 5:] = 0, 0
    integer_probes, _ = active_front_probes(integer_p, integer_t, 1e9)
    assert integer_probes["rupture_full_content_loss"]["content"].dtype == np.float64
    assert abs(integer_probes["rupture_full_content_loss"]["content"]).max() == .01
