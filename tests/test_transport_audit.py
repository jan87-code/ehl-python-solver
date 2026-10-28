"""Independent FD/analytical checks for a diagnostic, not a modified solver."""

import numpy as np
import pytest

from ehl_solver.constitutive import IsothermalFluid
from ehl_solver.property_reynolds import mass_fluxes
from ehl_solver.transport_audit import (
    mass_direction_components,
    nyquist_pressure_projection,
    pressure_mode_directions,
)


def test_complete_si_decomposition_matches_independent_flux_differences():
    x, y = np.linspace(-1e-3, 1e-3, 13), np.linspace(-.7e-3, .7e-3, 9)
    xx, yy = np.meshgrid(x/1e-3, y/1e-3)
    h = 2e-6*(1+.1*xx+.05*yy**2)
    p = 1e8*(1+.2*np.cos(xx)*np.cos(yy))
    theta = .8+.1*xx
    dp, dh, dt = 1e6*np.sin(xx+yy), .02*h*np.cos(xx), .03*np.sin(yy)
    fluid = IsothermalFluid(.04, 850.)
    terms = mass_direction_components(x, y, h, p, theta, fluid, .4,
                                      pressure_direction_pa=dp, film_direction_m=dh,
                                      content_direction=dt)
    assert len(terms) == 7
    step = 1e-4
    plus = mass_fluxes(x, y, h+step*dh, p+step*dp, theta+step*dt, fluid, .4)
    minus = mass_fluxes(x, y, h-step*dh, p-step*dp, theta-step*dt, fluid, .4)
    dx, dy = x[1]-x[0], y[1]-y[0]
    expected = ((np.diff(plus[0][1:-1]-minus[0][1:-1], axis=1)/dx
                 + np.diff(plus[1][:, 1:-1]-minus[1][:, 1:-1], axis=0)/dy)/(2*step))
    np.testing.assert_allclose(sum(terms.values()), expected, rtol=3e-8, atol=1e-8)


def test_uniform_alternating_density_null_but_diffusion_is_not_null():
    x, y = np.linspace(0, .001, 17), np.linspace(0, .0005, 9)
    h, p, theta = np.full((9, 17), 2e-6), np.full((9, 17), 1e7), np.ones((9, 17))
    dp = np.broadcast_to(1e6*(-1.)**np.arange(17), h.shape)
    fluid = IsothermalFluid(.04, 850.)
    terms = mass_direction_components(x, y, h, p, theta, fluid, .1,
                                      pressure_direction_pa=dp, film_direction_m=h*0,
                                      content_direction=h*0)
    rho, eta, _, _ = fluid.properties(p)
    symbol = 4*(rho*h**3/(12*eta))*dp/(x[1]-x[0])**2
    np.testing.assert_allclose(terms["pressure_diffusion"], symbol[1:-1, 1:-1], rtol=1e-13)
    for name in set(terms)-{"pressure_diffusion"}:
        np.testing.assert_array_equal(terms[name], 0)


def test_pressure_probes_have_zero_edges_force_and_no_input_mutation():
    x, y = np.linspace(-3e-4, 3e-4, 25), np.linspace(-3e-4, 3e-4, 17)
    xx, yy = np.meshgrid(x, y)
    p = 1e9*np.exp(-(xx**2+yy**2)/(1e-4)**2)
    original = p.copy()
    probes = pressure_mode_directions(x, y, p, 2.7e-4)
    assert len(probes) == 5
    for v in probes.values():
        np.testing.assert_array_equal(np.r_[v[0], v[-1], v[:, 0], v[:, -1]], 0)
        assert abs(v.sum()) < 1e-7
        assert abs(v).max() <= 1e6*(1+1e-14)
        np.testing.assert_array_equal(v[p <= .1*p.max()], 0)
    np.testing.assert_array_equal(p, original)
    with pytest.raises(ValueError):
        pressure_mode_directions(x, y, p*0, 2.7e-4)


def test_nyquist_projection_reports_raw_aliasing_without_filtering():
    x, y = np.linspace(-.001, .001, 25), np.linspace(-.001, .001, 17)
    flat = np.ones((17, 25))
    alternating = flat*(1+.4*(-1.)**np.arange(25))
    before = alternating.copy()
    smooth = nyquist_pressure_projection(x, y, flat, .00027)
    pattern = nyquist_pressure_projection(x, y, alternating, .00027)
    assert pattern["x"] > smooth["x"] + .2
    assert all(0 <= value <= 1 for value in pattern.values())
    np.testing.assert_array_equal(alternating, before)
    assert all(v == 0 for v in nyquist_pressure_projection(x, y, flat*0, .00027).values())


def test_centerline_projection_exposes_transverse_phase_cancellation():
    x, y = np.linspace(-.001, .001, 49), np.linspace(-.001, .001, 33)
    j, i = np.indices((y.size, x.size))
    p = 1+.4*(-1.)**(i+j)
    result = nyquist_pressure_projection(x, y, p, .00027)
    assert result["centerline_x"] > .35
    assert result["xy"] > .35
    assert result["x"] < .01


def test_invalid_si_direction_is_rejected():
    x, y = np.arange(5)*1e-6, np.arange(3)*1e-6
    field = np.ones((3, 5))
    with pytest.raises(ValueError):
        mass_direction_components(x, y, -field, field, field, IsothermalFluid(.04, 850), 1.,
                                  pressure_direction_pa=field, film_direction_m=field,
                                  content_direction=field)
