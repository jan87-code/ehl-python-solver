"""Independent conservation, symbols, consistency and JVP checks; no EHL claim."""

import numpy as np
import pytest

from ehl_solver.constitutive import IsothermalFluid
from ehl_solver.property_reynolds import mass_fluxes
from ehl_solver.transport_candidate import donor_flux_direction, donor_fluxes, face_divergence
from examples.benchmark_transport_candidate import smooth_errors


def test_nonuniform_conservation_and_independent_scalar_faces():
    x, y = np.array([0., .1, .3, .6]), np.array([0., .2, .5, 1., 1.2])
    rng = np.random.default_rng(238)
    h = 1e-6*(1+rng.random((5, 4)))
    p, theta = 1e7*rng.random(h.shape), rng.random(h.shape)
    f = IsothermalFluid(.04, 850.)
    qx, qy = donor_fluxes(x, y, h, p, theta, f, 2.)
    for j in range(5):
        for i in range(3):
            r0, e0, _, _ = f.properties(p[j, i])
            r1, e1, _, _ = f.properties(p[j, i+1])
            expected = -(r0*h[j, i]**3/e0+r1*h[j, i+1]**3/e1)/24
            expected *= (p[j, i+1]-p[j, i])/(x[i+1]-x[i])
            expected += 2*r0*h[j, i]*theta[j, i]
            assert qx[j, i] == pytest.approx(expected, rel=2e-15)
    for j in range(4):
        for i in range(4):
            r0, e0, _, _ = f.properties(p[j, i])
            r1, e1, _, _ = f.properties(p[j+1, i])
            expected = -(r0*h[j, i]**3/e0+r1*h[j+1, i]**3/e1)/24
            expected *= (p[j+1, i]-p[j, i])/(y[j+1]-y[j])
            assert qy[j, i] == pytest.approx(expected, rel=2e-15)
    wx, wy = .5*(x[2:]-x[:-2]), .5*(y[2:]-y[:-2])
    integral = np.sum(face_divergence(x, y, qx, qy)*wy[:, None]*wx)
    boundary = (qx[1:-1, -1]-qx[1:-1, 0])@wy+(qy[-1, 1:-1]-qy[0, 1:-1])@wx
    assert integral == pytest.approx(boundary, abs=1e-18)


@pytest.mark.parametrize("strength", [0., 1.])
def test_complete_flux_direction_with_independent_field_directions(strength):
    rng = np.random.default_rng(90)
    x, y = np.linspace(0, 1e-3, 11), np.linspace(0, 5e-4, 7)
    p = 1e8*(1+.1*rng.random((7, 11)))
    h, t = 1e-6*(1+.1*rng.random(p.shape)), .8+.1*rng.random(p.shape)
    dh, dp, dt = [scale*rng.normal(size=p.shape) for scale in (1e-7, 1e7, .1)]
    f = IsothermalFluid(.04, 850.)
    before = [a.copy() for a in (h, p, t)]
    analytic = donor_flux_direction(x, y, h, p, t, dh, dp, dt, f, 2., strength=strength)
    step = 1e-4
    plus = donor_fluxes(x, y, h+step*dh, p+step*dp, t+step*dt, f, 2., strength=strength)
    minus = donor_fluxes(x, y, h-step*dh, p-step*dp, t-step*dt, f, 2., strength=strength)
    for a, b, c in zip(analytic, plus, minus, strict=True):
        assert np.linalg.norm((b-c)/(2*step)-a)/np.linalg.norm(a) < 2e-8
    for a, b in zip(before, (h, p, t), strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("cells", [2, 4, 8, 16])
def test_donor_symbol_including_nonzero_nyquist_and_nonsymmetric_action(cells):
    x, y = np.arange(65)*1e-5, np.linspace(0, 1e-4, 5)
    phase = 2*np.pi*np.arange(65)/cells
    p = np.full((5, 65), 1e7)
    zero, h = np.zeros_like(p), np.full_like(p, 1e-6)
    # Density-pressure transport alone at a uniform state; take away known diffusion.
    fluid = IsothermalFluid(.04, 850.)
    dp = np.broadcast_to(1e6*np.cos(phase), p.shape)
    q = donor_flux_direction(x, y, h, p, np.ones_like(p), zero, dp, zero, fluid, 2.)
    q0 = donor_flux_direction(x, y, h, p, zero, zero, dp, zero, fluid, 0.)
    measured = face_divergence(x, y, q[0]-q0[0], q[1]-q0[1])
    _, _, dr, _ = fluid.properties(1e7)
    xi = 2*np.pi/cells
    expected = 2e-6*dr*1e6*((1-np.cos(xi))*np.cos(phase[1:-1])
                             - np.sin(xi)*np.sin(phase[1:-1]))/(x[1]-x[0])
    np.testing.assert_allclose(measured, np.broadcast_to(expected, measured.shape),
                               rtol=2e-12, atol=1e-11)
    if cells == 2:
        assert np.linalg.norm(measured) > 0


def test_constant_capacity_matches_reference_and_stationary_variable_film_content():
    x, y = np.linspace(0, .001, 33), np.linspace(0, .001, 5)
    p, h = np.zeros((5, 33)), np.full((5, 33), 1e-6)
    theta = np.broadcast_to(.8+.1*np.sin(2*np.pi*x/.001), p.shape).copy()
    f = IsothermalFluid(.04, 850.)
    for a, b in zip(donor_fluxes(x, y, h, p, theta, f, 2.),
                    mass_fluxes(x, y, h, p, theta, f, 2.), strict=True):
        np.testing.assert_allclose(a, b, rtol=2e-15, atol=1e-18)
    # Exact continuum solution: p=0, h*theta=constant, positive theta<=1.
    h = np.broadcast_to(1e-6*(1+.2*np.sin(2*np.pi*x/.001)), p.shape).copy()
    theta = .7e-6/h
    candidate = face_divergence(x, y, *donor_fluxes(x, y, h, p, theta, f, 2.))
    reference = face_divergence(x, y, *mass_fluxes(x, y, h, p, theta, f, 2.))
    assert abs(candidate).max() < 2e-14
    assert np.linalg.norm(reference) > 1e-3


def test_invalid_inputs_are_rejected_without_projection():
    x, y, p = np.arange(5.), np.arange(3.), np.zeros((3, 5))
    f = IsothermalFluid(.04, 850.)
    for axis, film, speed in ((x[::-1], p+1, 1.), (x, p, 1.), (x, p+1, -1.)):
        with pytest.raises(ValueError):
            donor_fluxes(axis, y, film, p, p+1, f, speed)
    with pytest.raises(ValueError):
        donor_flux_direction(x, y, p+1, p, p+1, np.zeros(2), p, p, f, 1.)
    with pytest.raises(ValueError):
        face_divergence(x, y, p, p)


@pytest.mark.parametrize("varying", [False, True])
def test_manufactured_source_consistency_orders(varying):
    errors = [smooth_errors(n, varying_content=varying) for n in (33, 65, 129)]
    for old, new in zip(errors, errors[1:], strict=False):
        for scheme in ("reference", "donor"):
            order = np.log2(old[scheme]/new[scheme])
            expected = 2. if scheme == "reference" and not varying else 1.
            assert expected-.15 < order < expected+.15
