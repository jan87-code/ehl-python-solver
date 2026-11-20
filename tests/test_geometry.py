from dataclasses import replace

import numpy as np
import pytest

from ehl_solver.backend import make_backend
from ehl_solver.convolution import fft_convolve_same
from ehl_solver.geometry import (
    AxialTiltGeometry,
    CircularAxialProfile,
    CircularPointContactGeometry,
    CompositeContactGeometry,
    EllipticalPointContactGeometry,
    LineContactGeometry,
    LogarithmicAxialProfile,
    ParabolicAxialProfile,
    ProfiledRollerGeometry,
    TabulatedAxialProfile,
    contact_geometry_from_config,
)
from ehl_solver.prototype import PointContactCase


def test_circular_geometry_is_bitwise_equal_to_frozen_ehl_paraboloid():
    case = PointContactCase()
    problem = case.problem()
    xx, yy = np.meshgrid(problem.x_m, problem.y_m, indexing='xy')
    frozen_geometry = xx**2 / (2 * case.radius_m) + yy**2 / (2 * case.radius_m)

    assert np.array_equal(problem.rigid_geometry_m, frozen_geometry)

    pressure = np.zeros(problem.shape)
    pressure[problem.shape[0] // 2, problem.shape[1] // 2] = 1e8
    expected_displacement = fft_convolve_same(
        pressure, problem.elastic_kernel, make_backend('numpy')
    )
    expected_film = (
        problem.central_film_offset_m + frozen_geometry + 0.25 * expected_displacement
    )
    film, displacement = problem.fields(pressure, compliance_factor=0.25)
    np.testing.assert_array_equal(displacement, expected_displacement)
    np.testing.assert_array_equal(film, expected_film)


def test_circular_and_elliptical_point_geometries_match_their_parabolas():
    x = np.array([-2e-6, 0.0, 2e-6])
    y = np.array([-3e-6, 0.0, 3e-6])
    xx, yy = np.meshgrid(x, y, indexing='xy')
    circle = CircularPointContactGeometry(0.01).h_geom(x, y)
    ellipse = EllipticalPointContactGeometry(0.01, 0.04).h_geom(x, y)

    np.testing.assert_array_equal(circle, xx**2 / 0.02 + yy**2 / 0.02)
    np.testing.assert_array_equal(ellipse, xx**2 / 0.02 + yy**2 / 0.08)


def test_line_geometry_is_translationally_invariant_transversely():
    x = np.array([-2e-6, 0.0, 2e-6])
    y = np.array([-3e-6, 0.0, 3e-6])
    geometry = LineContactGeometry(0.01).h_geom(x, y)

    np.testing.assert_array_equal(geometry, np.broadcast_to(geometry[1], geometry.shape))
    np.testing.assert_array_equal(geometry[:, 1], np.zeros(y.size))


def test_ehl_problem_uses_supplied_geometry_without_embedding_it_in_solver():
    case = PointContactCase()
    problem = case.problem()
    line = replace(problem, geometry=LineContactGeometry(case.radius_m))
    expected = np.broadcast_to(
        problem.x_m[None, :] ** 2 / (2 * case.radius_m), problem.shape
    )

    np.testing.assert_array_equal(line.rigid_geometry_m, expected)


def test_profiled_roller_geometry_applies_arbitrary_axial_profile():
    x = np.array([-2e-6, 0.0, 2e-6])
    y = np.array([-3e-6, 0.0, 3e-6])
    def profile(axial):
        return 0.5e-6 * (axial / 3e-6) ** 2

    geometry = ProfiledRollerGeometry(0.01, profile).h_geom(x, y)
    xx, yy = np.meshgrid(x, y, indexing='xy')

    np.testing.assert_allclose(
        geometry, xx**2 / 0.02 + 0.5e-6 * (yy / 3e-6) ** 2
    )


def test_profiled_roller_rejects_bad_profile_output():
    geometry = ProfiledRollerGeometry(0.01, lambda _axial: np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match='axial_profile_m must return'):
        geometry.h_geom(np.linspace(-1, 1, 3), np.linspace(-1, 1, 3))


def test_geometry_config_selects_all_shared_contact_geometries():
    cases = (
        (
            {'type': 'circular_point', 'radius_m': 0.01},
            CircularPointContactGeometry,
        ),
        (
            {'type': 'elliptical_point', 'radius_x_m': 0.01, 'radius_y_m': 0.04},
            EllipticalPointContactGeometry,
        ),
        ({'type': 'classical_line', 'radius_x_m': 0.01}, LineContactGeometry),
        (
            {'type': 'profiled_roller', 'radius_x_m': 0.01},
            ProfiledRollerGeometry,
        ),
    )

    for config, expected_type in cases:
        assert isinstance(contact_geometry_from_config(config), expected_type)

    x = np.array([-2e-6, 0.0, 2e-6])
    y = np.array([-3e-6, 0.0, 3e-6])
    no_crown = contact_geometry_from_config(
        {'type': 'profiled_roller', 'radius_x_m': 0.01}
    )
    expected = np.broadcast_to(x[None, :] ** 2 / 0.02, (y.size, x.size))
    np.testing.assert_array_equal(no_crown.h_geom(x, y), expected)


def test_tabulated_axial_profile_is_bounded_and_interpolated():
    profile = TabulatedAxialProfile(
        y_m=(-2e-3, 0.0, 2e-3),
        height_m=(0.0, 1e-6, 0.0),
    )
    np.testing.assert_allclose(
        profile(np.array([-1e-3, 0.0, 1e-3])),
        np.array([0.5e-6, 1e-6, 0.5e-6]),
    )
    with pytest.raises(ValueError, match='outside the tabulated profile'):
        profile(np.array([-2.1e-3]))

    geometry = contact_geometry_from_config(
        {
            'type': 'profiled_roller',
            'radius_x_m': 0.01,
            'axial_profile': {
                'type': 'tabulated',
                'y_m': [-2e-3, 0.0, 2e-3],
                'height_m': [0.0, 1e-6, 0.0],
            },
        }
    )
    x = np.array([-1e-6, 0.0, 1e-6])
    y = np.array([-1e-3, 0.0, 1e-3])
    expected = x[None, :] ** 2 / 0.02 + np.array([0.5e-6, 1e-6, 0.5e-6])[:, None]
    np.testing.assert_allclose(geometry.h_geom(x, y), expected)


def test_analytic_crowns_are_symmetric_and_have_expected_local_curvature():
    y = np.array([-1e-3, -1e-6, 0.0, 1e-6, 1e-3])
    radius = 0.2
    parabola = ParabolicAxialProfile(radius)(y)
    circular = CircularAxialProfile(radius)(y)
    np.testing.assert_array_equal(parabola, parabola[::-1])
    np.testing.assert_array_equal(circular, circular[::-1])
    assert circular[2] == 0.0
    assert circular[3] == pytest.approx(parabola[3], rel=1e-10)
    assert circular[-1] > parabola[-1]
    with pytest.raises(ValueError, match="within the circular crown radius"):
        CircularAxialProfile(radius)(np.array([radius * 1.01]))


def test_capped_logarithmic_crown_has_finite_symmetric_end_relief():
    half_length = 1e-3
    scale = 2e-6
    cap = 4e-6
    profile = LogarithmicAxialProfile(half_length, scale, cap)
    y = np.array([-half_length, -0.5 * half_length, 0.0, 0.5 * half_length, half_length])
    values = profile(y)
    np.testing.assert_allclose(values, values[::-1], atol=0)
    assert values[2] == 0.0
    assert values[3] == pytest.approx(-scale * np.log1p(-0.25))
    assert values[-1] == cap
    assert np.all(np.isfinite(values))
    assert profile(np.array([np.nextafter(half_length, 0.0)]))[0] == cap
    with pytest.raises(ValueError, match="outside the logarithmic profile length"):
        profile(np.array([half_length * 1.01]))


@pytest.mark.parametrize(
    ("config", "expected_type"),
    [
        ({"type": "parabolic", "radius_m": 0.2}, ParabolicAxialProfile),
        ({"type": "circular", "radius_m": 0.2}, CircularAxialProfile),
        (
            {
                "type": "logarithmic",
                "half_length_m": 1e-3,
                "scale_m": 2e-6,
                "maximum_relief_m": 4e-6,
            },
            LogarithmicAxialProfile,
        ),
    ],
)
def test_geometry_config_selects_analytic_profiles(config, expected_type):
    geometry = contact_geometry_from_config(
        {"type": "profiled_roller", "radius_x_m": 0.01, "axial_profile": config}
    )
    assert isinstance(geometry.axial_profile_m, expected_type)


def test_tilt_is_an_independent_additive_component_of_axial_profile():
    x = np.array([-1e-6, 0.0, 1e-6])
    y = np.array([-2e-3, 0.0, 2e-3])
    profile = ProfiledRollerGeometry(
        0.01, lambda axial: 0.5e-6 * (axial / 2e-3) ** 2
    )
    angle = 0.01
    composite = CompositeContactGeometry((profile, AxialTiltGeometry(angle)))
    xx, yy = np.meshgrid(x, y, indexing='xy')

    expected = (
        xx**2 / 0.02
        + 0.5e-6 * (yy / 2e-3) ** 2
        + yy * np.tan(angle)
    )
    np.testing.assert_allclose(composite.h_geom(x, y), expected)

    configured = contact_geometry_from_config(
        {
            'type': 'profiled_roller',
            'radius_x_m': 0.01,
            'axial_profile': {'type': 'cylindrical'},
            'axial_tilt_rad': angle,
        }
    )
    np.testing.assert_allclose(
        configured.h_geom(x, y),
        xx**2 / 0.02 + yy * np.tan(angle),
    )
    with pytest.raises(ValueError, match='currently defined for profiled_roller'):
        contact_geometry_from_config(
            {
                'type': 'classical_line',
                'radius_x_m': 0.01,
                'axial_tilt_rad': angle,
            }
        )


def test_composite_geometry_accepts_separate_time_dependent_components():
    class TimeDependentHeight:
        def h_geom(self, x_m, y_m, *, time_s=None):
            if time_s is None:
                raise ValueError('time_s is required for this test component')
            return np.full((len(y_m), len(x_m)), time_s * 1e-7)

    x = np.array([-1e-6, 0.0, 1e-6])
    y = np.array([-2e-3, 0.0, 2e-3])
    base = LineContactGeometry(0.01)
    dynamic = CompositeContactGeometry((base, TimeDependentHeight()))

    at_zero = dynamic.h_geom(x, y, time_s=0.0)
    later = dynamic.h_geom(x, y, time_s=2.0)
    np.testing.assert_array_equal(at_zero, base.h_geom(x, y))
    np.testing.assert_allclose(later - at_zero, 2e-7)
