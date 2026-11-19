from dataclasses import replace

import numpy as np
import pytest

from ehl_solver.backend import make_backend
from ehl_solver.convolution import fft_convolve_same
from ehl_solver.geometry import (
    CircularPointContactGeometry,
    EllipticalPointContactGeometry,
    LineContactGeometry,
    ProfiledRollerGeometry,
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
