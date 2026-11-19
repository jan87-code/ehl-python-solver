import math

import pytest

from ehl_solver.hertz import (
    circular_point_contact,
    elliptical_point_contact,
    pressure_elliptical,
    pressure_radial,
)


def test_hertz_load_integrates_analytically():
    load = 5.0
    h = circular_point_contact(load, 0.015, 113e9)
    assert 2 * math.pi * h.peak_pressure_pa * h.radius_m**2 / 3 == pytest.approx(load)
    assert pressure_radial(0, h) == pytest.approx(h.peak_pressure_pa)
    assert pressure_radial(h.radius_m, h) == pytest.approx(0)
    assert pressure_radial(h.radius_m * 2, h) == 0


def test_hertz_scaling():
    a = circular_point_contact(5, 0.015, 113e9)
    b = circular_point_contact(40, 0.015, 113e9)
    assert b.radius_m / a.radius_m == pytest.approx(2.0)


def test_hertz_bad_input():
    with pytest.raises(ValueError):
        circular_point_contact(0, 1, 1)


def test_elliptical_hertz_circle_limit_matches_circular_reference():
    circular = circular_point_contact(5.0, 0.015, 113e9)
    elliptical = elliptical_point_contact(5.0, 0.015, 0.015, 113e9)

    assert elliptical.semi_axis_x_m == pytest.approx(circular.radius_m, rel=1e-14)
    assert elliptical.semi_axis_y_m == pytest.approx(circular.radius_m, rel=1e-14)
    assert elliptical.peak_pressure_pa == pytest.approx(circular.peak_pressure_pa, rel=1e-14)
    assert elliptical.approach_m == pytest.approx(circular.approach_m, rel=1e-14)
    assert pressure_elliptical(0.0, 0.0, elliptical) == pytest.approx(
        circular.peak_pressure_pa
    )


def test_elliptical_hertz_reference_matches_known_axis_ratio_and_pressure():
    result = elliptical_point_contact(5.0, 0.015, 0.06, 113e9)

    assert result.axis_ratio_y_over_x == pytest.approx(2.5007063549272237, rel=1e-11)
    assert result.semi_axis_x_m == pytest.approx(61.56030628226593e-6, rel=1e-11)
    assert result.semi_axis_y_m == pytest.approx(153.94424913132873e-6, rel=1e-11)
    assert result.peak_pressure_pa == pytest.approx(251911009.97180778, rel=1e-11)
    assert pressure_elliptical(
        result.semi_axis_x_m, 0.0, result
    ) == pytest.approx(0.0, abs=1e-14)


def test_elliptical_hertz_swaps_axes_with_principal_curvature_orientation():
    wide_x = elliptical_point_contact(5.0, 0.06, 0.015, 113e9)
    wide_y = elliptical_point_contact(5.0, 0.015, 0.06, 113e9)

    assert wide_x.semi_axis_x_m == pytest.approx(wide_y.semi_axis_y_m, rel=1e-14)
    assert wide_x.semi_axis_y_m == pytest.approx(wide_y.semi_axis_x_m, rel=1e-14)


def test_elliptical_hertz_is_stable_near_the_circular_limit():
    result = elliptical_point_contact(5.0, 0.015, 0.015 * (1 + 1e-7), 113e9)

    assert result.axis_ratio_y_over_x == pytest.approx(1 + 2 / 3 * 1e-7, rel=1e-9)
    assert result.semi_axis_x_m > 0
    assert result.semi_axis_y_m > 0


@pytest.mark.parametrize(
    'values',
    [(0.0, 1.0, 1.0, 1.0), (1.0, float('inf'), 1.0, 1.0)],
)
def test_elliptical_hertz_rejects_invalid_input(values):
    with pytest.raises(ValueError):
        elliptical_point_contact(*values)
