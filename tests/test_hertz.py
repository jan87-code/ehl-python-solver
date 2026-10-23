import math

import pytest

from ehl_solver.hertz import circular_point_contact, pressure_radial


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
