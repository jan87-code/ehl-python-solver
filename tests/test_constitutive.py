from decimal import Decimal, localcontext

import numpy as np
import pytest

from ehl_solver.constitutive import IsothermalFluid


def test_si_laws_match_independent_high_precision_reference_and_derivatives():
    fluid = IsothermalFluid(0.04, 850)
    pressure = np.array([0, 1e5, 1e7, 1e8, 5e8, 1e9])
    r, e, dr, de = fluid.ratios(pressure)
    with localcontext() as ctx:
        ctx.prec = 60
        values = []
        for p in pressure:
            p = Decimal(str(p))
            a = Decimal("0.6e-9")
            b = Decimal("1.7e-9")
            v = (Decimal(".04") / Decimal("6.31e-5")).ln()
            exponent = ((1 + p / Decimal("1.96e8")).ln() * Decimal(".68")).exp()
            values.append([float(1 + a * p / (1 + b * p)), float((v * (exponent - 1)).exp())])
    np.testing.assert_allclose(r, np.array(values)[:, 0], rtol=2e-15)
    np.testing.assert_allclose(e, np.array(values)[:, 1], rtol=2e-14)
    for p in (1e6, 1e8, 1e9):
        delta = p * 1e-5
        plus = fluid.ratios(p + delta)
        minus = fluid.ratios(p - delta)
        exact = fluid.ratios(p)
        assert (plus[0] - minus[0]) / (2 * delta) == pytest.approx(exact[2], rel=1e-8)
        assert (plus[1] - minus[1]) / (2 * delta) == pytest.approx(exact[3], rel=1e-8)
    assert dr[0] == 0.6e-9
    assert np.all(np.diff(r) > 0) and np.all(np.diff(e) > 0)
    np.testing.assert_array_equal(fluid.ratios(pressure, strength=0)[0], np.ones(6))
    np.testing.assert_array_equal(fluid.ratios(pressure, strength=0)[1], np.ones(6))
    rho, eta, _, _ = fluid.properties(0)
    assert rho == 850 and eta == 0.04


def test_density_units_and_viscosity_origin_slope_and_asymptote():
    fluid = IsothermalFluid(0.04, 850)
    # Original density formula uses p in GPa, not Pa.
    r = fluid.ratios(np.array([1e8, 1e9]))[0]
    np.testing.assert_allclose(
        r, 1 + 0.6 * np.array([0.1, 1]) / (1 + 1.7 * np.array([0.1, 1])), rtol=1e-15
    )
    r, e, dr, de = fluid.ratios(0)
    assert e == 1 and r == 1
    assert de == pytest.approx(np.log(0.04 / 6.31e-5) * 0.68 / 1.96e8)
    assert fluid.ratios(1e13, strength=0)[0] == 1
    with pytest.raises(FloatingPointError):
        fluid.ratios(1e13)
    with pytest.raises(ValueError):
        fluid.ratios(-1e9)
    with pytest.raises(ValueError):
        IsothermalFluid(-1, 850)
