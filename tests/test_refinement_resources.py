from types import SimpleNamespace

import numpy as np
import pytest

from ehl_solver.benchmark_200n import Circular200N
from ehl_solver.constitutive import IsothermalFluid
from ehl_solver.property_load_newton import PropertyLoadSystem
from examples.compare_refined_meshes import apparent_order
from examples.run_property_load_verification import make_problem


def test_default_node_cap_and_explicit_bounded_extension():
    with pytest.raises(ValueError, match="budget"):
        Circular200N().problem(.5e-6, ny=129, nx=193)
    assert Circular200N().problem(.5e-6, ny=129, nx=193,
                                  maximum_physical_nodes=30000).shape == (129, 193)
    p, f = make_problem(129, 193), IsothermalFluid(.04, 850.)
    with pytest.raises(ValueError, match="exceeds 20000"):
        PropertyLoadSystem(p, f, .1, 1.)
    system = PropertyLoadSystem(p, f, .1, 1., maximum_physical_nodes=30000,
                                elasticity_mode="cached")
    assert system.n == 127*191
    for value in (True, 30001, 20000.):
        with pytest.raises(ValueError, match="explicitly"):
            PropertyLoadSystem(p, f, .1, 1., maximum_physical_nodes=value)
    with pytest.raises(ValueError, match="exceeds 30000"):
        PropertyLoadSystem(SimpleNamespace(shape=(201, 151)), f, .1, 1.,
                           maximum_physical_nodes=30000)


def test_unequal_spacing_fit_is_qualified_and_rejects_nonmonotone_data():
    h = np.array([1/2, 1/3, 1/4])
    result = apparent_order(h, 2+3*h**1.7)
    assert result["apparent_order"] == pytest.approx(1.7, abs=1e-10)
    assert result["three_point_extrapolation"] == pytest.approx(2., abs=1e-12)
    assert not result["qualified"]
    assert apparent_order(h, [1., 2., 1.5]) is None
    assert apparent_order(h, [1., 1., 1.]) is None
