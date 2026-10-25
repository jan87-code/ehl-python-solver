import numpy as np
import pytest

from examples.run_cavitating_domain import make_domain_problem


def test_domain_expansions_preserve_spacing_and_existing_nodes():
    baseline = make_domain_problem(0)
    for extra in (4, 8):
        expanded = make_domain_problem(extra)
        assert expanded.dx_m == pytest.approx(baseline.dx_m)
        assert expanded.dy_m == pytest.approx(baseline.dy_m)
        np.testing.assert_allclose(expanded.x_m[extra:-extra], baseline.x_m, atol=1e-18)
        np.testing.assert_allclose(expanded.y_m[extra:-extra], baseline.y_m, atol=1e-18)
        assert expanded.pressure_scale_pa == baseline.pressure_scale_pa
        assert expanded.e_star_pa == baseline.e_star_pa
        assert expanded.inlet_fractional_film_content == baseline.inlet_fractional_film_content


@pytest.mark.parametrize("extra", [-1, 1.5, True])
def test_invalid_domain_extensions_are_rejected(extra):
    with pytest.raises(ValueError):
        make_domain_problem(extra)
