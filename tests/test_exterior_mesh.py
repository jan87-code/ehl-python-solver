import json

import numpy as np
import pytest

from ehl_solver.exterior import exterior_source_budget, solve_exterior_response
from examples.run_exterior_mesh import compare_core, compare_fronts, reference, section_flux
from examples.run_exterior_response import fields
from examples.run_farfield_budget import ETA, H0, LY, P_REF, RX, RY, SPEED, L, graded_axis


def test_fixed_outer_reservoir_grading_converges_toward_independent_uniform_reference():
    target, ref = reference(0.125)
    errors = []
    for growth in (1.2, 1.05):
        axis = graded_axis(8, 8, 0.125, snap=False, growth=growth)
        x, y = axis * L, axis * LY
        h = H0 + x[None, :] ** 2 / (2 * RX) + y[:, None] ** 2 / (2 * RY)
        result = solve_exterior_response(
            x,
            y,
            h,
            ETA,
            SPEED,
            core_bounds_m=(-L, L, -LY, LY),
            length_scale_m=L,
            film_scale_m=H0,
            pressure_scale_pa=P_REF,
        )
        f = fields(x, y, h, result.pressure_pa, result.fractional_film_content)
        budget = exterior_source_budget(x, y, result, 1e9, [[0, 0]])
        errors.append(
            (
                abs(budget["total_load_n"] / target["load_n"] - 1),
                compare_core(f, ref),
                abs(budget["total_displacement_m"][0] / target["center_displacement_m"] - 1),
            )
        )
        assert x[0] == pytest.approx(-8 * L, rel=1e-15)
        assert y[-1] == pytest.approx(8 * LY, rel=1e-15)
        assert result.audit["mass_maximum"] < 1e-8
        assert result.audit["relative_local_balance"] < 1e-10
        assert result.audit["relative_global_balance"] < 1e-10
        assert result.audit["maximum_complementarity"] < 1e-10
        front = compare_fronts(f, ref)
        assert front["within_two_core_cells"]
        json.dumps(front, allow_nan=False)
        assert result.history[-1]["dense_work_estimate_bytes"] <= 64 * 1024**2
    assert all(fine < coarse for coarse, fine in zip(errors[0], errors[1], strict=True))
    assert max(errors[1]) < 0.005


def test_fixed_section_flow_has_si_volume_units_and_detects_missing_front():
    x, y = np.linspace(-2, 2, 33) * L, np.linspace(-2, 2, 33) * LY
    h = np.full((33, 33), H0)
    gradient = 1e6
    p = np.broadcast_to(gradient * x, h.shape)
    theta = np.ones_like(h)
    flow = section_flux(x, y, h, p, theta)
    assert flow["couette_m3_s"] == pytest.approx(2 * LY * SPEED * H0, rel=1e-14)
    assert flow["poiseuille_m3_s"] == pytest.approx(
        -2 * LY * H0**3 * gradient / (12 * ETA), rel=1e-14
    )
    f = fields(x, y, h, p, theta)
    changed = f | {"theta": theta.copy()}
    changed["theta"][:, x > 0] = 0.5
    assert not compare_fronts(f, changed)["within_two_core_cells"]
