"""Independent SI affine-bound checks; diagnostics never choose or clip a step."""

import numpy as np
import pytest

from ehl_solver.newton_diagnostics import affine_feasibility_limits


def fields():
    return [np.full((5, 7), v) for v in (2., .5, 3., 0., 0., 0.)]


def evaluate(arrays, **kwargs):
    return affine_feasibility_limits(*arrays, pressure_scale_pa=10., film_floor_m=1., **kwargs)


def test_uncapped_independent_limits_ties_and_strict_film_boundary():
    arrays = fields()
    p, theta, h, dp, dt, dh = arrays
    dp[1, 1] = dp[2, 3] = -4.
    dt[2, 3] = 1.
    dh[2, 3] = -4.
    snapshot = [a.copy() for a in arrays]
    result = evaluate(arrays, feasibility_tolerance=0.)
    assert result["first_alpha_limit"] == .5
    assert set(result["first_constraints"]) == {"pressure_lower", "content_upper", "film_lower"}
    assert result["limits"]["pressure_lower"]["cells_iy_ix"] == [[1, 1], [2, 3]]
    assert result["limits"]["pressure_lower"]["tie_count"] == 2
    assert result["limits"]["film_lower"]["strict"]
    assert not result["full_step_strictly_inside"]
    # Independent evaluation of the actual affine SI fields around the first boundary.
    for alpha, feasible in ((.49, True), (.5, False), (.51, False)):
        ok = (np.min((p+alpha*dp)[1:-1, 1:-1]) >= 0
              and np.max((theta+alpha*dt)[1:-1, 1:-1]) <= 1
              and np.min(h+alpha*dh) > 1)
        assert bool(ok) == feasible
    for actual, original in zip(arrays, snapshot, strict=True):
        np.testing.assert_array_equal(actual, original)


def test_bounds_above_one_and_zero_directions_are_not_clamped():
    arrays = fields()
    arrays[3][2, 3] = -.25
    arrays[4][2, 3] = -.125
    result = evaluate(arrays, feasibility_tolerance=0.)
    assert result["limits"]["pressure_lower"]["alpha_limit"] == 8.
    assert result["limits"]["content_lower"]["alpha_limit"] == 4.
    assert result["first_alpha_limit"] == 4.
    assert result["full_step_strictly_inside"]
    result = evaluate(fields())
    assert result["first_alpha_limit"] is None and result["first_constraints"] == []


def test_pressure_and_content_edges_are_fixed_but_full_grid_film_is_constrained():
    arrays = fields()
    arrays[3][0, 0] = -100.
    arrays[4][0, 0] = 100.
    arrays[5][0, 0] = -8.
    result = evaluate(arrays)
    assert result["limits"]["pressure_lower"]["alpha_limit"] is None
    assert result["limits"]["content_upper"]["alpha_limit"] is None
    assert result["first_alpha_limit"] == .25
    assert result["limits"]["film_lower"]["cells_iy_ix"] == [[0, 0]]


def test_si_unit_rescaling_and_existing_pressure_content_tolerance():
    arrays = fields()
    arrays[3][2, 3] = -4.
    arrays[4][2, 3] = -1.
    result = evaluate(arrays)
    assert result["limits"]["pressure_lower"]["alpha_limit"] == (2+1e-9)/4
    assert result["limits"]["content_lower"]["alpha_limit"] == .5+1e-10
    scaled = [a.copy() for a in arrays]
    for index in (0, 3):
        scaled[index] *= 1e9
    for index in (2, 5):
        scaled[index] *= 1e-6
    changed = affine_feasibility_limits(*scaled, pressure_scale_pa=1e10, film_floor_m=1e-6)
    for key, value in result["limits"].items():
        assert changed["limits"][key]["alpha_limit"] == value["alpha_limit"]


@pytest.mark.parametrize("failure", ["shape", "nonfinite", "pressure", "content", "film"])
def test_invalid_or_infeasible_initial_data_are_rejected(failure):
    arrays = fields()
    if failure == "shape":
        arrays[0] = arrays[0].T
    elif failure == "nonfinite":
        arrays[3][2, 3] = np.nan
    elif failure == "pressure":
        arrays[0][2, 3] = -1.
    elif failure == "content":
        arrays[1][2, 3] = 1.1
    else:
        arrays[2][0, 0] = 1.
    with pytest.raises(ValueError):
        evaluate(arrays)
