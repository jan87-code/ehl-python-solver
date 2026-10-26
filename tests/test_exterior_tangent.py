from dataclasses import replace

import numpy as np
import pytest

from ehl_solver.backend import make_backend
from ehl_solver.elasticity import elastic_deformation
from ehl_solver.exterior import (
    differentiate_exterior_response,
    retained_source_displacements,
    solve_exterior_response,
)
from examples.run_farfield_budget import ETA, H0, LY, P_REF, RX, RY, SPEED, L, graded_axis


@pytest.mark.parametrize("location", [-0.5, -4])
@pytest.mark.parametrize("graded", [False, True])
def test_film_direction_includes_interior_and_exterior_transport(location, graded):
    units = graded_axis(8, 8, 0.25, snap=False) if graded else np.linspace(-8, 8, 33)
    x, y = units * L, units * LY
    h = H0 + x[None, :] ** 2 / (2 * RX) + y[:, None] ** 2 / (2 * RY)
    dh = H0 * np.exp(-((x[None, :] / L - location) ** 2 + (y[:, None] / LY) ** 2) / 2)
    scales = dict(length_scale_m=L, film_scale_m=H0, pressure_scale_pa=P_REF)

    def solve(film):
        return solve_exterior_response(
            x, y, film, ETA, SPEED, core_bounds_m=(-2 * L, 2 * L, -2 * LY, 2 * LY), **scales
        )

    base = solve(h)
    tangent = differentiate_exterior_response(x, y, h, dh, base, ETA, SPEED, **scales)
    eps = 3e-4
    plus, minus = solve(h + eps * dh), solve(h - eps * dh)
    np.testing.assert_array_equal(plus.full_active_set, base.full_active_set)
    np.testing.assert_array_equal(minus.full_active_set, base.full_active_set)
    fd = (plus.pressure_pa - minus.pressure_pa) / (2 * eps)
    assert np.linalg.norm(fd - tangent.pressure_direction_pa) / np.linalg.norm(fd) < 1e-6
    ft = (plus.fractional_film_content - minus.fractional_film_content) / (2 * eps)
    assert np.linalg.norm(ft - tangent.content_direction) / np.linalg.norm(ft) < 1e-6
    assert tangent.linear_info["relative_row_backward_error"] < 1e-10
    assert np.linalg.norm(tangent.pressure_direction_pa[~base.retained_mask]) > 0
    direct = retained_source_displacements(
        x, y, replace(base, pressure_pa=tangent.pressure_direction_pa), 1e9
    )
    if not graded:
        fft = elastic_deformation(
            tangent.pressure_direction_pa, x[1] - x[0], y[1] - y[0], 1e9, make_backend("numpy")
        )
        np.testing.assert_allclose(
            direct["total_displacement_m"], fft[base.retained_mask], rtol=1e-10, atol=1e-22
        )
    else:
        a = retained_source_displacements(x, y, plus, 1e9)["total_displacement_m"]
        b = retained_source_displacements(x, y, minus, 1e9)["total_displacement_m"]
        np.testing.assert_allclose(
            direct["total_displacement_m"], (a - b) / (2 * eps), rtol=1e-6, atol=1e-19
        )
    np.testing.assert_allclose(
        direct["retained_displacement_m"] + direct["exterior_displacement_m"],
        direct["total_displacement_m"],
        rtol=1e-12,
        atol=1e-22,
    )
    with pytest.raises(ValueError, match="recorded full active set"):
        differentiate_exterior_response(
            x, y, h, dh, replace(base, full_active_set=None), ETA, SPEED, **scales
        )
    with pytest.raises(ValueError, match="supplied film"):
        differentiate_exterior_response(x, y, h + 0.1 * dh, dh, base, ETA, SPEED, **scales)
