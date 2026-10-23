import numpy as np
import pytest
from scipy.sparse.linalg import eigsh

from ehl_solver.reynolds import (
    assemble_reynolds_system,
    compare_reynolds_linear_solvers,
    dimensionless_reynolds_parameter,
    parabolic_point_contact_film,
    reynolds_diagnostics,
    solve_reynolds_pressure,
)

VISCOSITY_PA_S = 0.04
SURFACE_SPEED_1_M_S = 0.85
SURFACE_SPEED_2_M_S = 0.0
LENGTH_SCALE_M = 5e-4
FILM_SCALE_M = 20e-6
PRESSURE_SCALE_PA = (
    VISCOSITY_PA_S
    * (SURFACE_SPEED_1_M_S + SURFACE_SPEED_2_M_S)
    / 2.0
    * LENGTH_SCALE_M
    / FILM_SCALE_M**2
)
REYNOLDS_PARAMETER = dimensionless_reynolds_parameter(
    VISCOSITY_PA_S,
    SURFACE_SPEED_1_M_S,
    SURFACE_SPEED_2_M_S,
    LENGTH_SCALE_M,
    FILM_SCALE_M,
    PRESSURE_SCALE_PA,
)


def _parabolic_case(nx: int, ny: int, x_limits_m=(-2e-3, 1.5e-3)):
    x_m = np.linspace(*x_limits_m, nx)
    y_m = np.linspace(-1.5e-3, 1.5e-3, ny)
    x_grid_m, y_grid_m = np.meshgrid(x_m, y_m, indexing="xy")
    dimensional_film = parabolic_point_contact_film(
        x_grid_m, y_grid_m, FILM_SCALE_M, radius_x_m=0.01, radius_y_m=0.02
    )
    film = dimensional_film / FILM_SCALE_M
    dx = (x_m[1] - x_m[0]) / LENGTH_SCALE_M
    dy = (y_m[1] - y_m[0]) / LENGTH_SCALE_M
    return film, dx, dy


def _solve_parabolic_case(nx: int, ny: int, x_limits_m=(-2e-3, 1.5e-3)):
    film, dx, dy = _parabolic_case(nx, ny, x_limits_m)
    solution = solve_reynolds_pressure(
        film,
        dx,
        dy,
        REYNOLDS_PARAMETER,
        rtol=1e-10,
        atol=1e-12,
    )
    diagnostics = reynolds_diagnostics(
        solution,
        film,
        dx,
        dy,
        REYNOLDS_PARAMETER,
        pressure_scale_pa=PRESSURE_SCALE_PA,
        length_scale_m=LENGTH_SCALE_M,
    )
    return solution, diagnostics


def test_dimensional_scaling_and_mean_speed_factor_are_consistent():
    lambda_mean_speed = dimensionless_reynolds_parameter(
        0.04, 0.8, 0.2, 0.003, 4e-6, 120e6
    )
    lambda_speed_sum = 6 * 0.04 * (0.8 + 0.2) * 0.003 / (4e-6**2 * 120e6)
    dimensional_prefactor = FILM_SCALE_M**3 * PRESSURE_SCALE_PA / LENGTH_SCALE_M**2
    dimensional_convection = (
        12
        * VISCOSITY_PA_S
        * (SURFACE_SPEED_1_M_S + SURFACE_SPEED_2_M_S)
        / 2
        * FILM_SCALE_M
        / LENGTH_SCALE_M
    )

    assert lambda_mean_speed == pytest.approx(lambda_speed_sum, rel=1e-15)
    assert REYNOLDS_PARAMETER == pytest.approx(12.0, rel=1e-15)
    assert dimensional_convection / dimensional_prefactor == pytest.approx(
        REYNOLDS_PARAMETER, rel=1e-15
    )


def test_parabolic_film_has_si_minimum_and_correct_curvature():
    x = np.array([[-1e-3, 0.0, 1e-3]])
    y = np.zeros_like(x)
    film = parabolic_point_contact_film(x, y, 20e-6, 0.01, 0.02)

    np.testing.assert_allclose(film, [[70e-6, 20e-6, 70e-6]], rtol=1e-15)


def test_parabolic_pressure_and_integrated_load_converge_under_refinement():
    solutions_and_diagnostics = [_solve_parabolic_case(n, n) for n in (33, 65, 97)]
    pressures = [item[0].pressure for item in solutions_and_diagnostics]
    maxima = [item[1].maximum_pressure_pa for item in solutions_and_diagnostics]
    loads = [item[1].integrated_load_n for item in solutions_and_diagnostics]
    maximum_differences = [abs(maxima[i] - maxima[i + 1]) for i in range(2)]
    load_differences = [abs(loads[i] - loads[i + 1]) for i in range(2)]
    pressure_field_differences = [
        np.sqrt(np.mean((pressures[0] - pressures[1][::2, ::2]) ** 2)),
        np.sqrt(
            np.mean((pressures[1][::2, ::2] - pressures[2][::3, ::3]) ** 2)
        ),
    ]

    assert maxima[0] < maxima[1] < maxima[2]
    assert maximum_differences[1] < maximum_differences[0]
    assert pressure_field_differences[1] < pressure_field_differences[0]
    assert loads[0] > 0 and loads[1] > 0 and loads[2] > 0
    assert load_differences[1] < load_differences[0]
    assert abs(loads[2] - loads[1]) / loads[2] < 1e-3


def test_parabolic_full_film_domain_sensitivity_decreases_as_edges_move_out():
    extents = (1e-3, 1.5e-3, 2e-3, 3e-3, 4e-3)
    maxima = [
        _solve_parabolic_case(2 * round(extent / 0.05e-3) + 1, 41, (-extent, extent))[1]
        .maximum_pressure_pa
        for extent in extents
    ]
    differences = np.abs(np.diff(maxima))

    assert np.all(np.diff(maxima) > 0)
    assert differences[1] < differences[0]
    assert differences[2] < differences[1]
    assert differences[3] < differences[2]
    assert differences[-1] / maxima[-1] < 0.01


@pytest.mark.parametrize("shape", [(5, 7), (12, 9)])
def test_positive_film_assembled_operator_is_symmetric_positive_definite(shape):
    rng = np.random.default_rng(67)
    film = 0.2 + rng.random(shape)
    matrix, _ = assemble_reynolds_system(film, 0.13, 0.17, 2.0)

    assert matrix.dtype == np.float64
    np.testing.assert_allclose((matrix - matrix.T).toarray(), 0.0, atol=0.0)
    smallest_eigenvalue = eigsh(matrix, k=1, which="SA", return_eigenvectors=False)[0]
    assert smallest_eigenvalue > 0.0


def test_local_control_volume_and_boundary_flux_diagnostics():
    solution, diagnostics = _solve_parabolic_case(33, 33)

    assert diagnostics.minimum_pressure_pa < 0.0
    assert diagnostics.maximum_pressure_pa > 0.0
    assert diagnostics.integrated_load_n > 0.0
    assert diagnostics.normalized_algebraic_residual < 1e-9
    assert diagnostics.maximum_local_conservation_error < 1e-8
    assert diagnostics.maximum_local_conservation_error_relative < 1e-9
    assert abs(diagnostics.net_boundary_flux) < 1e-8
    assert diagnostics.west_boundary_flux != diagnostics.east_boundary_flux
    assert solution.algebraic_residual_relative_l2 == pytest.approx(
        diagnostics.normalized_algebraic_residual
    )


@pytest.mark.parametrize("n", [25, 41])
def test_direct_and_iterative_sparse_solvers_agree_and_report_timings(n):
    film, dx, dy = _parabolic_case(n, n)
    direct, iterative = compare_reynolds_linear_solvers(
        film, dx, dy, REYNOLDS_PARAMETER, rtol=1e-9, atol=1e-11
    )

    assert direct.method == "direct"
    assert iterative.method == "iterative"
    assert direct.elapsed_seconds >= 0.0
    assert iterative.elapsed_seconds >= 0.0
    assert direct.iterations is None
    assert iterative.iterations is not None and iterative.iterations > 0
    assert direct.relative_residual_l2 < 1e-11
    assert iterative.relative_residual_l2 < 1e-8
    np.testing.assert_allclose(iterative.pressure, direct.pressure, rtol=1e-7, atol=1e-9)
