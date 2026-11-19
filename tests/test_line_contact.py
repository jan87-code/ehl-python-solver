import math

import numpy as np
import pytest
from scipy.integrate import quad

from ehl_solver.backend import make_backend
from ehl_solver.geometry import LineContactGeometry
from ehl_solver.hertz import line_contact_hertz, pressure_line
from ehl_solver.line_contact import (
    PlaneStrainLineElasticityOperator,
    integrate_line_pressure,
    solve_dry_line_contact,
)

LINE_LOAD_N_M = 1000.0
RADIUS_M = 0.01
E_STAR_PA = 113e9
REFERENCE_HALF_WIDTH_M = line_contact_hertz(
    LINE_LOAD_N_M, RADIUS_M, E_STAR_PA, 1.0
).half_width_m
DOMAIN_HALF_WIDTH_M = 4.25 * REFERENCE_HALF_WIDTH_M


def test_hertz_line_reference_pressure_load_and_logarithmic_approach():
    reference = line_contact_hertz(
        LINE_LOAD_N_M, RADIUS_M, E_STAR_PA, DOMAIN_HALF_WIDTH_M
    )
    expected_half_width = math.sqrt(
        4 * LINE_LOAD_N_M * RADIUS_M / (math.pi * E_STAR_PA)
    )
    expected_peak_pressure = 2 * LINE_LOAD_N_M / (math.pi * expected_half_width)

    assert reference.half_width_m == pytest.approx(expected_half_width, rel=2e-15)
    assert reference.peak_pressure_pa == pytest.approx(expected_peak_pressure, rel=2e-15)
    assert pressure_line(0.0, reference) == pytest.approx(expected_peak_pressure)
    assert pressure_line(reference.half_width_m, reference) == 0.0
    assert pressure_line(1.1 * reference.half_width_m, reference) == 0.0
    assert quad(
        lambda x: pressure_line(x, reference),
        -reference.half_width_m,
        reference.half_width_m,
        epsabs=1e-10,
        epsrel=1e-12,
    )[0] == pytest.approx(LINE_LOAD_N_M, rel=2e-12)

    displacement_at_center = sum(
        quad(
            lambda x: pressure_line(x, reference) * math.log(
                DOMAIN_HALF_WIDTH_M / abs(x)
            ),
            lower,
            upper,
            epsabs=1e-9,
            epsrel=1e-12,
            limit=200,
        )[0]
        for lower, upper in (
            (-reference.half_width_m, 0.0),
            (0.0, reference.half_width_m),
        )
    ) * 2 / (math.pi * E_STAR_PA)
    assert reference.approach_m == pytest.approx(displacement_at_center, rel=2e-12)


def test_plane_strain_fft_operator_matches_direct_influence_quadrature():
    x_m = np.linspace(-DOMAIN_HALF_WIDTH_M, DOMAIN_HALF_WIDTH_M, 9)
    dx_m = x_m[1] - x_m[0]
    pressure_pa = np.linspace(1e6, 5e6, x_m.size)
    operator = PlaneStrainLineElasticityOperator(
        x_m, E_STAR_PA, DOMAIN_HALF_WIDTH_M
    )

    direct_displacement_m = []
    for target_x_m in x_m:
        integrated_pressure = 0.0
        for source_x_m, pressure in zip(x_m, pressure_pa, strict=True):
            lower = source_x_m - dx_m / 2
            upper = source_x_m + dx_m / 2
            singular_points = (
                [float(target_x_m)] if lower < target_x_m < upper else None
            )
            integral = quad(
                lambda source, target=float(target_x_m): math.log(
                    DOMAIN_HALF_WIDTH_M / abs(target - source)
                ),
                lower,
                upper,
                points=singular_points,
                epsabs=1e-14,
                epsrel=1e-12,
                limit=100,
            )[0]
            integrated_pressure += pressure * integral
        direct_displacement_m.append(
            2 * integrated_pressure / (math.pi * E_STAR_PA)
        )

    np.testing.assert_allclose(
        operator(pressure_pa),
        direct_displacement_m,
        rtol=5e-13,
        atol=1e-23,
    )


def test_line_pressure_integration_reports_force_per_length_and_moment():
    x_m = np.arange(-2.0, 3.0)
    loads = integrate_line_pressure(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), x_m)

    assert loads.line_load_n_m == pytest.approx(15.0)
    assert loads.moment_about_y_n == pytest.approx(10.0)


def test_dry_line_contact_matches_hertz_and_refines():
    reference = line_contact_hertz(
        LINE_LOAD_N_M, RADIUS_M, E_STAR_PA, DOMAIN_HALF_WIDTH_M
    )
    profile_errors = []
    approach_errors = []
    width_errors = []
    peak_pressure_errors = []

    for grid_size in (33, 65, 129):
        x_m = np.linspace(-DOMAIN_HALF_WIDTH_M, DOMAIN_HALF_WIDTH_M, grid_size)
        result = solve_dry_line_contact(
            x_m,
            LineContactGeometry(RADIUS_M),
            LINE_LOAD_N_M,
            E_STAR_PA,
            reference_length_m=DOMAIN_HALF_WIDTH_M,
        )
        analytical_pressure_pa = np.array([pressure_line(x, reference) for x in x_m])
        active = result.pressure_pa > 0
        numerical_half_width_m = np.max(np.abs(x_m[active]))

        assert result.relative_line_load_error == pytest.approx(0.0, abs=2e-14)
        assert result.loads.line_load_n_m == pytest.approx(LINE_LOAD_N_M, rel=2e-14)
        assert result.relative_kkt_residual <= 2e-6
        assert result.maximum_penetration_m < 2e-5 * reference.approach_m
        assert np.all(result.pressure_pa >= 0)
        profile_errors.append(
            np.linalg.norm(result.pressure_pa - analytical_pressure_pa)
            / np.linalg.norm(analytical_pressure_pa)
        )
        approach_errors.append(abs(result.approach_m / reference.approach_m - 1.0))
        width_errors.append(abs(numerical_half_width_m / reference.half_width_m - 1.0))
        peak_pressure_errors.append(
            abs(result.pressure_pa[grid_size // 2] / reference.peak_pressure_pa - 1.0)
        )

    assert profile_errors[0] > profile_errors[1] > profile_errors[2]
    assert approach_errors[0] > approach_errors[1] > approach_errors[2]
    assert width_errors[0] > width_errors[1] > width_errors[2]
    assert peak_pressure_errors[0] > peak_pressure_errors[1] > peak_pressure_errors[2]


def test_numpy_and_torch_cpu_line_elasticity_operator_parity():
    torch = pytest.importorskip("torch")
    x_m = np.linspace(-DOMAIN_HALF_WIDTH_M, DOMAIN_HALF_WIDTH_M, 17)
    pressure_pa = np.linspace(1e6, 5e6, x_m.size)
    numpy_displacement_m = PlaneStrainLineElasticityOperator(
        x_m, E_STAR_PA, DOMAIN_HALF_WIDTH_M
    )(pressure_pa)
    torch_displacement_m = PlaneStrainLineElasticityOperator(
        x_m,
        E_STAR_PA,
        DOMAIN_HALF_WIDTH_M,
        backend=make_backend("torch", "cpu"),
    )(pressure_pa)

    assert torch_displacement_m.dtype == torch.float64
    np.testing.assert_allclose(
        torch_displacement_m.numpy(), numpy_displacement_m, rtol=2e-13, atol=1e-23
    )


def test_cuda_line_elasticity_operator_matches_numpy_when_available():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("No CUDA or ROCm GPU available")
    x_m = np.linspace(-DOMAIN_HALF_WIDTH_M, DOMAIN_HALF_WIDTH_M, 17)
    pressure_pa = np.linspace(1e6, 5e6, x_m.size)
    numpy_displacement_m = PlaneStrainLineElasticityOperator(
        x_m, E_STAR_PA, DOMAIN_HALF_WIDTH_M
    )(pressure_pa)
    cuda_displacement_m = PlaneStrainLineElasticityOperator(
        x_m,
        E_STAR_PA,
        DOMAIN_HALF_WIDTH_M,
        backend=make_backend("torch", "cuda"),
    )(pressure_pa)

    assert cuda_displacement_m.dtype == torch.float64
    np.testing.assert_allclose(
        cuda_displacement_m.cpu().numpy(),
        numpy_displacement_m,
        rtol=2e-12,
        atol=1e-23,
    )
