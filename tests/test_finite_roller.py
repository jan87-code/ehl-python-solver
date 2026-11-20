import numpy as np
import pytest

from ehl_solver.backend import make_backend
from ehl_solver.contact_compliance import DirectHalfSpaceCompliance, HalfSpaceFFTCompliance
from ehl_solver.dry_contact import solve_finite_roller_dry_contact
from ehl_solver.elasticity import cell_integrated_kernel
from ehl_solver.geometry import contact_geometry_from_config
from ehl_solver.hertz import elliptical_point_contact
from examples.run_roller_end_loading import band_loads, feature_cells, profiles


def _case(angle=0.0, radius_y=1.0, nodes=33):
    x = np.linspace(-4e-4, 4e-4, nodes)
    y = (np.arange(nodes) - (nodes - 1) / 2) * (1e-3 / nodes)
    geometry = contact_geometry_from_config(
        {
            "type": "profiled_roller",
            "radius_x_m": 0.01,
            "axial_profile": {"type": "parabolic", "radius_m": radius_y},
            "axial_tilt_rad": angle,
        }
    )
    return x, y, geometry


def _solve(angle=0.0, backend=None):
    x, y, geometry = _case(angle)
    result = solve_finite_roller_dry_contact(
        x, y, geometry, 1e-3, 100.0, 113e9, backend=backend
    )
    return x, y, geometry, result


def test_finite_roller_load_complementarity_symmetry_and_direct_elasticity():
    x, y, geometry, result = _solve()
    pressure = result.pressure_pa
    assert result.loads.force_n == pytest.approx(100.0, rel=2e-14)
    assert result.loads.moment_x_n_m == pytest.approx(0.0, abs=1e-14)
    assert result.relative_kkt_residual <= 2e-6
    assert np.min(pressure) >= 0.0
    np.testing.assert_allclose(pressure, pressure[::-1, :], atol=1e-5, rtol=1e-11)
    np.testing.assert_allclose(pressure, pressure[:, ::-1], atol=1e-5, rtol=1e-11)
    kernel = cell_integrated_kernel(pressure.shape, x[1] - x[0], y[1] - y[0], 113e9)
    yi, xi = np.indices(pressure.shape)
    for row, col in ((0, 0), (16, 16), (0, 16), (32, 32)):
        direct = np.sum(pressure * kernel[row - yi + 32, col - xi + 32])
        assert result.elastic_displacement_m[row, col] == pytest.approx(direct, rel=1e-12)
        expected_gap = geometry.h_geom(x, y)[row, col] + direct - result.approach_m
        assert result.gap_m[row, col] == pytest.approx(expected_gap, abs=1e-18)


def test_finite_roller_tilt_reverses_axial_pressure_and_moment():
    _, _, _, positive = _solve(1e-4)
    _, _, _, negative = _solve(-1e-4)
    np.testing.assert_allclose(
        positive.pressure_pa, negative.pressure_pa[::-1], atol=1e-5, rtol=1e-11
    )
    assert positive.loads.force_n == pytest.approx(100.0, rel=2e-14)
    assert negative.loads.force_n == pytest.approx(100.0, rel=2e-14)
    assert positive.loads.moment_x_n_m > 0.0
    assert positive.loads.moment_x_n_m == pytest.approx(-negative.loads.moment_x_n_m)


def test_finite_roller_strong_tilt_moves_contact_footprint():
    _, _, _, untilted = _solve()
    _, _, _, tilted = _solve(3e-3)
    assert np.any(untilted.pressure_pa[-1] > 0.0)
    assert np.all(tilted.pressure_pa[-1] == 0.0)
    assert tilted.loads.moment_x_n_m > 0.0
    assert np.any(tilted.pressure_pa[0] > 0.0)


def test_finite_roller_requires_grid_cell_edges_at_physical_ends():
    x, y, geometry = _case()
    with pytest.raises(ValueError, match="cell edges"):
        solve_finite_roller_dry_contact(x, y, geometry, 1.1e-3, 100.0, 113e9)


def test_finite_roller_rejects_truncated_rolling_contact():
    _, y, geometry = _case(nodes=17)
    x = np.linspace(-2e-5, 2e-5, 17)
    with pytest.raises(RuntimeError, match="computational boundary"):
        solve_finite_roller_dry_contact(x, y, geometry, 1e-3, 100.0, 113e9)


@pytest.mark.parametrize(
    "profile",
    [
        {"type": "cylindrical"},
        {"type": "parabolic", "radius_m": 1.0},
        {"type": "circular", "radius_m": 1.0},
        {
            "type": "logarithmic",
            "half_length_m": 0.0005,
            "scale_m": 1e-7,
            "maximum_relief_m": 5e-7,
        },
        {
            "type": "tabulated",
            "y_m": [-0.0005, 0.0, 0.0005],
            "height_m": [1e-7, 0.0, 1e-7],
        },
    ],
)
def test_symmetric_profiles_produce_symmetric_finite_dry_contact(profile):
    x, y, _ = _case(nodes=17)
    geometry = contact_geometry_from_config(
        {"type": "profiled_roller", "radius_x_m": 0.01, "axial_profile": profile}
    )
    result = solve_finite_roller_dry_contact(x, y, geometry, 1e-3, 100.0, 113e9)
    assert result.loads.force_n == pytest.approx(100.0, rel=2e-14)
    assert result.loads.moment_x_n_m == pytest.approx(0.0, abs=1e-14)
    assert result.relative_kkt_residual <= 2e-6
    np.testing.assert_allclose(
        result.pressure_pa, result.pressure_pa[::-1], atol=1e-5, rtol=1e-11
    )


def test_zero_tilt_is_exact_and_large_crown_radius_approaches_cylinder():
    x, y, _ = _case()
    base = {"type": "profiled_roller", "radius_x_m": 0.01}
    zero = contact_geometry_from_config(base).h_geom(x, y)
    with_zero_tilt = contact_geometry_from_config({**base, "axial_tilt_rad": 0.0})
    np.testing.assert_array_equal(zero, with_zero_tilt.h_geom(x, y))
    plain_result = solve_finite_roller_dry_contact(
        x, y, contact_geometry_from_config(base), 1e-3, 100.0, 113e9
    )
    zero_result = solve_finite_roller_dry_contact(
        x, y, with_zero_tilt, 1e-3, 100.0, 113e9
    )
    np.testing.assert_array_equal(plain_result.pressure_pa, zero_result.pressure_pa)
    for kind in ("parabolic", "circular"):
        geometry = contact_geometry_from_config(
            {**base, "axial_profile": {"type": kind, "radius_m": 1e12}}
        )
        assert np.max(np.abs(geometry.h_geom(x, y) - zero)) < 2e-19
        crowned_result = solve_finite_roller_dry_contact(
            x, y, geometry, 1e-3, 100.0, 113e9
        )
        np.testing.assert_allclose(
            crowned_result.pressure_pa, plain_result.pressure_pa, rtol=1e-9, atol=1e-3
        )


def test_profiled_parabola_has_exact_elliptical_hertz_limit():
    load, rx, ry, e_star = 5.0, 0.015, 0.06, 113e9
    reference = elliptical_point_contact(load, rx, ry, e_star)
    x = np.linspace(-3 * reference.semi_axis_x_m, 3 * reference.semi_axis_x_m, 65)
    dy = 6 * reference.semi_axis_y_m / 64
    y = (np.arange(65) - 32) * dy
    geometry = contact_geometry_from_config(
        {
            "type": "profiled_roller",
            "radius_x_m": rx,
            "axial_profile": {"type": "parabolic", "radius_m": ry},
        }
    )
    result = solve_finite_roller_dry_contact(x, y, geometry, 65 * dy, load, e_star)
    assert result.approach_m == pytest.approx(reference.approach_m, rel=3e-3)
    assert result.pressure_pa[32, 32] == pytest.approx(reference.peak_pressure_pa, rel=3e-3)
    assert np.all(result.pressure_pa[[0, -1]] == 0.0)


def test_finite_roller_cuda_matches_numpy_reference():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("No CUDA or ROCm GPU available")
    _, _, _, cpu = _solve(1e-4)
    _, _, _, gpu = _solve(1e-4, make_backend("torch", "cuda"))
    np.testing.assert_allclose(
        gpu.pressure_pa.cpu().numpy(), cpu.pressure_pa, rtol=2e-5, atol=1e-3
    )
    assert gpu.loads.force_n == pytest.approx(cpu.loads.force_n, rel=2e-14)
    assert gpu.loads.moment_x_n_m == pytest.approx(cpu.loads.moment_x_n_m, rel=1e-10)


def test_direct_compliance_matches_fft_and_load_controlled_contact():
    x, y, geometry = _case(nodes=9)
    dx, dy = x[1] - x[0], y[1] - y[0]
    direct = DirectHalfSpaceCompliance((9, 9), dx, dy, 113e9)
    fft = HalfSpaceFFTCompliance((9, 9), dx, dy, 113e9)
    pressure = np.random.default_rng(42).random((9, 9)) * 1e9
    np.testing.assert_allclose(direct(pressure), fft(pressure), rtol=2e-15, atol=1e-20)
    matrix_norm = np.linalg.norm(direct._matrix, ord=2)
    assert direct.pressure_norm_bound_m_per_pa >= matrix_norm * (1 - 1e-14)
    direct_result = solve_finite_roller_dry_contact(
        x, y, geometry, 1e-3, 100.0, 113e9, compliance=direct, tolerance=1e-8
    )
    fft_result = solve_finite_roller_dry_contact(
        x, y, geometry, 1e-3, 100.0, 113e9, tolerance=1e-8
    )
    np.testing.assert_allclose(
        direct_result.pressure_pa, fft_result.pressure_pa, rtol=2e-5, atol=100
    )
    assert direct_result.loads.force_n == pytest.approx(100.0, rel=1e-14)
    assert direct_result.relative_kkt_residual <= 1e-8


def test_compliance_rejects_mismatched_spacing_and_excess_memory():
    x, y, geometry = _case(nodes=9)
    dx, dy = x[1] - x[0], y[1] - y[0]
    compliance = DirectHalfSpaceCompliance((9, 9), 2 * dx, dy, 113e9)
    with pytest.raises(ValueError, match="spacing"):
        solve_finite_roller_dry_contact(x, y, geometry, 1e-3, 100.0, 113e9, compliance=compliance)
    with pytest.raises(MemoryError, match="budget"):
        DirectHalfSpaceCompliance((9, 9), dx, dy, 113e9, memory_limit_bytes=128)


def test_smooth_logarithmic_profile_is_bounded_monotone_and_c1():
    profile = contact_geometry_from_config(
        {
            "type": "profiled_roller",
            "radius_x_m": 0.01,
            "axial_profile": {
                "type": "smooth_logarithmic",
                "half_length_m": 5e-4,
                "scale_m": 7.5e-8,
                "maximum_relief_m": 1.25e-7,
                "blend_start_fraction": 0.8,
            },
        }
    ).axial_profile_m
    y = np.linspace(-5e-4, 5e-4, 10001)
    relief = profile(y)
    np.testing.assert_allclose(relief, relief[::-1], atol=1e-22)
    assert relief[len(y) // 2] == 0
    assert relief[-1] == pytest.approx(1.25e-7)
    assert np.all(np.diff(relief[len(y) // 2 :]) >= 0)
    eps = 1e-10
    for location in (0.8 * 5e-4, 5e-4):
        if location < 5e-4:
            values = profile(np.array([location - eps, location, location + eps]))
            assert (values[1] - values[0]) / eps == pytest.approx(
                (values[2] - values[1]) / eps, rel=2e-5
            )
        else:
            values = profile(np.array([location - eps, location]))
            assert abs((values[1] - values[0]) / eps) < 1e-8


def test_smooth_logarithmic_rejects_nonmonotone_blend():
    with pytest.raises(ValueError, match="monotone"):
        contact_geometry_from_config(
            {
                "type": "profiled_roller",
                "radius_x_m": 0.01,
                "axial_profile": {
                    "type": "smooth_logarithmic",
                    "half_length_m": 5e-4,
                    "scale_m": 1e-7,
                    "maximum_relief_m": 1e-8,
                    "blend_start_fraction": 0.8,
                },
            }
        )


def test_end_band_integral_handles_partial_cells_and_resolves_features():
    ny, nx = 37, 5
    dy, dx = 1e-3 / ny, 1e-5
    y = (np.arange(ny) - (ny - 1) / 2) * dy
    pressure = np.ones((ny, nx)) * 1e9
    for fraction in (0.02, 0.05, 0.1):
        expected = 1e9 * nx * dx * 1e-3 * fraction
        assert band_loads(pressure, dx, dy, y, fraction) == pytest.approx([expected, expected])
    assert feature_cells(pressure, y) == [4, 4]
    pressure[-4:] = 0
    assert feature_cells(pressure, y)[1] == 0


def test_study_profiles_match_end_relief_and_tabulate_hard_log():
    definitions = profiles()
    coordinates = np.linspace(-5e-4, 5e-4, 2001)
    values = {}
    for name, profile in definitions.items():
        geometry = contact_geometry_from_config(
            {"type": "profiled_roller", "radius_x_m": 0.01, "axial_profile": profile}
        )
        values[name] = geometry.axial_profile_m(coordinates)
        assert values[name][1000] == pytest.approx(0.0, abs=1e-20)
        expected_end = 0 if name == "cylindrical" else 125e-9
        assert values[name][0] == pytest.approx(expected_end, abs=1e-19)
        assert values[name][-1] == pytest.approx(expected_end, abs=1e-19)
    np.testing.assert_allclose(values["log_tabulated"], values["log_hard"], atol=1e-11, rtol=0)
