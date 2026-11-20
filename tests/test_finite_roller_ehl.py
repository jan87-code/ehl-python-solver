import numpy as np
import pytest

from ehl_solver.finite_roller_ehl import (
    FiniteRollerEHLCase,
    finite_roller_hertz_seed,
    finite_roller_metrics,
    mirror_finite_roller_seed,
    transfer_finite_roller_seed,
    verify_finite_roller_cuda_parity,
)
from ehl_solver.geometry import BlendedParabolicAxialProfile
from ehl_solver.limited_system import LimitedLoadProbe


def small_case(**kwargs):
    return FiniteRollerEHLCase(nx=17, ny=33, **kwargs)


def test_blended_parabolic_profile_is_bounded_symmetric_and_c1():
    half_length = 0.5e-3
    relief = 2e-6
    profile = BlendedParabolicAxialProfile(half_length, relief, 0.1)
    y = np.linspace(-half_length, half_length, 1001)
    g = profile(y)
    assert g[500] == 0.0
    assert g[0] == pytest.approx(relief)
    assert np.all(g >= 0.0)
    assert np.all(g <= relief)
    np.testing.assert_allclose(g, g[::-1], rtol=0.0, atol=1e-18)

    start = 0.8 * half_length
    step = 1e-9
    values = profile(np.array([start - step, start, start + step, half_length - step]))
    assert values[1] == pytest.approx(relief * 0.8**2)
    assert abs(values[2] - values[1]) / step == pytest.approx(
        2 * relief * 0.8 / half_length, rel=2e-5
    )
    assert (relief - values[3]) / step < 1e-4
    with pytest.raises(ValueError, match="outside"):
        profile(np.array([half_length + 1e-9]))


def test_finite_roller_case_composes_crown_and_tilt_on_rectangular_grid():
    case = small_case(axial_tilt_rad=0.00025)
    problem = case.problem()
    assert problem.shape == (33, 17)
    assert problem.x_m.size != problem.y_m.size
    center = problem.rigid_geometry_m[case.ny // 2, case.nx // 2]
    assert center == pytest.approx(0.0, abs=1e-18)
    expected_end = case.end_relief_m + case.half_length_m * np.tan(case.axial_tilt_rad)
    assert problem.rigid_geometry_m[-1, case.nx // 2] == pytest.approx(expected_end)
    assert problem.rigid_geometry_m[0, case.nx // 2] == pytest.approx(
        case.end_relief_m - case.half_length_m * np.tan(case.axial_tilt_rad)
    )
    expected_lambda = (
        12
        * case.viscosity_pa_s
        * case.entrainment_m_s
        * problem.length_scale_m
        / (problem.film_scale_m**2 * problem.pressure_scale_pa)
    )
    assert problem.reynolds_parameter == pytest.approx(expected_lambda)


def test_hertz_seed_is_feasible_and_tilt_transfer_preserves_pressure_state():
    source_case = small_case()
    source_system = source_case.system()
    seed = finite_roller_hertz_seed(source_case, source_system)
    source_probe = LimitedLoadProbe(source_case, transport_fraction=0.0)
    assert source_probe.feasible(seed)
    pressure, theta, film = source_probe.fields(seed)
    assert source_system.problem.load_n(pressure) == pytest.approx(source_case.load_n)
    assert film.min() > source_system.problem.film_floor_m
    assert theta.min() == pytest.approx(1.0)

    target_case = small_case(axial_tilt_rad=0.00025)
    target_system = target_case.system()
    transferred = transfer_finite_roller_seed(
        source_case, seed, target_case, target_system
    )
    target_probe = LimitedLoadProbe(target_case, transport_fraction=0.0)
    assert target_probe.feasible(transferred)
    np.testing.assert_allclose(transferred[:-1], seed[:-1], rtol=1e-14, atol=1e-14)
    expected_offset_change = target_case.half_length_m * np.tan(target_case.axial_tilt_rad)
    assert (
        transferred[-1] - seed[-1]
    ) * target_system.problem.film_scale_m == pytest.approx(expected_offset_change)


def test_mirrored_seed_reverses_axial_fields_for_opposite_tilt():
    positive = small_case(axial_tilt_rad=0.00025)
    negative = small_case(axial_tilt_rad=-0.00025)
    positive_state = finite_roller_hertz_seed(positive)
    negative_state = mirror_finite_roller_seed(positive, positive_state, negative)
    n = (positive.ny - 2) * (positive.nx - 2)
    positive_pressure = positive_state[:n].reshape(positive.ny - 2, positive.nx - 2)
    negative_pressure = negative_state[:n].reshape(negative.ny - 2, negative.nx - 2)
    np.testing.assert_allclose(negative_pressure, positive_pressure[::-1], rtol=1e-14, atol=1e-14)
    assert negative_state[-1] == pytest.approx(positive_state[-1])


def test_finite_roller_metrics_include_end_rows_windows_and_flux_profiles():
    case = small_case()
    probe = LimitedLoadProbe(case, transport_fraction=0.0)
    state = finite_roller_hertz_seed(case)
    metrics, profiles = finite_roller_metrics(case, probe, state)
    assert metrics["load_integrated_n"] == pytest.approx(case.load_n)
    assert metrics["outermost_axial_row_load_n"] == [0.0, 0.0]
    assert metrics["boundary_pressure_max_pa"] == 0.0
    assert metrics["outer_band_load_n_left_right"]["2"][0] == pytest.approx(
        metrics["outer_band_load_n_left_right"]["2"][1]
    )
    assert set(metrics["pressurized_support_clearance_by_peak_fraction"]) == {
        "left",
        "right",
    }
    assert profiles["qx_centerline_kg_m_s"].shape == (case.nx - 1,)
    assert profiles["qy_centerline_kg_m_s"].shape == (case.ny - 1,)


def test_finite_roller_cuda_fields_and_donor_residual_match_cpu():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is optional for CPU-only test environments")
    case = small_case()
    state = finite_roller_hertz_seed(case)
    probe = LimitedLoadProbe(case, transport_fraction=0.0)
    parity = verify_finite_roller_cuda_parity(case, state, probe)
    assert parity["passed"]
