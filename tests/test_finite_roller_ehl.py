import numpy as np
import pytest
from scipy.integrate import trapezoid

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


def test_accepted_state_transfer_to_axially_refined_grid_preserves_feasibility():
    source_case = FiniteRollerEHLCase(
        nx=17, ny=33, axial_tilt_rad=0.0005
    )
    source_state = finite_roller_hertz_seed(source_case)
    target_case = FiniteRollerEHLCase(
        nx=17, ny=49, axial_tilt_rad=0.0005
    )
    target_system = target_case.system()
    target_state = transfer_finite_roller_seed(
        source_case, source_state, target_case, target_system
    )
    target_probe = LimitedLoadProbe(target_case, transport_fraction=0.0)
    pressure, theta, film = target_probe.fields(target_state)

    assert target_probe.feasible(target_state)
    assert target_system.problem.load_n(pressure) == pytest.approx(target_case.load_n)
    assert pressure[[0, -1], :].max() == 0.0
    assert np.all((theta >= 0.0) & (theta <= 1.0))
    assert film.min() > target_system.problem.film_floor_m


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
    axial_boundary_flows = metrics["audit"]["mass"]["outward_mass_flow_kg_s"][2:4]
    expected_axial_imbalance = abs(
        axial_boundary_flows[0] - axial_boundary_flows[1]
    ) / max(sum(abs(flow) for flow in axial_boundary_flows), 1e-300)
    assert metrics["relative_axial_boundary_flow_imbalance"] == pytest.approx(
        expected_axial_imbalance
    )
    assert metrics["relative_axial_boundary_flow_imbalance"] == pytest.approx(
        0.0, abs=1e-14
    )
    assert metrics["outer_band_load_n_left_right"]["2"][0] == pytest.approx(
        metrics["outer_band_load_n_left_right"]["2"][1]
    )
    assert set(metrics["pressurized_support_clearance_by_peak_fraction"]) == {
        "left",
        "right",
    }
    assert profiles["qx_centerline_kg_m_s"].shape == (case.nx - 1,)
    assert profiles["qy_centerline_kg_m_s"].shape == (case.ny - 1,)


def test_finite_roller_metrics_preserve_offset_and_recovered_angle_in_border_state():
    case = small_case(axial_tilt_rad=0.0005)
    seed = finite_roller_hertz_seed(case)
    probe = LimitedLoadProbe(
        case, transport_fraction=0.0, moment_target_n_m=0.0
    )
    controlled_state = np.r_[seed, 0.8]
    metrics, _ = finite_roller_metrics(case, probe, controlled_state)
    assert metrics["h0_m"] == pytest.approx(
        seed[-1] * probe.system.problem.film_scale_m
    )
    assert metrics["axial_tilt_rad"] == pytest.approx(0.8 * probe.angle_scale_rad)
    assert metrics["moment_target_n_m"] == 0.0
    assert metrics["moment_about_x_n_m"] == metrics["physical_contact_moment_x_n_m"]
    assert metrics["pressure_y_first_moment_n_m"] == pytest.approx(
        -metrics["physical_contact_moment_x_n_m"]
    )
    assert metrics["moment_residual_n_m"] == pytest.approx(
        metrics["physical_contact_moment_x_n_m"]
    )


def test_finite_roller_metrics_integrate_a_fixed_physical_spike_window():
    case = small_case()
    probe = LimitedLoadProbe(case, transport_fraction=0.0)
    state = finite_roller_hertz_seed(case)
    x_m = probe.system.problem.x_m
    y_m = probe.system.problem.y_m
    pressure, _, _ = probe.fields(state)
    bounds = (
        float((x_m[3] + x_m[4]) / 2.0),
        float((x_m[6] + x_m[7]) / 2.0),
    )

    metrics, _ = finite_roller_metrics(
        case, probe, state, spike_window_m=bounds
    )

    coordinates = np.r_[
        bounds[0],
        x_m[(x_m > bounds[0]) & (x_m < bounds[1])],
        bounds[1],
    ]
    pressure_in_window = np.vstack(
        [np.interp(coordinates, x_m, row) for row in pressure]
    )
    expected_centerline = trapezoid(
        np.interp(coordinates, x_m, pressure[len(y_m) // 2]), x=coordinates
    )
    expected_load = trapezoid(
        trapezoid(pressure_in_window, x=coordinates, axis=1), x=y_m
    )
    window = metrics["fixed_spike_window"]
    assert window["x_bounds_m"] == list(bounds)
    assert window["centerline_pressure_integral_pa_m"] == pytest.approx(
        expected_centerline
    )
    assert window["integrated_pressure_load_n"] == pytest.approx(expected_load)

    with pytest.raises(ValueError, match="within"):
        finite_roller_metrics(
            case,
            probe,
            state,
            spike_window_m=(float(x_m[0] - 1e-9), float(x_m[2])),
        )


def test_finite_roller_case_allows_the_bounded_769_by_513_grid():
    case = FiniteRollerEHLCase(
        nx=769,
        ny=513,
        maximum_physical_nodes=450000,
    )
    assert case.nx * case.ny <= case.maximum_physical_nodes
    with pytest.raises(ValueError, match="exceeds"):
        FiniteRollerEHLCase(
            nx=769,
            ny=513,
            maximum_physical_nodes=300000,
        )


def test_finite_roller_case_allows_769_by_577_within_node_budget():
    case = FiniteRollerEHLCase(
        nx=769,
        ny=577,
        maximum_physical_nodes=450000,
    )
    assert case.nx * case.ny == 443713
    with pytest.raises(ValueError, match="exceeds"):
        FiniteRollerEHLCase(
            nx=769,
            ny=577,
            maximum_physical_nodes=300000,
        )


def test_finite_roller_cuda_fields_and_donor_residual_match_cpu():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is optional for CPU-only test environments")
    case = small_case()
    state = finite_roller_hertz_seed(case)
    probe = LimitedLoadProbe(case, transport_fraction=0.0)
    parity = verify_finite_roller_cuda_parity(case, state, probe)
    assert parity["passed"]
