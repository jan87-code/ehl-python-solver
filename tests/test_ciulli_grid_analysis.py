import json
from pathlib import Path

import numpy as np
import pytest

from ehl_solver.hertz import circular_point_contact
from ehl_solver.limited_audit import audit_limited_state
from ehl_solver.limited_newton import solve_limited_load
from ehl_solver.limited_system import LimitedLoadProbe
from ehl_solver.prototype import PointContactCase, hertz_guess
from examples.analyze_ciulli_donor_grid import (
    _fixed_spike_region_pressure,
    _midplane_comparison,
    _outlet_integrated_pressure,
    _pairwise_field_archive,
    _scalar_refinement,
    main_contact_structure_comparison,
    pairwise_513_to_1025_convergence,
    profile_error_metrics,
    regional_mesh_comparison,
)
from examples.run_ciulli_constitutive_sensitivity import (
    _BarusFluid,
    _ConstitutiveCase,
)


def test_profile_error_metrics_report_signed_bias_and_digitization_coverage():
    result = profile_error_metrics([0.4, 0.5], [0.41, 0.47], [0.02, 0.03])
    assert result["bias_um"] == pytest.approx(-0.01)
    assert result["rmse_um"] == pytest.approx(np.sqrt(0.0005))
    assert result["maximum_absolute_error_um"] == pytest.approx(0.03)
    assert result["points_within_digitization_estimate"] == 2
    assert result["point_count"] == 2


def test_profile_error_metrics_reject_mismatched_or_invalid_profiles():
    with pytest.raises(ValueError):
        profile_error_metrics([0.4], [0.4, 0.5], [0.02])
    with pytest.raises(ValueError):
        profile_error_metrics([0.4], [float("nan")], [0.02])
    with pytest.raises(ValueError):
        profile_error_metrics([0.4], [0.4], [-0.02])


def test_midplane_comparison_reports_common_coordinate_units():
    axis = np.array([-100e-6, 0.0, 100e-6])
    coarse = {
        "x_m": axis,
        "y_m": axis,
        "pressure_pa": np.full((3, 3), 100e6),
        "film_m": np.full((3, 3), 100e-9),
    }
    fine = {
        "x_m": axis,
        "y_m": axis,
        "pressure_pa": np.full((3, 3), 101e6),
        "film_m": np.full((3, 3), 101e-9),
    }
    result = _midplane_comparison(coarse, fine)
    assert result["longitudinal"]["pressure_pa"]["rms"] == pytest.approx(1.0)
    assert result["transverse"]["pressure_pa"]["linf"] == pytest.approx(1.0)
    assert result["longitudinal"]["film_m"]["rms"] == pytest.approx(1.0)
    assert result["transverse"]["film_m"]["linf"] == pytest.approx(1.0)


def test_midplane_comparison_samples_the_declared_si_window():
    axis = np.linspace(-400e-6, 400e-6, 5)
    xx, yy = np.meshgrid(axis, axis)
    pressure = 600e6 + 1e12 * xx + 0.3e12 * yy
    fine_pressure = pressure + 1e12 * xx
    film = 1e-6 + 1e-3 * xx
    fine_film = film + 1e-3 * xx
    coarse = {
        "x_m": axis,
        "y_m": axis,
        "pressure_pa": pressure,
        "film_m": film,
    }
    fine = {
        "x_m": axis,
        "y_m": axis,
        "pressure_pa": fine_pressure,
        "film_m": fine_film,
    }
    result = _midplane_comparison(coarse, fine)
    expected_rms_mpa = np.sqrt(np.mean(np.linspace(-100, 100, 401) ** 2))
    assert result["longitudinal"]["pressure_pa"]["rms"] == pytest.approx(
        expected_rms_mpa
    )
    assert result["transverse"]["pressure_pa"]["rms"] == pytest.approx(0.0, abs=1e-12)
    assert result["longitudinal"]["film_m"]["rms"] == pytest.approx(
        expected_rms_mpa
    )
    assert result["transverse"]["film_m"]["rms"] == pytest.approx(0.0, abs=1e-12)


def test_regional_comparison_separates_main_contact_and_outlet_error():
    base_case = json.loads(
        (Path(__file__).parents[1] / "cases/ciulli_15n_donor.json").read_text()
    )
    x_coarse = np.linspace(-0.4e-3, 0.4e-3, 5)
    y_coarse = np.linspace(-0.4e-3, 0.4e-3, 5)
    x_fine = np.linspace(-0.4e-3, 0.4e-3, 41)
    y_fine = np.linspace(-0.4e-3, 0.4e-3, 9)
    coarse = {
        "case": {**base_case, "nx": 5, "ny": 5},
        "x_m": x_coarse,
        "y_m": y_coarse,
        "pressure_pa": np.full((5, 5), 10e6),
        "film_m": np.full((5, 5), 100e-9),
        "theta": np.ones((5, 5)),
    }
    xx, yy = np.meshgrid(x_fine, y_fine)
    pressure_fine = np.full((9, 41), 10e6)
    pressure_fine[xx > 60e-6] += 2e6
    fine = {
        "case": {**base_case, "nx": 41, "ny": 9},
        "x_m": x_fine,
        "y_m": y_fine,
        "pressure_pa": pressure_fine,
        "film_m": np.full((9, 41), 101e-9),
        "theta": np.ones((9, 41)),
    }
    result = regional_mesh_comparison(coarse, fine, spacing_um=1.0)
    regions = result["regions"]
    assert regions["main_contact_excluding_spike"]["pressure_relative_l2"] == pytest.approx(
        0.0, abs=1e-14
    )
    assert regions["outlet_spike"]["pressure_relative_l2"] > 0
    assert regions["full_contact"]["film_rms_nm"] == pytest.approx(1.0)
    assert sum(result["contact_pressure_error_energy_share"].values()) == pytest.approx(
        100.0
    )
    assert result["fields"]["pressure_difference_pa"].shape == (151, 276)
    structure = main_contact_structure_comparison(coarse, fine, spacing_um=1.0)
    assert structure["hertz_contact_radius_um"] == pytest.approx(100.0)
    assert "x_peak_5_um" in structure[
        "pressure_ridge_maxima_x_0_to_60_um_by_lateral_coordinate"
    ][0]
    assert "x_peak_41_um" in structure[
        "pressure_ridge_maxima_x_0_to_60_um_by_lateral_coordinate"
    ][0]
    assert sum(
        structure["regions"][key]["pressure_error_energy_share_of_main_percent"]
        for key in ("hertz_core", "transition")
    ) == pytest.approx(100.0)
    assert structure["hotspot_and_cavitation"]["cavitation_state_mismatch_percent_in_main"] == 0
    pairwise_fields = _pairwise_field_archive([coarse, fine])
    assert pairwise_fields["5_to_41_pressure_difference_pa"].shape == (301, 551)


def test_scalar_refinement_reports_order_only_for_monotone_contraction():
    result = _scalar_refinement([1.25, 1.0625, 1.015625])
    assert result["observed_order"] == pytest.approx(2.0)
    assert result["richardson_extrapolated_value"] == pytest.approx(1.0)
    assert result["fine_grid_gci_percent_fs_1_25"] > 0

    oscillatory = _scalar_refinement([1.0, 1.2, 1.1])
    assert not oscillatory["monotone_contraction"]
    assert oscillatory["observed_order"] is None
    assert oscillatory["richardson_extrapolated_value"] is None

    four_grid = _scalar_refinement(
        [8.0, 4.0, 3.0, 2.75], [65, 129, 257, 513]
    )
    assert four_grid["observed_orders_between_successive_changes"] == pytest.approx(
        [2.0, 2.0]
    )
    assert four_grid["observed_order"] == pytest.approx(2.0)
    assert four_grid["richardson_extrapolated_value"] == pytest.approx(8 / 3)
    assert four_grid["fine_grid_gci_percent_fs_1_25"] > 0

    four_grid_oscillatory = _scalar_refinement(
        [1.0, 1.2, 1.1, 1.05], [65, 129, 257, 513]
    )
    assert not four_grid_oscillatory["monotone_contraction"]
    assert four_grid_oscillatory["observed_order"] is None


def test_513_to_1025_analysis_rejects_other_grid_pairs():
    runs = [{"case": {"nx": 257}}, {"case": {"nx": 513}}]
    with pytest.raises(ValueError, match="513x513 and 1025x1025"):
        pairwise_513_to_1025_convergence(runs)


def test_outlet_integral_reports_absolute_and_baseline_subtracted_pressure():
    axis = np.linspace(-400e-6, 400e-6, 41)
    xx, yy = np.meshgrid(axis, axis)
    pressure = 100e6 + 100e6 * np.exp(-((xx - 100e-6) / 10e-6) ** 2)
    case = json.loads(
        (Path(__file__).parents[1] / "cases/ciulli_15n_donor.json").read_text()
    )
    run = {
        "case": case,
        "x_m": axis,
        "y_m": axis,
        "pressure_pa": pressure,
    }
    result = _outlet_integrated_pressure(
        run,
        {
            "identified": True,
            "peak_x_m": 100e-6,
            "half_prominence_width_m": 40e-6,
        },
    )
    assert result["defined"]
    assert result["pressure_integral_n"] > 0
    assert result["centerline_pressure_integral_pa_m"] > 0
    assert result["centerline_excess_above_local_linear_baseline_pa_m"] > 0


def test_fixed_spike_region_integral_uses_common_physical_window():
    axis = np.linspace(-400e-6, 400e-6, 41)
    case = json.loads(
        (Path(__file__).parents[1] / "cases/ciulli_15n_donor.json").read_text()
    )
    run = {
        "case": case,
        "x_m": axis,
        "y_m": axis,
        "pressure_pa": np.full((41, 41), 100e6),
    }
    result = _fixed_spike_region_pressure(run)
    hertz_radius = circular_point_contact(
        case["load_n"], case["radius_m"], case["contact_modulus_pa"]
    ).radius_m
    expected = 100e6 * 25e-6 * 2 * hertz_radius
    assert result["pressure_integral_n"] == pytest.approx(expected, rel=1e-12)
    assert result["x_window_um"] == pytest.approx([75.0, 100.0])


def test_barus_sensitivity_fluid_matches_reported_zero_pressure_slope():
    case = PointContactCase(
        **json.loads(
            (Path(__file__).parents[1] / "cases/ciulli_15n_donor.json").read_text()
        )
    )
    fluid = _BarusFluid(case, constant_density=False)
    rho, eta, drho, deta = fluid.properties(np.array([0.0, 1e6]))
    assert rho[0] == pytest.approx(case.density_kg_m3)
    assert eta[0] == pytest.approx(case.viscosity_pa_s)
    assert deta[0] / eta[0] == pytest.approx(case.alpha_per_pa)
    assert rho[1] > rho[0]
    assert drho[1] > 0

    constant_density = _BarusFluid(case, constant_density=True)
    rho_constant, _, drho_constant, _ = constant_density.properties(np.array([1e8]))
    assert rho_constant[0] == pytest.approx(case.density_kg_m3)
    assert drho_constant[0] == 0


def test_constitutive_override_reaches_the_limited_probe_equations():
    case = PointContactCase(
        **json.loads(
            (Path(__file__).parents[1] / "cases/ciulli_15n_donor.json").read_text()
        )
    )
    state = hertz_guess(case, case.system())[0]
    baseline_residual = LimitedLoadProbe(
        case, transport_fraction=0.0
    ).residual(state)
    alternative = _ConstitutiveCase(
        case, _BarusFluid(case, constant_density=False)
    )
    alternative_residual = LimitedLoadProbe(
        alternative, transport_fraction=0.0
    ).residual(state)
    assert not np.allclose(baseline_residual, alternative_residual)


def test_ciulli_33_grid_donor_regression_closes_all_independent_equations():
    case = PointContactCase(
        **json.loads(
            (Path(__file__).parents[1] / "cases/ciulli_15n_donor.json").read_text()
        )
    )
    root = solve_limited_load(case, transport_fraction=0.0)
    audit = audit_limited_state(
        LimitedLoadProbe(case, transport_fraction=0.0), root.state, direct=True
    )
    assert audit["numerically_accepted"]
    assert audit["force_n"] == pytest.approx(15.0, abs=1e-10)
    assert audit["mass"]["mass_maximum"] <= 1e-8
    assert audit["mass"]["relative_local_balance"] <= 1e-10
    assert audit["mass"]["relative_global_balance"] <= 1e-10
    assert audit["mass"]["maximum_complementarity"] <= 1e-10
    assert audit["film_reconstruction_error_m"] < 1e-18
    assert audit["minimum_film_m"] == pytest.approx(216.7062e-9, rel=0.01)
    assert audit["central_film_m"] == pytest.approx(456.1365e-9, rel=0.01)
    pressure_pa, _, _ = LimitedLoadProbe(case, transport_fraction=0.0).fields(root.state)
    assert np.max(pressure_pa) == pytest.approx(760.6064e6, rel=0.01)
