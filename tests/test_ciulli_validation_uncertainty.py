import numpy as np
import pytest

from examples.analyze_ciulli_validation_uncertainty import (
    EXPECTED_GRIDS,
    _minimum_position_offsets,
    _pair_metrics,
    _profile_error_summary,
    _profile_region_assessment,
    _validate_sequence,
)


def _case(grid):
    return {
        "nx": grid,
        "ny": grid,
        "load_n": 15.0,
        "radius_m": 5.1595e-3,
        "contact_modulus_pa": 58.044375e9,
        "viscosity_pa_s": 0.497,
        "density_kg_m3": 892.0,
        "entrainment_m_s": 0.2,
        "alpha_per_pa": 2.87e-8,
        "domain_half_width_x_m": 0.4e-3,
        "domain_half_width_y_m": 0.4e-3,
        "transport": "donor",
    }


def test_validation_sequence_requires_all_five_nested_donor_grids():
    runs = [{"case": _case(grid)} for grid in EXPECTED_GRIDS]
    assert [run["case"]["nx"] for run in _validate_sequence(runs)] == list(
        EXPECTED_GRIDS
    )

    with pytest.raises(ValueError, match="Expected square donor grids"):
        _validate_sequence(runs[:-1])

    wrong_transport = [{"case": _case(grid)} for grid in EXPECTED_GRIDS]
    wrong_transport[-1]["case"]["transport"] = "limited"
    with pytest.raises(ValueError, match="identical donor physics"):
        _validate_sequence(wrong_transport)


def test_pair_metrics_separate_bulk_pressure_and_film_profile_changes():
    axis_coarse = np.linspace(-0.4e-3, 0.4e-3, 5)
    axis_fine = np.linspace(-0.4e-3, 0.4e-3, 9)
    coarse = {
        "case": _case(5),
        "x_m": axis_coarse,
        "y_m": axis_coarse,
        "pressure_pa": np.full((5, 5), 100e6),
        "film_m": np.full((5, 5), 100e-9),
        "theta": np.ones((5, 5)),
        "audit": {"central_film_m": 100e-9},
    }
    fine = {
        "case": _case(9),
        "x_m": axis_fine,
        "y_m": axis_fine,
        "pressure_pa": np.full((9, 9), 101e6),
        "film_m": np.full((9, 9), 101e-9),
        "theta": np.ones((9, 9)),
        "audit": {"central_film_m": 101e-9},
    }

    result = _pair_metrics(coarse, fine)
    assert result["scalars"]["h_c_nm"]["change"] == pytest.approx(1.0)
    assert result["regions"]["main_contact_excluding_spike"][
        "pressure_rms_mpa_over_positive_support"
    ] == pytest.approx(1.0)
    assert result["midplane_profiles"]["longitudinal"]["film_m"]["rms"] == pytest.approx(
        1.0
    )
    assert "not Richardson" in result["interpretation"]


def test_profile_error_summary_reports_bias_error_and_digitization_coverage():
    result = _profile_error_summary(
        np.array([100.0, 110.0, 120.0]),
        np.array([101.0, 108.0, 123.0]),
        np.array([5.0, 2.0, 4.0]),
    )
    assert result["rmse_nm"] == pytest.approx(np.sqrt(14 / 3))
    assert result["bias_nm"] == pytest.approx(2 / 3)
    assert result["maximum_absolute_deviation_nm"] == pytest.approx(3.0)
    assert result["digitization_rms_nm"] == pytest.approx(np.sqrt(15))
    assert result["points_within_digitization_estimate"] == 3


def test_regional_profile_metrics_use_unique_points_and_separate_grid_sensitivity():
    coordinates = {
        "longitudinal": [-70.0, 0.0, 80.0, 100.0],
        "transverse": [-80.0, -60.0, 0.0, 60.0, 80.0],
    }
    offsets = dict(zip(EXPECTED_GRIDS, range(len(EXPECTED_GRIDS)), strict=True))
    rows = []
    for grid in EXPECTED_GRIDS:
        for direction, points in coordinates.items():
            for coordinate in points:
                rows.append(
                    {
                        "direction": direction,
                        "coordinate_um": coordinate,
                        "grid_n": grid,
                        "observed_film_nm": 100.0,
                        "digitization_uncertainty_nm": 2.0,
                        "predicted_film_nm": 100.0 + offsets[grid],
                    }
                )

    result = _profile_region_assessment(rows)
    by_name = {
        row["region"]: row for row in result["agreement_1025"]
    }
    assert by_name["longitudinal_bulk"]["point_count"] == 2
    assert by_name["longitudinal_outlet_constriction"]["point_count"] == 2
    assert by_name["transverse_bulk"]["point_count"] == 3
    assert by_name["transverse_side_constrictions"]["point_count"] == 2
    assert by_name["longitudinal_bulk"]["rmse_nm"] == pytest.approx(4.0)
    assert by_name["longitudinal_bulk"]["bias_nm"] == pytest.approx(4.0)

    final_pair = [
        row
        for row in result["adjacent_grid_sensitivity_at_digitized_coordinates"]
        if row["region"] == "transverse_side_constrictions"
    ][-1]
    assert final_pair["point_count"] == 2
    assert final_pair["grid_change_rms_nm"] == pytest.approx(1.0)
    assert final_pair["digitization_rms_nm"] == pytest.approx(2.0)
    assert final_pair["grid_rms_over_digitization_rms"] == pytest.approx(0.5)


def test_transverse_cut_minimum_offsets_match_each_symmetric_side():
    assert _minimum_position_offsets(
        "transverse", 85.9375, [-80.0, 80.0]
    ) == pytest.approx([-5.9375, 5.9375])
