import json

import numpy as np
import pytest

from ehl_solver.pressure_analysis import (
    centerline_peak_diagnostics,
    centerline_pressure_profile,
    compare_pressure_profiles,
    outlet_spike_diagnostics,
)
from examples.analyze_pressure_field import analyze


def test_centerline_profile_interpolates_to_physical_y_zero():
    x = np.array([-1.0, 0.5, 2.0])
    y = np.array([-2.0, 1.0, 3.0])
    pressure = np.array([2 * x - 2, 2 * x + 1, 2 * x + 3])

    actual_x, actual_pressure = centerline_pressure_profile(x, y, pressure)

    np.testing.assert_array_equal(actual_x, x)
    np.testing.assert_allclose(actual_pressure, 2 * x)


def test_common_coordinate_comparison_is_independent_of_grid_indices():
    reference_x = np.linspace(-1, 1, 5)
    comparison_x = np.linspace(-1, 1, 9)

    result = compare_pressure_profiles(
        reference_x,
        3 + 2 * reference_x,
        comparison_x,
        3 + 2 * comparison_x,
        window_m=(-0.75, 0.75),
        sample_count=101,
    )

    assert result["relative_l2"] < 1e-14
    assert result["maximum_absolute_difference_pa"] < 1e-14


def test_spike_diagnostic_reports_width_and_cell_resolution():
    x = np.arange(11, dtype=np.float64)
    pressure = np.array([0, 0, 1, 2, 4, 8, 4, 2, 1, 0, 0], dtype=np.float64)

    result = outlet_spike_diagnostics(x, pressure, window_m=(2.0, 8.0))

    assert result["identified"]
    assert result["peak_x_m"] == 5
    assert result["peak_pressure_pa"] == 8
    assert result["left_adjacent_pressure_pa"] == 4
    assert result["right_adjacent_pressure_pa"] == 4
    assert result["half_prominence_width_m"] == pytest.approx(1.75)
    assert result["cells_across_width"] == pytest.approx(1.75)


def test_spike_diagnostic_does_not_invent_peak_for_monotone_release():
    x = np.arange(8, dtype=np.float64)
    pressure = np.array([8, 6, 4, 2, 0, 0, 0, 0], dtype=np.float64)

    result = outlet_spike_diagnostics(x, pressure, window_m=(2.0, 7.0))

    assert result["identified"] is False
    assert "No interior peak" in result["reason"]


def test_common_coordinate_comparison_rejects_nonoverlap():
    with pytest.raises(ValueError, match="shared physical window"):
        compare_pressure_profiles(
            np.array([0, 1]),
            np.array([1, 0]),
            np.array([2, 3]),
            np.array([1, 0]),
            window_m=(0, 1),
        )


def test_centerline_requires_axis_to_span_zero():
    with pytest.raises(ValueError, match="spanning y=0"):
        centerline_pressure_profile(
            np.array([0, 1]),
            np.array([1, 2]),
            np.ones((2, 2)),
        )


def test_centerline_diagnostics_reports_local_peak_and_support():
    result = centerline_peak_diagnostics(
        np.array([-2, -1, 0, 1, 2], dtype=np.float64),
        np.array([0, 2, 4, 3, 0], dtype=np.float64),
    )

    assert result["maximum"] == {"x_m": 0.0, "pressure_pa": 4.0}
    assert result["interior_local_maxima"] == [
        {
            "x_m": 0.0,
            "pressure_pa": 4.0,
            "left_adjacent_pressure_pa": 2.0,
            "right_adjacent_pressure_pa": 3.0,
        }
    ]
    assert result["positive_pressure_extent_m"] == [-1.0, 1.0]


def test_field_analyzer_rejects_nonaccepted_iterates(tmp_path):
    field = tmp_path / "state.npz"
    np.savez(
        field,
        pressure_pa=np.ones((3, 3)),
        x_m=np.array([-1.0, 0.0, 1.0]),
        y_m=np.array([-1.0, 0.0, 1.0]),
    )
    (tmp_path / "summary.json").write_text(
        json.dumps({"status": "rejected_stage", "audit": {"numerically_accepted": False}})
    )

    with pytest.raises(ValueError, match="not an accepted root"):
        analyze(field)
