import numpy as np
import pytest

from examples.analyze_line_ehl_m20_grid_sensitivity import (
    _check_benchmark_signatures,
    _fixed_window_integrals,
    _profile_difference,
)


def test_profile_difference_reports_physical_rms_and_relative_l2():
    x = np.linspace(-1.0, 1.0, 5)
    coarse = np.ones(5)
    fine = np.full(9, 2.0)

    result = _profile_difference(x, coarse, np.linspace(-1.0, 1.0, 9), fine, (-0.5, 0.5))

    assert result["sample_count"] == 5
    assert result["rms_difference"] == pytest.approx(1.0)
    assert result["relative_l2_difference"] == pytest.approx(0.5)


def test_fixed_spike_window_integral_uses_fixed_bounds_and_baseline():
    x = np.linspace(0.0, 1.5, 16)
    pressure = 2.0e6 * x + 3.0e6
    b_m = 2.0e-4

    result = _fixed_window_integrals(x, pressure, b_m, (0.5, 1.0))

    expected_integral = (pressure[5] + pressure[10]) * 0.5 * (0.5 * b_m)
    assert result["pressure_integral_pa_m"] == pytest.approx(expected_integral)
    assert result["linear_baseline_excess_integral_pa_m"] == pytest.approx(0.0)


def test_fixed_window_baseline_respects_interpolated_endpoint_coordinates():
    x = np.linspace(0.0, 1.5, 16)
    pressure = 2.0e6 * x + 3.0e6

    result = _fixed_window_integrals(x, pressure, 2.0e-4, (0.53, 1.07))

    assert result["linear_baseline_excess_integral_pa_m"] == pytest.approx(
        0.0, abs=1e-12
    )


def test_profile_difference_rejects_empty_region():
    x = np.linspace(0.0, 1.0, 4)

    with pytest.raises(ValueError):
        _profile_difference(x, x, x, x, (-2.0, -1.0))


def test_comparison_rejects_mismatched_physics():
    def profile(nodes, load):
        return {
            "nodes": nodes,
            "summary": {
                "case_signature": {
                    "inputs_si": {"line_load_n_m": load},
                    "dimensionless_groups": {"M": 20.0, "L": 10.0},
                }
            },
        }

    with pytest.raises(ValueError, match="different benchmark physics"):
        _check_benchmark_signatures([profile(257, 1.0), profile(513, 2.0)])
