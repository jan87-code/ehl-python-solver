import csv

import numpy as np
import pytest

from examples.run_roller_relief_tilt import (
    HALF,
    edge_metrics,
    make_profile,
    refinement_gate,
    screen_gate,
    systematic_end_growth,
    write_table,
)


@pytest.mark.parametrize("kind", ["parabolic", "logarithmic"])
@pytest.mark.parametrize("zone", [0.1, 0.2, 0.3])
def test_study_profiles_are_symmetric_bounded_and_c1(kind, zone):
    profile = make_profile(kind, 1500, zone)
    y = np.linspace(-HALF, HALF, 10001)
    relief = profile(y)
    np.testing.assert_allclose(relief, relief[::-1], atol=1e-20)
    assert relief[len(y) // 2] == 0
    assert relief[-1] == pytest.approx(1500e-9)
    assert np.all(np.diff(relief[len(y) // 2 :]) >= 0)
    point, step = HALF * (1 - 2 * zone), 1e-10
    samples = profile(np.array([point - step, point, point + step]))
    assert (samples[1] - samples[0]) / step == pytest.approx(
        (samples[2] - samples[1]) / step, rel=3e-5
    )
    end = profile(np.array([HALF - step, HALF]))
    assert abs((end[1] - end[0]) / step) < 1e-6


def test_edge_distance_counts_empty_cells_instead_of_centres():
    y = (np.arange(21) - 10) * (1e-3 / 21)
    p = np.zeros((21, 5))
    p[5:15, 2] = 100
    metrics = edge_metrics(p, y)
    assert metrics["edge_distance_cells"] == [5, 6]
    assert metrics["edge_distance_m"] == pytest.approx([5e-3 / 21, 6e-3 / 21])
    p[0, 0] = 1e-12
    assert edge_metrics(p, y)["edge_distance_cells"][0] == 0


def test_end_window_peak_includes_partially_overlapping_cells():
    y = (np.arange(21) - 10) * (1e-3 / 21)
    pressure = np.zeros((21, 3))
    pressure[2, 1] = 123
    pressure[-3, 1] = 456
    assert edge_metrics(pressure, y)["end_window_peak_pa"] == [123, 456]


def test_refinement_gate_rejects_growth_and_user_threshold_failures():
    case = {
        "edge_distance_cells": [5, 5],
        "outermost_row_load_n": [0.0, 0.0],
        "end_band_load_n": {"2": [1.0, 1.0], "5": [3.0, 3.0], "10": [6.0, 6.0]},
        "approach_m": 2e-6,
        "moment_x_n_m": 0.01,
        "centroid_y_m": 1e-4,
        "end_window_peak_pa": [100e6, 100e6],
    }
    assert screen_gate(case)
    assert refinement_gate(case, case)
    assert not refinement_gate(case, {**case, "end_window_peak_pa": [103e6, 100e6]})
    assert not screen_gate({**case, "edge_distance_cells": [4, 5]})
    assert not screen_gate({**case, "outermost_row_load_n": [0.5, 0.0]})
    assert not screen_gate({
        **case, "end_band_load_n": {**case["end_band_load_n"], "2": [2.0, 1.0]}
    })
    assert systematic_end_growth(
        {"end_window_peak_pa": [1.0, 0.0]},
        {"end_window_peak_pa": [2.0, 0.0]},
        {"end_window_peak_pa": [3.0, 0.0]},
    )
    assert not systematic_end_growth(
        {"end_window_peak_pa": [1.0, 0.0]},
        {"end_window_peak_pa": [2.0, 0.0]},
        {"end_window_peak_pa": [1.9, 0.0]},
    )


def test_report_marks_boundary_and_refinement_growth_unresolved(tmp_path):
    base = {
        "kind": "parabolic", "relief_nm": 2000, "zone_fraction": 0.1,
        "tilt_rad": 0.0, "nx": 49, "ny": 257, "load_n": 100.0,
        "audit_kkt": 1e-6, "approach_m": 2e-6, "centroid_y_m": 0.0,
        "moment_x_n_m": 0.0, "peak_pa": 100e6, "peak_x_m": 0.0, "peak_y_m": 0.0,
        "footprint_y_m": [-4e-4, 4e-4], "edge_distance_cells": [5, 5],
        "edge_distance_m": [2e-5, 2e-5], "outermost_row_load_n": [0.0, 0.0],
        "end_window_peak_pa": [10e6, 10e6],
        "end_band_load_n": {"2": [0.0, 0.0], "5": [0.0, 0.0], "10": [1.0, 1.0]},
        "cpu_fft_relative_max": 1e-15, "direct_sample_relative_max": 1e-15,
    }
    records = {
        "coarse": base,
        "fine": {**base, "ny": 513, "peak_pa": 101e6},
        "boundary": {**base, "kind": "cylindrical", "edge_distance_cells": [0, 0]},
    }
    write_table(tmp_path, records)
    with (tmp_path / "cases.csv").open(newline="", encoding="utf-8") as file:
        rows = {row["case"]: row for row in csv.DictReader(file)}
    assert rows["fine"]["peak_status"] == "unresolved_increase"
    assert rows["coarse"]["peak_status"] == "unresolved_increase"
    assert rows["boundary"]["peak_status"] == "unresolved_boundary_contact"
