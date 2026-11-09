import csv
import json
from pathlib import Path

import pytest

from ehl_solver.prototype import PointContactCase

CASES_PATH = (
    Path(__file__).resolve().parents[1]
    / "cases"
    / "experimental_validation_candidates.json"
)


def load_cases():
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))["cases"]


def test_paper_benchmark_records_preserve_sources_and_measured_observations():
    cases = load_cases()
    assert {case["id"] for case in cases} == {
        "cameron_gohar_1966_oil_3_load_series",
        "gohar_1971_tungsten_carbide_glass",
        "ciulli_2008_steady_pure_rolling",
    }

    for case in cases:
        assert case["source"]["authors"]
        assert case["source"]["title"]
        assert case["experimental_observation"]["quantity"]
        assert case.get("blocking_gaps") or case.get("optional_reference_improvements")


def test_gohar_1971_reported_units_convert_to_si():
    case = next(case for case in load_cases() if case["id"].startswith("gohar_1971"))
    conditions = case["conditions"]

    assert conditions["load_n"] == pytest.approx(7.85 * 4.448221615)
    assert conditions["rolling_speed_m_s"] == pytest.approx(11.8 * 0.0254)
    assert conditions["viscosity_pa_s"] == pytest.approx(4.2e-6 * 6894.757293)
    assert conditions["pressure_viscosity_coefficient_per_pa"] == pytest.approx(
        2.52e-9 * 10
    )
    assert case["experimental_observation"]["reported_value"] == "second black fringe"
    assert case["experimental_observation"]["numerical_film_thickness_available"] is False


def test_ciulli_candidate_reproduces_reported_hertz_geometry_and_refines_consistently():
    case_data = next(case for case in load_cases() if case["id"].startswith("ciulli_2008"))
    solver_input = case_data["solver_input"]
    reference = PointContactCase(**solver_input)

    assert reference.hertz.radius_m == pytest.approx(
        case_data["conditions"]["reported_hertz_radius_m"], rel=1e-8
    )
    assert reference.hertz.peak_pressure_pa == pytest.approx(
        case_data["conditions"]["reported_peak_pressure_pa"], rel=0.03
    )
    assert case_data["experimental_observation"]["central_film_thickness_um_digitized_range"] == [
        0.45,
        0.47,
    ]

    for size in case_data["refinement_grid_sizes"]:
        refined = PointContactCase(**{**solver_input, "ny": size, "nx": size})
        assert refined.physical_signature() == reference.physical_signature()
        assert refined.domain_half_width_x_m == reference.domain_half_width_x_m
        assert refined.domain_half_width_y_m == reference.domain_half_width_y_m


def test_ciulli_figure13_profiles_are_digitized_for_both_s0_midplanes():
    case_data = next(case for case in load_cases() if case["id"].startswith("ciulli_2008"))
    csv_path = CASES_PATH.parent / case_data["experimental_observation"]["reference_csv"]
    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    assert len(rows) == 42
    for direction in ("transverse", "longitudinal"):
        profile = [row for row in rows if row["direction"] == direction]
        assert len(profile) == 21
        assert [int(row["coordinate_um"]) for row in profile] == list(range(-100, 101, 10))
        assert all(0.0 <= float(row["film_thickness_um"]) <= 0.7 for row in profile)
        assert all(float(row["digitization_uncertainty_um"]) > 0 for row in profile)

    transverse = [row for row in rows if row["direction"] == "transverse"]
    longitudinal = [row for row in rows if row["direction"] == "longitudinal"]
    assert min(float(row["film_thickness_um"]) for row in transverse) == pytest.approx(0.25)
    assert min(float(row["film_thickness_um"]) for row in longitudinal) == pytest.approx(0.36)
    assert (
        case_data["experimental_observation"][
            "pressure_dependent_refractive_index_correction_applied"
        ]
        is False
    )


def test_incomplete_historical_cases_are_not_presented_as_runnable_inputs():
    cases = load_cases()
    for case_id in (
        "cameron_gohar_1966_oil_3_load_series",
        "gohar_1971_tungsten_carbide_glass",
    ):
        case = next(case for case in cases if case["id"] == case_id)
        assert case["status"] == "source_data_only"
        assert "solver_input" not in case
