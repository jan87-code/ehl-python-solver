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
        assert case["blocking_gaps"]


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
        0.42,
        0.5,
    ]

    for size in case_data["refinement_grid_sizes"]:
        refined = PointContactCase(**{**solver_input, "ny": size, "nx": size})
        assert refined.physical_signature() == reference.physical_signature()
        assert refined.domain_half_width_x_m == reference.domain_half_width_x_m
        assert refined.domain_half_width_y_m == reference.domain_half_width_y_m


def test_incomplete_historical_cases_are_not_presented_as_runnable_inputs():
    cases = load_cases()
    for case_id in (
        "cameron_gohar_1966_oil_3_load_series",
        "gohar_1971_tungsten_carbide_glass",
    ):
        case = next(case for case in cases if case["id"] == case_id)
        assert case["status"] == "source_data_only"
        assert "solver_input" not in case
