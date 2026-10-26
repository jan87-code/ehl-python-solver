import json

import numpy as np
import pytest

from examples import run_scientific_validation as validation
from examples.run_rigid_point_farfield import audit_volume_flux, solve_case
from examples.scientific_figures import caption, extrema


def test_rigid_dimensional_similarity_preserves_content_and_scales_pressure_force():
    base, fields = solve_case(2, 0.5)
    taller, higher = solve_case(2, 0.5, h0_m=4e-7)
    viscous, thicker_oil = solve_case(2, 0.5, viscosity_pa_s=0.08, speed_m_s=0.003)
    np.testing.assert_allclose(higher["x_m"], 2 * fields["x_m"])
    np.testing.assert_allclose(higher["y_m"], 2 * fields["y_m"])
    np.testing.assert_allclose(higher["pressure_pa"], fields["pressure_pa"] / 8, atol=1e-9)
    np.testing.assert_allclose(higher["theta"], fields["theta"], atol=1e-13)
    assert taller["load_n"] == pytest.approx(base["load_n"] / 2, rel=1e-12)
    np.testing.assert_allclose(thicker_oil["pressure_pa"], 6 * fields["pressure_pa"], atol=1e-8)
    assert viscous["load_n"] == pytest.approx(6 * base["load_n"], rel=1e-12)
    for case in (base, taller, viscous):
        assert case["audit"]["mass_maximum"] < 1e-8
        assert case["audit"]["maximum_complementarity"] < 1e-10


def test_independent_si_flux_audit_matches_uniform_transport_and_detects_inconsistent_content():
    x, y = np.linspace(0, 0.001, 7), np.linspace(-0.0005, 0.0005, 5)
    fields = {
        "x_m": x,
        "y_m": y,
        "pressure_pa": np.zeros((5, 7)),
        "film_m": np.full((5, 7), 2e-6),
        "theta": np.full((5, 7), 0.7),
    }
    options = dict(
        viscosity_pa_s=0.04,
        speed_m_s=0.1,
        length_m=0.001,
        film_scale_m=2e-6,
        pressure_scale_pa=12 * 0.04 * 0.1 * 0.001 / (2e-6) ** 2,
    )
    audit = audit_volume_flux(fields, **options)
    # Outer control-volume faces lie halfway to boundary nodes: 3 interior rows.
    flow = 0.1 * 2e-6 * 0.7 * 3 * (y[1] - y[0])
    np.testing.assert_allclose(audit["outward_volume_flow_m3_s"], [-flow, flow, 0, 0])
    assert audit["mass_maximum"] == 0
    assert audit["net_volume_flow_m3_s"] == 0
    fields["theta"][2, 3] = 0.6
    corrupted = audit_volume_flux(fields, **options)
    assert corrupted["mass_maximum"] > 0.1
    assert corrupted["maximum_relative_local_balance"] > 0.01


@pytest.mark.parametrize("extent,step", [(0, 0.5), (2, 0), (1, 0.4), (200, 0.25)])
def test_rigid_study_rejects_invalid_or_excessive_grids(extent, step):
    with pytest.raises(ValueError):
        solve_case(extent, step)


def test_plot_metadata_reports_actual_sampled_extrema_and_operating_units():
    state, fields = solve_case(2, 0.5)
    original = {key: value.copy() for key, value in fields.items()}
    params = validation.PARAMETERS | {"target_load_n": state["load_n"]}
    audit = state["audit"] | {"load_n": state["load_n"], "film_coupling_max_error_m": 0}
    text = caption(fields, params, audit)
    for label in (
        "Ue=",
        "eta=",
        "E*=",
        "Rx,Ry=",
        "(ny,nx)=",
        "hmin=",
        "pmax=",
        "mass max=",
        "force error=",
        "inlet left",
        "outlet right",
        " MPa",
        " Pa s",
    ):
        assert label in text
    values = extrema(fields)
    assert values["minimum_film"]["value_si"] == pytest.approx(1e-7)
    assert values["minimum_film"]["x_m"] == 0
    assert values["minimum_film"]["y_m"] == 0
    for key, value in fields.items():
        np.testing.assert_array_equal(value, original[key])


def test_scientific_runner_preserves_failure_and_refuses_existing_output(tmp_path, monkeypatch):
    def fail(_directory):
        raise RuntimeError("injected independent benchmark rejection")

    monkeypatch.setattr(validation, "hertz_benchmark", fail)
    output = tmp_path / "new_results"
    with pytest.raises(RuntimeError, match="benchmark rejection"):
        validation.main(output)
    report = json.loads((output / "summary.json").read_text())
    assert report["status"] == "failed"
    assert report["failure"]["reason"] == "injected independent benchmark rejection"
    saved = (output / "summary.json").read_bytes()
    with pytest.raises(FileExistsError):
        validation.main(output)
    assert (output / "summary.json").read_bytes() == saved
