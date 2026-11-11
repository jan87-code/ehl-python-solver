import json
from pathlib import Path

import numpy as np
import pytest

from ehl_solver.limited_audit import audit_limited_state
from ehl_solver.limited_newton import solve_limited_load
from ehl_solver.limited_system import LimitedLoadProbe
from ehl_solver.prototype import PointContactCase, hertz_guess
from examples.analyze_ciulli_donor_grid import (
    _midplane_comparison,
    profile_error_metrics,
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
