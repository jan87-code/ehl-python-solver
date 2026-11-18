import numpy as np
import pytest

from examples.analyze_ciulli_validation_uncertainty import (
    EXPECTED_GRIDS,
    _pair_metrics,
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
