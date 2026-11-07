import json
from pathlib import Path

import numpy as np
import pytest

from ehl_solver.prototype import PointContactCase


@pytest.mark.parametrize(
    ("name", "size"),
    [
        ("nasa_d8317_table1_k1_probe.json", 33),
        ("nasa_d8317_table1_k1_probe_65.json", 65),
        ("nasa_d8317_table1_k1_probe_129.json", 129),
    ],
)
def test_nasa_probe_preserves_declared_groups_and_ambient_fluid_assumption(name, size):
    path = Path(__file__).resolve().parents[1] / "cases" / name
    case = PointContactCase(**json.loads(path.read_text()))
    e_prime = 2 * case.contact_modulus_pa
    assert case.viscosity_pa_s * case.entrainment_m_s / (e_prime * case.radius_m) == pytest.approx(
        0.1683e-11
    )
    assert case.load_n / (e_prime * case.radius_m**2) == pytest.approx(0.1106e-6)
    assert case.alpha_per_pa * e_prime == pytest.approx(4522)
    assert case.domain_half_width_x_m == pytest.approx(4 * case.hertz.radius_m)
    assert case.domain_half_width_y_m == pytest.approx(4 * case.hertz.radius_m)
    assert case.problem().shape == (size, size)

    _, eta, _, deta = case.fluid().properties(0.0)
    assert eta == pytest.approx(case.viscosity_pa_s)
    assert deta / eta == pytest.approx(case.alpha_per_pa)
    assert np.isfinite(case.fluid().viscosity_index)


def test_nasa_probe_refinement_has_same_physics_and_domain():
    cases = Path(__file__).resolve().parents[1] / "cases"
    coarse, middle, fine = (
        PointContactCase(**json.loads((cases / name).read_text()))
        for name in (
            "nasa_d8317_table1_k1_probe.json",
            "nasa_d8317_table1_k1_probe_65.json",
            "nasa_d8317_table1_k1_probe_129.json",
        )
    )
    for lower, upper in ((coarse, middle), (middle, fine)):
        assert lower.physical_signature() == upper.physical_signature()
        assert lower.domain_half_width_x_m == upper.domain_half_width_x_m
        assert lower.domain_half_width_y_m == upper.domain_half_width_y_m
        assert upper.nx - 1 == 2 * (lower.nx - 1)
        assert upper.ny - 1 == 2 * (lower.ny - 1)
