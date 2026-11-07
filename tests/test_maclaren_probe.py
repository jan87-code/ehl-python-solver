import json
import math
from pathlib import Path

import pytest

from ehl_solver.prototype import PointContactCase

CASES = Path(__file__).resolve().parents[1] / "cases"
CASE_NAMES = (
    "maclaren_pao4_20c_1p2_probe.json",
    "maclaren_pao4_20c_1p2_probe_65.json",
    "maclaren_pao4_20c_1p2_probe_129.json",
)


@pytest.mark.parametrize(("name", "size"), list(zip(CASE_NAMES, (33, 65, 129), strict=True)))
def test_pao4_paper_inputs_and_explicit_fluid_assumptions(name, size):
    case = PointContactCase(**json.loads((CASES / name).read_text()))
    contact_modulus = 1 / ((1 - 0.3**2) / 210e9 + (1 - 0.22**2) / 75e9)
    viscosity = 0.06576e-3 * math.exp(843.3 / (293.15 - 158.1))
    density = 829.3 - 0.6275 * 20
    assert case.contact_modulus_pa == pytest.approx(contact_modulus)
    assert case.viscosity_pa_s == pytest.approx(viscosity)
    assert case.density_kg_m3 == pytest.approx(density)
    assert case.alpha_per_pa == pytest.approx(17.6e-9)
    assert case.load_n == 50
    assert case.radius_m == pytest.approx(9.525e-3)
    assert case.entrainment_m_s == pytest.approx(1.2)
    assert case.problem().shape == (size, size)
    assert case.domain_half_width_x_m == pytest.approx(4 * case.hertz.radius_m)
    assert case.domain_half_width_y_m == pytest.approx(4 * case.hertz.radius_m)
    _, eta, _, deta = case.fluid().properties(0)
    assert eta == pytest.approx(viscosity)
    assert deta / eta == pytest.approx(case.alpha_per_pa)


def test_pao4_probe_refinements_preserve_physics_and_domain():
    cases = [PointContactCase(**json.loads((CASES / name).read_text())) for name in CASE_NAMES]
    for lower, upper in zip(cases, cases[1:], strict=False):
        assert lower.physical_signature() == upper.physical_signature()
        assert lower.domain_half_width_x_m == upper.domain_half_width_x_m
        assert lower.domain_half_width_y_m == upper.domain_half_width_y_m
        assert upper.nx - 1 == 2 * (lower.nx - 1)
        assert upper.ny - 1 == 2 * (lower.ny - 1)


def test_pao4_wide_domain_preserves_physics_and_spacing():
    inner = PointContactCase(**json.loads((CASES / CASE_NAMES[1]).read_text()))
    wide = PointContactCase(
        **json.loads((CASES / "maclaren_pao4_20c_1p2_probe_wide_97.json").read_text())
    )
    assert wide.physical_signature() == inner.physical_signature()
    assert wide.problem().shape == (97, 97)
    assert wide.domain_half_width_x_m == pytest.approx(6 * wide.hertz.radius_m)
    assert wide.domain_half_width_y_m == pytest.approx(6 * wide.hertz.radius_m)
    assert wide.problem().dx_m == pytest.approx(inner.problem().dx_m)
    assert wide.problem().dy_m == pytest.approx(inner.problem().dy_m)
