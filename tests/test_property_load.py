from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import trapezoid

from ehl_solver.cavitating_coupled import CavitatingElasticityProblem
from ehl_solver.constitutive import IsothermalFluid
from ehl_solver.property_load import PropertyLoadError, solve_property_at_load
from ehl_solver.property_reynolds import admissible, mass_audit
from examples.property_monolithic_reference import small_monolithic_reference
from examples.run_farfield_budget import H0, LY, P_REF, RX, RY, L


@pytest.fixture(scope="module")
def known():
    x = np.linspace(-3, 4, 9) * L
    y = np.linspace(-3, 3, 7) * LY
    p = CavitatingElasticityProblem(x, y, H0, RX, RY, 1e11, L, H0, P_REF * 100, 1)
    fluid = IsothermalFluid(0.04, 850)
    info, (pressure, theta, h, w) = small_monolithic_reference(p, fluid, 0.1)
    audit = mass_audit(
        x,
        y,
        h,
        pressure,
        theta,
        fluid,
        0.1,
        length_scale_m=L,
        film_scale_m=H0,
        pressure_scale_pa=P_REF * 100,
    )
    assert admissible(audit, pressure, theta, P_REF * 100), info
    q = float(trapezoid(trapezoid(pressure, x, axis=1), y))
    return p, fluid, q, pressure, theta, h, w


@pytest.fixture(scope="module")
def tiny():
    from ehl_solver.property_coupled import solve_property_coupling

    p = CavitatingElasticityProblem(
        np.arange(-2, 1) * L, np.arange(-1, 2) * LY, H0, RX, RY, 1e11, L, H0, P_REF * 100, 1
    )
    fluid = IsothermalFluid(0.04, 850)
    s = solve_property_coupling(p, fluid, 0.1)
    # Just outside the exact sampled force but inside the declared tolerance.
    return p, fluid, s.load_n * (1 + 1e-10)


def test_known_offset_and_fields_recovered_from_independent_nonlinear_target(known):
    p, fluid, q, pressure, theta, h, _ = known
    template = replace(p, central_film_offset_m=1.1 * H0)
    records = []
    r = solve_property_at_load(
        template,
        fluid,
        0.1,
        q,
        h0_bracket_m=(0.8 * H0, 1.3 * H0),
        evaluation_callback=lambda e, s: records.append(e),
    )
    assert r.relative_load_error <= 1e-8
    assert r.central_film_offset_m == pytest.approx(H0, rel=1e-7)
    assert (
        np.linalg.norm(r.coupled_solution.pressure_pa - pressure) / np.linalg.norm(pressure) < 1e-7
    )
    np.testing.assert_allclose(r.coupled_solution.fractional_film_content, theta, atol=1e-7)
    np.testing.assert_allclose(r.coupled_solution.film_thickness_m, h, rtol=1e-8)
    assert template.central_film_offset_m == 1.1 * H0
    np.testing.assert_array_equal(template.x_m, p.x_m)
    assert len(records) == len(r.evaluations) == len({e.central_film_offset_m for e in records})
    assert all(e.audit["mass_maximum"] <= 1e-8 and e.maximum_film_error_m <= 1e-17 for e in records)
    assert r.root_iterations > 0


def test_endpoint_match_is_accepted_with_original_force_tolerance(tiny):
    p, fluid, q, *_ = tiny
    r = solve_property_at_load(p, fluid, 0.1, q, h0_bracket_m=(H0, 1.3 * H0))
    assert r.relative_load_error <= 1e-8
    assert r.central_film_offset_m == H0
    assert r.root_iterations == 0


def test_unbracketed_target_and_budget_failure_keep_valid_evaluations(tiny):
    p, fluid, q, *_ = tiny
    with pytest.raises(PropertyLoadError, match="does not bracket") as exc:
        solve_property_at_load(p, fluid, 0.1, 10 * q, h0_bracket_m=(0.8 * H0, 1.3 * H0))
    assert len(exc.value.evaluations) == 3 and exc.value.last_solution is not None
    with pytest.raises(PropertyLoadError, match="evaluation budget") as exc:
        solve_property_at_load(
            p, fluid, 0.1, q, h0_bracket_m=(0.8 * H0, 1.3 * H0), maximum_evaluations=3
        )
    assert len(exc.value.evaluations) == 3 and exc.value.last_solution is not None


def test_inner_exhaustion_is_never_used_as_a_force_value(tiny, monkeypatch):
    import ehl_solver.property_load as module

    p, fluid, q, *_ = tiny
    original = module.solve_property_coupling
    calls = []

    def fail_second(*args, **kw):
        calls.append(1)
        if len(calls) == 2:
            kw["maximum_iterations_per_stage"] = 1
        return original(*args, **kw)

    monkeypatch.setattr(module, "solve_property_coupling", fail_second)
    with pytest.raises(PropertyLoadError, match="exhausted") as exc:
        solve_property_at_load(p, fluid, 0.1, q, h0_bracket_m=(0.8 * H0, 1.3 * H0))
    assert len(exc.value.evaluations) == 1
    assert exc.value.last_solution.history[-1]["compliance_factor"] == 1
    assert exc.value.failed_solution.history[-1]["compliance_factor"] == 0
    assert exc.value.inner_failure.history


@pytest.mark.parametrize("corruption", ["film", "load"])
def test_cached_audit_and_force_cannot_hide_corrupted_returned_fields(
    tiny, monkeypatch, corruption
):
    import ehl_solver.property_load as module

    p, fluid, q, *_ = tiny
    original = module.solve_property_coupling

    def corrupt(*args, **kw):
        s = original(*args, **kw)
        return (
            replace(s, film_thickness_m=1.01 * s.film_thickness_m)
            if corruption == "film"
            else replace(s, load_n=1.01 * s.load_n)
        )

    monkeypatch.setattr(module, "solve_property_coupling", corrupt)
    with pytest.raises(PropertyLoadError, match="independent") as exc:
        solve_property_at_load(p, fluid, 0.1, q, h0_bracket_m=(0.8 * H0, 1.3 * H0))
    assert not exc.value.evaluations and exc.value.failed_solution is not None


@pytest.mark.parametrize(
    "changes",
    [
        {"target_load_n": 0},
        {"h0_bracket_m": (-H0, H0)},
        {"maximum_evaluations": 2},
        {"relative_load_tolerance": 1e-4},
    ],
)
def test_invalid_load_requests_reject_before_solving(tiny, changes):
    p, fluid, q, *_ = tiny
    kw = {"target_load_n": q, "h0_bracket_m": (0.8 * H0, 1.3 * H0)} | changes
    with pytest.raises(ValueError):
        solve_property_at_load(p, fluid, 0.1, **kw)


def test_offset_stopping_rule_cannot_override_force_tolerance(tiny):
    p, fluid, q = tiny
    with pytest.raises(PropertyLoadError, match="force tolerance") as exc:
        solve_property_at_load(
            p, fluid, 0.1, q, h0_bracket_m=(0.8 * H0, 1.3 * H0), offset_tolerance_m=H0
        )
    assert exc.value.evaluations and exc.value.last_solution is not None
