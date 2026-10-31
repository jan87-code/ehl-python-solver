import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from ehl_solver.cavitating_newton import CavitatingLinearError
from ehl_solver.limited_audit import audit_limited_state, limited_mass_audit
from ehl_solver.limited_continuation import (
    LimitedPathError,
    continue_limited_load,
    continue_limited_transport,
)
from ehl_solver.limited_newton import (
    LimitedNewtonError,
    LimitedObserverError,
    _structural_content_pivots,
    solve_limited_load,
)
from ehl_solver.limited_system import LimitedLoadLinearization, LimitedLoadProbe
from ehl_solver.limited_transport import limited_fluxes
from ehl_solver.prototype import PointContactCase, hertz_guess
from ehl_solver.transport_candidate import donor_fluxes, face_divergence


@pytest.fixture(scope="module")
def donor_root():
    case = PointContactCase(
        **json.loads((Path(__file__).parents[1] / "cases/circular_50n.json").read_text())
    )
    root = solve_limited_load(case, transport_fraction=0.0)
    assert root.audit["numerically_accepted"] and root.transport_fraction == 0
    return root


def mathematical_state():
    case = PointContactCase(ny=9, nx=13)
    probe = LimitedLoadProbe(case)
    rng = np.random.default_rng(45)
    p = np.zeros((9, 13))
    theta = np.ones_like(p)
    p[1:-1, 1:-1] = 1e7 * (1 + 0.1 * rng.random((7, 11)))
    theta[1:-1, 1:-1] = 0.6 + 0.1 * rng.random((7, 11))
    z = probe.system.pack(SimpleNamespace(pressure_pa=p, fractional_film_content=theta), 1e-6)
    g = rng.normal(size=probe.size) * 1e-3
    return case, z, g


@pytest.mark.parametrize("fraction", [0.0, 0.4, 1.0])
def test_complete_blended_direction_and_sparse_preconditioner(fraction):
    case, z, g = mathematical_state()
    model = LimitedLoadLinearization(case, z, g, transport_fraction=fraction)
    analytic = model.jacobian_vector(z, g)
    step = 1e-4
    fd = (model.probe.residual(z + step * g) - model.probe.residual(z - step * g)) / (2 * step)
    assert np.linalg.norm(fd - analytic) / np.linalg.norm(analytic) < 1e-6
    p, theta, h = model.probe.fields(z)
    q_limited = limited_fluxes(
        model.problem.x_m, model.problem.y_m, h, p, theta, case.fluid(), case.entrainment_m_s
    )
    q_donor = donor_fluxes(
        model.problem.x_m, model.problem.y_m, h, p, theta, case.fluid(), case.entrainment_m_s
    )
    expected = face_divergence(
        model.problem.x_m,
        model.problem.y_m,
        q_donor[0] + fraction * (q_limited[0] - q_donor[0]),
        q_donor[1],
    )
    audit = limited_mass_audit(
        model.problem, case.fluid(), case.entrainment_m_s, h, p, theta, transport_fraction=fraction
    )
    factor = model.problem.length_scale_m / model.probe.system.base.flux_scale
    assert audit["mass_maximum"] == pytest.approx(abs(expected).max() * factor, rel=1e-13)
    # Matrix-free field approximation checked independently from nonlocal elasticity.
    from ehl_solver.limited_transport import freeze_limited_flux_jvp

    dh, dp, dt = model.probe.directions(z, g)
    limited = freeze_limited_flux_jvp(
        model.problem.x_m,
        model.problem.y_m,
        h,
        p,
        theta,
        dh,
        dp,
        dt,
        case.fluid(),
        case.entrainment_m_s,
    )
    offset = np.full_like(h, g[-1] * model.problem.film_scale_m)
    ql = limited(offset, dp, dt)
    from ehl_solver.transport_candidate import donor_flux_direction

    qd = donor_flux_direction(
        model.problem.x_m,
        model.problem.y_m,
        h,
        p,
        theta,
        offset,
        dp,
        dt,
        case.fluid(),
        case.entrainment_m_s,
    )
    expected = model.probe.mass((qd[0] + fraction * (ql[0] - qd[0]), qd[1]))
    assert (
        np.linalg.norm((model.frozen_matrix(z) @ g)[: model.n] - expected)
        / np.linalg.norm(expected)
        < 1e-12
    )


def test_zero_fraction_reproduces_reference_complete_operator(donor_root):
    root = donor_root
    case, z = root.case, root.state
    reference = case.system()
    full = z[: reference.n] >= z[reference.n : 2 * reference.n]
    model = LimitedLoadLinearization(
        case, z, np.zeros_like(z), active_full=full, transport_fraction=0.0
    )
    v = np.random.default_rng(23).normal(size=z.size) * 1e-4
    np.testing.assert_allclose(model.residual(z), reference.residual(z), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(
        model.jacobian_vector(z, v), reference.jacobian_vector(z, v), rtol=1e-12, atol=1e-12
    )


def test_recovery_rows_and_reference_point_keep_original_merit():
    case, z, g = mathematical_state()
    ref = z + 1e-3 * g
    n = (case.ny - 2) * (case.nx - 2)
    full = np.ones(n, dtype=bool)
    actual = LimitedLoadLinearization(case, z, g, active_full=full, reference_state=ref)
    at_reference = LimitedLoadLinearization(case, ref, g, active_full=full)
    np.testing.assert_array_equal(
        actual.jacobian_vector(z, g), at_reference.jacobian_vector(ref, g)
    )
    assert (actual.frozen_matrix(z) - at_reference.frozen_matrix(ref)).nnz == 0
    np.testing.assert_array_equal(actual.residual(z)[n : 2 * n], z[n : 2 * n])
    assert not np.array_equal(actual.residual(z), at_reference.residual(ref))
    r = actual.probe.residual(z)
    assert actual.merit_slope(z, r, g) == float(r @ actual.probe.derivative(z, g) / actual.size)
    with pytest.raises(ValueError):
        actual.reference_state[0] = 0
    with pytest.raises(ValueError):
        LimitedLoadLinearization(case, z, g, active_full=np.ones(n))


def test_nonlinear_root_and_independently_reconstructed_history(donor_root):
    root = donor_root
    case = root.case
    probe = LimitedLoadProbe(case, transport_fraction=0.0)
    z = hertz_guess(case, probe.system)[0]
    assert len(root.history) > 0
    stages = {t["stage"] for r in root.history for m in r["models"] for t in m["trials"]}
    assert stages == {"full", "affine_bound_backtrack"}
    for record in root.history:
        choices = [(m, t) for m in record["models"] for t in m["trials"] if t["accepted"]]
        assert len(choices) == 1
        model, trial = choices[0]
        linear = LimitedLoadLinearization(
            case,
            z,
            model["guide"],
            active_full=model["active_full"],
            preconditioner=model["preconditioner"],
            transport_fraction=0.0,
            reference_state=None
            if not model["reference_alpha"]
            else z + model["reference_alpha"] * model["guide"],
        )
        d = model["direction"]
        r = probe.residual(z)
        slope = float(r @ probe.derivative(z, d) / probe.size)
        assert slope == pytest.approx(model["slope"], rel=1e-13)
        assert (
            np.linalg.norm(linear.jacobian_vector(z, d) + linear.residual(z))
            / np.linalg.norm(linear.residual(z))
            <= 5e-9
        )
        assert slope < 0 and trial["merit"] <= record["merit"] + 1e-4 * trial["alpha"] * slope
        z = z + trial["alpha"] * d
        assert probe.feasible(z)
    np.testing.assert_array_equal(z, root.state)
    independent = audit_limited_state(probe, z, direct=True)
    assert independent["numerically_accepted"] and independent["direct_elasticity_checked"]
    assert not independent["full_limited_equations"]
    with pytest.raises(ValueError):
        root.state[0] = 0


def test_partial_transport_root_is_not_promoted_to_full_limited(donor_root):
    with pytest.raises(LimitedNewtonError) as exc:
        solve_limited_load(donor_root.case, donor_root.state, maximum_iterations=0)
    assert not exc.value.audit["numerically_accepted"]
    np.testing.assert_array_equal(exc.value.state, donor_root.state)
    result = solve_limited_load(
        donor_root.case, donor_root.state, transport_fraction=0.5, local_balance_tolerance=1e-12
    )
    assert result.audit["numerically_accepted"] and result.transport_fraction == 0.5
    assert not result.audit["full_limited_equations"]


@pytest.mark.parametrize(
    "options",
    [
        {"maximum_iterations": 31},
        {"maximum_iterations": True},
        {"maximum_krylov_cycles": 11},
        {"transport_fraction": 1.1},
        {"transport_fraction": True},
        {"local_balance_tolerance": 1e-9},
        {"local_balance_tolerance": 0.0},
        {"local_balance_tolerance": True},
    ],
)
def test_solver_budgets_and_fraction_are_enforced(options):
    with pytest.raises(ValueError):
        solve_limited_load(PointContactCase(), **options)


def test_observer_mutation_and_failure_are_separate_from_numerical_failure(donor_root):
    called = []

    def observer(i, state, audit):
        called.append(i)
        assert not state.flags.writeable
        audit["numerically_accepted"] = False

    r = solve_limited_load(
        donor_root.case, donor_root.state, transport_fraction=0.0, state_observer=observer
    )
    assert r.audit["numerically_accepted"] and called == [0]

    def bad(*args):
        raise RuntimeError("renderer failure")

    with pytest.raises(LimitedObserverError):
        solve_limited_load(
            donor_root.case, donor_root.state, transport_fraction=0.0, state_observer=bad
        )


def test_failed_krylov_models_are_retained(monkeypatch):
    import ehl_solver.limited_newton as module

    def fail(*args, **kwargs):
        raise CavitatingLinearError(
            "injected failure",
            {
                "linear_true_residuals": [0.1],
                "krylov_histories": [[0.2, 0.1]],
                "field_setup_failures": [],
            },
        )

    monkeypatch.setattr(module, "solve_cavitating_correction", fail)
    with pytest.raises(LimitedNewtonError) as exc:
        solve_limited_load(PointContactCase(ny=5, nx=7), transport_fraction=0.0)
    models = exc.value.history[0]["models"]
    assert 1 <= len(models) <= 8
    assert all(m["status"] == "linear_rejected" for m in models)
    assert all(m["diagnostics"]["krylov_histories"] == [[0.2, 0.1]] for m in models)


def test_structural_recovery_pivots_only_unmatched_free_content():
    full = np.array([True, False, False])
    matrix = np.zeros((7, 7))
    matrix[:3, :3] = np.eye(3)
    matrix[:3, 3:6] = np.diag([1.0, 0.0, 1.0])
    model = SimpleNamespace(n=3, state=np.zeros(7), frozen_matrix=lambda state: csr_matrix(matrix))
    np.testing.assert_array_equal(_structural_content_pivots(model, full), [1])


def test_load_retry_rolls_back_failed_state_and_reaudits_actual_targets(donor_root, monkeypatch):
    import ehl_solver.limited_continuation as module

    original = module.solve_limited_load
    seen = []

    def once(case, state, **kwargs):
        probe = LimitedLoadProbe(case, transport_fraction=kwargs["transport_fraction"])
        p, _, _ = probe.fields(state)
        seen.append((case.load_n, probe.system.problem.load_n(p)))
        if len(seen) == 1:
            raise LimitedNewtonError("injected load failure", state + 123, [], {})
        return original(case, state, **kwargs)

    monkeypatch.setattr(module, "solve_limited_load", once)
    path = continue_limited_load(donor_root, donor_root.case.load_n + 0.05)
    assert path.result.case.load_n == donor_root.case.load_n + 0.05
    assert path.result.audit["numerically_accepted"] and path.result.transport_fraction == 0
    assert path.attempts[0]["status"] == "rejected_stage"
    assert seen[1][1] == pytest.approx(donor_root.case.load_n, rel=1e-14)
    assert len(path.attempts) >= 3


def test_transport_retry_retains_last_stage_and_never_uses_rejected_partial(
    donor_root, monkeypatch
):
    import ehl_solver.limited_continuation as module

    original = module.solve_limited_load
    seen = []

    def reject_next(case, state, **kwargs):
        seen.append((kwargs["transport_fraction"], np.array(state, copy=True)))
        if kwargs["transport_fraction"] > 0:
            raise LimitedNewtonError("injected stage failure", state + 123, [], {})
        return original(case, state, **kwargs)

    monkeypatch.setattr(module, "solve_limited_load", reject_next)
    with pytest.raises(LimitedPathError) as exc:
        continue_limited_transport(donor_root.case, donor_root.state, maximum_attempts=3)
    assert exc.value.last_result.transport_fraction == 0
    assert seen[1][0] == 0.5 and seen[2][0] == 0.25
    np.testing.assert_array_equal(seen[1][1], seen[2][1])
    assert exc.value.attempts[-1]["status"] == "rejected_stage"


def test_continuation_option_and_seed_audit_failures(donor_root):
    with pytest.raises(ValueError):
        continue_limited_transport(donor_root.case, maximum_attempts=25)
    with pytest.raises(ValueError):
        continue_limited_load(donor_root, float("nan"))
    damaged = replace(donor_root, state=donor_root.state * 1.01)
    with pytest.raises(ValueError):
        continue_limited_load(damaged, 51.0)


def test_stage_observer_isolation_and_failure(donor_root):
    calls = []

    def observer(i, case, fraction, result):
        calls.append(i)
        assert not result.state.flags.writeable
        result.audit["numerically_accepted"] = False

    with pytest.raises(LimitedPathError) as exc:
        continue_limited_transport(
            donor_root.case, donor_root.state, maximum_attempts=1, stage_observer=observer
        )
    assert calls == [0] and exc.value.last_result.audit["numerically_accepted"]

    def bad(*args):
        raise RuntimeError("checkpoint failure")

    with pytest.raises(LimitedObserverError):
        continue_limited_transport(donor_root.case, donor_root.state, stage_observer=bad)
