"""Bounded experimental limited-capacity JFO/load Newton, CPU float64.

Case quantities are SI; state is interior P=p/pH,D=1-theta,h0/Href in
C order (ny,nx), +x mean entrainment. No projections or relaxed root gates.
A numerically accepted finite-grid root is not mesh/physical validation.
"""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from scipy.sparse import diags
from scipy.sparse.csgraph import maximum_bipartite_matching

from ehl_solver.cavitating_newton import CavitatingLinearError, solve_cavitating_correction
from ehl_solver.limited_audit import audit_limited_state
from ehl_solver.limited_system import LimitedLoadLinearization, LimitedLoadProbe
from ehl_solver.property_load_newton import _content_bound_blockers
from ehl_solver.prototype import hertz_guess


@dataclass(frozen=True)
class LimitedNewtonResult:
    case: object
    state: np.ndarray
    history: tuple
    audit: dict
    transport_fraction: float = 1.0


class LimitedNewtonError(RuntimeError):
    def __init__(self, message, state, history, audit=None):
        super().__init__(message)
        self.state = np.array(state, dtype=np.float64, copy=True)
        self.history = deepcopy(tuple(history))
        self.audit = deepcopy(audit)


class LimitedObserverError(RuntimeError):
    pass


def fraction_to_boundary(probe, state, direction):
    """Original P,D tolerances and strict full-source film feasibility."""
    n = probe.n
    margin = state[: 2 * n] + 1e-10
    change = direction[: 2 * n]
    bound = 1.0
    falling = change < -margin
    if np.any(falling):
        bound = min(bound, 0.99 * float(np.min(margin[falling] / -change[falling])))
    velocity = direction[n : 2 * n]
    gap = 1 + 1e-10 - state[n : 2 * n]
    rising = velocity > gap
    if np.any(rising):
        bound = min(bound, 0.99 * float(np.min(gap[rising] / velocity[rising])))
    _, _, film = probe.fields(state)
    dh, _, _ = probe.directions(state, direction)
    gap = film - probe.system.problem.film_floor_m
    closing = dh < -gap
    if np.any(closing):
        bound = min(bound, 0.99 * float(np.min(gap[closing] / -dh[closing])))
    return max(0.0, bound)


def _line_search(probe, state, direction, merit, slope):
    trials = []
    bound = fraction_to_boundary(probe, state, direction)
    if not np.isfinite(slope) or slope >= 0:
        return None, trials, bound
    for k in range(24):
        alpha = 1.0 if k == 0 else bound * 2.0 ** (-(k - 1))
        trial = state + alpha * direction
        value = None
        accepted = False
        reason = "pressure/content/film infeasible"
        if alpha > 0 and probe.feasible(trial):
            r = probe.residual(trial)
            value = float(r @ r / (2 * probe.size))
            accepted = bool(
                np.isfinite(value)
                and value <= merit + 1e-4 * alpha * slope
                and value < merit
                and not np.array_equal(state, trial)
            )
            reason = "feasible Armijo candidate" if accepted else "Armijo/no resolved progress"
        trials.append(
            dict(
                stage="full" if k == 0 else "affine_bound_backtrack",
                alpha=alpha,
                merit=value,
                accepted=False,
                reason=reason,
            )
        )
        if accepted:
            return trial, trials, bound
    return None, trials, bound


def _structural_content_pivots(model, full):
    """Recovery constraints for unmatched free-content columns; no state change."""
    matrix = model.frozen_matrix(model.state)
    n = model.n
    reduced = (
        matrix[:n, :n] @ diags(full.astype(float))
        + matrix[:n, n : 2 * n] @ diags((~full).astype(float))
    ).tocsr()
    reduced.eliminate_zeros()
    matching = maximum_bipartite_matching(reduced, perm_type="row")
    return np.flatnonzero((matching < 0) & (~full))


def solve_limited_load(
    case,
    initial_state=None,
    *,
    maximum_iterations=30,
    maximum_krylov_cycles=10,
    active_recovery=True,
    preconditioner="local",
    state_observer=None,
    transport_fraction=1.0,
    reference_recovery=False,
    local_balance_tolerance=1e-10,
):
    """Experimental bounded solve from an explicitly unaccepted ND guess.

    At most 30 corrections, 8 frozen models/correction, 10 GMRES cycles
    with restart 60 and 24 full/affine line-search trials. Original min/force
    residuals and the true one-sided derivative govern all nonlinear merit
    checks, even when a recovery model selects different complementary rows.
    """
    if (
        type(maximum_iterations) is not int
        or not 0 <= maximum_iterations <= 30
        or type(maximum_krylov_cycles) is not int
        or not 1 <= maximum_krylov_cycles <= 10
        or isinstance(local_balance_tolerance, bool)
        or not np.isfinite(local_balance_tolerance)
        or not 0 < local_balance_tolerance <= 1e-10
        or type(reference_recovery) is not bool
        or type(active_recovery) is not bool
        or preconditioner not in ("material", "local")
        or (state_observer is not None and not callable(state_observer))
    ):
        raise ValueError("Invalid bounded Newton/Krylov/recovery/observer options")
    probe = LimitedLoadProbe(case, transport_fraction=transport_fraction)
    state = (
        hertz_guess(case, probe.system)[0]
        if initial_state is None
        else np.array(initial_state, dtype=np.float64, copy=True)
    )
    if state.shape != (probe.size,) or not np.all(np.isfinite(state)):
        raise ValueError("Finite full P,D,h0/Href initial state required")
    history = []
    audit = None
    guide = np.zeros_like(state)
    try:
        for iteration in range(maximum_iterations + 1):
            if not probe.feasible(state):
                raise RuntimeError("Initial/current state violates original feasibility")
            audit = audit_limited_state(probe, state)
            audit["requested_local_balance_tolerance"] = local_balance_tolerance
            audit["requested_tolerance_satisfied"] = (
                audit["mass"]["relative_local_balance"] <= local_balance_tolerance
            )
            if state_observer is not None:
                snapshot = state.copy()
                snapshot.setflags(write=False)
                try:
                    state_observer(iteration, snapshot, deepcopy(audit))
                except Exception as exc:
                    raise LimitedObserverError(
                        "State observer failed; numerical solve not continued"
                    ) from exc
            if audit["numerically_accepted"] and audit["requested_tolerance_satisfied"]:
                final = audit_limited_state(probe, state, direct=True)
                final["requested_local_balance_tolerance"] = local_balance_tolerance
                final["requested_tolerance_satisfied"] = (
                    final["mass"]["relative_local_balance"] <= local_balance_tolerance
                )
                audit = final
                if not final["numerically_accepted"] or not final["requested_tolerance_satisfied"]:
                    raise RuntimeError("Independent direct elastic/final SI audit disagrees")
                snapshot = state.copy()
                snapshot.setflags(write=False)
                return LimitedNewtonResult(
                    case, snapshot, tuple(history), final, probe.transport_fraction
                )
            if iteration == maximum_iterations:
                raise RuntimeError("Limited Newton correction budget exhausted")
            residual = probe.residual(state)
            merit = float(residual @ residual / (2 * probe.size))
            record = dict(iteration=iteration, merit=merit, audit=audit, models=[])
            history.append(record)
            full = (state[: probe.n] >= state[probe.n : 2 * probe.n]).copy()
            seen = set()
            best = None
            mode = preconditioner
            reference = None
            reference_alpha = 0.0
            for recovery in range(8 if active_recovery else 1):
                model = LimitedLoadLinearization(
                    case,
                    state,
                    guide,
                    preconditioner=mode,
                    active_full=full,
                    transport_fraction=probe.transport_fraction,
                    reference_state=reference,
                )
                key = model.signature()
                if key in seen:
                    break
                seen.add(key)
                entry = dict(
                    recovery=recovery,
                    preconditioner=mode,
                    guide=guide.copy(),
                    active_full=full.copy(),
                    status="linear_pending",
                    reference_alpha=reference_alpha,
                    trials=[],
                )
                record["models"].append(entry)
                try:
                    linear = solve_cavitating_correction(
                        model,
                        state,
                        maximum_krylov_cycles=maximum_krylov_cycles,
                        gmres_restart=60,
                        schur_tolerance=1e-10,
                    )
                except CavitatingLinearError as exc:
                    entry.update(
                        status="linear_rejected", reason=str(exc), diagnostics=exc.diagnostics
                    )
                    pivots = (
                        _structural_content_pivots(model, full)
                        if active_recovery
                        else np.array([], dtype=int)
                    )
                    entry["structural_content_pivots"] = pivots.tolist()
                    if len(pivots):
                        full = full.copy()
                        full[pivots] = True
                    else:
                        mode = "material" if mode == "local" else "local"
                    continue
                direction = linear.direction
                model_residual = model.residual(state)
                measured = float(
                    np.linalg.norm(model.jacobian_vector(state, direction) + model_residual)
                    / max(np.linalg.norm(model_residual), 1e-300)
                )
                entry.update(
                    status="linear_solved",
                    direction=direction.copy(),
                    linear_true_residual=measured,
                    linear_true_residuals=linear.linear_true_residuals,
                    linear_iterations=linear.linear_iterations,
                    krylov_histories=linear.krylov_histories,
                    used_bordered_fallback=linear.used_bordered_fallback,
                    schur_indicator=linear.schur_indicator,
                    field_setup_failures=linear.field_setup_failures,
                )
                if not np.isfinite(measured) or measured > 5e-9:
                    entry.update(
                        status="linear_rejected",
                        reason="Independent reconstructed correction failed",
                    )
                    mode = "material" if mode == "local" else "local"
                    continue
                slope = float(residual @ probe.derivative(state, direction) / probe.size)
                entry["slope"] = slope
                trial, trials, bound = _line_search(probe, state, direction, merit, slope)
                entry.update(trials=trials, fraction_to_boundary=bound)
                predicted = state + direction
                proposed = full.copy()
                proposed[predicted[: probe.n] < -1e-10] = False
                proposed[predicted[probe.n : 2 * probe.n] < -1e-10] = True
                blockers = _content_bound_blockers(state, direction, probe.n)
                blockers = blockers[~full[blockers]]
                entry["content_bound_pivots"] = blockers.tolist()
                if len(blockers):
                    proposed = full.copy()
                    proposed[blockers] = True
                if trial is not None:
                    candidate = trials[-1]
                    if best is None or candidate["merit"] < best[2]["merit"]:
                        best = (trial.copy(), entry, candidate)
                    if candidate["alpha"] >= 0.1 or not active_recovery:
                        break
                guide = direction.copy()
                same_active = np.array_equal(full, proposed)
                full = proposed
                reference = None
                reference_alpha = 0.0
                if reference_recovery and same_active:
                    radii = (1e-3, 1e-4, 1e-2, 1e-5)
                    reference_alpha = min(
                        radii[(recovery // 2) % len(radii)],
                        0.5 * fraction_to_boundary(probe, state, guide),
                    )
                    candidate_reference = state + reference_alpha * guide
                    if reference_alpha > 0 and probe.feasible(candidate_reference):
                        reference = candidate_reference
                    else:
                        reference_alpha = 0.0
            if best is None:
                raise RuntimeError("All bounded limited/active recovery models exhausted")
            trial, entry, candidate = best
            candidate.update(
                accepted=True, reason="Accepted best feasible directional Armijo trial"
            )
            guide = trial - state
            state = trial
    except (RuntimeError, ValueError, FloatingPointError) as exc:
        if isinstance(exc, LimitedObserverError):
            raise
        raise LimitedNewtonError(str(exc), state, history, audit) from exc
    raise AssertionError("Unreachable bounded Newton exit")
