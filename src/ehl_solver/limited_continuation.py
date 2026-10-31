"""Bounded experimental transport/load paths; rejected states never seed retries.

All accepted stages retain their equation's explicit transport fraction.
Only fraction==1 solves the full limited flux. No partial path is promoted.
"""

import hashlib
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from types import SimpleNamespace

import numpy as np

from ehl_solver.limited_audit import audit_limited_state
from ehl_solver.limited_newton import LimitedNewtonError, LimitedObserverError, solve_limited_load
from ehl_solver.limited_system import LimitedLoadProbe


@dataclass(frozen=True)
class LimitedPath:
    result: object
    attempts: tuple
    kind: str


class LimitedPathError(RuntimeError):
    def __init__(self, message, last_result, attempts, last_failure=None):
        super().__init__(message)
        self.last_result = last_result
        self.attempts = tuple(attempts)
        self.last_failure = last_failure


def _compact(history):
    """Retain scalar/trial/Krylov diagnostics; hash replay vectors to bound RAM.

    The stage observer receives full objects before compaction for disk
    checkpointing. Latest result/failure still retain their full histories.
    """
    records = []
    for record in history:
        entry = {k: v for k, v in record.items() if k != "models"}
        models = []
        for model in record["models"]:
            item = {k: v for k, v in model.items() if k not in ("guide", "direction")}
            for key in ("guide", "direction"):
                if key in model:
                    item[key + "_sha256"] = hashlib.sha256(
                        np.asarray(model[key]).tobytes()
                    ).hexdigest()
            item["vectors_in_compact_history"] = False
            models.append(item)
        entry["models"] = models
        records.append(entry)
    return records


def _notify(observer, index, case, fraction, outcome):
    if observer is not None:
        try:
            if isinstance(outcome, LimitedNewtonError):
                snapshot = LimitedNewtonError(
                    str(outcome), outcome.state, outcome.history, outcome.audit
                )
            else:
                snapshot = replace(
                    outcome,
                    state=np.array(outcome.state, copy=True),
                    history=deepcopy(outcome.history),
                    audit=deepcopy(outcome.audit),
                )
            snapshot.state.setflags(write=False)
            observer(index, case, fraction, snapshot)
        except Exception as exc:
            raise LimitedObserverError("Stage observer failed; continuation stopped") from exc


def _options(maximum_attempts, observer):
    if (
        type(maximum_attempts) is not int
        or not 1 <= maximum_attempts <= 24
        or (observer is not None and not callable(observer))
    ):
        raise ValueError("At most 24 stage attempts; stage observer must be callable")


def _attempt(case, state, fraction, index, observer, *, audit_only=False):
    try:
        # Intermediate transport stages need headroom: otherwise a stage
        # barely inside its own gate can remain outside the exact MC gate.
        tolerance = 1e-12 if 0 < fraction < 1 else 1e-10
        outcome = solve_limited_load(
            case,
            state,
            transport_fraction=fraction,
            local_balance_tolerance=tolerance,
            maximum_iterations=0 if audit_only else 30,
        )
        status = "accepted_stage"
    except LimitedNewtonError as exc:
        outcome = exc
        status = "rejected_stage"
    _notify(observer, index, case, fraction, outcome)
    data = dict(
        status=status,
        case=asdict(case),
        transport_fraction=fraction,
        initial_state_sha256=None if state is None else hashlib.sha256(state.tobytes()).hexdigest(),
        audit=outcome.audit,
        history=_compact(outcome.history),
        audit_only=audit_only,
    )
    if status == "rejected_stage":
        data["reason"] = str(outcome)
    return outcome, data


def continue_limited_transport(
    case, initial_state=None, *, maximum_attempts=24, stage_observer=None
):
    """Conservative donor->MC homotopy, plus bounded bisection/rollback.

    Final success always independently checks the original fraction-one
    equation. Intermediate accepted stages do not establish MC convergence.
    Full stage vectors may be archived by the observer; attempts keep compact
    diagnostics to prevent path-history memory scaling as attempts*all vectors.
    """
    _options(maximum_attempts, stage_observer)
    pending = [
        0.0,
        0.5,
        0.9,
        0.99,
        0.999,
        0.9999,
        0.99999,
        0.999999,
        0.9999999,
        0.99999999,
        0.999999999,
    ]
    attempts = []
    last = None
    failure = None
    state = None if initial_state is None else np.array(initial_state, dtype=np.float64, copy=True)
    while len(attempts) < maximum_attempts:
        if not pending:
            if last is None:
                break
            next_fraction = 0.5 * (1.0 + last.transport_fraction)
            if next_fraction <= last.transport_fraction or next_fraction >= 1.0:
                break
            pending = [next_fraction]
        target = pending.pop(0)
        outcome, data = _attempt(
            case, state, target, len(attempts), stage_observer, audit_only=target == 1.0
        )
        attempts.append(data)
        if isinstance(outcome, LimitedNewtonError):
            failure = outcome
            if last is None:
                break
            mid = 0.5 * (last.transport_fraction + target)
            if mid - last.transport_fraction < 1e-12:
                break
            pending[:0] = [mid, target]
            state = last.state.copy()  # NEVER continue from the failed partial.
            continue
        last = outcome
        state = last.state.copy()
        if target == 1.0:
            return LimitedPath(last, tuple(attempts), "transport")
        # A nearby blended root may already satisfy the exact MC gates;
        # explicitly run the fraction-one final audit instead of promoting it.
        probe = LimitedLoadProbe(case)
        if audit_limited_state(probe, state)["numerically_accepted"]:
            pending = [1.0]
    raise LimitedPathError(
        "Transport continuation incomplete; full limited root not established",
        last,
        attempts,
        failure,
    )


def continue_limited_load(
    initial_result, target_load_n, *, maximum_attempts=24, stage_observer=None
):
    """SI force path at a fixed, explicit transport fraction and physical grid.

    Warm guesses preserve pressure/content and signed SI offset; load/scales
    change without force rescaling. Retry only from the last audited stage.
    Blended paths remain blended and cannot certify the full limited model.
    """
    _options(maximum_attempts, stage_observer)
    if isinstance(target_load_n, bool) or not np.isfinite(target_load_n) or target_load_n <= 0:
        raise ValueError("Positive finite SI target force required")
    case = initial_result.case
    fraction = initial_result.transport_fraction
    probe = LimitedLoadProbe(case, transport_fraction=fraction)
    if not audit_limited_state(probe, initial_result.state, direct=True)["numerically_accepted"]:
        raise ValueError("Initial load-path result fails independent stage audit")
    last = initial_result
    attempts = []
    failure = None
    step = min(0.25 * case.load_n, abs(target_load_n - case.load_n))
    while last.case.load_n != target_load_n and len(attempts) < maximum_attempts:
        old = last.case
        sign = np.sign(target_load_n - old.load_n)
        magnitude = min(step, abs(target_load_n - old.load_n))
        load = (
            target_load_n
            if magnitude == abs(target_load_n - old.load_n)
            else old.load_n + sign * magnitude
        )
        try:
            new = replace(old, load_n=float(load))
        except ValueError as exc:
            attempts.append(
                dict(status="rejected_case", target_load_n=float(load), reason=str(exc))
            )
            failure = exc
            step *= 0.5
            if step < 1e-8 * old.load_n:
                break
            continue
        old_probe = LimitedLoadProbe(old, transport_fraction=fraction)
        p, theta, _ = old_probe.fields(last.state)
        offset = last.state[-1] * old_probe.system.problem.film_scale_m
        new_probe = LimitedLoadProbe(new, transport_fraction=fraction)
        if not (
            np.array_equal(old_probe.system.problem.x_m, new_probe.system.problem.x_m)
            and np.array_equal(old_probe.system.problem.y_m, new_probe.system.problem.y_m)
        ):
            raise ValueError("Load continuation must preserve the physical grid")
        state = new_probe.system.pack(
            SimpleNamespace(pressure_pa=p, fractional_film_content=theta), offset
        )
        outcome, data = _attempt(new, state, fraction, len(attempts), stage_observer)
        data.update(seed_force_n=old_probe.system.problem.load_n(p), initial_offset_m=float(offset))
        attempts.append(data)
        if isinstance(outcome, LimitedNewtonError):
            failure = outcome
            step *= 0.5
            if step < 1e-8 * old.load_n:
                break
        else:
            last = outcome
            step *= 2
    if last.case.load_n == target_load_n:
        return LimitedPath(last, tuple(attempts), "load")
    raise LimitedPathError(
        "Load continuation incomplete; last accepted target retained", last, attempts, failure
    )
