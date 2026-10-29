"""Controlled N33 reference globalization experiments, not production JFO.

Keeps source-equation residuals, FD Jacobian, correction generators, projection,
20-correction budget and gates. Separates candidate competition from exact
one-sided projected-path Armijo. SI input/output fields, (ny,nx), +x entrainment;
P=p/pH and H0=h0/(a²/R), MATLAB F-order interiors internally.
"""

import argparse
import hashlib
import json
from functools import partial
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from examples.audit_matlab_checkpoint import audit
from examples.matlab_merit_direction import merit_direction
from examples.reproduce_matlab_newton import (
    ReferenceSystem,
    accepted,
    active_step,
    continuation,
    diagnostics,
    line_search,
    lm_step,
    merit,
    primary_step,
    save_state,
)
from examples.verify_matlab_reproduction import verify_state

POLICIES = ("compete_reference", "compete_armijo", "sequential_armijo")


def armijo_search(system, x, offset, dp, dh, old_merit, slope):
    if not np.isfinite(slope) or slope >= 0:
        return None, []
    lam, trials = 1.0, []
    for _ in range(int(system.cfg["max_backtrack"])):
        raw = x + lam * dp
        trial = np.maximum(raw, 0)
        record = {
            "lambda": lam,
            "clipped_nodes": int(np.count_nonzero(raw < 0)),
            "armijo_rhs": old_merit + system.cfg["armijo_c"] * lam * slope,
        }
        try:
            f, g, _ = system.evaluate(trial, offset + lam * dh)
            value = merit(f, g, system.cfg)
            record.update(
                valid=True, merit=value, merit_ratio=value / max(old_merit, np.finfo(float).eps)
            )
            trials.append(record)
            if value <= record["armijo_rhs"]:
                return (trial, offset + lam * dh), trials
        except ValueError:
            record["valid"] = False
            trials.append(record)
        lam *= 0.5
        if lam < system.cfg["min_lambda"]:
            break
    return None, trials


def select_candidate(system, x, offset, candidates, old_merit, policy):
    """Try original candidates; competition picks the lowest verified trial merit."""
    if policy not in POLICIES:
        raise ValueError("Unknown globalization policy")
    models, eligible = [], []
    for label, make in candidates:
        step = make()
        if step is None:
            models.append({"candidate": label, "linear_ok": False, "accepted": False})
            continue
        dp, dh, info = step
        slope, _, _ = merit_direction(system, x, offset, dp, dh)
        if policy == "compete_reference":
            trial, trials = line_search(system, x, offset, dp, dh, old_merit)
        else:
            trial, trials = armijo_search(system, x, offset, dp, dh, old_merit, slope)
        model = {
            **info,
            "candidate": label,
            "linear_ok": True,
            "true_slope": slope,
            "trials": trials,
            "accepted": False,
        }
        models.append(model)
        if trial is None:
            continue
        if (
            label == "active"
            and trials[-1]["merit_ratio"] >= system.cfg["as_max_accept_merit_ratio"]
        ):
            continue
        eligible.append((trials[-1]["merit"], len(models) - 1, trial))
        if policy == "sequential_armijo":
            break
    if not eligible:
        return None, models
    _, index, trial = min(eligible, key=lambda item: item[0])
    models[index]["accepted"] = True
    return trial, models


def solve_competing(system, x, offset, *, policy="compete_armijo", observer=None):
    if policy not in POLICIES:
        raise ValueError("Unknown globalization policy")
    x = np.array(x, dtype=np.float64, copy=True)
    history, reason = [], "iteration_budget"
    for iteration in range(int(system.cfg["max_newton"])):
        report = diagnostics(system, x, offset)
        if observer is not None:
            observer(iteration, x.copy(), offset, report)
        entry = {"iteration": iteration, **report, "models": []}
        history.append(entry)
        if accepted(system, report):
            reason = "converged"
            break
        f, g, fields = system.evaluate(x, offset)
        jp, jh, fd = system.jacobian(x, offset, fields)
        entry.update(fd)
        args = jp, jh, f, g, system.weights
        candidates = []
        if system.cfg["as_enable"] and report["fb_relative"] <= system.cfg["as_trigger_Rinf"]:
            candidates.append(
                (
                    "active",
                    partial(
                        active_step,
                        *args,
                        x,
                        fields["scaled_residual"].ravel(order="F"),
                        system.cfg,
                    ),
                )
            )
        candidates.append(("primary", partial(primary_step, *args, system.cfg)))
        if system.cfg["enable_bounded_lm"]:
            candidates.append(("lm", partial(lm_step, *args, x, system.cfg)))
        trial, models = select_candidate(
            system, x, offset, candidates, merit(f, g, system.cfg), policy
        )
        entry["models"] = models
        if trial is None:
            reason = "no_acceptable_step"
            break
        x, offset = trial
    report = diagnostics(system, x, offset)
    if observer is not None:
        observer(len(history), x.copy(), offset, report)
    success = accepted(system, report)
    return (
        x,
        offset,
        {
            "success": success,
            "reason": "converged" if success else reason,
            "iterations": len(history),
            **report,
            "history": history,
            "globalization_policy": policy,
        },
    )


def inspect_state(system, x, offset):
    """Evaluate all source candidates at a frozen stalled state, without updating it."""
    f, g, fields = system.evaluate(x, offset)
    jp, jh, _ = system.jacobian(x, offset, fields)
    args, old = (jp, jh, f, g, system.weights), merit(f, g, system.cfg)
    candidates = [
        ("primary", primary_step(*args, system.cfg)),
        ("active", active_step(*args, x, fields["scaled_residual"].ravel(order="F"), system.cfg)),
        ("lm", lm_step(*args, x, system.cfg)),
    ]
    records = []
    for label, step in candidates:
        if step is None:
            records.append({"candidate": label, "linear_ok": False})
            continue
        dp, dh, info = step
        slope, _df, _tangent = merit_direction(system, x, offset, dp, dh)
        source_trial, source_trials = line_search(system, x, offset, dp, dh, old)
        true_trial, true_trials = armijo_search(system, x, offset, dp, dh, old, slope)
        record = {
            "candidate": label,
            **info,
            "true_slope": slope,
            "old_merit": old,
            "true_slope_over_merit": slope / old,
            "source_trials": source_trials,
            "armijo_trials": true_trials,
            "source_accepted": source_trial is not None,
            "armijo_accepted": true_trial is not None,
            "directional_checks": [],
        }
        for alpha in (1e-3, 1e-4, 1e-5, 1e-6):
            try:
                ft, gt, _ = system.evaluate(np.maximum(x + alpha * dp, 0), offset + alpha * dh)
                difference = (merit(ft, gt, system.cfg) - old) / alpha
                record["directional_checks"].append(
                    {
                        "alpha": alpha,
                        "finite_difference_slope": difference,
                        "relative_slope_error": abs(difference - slope) / max(abs(slope), 1e-300),
                    }
                )
            except ValueError:
                record["directional_checks"].append({"alpha": alpha, "invalid": True})
        if true_trial is not None:
            record["trial_diagnostics"] = diagnostics(system, *true_trial)
        records.append(record)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--policy", choices=POLICIES, default="compete_armijo")
    parser.add_argument("--target-alpha", type=float)
    parser.add_argument("--stalled-results", type=Path)
    args = parser.parse_args()
    check = audit(args.checkpoint)
    if not check["passes_saved_matlab_gates"]:
        raise ValueError("Initial checkpoint fails original gates")
    args.output.mkdir(parents=True, exist_ok=False)
    data = loadmat(args.checkpoint, simplify_cells=True)
    system = ReferenceSystem(data["scale"], data["stage"], data["cfg"])
    x, offset = data["P"][1:-1, 1:-1].ravel(order="F"), data["H0"]
    summary = {
        "policy": args.policy,
        "checkpoint_audit": check,
        "production_jfo_modified": False,
        "matlab_runtime_execution": False,
    }
    if args.stalled_results:
        for index in (0, 1):
            item = json.loads((args.stalled_results / f"attempt_{index:03d}.json").read_text())
            with np.load(args.stalled_results / f"attempt_{index:03d}.npz") as saved:
                sx = saved["pressure_pa"][1:-1, 1:-1].ravel(order="F") / system.scale["pH"]
                sh = float(saved["offset_m"]) / system.scale["H_scale"]
            ss = ReferenceSystem(
                system.scale, {**system.stage, "alpha_scale": item["alpha"]}, system.cfg
            )
            records = inspect_state(ss, sx, sh)
            (args.output / f"stalled_{index:03d}.json").write_text(
                json.dumps(records, indent=2, allow_nan=False)
            )
    solver = partial(solve_competing, policy=args.policy)

    def observe(attempt, trial_system, tx, th, item):
        stem = args.output / f"attempt_{attempt:03d}"
        save_state(stem.with_suffix(".npz"), trial_system, tx, th)
        item["independent_si_audit"] = verify_state(
            stem.with_suffix(".npz"), alpha_fraction=trial_system.stage["alpha_scale"]
        )
        if item["success"] != item["independent_si_audit"]["passes_reference_gates"]:
            raise RuntimeError("Independent SI acceptance disagrees with solver")
        stem.with_suffix(".json").write_text(json.dumps(item, indent=2, allow_nan=False))
        print(
            "alpha",
            item["alpha"],
            "success",
            item["success"],
            "FB",
            item["fb_relative"],
            "iterations",
            item["iterations"],
            flush=True,
        )

    if args.target_alpha is not None:
        if (
            not np.isfinite(args.target_alpha)
            or not system.stage["alpha_scale"] < args.target_alpha <= 1
        ):
            raise ValueError("Target alpha must exceed checkpoint alpha and be <=1")
        trial_system = ReferenceSystem(
            system.scale, {**system.stage, "alpha_scale": args.target_alpha}, system.cfg
        )
        tx, th, item = solver(trial_system, x, offset)
        item["alpha"] = args.target_alpha
        observe(0, trial_system, tx, th, item)
        summary.update(
            success=item["success"],
            target_alpha=args.target_alpha,
            fb_relative=item["fb_relative"],
            iterations=item["iterations"],
        )
    else:
        _tx, _th, ramp = continuation(system, x, offset, observer=observe, trial_solver=solver)
        summary.update(
            last_verified_alpha=ramp["last_verified_alpha"],
            target_reached=ramp["target_reached"],
            stop=ramp["stop"],
        )
        (args.output / "continuation.json").write_text(json.dumps(ramp, indent=2, allow_nan=False))
    files = [
        Path(__file__),
        Path("examples/matlab_merit_direction.py"),
        Path("examples/reproduce_matlab_newton.py"),
        Path("examples/audit_matlab_checkpoint.py"),
        Path("examples/verify_matlab_reproduction.py"),
    ]
    snapshot = args.output / "source"
    snapshot.mkdir()
    summary["sources_sha256"] = {}
    for path in files:
        content = path.read_bytes()
        summary["sources_sha256"][str(path)] = hashlib.sha256(content).hexdigest()
        (snapshot / path.name).write_bytes(content)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
