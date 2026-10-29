"""Replay bounded mesh Newton models: independent bounds and original merit checks.

Reports numerical restrictions, never a physical instability or root promotion.
Requires exact captured dimensionless states and selected-row metadata. All
physical fields/bounds are reported in SI on (ny,nx), +x entrainment.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid

from ehl_solver.benchmark_200n import Circular200N
from ehl_solver.cavitating_newton import CavitatingLinearError, solve_cavitating_correction
from ehl_solver.newton_diagnostics import affine_feasibility_limits
from ehl_solver.property_load_newton import PropertyLoadSystem
from ehl_solver.property_reynolds import admissible, mass_audit
from examples.run_load_directed_200n import finite_json
from examples.run_scientific_validation import provenance, save_json


def summarize_history(history):
    """Aggregate recorded checks; deferred/superseded trials are not physical rejections."""
    rows, reasons, failed_linear = [], Counter(), []
    for i, row in enumerate(history):
        if row["iteration"] != i:
            raise ValueError("contiguous original iteration history required")
        models = row["models"]
        accepted = [(j, t) for j, m in enumerate(models) for t in m["trials"] if t["accepted"]]
        if len(accepted) > 1 or (not accepted and i != len(history)-1):
            raise ValueError("one accepted trial per completed iteration required")
        for j, model in enumerate(models):
            value = model["linear_true_residual"]
            if not np.isfinite(value) or value > 5e-9:
                failed_linear.append([i, j])
            reasons.update(t["reason"] for t in model["trials"] if not t["accepted"])
        rows.append({"iteration": i, "mass_maximum": row["audit"]["mass_maximum"],
                     "merit": row["merit"], "models": len(models),
                     "linear_iterations": sum(m["linear_iterations"] for m in models),
                     "active_changes": [m["active_changes"] for m in models],
                     "accepted_model": accepted[0][0] if accepted else None,
                     "accepted_alpha": accepted[0][1]["alpha"] if accepted else None,
                     "accepted_merit": accepted[0][1]["merit"] if accepted else None,
                     "active_recovery_failures": row.get("active_recovery_failures", [])})
    alphas = [r["accepted_alpha"] for r in rows if r["accepted_alpha"] is not None]
    return {"iterations": rows, "models_total": sum(r["models"] for r in rows),
            "linear_iterations_total": sum(r["linear_iterations"] for r in rows),
            "recorded_final_linear_gate_failures": failed_linear,
            "nonaccepted_trial_reasons": dict(reasons),
            "minimum_accepted_alpha": min(alphas) if alphas else None}


def _raw_limit(margin, speed, offset):
    values = np.full(margin.shape, np.inf)
    with np.errstate(over="ignore", invalid="raise", divide="ignore"):
        np.divide(margin, speed, out=values, where=speed > 0)
    value = float(values.min())
    if not np.isfinite(value):
        return {"alpha_limit": None, "cells_iy_ix": []}
    return {"alpha_limit": value,
            "cells_iy_ix": (np.argwhere(values == value)[:8]+offset).tolist()}


def replay_model(system, state, model, *, next_state=None):
    """Replay one selected linear model; internal state is P,D,h0/Href, not SI.

    No active mask is inferred from rounded saved SI content. Original recorded
    override indices reconstruct the selected rows. SI diagnostic bounds and
    exact internal-variable bounds are both reported: theta=1-D can lose tiny
    deficits during SI reconstruction, so they need not be bitwise equal.
    """
    if "active_override_indices" not in model:
        raise ValueError("exact selected-row metadata required; do not infer from rounded fields")
    raw_indices = np.asarray(model["active_override_indices"])
    if raw_indices.ndim != 1 or (raw_indices.size and raw_indices.dtype.kind not in "iu"):
        raise ValueError("integer selected-row override indices required")
    indices = raw_indices.astype(int)
    if (len(np.unique(indices)) != len(indices) or np.any(indices < 0)
            or np.any(indices >= system.n) or len(indices) != model["active_changes"]):
        raise ValueError("invalid selected-row override indices")
    full = system.full(state).copy()
    full[indices] = ~full[indices]
    view = system if model["recovery"] == 0 else system.correction_view(state, full)
    if model["recovery"] == 0 and len(indices):
        raise ValueError("canonical first model cannot override active rows")
    correction = solve_cavitating_correction(view, state, maximum_krylov_cycles=10,
                                             gmres_restart=60)
    direction = correction.direction
    residual = system.residual(state)
    slope = system.merit_slope(state, residual, direction)
    slope_error = abs(slope-model["slope"])/max(abs(model["slope"]), 1e-300)
    if not np.isfinite(slope_error) or slope_error > 1e-6:
        raise RuntimeError("replay differs from recorded original-merit slope")
    system.sync(state)
    p, theta, h, _, _ = system.base.unpack(state[:-1])
    problem, n = system.problem, system.n
    dp, dt = np.zeros(problem.shape), np.zeros(problem.shape)
    dp[1:-1, 1:-1] = direction[:n].reshape(dp[1:-1, 1:-1].shape)*problem.pressure_scale_pa
    dt[1:-1, 1:-1] = -direction[n:2*n].reshape(dt[1:-1, 1:-1].shape)
    dh = system.film_direction(state, direction)
    si = affine_feasibility_limits(p*problem.pressure_scale_pa, theta, h, dp, dt, dh,
                                   pressure_scale_pa=problem.pressure_scale_pa,
                                   film_floor_m=problem.film_floor_m)
    shape = (problem.shape[0]-2, problem.shape[1]-2)
    nd = {
        "pressure_lower": _raw_limit((state[:n]+1e-10).reshape(shape),
                                      -direction[:n].reshape(shape), 1),
        "content_upper": _raw_limit((state[n:2*n]+1e-10).reshape(shape),
                                      -direction[n:2*n].reshape(shape), 1),
        "content_lower": _raw_limit((1+1e-10-state[n:2*n]).reshape(shape),
                                      direction[n:2*n].reshape(shape), 1),
        "film_lower": _raw_limit(h-problem.film_floor_m, -dh, 0),
    }
    finite = [r["alpha_limit"] for r in nd.values() if r["alpha_limit"] is not None]
    first = min(finite) if finite else None
    safe = .99*first if first is not None and first < 1 else 1.
    source_bound = model.get("fraction_to_boundary")
    if source_bound is not None and not np.isclose(safe, source_bound, rtol=1e-7, atol=1e-15):
        raise RuntimeError("replay differs from original fraction-to-boundary")
    result = {"recovery": model["recovery"], "active_changes": model["active_changes"],
              "complete_true_linear_residual": correction.true_linear_residual,
              "original_slope_relative_difference": slope_error,
              "used_bordered_fallback": correction.used_bordered_fallback,
              "linear_iterations": correction.linear_iterations,
              "si_affine_limits": si, "internal_limits": nd,
              "internal_first_alpha_limit": first,
              "first_constraints": [k for k, v in nd.items()
                                    if first is not None and v["alpha_limit"] == first],
              "recomputed_fraction_to_boundary": safe, "stored_fraction_to_boundary": source_bound}
    sampled = []
    sample_indices = {0} if model["trials"] else set()
    evaluable = [j for j, t in enumerate(model["trials"]) if t["merit"] is not None]
    if evaluable:
        sample_indices.add(evaluable[0])
    for j in sorted(sample_indices):
        recorded = model["trials"][j]
        alpha = recorded["alpha"]
        trial = state+alpha*direction
        value = None
        armijo, progress = False, False
        try:
            if (alpha <= 0 or trial[:2*n].min() < -1e-10
                    or trial[n:2*n].max() > 1+1e-10):
                raise ValueError("pressure/content infeasible")
            r = system.residual(trial)
            value = float(r@r/(2*system.size))
            armijo = value <= float(residual@residual/(2*system.size))+1e-4*alpha*slope
            progress = value < float(residual@residual/(2*system.size))
            reason = "feasible original merit evaluated"
        except (ValueError, FloatingPointError) as exc:
            reason = str(exc)
        sampled.append({"source_trial_index": j, "alpha": alpha,
                        "source_reason": recorded["reason"], "recomputed_reason": reason,
                        "source_merit": recorded["merit"], "recomputed_merit": value,
                        "original_armijo_pass": bool(armijo),
                        "resolved_merit_progress": bool(progress)})
    result["sampled_trials"] = sampled
    selected = [t for t in model["trials"] if t["accepted"]]
    if selected:
        trial = state+selected[0]["alpha"]*direction
        r = system.residual(trial)
        merit = float(r@r/(2*system.size))
        result.update(accepted_alpha=selected[0]["alpha"], recomputed_merit=merit,
                      stored_merit=selected[0]["merit"],
                      original_armijo_pass=merit <= float(residual@residual/(2*system.size))
                      + 1e-4*selected[0]["alpha"]*slope)
        if next_state is not None:
            if np.shape(next_state) != np.shape(state) or not np.all(np.isfinite(next_state)):
                raise ValueError("finite exact next snapshot of the same shape required")
            difference = float(np.linalg.norm(trial-next_state)
                               /max(np.linalg.norm(next_state), 1e-300))
            result["next_state_relative_difference"] = difference
            if not np.isfinite(difference) or difference > 1e-7:
                raise RuntimeError("replayed accepted state differs from exact next snapshot")
    return result


def main(source, directory, selected_iterations, *, all_models=False):
    summary_bytes = (source / "summary.json").read_bytes()
    history_bytes = (source / "history.json").read_bytes()
    report = json.loads(summary_bytes)
    if not report.get("terminal_audit"):
        raise ValueError("completed diagnostic trial required")
    history = json.loads(history_bytes)
    directory.mkdir(parents=True, exist_ok=False)
    case = Circular200N()
    if report["case"] != case.parameters():
        raise ValueError("registered circular case required")
    ny, nx = report["shape_ny_nx"]
    problem = case.problem(.5e-6, ny=ny, nx=nx,
                           maximum_physical_nodes=report["maximum_physical_nodes"])
    system = PropertyLoadSystem(problem, case.fluid(), 2., 200., preconditioner="local",
                                elasticity_mode="cached",
                                maximum_physical_nodes=report["maximum_physical_nodes"],
                                field_factorization=report["field_factorization"])
    result = {"status": "incomplete", "runtime": provenance(),
              "source_summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
              "source_history_sha256": hashlib.sha256(history_bytes).hexdigest(),
              "history_summary": summarize_history(history), "replays": []}
    with np.load(source / "terminal_state.npz") as a:
        np.testing.assert_array_equal(a["x_m"], problem.x_m)
        np.testing.assert_array_equal(a["y_m"], problem.y_m)
        p, t, h, w = [a[k].copy() for k in ("pressure_pa", "theta", "film_m", "displacement_m")]
    exact_h, exact_w = problem.fields(p, central_film_offset_m=report["terminal_offset_m"])
    np.testing.assert_allclose(h, exact_h, rtol=0, atol=1e-10*problem.film_scale_m)
    np.testing.assert_allclose(w, exact_w, rtol=0, atol=1e-10*problem.film_scale_m)
    audit = mass_audit(problem.x_m, problem.y_m, h, p, t, case.fluid(), 2.,
                       length_scale_m=problem.length_scale_m, film_scale_m=problem.film_scale_m,
                       pressure_scale_pa=problem.pressure_scale_pa)
    force = float(trapezoid(trapezoid(p, problem.x_m, axis=1), problem.y_m))
    accepted = admissible(audit, p, t, problem.pressure_scale_pa) and abs(force/200-1) <= 1e-8
    if accepted != report["status"].startswith("original gates passed"):
        raise RuntimeError("independent terminal gates disagree with recorded classification")
    result.update(terminal_audit=audit, terminal_force_n=force,
                  terminal_original_gates_pass=bool(accepted),
                  film_reference_error_m=float(abs(h-exact_h).max()))
    snapshots = {s["iteration"]: s for s in report["snapshots"]}

    def read_snapshot(index):
        row = snapshots[index]
        path = source / row["state_file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("snapshot modified")
        with np.load(path) as a:
            state = a["state_dimensionless"].copy()
            np.testing.assert_array_equal(a["x_m"], problem.x_m)
            np.testing.assert_array_equal(a["y_m"], problem.y_m)
            system.sync(state)
            P, theta, film, displacement, offset = system.base.unpack(state[:-1])
            np.testing.assert_array_equal(a["pressure_pa"], P*problem.pressure_scale_pa)
            np.testing.assert_array_equal(a["theta"], theta)
            np.testing.assert_allclose(a["film_m"], film, rtol=0, atol=1e-10*problem.film_scale_m)
            np.testing.assert_allclose(a["displacement_m"], displacement,
                                       rtol=0, atol=1e-10*problem.film_scale_m)
            if float(a["offset_m"]) != offset:
                raise ValueError("snapshot offset differs from exact internal state")
            return state

    for index in selected_iterations:
        if not 0 <= index < len(history):
            raise ValueError("replay iteration outside recorded history")
        state = read_snapshot(index)
        row = history[index]
        models = (row["models"] if all_models else
                  [m for m in row["models"] if any(t["accepted"] for t in m["trials"])])
        if not models:
            models = row["models"][-1:]
        for model in models:
            try:
                next_state = read_snapshot(index+1) if index+1 in snapshots else None
                replay = replay_model(system, state, model, next_state=next_state)
            except CavitatingLinearError as exc:
                replay = {"linear_replay_failed": True, "reason": str(exc),
                          "diagnostics": exc.diagnostics}
            result["replays"].append({"iteration": index, **replay})
        if not models:
            result["replays"].append({"iteration": index, "no_replayable_model": True,
                                       "source_diagnostics": report.get("diagnostics", {})})
        save_json(directory / "summary.json", finite_json(result))
    if (hashlib.sha256((source / "summary.json").read_bytes()).hexdigest()
            != result["source_summary_sha256"]
            or hashlib.sha256((source / "history.json").read_bytes()).hexdigest()
            != result["source_history_sha256"]):
        raise RuntimeError("diagnostic source changed during replay")
    replay_failures = any(r.get("linear_replay_failed") or r.get("no_replayable_model")
                          or r.get("original_armijo_pass") is False for r in result["replays"])
    result["status"] = ("terminal audited; some model replays unverified" if replay_failures else
                        "terminal gates and selected raw-state replays audited; no root promotion")
    save_json(directory / "summary.json", finite_json(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--iterations", nargs="+", type=int, required=True)
    parser.add_argument("--all-models", action="store_true")
    args = parser.parse_args()
    main(args.input_directory, args.output_directory, args.iterations, all_models=args.all_models)
