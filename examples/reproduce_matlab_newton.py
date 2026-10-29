"""N33 pressure-only MATLAB Newton/continuation reproduction, CPU float64.

Diagnostic only: sampled elastic kernel, projected pressure trials and the
source's residual-relative decrease test are deliberate MATLAB compatibility
choices. This is not the production JFO solver or physical validation.
Internal pressure/offset are dimensionless P=p/pH, H0=h0/(a²/R), MATLAB F order.
CLI input/output physical fields are SI, (ny,nx), entrainment +x.
"""

import argparse
import hashlib
import json
import time
from functools import partial
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.linalg import LinAlgWarning, get_lapack_funcs, lu_factor, lu_solve

from examples.audit_matlab_checkpoint import audit, reference_fields

QSTAR = 2 * np.pi / 3


class ReferenceSystem:
    def __init__(self, scale, stage, cfg):
        self.scale, self.stage, self.cfg = dict(scale), dict(stage), dict(cfg)
        self.weights = np.full(31 * 31, 0.25**2)

    def evaluate(self, x, offset, *, signed_probe=False):
        p = np.zeros((33, 33), dtype=np.float64)
        p[1:-1, 1:-1] = x.reshape((31, 31), order="F")
        fields = reference_fields(p, offset, self.scale, self.stage, _signed_probe=signed_probe)
        f = fields["fb"].ravel(order="F")
        g = float(self.weights @ x - QSTAR)
        if not np.all(np.isfinite(f)):
            raise ValueError("Nonfinite reference residual")
        return f, g, fields

    def jacobian(self, x, offset, fields):
        """Source FD of scaled smooth R, then its regularized FB chain rule."""
        cfg = self.cfg
        jr = np.empty((x.size, x.size))
        rs = fields["scaled_residual"].ravel(order="F")
        one_sided = 0
        for j in range(x.size):
            delta = min(
                cfg["jac_fd_max"], max(cfg["jac_fd_min"], cfg["jac_fd_rel"] * max(1, abs(x[j])))
            )
            probes = []
            for sign in (1, -1):
                trial = x.copy()
                trial[j] += sign * delta
                try:
                    probes.append(
                        self.evaluate(trial, offset, signed_probe=True)[2]["scaled_residual"].ravel(
                            order="F"
                        )
                    )
                except ValueError:
                    probes.append(None)
            plus, minus = probes
            if plus is not None and minus is not None:
                jr[:, j] = (plus - minus) / (2 * delta)
            elif plus is not None:
                jr[:, j] = (plus - rs) / delta
                one_sided += 1
            elif minus is not None:
                jr[:, j] = (rs - minus) / delta
                one_sided += 1
            else:
                raise ValueError(f"Both FD probes invalid for column {j}")
        delta = max(cfg["jac_fd_min"], cfg["jac_fd_rel"] * max(1, abs(offset)))
        plus = self.evaluate(x, offset + delta, signed_probe=True)[2]["scaled_residual"]
        minus = self.evaluate(x, offset - delta, signed_probe=True)[2]["scaled_residual"]
        jrh = ((plus - minus) / (2 * delta)).ravel(order="F")
        s = np.sqrt(x * x + rs * rs + cfg["jac_fb_floor"] ** 2)
        a, b = x / s - 1, rs / s - 1
        jp = b[:, None] * jr
        jp[np.diag_indices(x.size)] += a
        return jp, b * jrh, {"one_sided_columns": one_sided}


def _factor(matrix):
    # LAPACK 1-norm reciprocal condition estimate, like MATLAB rcond;
    # do not form an inverse or silently accept singular factorizations.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", LinAlgWarning)
        try:
            factor = lu_factor(matrix)
        except (LinAlgWarning, ValueError):
            return None, 0.0
    rcond, info = get_lapack_funcs("gecon", (matrix,))(
        factor[0], np.linalg.norm(matrix, 1), norm="1"
    )
    return factor, float(rcond) if info == 0 else 0.0


def _linear_checks(jp, jh, f, g, w, dp, dh, load_scale):
    return (
        float(
            np.linalg.norm(jp @ dp + jh * dh + f, np.inf) / max(1e-12, np.linalg.norm(f, np.inf))
        ),
        float(abs(w @ dp + g) / max(load_scale, abs(g))),
    )


def schur_step(jp, jh, f, g, w, cfg):
    for reg in np.atleast_1d(cfg["jac_regularization"]):
        matrix = jp + reg * max(1, np.linalg.norm(jp, np.inf)) * np.eye(f.size)
        factor, rc = _factor(matrix)
        if rc < 1e-14 or not np.isfinite(rc):
            continue
        uv = lu_solve(factor, np.column_stack((-f, jh)))
        u, v = uv[:, 0], uv[:, 1]
        denom = w @ v
        floor = cfg["schur_den_floor"] * max(1, np.linalg.norm(w) * np.linalg.norm(v))
        if not np.isfinite(denom) or abs(denom) < floor:
            continue
        dh = float((g + w @ u) / denom)
        dp = u - v * dh
        pr, br = _linear_checks(jp, jh, f, g, w, dp, dh, cfg["load_scale"])
        if np.all(np.isfinite(dp)) and np.isfinite(dh) and pr < 1e-2 and br < 1e-8:
            return (
                dp,
                dh,
                {
                    "method": "schur",
                    "regularization": float(reg),
                    "rcond": rc,
                    "pressure_linear_relative": pr,
                    "load_linear_relative": br,
                },
            )
    return None


def direct_step(jp, jh, f, g, w, cfg):
    matrix = np.block([[jp, jh[:, None]], [w[None, :], np.zeros((1, 1))]])
    rows = np.maximum(np.linalg.norm(matrix, axis=1), 1e-12)
    row_scaled = matrix / rows[:, None]
    cols = np.maximum(np.linalg.norm(row_scaled, axis=0), 1e-12)
    factor, rc = _factor(row_scaled / cols)
    if rc < 1e-14 or not np.isfinite(rc):
        return None
    step = lu_solve(factor, -np.append(f, g) / rows) / cols
    dp, dh = step[:-1], float(step[-1])
    pr, br = _linear_checks(jp, jh, f, g, w, dp, dh, cfg["load_scale"])
    if np.all(np.isfinite(step)) and pr < 1e-2 and br < 1e-8:
        return (
            dp,
            dh,
            {
                "method": "direct",
                "rcond": rc,
                "pressure_linear_relative": pr,
                "load_linear_relative": br,
            },
        )
    return None


def active_step(jp, jh, f, g, w, x, rs, cfg):
    active = (x <= cfg["as_pressure_tol"]) & (rs >= 0)
    free = ~active
    if active.sum() < cfg["as_min_active_nodes"] or free.sum() < 2:
        return None
    dp = np.where(active, -x, 0)
    ff = f[free] + jp[np.ix_(free, active)] @ dp[active]
    gf = float(g + w[active] @ dp[active])
    args = jp[np.ix_(free, free)], jh[free], ff, gf, w[free], cfg
    reduced = schur_step(*args)
    if reduced is None and cfg["enable_bordered_direct"]:
        reduced = direct_step(*args)
    if reduced is None:
        return None
    dp[free], dh, info = reduced
    pr = float(
        np.linalg.norm(jp[free, :] @ dp + jh[free] * dh + f[free], np.inf)
        / max(np.linalg.norm(f, np.inf), 1e-12)
    )
    br = float(abs(w @ dp + g) / max(cfg["load_scale"], abs(g)))
    if max(pr, br) >= 1e-4:
        return None
    return (
        dp,
        dh,
        {
            **info,
            "method": "active_" + info["method"],
            "active_count": int(active.sum()),
            "pressure_linear_relative": pr,
            "load_linear_relative": br,
        },
    )


def lm_step(jp, jh, f, g, w, x, cfg):
    n = x.size
    sw = np.sqrt(cfg["merit_load_weight"])
    matrix = np.block(
        [[jp / np.sqrt(n), jh[:, None] / np.sqrt(n)], [sw * w[None, :] / QSTAR, np.zeros((1, 1))]]
    )
    r = np.append(f / np.sqrt(n), sw * g / QSTAR)
    cols = np.linalg.norm(matrix, axis=0)
    cols = np.maximum(cols, 1e-10 * max(1, np.max(cols)))
    scaled = matrix / cols
    best, best_ratio = None, np.inf
    for mu in np.atleast_1d(cfg["lm_mu_candidates"]):
        blocked = np.zeros(n, dtype=bool)
        for _ in range(int(cfg["lm_max_active_sweeps"])):
            free = np.append(~blocked, True)
            b = scaled[:, free]
            factor, _rc = _factor(b.T @ b + mu * np.eye(free.sum()))
            if factor is None:
                break
            d = np.zeros(n + 1)
            d[free] = lu_solve(factor, -b.T @ r) / cols[free]
            new = (x <= cfg["lm_zero_tol"]) & (d[:-1] < 0)
            if not np.any(new & ~blocked):
                break
            blocked |= new
        if factor is None:
            continue
        # Source compatibility only: projected tangent and explicit trust caps.
        d[:-1] = np.maximum(d[:-1], -np.maximum(x, 0))
        d *= min(
            1,
            cfg["lm_trust_dp_inf"] / max(np.linalg.norm(d[:-1], np.inf), np.finfo(float).eps),
            cfg["lm_trust_dH0"] / max(abs(d[-1]), np.finfo(float).eps),
        )
        pred, slope = r + matrix @ d, float(r @ (matrix @ d))
        ratio = float(pred @ pred / max(r @ r, np.finfo(float).eps))
        if (
            np.all(np.isfinite(d))
            and slope < -1e-13 * max(r @ r, np.finfo(float).eps)
            and ratio < best_ratio
        ):
            best_ratio = ratio
            best = (
                d[:-1],
                float(d[-1]),
                {
                    "method": "bounded_lm",
                    "mu": float(mu),
                    "blocked": int(blocked.sum()),
                    "predicted_merit_ratio": ratio,
                    "slope": slope,
                },
            )
    return best


def merit(f, g, cfg):
    return float(0.5 * (np.mean(f * f) + cfg["merit_load_weight"] * (g / QSTAR) ** 2))


def line_search(system, x, offset, dp, dh, old_merit):
    """Reproduce projected source decrease test; not directional Armijo."""
    cfg, lam = system.cfg, 1.0
    trials = []
    for _ in range(int(cfg["max_backtrack"])):
        raw = x + lam * dp
        trial = np.maximum(raw, 0)
        record = {"lambda": lam, "clipped_nodes": int(np.count_nonzero(raw < 0))}
        try:
            f, g, _fields = system.evaluate(trial, offset + lam * dh)
            value = merit(f, g, cfg)
            record.update(
                valid=True, merit=value, merit_ratio=value / max(old_merit, np.finfo(float).eps)
            )
            trials.append(record)
            if value < old_merit * (1 - cfg["armijo_c"] * lam):
                return (trial, offset + lam * dh), trials
        except ValueError:
            record["valid"] = False
            trials.append(record)
        lam *= 0.5
        if lam < cfg["min_lambda"]:
            break
    return None, trials


def diagnostics(system, x, offset):
    f, g, fields = system.evaluate(x, offset)
    return {
        "fb_relative": float(np.linalg.norm(f, np.inf) / max(1, np.max(x))),
        "load_relative": float(g / QSTAR),
        "film_min_m": float(np.min(fields["film"]) * system.scale["H_scale"]),
        "offset_m": float(offset * system.scale["H_scale"]),
        "pressure_max_pa": float(np.max(x) * system.scale["pH"]),
    }


def accepted(system, report):
    return (
        report["fb_relative"] <= system.cfg["tol_fb_inf"]
        and abs(report["load_relative"]) <= system.cfg["tol_load_rel"]
    )


def primary_step(jp, jh, f, g, w, cfg):
    step = schur_step(jp, jh, f, g, w, cfg)
    if step is None and cfg["enable_bordered_direct"]:
        step = direct_step(jp, jh, f, g, w, cfg)
    return step


def solve(system, x, offset):
    x = np.array(x, dtype=np.float64, copy=True)
    history, reason = [], "iteration_budget"
    for iteration in range(int(system.cfg["max_newton"])):
        report = diagnostics(system, x, offset)
        entry = {"iteration": iteration, **report, "models": []}
        history.append(entry)
        if accepted(system, report):
            reason = "converged"
            break
        f, g, fields = system.evaluate(x, offset)
        try:
            jp, jh, fd = system.jacobian(x, offset, fields)
        except ValueError as exc:
            reason = "jacobian_failure: " + str(exc)
            break
        entry.update(fd)
        old_merit = merit(f, g, system.cfg)
        args = jp, jh, f, g, system.weights
        candidates = []
        if system.cfg["as_enable"] and report["fb_relative"] <= system.cfg["as_trigger_Rinf"]:
            candidates.append(
                partial(
                    active_step, *args, x, fields["scaled_residual"].ravel(order="F"), system.cfg
                )
            )

        # Direct is tried only if the Schur linear solve fails, as in MATLAB.
        candidates.append(partial(primary_step, *args, system.cfg))
        if system.cfg["enable_bounded_lm"]:
            candidates.append(partial(lm_step, *args, x, system.cfg))
        updated = False
        for candidate in candidates:
            step = candidate()
            if step is None:
                entry["models"].append({"linear_ok": False})
                continue
            dp, dh, info = step
            trial, trials = line_search(system, x, offset, dp, dh, old_merit)
            model = {**info, "linear_ok": True, "trials": trials, "accepted": False}
            entry["models"].append(model)
            if trial is None:
                continue
            if (
                info["method"].startswith("active_")
                and trials[-1]["merit_ratio"] >= system.cfg["as_max_accept_merit_ratio"]
            ):
                continue
            x, offset = trial
            updated = model["accepted"] = True
            break
        if not updated:
            reason = "no_acceptable_step"
            break
    report = diagnostics(system, x, offset)
    return (
        x,
        offset,
        {
            "success": accepted(system, report),
            "reason": reason,
            "iterations": len(history),
            **report,
            "history": history,
        },
    )


def continuation(system, x, offset, *, observer=None):
    cfg = system.cfg
    good, delta = system.stage["alpha_scale"], cfg["alpha_step_initial"]
    attempts, stop = [], "attempt_budget"
    if not accepted(system, diagnostics(system, x, offset)):
        raise ValueError("Continuation requires a verified initial checkpoint")
    targets = np.atleast_1d(cfg["alpha_targets"])
    for attempt in range(int(cfg["alpha_max_attempts"])):
        if good >= targets[-1] - 1e-12:
            stop = "target_reached"
            break
        target = targets[np.flatnonzero(targets > good + 1e-12)[0]]
        increment = min(delta, target - good)
        alpha = float(good + increment)
        trial_system = ReferenceSystem(system.scale, {**system.stage, "alpha_scale": alpha}, cfg)
        trial_x, trial_offset, report = solve(trial_system, x, offset)
        item = {
            "attempt": attempt,
            "from_alpha": float(good),
            "alpha": alpha,
            "increment": float(increment),
            **report,
        }
        attempts.append(item)
        if report["success"]:
            x, offset, good = trial_x, trial_offset, alpha
            if observer is not None:
                observer(attempt, trial_system, x, offset, item)
            if report["iterations"] <= 6:
                delta = min(cfg["alpha_step_max"], delta * cfg["alpha_step_grow"])
            elif report["iterations"] >= 12:
                delta = max(cfg["alpha_step_min"], 0.7 * delta)
        else:
            if observer is not None:
                observer(attempt, trial_system, trial_x, trial_offset, item)
            next_delta = min(delta, increment) * cfg["alpha_step_shrink"]
            if next_delta < cfg["alpha_step_min"]:
                stop = "minimum_alpha_step"
                break
            delta = next_delta
    return (
        x,
        offset,
        {
            "last_verified_alpha": float(good),
            "target_reached": bool(good >= targets[-1] - 1e-12),
            "stop": stop,
            "attempts": attempts,
        },
    )


def save_state(path, system, x, offset):
    _f, _g, fields = system.evaluate(x, offset)
    p = np.zeros((33, 33))
    p[1:-1, 1:-1] = x.reshape((31, 31), order="F") * system.scale["pH"]
    np.savez_compressed(
        path,
        pressure_pa=p,
        film_m=fields["film"] * system.scale["H_scale"],
        displacement_m=fields["deformation"] * system.scale["H_scale"],
        offset_m=offset * system.scale["H_scale"],
        x_m=np.linspace(-4, 4, 33) * system.scale["a"],
        y_m=np.linspace(-4, 4, 33) * system.scale["a"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    verified = audit(args.checkpoint)
    if not verified["passes_saved_matlab_gates"]:
        raise ValueError("Checkpoint failed independently recomputed reference gates")
    data = loadmat(args.checkpoint, simplify_cells=True)
    system = ReferenceSystem(data["scale"], data["stage"], data["cfg"])
    x, offset = data["P"][1:-1, 1:-1].ravel(order="F"), data["H0"]
    start = time.perf_counter()
    # Same-physics recovery test must do real nonlinear work, not accept the seed.
    repaired_x, repaired_h, recovery = solve(system, x, offset + 0.001)
    save_state(args.output / "recovery.npz", system, repaired_x, repaired_h)
    (args.output / "recovery.json").write_text(json.dumps(recovery, indent=2, allow_nan=False))
    print("recovery", recovery["success"], recovery["fb_relative"], flush=True)

    def observe(attempt, trial_system, trial_x, trial_h, item):
        stem = args.output / f"attempt_{attempt:03d}"
        save_state(stem.with_suffix(".npz"), trial_system, trial_x, trial_h)
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

    _x, _h, ramp = continuation(system, x, offset, observer=observe)
    summary = {
        "checkpoint_audit": verified,
        "recovery_success": recovery["success"],
        "last_verified_alpha": ramp["last_verified_alpha"],
        "target_reached": ramp["target_reached"],
        "stop": ramp["stop"],
        "wall_seconds": time.perf_counter() - start,
        "matlab_runtime_execution": False,
        "python_jfo_solution": False,
        "starting_alpha": data["stage"]["alpha_scale"],
        "settings": {
            k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in data["cfg"].items()
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (args.output / "continuation.json").write_text(json.dumps(ramp, indent=2, allow_nan=False))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
