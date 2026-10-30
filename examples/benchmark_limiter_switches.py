"""Directional/merit probes only: no coupled candidate Newton solve.

The full probe residual uses the original mass scaling, min(P,1-theta),
relative force and full-source elastic coupling. All trial states remain
unaccepted EHL states even if a merit Armijo comparison passes.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ehl_solver.limited_system import LimitedLoadProbe as ResidualProbe
from ehl_solver.prototype import PointContactCase
from ehl_solver.prototype_cli import clean_json, digest, load_run


def smooth_complete_probe():
    """Weak unaccepted mathematical state: independent full residual FD check."""
    probe = ResidualProbe(PointContactCase(ny=9, nx=13))
    rng = np.random.default_rng(629)
    p = np.zeros(probe.system.problem.shape)
    theta = np.ones_like(p)
    p[1:-1, 1:-1] = 1e7 * (1 + 0.01 * rng.random((7, 11)))
    theta[1:-1, 1:-1] = 0.6 + 0.01 * rng.random((7, 11))
    state = probe.system.pack(SimpleNamespace(pressure_pa=p, fractional_film_content=theta), 1e-6)
    direction = rng.normal(size=probe.size) * 1e-3
    action = probe.freeze(state, direction)
    analytic = action(direction)
    errors = []
    for step in (1e-3, 1e-4, 1e-5):
        fd = (
            probe.residual(state + step * direction) - probe.residual(state - step * direction)
        ) / (2 * step)
        blocks = {
            "mass": slice(0, probe.n),
            "complementarity": slice(probe.n, 2 * probe.n),
            "load": slice(2 * probe.n, None),
        }
        errors.append(
            dict(
                step=step,
                relative_error=float(np.linalg.norm(fd - analytic) / np.linalg.norm(analytic)),
                block_relative_errors={
                    name: float(
                        np.linalg.norm((fd - analytic)[sl])
                        / max(np.linalg.norm(analytic[sl]), 1e-30)
                    )
                    for name, sl in blocks.items()
                },
            )
        )
    return dict(
        scope="weak unaccepted mathematical state; not physical load validation",
        errors=errors,
        full_residual_difference_check_passed=min(
            max(e["block_relative_errors"].values()) for e in errors
        )
        < 1e-6,
    )


def main(seed, output):
    case, _, _ = load_run(seed)
    if case.transport != "donor":
        raise ValueError("An accepted donor prototype seed is required")
    output.mkdir(parents=True, exist_ok=False)
    probe = ResidualProbe(case)
    with np.load(seed / "solution.npz", allow_pickle=False) as saved:
        state = saved["state_dimensionless"].copy()
    residual = probe.residual(state)
    merit = float(residual @ residual / (2 * probe.size))
    problem = probe.system.problem
    p, theta, _ = probe.fields(state)
    yy, xx = np.meshgrid(
        problem.y_m / case.hertz.radius_m, problem.x_m / case.hertz.radius_m, indexing="ij"
    )
    vp = 0.01 * p * np.sin(2 * xx + 0.4 * yy) / problem.pressure_scale_pa
    vd = -0.01 * theta * (1 - theta) * np.sin(xx + yy)
    n = probe.n
    directions = dict(
        pressure_elastic=np.r_[vp[1:-1, 1:-1].ravel(), np.zeros(n + 1)],
        content=np.r_[np.zeros(n), vd[1:-1, 1:-1].ravel(), 0.0],
        offset_1nm=np.r_[np.zeros(2 * n), 1e-9 / problem.film_scale_m],
    )
    rng = np.random.default_rng(349)
    a, b = rng.normal(size=probe.size), rng.normal(size=probe.size)
    rows = []
    for name, direction in directions.items():
        for sign in (1.0, -1.0):
            v = sign * direction
            derivative = probe.derivative(state, v)
            frozen = probe.freeze(state, v)
            agreement = float(
                np.linalg.norm(frozen(v) - derivative) / max(np.linalg.norm(derivative), 1e-30)
            )
            linearity = float(
                np.linalg.norm(frozen(a + b) - frozen(a) - frozen(b))
                / max(np.linalg.norm(frozen(a + b)), 1e-30)
            )
            slope = float(residual @ derivative / probe.size)
            legacy_slope = float(residual @ probe.derivative(state, v, legacy=True) / probe.size)
            finite = []
            for step in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
                r1, r2 = probe.residual(state + step * v), probe.residual(state + 2 * step * v)
                d = (4 * (r1 - residual) - (r2 - residual)) / (2 * step)
                m1, m2 = r1 @ r1 / (2 * probe.size), r2 @ r2 / (2 * probe.size)
                estimate = (4 * (m1 - merit) - (m2 - merit)) / (2 * step)
                finite.append(
                    dict(
                        step=step,
                        residual_relative_error=float(
                            np.linalg.norm(d - derivative) / max(np.linalg.norm(derivative), 1e-30)
                        ),
                        merit_slope=float(estimate),
                        merit_slope_relative_error=float(
                            abs(estimate - slope) / max(abs(slope), 1e-30)
                        ),
                    )
                )
            trials = []
            if slope < 0:
                for i in range(25):
                    alpha = 0.5**i
                    feasible = probe.feasible(state + alpha * v)
                    value = None
                    if feasible:
                        r = probe.residual(state + alpha * v)
                        value = float(r @ r / (2 * probe.size))
                    accepted = feasible and value <= merit + 1e-4 * alpha * slope and value < merit
                    trials.append(
                        dict(
                            alpha=alpha,
                            feasible=feasible,
                            merit=value,
                            armijo_comparison=bool(accepted),
                        )
                    )
                    if accepted:
                        break
            rows.append(
                dict(
                    name=name,
                    sign=sign,
                    slope=slope,
                    legacy_slope=legacy_slope,
                    guide_agreement=agreement,
                    frozen_linearity_error=linearity,
                    finite_differences=finite,
                    trials=trials,
                    no_trials_reason="nonnegative directional slope" if slope >= 0 else None,
                )
            )
    report = dict(
        scope=__doc__,
        seed=str(seed),
        source_state_sha256=digest(seed / "solution.npz"),
        shape_ny_nx=list(p.shape),
        case=vars(case),
        merit=merit,
        rows=rows,
        physical_validation_complete=False,
        candidate_is_converged=False,
        krylov_correction_or_preconditioner_tested=False,
        branch_policy="exact ties only; no tolerance snapping; finite steps may cross near ties",
        smooth_complete_residual_check=smooth_complete_probe(),
    )
    (output / "summary.json").write_text(json.dumps(clean_json(report), indent=2, allow_nan=False))
    np.savez_compressed(output / "base_residual.npz", state_dimensionless=state, residual=residual)
    print(json.dumps(clean_json(report), indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.seed, args.output)
