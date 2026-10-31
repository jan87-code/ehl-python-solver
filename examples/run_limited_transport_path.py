"""Archive a bounded transport path; only full fraction-one closure is success."""

import argparse
import json
import platform
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import scipy

from ehl_solver.limited_continuation import LimitedPathError, continue_limited_transport
from ehl_solver.limited_newton import LimitedNewtonError
from ehl_solver.limited_system import LimitedLoadProbe
from ehl_solver.prototype import PointContactCase
from ehl_solver.prototype_cli import clean_json, digest, load_run


def main(case_path, seed, output, maximum_attempts):
    if seed is not None:
        case, _, _ = load_run(seed)
        if case_path is not None and PointContactCase(**json.loads(case_path.read_text())) != case:
            raise ValueError("Seed and case must match exactly")
        with np.load(seed / "solution.npz", allow_pickle=False) as saved:
            initial = saved["state_dimensionless"].copy()
            probe = LimitedLoadProbe(case, transport_fraction=0.0)
            p, t, h = probe.fields(initial)
            for actual, key in [(p, "pressure_pa"), (t, "theta"), (h, "film_m")]:
                np.testing.assert_array_equal(actual, saved[key])
    else:
        case = (
            PointContactCase(**json.loads(case_path.read_text()))
            if case_path
            else PointContactCase()
        )
        initial = None
    output.mkdir(parents=True, exist_ok=False)
    report = dict(
        status="running",
        case=asdict(case),
        operator_transport="limited_capacity_target",
        target_transport_fraction=1.0,
        initial_state_accepted_for_target=False,
        seed=None if seed is None else dict(path=str(seed), sha256=digest(seed / "solution.npz")),
        maximum_attempts=maximum_attempts,
        maximum_corrections_per_attempt=30,
        maximum_models_per_correction=8,
        maximum_krylov_cycles=10,
        gmres_restart=60,
        runtime=dict(
            python=sys.version,
            numpy=np.__version__,
            scipy=scipy.__version__,
            platform=platform.platform(),
        ),
        stages=[],
        physical_validation_complete=False,
        source_sha256={},
    )
    package = Path(__import__("ehl_solver").__file__).parent
    for path in package.glob("*.py"):
        dest = output / "source/ehl_solver" / path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        report["source_sha256"][path.name] = digest(path)
    shutil.copyfile(__file__, output / "run_limited_transport_path.py")
    report["driver_sha256"] = digest(__file__)

    def write():
        (output / "summary.json").write_text(
            json.dumps(clean_json(report), indent=2, allow_nan=False)
        )

    def stage(index, stage_case, fraction, outcome):
        folder = output / f"attempt_{index:03d}"
        folder.mkdir(exist_ok=False)
        probe = LimitedLoadProbe(stage_case, transport_fraction=fraction)
        p, t, h = probe.fields(outcome.state)
        probe.system.sync(outcome.state)
        _, _, _, w, offset = probe.system.base.unpack(outcome.state[:-1])
        np.savez_compressed(
            folder / "state.npz",
            state_dimensionless=outcome.state,
            pressure_pa=p,
            theta=t,
            film_m=h,
            displacement_m=w,
            offset_m=offset,
            x_m=probe.system.problem.x_m,
            y_m=probe.system.problem.y_m,
        )
        info = dict(
            case=asdict(stage_case),
            transport_fraction=fraction,
            status="rejected_stage"
            if isinstance(outcome, LimitedNewtonError)
            else "accepted_stage",
            audit=outcome.audit,
            history=outcome.history,
            state_sha256=digest(folder / "state.npz"),
        )
        if isinstance(outcome, LimitedNewtonError):
            info["reason"] = str(outcome)
        (folder / "summary.json").write_text(
            json.dumps(clean_json(info), indent=2, allow_nan=False)
        )
        report["stages"].append({k: v for k, v in info.items() if k != "history"})
        write()

    write()
    try:
        path = continue_limited_transport(
            case, initial, maximum_attempts=maximum_attempts, stage_observer=stage
        )
        result = path.result
        report.update(
            status="accepted_numerical_root",
            transport_fraction=result.transport_fraction,
            audit=result.audit,
            attempts=path.attempts,
        )
    except LimitedPathError as exc:
        result = exc.last_result
        report.update(
            status="rejected_path",
            reason=str(exc),
            attempts=exc.attempts,
            last_accepted_fraction=None if result is None else result.transport_fraction,
            last_stage_audit=None if result is None else result.audit,
        )
    if result is not None:
        np.savez_compressed(output / "last_accepted_state.npz", state_dimensionless=result.state)
        report["last_accepted_state_sha256"] = digest(output / "last_accepted_state.npz")
    write()
    print(
        report["status"],
        report.get("transport_fraction", report.get("last_accepted_fraction")),
        flush=True,
    )
    return 0 if report["status"] == "accepted_numerical_root" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path)
    parser.add_argument("--seed", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-attempts", type=int, choices=range(1, 25), default=24)
    args = parser.parse_args()
    raise SystemExit(main(args.case, args.seed, args.output, args.max_attempts))
