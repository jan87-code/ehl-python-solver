"""Experimental limited nonlinear run; numerical closure is not physical accuracy."""

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ehl_solver.limited_audit import audit_limited_state
from ehl_solver.limited_newton import LimitedNewtonError, solve_limited_load
from ehl_solver.limited_system import LimitedLoadProbe
from ehl_solver.prototype import PointContactCase
from ehl_solver.prototype_cli import clean_json, digest, load_run


def main(case_path, seed, output, maximum_iterations):
    case = (
        PointContactCase(**json.loads(case_path.read_text())) if case_path else PointContactCase()
    )
    probe = LimitedLoadProbe(case)
    initial = None
    initialization = dict(method="Hertz unaccepted guess", accepted=False)
    if seed is not None:
        old, _, _ = load_run(seed)
        if old != case:
            raise ValueError("Seed must have identical physical parameters, grid and domain")
        with np.load(seed / "solution.npz", allow_pickle=False) as saved:
            initial = saved["state_dimensionless"].copy()
            p, t, h = probe.fields(initial)
            for actual, name in [(p, "pressure_pa"), (t, "theta"), (h, "film_m")]:
                np.testing.assert_array_equal(actual, saved[name])
        initialization = dict(
            method="re-audited reference root; unaccepted for limited equations",
            source=str(seed),
            source_sha256=digest(seed / "solution.npz"),
            accepted=False,
        )
    output.mkdir(parents=True, exist_ok=False)
    package = Path(__import__("ehl_solver").__file__).parent
    hashes = {}
    for path in package.glob("*.py"):
        dest = output / "source/ehl_solver" / path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        hashes[path.name] = digest(path)
    shutil.copyfile(__file__, output / "run_limited_case.py")
    report = dict(
        status="running",
        case=asdict(case),
        operator_transport="limited_capacity",
        case_transport_used_for_reference_kinematics_only=True,
        initialization=initialization,
        maximum_corrections=maximum_iterations,
        maximum_models_per_correction=8,
        maximum_krylov_cycles=10,
        gmres_restart=60,
        source_sha256=hashes,
        driver_sha256=digest(__file__),
        iterations=[],
        physical_validation_complete=False,
    )

    def write():
        (output / "summary.json").write_text(
            json.dumps(clean_json(report), indent=2, allow_nan=False)
        )

    def save(path, state):
        probe.system.sync(state)
        p, t, h, w, offset = probe.system.base.unpack(state[:-1])
        np.savez_compressed(
            path,
            state_dimensionless=state,
            x_m=probe.system.problem.x_m,
            y_m=probe.system.problem.y_m,
            pressure_pa=p * probe.system.problem.pressure_scale_pa,
            theta=t,
            film_m=h,
            displacement_m=w,
            offset_m=offset,
        )

    def observe(i, state, audit):
        save(output / f"state_{i:03d}.npz", state)
        report["iterations"].append(dict(iteration=i, audit=audit))
        write()

    write()
    try:
        result = solve_limited_load(
            case, initial, maximum_iterations=maximum_iterations, state_observer=observe
        )
        state = result.state
        report.update(status="accepted_numerical_root", audit=result.audit, history=result.history)
    except LimitedNewtonError as exc:
        state = exc.state
        report.update(status="rejected", reason=str(exc), audit=exc.audit, history=exc.history)
    save(output / "solution.npz", state)
    report["state_sha256"] = digest(output / "solution.npz")
    write()
    if report["status"] == "accepted_numerical_root":
        independent = audit_limited_state(probe, state, direct=True)
        assert independent["numerically_accepted"]
        report["separate_final_audit"] = independent
        write()
    print(
        json.dumps(
            clean_json(
                {
                    k: v
                    for k, v in report.items()
                    if k not in ("source_sha256", "history", "iterations")
                }
            ),
            indent=2,
            allow_nan=False,
        )
    )
    return 0 if report["status"] == "accepted_numerical_root" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path)
    parser.add_argument("--seed", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-iterations", type=int, default=30, choices=range(31))
    args = parser.parse_args()
    raise SystemExit(main(args.case, args.seed, args.output, args.max_iterations))
