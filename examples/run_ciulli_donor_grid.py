"""Reproduce and archive one accepted Ciulli 15 N donor-flux grid root."""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import scipy

from ehl_solver.limited_audit import audit_limited_state
from ehl_solver.limited_newton import LimitedNewtonError, solve_limited_load
from ehl_solver.limited_system import LimitedLoadProbe
from ehl_solver.prototype import PointContactCase
from ehl_solver.prototype_cli import clean_json


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _save_state(path, probe, state):
    pressure, theta, film = probe.fields(state)
    problem = probe.system.problem
    with Path(path).open("wb") as stream:
        np.savez_compressed(
            stream,
            state_dimensionless=state,
            pressure_pa=pressure,
            theta=theta,
            film_m=film,
            x_m=problem.x_m,
            y_m=problem.y_m,
        )


def _revision():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def run(case_path, output, grid, half_width_mm):
    started = time.perf_counter()
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    base = PointContactCase(**json.loads(Path(case_path).read_text()))
    if base.transport != "donor" or grid < 5 or grid % 2 == 0:
        raise ValueError("A donor case and an odd grid >=5 are required")
    half_width_m = half_width_mm * 1e-3
    case = replace(
        base,
        ny=grid,
        nx=grid,
        domain_half_width_x_m=half_width_m,
        domain_half_width_y_m=half_width_m,
    )
    output.mkdir(parents=True)
    probe = LimitedLoadProbe(case, transport_fraction=0.0)
    metadata = dict(
        status="running",
        case=asdict(case),
        transport_fraction=0.0,
        equation="donor-flux JFO/Elrod-Adams",
        coordinates="(ny,nx), x follows +entrainment; fields are SI",
        command=f"python examples/run_ciulli_donor_grid.py --case {case_path} "
        f"--grid {grid} --half-width-mm {half_width_mm:g} --output {output}",
        runtime=dict(
            python=sys.version,
            numpy=np.__version__,
            scipy=scipy.__version__,
            platform=platform.platform(),
            git_revision=_revision(),
        ),
        fluid=asdict(case.fluid()),
        maximum_corrections=30,
        maximum_models_per_correction=8,
        maximum_krylov_cycles=10,
        gmres_restart=60,
    )
    metadata["case_sha256"] = _digest(case_path)
    source_files = [
        Path(__file__),
        Path(__import__("ehl_solver.limited_newton", fromlist=["__file__"]).__file__),
        Path(__import__("ehl_solver.limited_system", fromlist=["__file__"]).__file__),
        Path(__import__("ehl_solver.limited_audit", fromlist=["__file__"]).__file__),
    ]
    metadata["source_sha256"] = {path.name: _digest(path) for path in source_files}
    metadata["run_script_sha256"] = _digest(__file__)
    summary_path = output / "summary.json"

    def write_summary():
        summary_path.write_text(json.dumps(clean_json(metadata), indent=2, allow_nan=False))

    write_summary()
    latest = {}

    def observe(iteration, state, audit):
        latest["state"] = state.copy()
        metadata["latest_iteration"] = iteration
        metadata["latest_audit"] = audit
        _save_state(output / "latest_state.npz", probe, state)
        write_summary()

    try:
        root = solve_limited_load(
            case,
            transport_fraction=0.0,
            state_observer=observe,
        )
        audit = audit_limited_state(probe, root.state, direct=True)
        if not audit["numerically_accepted"]:
            raise RuntimeError("Independent direct endpoint audit rejected the donor root")
        _save_state(output / "state.npz", probe, root.state)
        metadata.update(
            status="accepted_numerical_root",
            audit=audit,
            newton_corrections=len(root.history),
            elapsed_seconds=time.perf_counter() - started,
            history=[
                {
                    "iteration": record["iteration"],
                    "merit": record["merit"],
                    "models": [
                        {
                            "recovery": model.get("recovery"),
                            "status": model.get("status"),
                            "reason": model.get("reason"),
                            "linear_true_residual": model.get("linear_true_residual"),
                            "linear_iterations": model.get("linear_iterations"),
                            "accepted_step": next(
                                (
                                    trial.get("alpha")
                                    for trial in model.get("trials", [])
                                    if trial.get("accepted")
                                ),
                                None,
                            ),
                        }
                        for model in record["models"]
                    ],
                }
                for record in root.history
            ],
            state_sha256=_digest(output / "state.npz"),
        )
    except (LimitedNewtonError, RuntimeError, ValueError) as exc:
        candidate = latest.get("state")
        if candidate is not None:
            _save_state(output / "candidate_unaccepted.npz", probe, candidate)
        metadata.update(
            status="rejected",
            failure_reason=str(exc),
            elapsed_seconds=time.perf_counter() - started,
        )
        if isinstance(exc, LimitedNewtonError):
            metadata["failure_audit"] = exc.audit
    write_summary()
    if metadata["status"] != "accepted_numerical_root":
        raise RuntimeError(
            f"Ciulli donor run failed; see {summary_path}: {metadata.get('failure_reason')}"
        )
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, default=Path("cases/ciulli_15n_donor.json"))
    parser.add_argument("--grid", type=int, required=True)
    parser.add_argument("--half-width-mm", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run(args.case, args.output, args.grid, args.half_width_mm)
    audit = report["audit"]
    print(
        f"{report['status']}: {args.grid}x{args.grid}, "
        f"half-width={args.half_width_mm:g} mm, "
        f"hmin/hc={audit['minimum_film_m']*1e9:.6g}/"
        f"{audit['central_film_m']*1e9:.6g} nm, "
        f"pmax={_maximum_pressure_mpa(args.output / 'state.npz'):.6g} MPa"
    )
    return 0


def _maximum_pressure_mpa(path):
    with np.load(path, allow_pickle=False) as archive:
        return float(np.max(archive["pressure_pa"]) / 1e6)


if __name__ == "__main__":
    raise SystemExit(main())
