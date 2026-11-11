"""Reproduce and archive one accepted Ciulli 15 N donor-flux grid root."""

import argparse
import ctypes
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import scipy
from scipy.interpolate import RegularGridInterpolator

from ehl_solver.limited_audit import audit_limited_state
from ehl_solver.limited_newton import (
    LimitedNewtonError,
    LimitedObserverError,
    solve_limited_load,
)
from ehl_solver.limited_system import LimitedLoadProbe
from ehl_solver.prototype import PointContactCase
from ehl_solver.prototype_cli import clean_json
from ehl_solver.visualization import write_run_summary


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _save_state(path, probe, state, case):
    pressure, theta, film = probe.fields(state)
    problem = probe.system.problem
    deformation = (
        film
        - state[-1] * problem.film_scale_m
        - problem.rigid_geometry_m
    )
    density, viscosity, _, _ = case.fluid().properties(pressure)
    with Path(path).open("wb") as stream:
        np.savez_compressed(
            stream,
            state_dimensionless=state,
            pressure_pa=pressure,
            theta=theta,
            film_m=film,
            deformation_m=deformation,
            density_kg_m3=density,
            viscosity_pa_s=viscosity,
            x_m=problem.x_m,
            y_m=problem.y_m,
            case_json=json.dumps(asdict(case), sort_keys=True),
        )


def _seed_state(seed_path, case, probe):
    source = Path(seed_path)
    if source.is_dir():
        summary = json.loads((source / "summary.json").read_text())
        if (
            summary.get("status") != "accepted_numerical_root"
            or summary.get("transport_fraction") != 0
            or summary.get("audit", {}).get("numerically_accepted") is not True
        ):
            raise ValueError(f"{source} is not an accepted donor-flux seed")
        archive_path = source / "state.npz"
        if _digest(archive_path) != summary.get("state_sha256"):
            raise ValueError(f"{source} seed checksum does not match its summary")
        source_case_data = summary["case"]
        provenance = {
            "path": str(source),
            "kind": "accepted donor root",
            "state_sha256": summary["state_sha256"],
        }
    elif source.is_file():
        archive_path = source
        source_case_data = None
        provenance = {
            "path": str(source),
            "kind": "checkpoint or state archive",
            "state_sha256": _digest(source),
        }
    else:
        raise FileNotFoundError(f"Seed does not exist: {source}")

    with np.load(archive_path, allow_pickle=False) as archive:
        fields = {name: archive[name].copy() for name in archive.files}
    if source_case_data is None:
        if "case_json" not in fields:
            raise ValueError("Checkpoint seed must include its source case metadata")
        source_case_data = json.loads(str(fields["case_json"]))
    source_case = PointContactCase(**source_case_data)
    if source_case.physical_signature() != case.physical_signature():
        raise ValueError("Seed and target must use identical physical parameters")
    if (
        source_case.domain_half_width_x_m != case.domain_half_width_x_m
        or source_case.domain_half_width_y_m != case.domain_half_width_y_m
    ):
        raise ValueError("Seed and target must use the identical physical domain")
    state = fields.get("state_dimensionless")
    pressure, theta = fields.get("pressure_pa"), fields.get("theta")
    old_x, old_y = fields.get("x_m"), fields.get("y_m")
    if (
        state is None
        or pressure is None
        or theta is None
        or old_x is None
        or old_y is None
        or pressure.shape != (len(old_y), len(old_x))
        or theta.shape != pressure.shape
        or state.ndim != 1
        or not all(np.all(np.isfinite(value)) for value in (state, pressure, theta))
    ):
        raise ValueError("Seed must contain finite state, pressure, content and SI axes")
    if source_case.nx > case.nx or source_case.ny > case.ny:
        raise ValueError("Seed cannot coarsen either target grid axis")
    if (
        source_case.nx == case.nx
        and source_case.ny == case.ny
        and old_x.shape == probe.system.problem.x_m.shape
        and old_y.shape == probe.system.problem.y_m.shape
    ):
        target_state = state.astype(np.float64, copy=True)
        provenance["interpolation"] = "none; exact-grid restart"
    else:
        new_x, new_y = probe.system.problem.x_m, probe.system.problem.y_m
        yy, xx = np.meshgrid(new_y, new_x, indexing="ij")
        points = np.column_stack((yy.ravel(), xx.ravel()))
        pressure_new = RegularGridInterpolator((old_y, old_x), pressure)(points).reshape(
            case.ny, case.nx
        )
        theta_new = RegularGridInterpolator((old_y, old_x), theta)(points).reshape(
            case.ny, case.nx
        )
        offset_m = state[-1] * probe.system.problem.film_scale_m
        target_state = np.r_[
            (pressure_new[1:-1, 1:-1] / probe.system.problem.pressure_scale_pa).ravel(),
            (1 - theta_new[1:-1, 1:-1]).ravel(),
            offset_m / probe.system.problem.film_scale_m,
        ].astype(np.float64, copy=False)
        provenance["interpolation"] = "bilinear pressure/content; physical offset preserved"
    if not probe.feasible(target_state):
        raise ValueError("Prolongated seed violates target-grid feasibility")
    provenance.update(
        source_grid_ny_nx=[source_case.ny, source_case.nx],
        target_grid_ny_nx=[case.ny, case.nx],
        accepted_as_solution=False,
    )
    return target_state, provenance


def _persist_solver_record(output, record, previous_active):
    diagnostics = Path(output) / "diagnostics"
    diagnostics.mkdir(exist_ok=True)
    metadata = {key: value for key, value in record.items() if key != "models"}
    models, active_masks = [], {}
    current_active = previous_active
    for index, model in enumerate(record.get("models", [])):
        details = {
            key: value
            for key, value in model.items()
            if key not in ("direction", "guide", "active_full")
        }
        direction, guide = model.get("direction"), model.get("guide")
        if isinstance(direction, np.ndarray):
            details["correction_norm_l2"] = float(np.linalg.norm(direction))
            details["correction_norm_linf"] = float(np.max(abs(direction)))
            accepted = next(
                (trial for trial in model.get("trials", []) if trial.get("accepted")),
                None,
            )
            details["accepted_correction_norm_l2"] = (
                float(accepted["alpha"] * details["correction_norm_l2"])
                if accepted is not None
                else None
            )
            details["accepted_correction_norm_linf"] = (
                float(accepted["alpha"] * details["correction_norm_linf"])
                if accepted is not None
                else None
            )
        else:
            details.update(
                correction_norm_l2=None,
                correction_norm_linf=None,
                accepted_correction_norm_l2=None,
                accepted_correction_norm_linf=None,
            )
        details["guide_norm_l2"] = (
            float(np.linalg.norm(guide)) if isinstance(guide, np.ndarray) else None
        )
        active = model.get("active_full")
        if isinstance(active, np.ndarray):
            active = np.asarray(active, dtype=bool)
            details["active_full_count"] = int(np.count_nonzero(active))
            details["active_full_size"] = int(active.size)
            details["active_set_changed_nodes"] = (
                int(np.count_nonzero(active != current_active))
                if current_active is not None and current_active.shape == active.shape
                else None
            )
            active_masks[f"model_{index:02d}_active_full_packed"] = np.packbits(active)
            active_masks[f"model_{index:02d}_active_full_shape"] = np.asarray(
                active.shape, dtype=np.int64
            )
            current_active = active.copy()
        else:
            details.update(
                active_full_count=None,
                active_full_size=None,
                active_set_changed_nodes=None,
            )
        models.append(details)
    metadata["models"] = models
    metadata["active_mask_archive"] = None
    if active_masks:
        mask_path = diagnostics / f"active_sets_{record['iteration']:04d}.npz"
        np.savez_compressed(mask_path, **active_masks)
        metadata["active_mask_archive"] = str(mask_path)
    record_path = diagnostics / f"iteration_{record['iteration']:04d}.json"
    record_path.write_text(json.dumps(clean_json(metadata), indent=2, allow_nan=False))
    return str(record_path), current_active


def _reconcile_persisted_history(output, compact_history):
    paths = []
    for record in compact_history:
        path = Path(output) / "diagnostics" / f"iteration_{record['iteration']:04d}.json"
        detail = json.loads(path.read_text())
        if len(detail["models"]) != len(record["models"]):
            raise RuntimeError(f"Persisted solver history is incomplete: {path}")
        accepted = []
        for model_detail, model_summary in zip(
            detail["models"], record["models"], strict=True
        ):
            alpha = model_summary["accepted_step"]
            if alpha is None:
                continue
            matches = [
                trial
                for trial in model_detail["trials"]
                if trial.get("alpha") == alpha
            ]
            if len(matches) != 1:
                raise RuntimeError(f"Accepted line-search step is ambiguous in {path}")
            matches[0].update(
                accepted=True,
                reason="Accepted best feasible directional Armijo trial",
            )
            model_detail["accepted_step"] = alpha
            correction_norm = model_detail.get("correction_norm_l2")
            model_detail["accepted_correction_norm_l2"] = (
                alpha * correction_norm if correction_norm is not None else None
            )
            accepted.append(model_detail["recovery"])
        if len(accepted) > 1:
            raise RuntimeError(f"Multiple accepted recovery models in {path}")
        detail["accepted_recovery"] = accepted[0] if accepted else None
        detail["accepted_alpha"] = next(
            (
                model["accepted_step"]
                for model in detail["models"]
                if model.get("accepted_step") is not None
            ),
            None,
        )
        path.write_text(json.dumps(clean_json(detail), indent=2, allow_nan=False))
        paths.append(str(path))
    return paths


def _revision():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _scipy_openblas_thread_count():
    library_paths = Path(scipy.__file__).resolve().parent.parent.glob(
        "scipy.libs/*openblas*"
    )
    loader = getattr(ctypes, "WinDLL", ctypes.CDLL)
    for path in library_paths:
        try:
            library = loader(str(path))
        except OSError:
            continue
        getter = getattr(library, "scipy_openblas_get_num_threads", None)
        if getter is not None:
            getter.restype = ctypes.c_int
            return int(getter())
    return None


def _numpy_openblas_thread_count():
    library_paths = Path(np.__file__).resolve().parent.parent.glob(
        "numpy.libs/*openblas*"
    )
    loader = getattr(ctypes, "WinDLL", ctypes.CDLL)
    for path in library_paths:
        try:
            library = loader(str(path))
        except OSError:
            continue
        for symbol in (
            "openblas_get_num_threads",
            "openblas_get_num_threads_",
            "openblas_get_num_threads64_",
        ):
            getter = getattr(library, symbol, None)
            if getter is not None:
                getter.restype = ctypes.c_int
                return int(getter())
    return None


def run(
    case_path,
    output,
    grid,
    half_width_mm,
    *,
    seed=None,
    maximum_physical_nodes=None,
):
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
        maximum_physical_nodes=(
            base.maximum_physical_nodes
            if maximum_physical_nodes is None
            else maximum_physical_nodes
        ),
    )
    probe = LimitedLoadProbe(case, transport_fraction=0.0)
    initial_state, seed_provenance = (
        _seed_state(seed, case, probe)
        if seed is not None
        else (None, {"kind": "Hertz initial guess; not a solution"})
    )
    output.mkdir(parents=True)
    command = [
        "python",
        "examples/run_ciulli_donor_grid.py",
        "--case",
        str(case_path),
        "--grid",
        str(grid),
        "--half-width-mm",
        f"{half_width_mm:g}",
        "--output",
        str(output),
    ]
    if seed is not None:
        command.extend(("--seed", str(seed)))
    if maximum_physical_nodes is not None:
        command.extend(("--maximum-physical-nodes", str(maximum_physical_nodes)))
    metadata = dict(
        status="running",
        case=asdict(case),
        transport_fraction=0.0,
        equation="donor-flux JFO/Elrod-Adams",
        coordinates="(ny,nx), x follows +entrainment; fields are SI",
        command=" ".join(command),
        runtime=dict(
            python=sys.version,
            numpy=np.__version__,
            scipy=scipy.__version__,
            platform=platform.platform(),
            git_revision=_revision(),
            cpu_count=os.cpu_count(),
            thread_environment={
                key: os.environ.get(key)
                for key in (
                    "OPENBLAS_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "BLIS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
            numpy_openblas_active_threads=_numpy_openblas_thread_count(),
            scipy_openblas_active_threads=_scipy_openblas_thread_count(),
            linear_algebra_backend=(
                "SciPy sparse GMRES/SuperLU; NumPy and SciPy OpenBLAS "
                "thread counts queried independently. NumPy FFT and Python "
                "matrix-free operations are not configured as parallel solvers."
            ),
        ),
        fluid=asdict(case.fluid()),
        maximum_corrections=30,
        maximum_models_per_correction=8,
        maximum_krylov_cycles=10,
        gmres_restart=60,
        seed=seed_provenance,
        field_archive_keys=[
            "pressure_pa",
            "film_m",
            "deformation_m",
            "density_kg_m3",
            "viscosity_pa_s",
            "theta",
        ],
        history_directory="diagnostics",
        checkpoint_directory="checkpoints",
        termination_reason=None,
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
        _save_state(output / "latest_state.npz", probe, state, case)
        checkpoint = output / "checkpoints" / f"checkpoint_{iteration:04d}.npz"
        checkpoint.parent.mkdir(exist_ok=True)
        _save_state(checkpoint, probe, state, case)
        metadata["latest_checkpoint"] = str(checkpoint)
        write_summary()

    previous_active = None
    diagnostic_paths = []

    def observe_diagnostics(iteration, record):
        nonlocal previous_active
        path, previous_active = _persist_solver_record(output, record, previous_active)
        diagnostic_paths.append(path)
        metadata["latest_diagnostic_record"] = path
        metadata["diagnostic_record_count"] = len(diagnostic_paths)
        write_summary()

    try:
        root = solve_limited_load(
            case,
            initial_state=initial_state,
            transport_fraction=0.0,
            state_observer=observe,
            diagnostic_observer=observe_diagnostics,
        )
        audit = audit_limited_state(probe, root.state, direct=True)
        if not audit["numerically_accepted"]:
            raise RuntimeError("Independent direct endpoint audit rejected the donor root")
        _save_state(output / "state.npz", probe, root.state, case)
        metadata.update(
            status="accepted_numerical_root",
            termination_reason="independent direct endpoint audit accepted",
            audit=audit,
            newton_corrections=len(root.history),
            diagnostic_records=diagnostic_paths,
            elapsed_seconds=time.perf_counter() - started,
            history=[
                {
                    "iteration": record["iteration"],
                    "merit": record["merit"],
                    "nonlinear_residual_l2": record["nonlinear_residual_l2"],
                    "nonlinear_residual_linf": record["nonlinear_residual_linf"],
                    "force_residual": record["force_residual"],
                    "mass_residual_linf": record["mass_residual_linf"],
                    "complementarity_residual_linf": record[
                        "complementarity_residual_linf"
                    ],
                    "models": [
                        {
                            "recovery": model.get("recovery"),
                            "status": model.get("status"),
                            "reason": model.get("reason"),
                            "linear_true_residual": model.get("linear_true_residual"),
                            "linear_iterations": model.get("linear_iterations"),
                            "krylov_histories": model.get("krylov_histories"),
                            "correction_norm_l2": (
                                float(np.linalg.norm(model["direction"]))
                                if isinstance(model.get("direction"), np.ndarray)
                                else None
                            ),
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
        metadata["diagnostic_records"] = _reconcile_persisted_history(
            output, metadata["history"]
        )
    except (LimitedNewtonError, LimitedObserverError, RuntimeError, ValueError) as exc:
        candidate = latest.get("state")
        if isinstance(exc, LimitedNewtonError):
            active = None
            failure_records = []
            for record in exc.history:
                path, active = _persist_solver_record(output, record, active)
                failure_records.append(path)
            metadata["failure_history"] = failure_records
            metadata["diagnostic_record_count"] = len(exc.history)
            candidate = exc.state
        if candidate is not None:
            _save_state(output / "candidate_unaccepted.npz", probe, candidate, case)
        metadata.update(
            status="rejected",
            failure_reason=str(exc),
            termination_reason=str(exc),
            elapsed_seconds=time.perf_counter() - started,
        )
        if isinstance(exc, LimitedNewtonError):
            metadata["failure_audit"] = exc.audit
    figure_state = output / (
        "state.npz"
        if metadata["status"] == "accepted_numerical_root"
        else "candidate_unaccepted.npz"
    )
    if not figure_state.exists():
        figure_state = output / "latest_state.npz"
    if figure_state.exists():
        with np.load(figure_state, allow_pickle=False) as archive:
            figure_path, profiles_path = write_run_summary(
                archive["x_m"],
                archive["y_m"],
                archive["pressure_pa"],
                archive["film_m"],
                directory=output,
                case_name=Path(case_path).stem,
                load_n=case.load_n,
                formulation="donor-flux",
                status=metadata["status"],
                force_audit=metadata.get("audit", metadata.get("failure_audit")),
                newton_corrections=metadata.get(
                    "newton_corrections", metadata.get("diagnostic_record_count")
                ),
                backend="CPU (NumPy/SciPy)",
                runtime_seconds=metadata.get("elapsed_seconds"),
            )
        metadata["run_summary_figure"] = figure_path.name
        metadata["run_summary_profiles"] = profiles_path.name
    else:
        metadata["run_summary_unavailable_reason"] = (
            "No valid or checkpointed field was available for plotting."
        )
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
    parser.add_argument("--seed", type=Path)
    parser.add_argument("--maximum-physical-nodes", type=int)
    args = parser.parse_args(argv)
    report = run(
        args.case,
        args.output,
        args.grid,
        args.half_width_mm,
        seed=args.seed,
        maximum_physical_nodes=args.maximum_physical_nodes,
    )
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
