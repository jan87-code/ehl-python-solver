"""Weak mathematical limited-flux correction benchmarks; no physical roots.

All budgets remain 10 Krylov cycles, restart 60, original true-residual and
feasibility/Armijo gates. Single correction trials are explicitly unaccepted
EHL states. No nonlinear iteration or load-continuation driver is provided.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ehl_solver.cavitating_newton import CavitatingLinearError, solve_cavitating_correction
from ehl_solver.limited_system import LimitedLoadLinearization, LimitedLoadProbe
from ehl_solver.prototype import PointContactCase
from ehl_solver.prototype_cli import clean_json


def mathematical_state(ny, nx):
    case = PointContactCase(load_n=1.0, ny=ny, nx=nx)
    probe = LimitedLoadProbe(case)
    rng = np.random.default_rng(523)
    p, theta = np.zeros((ny, nx)), np.ones((ny, nx))
    p[1:-1, 1:-1] = (
        0.02 * probe.system.problem.pressure_scale_pa * (1 + 0.1 * rng.random((ny - 2, nx - 2)))
    )
    state = probe.system.pack(SimpleNamespace(pressure_pa=p, fractional_film_content=theta), 2e-6)
    guide = rng.normal(size=probe.size) * 1e-3
    return case, state, guide


def trial_history(model, state, direction):
    residual = model.residual(state)
    merit = float(residual @ residual / (2 * model.size))
    slope = model.merit_slope(state, residual, direction)
    predicted = float(residual @ model.jacobian_vector(state, direction) / model.size)
    report = dict(
        merit=merit, true_directional_slope=slope, frozen_model_slope=predicted, trials=[]
    )
    if slope >= 0:
        report["reason"] = "nonnegative actual directional slope; no line search"
        return report
    margin = state[: 2 * model.n] + 1e-10
    change = direction[: 2 * model.n]
    bound = 1.0
    decreasing = change < -margin
    if np.any(decreasing):
        bound = min(bound, 0.99 * float(np.min(margin[decreasing] / -change[decreasing])))
    deficit = state[model.n : 2 * model.n]
    velocity = direction[model.n : 2 * model.n]
    rising = velocity > 1 + 1e-10 - deficit
    if np.any(rising):
        bound = min(bound, 0.99 * float(np.min((1 + 1e-10 - deficit[rising]) / velocity[rising])))
    _, _, film = model.probe.fields(state)
    dh, _, _ = model.probe.directions(state, direction)
    closing = dh < -(film - model.problem.film_floor_m)
    if np.any(closing):
        bound = min(
            bound, 0.99 * float(np.min((film[closing] - model.problem.film_floor_m) / -dh[closing]))
        )
    report["fraction_to_boundary"] = bound
    for k in range(24):
        alpha = 1.0 if k == 0 else bound * 2.0 ** (-(k - 1))
        trial = state + alpha * direction
        feasible = model.probe.feasible(trial)
        value = None
        if feasible:
            r = model.probe.residual(trial)
            value = float(r @ r / (2 * model.size))
        good = feasible and value <= merit + 1e-4 * alpha * slope and value < merit
        report["trials"].append(
            dict(
                stage="full" if k == 0 else "affine_bound_backtrack",
                alpha=alpha,
                feasible=feasible,
                merit=value,
                armijo_comparison=bool(good),
            )
        )
        if good:
            break
    return report


def main(output):
    output.mkdir(parents=True, exist_ok=False)
    report = dict(
        scope=__doc__,
        operator_transport="limited_capacity",
        case_transport_used_for_reference_kinematics_only=True,
        rows=[],
        physical_validation_complete=False,
        candidate_is_converged=False,
    )

    def write():
        (output / "summary.json").write_text(
            json.dumps(clean_json(report), indent=2, allow_nan=False)
        )

    write()
    for shape in ((5, 7), (9, 13), (17, 25)):
        for mode in ("material", "local"):
            case, state, guide = mathematical_state(*shape)
            model = LimitedLoadLinearization(case, state, guide, preconditioner=mode)
            matrix = model.frozen_matrix(state)
            row = dict(
                shape_ny_nx=shape,
                case=asdict(case),
                preconditioner=mode,
                maximum_krylov_cycles=10,
                gmres_restart=60,
                matrix_nnz=matrix.nnz,
                sparse_payload_bytes=int(
                    matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
                ),
                payload_is_total_peak_memory=False,
                status="in_progress",
                nondimensionalization=dict(
                    length_m=model.problem.length_scale_m,
                    film_m=model.problem.film_scale_m,
                    pressure_pa=model.problem.pressure_scale_pa,
                    flux_kg_m_s=model.probe.system.base.flux_scale,
                    mass_row_scale=model.probe.system.base.mass_scale,
                ),
            )
            report["rows"].append(row)
            write()
            try:
                correction = solve_cavitating_correction(
                    model, state, maximum_krylov_cycles=10, gmres_restart=60
                )
                r = model.residual(state)
                measured = float(
                    np.linalg.norm(model.jacobian_vector(state, correction.direction) + r)
                    / np.linalg.norm(r)
                )
                row.update(
                    status="linear_model_solved",
                    independent_true_relative_residual=measured,
                    correction=asdict(correction),
                    trial_diagnostics=trial_history(model, state, correction.direction),
                )
                if model.size <= 128:
                    dense = np.column_stack(
                        [model.jacobian_vector(state, e) for e in np.eye(model.size)]
                    )
                    direct = np.linalg.solve(dense, -r)
                    row["dense_oracle_relative_error"] = float(
                        np.linalg.norm(direct - correction.direction) / np.linalg.norm(direct)
                    )
                np.savez_compressed(
                    output / f"correction_{shape[0]}x{shape[1]}_{mode}.npz",
                    state_dimensionless=state,
                    guide_dimensionless=guide,
                    correction_dimensionless=correction.direction,
                    residual=r,
                    x_m=model.problem.x_m,
                    y_m=model.problem.y_m,
                )
            except CavitatingLinearError as exc:
                row.update(
                    status="linear_model_rejected", reason=str(exc), diagnostics=exc.diagnostics
                )
            write()
    case, state, guide = mathematical_state(9, 13)
    for name, cycles, restart, schur in [
        ("forced_bordered", 10, 60, 2.0),
        ("exhausted_krylov", 1, 1, 1e-10),
    ]:
        model = LimitedLoadLinearization(case, state, guide)
        row = dict(
            name=name, maximum_krylov_cycles=cycles, gmres_restart=restart, schur_tolerance=schur
        )
        try:
            correction = solve_cavitating_correction(
                model,
                state,
                maximum_krylov_cycles=cycles,
                gmres_restart=restart,
                schur_tolerance=schur,
            )
            row.update(status="linear_model_solved", correction=asdict(correction))
        except CavitatingLinearError as exc:
            row.update(status="linear_model_rejected", reason=str(exc), diagnostics=exc.diagnostics)
        report.setdefault("recovery_checks", []).append(row)
        write()
    print(json.dumps(clean_json(report), indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
