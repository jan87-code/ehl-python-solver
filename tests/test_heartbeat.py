import io
import json
import time

import pytest

from ehl_solver.heartbeat import SolverHeartbeat


def test_heartbeat_flushes_periodic_scalar_records_and_limit_warnings(tmp_path):
    path = tmp_path / "heartbeat.jsonl"
    output = io.StringIO()
    heartbeat = SolverHeartbeat(path, interval_seconds=0.02, stream=output)
    heartbeat.begin_correction(
        28, maximum_newton_iterations=30, maximum_gmres_iterations=4
    )
    heartbeat.update(
        stage="preconditioner_setup",
        emit=True,
        cumulative_jvp_seconds=0.0,
        cumulative_preconditioner_seconds=0.0,
    )
    time.sleep(0.05)
    heartbeat.update(
        stage="gmres",
        emit=True,
        gmres_iterations=4,
        gmres_restart_count=3,
        gmres_stagnation_restarts=3,
        gmres_residual_estimate=0.25,
    )
    heartbeat.close()

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) >= 4
    assert "HEARTBEAT " in output.getvalue()
    last = records[-2]
    assert last["stage"] == "gmres"
    assert last["iteration"] == 28
    assert last["gmres_iterations"] == 4
    assert last["gmres_restart_count"] == 3
    assert last["gmres_stagnation_restarts"] == 3
    assert "gmres_restarts_without_meaningful_reduction" in last["warning"]
    assert "approaching_newton_iteration_limit" in last["warning"]
    assert "approaching_gmres_iteration_limit" in last["warning"]
    assert all(
        not isinstance(value, (list, dict))
        for record in records
        for value in record.values()
    )


def test_heartbeat_rejects_non_scalar_or_nonfinite_metrics(tmp_path):
    heartbeat = SolverHeartbeat(
        tmp_path / "heartbeat.jsonl", interval_seconds=1, stream=io.StringIO()
    )
    with pytest.raises(TypeError, match="scalar"):
        heartbeat.update(stage="gmres", values=[1, 2])
    with pytest.raises(ValueError, match="finite"):
        heartbeat.update(stage="gmres", residual=float("nan"))
    heartbeat.close()


def test_heartbeat_includes_elapsed_superlu_time_during_active_call(tmp_path):
    path = tmp_path / "heartbeat.jsonl"
    heartbeat = SolverHeartbeat(path, interval_seconds=0.01, stream=io.StringIO())
    heartbeat.begin_correction(
        1, maximum_newton_iterations=30, maximum_gmres_iterations=200
    )
    heartbeat.update(
        stage="superlu_factorization",
        emit=True,
        cumulative_preconditioner_seconds=4.0,
        cumulative_superlu_seconds=2.0,
    )
    time.sleep(0.03)
    heartbeat.close()

    active = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if '"stage":"superlu_factorization"' in line
    ]
    assert active
    assert active[-1]["cumulative_preconditioner_seconds"] > 4.0
    assert active[-1]["cumulative_superlu_seconds"] > 2.0
    assert active[-1]["current_superlu_operation_elapsed_seconds"] > 0
