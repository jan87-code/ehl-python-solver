import io
import json
import time

import pytest

from ehl_solver.heartbeat import SolverHeartbeat, _CorrectionTiming


def test_correction_timing_partitions_wall_time_and_nests_gmres():
    timing = _CorrectionTiming(0.0)
    for stage, now in (
        ("audit", 0.0),
        ("model_setup", 2.0),
        ("superlu_factorization", 5.0),
        ("gmres_start", 9.0),
        ("jvp", 10.0),
        ("jvp_complete", 13.0),
        ("superlu_solve", 14.0),
        ("superlu_solve_complete", 16.0),
        ("gmres_end", 18.0),
        ("jvp", 19.0),
        ("jvp_complete", 20.0),
        ("gmres_true_residual", 22.0),
        ("line_search", 23.0),
        ("active_set_recovery", 26.0),
        ("checkpoint", 28.0),
    ):
        timing.advance(stage, now)
    values = timing.snapshot(30.0)
    assert values["timing_total_seconds"] == 30.0
    assert values["timing_gmres_total_seconds"] == 9.0
    assert values["timing_gmres_vector_seconds"] == 4.0
    assert values["timing_jvp_seconds"] == 4.0
    assert values["timing_preconditioner_application_seconds"] == 2.0
    assert values["timing_true_residual_seconds"] == 3.0
    assert values["timing_coefficient_assembly_seconds"] == 3.0
    assert values["timing_superlu_factorization_seconds"] == 4.0
    assert values["timing_other_seconds"] == 1.0
    assert sum(
        value for name, value in values.items()
        if name not in ("timing_total_seconds", "timing_gmres_total_seconds")
    ) == 30.0
    timing.advance("linear_solve", 30.0)
    timing.advance("gmres_start", 31.0)
    timing.advance("jvp", 32.0)
    timing.advance("jvp_complete", 34.0)
    timing.advance("gmres_end", 35.0)
    second_model = timing.snapshot(35.0)
    assert second_model["timing_gmres_total_seconds"] == 13.0
    assert second_model["timing_jvp_seconds"] == 6.0


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


def test_cpu_timing_is_opt_in_for_shared_heartbeat(tmp_path):
    path = tmp_path / "heartbeat.jsonl"
    heartbeat = SolverHeartbeat(path, interval_seconds=1, stream=io.StringIO())
    heartbeat.begin_correction(
        0, maximum_newton_iterations=30, maximum_gmres_iterations=200
    )
    heartbeat.update(stage="audit", emit=True)
    heartbeat.begin_correction(
        1, maximum_newton_iterations=30, maximum_gmres_iterations=200,
        timing_enabled=True,
    )
    heartbeat.update(stage="audit", emit=True)
    heartbeat.close()
    records = [json.loads(line) for line in path.read_text().splitlines()]
    first_audit, second_audit = [record for record in records if record["stage"] == "audit"]
    assert "timing_total_seconds" not in first_audit
    assert second_audit["timing_total_seconds"] >= 0
    assert "timing_other_seconds" in second_audit


def test_gmres_limit_warning_uses_current_rhs_iteration(tmp_path):
    path = tmp_path / "heartbeat.jsonl"
    heartbeat = SolverHeartbeat(path, interval_seconds=1, stream=io.StringIO())
    heartbeat.begin_correction(
        0, maximum_newton_iterations=30, maximum_gmres_iterations=5
    )
    heartbeat.update(
        stage="gmres", emit=True, gmres_iterations=7, gmres_solve_iteration=1
    )
    heartbeat.close()
    record = [json.loads(line) for line in path.read_text().splitlines()][-2]
    assert "approaching_gmres_iteration_limit" not in record.get("warning", "")
