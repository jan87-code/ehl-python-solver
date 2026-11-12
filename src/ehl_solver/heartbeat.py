"""Low-overhead scalar progress reporting for long solver corrections."""

import json
import math
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


class SolverHeartbeat:
    def __init__(self, path, *, interval_seconds=30.0, stream=None):
        if (
            isinstance(interval_seconds, bool)
            or not isinstance(interval_seconds, (int, float))
            or not math.isfinite(interval_seconds)
            or interval_seconds <= 0
        ):
            raise ValueError("interval_seconds must be finite and positive")
        self.path = Path(path)
        self._file = self.path.open("a", encoding="utf-8", buffering=1)
        self._stream = sys.stdout if stream is None else stream
        self._interval = float(interval_seconds)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._started = time.perf_counter()
        self._correction_started = None
        self._stage_started = self._started
        self._last_jvp_at = None
        self._state = {"stage": "starting", "heartbeat_status": "running"}
        self._sequence = 0
        self._last_emitted_gmres_iteration = 0
        self._error = None
        self._thread = threading.Thread(
            target=self._run, name="ehl-solver-heartbeat", daemon=True
        )
        self._thread.start()
        self.update(stage="run_start", emit=True)

    def begin_correction(
        self, iteration, *, maximum_newton_iterations, maximum_gmres_iterations
    ):
        self._raise_if_failed()
        with self._lock:
            now = time.perf_counter()
            self._correction_started = now
            self._stage_started = now
            self._last_jvp_at = now
            self._state = {
                "iteration": int(iteration),
                "stage": "checkpoint",
                "heartbeat_status": "running",
                "maximum_newton_iterations": int(maximum_newton_iterations),
                "maximum_gmres_iterations": int(maximum_gmres_iterations),
            }
            self._last_emitted_gmres_iteration = 0
            self._emit_locked(now)

    def update(self, *, stage, iteration=None, emit=False, **metrics):
        self._raise_if_failed()
        if not isinstance(stage, str) or not stage:
            raise ValueError("stage must be a nonempty string")
        for name, value in metrics.items():
            if not isinstance(value, (str, int, float, bool, type(None))):
                raise TypeError(f"heartbeat field {name!r} must be a scalar")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"heartbeat field {name!r} must be finite")
        with self._lock:
            now = time.perf_counter()
            previous_stage = self._state.get("stage")
            if previous_stage != stage:
                self._stage_started = now
            self._state["stage"] = stage
            if iteration is not None:
                self._state["iteration"] = int(iteration)
            previous_jvp = self._state.get("cumulative_jvp_seconds")
            self._state.update(metrics)
            jvp_seconds = self._state.get("cumulative_jvp_seconds")
            if (
                isinstance(jvp_seconds, (int, float))
                and jvp_seconds > (previous_jvp or 0)
            ):
                self._last_jvp_at = now
            gmres_iterations = self._state.get("gmres_iterations", 0)
            emit_for_gmres = (
                isinstance(gmres_iterations, int)
                and gmres_iterations - self._last_emitted_gmres_iteration >= 5
            )
            emit_for_restart = stage == "gmres_restart"
            if emit or emit_for_gmres or emit_for_restart:
                self._emit_locked(now)
                if isinstance(gmres_iterations, int):
                    self._last_emitted_gmres_iteration = gmres_iterations
        self._raise_if_failed()

    def end_correction(self, iteration, **metrics):
        self.update(
            stage="correction_complete", iteration=iteration, emit=True, **metrics
        )
        with self._lock:
            self._correction_started = None

    def close(self, *, status="finished"):
        if self._thread.is_alive():
            with self._lock:
                self._state["heartbeat_status"] = status
                self._state["stage"] = "run_end"
                self._emit_locked(time.perf_counter())
            self._stop.set()
            self._thread.join()
        self._file.close()
        self._raise_if_failed()

    def _run(self):
        while not self._stop.wait(self._interval):
            try:
                with self._lock:
                    if self._correction_started is not None:
                        self._emit_locked(time.perf_counter())
            except OSError as exc:
                self._error = exc
                self._stop.set()

    def _emit_locked(self, now):
        correction_elapsed = (
            None
            if self._correction_started is None
            else now - self._correction_started
        )
        stage_elapsed = now - self._stage_started
        warnings = []
        stage = self._state.get("stage")
        if correction_elapsed is not None and correction_elapsed >= 300:
            warnings.append("correction_over_300_seconds")
        if stage == "preconditioner_setup" and stage_elapsed >= 60:
            warnings.append("preconditioner_setup_over_60_seconds")
        if stage == "superlu_factorization" and stage_elapsed >= 60:
            warnings.append("superlu_factorization_over_60_seconds")
        if stage == "preconditioner_apply" and stage_elapsed >= 10:
            warnings.append("preconditioner_apply_over_10_seconds")
        if stage == "superlu_solve" and stage_elapsed >= 10:
            warnings.append("superlu_solve_over_10_seconds")
        if self._state.get("preconditioner_application_seconds", 0.0) >= 10:
            warnings.append("preconditioner_application_over_10_seconds")
        if self._state.get("preconditioner_setup_seconds", 0.0) >= 60:
            warnings.append("preconditioner_setup_over_60_seconds")
        if stage == "audit" and stage_elapsed >= 60:
            warnings.append("audit_over_60_seconds")
        if stage in {
            "preconditioner_setup",
            "preconditioner_apply",
            "superlu_factorization",
            "superlu_solve",
            "matrix_assembly",
        }:
            if self._last_jvp_at is not None and now - self._last_jvp_at >= 30:
                warnings.append("cpu_stage_without_recent_cuda_jvp")
        if self._state.get("gmres_stagnation_restarts", 0) >= 3:
            warnings.append("gmres_restarts_without_meaningful_reduction")
        iteration = self._state.get("iteration")
        max_newton = self._state.get("maximum_newton_iterations")
        if (
            isinstance(iteration, int)
            and isinstance(max_newton, int)
            and iteration >= max_newton - 2
        ):
            warnings.append("approaching_newton_iteration_limit")
        gmres_iterations = self._state.get("gmres_iterations")
        max_gmres = self._state.get("maximum_gmres_iterations")
        if (
            isinstance(gmres_iterations, int)
            and isinstance(max_gmres, int)
            and gmres_iterations >= math.ceil(0.9 * max_gmres)
        ):
            warnings.append("approaching_gmres_iteration_limit")

        self._sequence += 1
        record = {
            "sequence": self._sequence,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "run_elapsed_seconds": now - self._started,
            "correction_elapsed_seconds": correction_elapsed,
            "stage_elapsed_seconds": stage_elapsed,
            **self._state,
        }
        if stage in {
            "preconditioner_setup",
            "preconditioner_apply",
            "superlu_factorization",
            "superlu_solve",
        }:
            cumulative_preconditioner = self._state.get(
                "cumulative_preconditioner_seconds"
            )
            if isinstance(cumulative_preconditioner, (int, float)):
                record["cumulative_preconditioner_seconds"] = (
                    cumulative_preconditioner + stage_elapsed
                )
        if stage in {"superlu_factorization", "superlu_solve"}:
            cumulative_superlu = self._state.get("cumulative_superlu_seconds")
            if isinstance(cumulative_superlu, (int, float)):
                record["cumulative_superlu_seconds"] = (
                    cumulative_superlu + stage_elapsed
                )
                record["current_superlu_operation_elapsed_seconds"] = stage_elapsed
        warning = self._state.get("warning")
        if warning:
            warnings.append(str(warning))
        if warnings:
            record["warning"] = ";".join(dict.fromkeys(warnings))
        line = json.dumps(record, separators=(",", ":"), allow_nan=False)
        self._file.write(line + "\n")
        self._file.flush()
        print(f"HEARTBEAT {line}", file=self._stream, flush=True)

    def _raise_if_failed(self):
        if self._error is not None:
            raise OSError("heartbeat writer failed") from self._error
