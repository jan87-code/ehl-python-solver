"""Parent comparison validates output identity; physical gates run in workers."""

import copy
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from examples import benchmark_reduced_material as benchmark


@pytest.fixture
def pair(tmp_path):
    # Synthetic comparison-only fields, not asserted to be physical 200 N roots.
    fields = {"pressure_pa": np.ones((3, 5)), "theta": np.ones((3, 5)),
              "film_m": np.full((3, 5), 1e-6)}
    hashes = {name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in fields.items()}
    rows = {}
    for mode in ("full", "reduced"):
        directory = tmp_path / f"grid_0_{mode}"
        directory.mkdir()
        np.savez_compressed(directory / "solution.npz", **fields)
        rows[mode] = {"status": "reference-audited nonlinear repair passed",
                      "source": {"source_sha256": "state", "summary_sha256": "summary"},
                      "field_factorization": mode, "elasticity_mode": "cached",
                      "measurement_histories": [[{"models": [{"field_factorization": mode}]}]],
                      "measurement_field_sha256": [hashes.copy()],
                      "audit": {"offset_m": 1e-6}, "elapsed_s": 2 if mode == "full" else 1}
    return tmp_path, fields, rows


def test_identical_audited_outputs_compare_and_overwrite_is_refused(pair):
    directory, _, rows = pair
    comparison = benchmark.compare_pair(directory, 0, rows)
    assert comparison["pressure_relative_l2_difference"] == 0
    assert comparison["film_maximum_difference_m"] == 0
    assert comparison["full_over_reduced_repair_runtime"] == 2
    with pytest.raises(FileExistsError):
        benchmark.main([], directory, 1)


@pytest.mark.parametrize("failure", ["unaccepted", "snapshot", "mode", "recovery"])
def test_unaccepted_or_mismatched_workers_are_not_compared(pair, failure):
    directory, _, initial = pair
    rows = copy.deepcopy(initial)
    if failure == "unaccepted":
        rows["reduced"]["status"] = "rejected nonlinear solve"
    elif failure == "snapshot":
        rows["reduced"]["source"]["source_sha256"] = "changed"
    elif failure == "mode":
        rows["reduced"]["field_factorization"] = "full"
    else:
        rows["reduced"]["measurement_histories"][0][0]["models"][0]["field_factorization"] = "full"
    with pytest.raises(ValueError):
        benchmark.compare_pair(directory, 0, rows)


@pytest.mark.parametrize("failure", ["modified", "nonfinite", "different_root", "zero"])
def test_corrupt_or_disagreeing_outputs_cannot_pass(pair, failure):
    directory, fields, rows = pair
    actual = {k: a.copy() for k, a in fields.items()}
    if failure == "modified":
        actual["pressure_pa"][1, 2] += .1
    elif failure == "nonfinite":
        actual["pressure_pa"][1, 2] = np.nan
    elif failure == "different_root":
        actual["pressure_pa"][1, 2] += .1
        rows["reduced"]["measurement_field_sha256"][-1]["pressure_pa"] = hashlib.sha256(
            actual["pressure_pa"].tobytes()).hexdigest()
    else:
        for mode in ("full", "reduced"):
            zero = {k: a.copy() for k, a in fields.items()}
            zero["pressure_pa"][:] = 0
            rows[mode]["measurement_field_sha256"][-1]["pressure_pa"] = hashlib.sha256(
                zero["pressure_pa"].tobytes()).hexdigest()
            np.savez_compressed(directory / f"grid_0_{mode}/solution.npz", **zero)
        with pytest.raises(ValueError, match="nonzero"):
            benchmark.compare_pair(directory, 0, rows)
        return
    np.savez_compressed(directory / "grid_0_reduced/solution.npz", **actual)
    with pytest.raises((ValueError, RuntimeError)):
        benchmark.compare_pair(directory, 0, rows)


def test_failed_worker_is_retained_and_other_mode_is_still_attempted(tmp_path, monkeypatch):
    import json

    calls = []

    def fail(command, **kwargs):
        mode = command[command.index("--worker-mode")+1]
        output = type(tmp_path)(command[command.index("--output-directory")+1])
        output.mkdir()
        (output / "summary.json").write_text(json.dumps({"status": "rejected nonlinear solve"}))
        calls.append(mode)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(benchmark.subprocess, "run", fail)
    monkeypatch.setattr(benchmark, "provenance", lambda: {})
    target = tmp_path / "comparison"
    with pytest.raises(RuntimeError, match="worker failed"):
        benchmark.main([tmp_path / "source"], target, 1)
    assert calls == ["full", "reduced"]
    report = json.loads((target / "summary.json").read_text())
    assert report["status"] == "worker failures; incomplete"
