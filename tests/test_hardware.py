import json
import subprocess
import sys
from dataclasses import replace

import pytest

from ehl_solver.backend import Backend
from ehl_solver.hardware import (
    BackendStatus,
    Benchmark,
    HardwareReport,
    estimate_convolution_memory,
    select_backend,
)


def _report(*, hip=False, torch=False, free_memory=10**12):
    return HardwareReport(
        system="Linux",
        machine="x86_64",
        processor="test CPU",
        cpu_flags=("sse2",),
        backends={
            "numpy": BackendStatus("numpy", True, True, "cpu", float64=True, fft=True),
            "hip": BackendStatus(
                "hip",
                hip,
                hip,
                "hip:0" if hip else None,
                "gfx1032" if hip else None,
                None if hip else "extension missing",
                hip,
                hip,
                free_memory if hip else None,
                2 * free_memory if hip else None,
                "test-hip",
                False,
            ),
            "torch": BackendStatus(
                "torch",
                torch,
                torch,
                "cuda:0" if torch else None,
                "test-gpu" if torch else None,
                None if torch else "torch missing",
                torch,
                torch,
                free_memory if torch else None,
                2 * free_memory if torch else None,
                "test-torch",
                torch,
            ),
        },
    )


def test_estimated_memory_matches_full_padded_complex_buffers_and_margin():
    estimate = estimate_convolution_memory((5, 7), (9, 13), safety_factor=1.5)
    padded_cells = (5 + 9 - 1) * (7 + 13 - 1)
    hip_raw = 32 * padded_cells + 8 * (2 * 5 * 7 + 9 * 13)
    half_spectrum_cells = (5 + 9 - 1) * ((7 + 13 - 1) // 2 + 1)
    torch_raw = 8 * (5 * 7 + 9 * 13 + padded_cells) + 16 * 3 * half_spectrum_cells
    assert estimate.padded_shape == (13, 19)
    assert estimate.hip_bytes == pytest.approx(hip_raw * 1.5)
    assert estimate.torch_bytes == pytest.approx(torch_raw * 1.5)


@pytest.mark.parametrize(
    "shape,factor",
    [((0, 3), 1.5), ((3, 3), 0.9)],
)
def test_invalid_memory_estimate(shape, factor):
    with pytest.raises(ValueError):
        estimate_convolution_memory(shape, safety_factor=factor)


def test_auto_uses_cpu_when_gpu_evidence_is_missing():
    selection = select_backend("auto", grid_shape=(5, 7), report=_report())
    assert selection.backend.name == "numpy"
    assert selection.strategy == "fastest measured end-to-end (numpy)"


def test_forced_gpu_mode_reports_validation_failure():
    with pytest.raises(RuntimeError, match="Forced hip backend is not operational"):
        select_backend("hip", report=_report())
    with pytest.raises(RuntimeError, match="Forced torch backend is not operational"):
        select_backend("torch", report=_report())


def test_forced_gpu_mode_refuses_unverified_or_insufficient_memory():
    report = _report(hip=True, free_memory=1)
    with pytest.raises(RuntimeError, match="lacks verified free memory"):
        select_backend("hip", grid_shape=(33, 33), report=report)


def test_auto_chooses_fastest_validated_candidate(monkeypatch, tmp_path):
    from ehl_solver import hardware

    monkeypatch.setenv("EHL_BACKEND_CACHE_PATH", str(tmp_path / "cache.json"))
    monkeypatch.setattr(
        hardware,
        "_benchmark_numpy",
        lambda report, shape, repeats: Benchmark("numpy", 0, 8),
    )
    monkeypatch.setattr(
        hardware,
        "_benchmark_isolated_gpu",
        lambda report, name, shape, repeats: Benchmark(name, 1, 2)
        if name == "hip"
        else Benchmark(name, 1, 5),
    )
    monkeypatch.setattr(
        hardware,
        "make_backend",
        lambda name, device="auto": Backend(name, object(), device, name),
    )
    selection = hardware.select_backend(
        "auto", grid_shape=(5, 7), report=_report(hip=True, torch=True)
    )
    assert selection.backend.name == "hip"
    assert selection.strategy == "fastest measured end-to-end (hip)"


def test_memory_gate_excludes_gpu_that_cannot_fit(monkeypatch):
    from ehl_solver import hardware

    monkeypatch.setattr(
        hardware,
        "_benchmark_numpy",
        lambda report, shape, repeats: Benchmark("numpy", 0, 8),
    )
    monkeypatch.setattr(
        hardware,
        "_benchmark_isolated_gpu",
        lambda *args: pytest.fail("GPU should be excluded before benchmarking"),
    )
    report = _report(hip=True, free_memory=1)
    selection = select_backend("auto", grid_shape=(17, 13), report=report)
    assert selection.backend.name == "numpy"


def test_auto_restricts_to_explicit_device(monkeypatch):
    from ehl_solver import hardware

    monkeypatch.setattr(
        hardware,
        "_benchmark_numpy",
        lambda report, shape, repeats: Benchmark("numpy", 0, 8),
    )
    seen = []

    def benchmark(report, name, shape, repeats):
        seen.append(name)
        return Benchmark(name, 1, 2)

    monkeypatch.setattr(hardware, "_benchmark_isolated_gpu", benchmark)
    monkeypatch.setattr(
        hardware,
        "make_backend",
        lambda name, device="auto": Backend(name, object(), device, name),
    )
    selection = hardware.select_backend(
        "auto", "hip", grid_shape=(5, 7), report=_report(hip=True, torch=True)
    )
    assert seen == ["hip"]
    assert selection.backend.name == "hip"


def test_torch_cpu_is_explicit_and_does_not_fall_back_to_numpy(monkeypatch):
    from ehl_solver import hardware

    report = _report(torch=True)
    monkeypatch.setattr(
        hardware,
        "make_backend",
        lambda name, device="auto": Backend(name, object(), device, name),
    )
    selection = select_backend("torch", "cpu", report=report)
    assert selection.backend.name == "torch"
    assert selection.backend.device == "cpu"


def test_torch_auto_uses_validated_torch_cpu_if_gpu_unavailable(monkeypatch):
    from ehl_solver import hardware

    status = BackendStatus(
        name="torch",
        installed=True,
        operational=False,
        reason="No CUDA/ROCm GPU detected",
        cpu_operational=True,
    )
    report = _report()
    report = replace(report, backends={**report.backends, "torch": status})
    monkeypatch.setattr(
        hardware,
        "make_backend",
        lambda name, device="auto": Backend(name, object(), device, name),
    )
    selection = select_backend("torch", "auto", report=report)
    assert selection.backend.name == "torch"
    assert selection.backend.device == "cpu"


def test_numpy_probe_reports_float64_fft_capability():
    result = subprocess.run(
        [sys.executable, "-m", "ehl_solver._backend_probe", "--backend", "numpy"],
        check=True,
        capture_output=True,
        text=True,
    )
    probe = json.loads(result.stdout)
    assert probe["operational"]
    assert probe["float64"]
    assert probe["fft"]


def test_cached_benchmark_reuses_hardware_and_workload_result(monkeypatch, tmp_path):
    from ehl_solver import hardware

    cache_path = tmp_path / "benchmarks.json"
    monkeypatch.setenv("EHL_BACKEND_CACHE_PATH", str(cache_path))
    calls = []

    class Result:
        returncode = 0
        stdout = json.dumps(
            {"operational": True, "device_time_ms": 1.25, "end_to_end_ms": 3.5}
        )
        stderr = ""

    monkeypatch.setattr(
        hardware.subprocess,
        "run",
        lambda *args, **kwargs: calls.append(args) or Result(),
    )
    report = _report(hip=True)
    fresh = hardware._benchmark_isolated_gpu(report, "hip", (5, 7), 3)
    cached = hardware._benchmark_isolated_gpu(
        replace(report, backends={
            **report.backends,
            "hip": replace(report.backends["hip"], free_memory_bytes=1234),
        }),
        "hip",
        (5, 7),
        3,
    )
    assert fresh is not None and not fresh.cached
    assert cached is not None and cached.cached
    assert len(calls) == 1
    assert cache_path.exists()


def test_numpy_benchmark_is_cached_by_workload(monkeypatch, tmp_path):
    from ehl_solver import hardware

    cache_path = tmp_path / "benchmarks.json"
    monkeypatch.setenv("EHL_BACKEND_CACHE_PATH", str(cache_path))
    calls = []
    monkeypatch.setattr(
        hardware,
        "elastic_deformation",
        lambda *args: calls.append(args) or object(),
    )
    report = _report()
    fresh = hardware._benchmark_numpy(report, (5, 7), 1)
    cached = hardware._benchmark_numpy(report, (5, 7), 1)
    assert fresh.backend == cached.backend == "numpy"
    assert not fresh.cached
    assert cached.cached
    assert len(calls) == 2


def test_probe_reports_crash_without_propagating(monkeypatch):
    from ehl_solver import hardware

    class Result:
        returncode = -4
        stdout = ""
        stderr = ""

    monkeypatch.setattr(hardware.subprocess, "run", lambda *args, **kwargs: Result())
    monkeypatch.setattr(hardware, "_package_installed", lambda backend: True)
    result = hardware._probe_process("torch")
    assert result["installed"]
    assert not result["operational"]
    assert "signal 4" in result["reason"]


def test_detect_hardware_keeps_optional_runtime_probes_isolated():
    from ehl_solver.hardware import detect_hardware

    report = detect_hardware()
    assert report.backends["numpy"].operational
    assert report.backends["numpy"].cpu_operational
    assert report.backends["numpy"].float64
    assert report.backends["numpy"].fft
    assert set(report.backends) == {"numpy", "hip", "torch"}


def test_real_hip_validation_when_extension_is_built():
    import importlib.util

    if importlib.util.find_spec("ehl_solver._hip_native") is None:
        pytest.skip("Native HIP extension is not built")
    from ehl_solver.hardware import detect_hardware

    status = detect_hardware().backends["hip"]
    if not status.operational:
        pytest.skip(f"HIP backend not available: {status.reason}")
    assert status.float64
    assert status.fft
    assert status.architecture
    assert status.free_memory_bytes


def test_invalid_selection_mode_and_device():
    with pytest.raises(ValueError, match="mode"):
        select_backend("invalid", report=_report())
    with pytest.raises(ValueError, match="device"):
        select_backend("auto", "invalid", report=_report())
    with pytest.raises(ValueError, match="torch mode"):
        select_backend("torch", "hip", report=_report())
