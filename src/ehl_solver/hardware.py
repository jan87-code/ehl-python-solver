"""Isolated hardware probes and conservative benchmark-based backend selection."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ehl_solver.backend import Backend, make_backend
from ehl_solver.elasticity import elastic_deformation

_PROBE_TIMEOUT_SECONDS = 90
_MEMORY_SAFETY_FACTOR = 1.5


@dataclass(frozen=True)
class BackendStatus:
    name: str
    installed: bool
    operational: bool
    device: str | None = None
    architecture: str | None = None
    reason: str | None = None
    float64: bool = False
    fft: bool = False
    free_memory_bytes: int | None = None
    total_memory_bytes: int | None = None
    version: str | None = None
    cpu_operational: bool = False


@dataclass(frozen=True)
class HardwareReport:
    system: str
    machine: str
    processor: str
    cpu_flags: tuple[str, ...]
    backends: dict[str, BackendStatus]


@dataclass(frozen=True)
class MemoryEstimate:
    grid_shape: tuple[int, int]
    kernel_shape: tuple[int, int]
    padded_shape: tuple[int, int]
    hip_bytes: int
    torch_bytes: int
    safety_factor: float


@dataclass(frozen=True)
class Benchmark:
    backend: str
    device_time_ms: float
    end_to_end_ms: float
    cached: bool = False


@dataclass(frozen=True)
class BackendSelection:
    backend: Backend
    mode: str
    strategy: str
    report: HardwareReport
    memory: MemoryEstimate
    benchmarks: tuple[Benchmark, ...] = ()


def _cpu_flags() -> tuple[str, ...]:
    flags: set[str] = set()
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith(("flags", "features")) and ":" in line:
                flags.update(line.split(":", 1)[1].split())
                break
    except OSError:
        pass
    return tuple(sorted(flags))


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith(("model name", "hardware")) and ":" in line:
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.uname().processor or "unknown"


def estimate_convolution_memory(
    grid_shape: tuple[int, int],
    kernel_shape: tuple[int, int] | None = None,
    safety_factor: float = _MEMORY_SAFETY_FACTOR,
) -> MemoryEstimate:
    """Estimate peak GPU allocation for the current 2-D full-convolution layout."""
    if len(grid_shape) != 2 or any(
        not isinstance(size, int) or isinstance(size, bool) or size < 1 for size in grid_shape
    ):
        raise ValueError("grid_shape must contain two positive integers (ny, nx)")
    if kernel_shape is None:
        kernel_shape = (2 * grid_shape[0] - 1, 2 * grid_shape[1] - 1)
    if len(kernel_shape) != 2 or any(
        not isinstance(size, int) or isinstance(size, bool) or size < 1 for size in kernel_shape
    ):
        raise ValueError("kernel_shape must contain two positive integers")
    if not np.isfinite(safety_factor) or safety_factor < 1:
        raise ValueError("safety_factor must be finite and at least 1")

    ny, nx = grid_shape
    ky, kx = kernel_shape
    py, px = ny + ky - 1, nx + kx - 1
    grid_cells, kernel_cells, padded_cells = ny * nx, ky * kx, py * px
    # Native HIP holds two complex128 padded arrays and the input, kernel and output.
    hip_raw = 32 * padded_cells + 8 * (2 * grid_cells + kernel_cells)
    # PyTorch real inputs, full-sized inverse result, two inputs plus product spectra.
    half_spectrum_cells = py * (px // 2 + 1)
    torch_raw = (
        8 * (grid_cells + kernel_cells + padded_cells)
        + 16 * 3 * half_spectrum_cells
    )
    return MemoryEstimate(
        grid_shape=grid_shape,
        kernel_shape=kernel_shape,
        padded_shape=(py, px),
        hip_bytes=int(np.ceil(hip_raw * safety_factor)),
        torch_bytes=int(np.ceil(torch_raw * safety_factor)),
        safety_factor=safety_factor,
    )


def _cache_path() -> Path:
    configured = os.environ.get("EHL_BACKEND_CACHE_PATH")
    if configured:
        return Path(configured).expanduser()
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "ehl-solver" / "backend-benchmarks.json"


def _fingerprint(report: HardwareReport) -> str:
    backends = {}
    for key, status in sorted(report.backends.items()):
        stable_status = asdict(status)
        stable_status.pop("free_memory_bytes", None)
        backends[key] = stable_status
    details = {
        "system": report.system,
        "machine": report.machine,
        "processor": report.processor,
        "cpu_flags": report.cpu_flags,
        "backends": backends,
        "python": sys.version,
        "numpy": np.__version__,
        "selection_schema": 1,
    }
    return hashlib.sha256(json.dumps(details, sort_keys=True).encode()).hexdigest()


def _read_cache() -> dict[str, Any]:
    try:
        cache = json.loads(_cache_path().read_text(encoding="utf-8"))
        return cache if isinstance(cache, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_cache(cache: dict[str, Any]) -> None:
    path = _cache_path()
    temporary_path: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as temporary:
            json.dump(cache, temporary, sort_keys=True)
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, path)
    except OSError:
        # Cache is an optimization; selection still proceeds with fresh measurements.
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def _probe_json(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise json.JSONDecodeError("no JSON object in subprocess output", stdout, 0)


def _package_installed(backend: str) -> bool:
    module = {"hip": "ehl_solver._hip_native", "torch": "torch", "numpy": "numpy"}[backend]
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _probe_process(backend: str, shape: tuple[int, int] | None = None) -> dict[str, Any]:
    command = [sys.executable, "-m", "ehl_solver._backend_probe", "--backend", backend]
    if shape is not None:
        command.extend(["--shape", str(shape[0]), str(shape[1])])
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "installed": _package_installed(backend),
            "operational": False,
            "reason": str(exc),
        }
    try:
        data = _probe_json(result.stdout)
    except json.JSONDecodeError:
        reason = result.stderr.strip() or f"probe exited with status {result.returncode}"
        if result.returncode < 0:
            reason = f"isolated probe terminated by signal {-result.returncode}"
        return {
            "installed": _package_installed(backend),
            "operational": False,
            "reason": reason,
        }
    data.setdefault("installed", _package_installed(backend))
    if result.returncode != 0:
        data["operational"] = False
        data["reason"] = data.get("reason") or result.stderr.strip()
    return data


def detect_hardware() -> HardwareReport:
    """Collect CPU facts and test optional runtimes in crash-isolated subprocesses."""
    cpu_probe = _probe_process("numpy")
    cpu = BackendStatus(
        name="numpy",
        installed=True,
        operational=bool(cpu_probe.get("operational", False)),
        device="cpu",
        architecture=platform.machine(),
        reason=cpu_probe.get("reason")
        or (
            "NumPy float64 FFT capability checked"
            if cpu_probe.get("operational", False)
            else "NumPy float64 FFT probe failed"
        ),
        float64=bool(cpu_probe.get("float64", False)),
        fft=bool(cpu_probe.get("fft", False)),
        version=cpu_probe.get("version"),
        cpu_operational=bool(cpu_probe.get("operational", False)),
    )
    statuses: dict[str, BackendStatus] = {"numpy": cpu}
    for name in ("hip", "torch"):
        info = _probe_process(name)
        statuses[name] = BackendStatus(
            name=name,
            installed=bool(info.get("installed", False)),
            operational=bool(info.get("operational", False)),
            device=info.get("device"),
            architecture=info.get("architecture"),
            reason=info.get("reason"),
            float64=bool(info.get("float64", False)),
            fft=bool(info.get("fft", False)),
            free_memory_bytes=info.get("free_memory_bytes"),
            total_memory_bytes=info.get("total_memory_bytes"),
            version=info.get("version"),
        )
    return HardwareReport(
        system=platform.system(),
        machine=platform.machine(),
        processor=_cpu_model(),
        cpu_flags=_cpu_flags(),
        backends=statuses,
    )


def _benchmark_key(
    report: HardwareReport, name: str, shape: tuple[int, int], repeats: int
) -> str:
    return json.dumps(
        {
            "hardware": _fingerprint(report),
            "backend": name,
            "shape": shape,
            "kernel_shape": (2 * shape[0] - 1, 2 * shape[1] - 1),
            "dtype": "float64/complex128",
            "operation": "cell-integrated-elastic-deformation",
            "grid_spacing_m": (2.0e-6, 2.3e-6),
            "effective_modulus_pa": 113.0e9,
            "repeats": repeats,
        },
        sort_keys=True,
    )


def _benchmark_numpy(
    report: HardwareReport, shape: tuple[int, int], repeats: int
) -> Benchmark:
    key = _benchmark_key(report, "numpy", shape, repeats)
    cached = _read_cache().get(key)
    if cached is not None:
        try:
            return Benchmark(
                "numpy",
                float(cached["device_time_ms"]),
                float(cached["end_to_end_ms"]),
                True,
            )
        except (KeyError, TypeError, ValueError):
            pass
    backend = make_backend("numpy")
    rng = np.random.default_rng(7001)
    pressure = rng.uniform(0.0, 2.0e8, shape)
    samples = []
    for _ in range(repeats + 1):
        start = time.perf_counter()
        elastic_deformation(pressure, 2.0e-6, 2.3e-6, 113.0e9, backend)
        duration = (time.perf_counter() - start) * 1000
        samples.append(duration)
    value = {
        "device_time_ms": 0.0,
        "end_to_end_ms": float(np.median(samples[1:])),
    }
    cache = _read_cache()
    cache[key] = value
    _write_cache(cache)
    return Benchmark("numpy", **value)


def _benchmark_isolated_gpu(
    report: HardwareReport,
    name: str,
    shape: tuple[int, int],
    repeats: int,
) -> Benchmark | None:
    key = _benchmark_key(report, name, shape, repeats)
    cache = _read_cache()
    cached = cache.get(key)
    if cached is not None:
        try:
            return Benchmark(
                name,
                float(cached["device_time_ms"]),
                float(cached["end_to_end_ms"]),
                True,
            )
        except (KeyError, TypeError, ValueError):
            pass
    command = [
        sys.executable,
        "-m",
        "ehl_solver._backend_probe",
        "--backend",
        name,
        "--benchmark",
        "--shape",
        str(shape[0]),
        str(shape[1]),
        "--repeats",
        str(repeats),
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_SECONDS,
        )
        data = _probe_json(result.stdout)
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return None
    if result.returncode != 0 or not data.get("operational"):
        return None
    value = {
        "device_time_ms": float(data["device_time_ms"]),
        "end_to_end_ms": float(data["end_to_end_ms"]),
    }
    cache[key] = value
    _write_cache(cache)
    return Benchmark(name, **value)


def select_backend(
    mode: str = "auto",
    device: str = "auto",
    grid_shape: tuple[int, int] = (129, 129),
    *,
    repeats: int = 3,
    report: HardwareReport | None = None,
) -> BackendSelection:
    """Choose a backend without allowing forced modes to fall back."""
    if mode not in ("auto", "cpu", "hip", "torch"):
        raise ValueError("mode must be one of 'auto', 'cpu', 'hip', or 'torch'")
    if device not in ("auto", "cpu", "hip", "cuda"):
        raise ValueError("device must be one of 'auto', 'cpu', 'hip', or 'cuda'")
    if repeats < 1:
        raise ValueError("repeats must be positive")
    memory = estimate_convolution_memory(grid_shape)
    if report is None:
        report = detect_hardware()

    if mode == "cpu":
        if device not in ("auto", "cpu"):
            raise ValueError("cpu mode only supports device='cpu'")
        if not report.backends["numpy"].operational:
            raise RuntimeError(
                "Forced NumPy CPU backend is not operational: "
                f"{report.backends['numpy'].reason or 'float64/FFT probe failed'}"
            )
        return BackendSelection(
            make_backend("numpy"), mode, "explicit CPU reference", report, memory
        )

    if mode in ("hip", "torch"):
        if mode == "torch" and device not in ("auto", "cpu", "cuda"):
            raise ValueError("torch mode supports device='auto', 'cpu', or 'cuda'")
        target = "hip" if mode == "hip" else device
        if target == "auto":
            torch_status = report.backends["torch"]
            estimated_bytes = memory.torch_bytes
            target = (
                "cuda"
                if torch_status.operational
                and torch_status.free_memory_bytes is not None
                and torch_status.free_memory_bytes >= estimated_bytes
                else "cpu"
                if mode == "torch" and torch_status.cpu_operational
                else "cuda"
            )
        if mode == "hip" and device not in ("auto", "hip"):
            raise ValueError("hip mode only supports device='hip'")
        if target == "cpu":
            if mode != "torch":
                raise ValueError("Only torch mode supports device='cpu'")
            status = report.backends["torch"]
            if not status.cpu_operational:
                raise RuntimeError(
                    f"Forced torch CPU backend is not operational: "
                    f"{status.reason or 'CPU capability probe failed'}"
                )
            backend = make_backend("torch", "cpu")
            return BackendSelection(
                backend, mode, "explicit PyTorch CPU", report, memory
            )
        name = "hip" if mode == "hip" else "torch"
        status = report.backends[name]
        if not status.operational or not status.float64 or not status.fft:
            raise RuntimeError(
                f"Forced {name} backend is not operational for float64 FFT: "
                f"{status.reason or 'probe failed'}"
            )
        requirement = memory.hip_bytes if name == "hip" else memory.torch_bytes
        if status.free_memory_bytes is None or status.free_memory_bytes < requirement:
            raise RuntimeError(
                f"Forced {name} backend lacks verified free memory "
                f"(requires {requirement} bytes with safety margin)"
            )
        backend = make_backend("hip") if name == "hip" else make_backend("torch", "cuda")
        return BackendSelection(
            backend, mode, f"explicit {name} GPU", report, memory
        )

    if device == "cpu":
        if not report.backends["numpy"].operational:
            raise RuntimeError(
                "NumPy CPU backend is not operational: "
                f"{report.backends['numpy'].reason or 'float64/FFT probe failed'}"
            )
        return BackendSelection(
            make_backend("numpy"), mode, "CPU explicitly requested", report, memory
        )

    candidates: list[Benchmark] = []
    if device in ("auto", "cpu") and report.backends["numpy"].operational:
        candidates.append(_benchmark_numpy(report, grid_shape, repeats))
    for name in ("hip", "torch"):
        if device not in ("auto", name, "cuda" if name == "torch" else name):
            continue
        status = report.backends[name]
        if (
            not status.operational
            or not status.float64
            or not status.fft
            or status.free_memory_bytes is None
        ):
            continue
        requirement = memory.hip_bytes if name == "hip" else memory.torch_bytes
        if status.free_memory_bytes < requirement:
            continue
        benchmark = _benchmark_isolated_gpu(report, name, grid_shape, repeats)
        if benchmark is not None:
            candidates.append(benchmark)
    if not candidates:
        if device not in ("auto", "cpu"):
            raise RuntimeError(
                f"Requested device {device!r} has no validated backend or usable benchmark"
            )
        if not report.backends["numpy"].operational:
            raise RuntimeError(
                "No validated backend is available and NumPy CPU validation failed: "
                f"{report.backends['numpy'].reason or 'unknown error'}"
            )
        return BackendSelection(
            make_backend("numpy"),
            mode,
            "CPU conservative default; no benchmark evidence",
            report,
            memory,
        )
    best = min(candidates, key=lambda item: item.end_to_end_ms)
    backend = (
        make_backend("numpy")
        if best.backend == "numpy"
        else make_backend("hip")
        if best.backend == "hip"
        else make_backend("torch", "cuda")
    )
    return BackendSelection(
        backend,
        mode,
        f"fastest measured end-to-end ({best.backend})",
        report,
        memory,
        tuple(candidates),
    )


def report_as_dict(report: HardwareReport) -> dict[str, Any]:
    return {
        "system": report.system,
        "machine": report.machine,
        "processor": report.processor,
        "cpu_flags": list(report.cpu_flags),
        "backends": {name: asdict(status) for name, status in report.backends.items()},
    }


def selection_as_dict(selection: BackendSelection) -> dict[str, Any]:
    return {
        "mode": selection.mode,
        "selected_backend": selection.backend.name,
        "selected_device": selection.backend.device,
        "strategy": selection.strategy,
        "memory_estimate": asdict(selection.memory),
        "benchmarks": [asdict(item) for item in selection.benchmarks],
        "hardware": report_as_dict(selection.report),
    }
