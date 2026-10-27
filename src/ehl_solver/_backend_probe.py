"""Child-process probes for optional runtimes that may crash during import/init."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np


def probe_numpy() -> dict:
    values = np.arange(15, dtype=np.float64).reshape(3, 5)
    spectrum = np.fft.rfft2(values)
    result = np.fft.irfft2(spectrum, s=values.shape)
    return {
        "installed": True,
        "operational": bool(np.allclose(result, values, rtol=1e-13, atol=1e-13)),
        "device": "cpu",
        "architecture": None,
        "float64": values.dtype == np.float64,
        "fft": spectrum.dtype == np.complex128,
        "version": np.__version__,
    }


def probe_hip(shape: tuple[int, int], benchmark: bool, repeats: int) -> dict:
    try:
        from ehl_solver import _hip_native
    except ImportError as exc:
        return {"installed": False, "operational": False, "reason": str(exc)}
    count = _hip_native.device_count()
    if count < 1:
        return {"installed": True, "operational": False, "reason": "No HIP devices detected"}
    info = _hip_native.device_info()
    pressure = np.zeros((5, 7), dtype=np.float64)
    pressure[1, 3] = 2.0
    kernel = np.arange(1, 13, dtype=np.float64).reshape(3, 4)
    result = _hip_native.convolve_same(pressure, kernel)
    # Independent direct summation verifies float64 linear convolution, not an FFT round trip.
    direct = np.zeros_like(pressure)
    for y in range(pressure.shape[0]):
        for x in range(pressure.shape[1]):
            for ky in range(kernel.shape[0]):
                for kx in range(kernel.shape[1]):
                    oy, ox = y + ky - kernel.shape[0] // 2, x + kx - kernel.shape[1] // 2
                    if 0 <= oy < pressure.shape[0] and 0 <= ox < pressure.shape[1]:
                        direct[oy, ox] += pressure[y, x] * kernel[ky, kx]
    valid = (
        result.dtype == np.float64
        and np.allclose(result, direct, rtol=2e-12, atol=2e-12)
    )
    response = {
        "installed": True,
        "operational": bool(valid),
        "reason": None if valid else "HIP float64 convolution disagrees with direct reference",
        "device": "hip:0",
        "architecture": info["architecture"],
        "device_name": info["name"],
        "float64": result.dtype == np.float64,
        "fft": bool(valid),
        "free_memory_bytes": info["free_memory_bytes"],
        "total_memory_bytes": info["total_memory_bytes"],
        "version": f"HIP {info['runtime_version']}; hipFFT {info['hipfft_version']}",
    }
    if not benchmark or not valid:
        return response

    from ehl_solver.backend import make_backend
    from ehl_solver.elasticity import cell_integrated_kernel, elastic_deformation

    ny, nx = shape
    pressure = np.random.default_rng(8102).uniform(0.0, 2.0e8, (ny, nx))
    backend = make_backend("hip")
    kernel = cell_integrated_kernel((ny, nx), 2.0e-6, 2.3e-6, 113.0e9, backend)
    elastic_deformation(pressure, 2.0e-6, 2.3e-6, 113.0e9, backend)
    device_times = []
    for _ in range(repeats):
        _, device_ms = _hip_native.convolve_same_timed(
            np.ascontiguousarray(pressure), np.ascontiguousarray(kernel)
        )
        device_times.append(float(device_ms))
    full_times = []
    for _ in range(repeats):
        start = time.perf_counter()
        elastic_deformation(pressure, 2.0e-6, 2.3e-6, 113.0e9, backend)
        full_times.append((time.perf_counter() - start) * 1000.0)
    response.update(
        device_time_ms=float(np.median(device_times)),
        end_to_end_ms=float(np.median(full_times)),
    )
    return response


def probe_torch(shape: tuple[int, int], benchmark: bool, repeats: int) -> dict:
    try:
        import torch
    except BaseException as exc:
        # This process is disposable: report import crashes, including SIGILL-like errors.
        return {"installed": False, "operational": False, "reason": f"{type(exc).__name__}: {exc}"}
    response = {
        "installed": True,
        "operational": False,
        "cpu_operational": False,
        "version": torch.__version__,
        "reason": "No CUDA/ROCm GPU detected",
    }
    try:
        cpu_values = torch.arange(35, dtype=torch.float64).reshape(5, 7)
        cpu_roundtrip = torch.fft.irfft2(torch.fft.rfft2(cpu_values), s=cpu_values.shape)
        response["cpu_operational"] = bool(
            cpu_roundtrip.dtype == torch.float64
            and torch.allclose(cpu_roundtrip, cpu_values, rtol=1e-12, atol=1e-12)
        )
    except BaseException as exc:
        response["reason"] = f"PyTorch CPU validation failed: {type(exc).__name__}: {exc}"
    if not torch.cuda.is_available():
        return response
    try:
        device = torch.device("cuda:0")
        properties = torch.cuda.get_device_properties(device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        from ehl_solver.backend import make_backend
        from ehl_solver.convolution import fft_convolve_same

        pressure = np.arange(35, dtype=np.float64).reshape(5, 7)
        kernel = np.arange(1, 13, dtype=np.float64).reshape(3, 4)
        source = torch.as_tensor(pressure, dtype=torch.float64, device=device)
        spectrum = torch.fft.rfft2(source)
        result = fft_convolve_same(pressure, kernel, make_backend("torch", "cuda"))
        torch.cuda.synchronize(device)
        direct = np.zeros_like(pressure)
        for y in range(pressure.shape[0]):
            for x in range(pressure.shape[1]):
                for ky in range(kernel.shape[0]):
                    for kx in range(kernel.shape[1]):
                        oy = y + ky - kernel.shape[0] // 2
                        ox = x + kx - kernel.shape[1] // 2
                        if 0 <= oy < pressure.shape[0] and 0 <= ox < pressure.shape[1]:
                            direct[oy, ox] += pressure[y, x] * kernel[ky, kx]
        valid = (
            result.dtype == torch.float64
            and spectrum.dtype == torch.complex128
            and np.allclose(result.cpu().numpy(), direct, rtol=2e-12, atol=2e-12)
        )
        response.update(
            device="cuda:0",
            architecture=getattr(properties, "gcnArchName", None)
            or getattr(properties, "name", None),
            free_memory_bytes=int(free_bytes),
            total_memory_bytes=int(total_bytes),
            float64=bool(valid),
            fft=bool(valid),
            operational=bool(valid),
            reason=None if valid else "PyTorch float64 FFT validation failed",
        )
        if not benchmark or not valid:
            return response

        from ehl_solver.elasticity import cell_integrated_kernel, elastic_deformation

        backend = make_backend("torch", "cuda")
        ny, nx = shape
        pressure = np.random.default_rng(8102).uniform(0.0, 2.0e8, (ny, nx))
        kernel = cell_integrated_kernel(
            (ny, nx), 2.0e-6, 2.3e-6, 113.0e9, backend
        )
        elastic_deformation(pressure, 2.0e-6, 2.3e-6, 113.0e9, backend)
        torch.cuda.synchronize(device)
        device_times = []
        for _ in range(repeats):
            start_event, stop_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            start_event.record()
            output = fft_convolve_same(pressure, kernel, backend)
            stop_event.record()
            torch.cuda.synchronize(device)
            output.cpu().numpy()
            device_times.append(float(start_event.elapsed_time(stop_event)))
        full_times = []
        for _ in range(repeats):
            start = time.perf_counter()
            output = elastic_deformation(
                pressure, 2.0e-6, 2.3e-6, 113.0e9, backend
            )
            torch.cuda.synchronize(device)
            output.cpu().numpy()
            full_times.append((time.perf_counter() - start) * 1000.0)
        response.update(
            device_time_ms=float(np.median(device_times)),
            end_to_end_ms=float(np.median(full_times)),
        )
        return response
    except BaseException as exc:
        response.update(
            operational=False,
            reason=f"GPU validation failed: {type(exc).__name__}: {exc}",
        )
        return response


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("numpy", "hip", "torch"), required=True)
    parser.add_argument("--shape", nargs=2, type=int, default=(129, 129))
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    try:
        if min(args.shape) < 1 or args.repeats < 1:
            raise ValueError("shape dimensions and repeats must be positive")
        if args.backend == "numpy":
            result = probe_numpy()
        elif args.backend == "hip":
            result = probe_hip(tuple(args.shape), args.benchmark, args.repeats)
        else:
            result = probe_torch(tuple(args.shape), args.benchmark, args.repeats)
    except BaseException as exc:
        result = {
            "installed": True,
            "operational": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
