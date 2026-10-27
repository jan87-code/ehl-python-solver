"""Backend selection; PyTorch's 'cuda' device also denotes AMD ROCm."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Backend:
    name: str
    xp: object
    device: str
    platform: str
    native: object | None = None


def make_backend(name: str = "numpy", device: str = "auto") -> Backend:
    if name == "numpy":
        if device not in ("auto", "cpu"):
            raise ValueError("NumPy backend supports CPU only")
        return Backend("numpy", np, "cpu", "CPU")
    if name == "hip":
        if device not in ("auto", "hip"):
            raise ValueError("HIP backend supports the HIP device only")
        try:
            from ehl_solver import _hip_native
        except ImportError as exc:
            raise RuntimeError(
                "The native HIP extension is not built; see the README HIP build procedure"
            ) from exc
        device_count = _hip_native.device_count()
        if device_count < 1:
            raise RuntimeError("The HIP runtime reports no available GPU devices")
        return Backend("hip", np, "hip:0", "AMD HIP/ROCm", _hip_native)
    if name != "torch":
        raise ValueError(f"Unknown backend {name!r}")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is not installed; see README GPU setup") from exc
    selected = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if selected == "auto":
        selected = "cpu"
    if selected == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("PyTorch sees no CUDA/ROCm GPU. Check GPU, drivers and wheel.")
    if selected not in ("cpu", "cuda"):
        raise ValueError("Device must be 'cpu', 'cuda', or 'auto'")
    platform = "AMD ROCm" if torch.version.hip else "NVIDIA CUDA" if torch.version.cuda else "CPU"
    if selected == "cpu":
        platform = "CPU (PyTorch)"
    return Backend("torch", torch, selected, platform)


def as_float64(array, backend: Backend):
    if backend.name in ("numpy", "hip"):
        return np.asarray(array, dtype=np.float64)
    return backend.xp.as_tensor(array, dtype=backend.xp.float64, device=backend.device)
