"""Linear convolution (not circular), centered 'same' extraction.

`kernel` is a general discrete impulse response, not yet a physically validated
half-space elasticity influence coefficient kernel. Kernel center uses floor(size/2).
"""
import numpy as np

from ehl_solver.backend import Backend, as_float64


def fft_convolve_same(signal, kernel, backend: Backend):
    signal = as_float64(signal, backend)
    kernel = as_float64(kernel, backend)
    if signal.ndim != 2 or kernel.ndim != 2:
        raise ValueError("Expected 2-D signal and kernel")
    if backend.name == "hip":
        signal = np.ascontiguousarray(signal, dtype=np.float64)
        kernel = np.ascontiguousarray(kernel, dtype=np.float64)
        return backend.native.convolve_same(signal, kernel)
    ny, nx = signal.shape
    ky, kx = kernel.shape
    full_shape = (ny + ky - 1, nx + kx - 1)
    if backend.name == "numpy":
        conv = np.fft.irfft2(
            np.fft.rfft2(signal, s=full_shape) * np.fft.rfft2(kernel, s=full_shape),
            s=full_shape,
        )
    else:
        torch = backend.xp
        conv = torch.fft.irfft2(
            torch.fft.rfft2(signal, s=full_shape) * torch.fft.rfft2(kernel, s=full_shape),
            s=full_shape,
        )
    sy, sx = ky // 2, kx // 2
    return conv[sy : sy + ny, sx : sx + nx]
