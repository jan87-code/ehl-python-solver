"""Smoke checks for analytical formulae and optional GPU convolution."""
import argparse

import numpy as np

from ehl_solver.backend import make_backend
from ehl_solver.convolution import fft_convolve_same
from ehl_solver.hertz import circular_point_contact


def main():
    p = argparse.ArgumentParser(description="EHL numerical foundation smoke test")
    p.add_argument("--backend", choices=["numpy", "torch", "hip"], default="numpy")
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "hip"], default="auto")
    args = p.parse_args()
    backend = make_backend(args.backend, args.device)
    signal = np.zeros((7, 9), dtype=np.float64)
    signal[3, 4] = 1.0
    kernel = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]], dtype=np.float64)
    result = fft_convolve_same(signal, kernel, backend)
    center = float(result[3, 4])
    assert abs(center - 4.0) < 1e-12
    h = circular_point_contact(5.0, 0.015, 113e9)
    print(f"Backend: {backend.name}; device: {backend.device}; platform: {backend.platform}")
    print(f"FFT impulse central result: {center:.12g} (expected 4)")
    print(f"Hertz dry contact: a={h.radius_m:.9g} m; p0={h.peak_pressure_pa:.9g} Pa")
    print("Foundation smoke check passed (this is NOT a complete EHL solution).")


if __name__ == "__main__":
    main()
