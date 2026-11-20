"""Reproduce the finite-roller dry-contact grid, domain and tilt study."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d

from ehl_solver.backend import make_backend
from ehl_solver.dry_contact import solve_finite_roller_dry_contact
from ehl_solver.elasticity import cell_integrated_kernel
from ehl_solver.geometry import contact_geometry_from_config

LENGTH_M = 1e-3
LOAD_N = 100.0
E_STAR_PA = 113e9
RADIUS_X_M = 0.01
CROWN_RADIUS_M = 1.0
TILT_RAD = 3e-3


def _solve(nodes_x, nodes_y, half_width_m, tilt, *, backend, cylindrical=False):
    x = np.linspace(-half_width_m, half_width_m, nodes_x)
    y = (np.arange(nodes_y) - (nodes_y - 1) / 2.0) * LENGTH_M / nodes_y
    profile = (
        {"type": "cylindrical"}
        if cylindrical
        else {"type": "parabolic", "radius_m": CROWN_RADIUS_M}
    )
    geometry = contact_geometry_from_config(
        {
            "type": "profiled_roller",
            "radius_x_m": RADIUS_X_M,
            "axial_profile": profile,
            "axial_tilt_rad": tilt,
        }
    )
    start = time.perf_counter()
    result = solve_finite_roller_dry_contact(
        x, y, geometry, LENGTH_M, LOAD_N, E_STAR_PA, backend=backend
    )
    elapsed = time.perf_counter() - start
    if backend.name == "torch":
        pressure = result.pressure_pa.detach().cpu().numpy()
        displacement = result.elastic_displacement_m.detach().cpu().numpy()
        gap = result.gap_m.detach().cpu().numpy()
    else:
        pressure = result.pressure_pa
        displacement = result.elastic_displacement_m
        gap = result.gap_m
    dx, dy = x[1] - x[0], y[1] - y[0]
    kernel = cell_integrated_kernel(pressure.shape, dx, dy, E_STAR_PA)
    direct = convolve2d(pressure, kernel, mode="same")
    geometry_field = geometry.h_geom(x, y)
    direct_gap = geometry_field + direct - result.approach_m
    active = pressure > pressure.max() * 1e-8
    edge = np.abs(y) >= 0.4 * LENGTH_M
    rows, cols = np.nonzero(active)
    area = dx * dy
    metrics = {
        "nodes_x": nodes_x,
        "nodes_y": nodes_y,
        "x_half_domain_m": half_width_m,
        "roller_length_m": LENGTH_M,
        "tilt_rad": tilt,
        "profile": profile,
        "backend": f"{backend.name}:{backend.device}",
        "elapsed_seconds": elapsed,
        "iterations": result.iterations,
        "force_n": float(np.sum(pressure) * area),
        "force_relative_error": float(np.sum(pressure) * area / LOAD_N - 1.0),
        "moment_x_n_m": float(-np.sum(pressure * y[:, None]) * area),
        "moment_y_n_m": float(np.sum(pressure * x[None, :]) * area),
        "pressure_min_pa": float(pressure.min()),
        "pressure_peak_pa": float(pressure.max()),
        "approach_m": result.approach_m,
        "kkt_residual": result.relative_kkt_residual,
        "direct_elasticity_relative_max_error": float(
            np.max(np.abs(direct - displacement)) / np.max(np.abs(direct))
        ),
        "direct_gap_max_error_m": float(np.max(np.abs(direct_gap - gap))),
        "direct_gap_min_m": float(direct_gap.min()),
        "direct_active_gap_max_abs_m": float(np.max(np.abs(direct_gap[active]))),
        "direct_complementarity_max_pa_m": float(np.max(np.abs(pressure * direct_gap))),
        "x_symmetry_relative_l2": float(
            np.linalg.norm(pressure - pressure[:, ::-1]) / np.linalg.norm(pressure)
        ),
        "y_symmetry_relative_l2": float(
            np.linalg.norm(pressure - pressure[::-1, :]) / np.linalg.norm(pressure)
        ),
        "footprint_threshold_relative_peak": 1e-8,
        "footprint_x_min_m": float(x[cols].min()),
        "footprint_x_max_m": float(x[cols].max()),
        "footprint_y_min_m": float(y[rows].min()),
        "footprint_y_max_m": float(y[rows].max()),
        "footprint_area_m2": float(active.sum() * area),
        "negative_edge_load_fraction": float(pressure[y <= -0.4 * LENGTH_M].sum() / pressure.sum()),
        "positive_edge_load_fraction": float(pressure[y >= 0.4 * LENGTH_M].sum() / pressure.sum()),
        "combined_edge_load_fraction": float(pressure[edge].sum() / pressure.sum()),
        "pressure_centroid_y_m": float(np.sum(pressure * y[:, None]) / np.sum(pressure)),
    }
    return {"x": x, "y": y, "pressure": pressure, "geometry": geometry_field, "metrics": metrics}


def _plot(path, result):
    x, y, p = result["x"], result["y"], result["pressure"]
    m = result["metrics"]
    x_mm, y_mm = x * 1e3, y * 1e3
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    image = axes[0, 0].pcolormesh(x_mm, y_mm, p * 1e-6, shading="nearest")
    fig.colorbar(image, ax=axes[0, 0], label="Pressure [MPa]")
    axes[0, 0].set(xlabel="x [mm]", ylabel="y [mm]", title="Dry-contact pressure")
    axes[0, 1].plot(x_mm, p[len(y) // 2, :] * 1e-6)
    axes[0, 1].set(xlabel="x [mm]", ylabel="Pressure [MPa]", title="Rolling centreline")
    axes[0, 2].plot(y_mm, p[:, len(x) // 2] * 1e-6)
    axes[0, 2].set(xlabel="y [mm]", ylabel="Pressure [MPa]", title="Axial centreline pressure")
    axes[1, 0].plot(y_mm, result["geometry"][:, len(x) // 2] * 1e9)
    axes[1, 0].set(xlabel="y [mm]", ylabel="Gap [nm]", title="Undeformed centreline gap")
    axes[1, 1].pcolormesh(x_mm, y_mm, p > p.max() * 1e-8, shading="nearest")
    axes[1, 1].axhline(-0.4, color="red", linestyle="--")
    axes[1, 1].axhline(0.4, color="red", linestyle="--")
    axes[1, 1].set(xlabel="x [mm]", ylabel="y [mm]", title="Footprint and outer 10% bands")
    axes[1, 2].axis("off")
    axes[1, 2].text(
        0.0, 1.0,
        f"Load: {m['force_n']:.6g} N\n"
        f"Mx: {m['moment_x_n_m']:.6g} N m\n"
        f"My: {m['moment_y_n_m']:.6g} N m\n"
        f"Approach: {m['approach_m'] * 1e9:.4g} nm\n"
        f"Peak: {m['pressure_peak_pa'] * 1e-6:.4g} MPa\n"
        f"Edge load, -y / +y: {m['negative_edge_load_fraction']:.3%} / "
        f"{m['positive_edge_load_fraction']:.3%}\n"
        f"Tilt: {m['tilt_rad']:.3g} rad\n"
        f"Grid: {m['nodes_y']} x {m['nodes_x']}",
        va="top",
    )
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("cuda", "numpy"), default="cuda")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    backend = make_backend("torch", "cuda") if args.backend == "cuda" else make_backend()
    definitions = {
        "crown_17": (17, 17, 4e-4, 0.0, False),
        "crown_33": (33, 33, 4e-4, 0.0, False),
        "crown_65": (65, 65, 4e-4, 0.0, False),
        "crown_wide": (81, 65, 5e-4, 0.0, False),
        "cylinder_65": (65, 65, 4e-4, 0.0, True),
        "tilt_positive": (65, 65, 4e-4, TILT_RAD, False),
        "tilt_negative": (65, 65, 4e-4, -TILT_RAD, False),
    }
    results = {}
    for name, (nx, ny, width, tilt, cylinder) in definitions.items():
        print(f"Solving {name}", flush=True)
        results[name] = _solve(nx, ny, width, tilt, backend=backend, cylindrical=cylinder)
    print("Solving independent NumPy reference", flush=True)
    cpu = _solve(65, 65, 4e-4, TILT_RAD, backend=make_backend())
    positive = results["tilt_positive"]
    negative = results["tilt_negative"]
    p = positive["pressure"]
    cpu_p = cpu["pressure"]
    wide = results["crown_wide"]["pressure"][:, 8:-8]
    regular = results["crown_65"]["pressure"]
    summary = {
        "cases": {name: result["metrics"] for name, result in results.items()},
        "cpu_reference": cpu["metrics"],
        "cuda_cpu_pressure_relative_l2": float(np.linalg.norm(p - cpu_p) / np.linalg.norm(cpu_p)),
        "tilt_reversal_pressure_relative_l2": float(
            np.linalg.norm(p - negative["pressure"][::-1, :]) / np.linalg.norm(p)
        ),
        "wider_domain_pressure_relative_l2": float(
            np.linalg.norm(wide - regular) / np.linalg.norm(regular)
        ),
        "note": (
            "Outer axial cell edges are the physical roller ends; the elastic "
            "operator is a half-space, not finite-body elasticity. Edge bands "
            "are |y| >= 0.4 roller lengths. CPU direct convolution audits "
            "the saved CUDA fields."
        ),
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    for name, result in results.items():
        np.savez_compressed(
            args.output_dir / f"{name}.npz",
            x_m=result["x"],
            y_m=result["y"],
            pressure_pa=result["pressure"],
            geometry_m=result["geometry"],
        )
    for name in ("crown_65", "cylinder_65", "tilt_positive", "tilt_negative"):
        _plot(args.output_dir / f"{name}.png", results[name])
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
