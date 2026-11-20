"""Reproduce the matched finite-roller dry-contact end-loading study."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ehl_solver.backend import make_backend
from ehl_solver.dry_contact import solve_finite_roller_dry_contact
from ehl_solver.elasticity import ElasticityOperator, cell_integrated_kernel
from ehl_solver.geometry import LogarithmicAxialProfile, contact_geometry_from_config

LENGTH = 1e-3
HALF = LENGTH / 2
LOAD = 100.0
MODULUS = 113e9
RELIEF = 125e-9
LOG_SCALE = RELIEF / -np.log(1.0 - 0.9**2)
GRIDS = ((33, 65), (49, 129), (65, 257))
TILTS = (0.0, 0.003, -0.003)


def profiles():
    coordinates = np.unique(
        np.concatenate((np.linspace(-HALF, HALF, 1001), [-0.9 * HALF, 0.9 * HALF]))
    )
    hard = LogarithmicAxialProfile(HALF, LOG_SCALE, RELIEF)
    common = {"half_length_m": HALF, "scale_m": LOG_SCALE, "maximum_relief_m": RELIEF}
    circular_radius = (HALF**2 + RELIEF**2) / (2 * RELIEF)
    return {
        "cylindrical": {"type": "cylindrical"},
        "parabolic": {"type": "parabolic", "radius_m": HALF**2 / (2 * RELIEF)},
        "circular": {"type": "circular", "radius_m": circular_radius},
        "log_hard": {"type": "logarithmic", **common},
        "log_smooth": {"type": "smooth_logarithmic", **common, "blend_start_fraction": 0.8},
        "log_tabulated": {
            "type": "tabulated",
            "y_m": coordinates.tolist(),
            "height_m": hard(coordinates).tolist(),
        },
    }


def band_loads(p, dx, dy, y, fraction):
    """Integrate fixed physical bands using partial cell overlap."""
    lo, hi = y - dy / 2, y + dy / 2
    widths_negative = np.maximum(0.0, np.minimum(hi, -HALF + fraction * LENGTH) - lo)
    widths_positive = np.maximum(0.0, hi - np.maximum(lo, HALF - fraction * LENGTH))
    line = p.sum(axis=1) * dx
    return [float(line @ widths_negative), float(line @ widths_positive)]


def feature_cells(p, y, fraction=0.1):
    """Count contiguous half-peak axial line-load cells within each end band."""
    line = p.sum(axis=1)
    counts = []
    for sign in (-1, 1):
        indices = np.flatnonzero(sign * y >= HALF - fraction * LENGTH)
        if not len(indices) or np.max(line[indices]) == 0:
            counts.append(0)
            continue
        peak = int(indices[np.argmax(line[indices])])
        selected = line >= line[peak] / 2
        left = right = peak
        while left > indices[0] and selected[left - 1]:
            left -= 1
        while right < indices[-1] and selected[right + 1]:
            right += 1
        counts.append(right - left + 1)
    return counts


def calculate(name, profile, nx, ny, tilt, backend, *, x_half_width=4e-4):
    x = np.linspace(-x_half_width, x_half_width, nx)
    y = (np.arange(ny) - (ny - 1) / 2) * LENGTH / ny
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
    geometry = contact_geometry_from_config({
        "type": "profiled_roller", "radius_x_m": 0.01,
        "axial_profile": profile, "axial_tilt_rad": tilt,
    })
    start = time.perf_counter()
    result = solve_finite_roller_dry_contact(
        x, y, geometry, LENGTH, LOAD, MODULUS, backend=backend
    )
    if backend.name == "torch":
        backend.xp.cuda.synchronize()
        p = result.pressure_pa.detach().cpu().numpy()
        displacement = result.elastic_displacement_m.detach().cpu().numpy()
    else:
        p, displacement = result.pressure_pa, result.elastic_displacement_m
    elapsed = time.perf_counter() - start
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(displacement)) or np.min(p) < 0:
        raise RuntimeError(f"non-finite or negative result for {name}, {nx}x{ny}, {tilt}")

    cpu = ElasticityOperator((ny, nx), dx, dy, MODULUS)(p)
    kernel = cell_integrated_kernel((ny, nx), dx, dy, MODULUS)
    rows, cols = np.indices(p.shape)
    peak_y, peak_x = np.unravel_index(np.argmax(p), p.shape)
    audit_points = [(0, 0), (0, nx // 2), (ny // 2, nx // 2), (peak_y, peak_x), (ny - 1, nx - 1)]
    direct_error = max(
        abs(np.sum(p * kernel[row - rows + ny - 1, col - cols + nx - 1]) - cpu[row, col])
        for row, col in audit_points
    ) / np.max(np.abs(cpu))
    gap = geometry.h_geom(x, y) + cpu - result.approach_m
    active = p > LOAD * 1e-12 / p.size / (dx * dy)
    scale = max(np.max(np.abs(geometry.h_geom(x, y))), abs(result.approach_m))
    kkt = max(np.max(np.abs(gap[active])), np.max(np.maximum(-gap[~active], 0), initial=0)) / scale
    if (
        not np.isfinite(kkt)
        or not np.isfinite(direct_error)
        or abs(p.sum() * dx * dy / LOAD - 1) > 1e-12
        or kkt > 2e-6
        or direct_error > 1e-12
    ):
        raise RuntimeError(f"independent acceptance audit failed for {name}, {nx}x{ny}, {tilt}")
    mask = p > p.max() * 1e-8
    yy, xx = np.nonzero(mask)
    bands = {str(percent): band_loads(p, dx, dy, y, percent / 100) for percent in (10, 5, 2)}
    data = {
        "profile": name, "nx": nx, "ny": ny, "tilt_rad": tilt,
        "backend": f"{backend.name}:{backend.device}", "seconds": elapsed,
        "iterations": result.iterations, "load_n": float(p.sum() * dx * dy),
        "approach_m": result.approach_m, "peak_pa": float(p[peak_y, peak_x]),
        "peak_x_m": float(x[peak_x]), "peak_y_m": float(y[peak_y]),
        "moment_x_n_m": float(-np.sum(p * y[:, None]) * dx * dy),
        "centroid_y_m": float(np.sum(p * y[:, None]) / p.sum()),
        "footprint_area_m2": float(mask.sum() * dx * dy),
        "footprint_x_m": [float(x[xx].min()), float(x[xx].max())],
        "footprint_y_m": [float(y[yy].min()), float(y[yy].max())],
        "end_band_load_n": bands, "end_half_peak_cells_10pct": feature_cells(p, y),
        "outermost_row_load_n": [float(p[0].sum() * dx * dy), float(p[-1].sum() * dx * dy)],
        "audit_kkt": float(kkt), "audit_min_gap_m": float(gap.min()),
        "cpu_fft_relative_max": float(np.max(np.abs(cpu - displacement)) / np.max(np.abs(cpu))),
        "direct_sample_relative_max": float(direct_error),
        "pressure_min_pa": float(p.min()),
        "x_symmetry_relative_l2": float(np.linalg.norm(p - p[:, ::-1]) / np.linalg.norm(p)),
    }
    return x, y, p, data


def make_plots(output, data):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for name, result in data.items():
        x, y, p, metrics = result
        axes[0, 0].plot(y * 1e3, p[:, len(x) // 2] * 1e-6, label=name)
        axes[0, 1].plot(x * 1e3, p[len(y) // 2, :] * 1e-6, label=name)
        axes[1, 0].plot(y * 1e3, p.sum(axis=1) * (x[1] - x[0]) / 1e3, label=name)
        axes[1, 1].scatter(metrics["peak_y_m"] * 1e3, metrics["peak_pa"] * 1e-6, label=name)
    axes[0, 0].set(xlabel="y [mm]", ylabel="Pressure [MPa]", title="Axial centreline")
    axes[0, 1].set(xlabel="x [mm]", ylabel="Pressure [MPa]", title="Rolling centreline")
    axes[1, 0].set(xlabel="y [mm]", ylabel="Line load [kN/m]", title="Integrated rolling pressure")
    axes[1, 1].set(xlabel="Peak y [mm]", ylabel="Peak pressure [MPa]", title="Nodal peaks")
    axes[0, 0].legend(fontsize=7)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def make_case_plot(output, result, profile):
    x, y, p, metrics = result
    geometry = contact_geometry_from_config({
        "type": "profiled_roller", "radius_x_m": 0.01,
        "axial_profile": profile, "axial_tilt_rad": metrics["tilt_rad"],
    })
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    image = axes[0, 0].pcolormesh(x * 1e3, y * 1e3, p * 1e-6, shading="nearest")
    fig.colorbar(image, ax=axes[0, 0], label="Pressure [MPa]")
    axes[0, 0].set(xlabel="x [mm]", ylabel="y [mm]", title="Dry-contact pressure")
    axes[0, 1].plot(x * 1e3, p[len(y) // 2, :] * 1e-6)
    axes[0, 1].set(xlabel="x [mm]", ylabel="Pressure [MPa]", title="Rolling centreline")
    axes[0, 2].plot(y * 1e3, p[:, len(x) // 2] * 1e-6)
    axes[0, 2].set(xlabel="y [mm]", ylabel="Pressure [MPa]", title="Axial centreline")
    axes[1, 0].plot(y * 1e3, geometry.h_geom(x, y)[:, len(x) // 2] * 1e9)
    axes[1, 0].set(xlabel="y [mm]", ylabel="Gap [nm]", title="Undeformed centreline gap")
    axes[1, 1].pcolormesh(x * 1e3, y * 1e3, p > p.max() * 1e-8, shading="nearest")
    for boundary in (-0.4, 0.4):
        axes[1, 1].axhline(boundary, color="red", linestyle="--")
    axes[1, 1].set(xlabel="x [mm]", ylabel="y [mm]", title="Footprint and 10% end bands")
    axes[1, 2].axis("off")
    negative, positive = metrics["end_band_load_n"]["10"]
    axes[1, 2].text(
        0, 1,
        f"{metrics['profile']}; tilt {metrics['tilt_rad']:+.3f} rad\n"
        f"Load: {metrics['load_n']:.6g} N\n"
        f"Mx: {metrics['moment_x_n_m']:.6g} N m\n"
        f"Approach: {metrics['approach_m'] * 1e9:.4g} nm\n"
        f"Peak: {metrics['peak_pa'] * 1e-6:.4g} MPa\n"
        f"End loads -y / +y: {negative:.3g} / {positive:.3g} N\n"
        f"Grid (ny,nx): {metrics['ny']} x {metrics['nx']}",
        va="top",
    )
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    backend = make_backend("torch", "cuda")
    definitions = profiles()
    (args.output_dir / "profiles.json").write_text(
        json.dumps(definitions, indent=2), encoding="utf-8"
    )
    metrics, fields = {}, {}
    for name, profile in definitions.items():
        for nx, ny in GRIDS:
            for tilt in TILTS:
                key = f"{name}_{nx}x{ny}_{tilt:+.3f}"
                print(f"Solving {key}", flush=True)
                result = calculate(name, profile, nx, ny, tilt, backend)
                x, y, p, case = result
                metrics[key], fields[key] = case, result
                np.savez_compressed(args.output_dir / f"{key}.npz", x_m=x, y_m=y, pressure_pa=p)
                (args.output_dir / "metrics.json").write_text(
                    json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
                )
    for tilt in TILTS:
        make_plots(
            args.output_dir / f"profiles_{tilt:+.3f}.png",
            {name: fields[f"{name}_65x257_{tilt:+.3f}"] for name in definitions},
        )
        for name, profile in definitions.items():
            make_case_plot(
                args.output_dir / f"case_{name}_{tilt:+.3f}.png",
                fields[f"{name}_65x257_{tilt:+.3f}"], profile,
            )
    print("Independent CPU parity case", flush=True)
    cpu = calculate("log_smooth", definitions["log_smooth"], 49, 129, 0.003, make_backend())
    gpu = fields["log_smooth_49x129_+0.003"]
    parity = {
        "cpu_seconds": cpu[3]["seconds"], "cuda_seconds": gpu[3]["seconds"],
        "pressure_relative_l2": float(np.linalg.norm(cpu[2] - gpu[2]) / np.linalg.norm(cpu[2])),
        "cpu_metrics": cpu[3],
    }
    (args.output_dir / "parity.json").write_text(json.dumps(parity, indent=2), encoding="utf-8")
    print(json.dumps(parity, indent=2), flush=True)
    wide = calculate(
        "log_smooth_wide", definitions["log_smooth"], 81, 257, 0.003, backend,
        x_half_width=5e-4,
    )
    regular = fields["log_smooth_65x257_+0.003"]
    domain = {
        "wide_metrics": wide[3],
        "pressure_relative_l2": float(
            np.linalg.norm(wide[2][:, 8:-8] - regular[2]) / np.linalg.norm(regular[2])
        ),
    }
    np.savez_compressed(
        args.output_dir / "log_smooth_wide.npz",
        x_m=wide[0], y_m=wide[1], pressure_pa=wide[2],
    )
    (args.output_dir / "domain.json").write_text(json.dumps(domain, indent=2), encoding="utf-8")
    print(json.dumps(domain, indent=2), flush=True)


if __name__ == "__main__":
    main()
