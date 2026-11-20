"""Screen smooth roller relief and tilt, then refine the first viable candidates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ehl_solver.backend import make_backend
from ehl_solver.geometry import SmoothLogarithmicAxialProfile
from examples.run_roller_end_loading import HALF, LENGTH, LOAD, LOG_SCALE, RELIEF, calculate

TILTS = (0.0, 0.0001, 0.00025, 0.0005, 0.001, 0.003)
RELIEFS_NM = (125, 250, 500, 1000, 1500, 2000)
ZONES = (0.1, 0.2, 0.3)
SCREEN = (49, 257)
REFINED = (49, 513)


def parabolic_blend(y, end_relief, start):
    """Parabola blended C1 to zero end slope over the specified end zone."""
    s = np.abs(np.asarray(y, dtype=np.float64)) / HALF
    if np.any(~np.isfinite(s)) or np.any(s > 1):
        raise ValueError("profile coordinates must be finite and within roller ends")
    if not np.isfinite(end_relief) or end_relief <= 0 or not 0 < start < 1:
        raise ValueError("invalid end relief or blend start")
    result = end_relief * s**2
    mask = s > start
    t = (s[mask] - start) / (1 - start)
    g0 = end_relief * start**2
    d0 = 2 * end_relief * start
    result[mask] = (
        (2 * t**3 - 3 * t**2 + 1) * g0
        + (t**3 - 2 * t**2 + t) * (1 - start) * d0
        + (-2 * t**3 + 3 * t**2) * end_relief
    )
    return result


def make_profile(kind, relief_nm, zone):
    if kind == "cylindrical":
        return {"type": "cylindrical"}
    relief = relief_nm * 1e-9
    start = 1 - 2 * zone
    if kind == "parabolic":
        return lambda y: parabolic_blend(y, relief, start)
    if kind == "logarithmic":
        return SmoothLogarithmicAxialProfile(
            HALF, LOG_SCALE * relief / RELIEF, relief, start
        )
    raise ValueError(f"unknown profile: {kind}")


def edge_metrics(p, y):
    """Report cell-edge distances and peak values in fixed end windows."""
    dy = float(y[1] - y[0])
    active = np.any(p > 0, axis=1)
    indices = np.flatnonzero(active)
    cells = [int(indices[0]), int(len(y) - 1 - indices[-1])]
    outer = [
        float(np.max(p[y - dy / 2 < -HALF + 0.1 * LENGTH], initial=0)),
        float(np.max(p[y + dy / 2 > HALF - 0.1 * LENGTH], initial=0)),
    ]
    return {
        "edge_distance_cells": cells,
        "edge_distance_m": [distance * dy for distance in cells],
        "end_window_peak_pa": outer,
        "edge_distance_pressure_threshold_pa": 0.0,
    }


def screen_gate(metrics):
    """Require five completely inactive end cells and the user load thresholds."""
    return (
        min(metrics["edge_distance_cells"]) >= 5
        and max(metrics["outermost_row_load_n"]) < 0.005 * LOAD
        and max(metrics["end_band_load_n"]["2"]) < 0.02 * LOAD
    )


def refinement_gate(coarse, fine):
    """Apply documented stability thresholds and reject growing end-window peaks."""
    if not (screen_gate(coarse) and screen_gate(fine)):
        return False
    if abs(fine["approach_m"] - coarse["approach_m"]) > 0.01 * abs(fine["approach_m"]):
        return False
    if abs(fine["moment_x_n_m"] - coarse["moment_x_n_m"]) > 0.001:
        return False
    if abs(fine["centroid_y_m"] - coarse["centroid_y_m"]) > 10e-6:
        return False
    for window in ("2", "5", "10"):
        if max(
            abs(a - b) for a, b in zip(
                coarse["end_band_load_n"][window], fine["end_band_load_n"][window], strict=True
            )
        ) > 1.0:
            return False
    for old, new in zip(coarse["end_window_peak_pa"], fine["end_window_peak_pa"], strict=True):
        if new > old * 1.01 + 1e6:
            return False
    return True


def systematic_end_growth(first, second, third):
    """Detect two consecutive increases in either fixed end-window peak."""
    return any(
        a < b < c for a, b, c in zip(
            first["end_window_peak_pa"], second["end_window_peak_pa"],
            third["end_window_peak_pa"], strict=True
        )
    )


def key(kind, relief, zone, tilt, nx, ny):
    return f"{kind}_r{relief}_z{int(round(zone * 100))}_a{tilt:.5f}_{nx}x{ny}"


def run_case(out, records, kind, relief, zone, tilt, grid, backend):
    nx, ny = grid
    name = key(kind, relief, zone, tilt, nx, ny)
    if name in records:
        return records[name]
    print(f"Solving {name}", flush=True)
    profile = make_profile(kind, relief, zone)
    x, y, p, metrics = calculate(name, profile, nx, ny, tilt, backend)
    metrics.update(edge_metrics(p, y))
    metrics.update({"kind": kind, "relief_nm": relief, "zone_fraction": zone})
    np.savez_compressed(out / f"{name}.npz", x_m=x, y_m=y, pressure_pa=p)
    records[name] = metrics
    (out / "metrics.json").write_text(
        json.dumps(records, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return metrics


def write_table(out, records):
    rows = []
    for name, m in records.items():
        comparable = sorted(
            (
                other for other in records.values()
                if all(
                    other[field] == m[field]
                    for field in ("kind", "relief_nm", "zone_fraction", "tilt_rad", "nx")
                )
            ),
            key=lambda other: other["ny"],
        )
        earlier = [other for other in comparable if other["ny"] < m["ny"]]
        later = [other for other in comparable if other["ny"] > m["ny"]]
        if min(m["edge_distance_cells"]) == 0:
            peak_status = "unresolved_boundary_contact"
        elif later:
            peak_status = (
                "unresolved_increase" if later[0]["peak_pa"] > m["peak_pa"]
                else "no_increase_observed"
            )
        elif earlier:
            peak_status = (
                "unresolved_increase" if m["peak_pa"] > earlier[-1]["peak_pa"]
                else "no_increase_observed"
            )
        else:
            peak_status = "refinement_not_assessed"
        row = {
            "case": name, "kind": m["kind"], "relief_nm": m["relief_nm"],
            "zone_fraction": m["zone_fraction"], "tilt_rad": m["tilt_rad"],
            "nx": m["nx"], "ny": m["ny"], "load_n": m["load_n"],
            "kkt": m["audit_kkt"], "approach_nm": m["approach_m"] * 1e9,
            "centroid_um": m["centroid_y_m"] * 1e6, "moment_x_n_m": m["moment_x_n_m"],
            "peak_mpa": m["peak_pa"] * 1e-6,
            "peak_x_um": m["peak_x_m"] * 1e6, "peak_y_um": m["peak_y_m"] * 1e6,
            "peak_status": peak_status,
            "cpu_fft_relative_max": m["cpu_fft_relative_max"],
            "direct_sample_relative_max": m["direct_sample_relative_max"],
            "footprint_y_min_um": m["footprint_y_m"][0] * 1e6,
            "footprint_y_max_um": m["footprint_y_m"][1] * 1e6,
        }
        for i, side in enumerate(("negative", "positive")):
            row[f"{side}_edge_cells"] = m["edge_distance_cells"][i]
            row[f"{side}_edge_um"] = m["edge_distance_m"][i] * 1e6
            row[f"{side}_outer_row_n"] = m["outermost_row_load_n"][i]
            row[f"{side}_window_peak_mpa"] = m["end_window_peak_pa"][i] * 1e-6
            for band in ("2", "5", "10"):
                row[f"{side}_band_{band}pct_n"] = m["end_band_load_n"][band][i]
        rows.append(row)
    if not rows:
        raise ValueError("no study cases to export")
    with (out / "cases.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def plot_screen(out, records):
    clearance = np.empty((len(RELIEFS_NM), len(TILTS)))
    for i, relief in enumerate(RELIEFS_NM):
        for j, tilt in enumerate(TILTS):
            clearance[i, j] = max(
                min(records[key(kind, relief, zone, tilt, *SCREEN)]["edge_distance_m"]) * 1e6
                for kind in ("parabolic", "logarithmic") for zone in ZONES
            )
    fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    image = axis.imshow(clearance, origin="lower", aspect="auto")
    axis.set_xticks(range(len(TILTS)), [f"{tilt * 1000:g}" for tilt in TILTS])
    axis.set_yticks(range(len(RELIEFS_NM)), RELIEFS_NM)
    axis.set(xlabel="Tilt [mrad]", ylabel="End relief [nm]",
             title="Best screened clearance of the nearer end")
    fig.colorbar(image, ax=axis, label="Distance [micrometres]")
    fig.savefig(out / "screen_clearance.png", dpi=160)
    plt.close(fig)


def diagnostic_refinements(out, records):
    """Check near-threshold and high-tilt failures without tuning beyond the matrix."""
    diagnostics = {}
    for tilt, relief in ((0.0, 1500), (0.0005, 1500), (0.001, 2000), (0.003, 2000)):
        options = [
            records[key(kind, relief, zone, tilt, *SCREEN)]
            for kind in ("parabolic", "logarithmic") for zone in ZONES
        ]
        best = min(options, key=lambda m: max(m["end_band_load_n"]["2"]))
        fine = run_case(
            out, records, best["kind"], relief, best["zone_fraction"], tilt,
            REFINED, make_backend("torch", "cuda"),
        )
        diagnostics[str(tilt)] = {
            "screen_case": key(best["kind"], relief, best["zone_fraction"], tilt, *SCREEN),
            "refined_case": key(best["kind"], relief, best["zone_fraction"], tilt, *REFINED),
            "passes": refinement_gate(best, fine),
        }
        if tilt in (0.001, 0.003):
            extra = []
            for candidate in options:
                refined = run_case(
                    out, records, candidate["kind"], relief, candidate["zone_fraction"], tilt,
                    REFINED, make_backend("torch", "cuda"),
                )
                extra.append({
                    "case": key(
                        candidate["kind"], relief, candidate["zone_fraction"], tilt, *REFINED
                    ),
                    "passes": refinement_gate(candidate, refined),
                })
            diagnostics[str(tilt)]["all_profiles"] = extra
    (out / "diagnostic_refinements.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8"
    )
    return diagnostics


def extended_checks(out, records, minima):
    """Check a third grid for each tilt and a fourth if peaks keep growing."""
    decisions = {}
    for tilt_text, item in minima.items():
        if item is None:
            decisions[tilt_text] = None
            continue
        tilt, relief = float(tilt_text), item["relief_nm"]
        choice = max(
            item["passing_profiles"],
            key=lambda profile: min(
                records[key(profile["kind"], relief, profile["zone_fraction"], tilt, *REFINED)]
                ["edge_distance_m"]
            ),
        )
        kind, zone = choice["kind"], choice["zone_fraction"]
        coarse = records[key(kind, relief, zone, tilt, *SCREEN)]
        fine = records[key(kind, relief, zone, tilt, *REFINED)]
        third = run_case(out, records, kind, relief, zone, tilt, (49, 1025),
                         make_backend("torch", "cuda"))
        growth = systematic_end_growth(coarse, fine, third)
        last = third
        fourth_key = None
        if growth:
            fourth = run_case(out, records, kind, relief, zone, tilt, (49, 2049),
                              make_backend("torch", "cuda"))
            growth = systematic_end_growth(fine, third, fourth)
            last = fourth
            fourth_key = key(kind, relief, zone, tilt, 49, 2049)
        passed = (
            refinement_gate(coarse, fine)
            and refinement_gate(fine, third)
            and (fourth_key is None or refinement_gate(third, last))
            and not growth
        )
        decisions[tilt_text] = {
            "relief_nm": relief, "profile": choice,
            "third_case": key(kind, relief, zone, tilt, 49, 1025),
            "fourth_case": fourth_key, "systematic_end_growth": growth, "passes": passed,
        }
    (out / "extended_checks.json").write_text(
        json.dumps(decisions, indent=2) + "\n", encoding="utf-8"
    )
    return decisions


def reference_checks(out, records, minima):
    passing = [(float(tilt), item) for tilt, item in minima.items() if item is not None]
    if not passing:
        tilt, relief, choice = 0.0, 2000, {"kind": "parabolic", "zone_fraction": 0.3}
        run_case(out, records, choice["kind"], relief, choice["zone_fraction"], tilt, REFINED,
                 make_backend("torch", "cuda"))
    else:
        tilt, item = max(passing, key=lambda pair: pair[0])
        relief = item["relief_nm"]
        choice = max(
            item["passing_profiles"],
            key=lambda profile: min(
                records[
                    key(profile["kind"], relief, profile["zone_fraction"], tilt, *REFINED)
                ]["edge_distance_m"]
            ),
        )
    kind, zone = choice["kind"], choice["zone_fraction"]
    profile = make_profile(kind, relief, zone)
    gpu_name = key(kind, relief, zone, tilt, *REFINED)
    with np.load(out / f"{gpu_name}.npz") as field:
        x, y, gpu_pressure = field["x_m"], field["y_m"], field["pressure_pa"]
    cpu = calculate("cpu_reference", profile, *REFINED, tilt, make_backend())
    wide_metrics = run_case(
        out, records, kind, relief, zone, tilt, (65, REFINED[1]),
        make_backend("torch", "cuda"),
    )
    extended = run_case(
        out, records, kind, relief, zone, tilt, (49, 1025),
        make_backend("torch", "cuda"),
    )
    wide_name = key(kind, relief, zone, tilt, 65, REFINED[1])
    with np.load(out / f"{wide_name}.npz") as field:
        wide_x, wide_p = field["x_m"], field["pressure_pa"]
    dy = float(y[1] - y[0])
    rolling = gpu_pressure.sum(axis=0) * dy
    wide_rolling = wide_p.sum(axis=0) * dy
    interpolated = np.interp(x, wide_x, wide_rolling)
    result = {
        "case": gpu_name, "cpu_seconds": cpu[3]["seconds"],
        "cuda_seconds": records[gpu_name]["seconds"],
        "cpu_cuda_pressure_relative_l2": float(
            np.linalg.norm(cpu[2] - gpu_pressure) / np.linalg.norm(cpu[2])
        ),
        "rolling_profile_49_vs_65_relative_l2": float(
            np.linalg.norm(rolling - interpolated) / np.linalg.norm(interpolated)
        ),
        "third_grid_gate": refinement_gate(records[gpu_name], extended),
        "third_grid_case": key(kind, relief, zone, tilt, 49, 1025),
        "rolling_65_metrics": wide_metrics,
        "cpu_metrics": cpu[3],
    }
    (out / "reference_checks.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    out = args.output_dir
    records = {}
    backend = make_backend("torch", "cuda")
    for tilt in TILTS:
        run_case(out, records, "cylindrical", 0, 0, tilt, SCREEN, backend)
        for relief in RELIEFS_NM:
            for kind in ("parabolic", "logarithmic"):
                for zone in ZONES:
                    run_case(out, records, kind, relief, zone, tilt, SCREEN, backend)

    minima = {}
    for tilt in TILTS:
        minima[str(tilt)] = None
        for relief in RELIEFS_NM:
            passed = []
            for kind in ("parabolic", "logarithmic"):
                for zone in ZONES:
                    coarse = records[key(kind, relief, zone, tilt, *SCREEN)]
                    if screen_gate(coarse):
                        fine = run_case(out, records, kind, relief, zone, tilt, REFINED, backend)
                        if refinement_gate(coarse, fine):
                            passed.append({"kind": kind, "zone_fraction": zone})
            if passed:
                minima[str(tilt)] = {"relief_nm": relief, "passing_profiles": passed}
                break
    (out / "two_grid_candidates.json").write_text(
        json.dumps(minima, indent=2) + "\n", encoding="utf-8"
    )
    print("Provisional two-grid candidates:", json.dumps(minima), flush=True)
    print("Diagnostic refinements:", json.dumps(diagnostic_refinements(out, records)), flush=True)
    decisions = extended_checks(out, records, minima)
    print("Extended checks:", json.dumps(decisions), flush=True)
    verified = {
        tilt: (
            {"relief_nm": item["relief_nm"], "passing_profiles": [item["profile"]]}
            if item is not None and item["passes"] else None
        )
        for tilt, item in decisions.items()
    }
    print("Focused reference:", json.dumps(reference_checks(out, records, verified)), flush=True)
    write_table(out, records)
    plot_screen(out, records)


if __name__ == "__main__":
    main()
