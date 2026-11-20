"""Compare accepted Venner/ten Napel M=20 line-EHL grid results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if __package__ == "examples":
    from .run_line_ehl_physical_diagnostics import (
        VENNER_FIG5_FILM,
        VENNER_FIG5_FILM_X,
        VENNER_FIG5_PRESSURE,
        VENNER_FIG5_X,
        VENNER_FILM_UNCERTAINTY,
        VENNER_PRESSURE_UNCERTAINTY,
        VENNER_SPIKE_MINIMUM,
        VENNER_SPIKE_MINIMUM_X,
        VENNER_SPIKE_PEAK,
        VENNER_SPIKE_PEAK_X,
        VENNER_X_UNCERTAINTY,
        _venner_case,
    )
else:
    from run_line_ehl_physical_diagnostics import (
        VENNER_FIG5_FILM,
        VENNER_FIG5_FILM_X,
        VENNER_FIG5_PRESSURE,
        VENNER_FIG5_X,
        VENNER_FILM_UNCERTAINTY,
        VENNER_PRESSURE_UNCERTAINTY,
        VENNER_SPIKE_MINIMUM,
        VENNER_SPIKE_MINIMUM_X,
        VENNER_SPIKE_PEAK,
        VENNER_SPIKE_PEAK_X,
        VENNER_X_UNCERTAINTY,
        _venner_case,
    )

ACCEPTED_DIRECTORIES = {
    257: Path("results/line_ehl_m20_newton_257_finalized"),
    513: Path("results/line_ehl_m20_newton_513_refine_v3"),
    1025: Path("results/line_ehl_m20_newton_1025_refine_v1"),
    2049: Path("results/line_ehl_m20_newton_2049_refine_v1"),
    4097: Path("results/line_ehl_m20_newton_4097_refine_v1"),
}
REGIONS = {
    "full_domain": (-4.0, 1.5),
    "contact": (-2.0, 1.5),
    "bulk": (-0.5, 0.5),
    "outlet": (0.5, 1.5),
    "spike_window": (0.85, 0.95),
}
SPIKE_WINDOW = REGIONS["spike_window"]
FIGURE_6_PEAK_X_UNCERTAINTY = 0.0025
FIGURE_6_PEAK_PRESSURE_UNCERTAINTY = 0.001


def _profile_difference(
    coarse_x: np.ndarray,
    coarse_values: np.ndarray,
    fine_x: np.ndarray,
    fine_values: np.ndarray,
    bounds: tuple[float, float],
) -> dict[str, float | int]:
    mask = (
        (fine_x >= bounds[0])
        & (fine_x <= bounds[1])
        & (fine_x >= coarse_x[0])
        & (fine_x <= coarse_x[-1])
    )
    x = fine_x[mask]
    if x.size == 0:
        raise ValueError("profile comparison region contains no grid samples")
    fine = fine_values[mask]
    difference = fine - np.interp(x, coarse_x, coarse_values)
    denominator = np.linalg.norm(fine)
    return {
        "sample_count": int(x.size),
        "rms_difference": float(np.sqrt(np.mean(difference**2))),
        "relative_l2_difference": (
            float(np.linalg.norm(difference) / denominator)
            if denominator > 0.0
            else float("nan")
        ),
    }


def _reference_errors(
    source_x: np.ndarray,
    source_values: np.ndarray,
    grid_x: np.ndarray,
    grid_values: np.ndarray,
    scale: float,
    mask: np.ndarray | None = None,
) -> dict[str, float | int]:
    selected = np.isfinite(source_x) & np.isfinite(source_values)
    if mask is not None:
        selected &= mask
    difference = np.interp(
        source_x[selected], grid_x, grid_values
    ) - source_values[selected]
    if difference.size == 0:
        raise ValueError("reference region contains no digitized samples")
    return {
        "sample_count": int(difference.size),
        "rmse": float(np.sqrt(np.mean(difference**2)) * scale),
        "bias": float(np.mean(difference) * scale),
        "maximum_absolute_deviation": float(
            np.max(np.abs(difference)) * scale
        ),
    }


def _fixed_window_integrals(
    x_over_b: np.ndarray,
    pressure_pa: np.ndarray,
    half_width_m: float,
    window: tuple[float, float] = SPIKE_WINDOW,
) -> dict[str, float]:
    left, right = window
    if x_over_b[0] > left or x_over_b[-1] < right:
        raise ValueError("profile does not span the fixed spike window")
    interior = (x_over_b > left) & (x_over_b < right)
    x_window = np.concatenate(([left], x_over_b[interior], [right]))
    pressure_window = np.interp(x_window, x_over_b, pressure_pa)
    x_m = x_window * half_width_m
    baseline = np.interp(x_window, [left, right], pressure_window[[0, -1]])
    return {
        "window_x_over_b": [float(left), float(right)],
        "pressure_integral_pa_m": _trapezoid(pressure_window, x_m),
        "linear_baseline_excess_integral_pa_m": float(
            _trapezoid(pressure_window - baseline, x_m)
        ),
    }


def _trapezoid(values: np.ndarray, coordinates: np.ndarray) -> float:
    return float(
        np.sum(
            0.5
            * (values[1:] + values[:-1])
            * np.diff(coordinates)
        )
    )


def _load_profile(nodes: int, directory: Path) -> dict:
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "accepted_numerical_root":
        raise ValueError(f"{directory} is not an accepted numerical root")
    if summary["case_signature"]["grid"]["nodes"] != nodes:
        raise ValueError(f"{directory} summary does not match {nodes} nodes")
    data = np.genfromtxt(directory / "profiles.csv", delimiter=",", names=True)
    if data.size != nodes:
        raise ValueError(f"{directory} profile does not contain {nodes} rows")
    return {
        "nodes": nodes,
        "directory": str(directory),
        "summary": summary,
        "x_over_b": np.asarray(data["x_over_b"], dtype=np.float64),
        "pressure_pa": np.asarray(data["pressure_pa"], dtype=np.float64),
        "pressure_over_hertz": np.asarray(
            data["pressure_over_hertz"], dtype=np.float64
        ),
        "film_m": np.asarray(data["film_m"], dtype=np.float64),
        "film_R_over_b2": np.asarray(data["film_R_over_b2"], dtype=np.float64),
    }


def _check_benchmark_signatures(profiles: list[dict]) -> None:
    first = profiles[0]["summary"]["case_signature"]
    expected_inputs = first["inputs_si"]
    for profile in profiles:
        signature = profile["summary"]["case_signature"]
        groups = signature["dimensionless_groups"]
        if (
            signature["inputs_si"] != expected_inputs
            or not np.isclose(groups["M"], 20.0, rtol=1e-12, atol=0.0)
            or not np.isclose(groups["L"], 10.0, rtol=1e-12, atol=0.0)
        ):
            raise ValueError(f"{profile['nodes']}-node root has different benchmark physics")


def _grid_metrics(profile: dict, b_m: float, p_h_pa: float) -> dict:
    root = profile["summary"]["root"]
    features = root["features"]
    x = profile["x_over_b"]
    film_nm = profile["film_m"] * 1.0e9
    central_film_nm = float(np.interp(0.0, x, film_nm))
    outlet = (x >= 0.5) & (x <= 1.5)
    outlet_indices = np.flatnonzero(outlet)
    minimum_index = outlet_indices[np.argmin(film_nm[outlet])]
    spike_x_m = features["petrusevich_spike_location_m"]
    spike_width_m = features["petrusevich_spike_width_m"]
    spike_x_over_b = (
        float(spike_x_m / b_m) if spike_x_m is not None else None
    )
    spike_pressure_over_hertz = (
        float(np.interp(spike_x_over_b, x, profile["pressure_over_hertz"]))
        if spike_x_over_b is not None
        else None
    )
    return {
        "nodes": profile["nodes"],
        "dx_um": float((x[1] - x[0]) * b_m * 1.0e6),
        "h_c_nm": float(root["h_c_m"] * 1.0e9),
        "h_min_nm": float(root["h_min_m"] * 1.0e9),
        "h_min_x_over_b": float(
            features["minimum_film_location_m"] / b_m
        ),
        "outlet_minimum_nm": float(film_nm[minimum_index]),
        "outlet_minimum_x_over_b": float(x[minimum_index]),
        "outlet_depth_from_hc_nm": float(
            central_film_nm - film_nm[minimum_index]
        ),
        "p_max_mpa": float(root["p_max_pa"] / 1.0e6),
        "p_max_x_over_b": float(root["p_max_x_m"] / b_m),
        "spike_x_over_b": spike_x_over_b,
        "spike_location_error_x_over_b": (
            float(spike_x_over_b - VENNER_SPIKE_PEAK_X)
            if spike_x_over_b is not None
            else None
        ),
        "spike_pressure_over_hertz": spike_pressure_over_hertz,
        "spike_pressure_residual_over_hertz": (
            float(spike_pressure_over_hertz - VENNER_SPIKE_PEAK)
            if spike_pressure_over_hertz is not None
            else None
        ),
        "spike_width_um": (
            float(spike_width_m * 1.0e6)
            if spike_width_m is not None
            else None
        ),
        "spike_width_cells": (
            float(spike_width_m / (x[1] - x[0]) / b_m)
            if spike_width_m is not None
            else None
        ),
        "detector_half_prominence_integral_pa_m": features[
            "petrusevich_spike_integrated_pressure_pa_m"
        ],
        **_fixed_window_integrals(
            x, profile["pressure_pa"], b_m
        ),
        "load_relative_error": root["relative_load_error"],
        "local_mass_defect": root["audit"]["relative_local_balance"],
        "global_mass_defect": root["audit"]["relative_global_balance"],
        "complementarity": root["audit"]["maximum_complementarity"],
        "direct_elasticity_relative_error": root["audit"][
            "direct_elasticity_relative_error"
        ],
        "film_reconstruction_maximum_m": root["audit"][
            "direct_film_reconstruction_maximum_m"
        ],
        "maximum_viscosity_pa_s": root["maximum_viscosity_pa_s"],
        "maximum_density_kg_m3": root["maximum_density_kg_m3"],
    }


def _reference_table(profiles: list[dict], b_m: float, p_h_pa: float, film_scale_m: float):
    results = []
    for profile in profiles:
        x = profile["x_over_b"]
        row = {"nodes": profile["nodes"]}
        pressure = profile["pressure_over_hertz"]
        film = profile["film_R_over_b2"]
        row["fig5_pressure_all"] = _reference_errors(
            VENNER_FIG5_X,
            VENNER_FIG5_PRESSURE,
            x,
            pressure,
            p_h_pa / 1.0e6,
        )
        row["fig5_pressure_bulk"] = _reference_errors(
            VENNER_FIG5_X,
            VENNER_FIG5_PRESSURE,
            x,
            pressure,
            p_h_pa / 1.0e6,
            (VENNER_FIG5_X >= -0.8) & (VENNER_FIG5_X <= 0.8),
        )
        row["fig5_pressure_core"] = _reference_errors(
            VENNER_FIG5_X,
            VENNER_FIG5_PRESSURE,
            x,
            pressure,
            p_h_pa / 1.0e6,
            np.abs(VENNER_FIG5_X) <= 0.5,
        )
        row["fig5_pressure_outlet"] = _reference_errors(
            VENNER_FIG5_X,
            VENNER_FIG5_PRESSURE,
            x,
            pressure,
            p_h_pa / 1.0e6,
            VENNER_FIG5_X >= 0.5,
        )
        row["fig5_film_bulk"] = _reference_errors(
            VENNER_FIG5_FILM_X,
            VENNER_FIG5_FILM,
            x,
            film,
            film_scale_m * 1.0e9,
            (VENNER_FIG5_FILM_X >= -0.8)
            & (VENNER_FIG5_FILM_X <= 0.6),
        )
        row["fig5_film_outlet"] = _reference_errors(
            VENNER_FIG5_FILM_X,
            VENNER_FIG5_FILM,
            x,
            film,
            film_scale_m * 1.0e9,
            VENNER_FIG5_FILM_X >= 0.8,
        )
        spike_reference_x = np.array(
            [VENNER_SPIKE_MINIMUM_X, VENNER_SPIKE_PEAK_X]
        )
        spike_reference_pressure = np.array(
            [VENNER_SPIKE_MINIMUM, VENNER_SPIKE_PEAK]
        )
        row["fig6_spike_points"] = _reference_errors(
            spike_reference_x,
            spike_reference_pressure,
            x,
            pressure,
            p_h_pa / 1.0e6,
        )
        row["fig6_pre_spike_pressure_residual_over_hertz"] = float(
            np.interp(VENNER_SPIKE_MINIMUM_X, x, pressure)
            - VENNER_SPIKE_MINIMUM
        )
        row["fig6_peak_pressure_residual_over_hertz"] = float(
            np.interp(VENNER_SPIKE_PEAK_X, x, pressure) - VENNER_SPIKE_PEAK
        )
        row["fig6_pre_spike_pressure_residual_mpa"] = (
            row["fig6_pre_spike_pressure_residual_over_hertz"]
            * p_h_pa
            / 1.0e6
        )
        row["fig6_peak_pressure_residual_mpa"] = (
            row["fig6_peak_pressure_residual_over_hertz"]
            * p_h_pa
            / 1.0e6
        )
        row["central_film_nm"] = float(
            np.interp(0.0, x, profile["film_m"]) * 1.0e9
        )
        row["central_film_minus_published_nm"] = (
            row["central_film_nm"] - 0.0866 * film_scale_m * 1.0e9
        )
        row["minimum_film_nm"] = float(np.min(profile["film_m"]) * 1.0e9)
        results.append(row)
    return results


def _scalar_sensitivity(grid_rows: list[dict]) -> list[dict]:
    changes = []
    for coarse, fine in zip(grid_rows, grid_rows[1:], strict=False):
        row = {
            "coarse_nodes": coarse["nodes"],
            "fine_nodes": fine["nodes"],
        }
        for name in (
            "h_c_nm",
            "h_min_nm",
            "outlet_minimum_nm",
            "outlet_depth_from_hc_nm",
            "p_max_mpa",
            "spike_pressure_over_hertz",
            "spike_width_um",
            "spike_width_cells",
            "detector_half_prominence_integral_pa_m",
            "pressure_integral_pa_m",
            "linear_baseline_excess_integral_pa_m",
        ):
            coarse_value = coarse[name]
            fine_value = fine[name]
            row[f"{name}_change"] = (
                float(fine_value - coarse_value)
                if coarse_value is not None and fine_value is not None
                else None
            )
        for name in (
            "h_min_x_over_b",
            "outlet_minimum_x_over_b",
            "spike_x_over_b",
        ):
            coarse_value = coarse[name]
            fine_value = fine[name]
            row[f"{name}_change"] = (
                float(fine_value - coarse_value)
                if coarse_value is not None and fine_value is not None
                else None
            )
        changes.append(row)
    return changes


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("cannot write an empty table")
    flattened = []
    for row in rows:
        item = {}
        for key, value in row.items():
            if isinstance(value, dict):
                item.update({f"{key}_{name}": field for name, field in value.items()})
            else:
                item[key] = value
        flattened.append(item)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flattened[0]))
        writer.writeheader()
        writer.writerows(flattened)


def _write_profile_plot(
    path: Path,
    profiles: list[dict],
    b_m: float,
    p_h_pa: float,
    film_scale_m: float,
):
    colors = {
        257: "tab:blue",
        513: "tab:orange",
        1025: "tab:green",
        2049: "tab:red",
        4097: "tab:purple",
    }
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    pressure_ax, film_ax, pressure_residual_ax, film_residual_ax = axes.flat
    for profile in profiles:
        nodes = profile["nodes"]
        color = colors.get(nodes)
        x = profile["x_over_b"]
        pressure_ax.plot(
            x,
            profile["pressure_over_hertz"],
            label=f"{nodes} nodes",
            color=color,
            linewidth=1.1,
        )
        film_ax.plot(
            x,
            profile["film_R_over_b2"],
            label=f"{nodes} nodes",
            color=color,
            linewidth=1.1,
        )
        pressure_residual_ax.plot(
            VENNER_FIG5_X,
            (
                np.interp(VENNER_FIG5_X, x, profile["pressure_over_hertz"])
                - VENNER_FIG5_PRESSURE
            )
            * p_h_pa
            / 1.0e6,
            marker="o",
            markersize=3,
            label=f"{nodes} nodes",
            color=color,
        )
        film_residual_ax.plot(
            VENNER_FIG5_FILM_X,
            (
                np.interp(VENNER_FIG5_FILM_X, x, profile["film_R_over_b2"])
                - VENNER_FIG5_FILM
            )
            * film_scale_m
            * 1.0e9,
            marker="o",
            markersize=3,
            label=f"{nodes} nodes",
            color=color,
        )

    pressure_ax.errorbar(
        VENNER_FIG5_X,
        VENNER_FIG5_PRESSURE,
        xerr=VENNER_X_UNCERTAINTY,
        yerr=VENNER_PRESSURE_UNCERTAINTY,
        fmt="ko",
        markersize=3,
        capsize=2,
        label="Venner Fig. 5",
    )
    pressure_ax.errorbar(
        [VENNER_SPIKE_MINIMUM_X],
        [VENNER_SPIKE_MINIMUM],
        xerr=VENNER_X_UNCERTAINTY,
        yerr=VENNER_PRESSURE_UNCERTAINTY,
        fmt="kd",
        markersize=4,
        capsize=2,
        label="Venner Table 1 pre-spike",
    )
    pressure_ax.errorbar(
        [VENNER_SPIKE_PEAK_X],
        [VENNER_SPIKE_PEAK],
        xerr=FIGURE_6_PEAK_X_UNCERTAINTY,
        yerr=FIGURE_6_PEAK_PRESSURE_UNCERTAINTY,
        fmt="ks",
        markersize=4,
        capsize=2,
        label="Venner Fig. 6 peak",
    )
    pressure_ax.set(xlim=(-1.5, 1.5), xlabel="x/b", ylabel="p/p_H")
    pressure_ax.set_title("Pressure profile and digitized reference")
    pressure_ax.legend(fontsize=8, ncol=2)

    film_ax.errorbar(
        VENNER_FIG5_FILM_X,
        VENNER_FIG5_FILM,
        xerr=VENNER_X_UNCERTAINTY,
        yerr=VENNER_FILM_UNCERTAINTY,
        fmt="ko",
        markersize=3,
        capsize=2,
        label="Venner Fig. 5",
    )
    film_ax.set(xlim=(-2.0, 1.5), xlabel="x/b", ylabel="h R/b²")
    film_ax.set_title("Film profile, including outlet constriction")
    film_ax.legend(fontsize=8)

    pressure_residual_ax.axhline(0.0, color="black", linewidth=0.8)
    pressure_residual_ax.axvspan(*SPIKE_WINDOW, color="grey", alpha=0.15)
    pressure_residual_ax.set(
        xlabel="x/b",
        ylabel="Numerical - digitized pressure (MPa)",
        title="Figure 5 pressure residuals at digitized abscissae",
    )
    pressure_residual_ax.legend(fontsize=8)

    film_residual_ax.axhline(0.0, color="black", linewidth=0.8)
    film_residual_ax.axvspan(0.8, 1.4, color="grey", alpha=0.15)
    film_residual_ax.set(
        xlabel="x/b",
        ylabel="Numerical - digitized film (nm)",
        title="Figure 5 film residuals at digitized abscissae",
    )
    film_residual_ax.legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_grid_difference_plot(
    path: Path, profiles: list[dict], b_m: float, p_h_pa: float
):
    colors = ("tab:purple", "tab:brown", "tab:red", "tab:cyan")
    fig, axes = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, constrained_layout=True
    )
    pressure_ax, film_ax = axes
    for index, (coarse, fine) in enumerate(
        zip(profiles, profiles[1:], strict=False)
    ):
        x = coarse["x_over_b"]
        pressure_difference = (
            np.interp(x, fine["x_over_b"], fine["pressure_pa"])
            - coarse["pressure_pa"]
        ) / 1.0e6
        film_difference = (
            np.interp(x, fine["x_over_b"], fine["film_m"]) - coarse["film_m"]
        ) * 1.0e9
        label = f"{coarse['nodes']}→{fine['nodes']}"
        pressure_ax.plot(x, pressure_difference, label=label, color=colors[index])
        film_ax.plot(x, film_difference, label=label, color=colors[index])
    for axis in axes:
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.axvline(0.0, color="grey", linewidth=0.6)
        axis.set_xlim(-4.0, 1.5)
        axis.legend()
    pressure_ax.axvspan(*SPIKE_WINDOW, color="grey", alpha=0.15)
    pressure_ax.set_ylabel("Fine - coarse pressure (MPa)")
    pressure_ax.set_title("Nested-grid profile differences")
    film_ax.axvspan(0.5, 1.5, color="grey", alpha=0.15)
    film_ax.set(xlabel="x/b", ylabel="Fine - coarse film (nm)")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_outlet_plot(path: Path, profiles: list[dict]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True, constrained_layout=True)
    for profile in profiles:
        x = profile["x_over_b"]
        label = f"{profile['nodes']} nodes"
        axes[0].plot(x, profile["pressure_over_hertz"], label=label)
        axes[1].plot(x, profile["film_R_over_b2"], label=label)
    axes[0].errorbar(
        [VENNER_SPIKE_PEAK_X],
        [VENNER_SPIKE_PEAK],
        xerr=FIGURE_6_PEAK_X_UNCERTAINTY,
        yerr=FIGURE_6_PEAK_PRESSURE_UNCERTAINTY,
        fmt="ks",
        capsize=2,
        label="Venner Fig. 6 peak",
    )
    axes[0].plot(
        [VENNER_SPIKE_MINIMUM_X],
        [VENNER_SPIKE_MINIMUM],
        "kd",
        label="Venner Table 1 pre-spike",
    )
    outlet = (VENNER_FIG5_FILM_X >= 0.78) & (VENNER_FIG5_FILM_X <= 1.05)
    axes[1].errorbar(
        VENNER_FIG5_FILM_X[outlet],
        VENNER_FIG5_FILM[outlet],
        xerr=VENNER_X_UNCERTAINTY,
        yerr=VENNER_FILM_UNCERTAINTY,
        fmt="ko",
        capsize=2,
        label="Venner Fig. 5",
    )
    axes[0].set(xlim=(0.78, 1.05), ylabel="p/p_H", title="Outlet pressure spike")
    axes[1].set(xlabel="x/b", ylabel="h R/b²", title="Outlet film constriction")
    for axis in axes:
        axis.legend(fontsize=8, ncol=2)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def analyze(
    directories: dict[int, Path] | None = None,
    output_directory: Path | None = None,
) -> dict:
    directories = ACCEPTED_DIRECTORIES if directories is None else directories
    profiles = [
        _load_profile(nodes, path)
        for nodes, path in sorted(directories.items())
        if path.exists()
    ]
    if [profile["nodes"] for profile in profiles][:3] != [257, 513, 1025]:
        raise ValueError("accepted 257-, 513- and 1025-node roots are required")
    _check_benchmark_signatures(profiles)
    case, hertz, _, _, _ = _venner_case(1025)
    b_m = hertz.half_width_m
    p_h_pa = hertz.peak_pressure_pa
    grid_rows = [_grid_metrics(profile, b_m, p_h_pa) for profile in profiles]
    reference_rows = _reference_table(
        profiles, b_m, p_h_pa, case.film_scale_m
    )
    sensitivity_rows = []
    for coarse, fine in zip(profiles, profiles[1:], strict=False):
        for region, bounds in REGIONS.items():
            for variable, coarse_values, fine_values, unit in (
                (
                    "pressure",
                    coarse["pressure_pa"] / 1.0e6,
                    fine["pressure_pa"] / 1.0e6,
                    "MPa",
                ),
                (
                    "film",
                    coarse["film_m"] * 1.0e9,
                    fine["film_m"] * 1.0e9,
                    "nm",
                ),
            ):
                sensitivity_rows.append(
                    {
                        "coarse_nodes": coarse["nodes"],
                        "fine_nodes": fine["nodes"],
                        "region": region,
                        "variable": variable,
                        "unit": unit,
                        **_profile_difference(
                            coarse["x_over_b"],
                            coarse_values,
                            fine["x_over_b"],
                            fine_values,
                            bounds,
                        ),
                    }
                )
    report = {
        "benchmark": "Venner and ten Napel (1990), M=20, L=10",
        "fixed_spike_window_x_over_b": list(SPIKE_WINDOW),
        "fixed_window_definition": (
            "Integral of dimensional pressure over x in [0.85b,0.95b]; "
            "also report excess above the straight line joining the two "
            "window-end pressure values."
        ),
        "grid_metrics": grid_rows,
        "scalar_sensitivity": _scalar_sensitivity(grid_rows),
        "reference_comparison": reference_rows,
        "grid_sensitivity": sensitivity_rows,
        "reference_uncertainties_dimensionless": {
            "figure_5_x_over_b": VENNER_X_UNCERTAINTY,
            "figure_5_pressure_over_hertz": VENNER_PRESSURE_UNCERTAINTY,
            "figure_5_hR_over_b2": VENNER_FILM_UNCERTAINTY,
            "figure_6_peak_x_over_b": FIGURE_6_PEAK_X_UNCERTAINTY,
            "figure_6_peak_pressure_over_hertz": (
                FIGURE_6_PEAK_PRESSURE_UNCERTAINTY
            ),
        },
        "interpretation": (
            "Grid differences are observed sensitivities, not Richardson/GCI "
            "error estimates. The digitized comparison is separate from "
            "grid sensitivity and does not include constitutive-model "
            "uncertainty."
        ),
    }
    if output_directory is not None:
        if output_directory.exists():
            raise FileExistsError(
                f"refusing to overwrite existing output directory {output_directory}"
            )
        output_directory.mkdir(parents=True)
        (output_directory / "summary.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _write_csv(output_directory / "grid_metrics.csv", grid_rows)
        _write_csv(
            output_directory / "scalar_sensitivity.csv",
            report["scalar_sensitivity"],
        )
        _write_csv(output_directory / "reference_comparison.csv", reference_rows)
        _write_csv(output_directory / "grid_sensitivity.csv", sensitivity_rows)
        _write_profile_plot(
            output_directory / "profiles_and_reference.png",
            profiles,
            b_m,
            p_h_pa,
            case.film_scale_m,
        )
        _write_grid_difference_plot(
            output_directory / "grid_differences.png",
            profiles,
            b_m,
            p_h_pa,
        )
        _write_outlet_plot(output_directory / "outlet_detail.png", profiles)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    report = analyze(output_directory=args.output_dir)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
