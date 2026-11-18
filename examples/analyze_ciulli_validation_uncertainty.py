"""Compare saved accepted Ciulli donor grids and Figure 13 profiles."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from ehl_solver.pressure_analysis import (
    centerline_pressure_profile,
    outlet_spike_diagnostics,
)
from examples.analyze_ciulli_donor_grid import (
    OUTLET_WINDOW_M,
    _centerline_film,
    _fixed_spike_region_pressure,
    _load_run,
    _midplane_comparison,
    _outlet_integrated_pressure,
    _profile_comparison,
    _transverse_film,
    profile_error_metrics,
    regional_mesh_comparison,
)

EXPECTED_GRIDS = (65, 129, 257, 513, 1025)
PROFILE_DIRECTIONS = ("longitudinal", "transverse")
PROFILE_REGIONS = {
    "longitudinal_bulk": {
        "direction": "longitudinal",
        "bounds_um": [-70.0, 70.0],
        "description": "x in [-70,70] um, y=0",
    },
    "longitudinal_outlet_constriction": {
        "direction": "longitudinal",
        "bounds_um": [80.0, 100.0],
        "description": "x in [80,100] um, y=0",
    },
    "transverse_bulk": {
        "direction": "transverse",
        "bounds_um": [-60.0, 60.0],
        "description": "y in [-60,60] um, x=0",
    },
    "transverse_side_constrictions": {
        "direction": "transverse",
        "absolute_minimum_um": 70.0,
        "description": "70 <= |y| <= 100 um, x=0",
    },
}


def _validate_sequence(runs):
    ordered = sorted(runs, key=lambda run: run["case"]["nx"])
    grids = tuple(run["case"]["nx"] for run in ordered)
    if grids != EXPECTED_GRIDS or any(
        run["case"]["ny"] != run["case"]["nx"] for run in ordered
    ):
        raise ValueError(f"Expected square donor grids {EXPECTED_GRIDS}, got {grids}")

    reference = ordered[0]["case"]
    physical_keys = (
        "load_n",
        "radius_m",
        "contact_modulus_pa",
        "viscosity_pa_s",
        "density_kg_m3",
        "entrainment_m_s",
        "alpha_per_pa",
        "domain_half_width_x_m",
        "domain_half_width_y_m",
        "transport",
    )
    if reference["transport"] != "donor" or any(
        any(run["case"][key] != reference[key] for key in physical_keys)
        for run in ordered[1:]
    ):
        raise ValueError("Runs must use identical donor physics and physical domain")
    if any(
        (right["case"]["nx"] - 1) != 2 * (left["case"]["nx"] - 1)
        for left, right in zip(ordered[:-1], ordered[1:], strict=True)
    ):
        raise ValueError("The accepted sequence must have refinement ratio two")
    return ordered


def _root_metrics(run, summary):
    pressure = run["pressure_pa"]
    film = run["film_m"]
    j_min, i_min = np.unravel_index(np.argmin(film), film.shape)
    j_max, i_max = np.unravel_index(np.argmax(pressure), pressure.shape)
    dx_um = float(np.median(np.diff(run["x_m"])) * 1e6)
    pressure_x, centerline_pressure = centerline_pressure_profile(
        run["x_m"], run["y_m"], pressure
    )
    outlet = outlet_spike_diagnostics(
        pressure_x, centerline_pressure, window_m=OUTLET_WINDOW_M
    )
    if outlet.get("identified"):
        outlet_width_um = float(outlet["half_prominence_width_m"] * 1e6)
        outlet_width_cells = outlet_width_um / dx_um
        outlet_peak_x_um = float(outlet["peak_x_m"] * 1e6)
        outlet_peak_mpa = float(outlet["peak_pressure_pa"] / 1e6)
        outlet_integral = _outlet_integrated_pressure(run, outlet)
    else:
        outlet_width_um = outlet_width_cells = None
        outlet_peak_x_um = outlet_peak_mpa = None
        outlet_integral = {"defined": False, "reason": outlet.get("reason")}

    runtime = summary.get("runtime", {})
    return {
        "grid_ny_nx": [run["case"]["ny"], run["case"]["nx"]],
        "spacing_um": dx_um,
        "state_sha256": summary["state_sha256"],
        "linear_algebra_backend": summary.get("linear_backend")
        or runtime.get("linear_algebra_backend", "not recorded"),
        "numerically_accepted": run["audit"]["numerically_accepted"],
        "force_n": float(run["audit"]["force_n"]),
        "relative_local_balance": float(
            run["audit"]["mass"]["relative_local_balance"]
        ),
        "relative_global_balance": float(
            run["audit"]["mass"]["relative_global_balance"]
        ),
        "h_c_nm": float(run["audit"]["central_film_m"] * 1e9),
        "h_min_nm": float(np.min(film) * 1e9),
        "h_min_xy_um": [
            float(run["x_m"][i_min] * 1e6),
            float(run["y_m"][j_min] * 1e6),
        ],
        "p_max_mpa": float(np.max(pressure) / 1e6),
        "p_max_xy_um": [
            float(run["x_m"][i_max] * 1e6),
            float(run["y_m"][j_max] * 1e6),
        ],
        "fixed_outlet_corridor": _fixed_spike_region_pressure(run),
        "outlet_sharp_feature": {
            "centerline_peak_x_um": outlet_peak_x_um,
            "centerline_peak_mpa": outlet_peak_mpa,
            "centerline_75_175_half_prominence_width_um": outlet_width_um,
            "centerline_75_175_half_prominence_width_cells": outlet_width_cells,
            "variable_width_pressure_integral": outlet_integral,
            "interpretation": (
                "Unresolved centerline 75-175 um window diagnostic; distinct "
                "from the fixed-band widths used for the connected locus."
            ),
        },
    }


def _pair_metrics(coarse, fine):
    regional = regional_mesh_comparison(coarse, fine)
    regions = regional["regions"]
    midplanes = _midplane_comparison(coarse, fine)
    coarse_hc = float(coarse["audit"]["central_film_m"] * 1e9)
    fine_hc = float(fine["audit"]["central_film_m"] * 1e9)
    coarse_hmin = float(np.min(coarse["film_m"]) * 1e9)
    fine_hmin = float(np.min(fine["film_m"]) * 1e9)
    coarse_fixed = _fixed_spike_region_pressure(coarse)["pressure_integral_n"]
    fine_fixed = _fixed_spike_region_pressure(fine)["pressure_integral_n"]
    return {
        "comparison": (
            f"{coarse['case']['nx']}x{coarse['case']['ny']} -> "
            f"{fine['case']['nx']}x{fine['case']['ny']}"
        ),
        "scalars": {
            "h_c_nm": {
                "coarse": coarse_hc,
                "fine": fine_hc,
                "change": fine_hc - coarse_hc,
                "relative_change_percent": 100 * (fine_hc - coarse_hc) / coarse_hc,
            },
            "h_min_nm": {
                "coarse": coarse_hmin,
                "fine": fine_hmin,
                "change": fine_hmin - coarse_hmin,
                "relative_change_percent": (
                    100 * (fine_hmin - coarse_hmin) / coarse_hmin
                ),
            },
            "fixed_outlet_corridor_pressure_integral_n": {
                "coarse": coarse_fixed,
                "fine": fine_fixed,
                "change": fine_fixed - coarse_fixed,
                "relative_change_percent": (
                    100 * (fine_fixed - coarse_fixed) / coarse_fixed
                ),
            },
        },
        "regions": regions,
        "midplane_profiles": midplanes,
        "interpretation": (
            "Direct adjacent-grid differences on common coordinates; not "
            "Richardson extrapolation, GCI, or a formal uncertainty estimate."
        ),
    }


def _profile_rows(runs, rows):
    profiles = []
    for direction in PROFILE_DIRECTIONS:
        selected = sorted(
            (row for row in rows if row["direction"] == direction),
            key=lambda row: float(row["coordinate_um"]),
        )
        coordinate = np.array(
            [float(row["coordinate_um"]) for row in selected], dtype=np.float64
        )
        observed_nm = np.array(
            [float(row["film_thickness_um"]) * 1000 for row in selected]
        )
        uncertainty_nm = np.array(
            [float(row["digitization_uncertainty_um"]) * 1000 for row in selected]
        )
        for run in runs:
            axis = run["x_m"] if direction == "longitudinal" else run["y_m"]
            values = (
                _centerline_film(run)
                if direction == "longitudinal"
                else _transverse_film(run)
            )
            predicted_nm = np.interp(coordinate, axis * 1e6, values * 1e9)
            for i, x_um in enumerate(coordinate):
                profiles.append(
                    {
                        "direction": direction,
                        "coordinate_um": float(x_um),
                        "grid_n": run["case"]["nx"],
                        "observed_film_nm": float(observed_nm[i]),
                        "digitization_uncertainty_nm": float(uncertainty_nm[i]),
                        "predicted_film_nm": float(predicted_nm[i]),
                        "error_nm": float(predicted_nm[i] - observed_nm[i]),
                    }
                )
    return profiles


def _region_mask(region, coordinates_um):
    coordinates_um = np.asarray(coordinates_um, dtype=np.float64)
    if "absolute_minimum_um" in region:
        return abs(coordinates_um) >= region["absolute_minimum_um"]
    lower, upper = region["bounds_um"]
    return (coordinates_um >= lower) & (coordinates_um <= upper)


def _profile_error_summary(observed_nm, predicted_nm, uncertainty_nm):
    metrics = profile_error_metrics(
        np.asarray(observed_nm) / 1000,
        np.asarray(predicted_nm) / 1000,
        np.asarray(uncertainty_nm) / 1000,
    )
    return {
        "point_count": metrics["point_count"],
        "rmse_nm": metrics["rmse_um"] * 1000,
        "bias_nm": metrics["bias_um"] * 1000,
        "maximum_absolute_deviation_nm": (
            metrics["maximum_absolute_error_um"] * 1000
        ),
        "digitization_rms_nm": metrics["digitization_rms_um"] * 1000,
        "points_within_digitization_estimate": (
            metrics["points_within_digitization_estimate"]
        ),
    }


def _profile_region_assessment(profile_rows):
    agreement = []
    grid_sensitivity = []
    for region_name, region in PROFILE_REGIONS.items():
        direction = region["direction"]
        measured_rows = sorted(
            (
                row
                for row in profile_rows
                if row["direction"] == direction
                and row["grid_n"] == EXPECTED_GRIDS[0]
                and _region_mask(region, [row["coordinate_um"]])[0]
            ),
            key=lambda row: row["coordinate_um"],
        )
        observed_nm = np.array(
            [row["observed_film_nm"] for row in measured_rows]
        )
        uncertainty_nm = np.array(
            [row["digitization_uncertainty_nm"] for row in measured_rows]
        )
        coordinates_um = np.array(
            [row["coordinate_um"] for row in measured_rows]
        )
        predictions = {}
        for grid in EXPECTED_GRIDS:
            grid_rows = {
                row["coordinate_um"]: row["predicted_film_nm"]
                for row in profile_rows
                if row["direction"] == direction and row["grid_n"] == grid
            }
            predictions[grid] = np.array(
                [grid_rows[coordinate] for coordinate in coordinates_um]
            )

        final_metrics = _profile_error_summary(
            observed_nm,
            predictions[EXPECTED_GRIDS[-1]],
            uncertainty_nm,
        )
        agreement.append(
            {
                "region": region_name,
                "direction": direction,
                "definition": region["description"],
                "coordinate_bounds_um": (
                    region.get("bounds_um")
                    or [
                        -100.0,
                        -region["absolute_minimum_um"],
                        region["absolute_minimum_um"],
                        100.0,
                    ]
                ),
                **final_metrics,
            }
        )
        for coarse_grid, fine_grid in zip(
            EXPECTED_GRIDS[:-1], EXPECTED_GRIDS[1:], strict=True
        ):
            change_nm = predictions[fine_grid] - predictions[coarse_grid]
            grid_sensitivity.append(
                {
                    "region": region_name,
                    "direction": direction,
                    "comparison": f"{coarse_grid}x{coarse_grid} -> "
                    f"{fine_grid}x{fine_grid}",
                    "point_count": int(change_nm.size),
                    "grid_change_rms_nm": float(
                        np.sqrt(np.mean(change_nm**2))
                    ),
                    "grid_change_bias_nm": float(np.mean(change_nm)),
                    "grid_change_maximum_absolute_nm": float(
                        np.max(abs(change_nm))
                    ),
                    "digitization_rms_nm": final_metrics["digitization_rms_nm"],
                    "grid_rms_over_digitization_rms": float(
                        np.sqrt(np.mean(change_nm**2))
                        / final_metrics["digitization_rms_nm"]
                    ),
                }
            )
    return {
        "region_definitions": PROFILE_REGIONS,
        "agreement_1025": agreement,
        "adjacent_grid_sensitivity_at_digitized_coordinates": grid_sensitivity,
        "interpretation": (
            "Regional error metrics compare the 1025x1025 profile with "
            "digitized Figure 13 points. Adjacent-grid differences are "
            "observed sensitivities at those same points, not uncertainty "
            "bounds. Digitization estimates are not statistical confidence "
            "intervals; coordinate read-off uncertainty is not propagated."
        ),
    }


def _minimum_position_offsets(direction, model_coordinate_um, measured_coordinates_um):
    offsets = []
    for measured_coordinate in measured_coordinates_um:
        model_coordinate = model_coordinate_um
        if direction == "transverse":
            model_coordinate = float(
                np.copysign(abs(model_coordinate_um), measured_coordinate)
            )
        offsets.append(float(model_coordinate - measured_coordinate))
    return offsets


def _central_and_minimum_comparison(runs, profile_rows, root_metrics):
    central_film = []
    for direction in PROFILE_DIRECTIONS:
        central_rows = [
            row
            for row in profile_rows
            if row["direction"] == direction and row["coordinate_um"] == 0.0
        ]
        observed = central_rows[0]["observed_film_nm"]
        uncertainty = central_rows[0]["digitization_uncertainty_nm"]
        model = central_rows[-1]["predicted_film_nm"]
        central_film.append(
            {
                "profile": direction,
                "digitized_h_c_nm": observed,
                "digitization_uncertainty_nm": uncertainty,
                "model_h_c_1025_nm": model,
                "model_minus_digitized_nm": model - observed,
                "within_digitization_estimate": abs(model - observed)
                <= uncertainty,
            }
        )

    profile_minima = []
    for direction in PROFILE_DIRECTIONS:
        measured_rows = sorted(
            (
                row
                for row in profile_rows
                if row["direction"] == direction
                and row["grid_n"] == EXPECTED_GRIDS[0]
            ),
            key=lambda row: row["coordinate_um"],
        )
        measured_minimum = min(
            measured_rows, key=lambda row: row["observed_film_nm"]
        )
        ties = [
            row
            for row in measured_rows
            if row["observed_film_nm"]
            == measured_minimum["observed_film_nm"]
        ]
        run = runs[-1]
        axis = run["x_m"] if direction == "longitudinal" else run["y_m"]
        film = (
            _centerline_film(run)
            if direction == "longitudinal"
            else _transverse_film(run)
        )
        in_window = (axis >= -100e-6) & (axis <= 100e-6)
        cut_indices = np.flatnonzero(in_window)
        cut_min_index = cut_indices[np.argmin(film[in_window])]
        cut_minimum_nm = float(film[cut_min_index] * 1e9)
        cut_minimum_coordinate_um = float(axis[cut_min_index] * 1e6)
        if direction == "transverse":
            cut_minimum_coordinate_um = abs(cut_minimum_coordinate_um)
        measured_minimum_coordinates = [
            float(row["coordinate_um"]) for row in ties
        ]
        model_at_measured_minima_nm = [
            float(
                np.interp(
                    coordinate,
                    axis * 1e6,
                    film * 1e9,
                )
            )
            for coordinate in measured_minimum_coordinates
        ]
        profile_minima.append(
            {
                "profile": direction,
                "digitized_minimum_nm": measured_minimum["observed_film_nm"],
                "digitization_uncertainty_nm": measured_minimum[
                    "digitization_uncertainty_nm"
                ],
                "digitized_minimum_coordinates_um": measured_minimum_coordinates,
                "model_cut_minimum_1025_nm": cut_minimum_nm,
                "model_cut_minimum_coordinate_um": cut_minimum_coordinate_um,
                "cut_minimum_minus_digitized_minimum_nm": (
                    cut_minimum_nm - measured_minimum["observed_film_nm"]
                ),
                "model_value_at_digitized_minimum_coordinates_nm": (
                    model_at_measured_minima_nm
                ),
                "model_minus_measurement_at_digitized_minimum_nm": [
                    value - measured_minimum["observed_film_nm"]
                    for value in model_at_measured_minima_nm
                ],
                "minimum_position_offset_um": _minimum_position_offsets(
                    direction,
                    cut_minimum_coordinate_um,
                    measured_minimum_coordinates,
                ),
                "interpretation": (
                    "This is a 1-D midplane-cut minimum, not the global "
                    "2-D minimum."
                ),
            }
        )

    global_h_min = root_metrics[-1]["h_min_nm"]
    global_h_min_xy = root_metrics[-1]["h_min_xy_um"]
    lowest_digitized = min(
        (row for row in profile_rows if row["grid_n"] == EXPECTED_GRIDS[0]),
        key=lambda row: row["observed_film_nm"],
    )
    return {
        "central_film": central_film,
        "profile_minima": profile_minima,
        "global_h_min": {
            "model_h_min_1025_nm": global_h_min,
            "model_location_xy_um": global_h_min_xy,
            "h_min_513_nm": root_metrics[-2]["h_min_nm"],
            "change_513_to_1025_nm": (
                global_h_min - root_metrics[-2]["h_min_nm"]
            ),
            "lowest_digitized_profile_point_nm": lowest_digitized[
                "observed_film_nm"
            ],
            "lowest_digitized_profile_point_uncertainty_nm": lowest_digitized[
                "digitization_uncertainty_nm"
            ],
            "lowest_digitized_profile_point": {
                "direction": lowest_digitized["direction"],
                "coordinate_um": lowest_digitized["coordinate_um"],
            },
            "nominal_scalar_difference_nm": (
                global_h_min - lowest_digitized["observed_film_nm"]
            ),
            "directly_comparable": False,
            "interpretation": (
                "Figure 13 gives only x=0 and y=0 cuts. The global model "
                "minimum is off those cuts, so the nominal scalar difference "
                "is not a direct validation of global h_min."
            ),
        },
    }


def _outlet_integral_rows(root_metrics):
    rows = []
    for metric in root_metrics:
        variable = metric["outlet_sharp_feature"][
            "variable_width_pressure_integral"
        ]
        rows.append(
            {
                "grid_n": metric["grid_ny_nx"][1],
                "fixed_corridor_integral_n": metric["fixed_outlet_corridor"][
                    "pressure_integral_n"
                ],
                "variable_half_prominence_integral_n": variable.get(
                    "pressure_integral_n"
                ),
                "variable_half_prominence_x_window_um": (
                    variable.get("x_window_um")
                ),
                "centerline_pressure_integral_pa_m": variable.get(
                    "centerline_pressure_integral_pa_m"
                ),
                "centerline_excess_integral_pa_m": variable.get(
                    "centerline_excess_above_local_linear_baseline_pa_m"
                ),
            }
        )
    return rows


def _plot_profiles(runs, rows, path):
    colors = plt.get_cmap("viridis")(np.linspace(0.12, 0.86, len(runs)))
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for axis, direction in zip(axes, PROFILE_DIRECTIONS, strict=True):
        visible_films = []
        if direction == "longitudinal":
            axis.axvspan(-70, 70, color="#dceaf7", alpha=0.45, zorder=0)
            axis.axvspan(80, 100, color="#f8dfcf", alpha=0.55, zorder=0)
            title = "Longitudinal: bulk [-70,70], outlet [80,100] um"
        else:
            axis.axvspan(-60, 60, color="#dceaf7", alpha=0.45, zorder=0)
            axis.axvspan(-100, -70, color="#f8dfcf", alpha=0.55, zorder=0)
            axis.axvspan(70, 100, color="#f8dfcf", alpha=0.55, zorder=0)
            title = "Transverse: bulk |y|<=60, side regions |y|>=70 um"
        for color, run in zip(colors, runs, strict=True):
            x = run["x_m"] if direction == "longitudinal" else run["y_m"]
            h = (
                _centerline_film(run)
                if direction == "longitudinal"
                else _transverse_film(run)
            )
            visible_films.extend(h[abs(x) <= 100e-6] * 1e9)
            is_finest = run["case"]["nx"] == EXPECTED_GRIDS[-1]
            axis.plot(
                x * 1e6,
                h * 1e9,
                color=color,
                linewidth=2.2 if is_finest else 1.1,
                alpha=1.0 if is_finest else 0.75,
                label=f"Donor {run['case']['nx']}x{run['case']['ny']}",
            )
        measured = sorted(
            (row for row in rows if row["direction"] == direction),
            key=lambda row: float(row["coordinate_um"]),
        )
        observed_nm = np.array(
            [float(row["film_thickness_um"]) * 1000 for row in measured]
        )
        uncertainty_nm = np.array(
            [float(row["digitization_uncertainty_um"]) * 1000 for row in measured]
        )
        axis.errorbar(
            [float(row["coordinate_um"]) for row in measured],
            observed_nm,
            yerr=uncertainty_nm,
            fmt="ko",
            markersize=3,
            capsize=1.5,
            linewidth=0.8,
            label="Ciulli Fig. 13 digitization",
            zorder=5,
        )
        axis.set_xlabel(
            "x [um], y=0" if direction == "longitudinal" else "y [um], x=0"
        )
        axis.set_ylabel("Film thickness [nm]")
        axis.set_title(title)
        low = min(min(visible_films), float(np.min(observed_nm - uncertainty_nm)))
        high = max(max(visible_films), float(np.max(observed_nm + uncertainty_nm)))
        padding = max(5.0, 0.06 * (high - low))
        axis.set_xlim(-100, 100)
        axis.set_ylim(low - padding, high + padding)
        axis.grid(True, alpha=0.25)
        handles, _ = axis.get_legend_handles_labels()
        handles.extend(
            [
                Patch(facecolor="#dceaf7", alpha=0.65, label="Bulk region"),
                Patch(
                    facecolor="#f8dfcf",
                    alpha=0.75,
                    label="Outlet/side-constriction region",
                ),
            ]
        )
        axis.legend(handles=handles, fontsize=8)
    figure.suptitle(
        "Ciulli Fig. 13, S=0: accepted donor roots on the fixed 0.8 mm domain"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_profile_residuals(profile_rows, path):
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for axis, direction in zip(axes, PROFILE_DIRECTIONS, strict=True):
        selected = sorted(
            (
                row
                for row in profile_rows
                if row["direction"] == direction
                and row["grid_n"] == EXPECTED_GRIDS[-1]
            ),
            key=lambda row: row["coordinate_um"],
        )
        coordinates = np.array([row["coordinate_um"] for row in selected])
        errors = np.array([row["error_nm"] for row in selected])
        uncertainties = np.array(
            [row["digitization_uncertainty_nm"] for row in selected]
        )
        if direction == "longitudinal":
            axis.axvspan(-70, 70, color="#dceaf7", alpha=0.45, zorder=0)
            axis.axvspan(80, 100, color="#f8dfcf", alpha=0.55, zorder=0)
        else:
            axis.axvspan(-60, 60, color="#dceaf7", alpha=0.45, zorder=0)
            axis.axvspan(-100, -70, color="#f8dfcf", alpha=0.55, zorder=0)
            axis.axvspan(70, 100, color="#f8dfcf", alpha=0.55, zorder=0)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.errorbar(
            coordinates,
            errors,
            yerr=uncertainties,
            fmt="o",
            color="#b2182b",
            capsize=2,
            markersize=4,
        )
        axis.set_xlim(-100, 100)
        axis.set_xlabel(
            "x [um], y=0" if direction == "longitudinal" else "y [um], x=0"
        )
        axis.set_ylabel("1025-grid minus digitized film [nm]")
        axis.set_title(direction.capitalize())
        axis.grid(True, alpha=0.25)
    figure.suptitle(
        "Figure 13 pointwise residuals with digitization estimates "
        "(not confidence intervals)"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_uncertainty(runs, root_metrics, pairs, path):
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    grids = np.array([run["case"]["nx"] for run in runs])
    hc = np.array([row["h_c_nm"] for row in root_metrics])
    hmin = np.array([row["h_min_nm"] for row in root_metrics])
    axes[0, 0].plot(grids, hc, "o-", label="h_c")
    axes[0, 0].plot(grids, hmin, "s-", label="h_min")
    axes[0, 0].set_xscale("log", base=2)
    axes[0, 0].set_xticks(grids, labels=[str(value) for value in grids])
    axes[0, 0].set_ylabel("Film thickness [nm]")
    axes[0, 0].set_title("Scalar sequence")
    axes[0, 0].legend()

    labels = [pair["comparison"].split(" -> ")[0].split("x")[0] for pair in pairs]
    longitudinal = [
        pair["midplane_profiles"]["longitudinal"]["film_m"]["rms"]
        for pair in pairs
    ]
    transverse = [
        pair["midplane_profiles"]["transverse"]["film_m"]["rms"]
        for pair in pairs
    ]
    axes[0, 1].plot(labels, longitudinal, "o-", label="Longitudinal")
    axes[0, 1].plot(labels, transverse, "s-", label="Transverse")
    axes[0, 1].set_ylabel("Adjacent-grid film RMS [nm]")
    axes[0, 1].set_title("Centerline profile differences")
    axes[0, 1].legend()

    main_pressure = [
        pair["regions"]["main_contact_excluding_spike"]["pressure_relative_l2"]
        * 100
        for pair in pairs
    ]
    full_pressure = [
        pair["regions"]["full_contact"]["pressure_relative_l2"] * 100
        for pair in pairs
    ]
    axes[1, 0].plot(labels, main_pressure, "o-", label="Main contact")
    axes[1, 0].plot(labels, full_pressure, "s-", label="Full contact")
    axes[1, 0].set_ylabel("Adjacent-grid pressure relative L2 [%]")
    axes[1, 0].set_title("Bulk pressure differences")
    axes[1, 0].legend()

    fixed_integral = np.array(
        [
            row["fixed_outlet_corridor"]["pressure_integral_n"]
            for row in root_metrics
        ]
    )
    axes[1, 1].plot(grids, fixed_integral, "o-")
    axes[1, 1].set_xscale("log", base=2)
    axes[1, 1].set_xticks(grids, labels=[str(value) for value in grids])
    axes[1, 1].set_ylabel("Fixed outlet-corridor pressure integral [N]")
    axes[1, 1].set_title("x=[75,100] um, |y|<=a_H")
    for axis in axes.flat:
        axis.grid(True, alpha=0.25)
    figure.suptitle(
        "Observed grid sensitivity only; no Richardson/GCI extrapolation"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_csv(path, rows, fieldnames):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyze(mesh_directories, profile_csv, output_directory):
    directories = [Path(path) for path in mesh_directories]
    runs = _validate_sequence([_load_run(path) for path in directories])
    summaries = {}
    for run in runs:
        summary_path = Path(run["directory"]) / "summary.json"
        summaries[run["case"]["nx"]] = json.loads(summary_path.read_text())
    with Path(profile_csv).open(newline="") as stream:
        digitized_rows = list(csv.DictReader(stream))
    root_metrics = [
        _root_metrics(run, summaries[run["case"]["nx"]]) for run in runs
    ]
    pairs = [
        _pair_metrics(coarse, fine)
        for coarse, fine in zip(runs[:-1], runs[1:], strict=True)
    ]
    finest_profiles = {
        direction: _profile_comparison(runs[-1], digitized_rows, direction)
        for direction in PROFILE_DIRECTIONS
    }
    profile_predictions = _profile_rows(runs, digitized_rows)
    profile_regions = _profile_region_assessment(profile_predictions)
    film_features = _central_and_minimum_comparison(
        runs, profile_predictions, root_metrics
    )
    outlet_integrals = _outlet_integral_rows(root_metrics)
    result = {
        "case": {
            key: value
            for key, value in runs[-1]["case"].items()
            if key not in ("nx", "ny", "maximum_physical_nodes")
        },
        "equation": "donor-flux JFO/Elrod-Adams",
        "grid_sequence": root_metrics,
        "adjacent_grid_differences": pairs,
        "figure13_profile_comparison_1025": finest_profiles,
        "figure13_profile_region_assessment": profile_regions,
        "central_and_minimum_film_comparison": film_features,
        "outlet_pressure_integrals": outlet_integrals,
        "uncertainty_interpretation": (
            "These are observed differences between accepted discrete roots. "
            "The sequence is not treated as asymptotic; no Richardson/GCI "
            "estimate or formal continuum uncertainty is claimed. p_max and "
            "outlet widths are unresolved sharp-feature diagnostics. "
            "Experimental and model-form sensitivities are not combined."
        ),
        "no_solver_run": True,
    }

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "uncertainty_summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)
    )
    root_csv = [
        {
            "grid_n": row["grid_ny_nx"][1],
            "spacing_um": row["spacing_um"],
            "state_sha256": row["state_sha256"],
            "linear_algebra_backend": row["linear_algebra_backend"],
            "h_c_nm": row["h_c_nm"],
            "h_min_nm": row["h_min_nm"],
            "p_max_mpa": row["p_max_mpa"],
            "fixed_outlet_corridor_pressure_integral_n": row[
                "fixed_outlet_corridor"
            ]["pressure_integral_n"],
            "force_n": row["force_n"],
            "relative_local_balance": row["relative_local_balance"],
            "relative_global_balance": row["relative_global_balance"],
        }
        for row in root_metrics
    ]
    _write_csv(
        output_directory / "grid_roots.csv",
        root_csv,
        list(root_csv[0]),
    )
    pair_csv = []
    for pair in pairs:
        main = pair["regions"]["main_contact_excluding_spike"]
        full = pair["regions"]["full_contact"]
        pair_csv.append(
            {
                "comparison": pair["comparison"],
                "h_c_change_nm": pair["scalars"]["h_c_nm"]["change"],
                "h_c_relative_change_percent": pair["scalars"]["h_c_nm"][
                    "relative_change_percent"
                ],
                "h_min_change_nm": pair["scalars"]["h_min_nm"]["change"],
                "h_min_relative_change_percent": pair["scalars"]["h_min_nm"][
                    "relative_change_percent"
                ],
                "longitudinal_film_profile_rms_nm": pair["midplane_profiles"][
                    "longitudinal"
                ]["film_m"]["rms"],
                "transverse_film_profile_rms_nm": pair["midplane_profiles"][
                    "transverse"
                ]["film_m"]["rms"],
                "main_pressure_relative_l2_percent": (
                    main["pressure_relative_l2"] * 100
                ),
                "main_pressure_rms_mpa": main[
                    "pressure_rms_mpa_over_positive_support"
                ],
                "full_pressure_relative_l2_percent": (
                    full["pressure_relative_l2"] * 100
                ),
                "full_pressure_rms_mpa": full[
                    "pressure_rms_mpa_over_positive_support"
                ],
                "fixed_outlet_integral_change_n": pair["scalars"][
                    "fixed_outlet_corridor_pressure_integral_n"
                ]["change"],
                "fixed_outlet_integral_change_percent": pair["scalars"][
                    "fixed_outlet_corridor_pressure_integral_n"
                ]["relative_change_percent"],
            }
        )
    _write_csv(
        output_directory / "adjacent_grid_differences.csv",
        pair_csv,
        list(pair_csv[0]),
    )
    sharp_rows = [
        {
            "grid_n": row["grid_ny_nx"][1],
            "p_max_mpa": row["p_max_mpa"],
            "p_max_x_um": row["p_max_xy_um"][0],
            "p_max_y_um": row["p_max_xy_um"][1],
            "centerline_outlet_peak_x_um": row["outlet_sharp_feature"][
                "centerline_peak_x_um"
            ],
            "centerline_outlet_peak_mpa": row["outlet_sharp_feature"][
                "centerline_peak_mpa"
            ],
            "centerline_75_175_half_prominence_width_um": row[
                "outlet_sharp_feature"
            ]["centerline_75_175_half_prominence_width_um"],
            "centerline_75_175_half_prominence_width_cells": row[
                "outlet_sharp_feature"
            ]["centerline_75_175_half_prominence_width_cells"],
        }
        for row in root_metrics
    ]
    _write_csv(
        output_directory / "sharp_feature_diagnostics.csv",
        sharp_rows,
        list(sharp_rows[0]),
    )
    _write_csv(
        output_directory / "figure13_profile_predictions.csv",
        profile_predictions,
        list(profile_predictions[0]),
    )
    region_agreement_rows = [
        {
            "region": row["region"],
            "direction": row["direction"],
            "definition": row["definition"],
            "coordinate_ranges_um": json.dumps(row["coordinate_bounds_um"]),
            "point_count": row["point_count"],
            "rmse_nm": row["rmse_nm"],
            "bias_nm": row["bias_nm"],
            "maximum_absolute_deviation_nm": row[
                "maximum_absolute_deviation_nm"
            ],
            "digitization_rms_nm": row["digitization_rms_nm"],
            "points_within_digitization_estimate": row[
                "points_within_digitization_estimate"
            ],
        }
        for row in profile_regions["agreement_1025"]
    ]
    _write_csv(
        output_directory / "profile_region_agreement_1025.csv",
        region_agreement_rows,
        list(region_agreement_rows[0]),
    )
    _write_csv(
        output_directory / "profile_region_grid_sensitivity.csv",
        profile_regions["adjacent_grid_sensitivity_at_digitized_coordinates"],
        list(
            profile_regions[
                "adjacent_grid_sensitivity_at_digitized_coordinates"
            ][0]
        ),
    )
    _write_csv(
        output_directory / "outlet_pressure_integrals.csv",
        outlet_integrals,
        list(outlet_integrals[0]),
    )
    (output_directory / "central_and_minimum_film_comparison.json").write_text(
        json.dumps(film_features, indent=2, allow_nan=False)
    )
    (output_directory / "figure13_profile_metrics_1025.json").write_text(
        json.dumps(finest_profiles, indent=2, allow_nan=False)
    )
    _plot_profiles(
        runs,
        digitized_rows,
        output_directory / "figure13_donor_profile_regions.png",
    )
    _plot_profile_residuals(
        profile_predictions,
        output_directory / "figure13_profile_residuals.png",
    )
    _plot_uncertainty(
        runs,
        root_metrics,
        pairs,
        output_directory / "donor_grid_sensitivity.png",
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--profiles",
        type=Path,
        default=Path("cases/ciulli_2008_figure13_s0_profiles.csv"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = analyze(args.mesh, args.profiles, args.output_dir)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
