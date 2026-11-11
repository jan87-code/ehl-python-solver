"""Reanalyze accepted Ciulli donor grids and compare them with Figure 13."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid
from scipy.interpolate import RegularGridInterpolator

from ehl_solver.hertz import circular_point_contact
from ehl_solver.pressure_analysis import (
    centerline_pressure_profile,
    outlet_spike_diagnostics,
)

MAIN_X_M = (-100e-6, 60e-6)
MAIN_Y_M = (-75e-6, 75e-6)
OUTLET_WINDOW_M = (75e-6, 175e-6)
PROFILE_WINDOW_M = (-100e-6, 100e-6)
REGIONAL_WINDOW_M = (-100e-6, 175e-6, -75e-6, 75e-6)
REGION_BOUNDS_UM = {
    "full_contact": (-100.0, 100.0),
    "main_contact_excluding_spike": (-100.0, 60.0),
    "outlet_spike": (60.0, 100.0),
    "pressure_release": (100.0, 175.0),
}


def profile_error_metrics(observed_um, computed_um, digitization_uncertainty_um):
    observed = np.asarray(observed_um, dtype=np.float64)
    computed = np.asarray(computed_um, dtype=np.float64)
    uncertainty = np.asarray(digitization_uncertainty_um, dtype=np.float64)
    if (
        observed.ndim != 1
        or computed.shape != observed.shape
        or uncertainty.shape != observed.shape
        or observed.size == 0
        or not all(np.all(np.isfinite(v)) for v in (observed, computed, uncertainty))
        or np.any(uncertainty < 0)
    ):
        raise ValueError("finite equal-length profiles and nonnegative uncertainty required")
    error = computed - observed
    rmse = float(np.sqrt(np.mean(error**2)))
    scale_mean = float(np.mean(abs(observed)))
    scale_range = float(np.ptp(observed))
    return {
        "bias_um": float(np.mean(error)),
        "rmse_um": rmse,
        "maximum_absolute_error_um": float(np.max(abs(error))),
        "nrmse_mean_percent": 100 * rmse / scale_mean if scale_mean else None,
        "nrmse_range_percent": 100 * rmse / scale_range if scale_range else None,
        "digitization_rms_um": float(np.sqrt(np.mean(uncertainty**2))),
        "points_within_digitization_estimate": int(
            np.count_nonzero(
                abs(error) <= uncertainty + 10 * np.finfo(np.float64).eps
            )
        ),
        "point_count": int(observed.size),
    }


def _load_run(directory):
    directory = Path(directory)
    summary = json.loads((directory / "summary.json").read_text())
    if (
        summary.get("status") != "accepted_numerical_root"
        or summary.get("transport_fraction") != 0
        or summary.get("audit", {}).get("numerically_accepted") is not True
    ):
        raise ValueError(f"{directory} is not an accepted donor-flux root")
    state_path = directory / "state.npz"
    digest = hashlib.sha256(state_path.read_bytes()).hexdigest()
    if digest != summary.get("state_sha256"):
        raise ValueError(f"{directory} field checksum does not match its summary")
    with np.load(state_path, allow_pickle=False) as archive:
        fields = {name: archive[name].copy() for name in archive.files}
    x, y = fields.get("x_m"), fields.get("y_m")
    if (
        fields.get("film_m", np.empty(0)).shape != (len(y), len(x))
        or fields.get("pressure_pa", np.empty(0)).shape != (len(y), len(x))
    ):
        raise ValueError(f"{directory} fields do not match coordinates")
    return {
        "directory": str(directory),
        "case": summary["case"],
        "audit": summary["audit"],
        **fields,
    }


def _centerline_film(run):
    return np.array(
        [np.interp(0.0, run["y_m"], run["film_m"][:, i]) for i in range(len(run["x_m"]))]
    )


def _transverse_film(run):
    return np.array(
        [np.interp(0.0, run["x_m"], run["film_m"][j]) for j in range(len(run["y_m"]))]
    )


def _profile_comparison(run, rows, direction):
    selected = [row for row in rows if row["direction"] == direction]
    coordinate = np.array([float(row["coordinate_um"]) for row in selected]) * 1e-6
    observed = np.array([float(row["film_thickness_um"]) for row in selected])
    uncertainty = np.array([float(row["digitization_uncertainty_um"]) for row in selected])
    if direction == "longitudinal":
        predicted = np.interp(coordinate, run["x_m"], _centerline_film(run)) * 1e6
    else:
        predicted = np.interp(coordinate, run["y_m"], _transverse_film(run)) * 1e6
    return profile_error_metrics(observed, predicted, uncertainty)


def _sample_2d(run, field, xx, yy):
    points = np.column_stack((yy.ravel(), xx.ravel()))
    interpolator = RegularGridInterpolator((run["y_m"], run["x_m"]), field)
    return interpolator(points).reshape(xx.shape)


def _field_comparison(coarse, fine):
    x = np.linspace(*MAIN_X_M, 321)
    y = np.linspace(*MAIN_Y_M, 301)
    xx, yy = np.meshgrid(x, y)
    p0 = _sample_2d(coarse, coarse["pressure_pa"], xx, yy)
    p1 = _sample_2d(fine, fine["pressure_pa"], xx, yy)
    pressure_mask = (p0 > 1e3) | (p1 > 1e3)
    dp = p1 - p0
    h0 = _sample_2d(coarse, coarse["film_m"], xx, yy)
    h1 = _sample_2d(fine, fine["film_m"], xx, yy)
    dh = h1 - h0
    return {
        "common_region_x_um": [MAIN_X_M[0] * 1e6, MAIN_X_M[1] * 1e6],
        "common_region_y_um": [MAIN_Y_M[0] * 1e6, MAIN_Y_M[1] * 1e6],
        "pressure_relative_l2_over_positive_support": float(
            np.linalg.norm(dp[pressure_mask]) / np.linalg.norm(p0[pressure_mask])
        ),
        "pressure_linf_mpa": float(np.max(abs(dp[pressure_mask])) / 1e6),
        "pressure_rms_mpa": float(np.sqrt(np.mean(dp[pressure_mask] ** 2)) / 1e6),
        "pressure_support_mismatch_percent": float(
            100 * np.mean((p0 > 1e3) != (p1 > 1e3))
        ),
        "film_relative_l2": float(np.linalg.norm(dh) / np.linalg.norm(h0)),
        "film_linf_nm": float(np.max(abs(dh)) * 1e9),
        "film_rms_nm": float(np.sqrt(np.mean(dh**2)) * 1e9),
    }


def _midplane_comparison(coarse, fine):
    profile_comparisons = {}
    coordinates = np.linspace(*PROFILE_WINDOW_M, 401)
    for direction, coordinate_key, other_key in (
        ("longitudinal", "x_m", "y_m"),
        ("transverse", "y_m", "x_m"),
    ):
        index0 = int(np.argmin(abs(coarse[other_key])))
        index1 = int(np.argmin(abs(fine[other_key])))
        axis0, axis1 = coarse[coordinate_key], fine[coordinate_key]
        result = {}
        for field in ("pressure_pa", "film_m"):
            values0 = (
                coarse[field][index0, :]
                if direction == "longitudinal"
                else coarse[field][:, index0]
            )
            values1 = (
                fine[field][index1, :]
                if direction == "longitudinal"
                else fine[field][:, index1]
            )
            profile0 = np.interp(
                coordinates, axis0, values0
            )
            profile1 = np.interp(
                coordinates, axis1, values1
            )
            difference = profile1 - profile0
            mask = (
                (profile0 > 1e3) | (profile1 > 1e3)
                if field == "pressure_pa"
                else np.ones_like(profile0, dtype=bool)
            )
            unit_scale = 1e-6 if field == "pressure_pa" else 1e9
            result[field] = {
                "relative_l2": float(
                    np.linalg.norm(difference[mask]) / np.linalg.norm(profile0[mask])
                ),
                "rms": float(np.sqrt(np.mean(difference[mask] ** 2)) * unit_scale),
                "linf": float(np.max(abs(difference[mask])) * unit_scale),
                "units": "MPa" if field == "pressure_pa" else "nm",
            }
        profile_comparisons[direction] = result
    return profile_comparisons


def _integral(values, x, y):
    return float(trapezoid(trapezoid(values, x=x, axis=1), x=y, axis=0))


def _regional_metrics(coarse_p, fine_p, coarse_h, fine_h, x, y, bounds_um):
    x0, x1 = bounds_um
    region = (
        (x[None, :] >= x0 * 1e-6)
        & (x[None, :] <= x1 * 1e-6)
        & (y[:, None] >= REGIONAL_WINDOW_M[2])
        & (y[:, None] <= REGIONAL_WINDOW_M[3])
    )
    pressure_support = region & ((coarse_p > 1e3) | (fine_p > 1e3))
    dp, dh = fine_p - coarse_p, fine_h - coarse_h
    p_error_integral = _integral(dp**2 * pressure_support, x, y)
    p_reference_integral = _integral(coarse_p**2 * pressure_support, x, y)
    h_error_integral = _integral(dh**2 * region, x, y)
    h_reference_integral = _integral(coarse_h**2 * region, x, y)
    if not np.any(pressure_support) or not np.any(region):
        raise ValueError("Regional comparison window has no sampled support")
    p_values = np.where(pressure_support, abs(dp), -np.inf)
    h_values = np.where(region, abs(dh), -np.inf)
    p_j, p_i = np.unravel_index(int(np.argmax(p_values)), p_values.shape)
    h_j, h_i = np.unravel_index(int(np.argmax(h_values)), h_values.shape)
    p_area = _integral(pressure_support.astype(np.float64), x, y)
    area = _integral(region.astype(np.float64), x, y)
    return {
        "bounds_x_um": [x0, x1],
        "pressure_relative_l2": float(
            np.sqrt(p_error_integral / p_reference_integral)
        ),
        "pressure_rms_mpa_over_positive_support": float(
            np.sqrt(p_error_integral / p_area) / 1e6
        ),
        "pressure_linf_mpa": float(abs(dp[p_j, p_i]) / 1e6),
        "pressure_linf_xy_um": [float(x[p_i] * 1e6), float(y[p_j] * 1e6)],
        "positive_pressure_support_percent": float(100 * p_area / area),
        "pressure_support_mismatch_percent": float(
            100
            * _integral(((coarse_p > 1e3) != (fine_p > 1e3)) & region, x, y)
            / area
        ),
        "film_relative_l2": float(np.sqrt(h_error_integral / h_reference_integral)),
        "film_rms_nm": float(np.sqrt(h_error_integral / area) * 1e9),
        "film_linf_nm": float(abs(dh[h_j, h_i]) * 1e9),
        "film_linf_xy_um": [float(x[h_i] * 1e6), float(y[h_j] * 1e6)],
        "pressure_error_squared_integral_pa2_m2": p_error_integral,
    }


def regional_mesh_comparison(coarse, fine, *, spacing_um=0.5):
    """Compare nested accepted fields on common physical coordinates; no solve."""
    if (
        not np.isfinite(spacing_um)
        or spacing_um <= 0
        or coarse["case"]["domain_half_width_x_m"]
        != fine["case"]["domain_half_width_x_m"]
        or coarse["case"]["domain_half_width_y_m"]
        != fine["case"]["domain_half_width_y_m"]
        or coarse["case"]["nx"] >= fine["case"]["nx"]
        or coarse["case"]["ny"] >= fine["case"]["ny"]
    ):
        raise ValueError("Increasing grids on the same physical domain are required")
    x0, x1, y0, y1 = REGIONAL_WINDOW_M
    count_x = round((x1 - x0) * 1e6 / spacing_um) + 1
    count_y = round((y1 - y0) * 1e6 / spacing_um) + 1
    x = np.linspace(x0, x1, count_x)
    y = np.linspace(y0, y1, count_y)
    xx, yy = np.meshgrid(x, y)
    p0 = _sample_2d(coarse, coarse["pressure_pa"], xx, yy)
    p1 = _sample_2d(fine, fine["pressure_pa"], xx, yy)
    h0 = _sample_2d(coarse, coarse["film_m"], xx, yy)
    h1 = _sample_2d(fine, fine["film_m"], xx, yy)
    regions = {
        name: _regional_metrics(p0, p1, h0, h1, x, y, bounds)
        for name, bounds in REGION_BOUNDS_UM.items()
    }
    contact_error = regions["full_contact"][
        "pressure_error_squared_integral_pa2_m2"
    ]
    for metric in regions.values():
        metric["fraction_of_contact_pressure_error_squared_percent"] = (
            100 * metric["pressure_error_squared_integral_pa2_m2"] / contact_error
            if contact_error
            else None
        )
        metric.pop("pressure_error_squared_integral_pa2_m2")

    centerline_index = int(np.argmin(abs(y)))
    profile_mask = (x >= 60e-6) & (x <= 100e-6)
    profile_x = x[profile_mask]
    profile_coarse = p0[centerline_index, profile_mask]
    profile_fine = p1[centerline_index, profile_mask]
    shifts = np.arange(-12.5e-6, 12.5001e-6, 0.125e-6)
    best_shift = None
    for shift in shifts:
        shifted = np.interp(
            profile_x + shift,
            profile_x,
            profile_fine,
            left=np.nan,
            right=np.nan,
        )
        valid = np.isfinite(shifted)
        rms = float(np.sqrt(np.mean((shifted[valid] - profile_coarse[valid]) ** 2)))
        if best_shift is None or rms < best_shift["rms_pa"]:
            best_shift = {
                "translation_um": float(shift * 1e6),
                "rms_pa": rms,
                "valid_samples": int(np.count_nonzero(valid)),
            }
    peak_data = {}
    for label, run in (("coarse", coarse), ("fine", fine)):
        pressure_x, pressure = centerline_pressure_profile(
            run["x_m"], run["y_m"], run["pressure_pa"]
        )
        window_tolerance = 8 * np.finfo(np.float64).eps * 100e-6
        indices = np.flatnonzero(
            (pressure_x >= 75e-6 - window_tolerance)
            & (pressure_x <= 100e-6 + window_tolerance)
        )
        window_maximum = None
        if indices.size:
            maximum_index = indices[np.argmax(pressure[indices])]
            window_maximum = {
                "x_um": float(pressure_x[maximum_index] * 1e6),
                "pressure_mpa": float(pressure[maximum_index] / 1e6),
            }
        peak_data[label] = {
            "window_maximum": window_maximum,
            "strict_interior_peaks": [
                {
                    "x_um": float(pressure_x[i] * 1e6),
                    "pressure_mpa": float(pressure[i] / 1e6),
                }
                for i in indices
                if 0 < i < len(pressure) - 1
                and pressure[i] > pressure[i - 1]
                and pressure[i] > pressure[i + 1]
            ],
        }
    return {
        "coarse_grid": [coarse["case"]["ny"], coarse["case"]["nx"]],
        "fine_grid": [fine["case"]["ny"], fine["case"]["nx"]],
        "common_grid_spacing_um": spacing_um,
        "common_window_x_um": [x0 * 1e6, x1 * 1e6],
        "common_window_y_um": [y0 * 1e6, y1 * 1e6],
        "regions": regions,
        "contact_pressure_error_energy_share": {
            "main_contact_excluding_spike_percent": regions[
                "main_contact_excluding_spike"
            ]["fraction_of_contact_pressure_error_squared_percent"],
            "outlet_spike_percent": regions["outlet_spike"][
                "fraction_of_contact_pressure_error_squared_percent"
            ],
        },
        "outlet_centerline_profile": {
            "x_um": (profile_x * 1e6).tolist(),
            "coarse_pressure_mpa": (profile_coarse / 1e6).tolist(),
            "fine_pressure_mpa": (profile_fine / 1e6).tolist(),
            "raw_rms_mpa": float(
                np.sqrt(np.mean((profile_fine - profile_coarse) ** 2)) / 1e6
            ),
            "best_translation_diagnostic": {
                **best_shift,
                "rms_mpa": best_shift["rms_pa"] / 1e6,
                "not_peak_location_estimate": True,
            },
            "peak_detection": peak_data,
        },
        "fields": {
            "x_m": x,
            "y_m": y,
            "pressure_coarse_pa": p0,
            "pressure_fine_pa": p1,
            "pressure_difference_pa": p1 - p0,
            "film_coarse_m": h0,
            "film_fine_m": h1,
            "film_difference_m": h1 - h0,
            "theta_coarse": _sample_2d(coarse, coarse["theta"], xx, yy),
            "theta_fine": _sample_2d(fine, fine["theta"], xx, yy),
        },
    }


def main_contact_structure_comparison(coarse, fine, *, spacing_um=0.5):
    """Partition main contact by the independent dry-Hertz footprint."""
    regional = regional_mesh_comparison(coarse, fine, spacing_um=spacing_um)
    fields = regional["fields"]
    x, y = fields["x_m"], fields["y_m"]
    xx, yy = np.meshgrid(x, y)
    x0, x1 = REGION_BOUNDS_UM["main_contact_excluding_spike"]
    main = (
        (xx >= x0 * 1e-6)
        & (xx <= x1 * 1e-6)
        & (abs(yy) <= 75e-6)
    )
    case = coarse["case"]
    hertz_radius = circular_point_contact(
        case["load_n"], case["radius_m"], case["contact_modulus_pa"]
    ).radius_m
    core = main & (xx**2 + yy**2 <= hertz_radius**2)
    transition = main & ~core
    pressure_difference = fields["pressure_difference_pa"]
    film_difference = fields["film_difference_m"]
    p0, p1 = fields["pressure_coarse_pa"], fields["pressure_fine_pa"]
    h0 = fields["film_coarse_m"]
    pressure_support = (p0 > 1e3) | (p1 > 1e3)
    main_error_integral = _integral(
        pressure_difference**2 * main * pressure_support, x, y
    )
    regions = {}
    for name, mask in (("main_contact", main), ("hertz_core", core), ("transition", transition)):
        p_mask = mask & pressure_support
        p_error = _integral(pressure_difference**2 * p_mask, x, y)
        p_reference = _integral(p0**2 * p_mask, x, y)
        h_error = _integral(film_difference**2 * mask, x, y)
        h_reference = _integral(h0**2 * mask, x, y)
        area = _integral(mask.astype(np.float64), x, y)
        p_area = _integral(p_mask.astype(np.float64), x, y)
        if area <= 0 or p_area <= 0 or p_reference <= 0 or h_reference <= 0:
            raise ValueError(f"{name} mask has insufficient sampled area/support")
        p_abs = np.where(p_mask, abs(pressure_difference), -np.inf)
        h_abs = np.where(mask, abs(film_difference), -np.inf)
        pj, pi = np.unravel_index(int(np.argmax(p_abs)), p_abs.shape)
        hj, hi = np.unravel_index(int(np.argmax(h_abs)), h_abs.shape)
        regions[name] = {
            "area_fraction_of_main_percent": float(
                100 * area / _integral(main.astype(np.float64), x, y)
            ),
            "pressure_relative_l2": float(np.sqrt(p_error / p_reference)),
            "pressure_rms_mpa": float(np.sqrt(p_error / p_area) / 1e6),
            "pressure_linf_mpa": float(abs(pressure_difference[pj, pi]) / 1e6),
            "pressure_linf_xy_um": [float(x[pi] * 1e6), float(y[pj] * 1e6)],
            "pressure_error_energy_share_of_main_percent": float(
                100 * p_error / main_error_integral
            ),
            "film_relative_l2": float(np.sqrt(h_error / h_reference)),
            "film_rms_nm": float(np.sqrt(h_error / area) * 1e9),
            "film_linf_nm": float(abs(film_difference[hj, hi]) * 1e9),
            "film_linf_xy_um": [float(x[hi] * 1e6), float(y[hj] * 1e6)],
        }

    hotspot_x_m, hotspot_y_m = 50e-6, -56.5e-6
    hotspot_window = (
        main
        & (abs(xx - hotspot_x_m) <= 10e-6)
        & (abs(yy - hotspot_y_m) <= 10e-6)
    )
    hotspot_error = _integral(
        pressure_difference**2 * hotspot_window * pressure_support, x, y
    )
    cavitated0 = (fields["theta_coarse"] < 1 - 1e-10) & (p0 <= 1e3)
    cavitated1 = (fields["theta_fine"] < 1 - 1e-10) & (p1 <= 1e3)
    hotspot_metrics = regions["main_contact"]
    hotspot_metrics = {
        "center_um": [hotspot_x_m * 1e6, hotspot_y_m * 1e6],
        "half_width_um": 10.0,
        "pressure_error_energy_share_of_main_percent": float(
            100 * hotspot_error / main_error_integral
        ),
        "cavitation_state_mismatch_percent_in_main": float(
            100 * np.mean(cavitated0[main] != cavitated1[main])
        ),
        "cavitated_area_percent_65": float(100 * np.mean(cavitated0[main])),
        "cavitated_area_percent_129": float(100 * np.mean(cavitated1[main])),
        "theta_linf_difference_in_main": float(
            np.max(abs(fields["theta_fine"][main] - fields["theta_coarse"][main]))
        ),
        "max_pressure_difference_xy_um": hotspot_metrics["pressure_linf_xy_um"],
        "max_pressure_difference_mpa": hotspot_metrics["pressure_linf_mpa"],
    }

    ridge = []
    ridge_x = x[(x >= 0) & (x <= 60e-6)]
    for y_um in (-70.0, -65.0, -60.0, -56.5, -50.0, -45.0, -40.0, 0.0,
                 40.0, 45.0, 50.0, 56.5, 60.0, 65.0, 70.0):
        y_value = y_um * 1e-6
        entries = {"y_um": y_um}
        for label, run in (("65", coarse), ("129", fine)):
            profile = _sample_2d(
                run,
                run["pressure_pa"],
                ridge_x[None, :],
                np.full((1, len(ridge_x)), y_value),
            )[0]
            peak_index = int(np.argmax(profile))
            entries[f"x_peak_{label}_um"] = float(ridge_x[peak_index] * 1e6)
            entries[f"p_peak_{label}_mpa"] = float(profile[peak_index] / 1e6)
        ridge.append(entries)
    return {
        "hertz_contact_radius_um": float(hertz_radius * 1e6),
        "partition_definition": (
            "Core is the main-contact portion inside the dry Hertz radius; "
            "transition is the remainder of the declared main-contact rectangle."
        ),
        "regions": regions,
        "hotspot_and_cavitation": hotspot_metrics,
        "pressure_ridge_maxima_x_0_to_60_um_by_lateral_coordinate": ridge,
        "interpretation_note": (
            "The Hertz footprint is a geometric diagnostic partition, not a "
            "separate Reynolds or cavitation boundary."
        ),
    }


def _scalar_refinement(values):
    q65, q129, q257 = map(float, values)
    coarse_change, fine_change = q129 - q65, q257 - q129
    result = {
        "values_65_129_257": [q65, q129, q257],
        "change_65_to_129": coarse_change,
        "change_129_to_257": fine_change,
        "fine_to_coarse_change_ratio": (
            abs(fine_change / coarse_change) if coarse_change else None
        ),
        "monotone_contraction": bool(
            coarse_change * fine_change > 0
            and abs(fine_change) < abs(coarse_change)
        ),
        "observed_order": None,
        "richardson_extrapolated_value": None,
        "fine_grid_gci_percent_fs_1_25": None,
        "coarse_grid_gci_percent_fs_1_25": None,
        "asymptotic_gci_ratio": None,
    }
    if result["monotone_contraction"]:
        order = float(np.log(abs(coarse_change / fine_change)) / np.log(2.0))
        denominator = 2**order - 1
        coarse_gci = (
            1.25 * abs(coarse_change / q129) / denominator * 100 if q129 else None
        )
        fine_gci = (
            1.25 * abs(fine_change / q257) / denominator * 100 if q257 else None
        )
        result.update(
            observed_order=order,
            richardson_extrapolated_value=float(q257 + fine_change / denominator),
            coarse_grid_gci_percent_fs_1_25=coarse_gci,
            fine_grid_gci_percent_fs_1_25=fine_gci,
            asymptotic_gci_ratio=(
                float(fine_gci / (coarse_gci / 2**order))
                if coarse_gci and fine_gci
                else None
            ),
        )
    return result


def _outlet_integrated_pressure(run, spike):
    if not spike.get("identified"):
        return {
            "defined": False,
            "reason": spike.get("reason"),
        }
    p = run["pressure_pa"]
    peak_x = spike["peak_x_m"]
    width = spike["half_prominence_width_m"]
    a_h = circular_point_contact(
        run["case"]["load_n"],
        run["case"]["radius_m"],
        run["case"]["contact_modulus_pa"],
    ).radius_m
    integration_spacing = 0.25e-6
    x_sample = np.linspace(
        peak_x - width / 2,
        peak_x + width / 2,
        max(3, int(np.ceil(width / integration_spacing)) + 1),
    )
    y_sample = np.linspace(-a_h, a_h, int(np.ceil(2 * a_h / integration_spacing)) + 1)
    xx, yy = np.meshgrid(x_sample, y_sample)
    pressure_sample = _sample_2d(run, p, xx, yy)
    contribution_n = _integral(pressure_sample, x_sample, y_sample)
    centerline_x, centerline_pressure = centerline_pressure_profile(
        run["x_m"], run["y_m"], p
    )
    centerline_sample = np.interp(x_sample, centerline_x, centerline_pressure)
    local_baseline = np.interp(
        x_sample,
        [x_sample[0], x_sample[-1]],
        [centerline_sample[0], centerline_sample[-1]],
    )
    return {
        "defined": True,
        "x_window_um": [float((peak_x - width / 2) * 1e6), float((peak_x + width / 2) * 1e6)],
        "y_window_um": [-float(a_h * 1e6), float(a_h * 1e6)],
        "pressure_integral_n": contribution_n,
        "centerline_pressure_integral_pa_m": float(
            trapezoid(centerline_sample, x=x_sample)
        ),
        "centerline_excess_above_local_linear_baseline_pa_m": float(
            trapezoid(np.maximum(0.0, centerline_sample - local_baseline), x=x_sample)
        ),
        "interpolation_spacing_um": integration_spacing * 1e6,
        "interpretation": (
            "Integrated interpolated absolute pressure over the detected "
            "centerline half-prominence width and the Hertz-radius lateral span. "
            "The 1D excess integral subtracts the local linear endpoint baseline."
        ),
    }


def three_grid_convergence(mesh_runs):
    """Assess 65/129/257 nested-grid behavior without forcing extrapolation."""
    runs = sorted(mesh_runs, key=lambda run: run["case"]["nx"])
    if len(runs) != 3 or [run["case"]["nx"] for run in runs] != [65, 129, 257]:
        raise ValueError("Exactly the accepted 65x65, 129x129 and 257x257 roots are required")
    if any(
        run["case"]["ny"] != run["case"]["nx"]
        or run["case"]["transport"] != "donor"
        or run["case"]["domain_half_width_x_m"] != runs[0]["case"]["domain_half_width_x_m"]
        or run["case"]["domain_half_width_y_m"] != runs[0]["case"]["domain_half_width_y_m"]
        for run in runs
    ):
        raise ValueError("The three runs must be square donor grids on one physical domain")
    intervals = [run["case"]["nx"] - 1 for run in runs]
    if intervals[1] / intervals[0] != 2 or intervals[2] / intervals[1] != 2:
        raise ValueError("The nested grid refinement ratio must be two")
    values = {}
    for key, getter in (
        ("h_min_nm", lambda run: np.min(run["film_m"]) * 1e9),
        (
            "h_c_nm",
            lambda run: run["film_m"][len(run["y_m"]) // 2, len(run["x_m"]) // 2]
            * 1e9,
        ),
        ("p_max_mpa", lambda run: np.max(run["pressure_pa"]) / 1e6),
        ("integrated_load_n", lambda run: run["audit"]["force_n"]),
    ):
        values[key] = _scalar_refinement([getter(run) for run in runs])
    root_audits = []
    scalar_locations = []
    for run in runs:
        j_min, i_min = np.unravel_index(
            int(np.argmin(run["film_m"])), run["film_m"].shape
        )
        j_max, i_max = np.unravel_index(
            int(np.argmax(run["pressure_pa"])), run["pressure_pa"].shape
        )
        scalar_locations.append(
            {
                "grid": [run["case"]["ny"], run["case"]["nx"]],
                "h_min_xy_um": [
                    float(run["x_m"][i_min] * 1e6),
                    float(run["y_m"][j_min] * 1e6),
                ],
                "p_max_xy_um": [
                    float(run["x_m"][i_max] * 1e6),
                    float(run["y_m"][j_max] * 1e6),
                ],
            }
        )
        audit = run["audit"]
        root_audits.append(
            {
                "grid": [run["case"]["ny"], run["case"]["nx"]],
                "numerically_accepted": audit["numerically_accepted"],
                "force_n": audit["force_n"],
                "force_relative_error": audit["force_relative_error"],
                "relative_local_balance": audit["mass"]["relative_local_balance"],
                "relative_global_balance": audit["mass"]["relative_global_balance"],
                "maximum_si_mass_divergence_kg_m2_s": audit["mass"][
                    "maximum_si_mass_divergence_kg_m2_s"
                ],
                "maximum_complementarity": audit["mass"]["maximum_complementarity"],
                "film_reconstruction_error_m": audit["film_reconstruction_error_m"],
                "direct_elasticity_checked": audit["direct_elasticity_checked"],
            }
        )

    pairwise = []
    for coarse, fine in zip(runs, runs[1:], strict=False):
        fields = regional_mesh_comparison(coarse, fine)
        structure = main_contact_structure_comparison(coarse, fine)
        pairwise.append(
            {
                "comparison": f"{coarse['case']['nx']} -> {fine['case']['nx']}",
                "regions": fields["regions"],
                "hertz_core_partition": structure["regions"],
                "main_contact_profile_differences": _midplane_comparison(coarse, fine),
                "hotspot_and_cavitation": structure["hotspot_and_cavitation"],
            }
        )
    field_orders = {}
    for region_key in ("main_contact_excluding_spike", "outlet_spike", "full_contact"):
        for field_key in ("pressure_rms_mpa_over_positive_support", "film_rms_nm"):
            first = pairwise[0]["regions"][region_key][field_key]
            second = pairwise[1]["regions"][region_key][field_key]
            field_orders[f"{region_key}_{field_key}"] = {
                "difference_norms_65_to_129_and_129_to_257": [first, second],
                "contraction_ratio": float(first / second) if second else None,
                "observed_order_of_difference_norm": (
                    float(np.log(first / second) / np.log(2.0))
                    if first > second > 0
                    else None
                ),
                "field_richardson_gci_not_reported": True,
            }
    spikes = {}
    for run in runs:
        px, pp = centerline_pressure_profile(run["x_m"], run["y_m"], run["pressure_pa"])
        diag = outlet_spike_diagnostics(px, pp, window_m=OUTLET_WINDOW_M)
        spikes[f"{run['case']['nx']}x{run['case']['ny']}"] = {
            **diag,
            "integrated_pressure": _outlet_integrated_pressure(run, diag),
        }
    return {
        "grid_intervals": intervals,
        "refinement_ratio": 2.0,
        "same_domain_and_equation": True,
        "scalars": values,
        "scalar_locations": scalar_locations,
        "independent_root_audits": root_audits,
        "pairwise_field_comparisons": pairwise,
        "field_difference_norm_trends": field_orders,
        "outlet_spike": spikes,
        "interpretation": (
            "Scalar Richardson/GCI estimates are descriptive only when the "
            "corresponding signed changes contract monotonically. Field norm "
            "orders are contraction diagnostics, not field extrapolations."
        ),
    }


def analyze(mesh_directories, domain_directories, profile_csv):
    mesh = sorted((_load_run(path) for path in mesh_directories), key=lambda r: r["case"]["nx"])
    domain = sorted(
        (_load_run(path) for path in domain_directories),
        key=lambda r: r["case"]["domain_half_width_x_m"],
    )
    with Path(profile_csv).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    summary = {
        "equation": "donor-flux JFO/Elrod-Adams",
        "mesh_fixed_domain": {"roots": [], "adjacent_comparisons": []},
        "domain_fixed_spacing": {"roots": [], "adjacent_comparisons": []},
        "finest_mesh_experimental_profiles": {},
        "outlet_spike": {},
    }
    for label, sequence in (("mesh_fixed_domain", mesh), ("domain_fixed_spacing", domain)):
        for run in sequence:
            pmax = float(np.max(run["pressure_pa"]))
            j, i = np.unravel_index(np.argmax(run["pressure_pa"]), run["pressure_pa"].shape)
            center = len(run["x_m"]) // 2, len(run["y_m"]) // 2
            centerline_y = int(np.argmin(abs(run["y_m"])))
            cavitated = np.flatnonzero(
                (run["x_m"] >= 0)
                & (run["pressure_pa"][centerline_y] <= 1e3)
                & (run["theta"][centerline_y] < 1 - 1e-10)
            )
            summary[label]["roots"].append(
                {
                    "directory": run["directory"],
                    "grid_ny_nx": [len(run["y_m"]), len(run["x_m"])],
                    "half_width_mm": [
                        run["case"]["domain_half_width_x_m"] * 1e3,
                        run["case"]["domain_half_width_y_m"] * 1e3,
                    ],
                    "spacing_um": [
                        float(np.median(np.diff(run["x_m"])) * 1e6),
                        float(np.median(np.diff(run["y_m"])) * 1e6),
                    ],
                    "h_min_nm": float(np.min(run["film_m"]) * 1e9),
                    "h_c_nm": float(run["film_m"][center[1], center[0]] * 1e9),
                    "p_max_mpa": pmax / 1e6,
                    "p_max_xy_um": [float(run["x_m"][i] * 1e6), float(run["y_m"][j] * 1e6)],
                    "centerline_cavitation_start_um": (
                        float(run["x_m"][cavitated[0]] * 1e6) if cavitated.size else None
                    ),
                    "force_n": run["audit"]["force_n"],
                    "relative_local_balance": run["audit"]["mass"]["relative_local_balance"],
                    "relative_global_balance": run["audit"]["mass"]["relative_global_balance"],
                    "scaled_max_mass_residual": run["audit"]["mass"]["mass_maximum"],
                }
            )
        for left, right in zip(sequence, sequence[1:], strict=False):
            summary[label]["adjacent_comparisons"].append(
                {
                    "comparison": f"{left['case']['ny']} -> {right['case']['ny']}",
                    **_field_comparison(left, right),
                    "midplane_profiles": _midplane_comparison(left, right),
                }
            )
    finest = mesh[-1]
    summary["finest_mesh_experimental_profiles"] = {
        direction: _profile_comparison(finest, rows, direction)
        for direction in ("longitudinal", "transverse")
    }
    for run in mesh:
        pressure_x, pressure = centerline_pressure_profile(
            run["x_m"], run["y_m"], run["pressure_pa"]
        )
        summary["outlet_spike"][f"{run['case']['nx']}x{run['case']['ny']}"] = (
            outlet_spike_diagnostics(
                pressure_x, pressure, window_m=OUTLET_WINDOW_M
            )
        )
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", nargs=3, type=Path, required=True)
    parser.add_argument("--domain", nargs=3, type=Path, required=True)
    parser.add_argument(
        "--profiles", type=Path, default=Path("cases/ciulli_2008_figure13_s0_profiles.csv")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--regional-fields", type=Path)
    args = parser.parse_args(argv)
    result = analyze(args.mesh, args.domain, args.profiles)
    if args.regional_fields is not None:
        mesh_runs = sorted(
            (_load_run(path) for path in args.mesh), key=lambda r: r["case"]["nx"]
        )
        regional = regional_mesh_comparison(mesh_runs[1], mesh_runs[2])
        result["regional_mesh_diagnostics_65_to_129"] = {
            key: value for key, value in regional.items() if key != "fields"
        }
        result["main_contact_structure_65_to_129"] = (
            main_contact_structure_comparison(mesh_runs[1], mesh_runs[2])
        )
        with args.regional_fields.open("wb") as stream:
            np.savez_compressed(stream, **regional["fields"])
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
