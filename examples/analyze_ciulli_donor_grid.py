"""Reanalyze accepted Ciulli donor grids and compare them with Figure 13."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid
from scipy.interpolate import RegularGridInterpolator

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
    coordinates = np.linspace(*PROFILE_WINDOW_M, 401) * 1e-6
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
        },
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
        with args.regional_fields.open("wb") as stream:
            np.savez_compressed(stream, **regional["fields"])
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
