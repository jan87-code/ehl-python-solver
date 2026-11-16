"""Compare saved 513/1025 Ciulli fields and partition outlet-feature changes."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid

from examples.analyze_ciulli_donor_grid import _load_run, _sample_2d
from examples.diagnose_ciulli_core_ridge import _outlet_feature_locus

_GRID_SIZES = (513, 1025)
_CONTACT_BOUNDS_UM = (-100.0, 100.0, -75.0, 75.0)
_TRACKED_ARC_HALF_WIDTH_UM = 59.375
_FIXED_WINDOW_HALF_WIDTH_UM = 5.0


def _integral(values, x_m, y_m):
    return float(trapezoid(trapezoid(values, x=x_m, axis=1), x=y_m, axis=0))


def _track_coordinates(locus, key):
    rows = sorted(
        (
            row
            for row in locus["pressure_maximum_locus"]
            if row.get(key) is not None
            and abs(row["y_um"]) <= _TRACKED_ARC_HALF_WIDTH_UM + 1e-9
        ),
        key=lambda row: row["y_um"],
    )
    if len(rows) < 2:
        raise ValueError(f"Outlet tracker produced insufficient {key} samples")
    return (
        np.array([row["y_um"] for row in rows], dtype=np.float64),
        np.array([row[key] for row in rows], dtype=np.float64),
    )


def feature_corridor_metrics(
    x_m,
    y_m,
    pressure_coarse,
    pressure_fine,
    film_coarse,
    film_fine,
    tracks,
    *,
    half_width_um,
):
    """Partition common-grid field changes around both tracked outlet loci."""
    fields = (pressure_coarse, pressure_fine, film_coarse, film_fine)
    if (
        x_m.ndim != 1
        or y_m.ndim != 1
        or any(field.shape != (y_m.size, x_m.size) for field in fields)
        or any(not np.all(np.isfinite(field)) for field in fields)
        or not np.isfinite(half_width_um)
        or half_width_um <= 0
    ):
        raise ValueError("finite matching fields and a positive corridor width are required")

    xx, yy = np.meshgrid(x_m, y_m)
    arc = np.abs(yy) <= _TRACKED_ARC_HALF_WIDTH_UM * 1e-6
    corridor = np.zeros_like(pressure_coarse, dtype=bool)
    for pressure_y, pressure_x, film_y, film_x in tracks:
        pressure_center = np.interp(
            y_m * 1e6, pressure_y, pressure_x, left=np.nan, right=np.nan
        ) * 1e-6
        film_center = np.interp(
            y_m * 1e6, film_y, film_x, left=np.nan, right=np.nan
        ) * 1e-6
        corridor |= arc & (
            (abs(xx - pressure_center[:, None]) <= half_width_um * 1e-6)
            | (abs(xx - film_center[:, None]) <= half_width_um * 1e-6)
        )

    dp = pressure_fine - pressure_coarse
    dh = film_fine - film_coarse
    support = (pressure_coarse > 1e3) | (pressure_fine > 1e3)
    inside, outside = corridor, ~corridor
    pressure_total = _integral(dp**2 * support, x_m, y_m)
    film_total = _integral(dh**2, x_m, y_m)
    pressure_inside = _integral(dp**2 * inside * support, x_m, y_m)
    pressure_outside = _integral(dp**2 * outside * support, x_m, y_m)
    film_inside = _integral(dh**2 * inside, x_m, y_m)
    film_outside = _integral(dh**2 * outside, x_m, y_m)
    inside_area = _integral(inside.astype(np.float64), x_m, y_m)
    outside_area = _integral(outside.astype(np.float64), x_m, y_m)
    inside_pressure_area = _integral(
        (inside & support).astype(np.float64), x_m, y_m
    )
    outside_pressure_area = _integral(
        (outside & support).astype(np.float64), x_m, y_m
    )
    pressure_reference_outside = _integral(
        pressure_coarse**2 * outside * support, x_m, y_m
    )
    film_reference_outside = _integral(film_coarse**2 * outside, x_m, y_m)
    return {
        "corridor_area_percent_contact": 100
        * inside_area
        / _integral(np.ones_like(inside, dtype=np.float64), x_m, y_m),
        "pressure_difference_energy_share_inside_percent": 100
        * pressure_inside
        / pressure_total,
        "pressure_difference_energy_share_outside_percent": 100
        * pressure_outside
        / pressure_total,
        "pressure_relative_l2_outside_percent": (
            100 * np.sqrt(pressure_outside / pressure_reference_outside)
            if pressure_reference_outside
            else None
        ),
        "pressure_rms_inside_mpa": (
            np.sqrt(pressure_inside / inside_pressure_area) / 1e6
            if inside_pressure_area
            else None
        ),
        "pressure_rms_outside_mpa": (
            np.sqrt(pressure_outside / outside_pressure_area) / 1e6
            if outside_pressure_area
            else None
        ),
        "film_difference_energy_share_inside_percent": 100
        * film_inside
        / film_total
        if film_total
        else 0.0,
        "film_relative_l2_outside_percent": (
            100 * np.sqrt(film_outside / film_reference_outside)
            if film_reference_outside
            else None
        ),
        "film_rms_inside_nm": (
            np.sqrt(film_inside / inside_area) * 1e9 if inside_area else None
        ),
        "film_rms_outside_nm": (
            np.sqrt(film_outside / outside_area) * 1e9 if outside_area else None
        ),
    }


def _point_at_y(locus, y_um):
    rows = locus["pressure_maximum_locus"]
    row = min(rows, key=lambda candidate: abs(candidate["y_um"] - y_um))
    if abs(row["y_um"] - y_um) > 1e-8:
        raise ValueError(f"Outlet tracker has no sample at y={y_um:g} um")
    window_width_um = (
        row["fixed_window_bounds_um"][1] - row["fixed_window_bounds_um"][0]
    )
    return {
        "y_um": row["y_um"],
        "pressure_peak_x_um": row["spike_x_um"],
        "pressure_peak_mpa": row["spike_pressure_mpa"],
        "half_prominence_width_um": row["half_prominence_width_um"],
        "cells_across_width": row["cells_across_width"],
        "fixed_window_bounds_um": row["fixed_window_bounds_um"],
        "fixed_window_pressure_integral_n_per_m": (
            row["fixed_window_pressure_integral_n_per_m"]
        ),
        "fixed_window_average_pressure_mpa": (
            row["fixed_window_pressure_integral_n_per_m"]
            / (window_width_um * 1e-6)
            / 1e6
        ),
        "film_constriction_x_um": row["film_minimum_x_um"],
        "film_constriction_minimum_nm": row["film_minimum_nm"],
    }


def _global_film_minimum(run):
    minimum = float(np.min(run["film_m"]))
    tolerance = 8 * np.finfo(np.float64).eps * abs(minimum)
    j_i = np.argwhere(abs(run["film_m"] - minimum) <= tolerance)
    return {
        "h_min_nm": minimum * 1e9,
        "locations_xy_um": [
            [float(run["x_m"][i] * 1e6), float(run["y_m"][j] * 1e6)]
            for j, i in j_i
        ],
    }


def analyze(coarse_directory, fine_directory, *, spacing_um=0.5):
    """Post-process accepted 513/1025 roots; no equation solve is performed."""
    runs = [_load_run(coarse_directory), _load_run(fine_directory)]
    if tuple(run["case"]["nx"] for run in runs) != _GRID_SIZES or any(
        run["case"]["ny"] != run["case"]["nx"]
        or run["case"]["transport"] != "donor"
        or run["case"]["domain_half_width_x_m"]
        != runs[0]["case"]["domain_half_width_x_m"]
        or run["case"]["domain_half_width_y_m"]
        != runs[0]["case"]["domain_half_width_y_m"]
        for run in runs
    ):
        raise ValueError("Accepted same-domain 513x513 and 1025x1025 donor roots are required")
    if not np.isfinite(spacing_um) or spacing_um <= 0:
        raise ValueError("Common-grid spacing must be finite and positive")

    loci = [_outlet_feature_locus(run) for run in runs]
    x0, x1, y0, y1 = _CONTACT_BOUNDS_UM
    nx = round((x1 - x0) / spacing_um) + 1
    ny = round((y1 - y0) / spacing_um) + 1
    x_m = np.linspace(x0 * 1e-6, x1 * 1e-6, nx)
    y_m = np.linspace(y0 * 1e-6, y1 * 1e-6, ny)
    xx, yy = np.meshgrid(x_m, y_m)
    common = [
        {
            name: _sample_2d(run, run[name], xx, yy)
            for name in ("pressure_pa", "film_m")
        }
        for run in runs
    ]
    pressure_coarse, pressure_fine = (
        fields["pressure_pa"] for fields in common
    )
    film_coarse, film_fine = (fields["film_m"] for fields in common)
    pressure_difference = pressure_fine - pressure_coarse
    film_difference = film_fine - film_coarse
    pressure_support = (pressure_coarse > 1e3) | (pressure_fine > 1e3)
    contact = np.ones_like(pressure_coarse, dtype=bool)
    tracks = []
    for locus in loci:
        tracks.append(
            (
                *_track_coordinates(locus, "spike_x_um"),
                *_track_coordinates(locus, "film_minimum_x_um"),
            )
        )
    full_pressure_error = _integral(
        pressure_difference**2 * pressure_support, x_m, y_m
    )
    full_film_error = _integral(film_difference**2, x_m, y_m)
    full_contact_area = _integral(contact.astype(np.float64), x_m, y_m)
    pressure_support_area = _integral(
        pressure_support.astype(np.float64), x_m, y_m
    )
    full_pressure_reference = _integral(
        pressure_coarse**2 * pressure_support, x_m, y_m
    )
    full_film_reference = _integral(film_coarse**2, x_m, y_m)
    corridor_sensitivity = {
        f"half_width_{half_width:g}_um": feature_corridor_metrics(
            x_m,
            y_m,
            pressure_coarse,
            pressure_fine,
            film_coarse,
            film_fine,
            tracks,
            half_width_um=half_width,
        )
        for half_width in (2.5, 5.0, 10.0)
    }

    ridge_integrals = {}
    for run, locus in zip(runs, loci, strict=True):
        rows = sorted(
            (
                row
                for row in locus["pressure_maximum_locus"]
                if row["pressure_peak_found"]
                and abs(row["y_um"]) <= _TRACKED_ARC_HALF_WIDTH_UM + 1e-9
            ),
            key=lambda row: row["y_um"],
        )
        row_y_m = np.array([row["y_um"] for row in rows]) * 1e-6
        integrals = np.array(
            [row["fixed_window_pressure_integral_n_per_m"] for row in rows]
        )
        widths_m = np.array(
            [
                row["fixed_window_bounds_um"][1]
                - row["fixed_window_bounds_um"][0]
                for row in rows
            ]
        ) * 1e-6
        integrated_pressure_n = float(trapezoid(integrals, x=row_y_m))
        window_area_m2 = float(trapezoid(widths_m, x=row_y_m))
        grid = run["case"]["nx"]
        ridge_integrals[f"{grid}x{grid}"] = {
            "y_span_um": [float(row_y_m[0] * 1e6), float(row_y_m[-1] * 1e6)],
            "fixed_window_width_um": 2 * _FIXED_WINDOW_HALF_WIDTH_UM,
            "integrated_pressure_n_over_arc": integrated_pressure_n,
            "area_mean_pressure_mpa": (
                integrated_pressure_n / window_area_m2 / 1e6
            ),
        }

    scalar_values = {}
    for run in runs:
        grid = run["case"]["nx"]
        center = run["film_m"][run["case"]["ny"] // 2, run["case"]["nx"] // 2]
        scalar_values[f"{grid}x{grid}"] = {
            "h_c_nm": float(center * 1e9),
            "global_film_minimum": _global_film_minimum(run),
            "p_max_mpa": float(np.max(run["pressure_pa"]) / 1e6),
        }
    scalar_changes = {}
    for key in ("h_c_nm", "p_max_mpa"):
        first = scalar_values["513x513"][key]
        second = scalar_values["1025x1025"][key]
        scalar_changes[key] = {
            "absolute_change": second - first,
            "relative_change_percent": 100 * (second - first) / first,
        }
    coarse_minimum = scalar_values["513x513"]["global_film_minimum"]["h_min_nm"]
    fine_minimum = scalar_values["1025x1025"]["global_film_minimum"]["h_min_nm"]
    scalar_changes["h_min_nm"] = {
        "absolute_change": fine_minimum - coarse_minimum,
        "relative_change_percent": (
            100 * (fine_minimum - coarse_minimum) / coarse_minimum
        ),
    }

    outlet_points = {}
    for run, locus in zip(runs, loci, strict=True):
        grid = run["case"]["nx"]
        outlet_points[f"{grid}x{grid}"] = {
            f"y_{y_um:g}_um": _point_at_y(locus, y_um)
            for y_um in (0.0, _TRACKED_ARC_HALF_WIDTH_UM)
        }
        outlet_points[f"{grid}x{grid}"]["connected_locus"] = (
            locus["continuity_from_centerline_to_global_off_axis_peak"]
        )

    return {
        "method": {
            "source_roots": [run["directory"] for run in runs],
            "grid_ny_nx": [[513, 513], [1025, 1025]],
            "common_coordinate_domain_um": {
                "x": [x0, x1],
                "y": [y0, y1],
            },
            "common_grid_spacing_um": spacing_um,
            "pressure_support": "p_513 > 1 kPa or p_1025 > 1 kPa",
            "feature_corridor": (
                "Union of fixed +/-half-width x bands around each grid's "
                "tracked pressure maximum and downstream film minimum, over "
                f"|y| <= {_TRACKED_ARC_HALF_WIDTH_UM:g} um; corridors from both "
                "grids are included to retain grid-shifted features."
            ),
            "outlet_width_definition": (
                "Existing cut-wise half-prominence width relative to the higher "
                "local valley within x=20..120 um."
            ),
            "pressure_window": (
                "Fixed +/-5 um x window centered on the tracked pressure "
                "maximum; integrals are kg-free pressure integrals [N/m] per "
                "transverse length."
            ),
            "no_solver_run": True,
        },
        "bulk_field_comparison": {
            "full_contact_pressure_relative_l2_percent": (
                100 * np.sqrt(full_pressure_error / full_pressure_reference)
            ),
            "full_contact_pressure_rms_mpa": (
                np.sqrt(full_pressure_error / pressure_support_area) / 1e6
            ),
            "full_contact_film_relative_l2_percent": (
                100 * np.sqrt(full_film_error / full_film_reference)
            ),
            "full_contact_film_rms_nm": (
                np.sqrt(full_film_error / full_contact_area) * 1e9
            ),
        },
        "feature_corridor_sensitivity": corridor_sensitivity,
        "scalar_values_by_grid": scalar_values,
        "scalar_changes": scalar_changes,
        "tracked_outlet_points": outlet_points,
        "fixed_window_integral_over_connected_arc": ridge_integrals,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse", type=Path, required=True)
    parser.add_argument("--fine", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spacing-um", type=float, default=0.5)
    args = parser.parse_args(argv)
    result = analyze(args.coarse, args.fine, spacing_um=args.spacing_um)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(f"Wrote saved-field outlet refinement analysis to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
