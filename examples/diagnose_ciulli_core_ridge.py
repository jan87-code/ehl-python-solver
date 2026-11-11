"""Diagnose the Ciulli donor core ridge from accepted saved grids; performs no solve."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import trapezoid
from scipy.interpolate import RegularGridInterpolator

from ehl_solver.elasticity import cell_integrated_kernel, elastic_deformation
from ehl_solver.prototype import PointContactCase
from ehl_solver.transport_candidate import donor_fluxes

MAIN_BOUNDS_M = (-100e-6, 60e-6, -75e-6, 75e-6)
RIDGE_BOUNDS_M = (35e-6, 60e-6, 45e-6, 75e-6)
CORE_RIDGE_X_RANGE_M = (25e-6, 75e-6)
CORE_RIDGE_Y_RANGE_M = (40e-6, 75e-6)
COMMON_SPACING_M = 0.5e-6


def _read_root(directory):
    directory = Path(directory)
    summary = json.loads((directory / "summary.json").read_text())
    state_path = directory / "state.npz"
    if (
        summary.get("status") != "accepted_numerical_root"
        or summary.get("transport_fraction") != 0
        or hashlib.sha256(state_path.read_bytes()).hexdigest()
        != summary.get("state_sha256")
    ):
        raise ValueError(f"{directory} is not a checksum-verified accepted donor root")
    with np.load(state_path, allow_pickle=False) as archive:
        fields = {key: archive[key].copy() for key in archive.files}
    run = {"directory": str(directory), "summary": summary, **fields}
    case = PointContactCase(**summary["case"])
    dx, dy = float(np.diff(run["x_m"])[0]), float(np.diff(run["y_m"])[0])
    if "deformation_m" not in run:
        run["deformation_m"] = elastic_deformation(
            run["pressure_pa"], dx, dy, case.contact_modulus_pa
        )
    if "density_kg_m3" not in run or "viscosity_pa_s" not in run:
        density, viscosity, _, _ = case.fluid().properties(run["pressure_pa"])
        run["density_kg_m3"] = density
        run["viscosity_pa_s"] = viscosity
    return run


def _relative_metrics(reference, comparison):
    difference = np.asarray(comparison) - np.asarray(reference)
    return {
        "relative_l2": float(np.linalg.norm(difference) / np.linalg.norm(reference)),
        "rms": float(np.sqrt(np.mean(difference**2))),
        "linf": float(np.max(np.abs(difference))),
    }


def _symmetry_metrics(run):
    fields = {
        "pressure_pa": run["pressure_pa"],
        "film_m": run["film_m"],
        "deformation_m": run["deformation_m"],
        "theta": run["theta"],
        "density_kg_m3": run["density_kg_m3"],
        "viscosity_pa_s": run["viscosity_pa_s"],
    }
    return {
        name: _relative_metrics(values, values[::-1, :])
        for name, values in fields.items()
    }


def _interpolate(run, field, xx, yy):
    return RegularGridInterpolator(
        (run["y_m"], run["x_m"]), field, method="linear"
    )(np.column_stack((yy.ravel(), xx.ravel()))).reshape(xx.shape)


def _integral(values, x, y):
    return float(trapezoid(trapezoid(values, x=x, axis=1), x=y, axis=0))


def _fixed_region_comparison(coarse, fine, spacing_m=COMMON_SPACING_M):
    x0, x1, y0, y1 = MAIN_BOUNDS_M
    x = np.arange(x0, x1 + spacing_m / 2, spacing_m)
    y = np.arange(y0, y1 + spacing_m / 2, spacing_m)
    xx, yy = np.meshgrid(x, y)
    p0 = _interpolate(coarse, coarse["pressure_pa"], xx, yy)
    p1 = _interpolate(fine, fine["pressure_pa"], xx, yy)
    h0 = _interpolate(coarse, coarse["film_m"], xx, yy)
    h1 = _interpolate(fine, fine["film_m"], xx, yy)
    main = (p0 > 1e3) | (p1 > 1e3)
    ridge = (
        (xx >= RIDGE_BOUNDS_M[0])
        & (xx <= RIDGE_BOUNDS_M[1])
        & (np.abs(yy) >= RIDGE_BOUNDS_M[2])
        & (np.abs(yy) <= RIDGE_BOUNDS_M[3])
    )
    results = {}
    for label, mask in (
        ("main", main),
        ("fixed_core_ridge", ridge & main),
        ("main_excluding_fixed_core_ridge", main & ~ridge),
    ):
        if not np.any(mask):
            raise ValueError(f"empty comparison region {label}")
        dp, dh = p1[mask] - p0[mask], h1[mask] - h0[mask]
        results[label] = {
            "sample_count": int(mask.sum()),
            "pressure_rms_mpa": float(np.sqrt(np.mean(dp**2)) / 1e6),
            "pressure_linf_mpa": float(np.max(np.abs(dp)) / 1e6),
            "pressure_relative_l2": float(
                np.linalg.norm(dp) / np.linalg.norm(p0[mask])
            ),
            "pressure_difference_energy_share_of_main_percent": (
                float(np.sum(dp**2) / np.sum((p1[main] - p0[main]) ** 2) * 100)
                if label != "main"
                else 100.0
            ),
            "film_rms_nm": float(np.sqrt(np.mean(dh**2)) * 1e9),
            "film_linf_nm": float(np.max(np.abs(dh)) * 1e9),
        }
    return {
        "grid_pair": f"{coarse['pressure_pa'].shape[1]}->{fine['pressure_pa'].shape[1]}",
        "bounds_main_um": [v * 1e6 for v in MAIN_BOUNDS_M],
        "bounds_fixed_ridge_um": [35.0, 60.0, 45.0, 75.0, -75.0, -45.0],
        "common_spacing_um": spacing_m * 1e6,
        "regions": results,
    }


def _quadratic_peak(run, x0, y0):
    x, y, p = run["x_m"], run["y_m"], run["pressure_pa"] / 1e6
    ix = int(np.argmin(abs(x - x0)))
    iy = int(np.argmin(abs(y - y0)))
    if ix < 1 or iy < 1 or ix + 1 >= len(x) or iy + 1 >= len(y):
        return {"valid": False, "reason": "peak is too close to an array edge"}
    xx, yy = np.meshgrid((x[ix - 1 : ix + 2] - x[ix]) * 1e6,
                         (y[iy - 1 : iy + 2] - y[iy]) * 1e6)
    values = p[iy - 1 : iy + 2, ix - 1 : ix + 2].ravel()
    design = np.column_stack(
        (np.ones(9), xx.ravel(), yy.ravel(), xx.ravel() ** 2,
         xx.ravel() * yy.ravel(), yy.ravel() ** 2)
    )
    a, b, c, d, e, f = np.linalg.lstsq(design, values, rcond=None)[0]
    hessian = np.array([[2 * d, e], [e, 2 * f]])
    if np.max(np.linalg.eigvalsh(hessian)) >= 0:
        return {"valid": False, "reason": "3x3 quadratic fit is not a strict maximum"}
    vertex = np.linalg.solve(hessian, -np.array([b, c]))
    if np.any(np.abs(vertex) > np.array([np.diff(x)[ix - 1] * 1e6,
                                        np.diff(y)[iy - 1] * 1e6])):
        return {"valid": False, "reason": "quadratic vertex leaves local nodal cell"}
    fitted_x = x[ix] * 1e6 + vertex[0]
    fitted_y = y[iy] * 1e6 + vertex[1]
    if not (
        RIDGE_BOUNDS_M[0] * 1e6 <= fitted_x <= RIDGE_BOUNDS_M[1] * 1e6
        and RIDGE_BOUNDS_M[2] * 1e6 <= abs(fitted_y) <= RIDGE_BOUNDS_M[3] * 1e6
    ):
        return {"valid": False, "reason": "quadratic vertex leaves fixed ridge region"}
    fitted = a + b * vertex[0] + c * vertex[1] + d * vertex[0] ** 2
    fitted += e * vertex[0] * vertex[1] + f * vertex[1] ** 2
    return {
        "valid": True,
        "x_um": float(fitted_x),
        "y_um": float(fitted_y),
        "pressure_mpa": float(fitted),
        "nodal_pressure_mpa": float(p[iy, ix]),
        "fit": "least-squares local quadratic on 3x3 nodes; no high-order interpolation",
    }


def _one_dimensional_peak_and_width(
    coordinate_m, values_pa, center_index, spacing_m, bounds_m
):
    start = int(np.argmin(abs(coordinate_m - bounds_m[0])))
    stop = int(np.argmin(abs(coordinate_m - bounds_m[1]))) + 1
    coordinate = coordinate_m[start:stop] * 1e6
    values = values_pa[start:stop] / 1e6
    if len(values) < 3:
        return {"valid": False, "reason": "insufficient local cut samples"}
    i = int(np.argmax(values))
    if i == 0 or i == len(values) - 1:
        return {"valid": False, "reason": "local cut maximum touches analysis-window edge"}
    quadratic = np.polyfit(coordinate[i - 1 : i + 2], values[i - 1 : i + 2], 2)
    if quadratic[0] >= 0:
        fit = {"valid": False, "reason": "local quadratic fit is not concave"}
    else:
        vertex = -quadratic[1] / (2 * quadratic[0])
        fit = (
            {"valid": True, "coordinate_um": float(vertex),
             "peak_mpa": float(np.polyval(quadratic, vertex))}
            if coordinate[i - 1] <= vertex <= coordinate[i + 1]
            else {"valid": False, "reason": "quadratic vertex leaves adjacent samples"}
        )
    baseline = 0.5 * (values[0] + values[-1])
    half_level = baseline + 0.5 * (values[i] - baseline)
    left_candidates = np.flatnonzero(values[:i] <= half_level)
    right_candidates = np.flatnonzero(values[i + 1 :] <= half_level)
    if not len(left_candidates) or not len(right_candidates):
        width = None
    else:
        left = int(left_candidates[-1])
        right = i + 1 + int(right_candidates[0])
        left_crossing = coordinate[left] + (
            (half_level - values[left])
            * (coordinate[left + 1] - coordinate[left])
            / (values[left + 1] - values[left])
        )
        right_crossing = coordinate[right - 1] + (
            (half_level - values[right - 1])
            * (coordinate[right] - coordinate[right - 1])
            / (values[right] - values[right - 1])
        )
        width = float(right_crossing - left_crossing)
    return {
        "low_order_quadratic_peak": fit,
        "local_baseline_mpa": float(baseline),
        "half_prominence_width_um": width,
        "cells_across_width": width / (spacing_m * 1e6) if width is not None else None,
        "definition": (
            "1-D cut across the fixed core-ridge band, half-prominence relative "
            "to its endpoint baseline; width is undefined when either crossing "
            "is absent in that band."
        ),
    }


def _contiguous_threshold_width(
    coordinate_m, values_pa, center_index, threshold_pa, bounds_m
):
    lower = int(np.argmin(abs(coordinate_m - bounds_m[0])))
    upper = int(np.argmin(abs(coordinate_m - bounds_m[1])))
    first = center_index
    last = center_index
    while first > lower and values_pa[first - 1] >= threshold_pa:
        first -= 1
    while last < upper and values_pa[last + 1] >= threshold_pa:
        last += 1
    cells = last - first
    return {
        "threshold_pa": float(threshold_pa),
        "contiguous_cells": int(cells),
        "width_um": float((coordinate_m[last] - coordinate_m[first]) * 1e6),
        "limits_um": [
            float(coordinate_m[first] * 1e6),
            float(coordinate_m[last] * 1e6),
        ],
        "truncated_by_analysis_band": bool(first == lower or last == upper),
    }


def _peak_sampling(run):
    p = run["pressure_pa"]
    j, i = np.unravel_index(np.argmax(p), p.shape)
    x, y = run["x_m"], run["y_m"]
    dx, dy = float(np.median(np.diff(x))), float(np.median(np.diff(y)))
    region = (
        (x[None, :] >= RIDGE_BOUNDS_M[0])
        & (x[None, :] <= RIDGE_BOUNDS_M[1])
        & (np.abs(y[:, None]) >= RIDGE_BOUNDS_M[2])
        & (np.abs(y[:, None]) <= RIDGE_BOUNDS_M[3])
    )
    branch_results = {}
    for side, side_mask in (
        ("upper", region & (y[:, None] > 0)),
        ("lower", region & (y[:, None] < 0)),
    ):
        masked = np.where(side_mask, p, -np.inf)
        jj, ii = np.unravel_index(np.argmax(masked), p.shape)
        branch_results[side] = {
            "x_um": float(x[ii] * 1e6),
            "y_um": float(y[jj] * 1e6),
            "nodal_pressure_mpa": float(p[jj, ii] / 1e6),
            "density_kg_m3": float(run["density_kg_m3"][jj, ii]),
            "viscosity_pa_s": float(run["viscosity_pa_s"][jj, ii]),
            "film_nm": float(run["film_m"][jj, ii] * 1e9),
            "theta": float(run["theta"][jj, ii]),
            "quadratic_peak": _quadratic_peak(run, x[ii], y[jj]),
            "width_x": _one_dimensional_peak_and_width(
                x, p[jj], ii, dx, CORE_RIDGE_X_RANGE_M
            ),
            "width_y": _one_dimensional_peak_and_width(
                y, p[:, ii], jj, dy,
                (RIDGE_BOUNDS_M[2], RIDGE_BOUNDS_M[3])
                if side == "upper"
                else (-RIDGE_BOUNDS_M[3], -RIDGE_BOUNDS_M[2]),
            ),
            "90_percent_local_peak_width_x": _contiguous_threshold_width(
                x, p[jj], ii, 0.9 * p[jj, ii],
                (RIDGE_BOUNDS_M[0], RIDGE_BOUNDS_M[1]),
            ),
            "90_percent_local_peak_width_y": _contiguous_threshold_width(
                y, p[:, ii], jj, 0.9 * p[jj, ii],
                (RIDGE_BOUNDS_M[2], RIDGE_BOUNDS_M[3])
                if side == "upper"
                else (-RIDGE_BOUNDS_M[3], -RIDGE_BOUNDS_M[2]),
            ),
            "five_by_five_nodes_above_95_percent_peak": int(
                np.count_nonzero(
                    p[max(0, jj - 2) : jj + 3, max(0, ii - 2) : ii + 3]
                    >= 0.95 * p[jj, ii]
                )
            ),
            "nine_by_nine_nodes_above_90_percent_peak": int(
                np.count_nonzero(
                    p[max(0, jj - 4) : jj + 5, max(0, ii - 4) : ii + 5]
                    >= 0.90 * p[jj, ii]
                )
            ),
        }
    return {
        "global_nodal_pmax_mpa": float(p[j, i] / 1e6),
        "global_nodal_pmax_xy_um": [float(x[i] * 1e6), float(y[j] * 1e6)],
        "grid_spacing_um": [dx * 1e6, dy * 1e6],
        "fixed_roi_peak_branches": branch_results,
    }


def _ridge_geometry(run):
    y_values = np.arange(40e-6, 67.5e-6 + 1e-12, 2.5e-6)
    x_values = np.linspace(*CORE_RIDGE_X_RANGE_M, 1001)
    rows = []
    for side in (-1, 1):
        for y in side * y_values:
            profile = _interpolate(
                run,
                run["pressure_pa"],
                x_values[None, :],
                np.full((1, len(x_values)), y),
            )[0]
            i = int(np.argmax(profile))
            rows.append((float(y * 1e6), float(x_values[i] * 1e6), float(profile[i] / 1e6)))
    positive = np.array([row for row in rows if row[0] > 0])
    coefficients = np.polyfit(positive[:, 0], positive[:, 1], 2)
    points = positive[:, :2]
    middle_y = float(np.mean(positive[:, 0]))
    slope = float(2 * coefficients[0] * middle_y + coefficients[1])
    curvature = float(
        2 * coefficients[0] / (1 + slope**2) ** 1.5
    )
    return {
        "definition": (
            "Maximum along x=[25,75] um for fixed y cuts in the fixed core band "
            "|y|=[40,75] um; not selected around each global pmax."
        ),
        "fixed_cut_ridge_points_y_x_pressure": rows,
        "positive_branch_quadratic_x_um_vs_y_um": coefficients.tolist(),
        "positive_branch_length_um": float(
            np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))
        ),
        "positive_branch_end_to_end_orientation_deg_from_x": float(np.degrees(
            np.arctan2(
                points[-1, 0] - points[0, 0],
                points[-1, 1] - points[0, 1],
            )
        )),
        "positive_branch_midpoint_curvature_per_um": curvature,
    }


def _fixed_ridge_statistics(run, common_x, common_y, spacing_m=0.5e-6):
    x = np.arange(35e-6, 60e-6 + spacing_m / 2, spacing_m)
    y = np.arange(-75e-6, 75e-6 + spacing_m / 2, spacing_m)
    xx, yy = np.meshgrid(x, y)
    p = _interpolate(run, run["pressure_pa"], xx, yy)
    mask = (np.abs(yy) >= 45e-6) & (np.abs(yy) <= 75e-6)
    values = p[mask]
    common_xx, common_yy = np.meshgrid(common_x, common_y)
    common_pressure = _interpolate(
        run, run["pressure_pa"], common_xx, common_yy
    )
    common_mask = (
        (common_xx >= 35e-6)
        & (common_xx <= 60e-6)
        & (np.abs(common_yy) >= 45e-6)
        & (np.abs(common_yy) <= 75e-6)
    )
    common_peak_j, common_peak_i = np.unravel_index(
        np.argmax(np.where(common_mask, common_pressure, -np.inf)),
        common_pressure.shape,
    )
    cell_average = 0.25 * (
        run["pressure_pa"][:-1, :-1]
        + run["pressure_pa"][1:, :-1]
        + run["pressure_pa"][:-1, 1:]
        + run["pressure_pa"][1:, 1:]
    )
    cell_x = 0.5 * (run["x_m"][:-1] + run["x_m"][1:])
    cell_y = 0.5 * (run["y_m"][:-1] + run["y_m"][1:])
    cell_region = (
        (cell_x[None, :] >= 35e-6)
        & (cell_x[None, :] <= 60e-6)
        & (np.abs(cell_y[:, None]) >= 45e-6)
        & (np.abs(cell_y[:, None]) <= 75e-6)
    )
    weights = np.zeros_like(p)
    weights[mask] = 1.0
    small_boxes = {}
    for label, y_bounds in (
        ("upper", (50e-6, 65e-6)),
        ("lower", (-65e-6, -50e-6)),
    ):
        sx = np.arange(45e-6, 60e-6 + spacing_m / 2, spacing_m)
        sy = np.arange(y_bounds[0], y_bounds[1] + spacing_m / 2, spacing_m)
        sxx, syy = np.meshgrid(sx, sy)
        samples = _interpolate(run, run["pressure_pa"], sxx, syy)
        small_boxes[label] = {
            "bounds_um": {
                "x": [45.0, 60.0],
                "y": [value * 1e6 for value in y_bounds],
            },
            "mean_pressure_mpa": float(np.mean(samples) / 1e6),
            "0_5um_sample_grid_max_pressure_mpa": float(np.max(samples) / 1e6),
            "0_5um_sample_grid_max_xy_um": [
                float(sx[np.unravel_index(np.argmax(samples), samples.shape)[1]] * 1e6),
                float(sy[np.unravel_index(np.argmax(samples), samples.shape)[0]] * 1e6),
            ],
            "integrated_pressure_n": _integral(samples, sx, sy),
        }
    return {
        "bounds_um": {
            "x": [35.0, 60.0],
            "upper_y": [45.0, 75.0],
            "lower_y": [-75.0, -45.0],
        },
        "mean_pressure_mpa": float(np.mean(values) / 1e6),
        "peak_pressure_mpa": float(np.max(values) / 1e6),
        "common_513_grid_max_pressure_mpa": float(
            common_pressure[common_peak_j, common_peak_i] / 1e6
        ),
        "common_513_grid_max_xy_um": [
            float(common_x[common_peak_i] * 1e6),
            float(common_y[common_peak_j] * 1e6),
        ],
        "common_grid_interpolation": (
            "Piecewise bilinear interpolation from each native field onto the "
            "accepted 513x513 nodal coordinates; all four grids are nested."
        ),
        "cell_average_peak_pressure_mpa": float(np.max(cell_average[cell_region]) / 1e6),
        "integrated_pressure_n": _integral(p * weights, x, y),
        "area_um2": float(mask.sum() * spacing_m**2 * 1e12),
        "subcell_spacing_um": spacing_m * 1e6,
        "small_fixed_neighborhoods": small_boxes,
    }


def _flux_stencil(run, case, x_target=52e-6, y_target=59.375e-6):
    p, h, theta = run["pressure_pa"], run["film_m"], run["theta"]
    rho, eta = run["density_kg_m3"], run["viscosity_pa_s"]
    x, y = run["x_m"], run["y_m"]
    speed = case.entrainment_m_s
    dx, dy = float(np.diff(x)[0]), float(np.diff(y)[0])
    mobility = rho * h**3 / (12 * eta)
    qx_diff = -0.5 * (mobility[:, 1:] + mobility[:, :-1]) * np.diff(p, axis=1) / dx
    qx_adv = speed * (rho * h * theta)[:, :-1]
    qx, qy = donor_fluxes(x, y, h, p, theta, case.fluid(), speed)
    i, j = int(np.argmin(abs(x - x_target))), int(np.argmin(abs(y - y_target)))
    divergence = (
        np.diff(qx[1:-1], axis=1) / (0.5 * (x[2:] - x[:-2]))
        + np.diff(qy[:, 1:-1], axis=0) / (0.5 * (y[2:] - y[:-2]))[:, None]
    )
    residual_j, residual_i = j - 1, i - 1
    j_slice = slice(j - 1, j + 2)
    i_slice = slice(i - 1, i + 2)
    local_divergence = divergence[
        residual_j - 1 : residual_j + 2,
        residual_i - 1 : residual_i + 2,
    ]
    return {
        "sample_xy_um": [float(x[i] * 1e6), float(y[j] * 1e6)],
        "center": {
            "pressure_mpa": float(p[j, i] / 1e6),
            "film_nm": float(h[j, i] * 1e9),
            "density_kg_m3": float(rho[j, i]),
            "viscosity_pa_s": float(eta[j, i]),
            "mobility_m_s": float(mobility[j, i]),
            "theta": float(theta[j, i]),
        },
        "x_west_face": {
            "pressure_gradient_flux_kg_m_s": float(qx_diff[j, i - 1]),
            "donor_couette_flux_kg_m_s": float(qx_adv[j, i - 1]),
            "total_flux_kg_m_s": float(qx[j, i - 1]),
            "upwind_node_x_um": float(x[i - 1] * 1e6),
            "upwind_theta": float(theta[j, i - 1]),
            "pressure_gradient_pa_m": float((p[j, i] - p[j, i - 1]) / dx),
        },
        "x_east_face": {
            "pressure_gradient_flux_kg_m_s": float(qx_diff[j, i]),
            "donor_couette_flux_kg_m_s": float(qx_adv[j, i]),
            "total_flux_kg_m_s": float(qx[j, i]),
            "upwind_node_x_um": float(x[i] * 1e6),
            "upwind_theta": float(theta[j, i]),
            "pressure_gradient_pa_m": float((p[j, i + 1] - p[j, i]) / dx),
        },
        "y_south_total_flux_kg_m_s": float(qy[j - 1, i]),
        "y_north_total_flux_kg_m_s": float(qy[j, i]),
        "y_south_pressure_gradient_flux_kg_m_s": float(
            -0.5 * (mobility[j - 1, i] + mobility[j, i])
            * (p[j, i] - p[j - 1, i]) / dy
        ),
        "y_north_pressure_gradient_flux_kg_m_s": float(
            -0.5 * (mobility[j, i] + mobility[j + 1, i])
            * (p[j + 1, i] - p[j, i]) / dy
        ),
        "y_south_pressure_gradient_pa_m": float((p[j, i] - p[j - 1, i]) / dy),
        "y_north_pressure_gradient_pa_m": float((p[j + 1, i] - p[j, i]) / dy),
        "cell_divergence_kg_m2_s": float(
            (qx[j, i] - qx[j, i - 1]) / dx
            + (qy[j, i] - qy[j - 1, i]) / dy
        ),
        "neighboring_3x3_cell_divergence_kg_m2_s": divergence[
            residual_j - 1 : residual_j + 2,
            residual_i - 1 : residual_i + 2,
        ].tolist(),
        "neighboring_3x3_cells": {
            "x_nodes_um": (x[i - 1 : i + 2] * 1e6).tolist(),
            "y_nodes_um": (y[j - 1 : j + 2] * 1e6).tolist(),
            "pressure_mpa": (p[j_slice, i_slice] / 1e6).tolist(),
            "film_nm": (h[j_slice, i_slice] * 1e9).tolist(),
            "density_kg_m3": rho[j_slice, i_slice].tolist(),
            "viscosity_pa_s": eta[j_slice, i_slice].tolist(),
            "mobility_m_s": mobility[j_slice, i_slice].tolist(),
            "theta": theta[j_slice, i_slice].tolist(),
            "residual_kg_m2_s": local_divergence.tolist(),
            "x_west_pressure_gradient_flux_kg_m_s": qx_diff[
                j_slice, i - 2 : i + 1
            ].tolist(),
            "x_east_pressure_gradient_flux_kg_m_s": qx_diff[
                j_slice, i - 1 : i + 2
            ].tolist(),
            "x_west_couette_flux_kg_m_s": qx_adv[
                j_slice, i - 2 : i + 1
            ].tolist(),
            "x_east_couette_flux_kg_m_s": qx_adv[
                j_slice, i - 1 : i + 2
            ].tolist(),
            "x_west_total_flux_kg_m_s": qx[
                j_slice, i - 2 : i + 1
            ].tolist(),
            "x_east_total_flux_kg_m_s": qx[
                j_slice, i - 1 : i + 2
            ].tolist(),
            "x_west_donor_node_um": (x[i - 2 : i + 1] * 1e6).tolist(),
            "x_east_donor_node_um": (x[i - 1 : i + 2] * 1e6).tolist(),
            "y_south_total_flux_kg_m_s": qy[
                j - 2 : j + 1, i_slice
            ].tolist(),
            "y_north_total_flux_kg_m_s": qy[
                j - 1 : j + 2, i_slice
            ].tolist(),
        },
        "interpretation": (
            "The donor advection is intentionally first-order upwind from the "
            "west node; pressure mobility is arithmetic at faces. theta=1 "
            "means no cavitation/donor branch switch is active at this sample. "
            "No coefficient clipping or limiter is used by this donor stencil."
        ),
    }


def _manufactured_flux_order(case, grids=(65, 129, 257, 513)):
    bounds = 0.4e-3
    length = 2 * bounds
    k = 2 * np.pi / length
    outputs = []
    for n in grids:
        x = np.linspace(-bounds, bounds, n)
        y = x.copy()
        xx, yy = np.meshgrid(x, y)
        cx, cy = np.cos(k * xx), np.cos(k * yy)
        sx, sy = np.sin(k * xx), np.sin(k * yy)
        pressure = 3e8 + 1e8 * cx * cy
        px, py = -1e8 * k * sx * cy, -1e8 * k * cx * sy
        lap_p = -1e8 * k**2 * (cx * cy + cx * cy)
        film = 0.6e-6 * (1 + 0.12 * np.sin(k * xx + 0.2) * np.cos(k * yy - 0.1))
        hx = 0.6e-6 * 0.12 * k * np.cos(k * xx + 0.2) * np.cos(k * yy - 0.1)
        hy = -0.6e-6 * 0.12 * k * np.sin(k * xx + 0.2) * np.sin(k * yy - 0.1)
        theta = np.ones_like(pressure)
        rho, eta, drho, deta = case.fluid().properties(pressure)
        mobility = rho * film**3 / (12 * eta)
        mobility_p = mobility * (drho / rho - deta / eta)
        mobility_x = mobility_p * px + 3 * mobility * hx / film
        mobility_y = mobility_p * py + 3 * mobility * hy / film
        exact_diff = -(
            mobility_x * px + mobility_y * py + mobility * lap_p
        )
        exact_adv = case.entrainment_m_s * (drho * px * film + rho * hx)
        exact = exact_diff + exact_adv
        qx, qy = donor_fluxes(x, y, film, pressure, theta, case.fluid(),
                              case.entrainment_m_s)
        numerical = (
            np.diff(qx[1:-1], axis=1) / np.diff(x)[0]
            + np.diff(qy[:, 1:-1], axis=0) / np.diff(y)[0]
        )
        mobility = rho * film**3 / (12 * eta)
        qx_diff = (
            -0.5 * (mobility[:, 1:] + mobility[:, :-1])
            * np.diff(pressure, axis=1) / np.diff(x)[0]
        )
        qx_adv = case.entrainment_m_s * (rho * film)[:, :-1]
        qy_diff = (
            -0.5 * (mobility[1:] + mobility[:-1])
            * np.diff(pressure, axis=0) / np.diff(y)[:, None]
        )
        numerical_diff = (
            np.diff(qx_diff[1:-1], axis=1) / np.diff(x)[0]
            + np.diff(qy_diff[:, 1:-1], axis=0) / np.diff(y)[0]
        )
        numerical_adv = np.diff(qx_adv[1:-1], axis=1) / np.diff(x)[0]
        reference = exact[1:-1, 1:-1]
        error = numerical - reference
        outputs.append({
            "grid": n,
            "relative_l2": float(np.linalg.norm(error) / np.linalg.norm(reference)),
            "rms_kg_m2_s": float(np.sqrt(np.mean(error**2))),
            "diffusion_relative_l2": float(
                np.linalg.norm(numerical_diff - exact_diff[1:-1, 1:-1])
                / np.linalg.norm(exact_diff[1:-1, 1:-1])
            ),
            "advection_relative_l2": float(
                np.linalg.norm(numerical_adv - exact_adv[1:-1, 1:-1])
                / np.linalg.norm(exact_adv[1:-1, 1:-1])
            ),
            "smooth_full_film": True,
            "transport_is_first_order_donor": True,
        })
    for left, right in zip(outputs, outputs[1:], strict=False):
        left["observed_order_to_next_grid"] = float(
            np.log(left["rms_kg_m2_s"] / right["rms_kg_m2_s"]) / np.log(2)
        )
        left["diffusion_observed_order"] = float(
            np.log(left["diffusion_relative_l2"] / right["diffusion_relative_l2"])
            / np.log(2)
        )
        left["advection_observed_order"] = float(
            np.log(left["advection_relative_l2"] / right["advection_relative_l2"])
            / np.log(2)
        )
    return {
        "definition": (
            "Exact continuous divergence of q=-rho*h^3/(12 eta)*grad(p) "
            "+ U*rho*h e_x, evaluated against the production donor face flux "
            "on a smooth positive fully flooded manufactured field."
        ),
        "results": outputs,
    }


def _elasticity_synthetic(case, grids=(65, 129, 257, 513)):
    half_width = 0.4e-3
    gaussian_width = 100e-6
    peak_pressure = 100e6
    exact_center = peak_pressure * gaussian_width * np.sqrt(np.pi) / case.contact_modulus_pa
    results = []
    for n in grids:
        axis = np.linspace(-half_width, half_width, n)
        xx, yy = np.meshgrid(axis, axis)
        pressure = peak_pressure * np.exp(
            -(xx**2 + yy**2) / gaussian_width**2
        )
        displacement = elastic_deformation(
            pressure, float(axis[1] - axis[0]), float(axis[1] - axis[0]),
            case.contact_modulus_pa,
        )
        center = float(displacement[n // 2, n // 2])
        mirror_error = (
            np.linalg.norm(displacement - displacement[::-1])
            / np.linalg.norm(displacement)
        )
        results.append({
            "grid": n,
            "center_displacement_m": center,
            "analytic_infinite_gaussian_center_m": exact_center,
            "relative_center_error": abs(center - exact_center) / exact_center,
            "positive_for_positive_pressure": bool(np.min(displacement) >= 0),
            "transverse_symmetry_relative_l2": float(mirror_error),
        })
    for left, right in zip(results, results[1:], strict=False):
        left["observed_error_order_to_next_grid"] = float(
            np.log(left["relative_center_error"] / right["relative_center_error"]) / np.log(2)
        )
    small_n = 129
    axis = np.linspace(-half_width, half_width, small_n)
    xx, yy = np.meshgrid(axis, axis)
    gaussian = peak_pressure * np.exp(-(xx**2 + yy**2) / gaussian_width**2)
    shift = 7
    shifted = np.zeros_like(gaussian)
    shifted[:, shift:] = gaussian[:, :-shift]
    step = float(axis[1] - axis[0])
    original_response = elastic_deformation(
        gaussian, step, step, case.contact_modulus_pa
    )
    doubled_response = elastic_deformation(
        2 * gaussian, step, step, case.contact_modulus_pa
    )
    shifted_response = elastic_deformation(
        shifted, step, step, case.contact_modulus_pa
    )
    overlap = (
        original_response[10:-10, 10:-10]
    )
    translated_reference = shifted_response[10:-10, 10 + shift : -10 + shift]
    kernel = cell_integrated_kernel((1, 2), step, step, case.contact_modulus_pa)
    return {
        "kernel": (
            "Exact rectangular source-cell integral of 1/(pi E* r), including "
            "analytic self-cell integral; zero-padded linear FFT and centered crop."
        ),
        "interpretation": (
            "Gaussian center has analytic infinite-domain reference; the omitted "
            "finite-domain tail is exp(-16)."
        ),
        "results": results,
        "self_cell_coefficient_m_pa": float(kernel[0, 1]),
        "nearest_x_cell_coefficient_m_pa": float(kernel[0, 2]),
        "nearest_to_self_coefficient_ratio": float(kernel[0, 2] / kernel[0, 1]),
        "translation_relative_l2_away_from_boundaries": float(
            np.linalg.norm(translated_reference - overlap) / np.linalg.norm(overlap)
        ),
        "pressure_load_scaling_relative_l2": float(
            np.linalg.norm(doubled_response - 2 * original_response)
            / np.linalg.norm(2 * original_response)
        ),
    }


def _checkpoint_history(run):
    directory = Path(run["directory"])
    rows = []
    previous = None
    previous_cavitated = None
    x_bounds = np.array([35e-6, 60e-6])
    y_bounds = np.array([45e-6, 75e-6])
    for path in sorted((directory / "checkpoints").glob("checkpoint_*.npz")):
        with np.load(path, allow_pickle=False) as a:
            p, x, y = a["pressure_pa"], a["x_m"], a["y_m"]
            theta = a["theta"]
            fixed_roi = (
                (x[None, :] >= x_bounds[0])
                & (x[None, :] <= x_bounds[1])
                & (np.abs(y[:, None]) >= y_bounds[0])
                & (np.abs(y[:, None]) <= y_bounds[1])
            )
            state = {
                "iteration": int(path.stem.split("_")[-1]),
                "pmax_mpa": float(np.max(p) / 1e6),
                "pmax_xy_um": [
                    float(x[np.argmax(p) % len(x)] * 1e6),
                    float(y[np.argmax(p) // len(x)] * 1e6),
                ],
                "theta_min": float(np.min(theta)),
                "theta_less_than_one_cells": int(np.count_nonzero(theta < 1 - 1e-10)),
            }
            for side, side_mask in (
                ("upper", fixed_roi & (y[:, None] > 0)),
                ("lower", fixed_roi & (y[:, None] < 0)),
            ):
                masked = np.where(side_mask, p, -np.inf)
                ridge_j, ridge_i = np.unravel_index(np.argmax(masked), p.shape)
                state[f"{side}_ridge_peak_mpa"] = float(p[ridge_j, ridge_i] / 1e6)
                state[f"{side}_ridge_peak_xy_um"] = [
                    float(x[ridge_i] * 1e6),
                    float(y[ridge_j] * 1e6),
                ]
            if previous is not None:
                correction = p - previous
                state["pressure_change_rms_mpa"] = float(
                    np.sqrt(np.mean(correction**2)) / 1e6
                )
                state["pressure_correction_l2_mpa"] = float(
                    np.linalg.norm(correction) / 1e6
                )
                state["pressure_correction_linf_mpa"] = float(
                    np.max(np.abs(correction)) / 1e6
                )
            cavitated = theta < 1 - 1e-10
            if previous_cavitated is not None:
                state["cavitation_state_changed_nodes"] = int(
                    np.count_nonzero(cavitated != previous_cavitated)
                )
            rows.append(state)
            previous = p
            previous_cavitated = cavitated
    records = []
    for path in sorted((directory / "diagnostics").glob("iteration_*.json")):
        record = json.loads(path.read_text())
        records.append({
            "iteration": record["iteration"],
            "accepted_alpha": record["accepted_alpha"],
            "nonlinear_residual_linf": record["nonlinear_residual_linf"],
            "accepted_merit": record["accepted_merit"],
            "models": len(record["models"]),
            "accepted_recovery": record["accepted_recovery"],
            "linear_models": [
                {
                    "recovery": model["recovery"],
                    "status": model["status"],
                    "active_full_count": model["active_full_count"],
                    "active_full_size": model["active_full_size"],
                    "active_set_changed_nodes": model["active_set_changed_nodes"],
                    "linear_iterations": model["linear_iterations"],
                    "linear_true_residual": model["linear_true_residual"],
                    "linear_true_residuals": model["linear_true_residuals"],
                    "correction_norm_l2": model["correction_norm_l2"],
                    "correction_norm_linf": model["correction_norm_linf"],
                    "accepted_correction_norm_l2": model[
                        "accepted_correction_norm_l2"
                    ],
                    "accepted_correction_norm_linf": model[
                        "accepted_correction_norm_linf"
                    ],
                    "trials": [
                        {
                            "stage": trial["stage"],
                            "alpha": trial["alpha"],
                            "accepted": trial["accepted"],
                            "reason": trial["reason"],
                        }
                        for trial in model["trials"]
                    ],
                }
                for model in record["models"]
            ],
        })
    return {"checkpoint_fields": rows, "newton_records": records}


def diagnose(root_paths):
    runs = sorted((_read_root(path) for path in root_paths),
                  key=lambda run: run["pressure_pa"].shape[1])
    if [run["pressure_pa"].shape[1] for run in runs] != [65, 129, 257, 513]:
        raise ValueError("Provide the accepted 65, 129, 257 and 513 donor roots")
    case = PointContactCase(**runs[-1]["summary"]["case"])
    symmetry, peaks, ridge, fixed, flux = {}, {}, {}, {}, {}
    for run in runs:
        n = run["pressure_pa"].shape[1]
        symmetry[str(n)] = _symmetry_metrics(run)
        peaks[str(n)] = _peak_sampling(run)
        ridge[str(n)] = _ridge_geometry(run)
        fixed[str(n)] = _fixed_ridge_statistics(
            run, runs[-1]["x_m"], runs[-1]["y_m"]
        )
        upper_peak = peaks[str(n)]["fixed_roi_peak_branches"]["upper"]
        flux[str(n)] = {
            "at_grid_specific_upper_fixed_roi_peak": _flux_stencil(
                run, case, upper_peak["x_um"] * 1e-6, upper_peak["y_um"] * 1e-6
            ),
            "at_fixed_common_coordinate_52_59_375_um": _flux_stencil(run, case),
        }
    comparisons = [
        _fixed_region_comparison(a, b) for a, b in zip(runs, runs[1:], strict=False)
    ]
    return {
        "case": "15 N Ciulli donor-flux, fixed +/-0.4 mm domain",
        "diagnostic_runtime": {
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "thread_environment": {
                name: os.environ.get(name)
                for name in (
                    "OPENBLAS_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "BLIS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                )
            },
            "note": (
                "Post-processing only; SciPy FFT uses its default single-worker "
                "path. No EHL solve was run."
            ),
        },
        "source_roots": [run["directory"] for run in runs],
        "transverse_symmetry": symmetry,
        "pressure_x_reflection_asymmetry_not_a_symmetry_error": {
            str(run["pressure_pa"].shape[1]): _relative_metrics(
                run["pressure_pa"], run["pressure_pa"][:, ::-1]
            )
            for run in runs
        },
        "ridge_peak_sampling": peaks,
        "fixed_cut_ridge_geometry": ridge,
        "fixed_physical_ridge_neighborhoods": fixed,
        "fixed_region_pairwise_differences": comparisons,
        "core_flux_stencil": flux,
        "smooth_fully_flooded_flux_manufactured_order": _manufactured_flux_order(case),
        "independent_gaussian_elasticity_check": _elasticity_synthetic(case),
        "513_newton_history": _checkpoint_history(runs[-1]),
        "interpretation": (
            "Post-processing only; uses accepted saved roots and checkpoints. "
            "No simulation, field modification, or physical-parameter tuning."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs=4, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = diagnose(args.roots)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(f"Wrote core-ridge diagnostics to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
