"""Independent reading and spatial comparisons of experimental MC path results.

SI pressure [Pa], film [m], coordinates [m]; arrays float64 (ny,nx), +x
entrainment. Change metrics are not a spatial-convergence certificate.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from ehl_solver.limited_audit import audit_limited_state
from ehl_solver.limited_system import LimitedLoadProbe
from ehl_solver.prototype import PointContactCase


def file_sha256(path):
    """Hash provenance without loading large JSON histories into memory."""
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            result.update(block)
    return result.hexdigest()


@dataclass(frozen=True)
class LimitedRun:
    case: PointContactCase
    state: np.ndarray
    pressure_pa: np.ndarray
    theta: np.ndarray
    film_m: np.ndarray
    audit: dict


def _checksum(path, expected):
    if not isinstance(expected, str) or file_sha256(path) != expected:
        raise ValueError(f"Checksum mismatch: {path.name}")


def load_limited_path(directory):
    """Re-audit a completed fraction-one path; partial paths cannot be loaded.

    Verify both NPZ checksums, coordinates, ND/SI consistency, archived source
    hashes and the independent direct-elasticity/MC audit. Archived Python is
    never executed. This reader supports the current research path format,
    not arbitrary prototype/direct-solve result formats or a future v1 schema.
    """
    directory = Path(directory)
    # Histories can dominate a research bundle. Discard trace containers as
    # each dictionary is decoded; the field reader does not replay iterations.
    # The raw JSON text is still read in full, not claimed to be streamed.
    report = json.loads(
        (directory / "summary.json").read_text(),
        object_hook=lambda obj: {
            k: v for k, v in obj.items() if k not in ("history", "models", "attempts", "iterations")
        },
    )
    if (
        report.get("status") != "accepted_numerical_root"
        or report.get("operator_transport") != "limited_capacity_target"
        or type(report.get("transport_fraction")) not in (int, float)
        or report.get("transport_fraction") != 1.0
    ):
        raise ValueError("Completed exact MC path required; partial/blended results are unaccepted")
    stages = report.get("stages", [])
    if (
        not stages
        or stages[-1].get("status") != "accepted_stage"
        or type(stages[-1].get("transport_fraction")) not in (int, float)
        or stages[-1].get("transport_fraction") != 1.0
    ):
        raise ValueError("Accepted fraction-one final stage required")
    saved_audit = report.get("audit", {})
    if not all(
        saved_audit.get(k) is True
        for k in ("numerically_accepted", "full_limited_equations", "direct_elasticity_checked")
    ):
        raise ValueError("Recorded accepted direct MC final audit required")
    sources = report.get("source_sha256")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("Archived source checksums required")
    for name, expected in sources.items():
        if Path(name).name != name or not name.endswith(".py"):
            raise ValueError("Source names must be local Python basenames")
        _checksum(directory / "source/ehl_solver" / name, expected)
    _checksum(directory / "run_limited_transport_path.py", report.get("driver_sha256"))
    top = directory / "last_accepted_state.npz"
    stage = directory / f"attempt_{len(stages) - 1:03d}" / "state.npz"
    _checksum(top, report.get("last_accepted_state_sha256"))
    _checksum(stage, stages[-1].get("state_sha256"))
    case = PointContactCase(**report["case"])
    probe = LimitedLoadProbe(case)
    with np.load(top, allow_pickle=False) as data:
        state = data["state_dimensionless"].copy()
    if state.dtype != np.float64 or state.shape != (probe.size,) or not np.isfinite(state).all():
        raise ValueError("Finite float64 complete dimensionless state required")
    p, theta, film = probe.fields(state)
    _, _, _, displacement, offset = probe.system.base.unpack(state[:-1])
    problem = probe.system.problem
    expected_fields = dict(
        state_dimensionless=state,
        x_m=problem.x_m,
        y_m=problem.y_m,
        pressure_pa=p,
        theta=theta,
        film_m=film,
        displacement_m=displacement,
        offset_m=np.asarray(offset),
    )
    with np.load(stage, allow_pickle=False) as data:
        for name, expected in expected_fields.items():
            actual = data[name]
            if actual.dtype != np.float64 or not np.array_equal(actual, expected):
                raise ValueError(
                    f"Archived ND/SI field differs from current reconstruction: {name}"
                )
    audit = audit_limited_state(probe, state, direct=True)
    if not audit["numerically_accepted"] or not audit["full_limited_equations"]:
        raise ValueError("Independent exact MC/direct-elasticity audit failed")
    # The loader supplies its own audit; it does not trust saved residuals.
    fields = [np.array(a, dtype=np.float64, copy=True) for a in (state, p, theta, film)]
    for a in fields:
        a.setflags(write=False)
    return LimitedRun(case, *fields, audit)


def compare_fields(
    coarse_case, fine_case, coarse_pressure_pa, fine_pressure_pa, coarse_film_m, fine_film_m
):
    """Measure spatial change on identical physical case/domain, with no gates.

    This low-level sampling function also accepts mathematical test fields;
    it does not audit Reynolds/cavitation/load closure. Use load_limited_path
    first for accepted MC results. Bilinear sampling of fine fields at coarse
    nodes is labeled explicitly when the grids do not share all coarse nodes.
    """
    if coarse_case.physical_signature() != fine_case.physical_signature():
        raise ValueError("Identical physical case and transport reference required")
    if (
        coarse_case.domain_half_width_x_m != fine_case.domain_half_width_x_m
        or coarse_case.domain_half_width_y_m != fine_case.domain_half_width_y_m
    ):
        raise ValueError("Identical physical domain required for spatial comparison")
    if fine_case.nx < coarse_case.nx or fine_case.ny < coarse_case.ny:
        raise ValueError("Fine grid cannot coarsen either axis")

    def fields(case, p, h):
        p, h = np.asarray(p, dtype=np.float64), np.asarray(h, dtype=np.float64)
        if any(a.shape != (case.ny, case.nx) or not np.isfinite(a).all() for a in (p, h)):
            raise ValueError("Finite fields in (ny,nx) order required")
        if h.min() <= 0:
            raise ValueError("Strictly positive SI film required")
        x = np.linspace(-case.domain_half_width_x_m, case.domain_half_width_x_m, case.nx)
        y = np.linspace(-case.domain_half_width_y_m, case.domain_half_width_y_m, case.ny)
        return x, y, p, h

    cx, cy, cp, ch = fields(coarse_case, coarse_pressure_pa, coarse_film_m)
    fx, fy, fp, fh = fields(fine_case, fine_pressure_pa, fine_film_m)
    xx, yy = np.meshgrid(cx, cy)
    points = np.c_[yy.ravel(), xx.ravel()]

    def nearest(fine, coarse):
        right = np.clip(np.searchsorted(fine, coarse), 1, len(fine) - 1)
        return np.where(abs(fine[right] - coarse) < abs(fine[right - 1] - coarse), right, right - 1)

    ix, iy = nearest(fx, cx), nearest(fy, cy)
    # Coordinate tolerance is only for identifying shared nodes; it does not
    # merge limiter states or relax any numerical acceptance tolerance.
    coordinate_tolerance = 32 * np.finfo(np.float64).eps * max(abs(fx).max(), abs(fy).max())
    shared = bool(
        np.allclose(fx[ix], cx, rtol=0, atol=coordinate_tolerance)
        and np.allclose(fy[iy], cy, rtol=0, atol=coordinate_tolerance)
    )
    if shared:
        sampled = fp[np.ix_(iy, ix)]
    else:
        sampled = RegularGridInterpolator((fy, fx), fp, bounds_error=True)(points).reshape(cp.shape)
    denominator = np.linalg.norm(sampled)
    maximum = abs(fp).max()
    if not np.isfinite(denominator) or denominator == 0 or maximum == 0:
        raise ValueError("Nonzero finer pressure required for relative pressure norms")
    coarse_exit = (
        (xx / coarse_case.hertz.radius_m >= 0.5)
        & (xx / coarse_case.hertz.radius_m <= 1.5)
        & (np.abs(yy / coarse_case.hertz.radius_m) <= 1.2)
    )
    fine_xx, fine_yy = np.meshgrid(fx, fy)
    fine_exit = (
        (fine_xx / fine_case.hertz.radius_m >= 0.5)
        & (fine_xx / fine_case.hertz.radius_m <= 1.5)
        & (np.abs(fine_yy / fine_case.hertz.radius_m) <= 1.2)
    )
    exit_denominator = np.linalg.norm(sampled[coarse_exit])
    exit_maximum = abs(sampled[coarse_exit]).max()
    if exit_denominator == 0 or exit_maximum == 0 or not np.isfinite(exit_denominator):
        raise ValueError("Nonzero finer exit-region pressure required")
    cmin, fmin = np.unravel_index(ch.argmin(), ch.shape), np.unravel_index(fh.argmin(), fh.shape)
    radius = coarse_case.hertz.radius_m
    correlation = coarse_case.correlation()
    central_coarse = ch[coarse_case.ny // 2, coarse_case.nx // 2]
    central_fine = fh[fine_case.ny // 2, fine_case.nx // 2]
    return dict(
        axis_order="ny,nx",
        entrainment_direction="+x",
        pressure_unit="Pa",
        film_unit="m",
        coordinate_unit="m",
        hamrock_dowson_reference_m=dict(
            minimum_film=float(correlation.minimum_film_m),
            central_film=float(correlation.central_film_m),
        ),
        coarse_shape=[coarse_case.ny, coarse_case.nx],
        fine_shape=[fine_case.ny, fine_case.nx],
        coarse_nodes=coarse_case.ny * coarse_case.nx,
        fine_nodes=fine_case.ny * fine_case.nx,
        sampling="shared_nodes" if shared else "bilinear_fine_at_coarse_nodes",
        shared_node_coordinate_tolerance_m=float(coordinate_tolerance),
        pressure_relative_l2_on_coarse_nodes=float(np.linalg.norm(sampled - cp) / denominator),
        pressure_relative_max_on_coarse_nodes=float(abs(sampled - cp).max() / maximum),
        exit_region_definition="0.5 <= x/a <= 1.5 and |y/a| <= 1.2",
        exit_pressure_relative_l2_on_coarse_nodes=float(
            np.linalg.norm(sampled[coarse_exit] - cp[coarse_exit]) / exit_denominator
        ),
        exit_pressure_relative_max_on_coarse_nodes=float(
            abs(sampled[coarse_exit] - cp[coarse_exit]).max() / exit_maximum
        ),
        exit_pressure_coarse_peak_pa=float(cp[coarse_exit].max()),
        exit_pressure_fine_peak_pa=float(fp[fine_exit].max()),
        minimum_film_coarse_m=float(ch.min()),
        minimum_film_fine_m=float(fh.min()),
        minimum_film_relative_change=float(fh.min() / ch.min() - 1),
        minimum_film_coarse_relative_to_hd=float(ch.min() / correlation.minimum_film_m - 1),
        minimum_film_fine_relative_to_hd=float(fh.min() / correlation.minimum_film_m - 1),
        central_film_coarse_m=float(central_coarse),
        central_film_fine_m=float(central_fine),
        central_film_relative_change=float(central_fine / central_coarse - 1),
        central_film_coarse_relative_to_hd=float(central_coarse / correlation.central_film_m - 1),
        central_film_fine_relative_to_hd=float(central_fine / correlation.central_film_m - 1),
        minimum_position_coarse_x_over_a=float(cx[cmin[1]] / radius),
        minimum_position_coarse_abs_y_over_a=float(abs(cy[cmin[0]]) / radius),
        minimum_position_fine_x_over_a=float(fx[fmin[1]] / radius),
        minimum_position_fine_abs_y_over_a=float(abs(fy[fmin[0]]) / radius),
        numerical_acceptance_checked=False,
        mesh_convergence_assessed=False,
    )
