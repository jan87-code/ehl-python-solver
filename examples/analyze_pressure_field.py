"""Summarize one independently accepted pressure-field archive."""

import argparse
import json
from pathlib import Path

import numpy as np

from ehl_solver.pressure_analysis import (
    centerline_peak_diagnostics,
    centerline_pressure_profile,
    outlet_spike_diagnostics,
)


def analyze(field_path, outlet_window_um=None):
    field_path = Path(field_path)
    metadata_path = field_path.parent / "summary.json"
    if not metadata_path.is_file():
        raise ValueError(f"Acceptance metadata is required beside {field_path}")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("status") not in ("accepted_stage", "accepted_numerical_root"):
        raise ValueError(f"Field is not an accepted root: {metadata.get('status')}")
    audit = metadata.get("audit", {})
    if not audit.get("numerically_accepted", False):
        raise ValueError("Archived field lacks a passing independent numerical audit")
    with np.load(field_path, allow_pickle=False) as saved:
        required = {"pressure_pa", "x_m", "y_m"}
        if not required.issubset(saved.files):
            raise ValueError(f"Pressure archive must contain {sorted(required)}")
        pressure = np.array(saved["pressure_pa"], dtype=np.float64, copy=True)
        x = np.array(saved["x_m"], dtype=np.float64, copy=True)
        y = np.array(saved["y_m"], dtype=np.float64, copy=True)
        theta = (
            np.array(saved["theta"], dtype=np.float64, copy=True)
            if "theta" in saved.files
            else None
        )
    line_x, line_pressure = centerline_pressure_profile(x, y, pressure)
    outlet = (
        {"identified": False, "reason": "No outlet window supplied"}
        if outlet_window_um is None
        else outlet_spike_diagnostics(
            line_x,
            line_pressure,
            window_m=tuple(value * 1e-6 for value in outlet_window_um),
        )
    )
    cavitation = None
    if theta is not None:
        line_theta = np.array([np.interp(0.0, y, theta[:, i]) for i in range(x.size)])
        released = np.flatnonzero((line_pressure == 0) & (line_theta < 1))
        cavitation = (
            {
                "first_centerline_zero_pressure_deficit_x_m": float(x[released[0]]),
                "fractional_film_content": float(line_theta[released[0]]),
            }
            if released.size
            else None
        )
    maximum = np.unravel_index(np.argmax(pressure), pressure.shape)
    return {
        "field": str(field_path),
        "acceptance_status": metadata["status"],
        "transport_fraction": metadata.get("transport_fraction"),
        "shape_ny_nx": list(pressure.shape),
        "domain_x_m": [float(x[0]), float(x[-1])],
        "domain_y_m": [float(y[0]), float(y[-1])],
        "spacing_x_median_m": float(np.median(np.diff(x))),
        "spacing_y_median_m": float(np.median(np.diff(y))),
        "global_maximum": {
            "pressure_pa": float(pressure[maximum]),
            "x_m": float(x[maximum[1]]),
            "y_m": float(y[maximum[0]]),
        },
        "centerline": centerline_peak_diagnostics(line_x, line_pressure),
        "first_centerline_pressure_release": cavitation,
        "outlet_spike": outlet,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("field", type=Path, help="Accepted NPZ containing pressure_pa, x_m and y_m")
    parser.add_argument(
        "--outlet-window-um",
        nargs=2,
        type=float,
        metavar=("LEFT", "RIGHT"),
        help="Shoulder-to-shoulder outlet window in um, defined from the finest field",
    )
    args = parser.parse_args()
    print(json.dumps(analyze(args.field, args.outlet_window_um), indent=2, allow_nan=False))
