"""Bounded rigid point-contact extent, resolution and SI similarity checks.

No elasticity or constitutive extension. H=1+X^2+(Rx/Ry)Y^2, Lambda=1,
L=sqrt(2 Rx h0), P=p/p_ref, p_ref=12 eta Ue L/h0^2.
Extent A means x in [-A L,A L], y in [-A Ly,A Ly], Ly=sqrt(2 Ry h0).
step is dX=dy/Ly, explicitly dimensionless. Fields and operating inputs are SI.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid

from ehl_solver.cavitation import solve_cavitating_reynolds
from examples.scientific_figures import extrema, figure, save


def audit_volume_flux(
    fields, *, viscosity_pa_s, speed_m_s, length_m, film_scale_m, pressure_scale_pa
):
    """Independently difference complete SI face fluxes (not active-set rows).

    Original dimensionless divergence = SI divergence * L/q_ref.
    Integrated dimensional boundary flow [m^3/s] = q_ref L * net_hat.
    """
    x, y = fields["x_m"], fields["y_m"]
    p, h, theta = fields["pressure_pa"], fields["film_m"], fields["theta"]
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
    qx = (
        -0.5 * (h[:, 1:] ** 3 + h[:, :-1] ** 3) * np.diff(p, axis=1) / (12 * viscosity_pa_s * dx)
        + speed_m_s * 0.5 * (h[:, 1:] + h[:, :-1]) * theta[:, :-1]
    )
    qy = -0.5 * (h[1:, :] ** 3 + h[:-1, :] ** 3) * np.diff(p, axis=0) / (12 * viscosity_pa_s * dy)
    divergence = np.diff(qx[1:-1], axis=1) / dx + np.diff(qy[:, 1:-1], axis=0) / dy
    q_ref = film_scale_m**3 * pressure_scale_pa / (12 * viscosity_pa_s * length_m)
    outward = np.array(
        [
            -qx[1:-1, 0].sum() * dy,
            qx[1:-1, -1].sum() * dy,
            -qy[0, 1:-1].sum() * dx,
            qy[-1, 1:-1].sum() * dx,
        ]
    )
    local_scale = (np.abs(qx[1:-1, 1:]) + np.abs(qx[1:-1, :-1])) * dy + (
        np.abs(qy[1:, 1:-1]) + np.abs(qy[:-1, 1:-1])
    ) * dx
    return {
        "mass_maximum": float(np.max(np.abs(divergence)) * length_m / q_ref),
        "maximum_relative_local_balance": float(
            np.max(np.abs(divergence * dx * dy) / np.maximum(local_scale, np.finfo(float).tiny))
        ),
        "net_boundary_flux": float(outward.sum() / (q_ref * length_m)),
        "net_volume_flow_m3_s": float(outward.sum()),
        "outward_volume_flow_m3_s": outward.tolist(),
        "summed_volume_balance_m3_s": float(divergence.sum() * dx * dy),
        "maximum_complementarity": float(np.max(np.abs(p / pressure_scale_pa * (1 - theta)))),
    }


def solve_case(
    extent,
    step,
    *,
    h0_m=1e-7,
    radius_x_m=0.01,
    radius_y_m=0.02,
    viscosity_pa_s=0.04,
    speed_m_s=0.001,
):
    """Return one rigid state and SI audit; A,step are dimensionless as above."""
    values = (extent, step, h0_m, radius_x_m, radius_y_m, viscosity_pa_s, speed_m_s)
    if not all(np.isfinite(value) and value > 0 for value in values):
        raise ValueError("extent, spacing and dimensional parameters must be positive finite")
    half_intervals = extent / step
    if not np.isclose(half_intervals, round(half_intervals), rtol=0, atol=1e-10):
        raise ValueError("extent must be an integer number of grid spacings")
    nodes = 2 * round(half_intervals) + 1
    if nodes < 3 or nodes**2 > 70000:
        raise ValueError("rigid study requires 3 to 264 nodes per axis and at most 70000 nodes")
    length = float(np.sqrt(2 * radius_x_m * h0_m))
    transverse_length = float(np.sqrt(2 * radius_y_m * h0_m))
    pressure_scale = 12 * viscosity_pa_s * speed_m_s * length / h0_m**2
    unit_axis = np.linspace(-extent, extent, nodes)
    x, y = unit_axis * length, unit_axis * transverse_length
    film = h0_m + x[None, :] ** 2 / (2 * radius_x_m) + y[:, None] ** 2 / (2 * radius_y_m)
    solution = solve_cavitating_reynolds(film / h0_m, step, step * transverse_length / length, 1)
    fields = {
        "x_m": x,
        "y_m": y,
        "film_m": film,
        "pressure_pa": solution.pressure * pressure_scale,
        "theta": solution.fractional_film_content,
        "displacement_m": np.zeros_like(film),
    }
    audit = audit_volume_flux(
        fields,
        viscosity_pa_s=viscosity_pa_s,
        speed_m_s=speed_m_s,
        length_m=length,
        film_scale_m=h0_m,
        pressure_scale_pa=pressure_scale,
    )
    load = float(trapezoid(trapezoid(fields["pressure_pa"], x, axis=1), y))
    if (
        audit["mass_maximum"] > 1e-8
        or audit["maximum_relative_local_balance"] > 1e-10
        or abs(audit["net_boundary_flux"]) > 1e-8
        or audit["maximum_complementarity"] > 1e-10
        or solution.pressure.min() < -1e-10
        or solution.fractional_film_content.min() < -1e-10
        or solution.fractional_film_content.max() > 1 + 1e-10
    ):
        raise RuntimeError(f"rigid SI flux/admissibility audit failed: {audit}")
    return {
        "extent_over_natural_lengths": extent,
        "dX": step,
        "shape": [nodes, nodes],
        "h0_m": h0_m,
        "Rx_m": radius_x_m,
        "Ry_m": radius_y_m,
        "eta_pa_s": viscosity_pa_s,
        "Ue_m_s": speed_m_s,
        "L_m": length,
        "Ly_m": transverse_length,
        "p_ref_pa": pressure_scale,
        "Lambda": 1,
        "load_n": load,
        "load_dimensionless": load / (pressure_scale * length**2),
        "x_limits_m": [float(x[0]), float(x[-1])],
        "y_limits_m": [float(y[0]), float(y[-1])],
        "audit": audit,
        "active_set_iterations": solution.active_set_iterations,
        "sampled_extrema": extrema(fields),
        "maximum_rigid_slope": float(np.hypot(x[-1] / radius_x_m, y[-1] / radius_y_m)),
    }, fields


def main(directory):
    directory.mkdir(parents=True, exist_ok=False)
    report = {"scope": "rigid numerical far-field probe; no domain-independence claim", "cases": []}
    for extent, step in (
        (2, 0.5),
        (4, 0.5),
        (8, 0.5),
        (16, 0.5),
        (32, 0.5),
        (8, 0.25),
        (8, 0.125),
        (16, 0.25),
        (32, 0.25),
    ):
        entry, fields = solve_case(extent, step)
        report["cases"].append(entry)
        np.savez_compressed(directory / f"extent_{extent}_step_{step}.npz", **fields)
        (directory / "summary.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"Rigid A={extent}, dX={step}: W={entry['load_dimensionless']:.12g}; "
            f"mass={entry['audit']['mass_maximum']:.3e}",
            flush=True,
        )
    details = "\n".join(
        f"A={case['extent_over_natural_lengths']}, dX={case['dX']}, {case['shape']}: "
        f"Q={case['load_n']:.7g} N, "
        f"pmax={case['sampled_extrema']['maximum_pressure']['value_si'] / 1e6:.6g} MPa, "
        f"mass={case['audit']['mass_maximum']:.2e}\n"
        f"x=+/-{case['x_limits_m'][1] * 1e3:.6g} mm; "
        f"y=+/-{case['y_limits_m'][1] * 1e3:.6g} mm; pmax at "
        f"({case['sampled_extrema']['maximum_pressure']['x_m'] * 1e3:.6g},"
        f"{case['sampled_extrema']['maximum_pressure']['y_m'] * 1e3:.6g}) mm; "
        f"net flux={case['audit']['net_boundary_flux']:.2e}"
        for case in report["cases"]
    )
    context = (
        "Verified prescribed rigid states; hmin=0.1 um at (0,0); no deformation or load control\n"
        "Rx=0.01 m, Ry=0.02 m; E*=rigid; eta=0.04 Pa s, Ue=0.001 m/s; entrainment +x\n"
        "L=sqrt(2 Rx h0), Ly=sqrt(2 Ry h0); inlet=-A L, outlet=A L; y=+/-A Ly\n"
        "W=Q/(p_ref L^2); every grid and SI domain/extremum/flux is saved in summary.json\n"
        + details
    )
    fig, axes = figure("Rigid point-contact extent and resolution sensitivity", context, rows=2)
    for step in (0.5, 0.25, 0.125):
        cases = sorted(
            [case for case in report["cases"] if case["dX"] == step],
            key=lambda case: case["extent_over_natural_lengths"],
        )
        axes[0].plot(
            [case["extent_over_natural_lengths"] for case in cases],
            [case["load_dimensionless"] for case in cases],
            "o-",
            label=f"dX={step}",
        )
    axes[0].set(xlabel="Domain extent A [dimensionless]", ylabel="Integrated pressure W")
    axes[0].legend()
    cases = sorted(
        [case for case in report["cases"] if case["extent_over_natural_lengths"] == 8],
        key=lambda case: case["dX"],
        reverse=True,
    )
    axes[1].plot(
        [case["dX"] for case in cases], [case["load_dimensionless"] for case in cases], "o-"
    )
    axes[1].set(xlabel="Streamwise dX at fixed A=8 [dimensionless]", ylabel="Integrated pressure W")
    save(fig, directory, "rigid_extent_resolution")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory", type=Path, default=Path("results/rigid_point_farfield")
    )
    main(parser.parse_args().output_directory)
