"""Reproduce CPU Hertz/JFO/elastic-load evidence and mandatory scientific plots.

Use a fresh output directory and one BLAS/OpenMP thread. All dimensional
fields are saved in SI NPZ files; JSON retains exact inputs and histories.
This runner verifies existing equations without adding constitutive physics.
It does not port JFO to CUDA or claim physical film-height validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import trapezoid

from ehl_solver.elasticity import elastic_deformation
from ehl_solver.hertz import circular_point_contact
from examples import run_cavitating_boundaries, run_cavitating_newton
from examples import run_parabolic_cavitation_reference as parabolic
from examples import run_rigid_point_farfield as farfield
from examples.scientific_figures import (
    caption,
    domain_figure,
    extrema,
    field_figures,
    figure,
    grid_figure,
    newton_figures,
    save,
)

PARAMETERS = {
    "target_load_n": 0.002,
    "eta_pa_s": 0.04,
    "Ue_m_s": 0.425,
    "E_star_pa": 1e7,
    "Rx_m": 0.01,
    "Ry_m": 0.02,
    "L_m": 5e-4,
    "h_ref_m": 20e-6,
    "p_ref_pa": 21250,
    "Lambda": 12,
    "inlet_theta": 1,
}


def load_fields(path):
    with np.load(path, allow_pickle=False) as state:
        return {key: state[key].copy() for key in state.files}


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def provenance():
    root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = "unavailable"
    sources = list((root / "examples").glob("*.py"))
    sources += list((root / "src/ehl_solver").glob("*.py"))
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "git_head_at_run": commit,
        "source_sha256": {
            path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(sources)
        },
    }


def hertz_benchmark(directory):
    directory.mkdir()
    load, radius, modulus = 5.0, 0.015, 113e9
    reference = circular_point_contact(load, radius, modulus)
    cases = []
    for n in (33, 65, 129):
        x = np.linspace(-4 * reference.radius_m, 4 * reference.radius_m, n)
        xx, yy = np.meshgrid(x, x, indexing="xy")
        squared_radius = xx**2 + yy**2
        pressure = np.zeros((n, n))
        inside = squared_radius < reference.radius_m**2
        pressure[inside] = reference.peak_pressure_pa * np.sqrt(
            1 - squared_radius[inside] / reference.radius_m**2
        )
        displacement = elastic_deformation(pressure, x[1] - x[0], x[1] - x[0], modulus)
        dry_gap = squared_radius / (2 * radius) - reference.approach_m + displacement
        force = float(trapezoid(trapezoid(pressure, x, axis=1), x))
        entry = {
            "shape": [n, n],
            "load_n": force,
            "load_relative_error": abs(force - load) / load,
            "center_displacement_m": float(displacement[n // 2, n // 2]),
            "center_displacement_relative_error": abs(
                displacement[n // 2, n // 2] / reference.approach_m - 1
            ),
            "radius_m": reference.radius_m,
            "peak_pressure_pa": reference.peak_pressure_pa,
            "approach_m": reference.approach_m,
        }
        np.savez_compressed(
            directory / f"hertz_{n}.npz",
            x_m=x,
            y_m=x,
            pressure_pa=pressure,
            displacement_m=displacement,
            dry_gap_m=dry_gap,
        )
        context = (
            "Analytical dry Hertz pressure / DC-FFT displacement; no lubricant film or flow\n"
            f"Q={load:g} N; R={radius:g} m; E*={modulus:g} Pa; eta,Ue=not applicable\n"
            f"(ny,nx)=({n},{n}); x,y=[{x[0] * 1e3:.6g},{x[-1] * 1e3:.6g}] mm; "
            "+x is reference entrainment orientation (no inlet/outlet flow)\n"
            f"pmax={pressure.max() / 1e6:.9g} MPa at (0,0) mm; hmin=not applicable; "
            f"sampled minimum dry gap={dry_gap.min() * 1e6:.6g} um\n"
            f"force error={force - load:.3e} N; center displacement relative error="
            f"{entry['center_displacement_relative_error']:.3e}; mass residual=not applicable"
        )
        fig, axes = figure("Dry Hertz analytical reference and displacement", context, rows=2)
        axes[0].plot(x * 1e3, pressure[n // 2] / 1e6, "o-")
        axes[0].set(xlabel="x [mm]", ylabel="Pressure [MPa]")
        axes[1].plot(x * 1e3, displacement[n // 2] * 1e6, "o-", label="DC-FFT")
        axes[1].plot(
            x[abs(x) <= reference.radius_m] * 1e3,
            (reference.approach_m - x[abs(x) <= reference.radius_m] ** 2 / (2 * radius)) * 1e6,
            label="Exact inside contact",
        )
        axes[1].set(xlabel="x [mm]", ylabel="Combined displacement [um]")
        axes[1].legend()
        save(fig, directory, f"hertz_{n}")
        cases.append(entry)
    errors = [case["center_displacement_relative_error"] for case in cases]
    if not errors[2] < errors[1] < errors[0] or errors[-1] > 0.01:
        raise RuntimeError("Hertz displacement refinement failed")
    save_json(
        directory / "summary.json",
        {"Q_n": load, "R_m": radius, "E_star_pa": modulus, "cases": cases},
    )
    return cases


def reference_figures(directory, report):
    for case in report["cases"]:
        nx = case["shape"][1]
        fields = load_fields(directory / f"grid_17_{nx}.npz")
        x, y = fields["x_m"], fields["y_m"]
        center = len(y) // 2
        j, i = np.unravel_index(np.argmax(fields["pressure_pa"]), fields["pressure_pa"].shape)
        context = (
            "Rigid wide 2-D JFO centerline vs exact infinite-width finite-inlet limit\n"
            "hmin=20 um at x=0 (all y); Rx=0.01 m; Ry=infinite; E*=rigid; "
            "eta=0.04 Pa s; Ue=0.425 m/s; displacement=0\n"
            f"(ny,nx)=(17,{nx}); x=[{x[0] * 1e3:.6g},{x[-1] * 1e3:.6g}] mm; "
            f"y=[{y[0] * 1e3:.6g},{y[-1] * 1e3:.6g}] mm; inlet left, outlet right (+x)\n"
            f"pmax={fields['pressure_pa'].max() / 1e6:.8g} MPa at "
            f"({x[i] * 1e3:.6g},{y[j] * 1e3:.6g}) mm; prescribed film, no target Q\n"
            f"Centerline force={case['force_per_width_n_m']:.8g} N/m (not point load); "
            f"mass max={case['mass_maximum']:.3e}; net flux={case['net_boundary_flux']:.3e}; "
            f"complementarity={case['complementarity_maximum']:.3e}\n"
            f"verified discrete root; pressure relative error={case['pressure_relative_l2']:.3e}; "
            f"force/width relative error={case['relative_force_error']:.3e}"
        )
        fig, axes = figure("Matched analytical pressure and content comparison", context, rows=2)
        axes[0].plot(x * 1e3, fields["reference_pressure_pa"] / 1e6, label="Exact 1-D")
        axes[0].plot(x * 1e3, fields["pressure_pa"][center] / 1e6, ".--", label="Wide 2-D")
        axes[0].set(xlabel="x [mm], inlet to outlet", ylabel="Pressure [MPa]")
        axes[0].legend()
        axes[1].plot(x * 1e3, fields["reference_theta"], label="Exact 1-D")
        axes[1].plot(x * 1e3, fields["theta"][center], ".--", label="Wide 2-D")
        axes[1].axvline(case["analytical_rupture_m"] * 1e3, color="grey", linestyle=":")
        axes[1].set(xlabel="x [mm], inlet to outlet", ylabel="Fractional film content")
        axes[1].legend()
        save(fig, directory, f"analytical_comparison_{nx}")


def main(directory):
    directory.mkdir(parents=True, exist_ok=False)
    report = {
        "scope": "restricted-model numerical verification; physical validation open",
        "parameters": PARAMETERS,
        "runtime": provenance(),
        "stages": {},
    }

    def checkpoint(stage, data):
        report["stages"][stage] = data
        save_json(directory / "summary.json", report)
        print(f"Scientific validation: {stage} complete", flush=True)

    try:
        checkpoint("hertz", hertz_benchmark(directory / "hertz"))
        reference = parabolic.main(directory / "analytical_reference", plots=False)
        reference_figures(directory / "analytical_reference", reference)
        checkpoint("analytical_reference", reference)
        checkpoint("rigid_farfield", farfield.main(directory / "rigid_farfield"))
        grids = run_cavitating_newton.main(directory / "grids", plots=False)
        for case in grids["cases"]:
            label = f"grid_{case['shape'][0]}_{case['shape'][1]}"
            fields = load_fields(directory / "grids" / f"{label}.npz")
            case["sampled_extrema"] = extrema(fields)
            field_figures(directory / "grids", label, fields, PARAMETERS, case["audit"])
            newton_figures(
                directory / "grids", label, fields, PARAMETERS, case["audit"], case["history"]
            )
            if "nested_reference" in case:
                independent = load_fields(directory / "grids" / f"{label}_nested_reference.npz")
                context = caption(fields, PARAMETERS, case["audit"])
                context += (
                    "\nMatched Picard + scalar h0 root: relative pressure difference="
                    f"{case['nested_reference']['pressure_relative_difference']:.3e}"
                )
                fig, axes = figure(
                    "Matched monolithic / nested reference centerlines", context, rows=2
                )
                for state, title in ((fields, "Monolithic"), (independent, "Nested reference")):
                    center = len(state["y_m"]) // 2
                    axes[0].plot(
                        state["x_m"] * 1e3, state["pressure_pa"][center] / 1e6, ".-", label=title
                    )
                    axes[1].plot(
                        state["x_m"] * 1e3, state["film_m"][center] * 1e6, ".-", label=title
                    )
                axes[0].set(xlabel="x [mm], inlet to outlet", ylabel="Pressure [MPa]")
                axes[1].set(xlabel="x [mm], inlet to outlet", ylabel="Film thickness [um]")
                for axis in axes:
                    axis.legend()
                save(fig, directory / "grids", "matched_nested_reference")
        grid_figure(directory, grids["cases"], fields, PARAMETERS)
        checkpoint("loaded_grids", grids)
        boundaries = run_cavitating_boundaries.main(directory / "boundaries", plots=False)
        for case in boundaries["cases"]:
            if not case["target_reached"]:
                raise RuntimeError(f"boundary case {case['name']} failed: {case['termination']}")
            fields = load_fields(directory / "boundaries" / f"{case['name']}_final.npz")
            case["sampled_extrema"] = extrema(fields)
            field_figures(
                directory / "boundaries", case["name"], fields, PARAMETERS, case["final_audit"]
            )
        baseline = load_fields(directory / "boundaries/baseline_final.npz")
        domain_figure(directory, boundaries["cases"], baseline, PARAMETERS)
        checkpoint("loaded_boundaries", boundaries)
        report["status"] = "all numerical checks passed; domain independence not established"
        report["figures"] = [
            path.relative_to(directory).as_posix() for path in sorted(directory.rglob("*.png"))
        ]
        save_json(directory / "summary.json", report)
    except Exception as error:
        report["status"] = "failed"
        report["failure"] = {"type": type(error).__name__, "reason": str(error)}
        save_json(directory / "summary.json", report)
        raise
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory", type=Path, default=Path("results/scientific_validation")
    )
    args = parser.parse_args()
    main(args.output_directory)
