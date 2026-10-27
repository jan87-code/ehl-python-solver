"""Independent saved-field/force and globalization audit for a bounded load path."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid

from ehl_solver.benchmark_200n import Circular200N
from ehl_solver.property_reynolds import admissible, mass_audit
from examples.run_scientific_validation import provenance, save_json
from examples.scientific_figures import figure, save


def main(source, directory):
    summary_path = source / "summary.json"
    report = json.loads(summary_path.read_text())
    if report["status"] == "incomplete":
        raise ValueError("complete bounded assessment required")
    c = Circular200N()
    f = c.fluid()
    if report["case"] != c.parameters():
        raise ValueError("registered physical case required")
    audits = []
    models = 0
    switched = 0
    fallback = 0
    backtracked = 0
    maxlinear = 0.0
    for name, grid in report["grids"].items():
        ny, nx = grid["shape_ny_nx"]
        problem = c.problem(0.5e-6, ny=ny, nx=nx)
        for entry in grid["attempts"]:
            if not entry["accepted"]:
                continue
            last_entry = entry
            path = source / entry["state_file"]
            with np.load(path) as a:
                p, t, h, w = [a[k] for k in ("pressure_pa", "theta", "film_m", "displacement_m")]
                np.testing.assert_array_equal(a["x_m"], problem.x_m)
                np.testing.assert_array_equal(a["y_m"], problem.y_m)
            actual = mass_audit(
                problem.x_m,
                problem.y_m,
                h,
                p,
                t,
                f,
                2.0,
                length_scale_m=problem.length_scale_m,
                film_scale_m=problem.film_scale_m,
                pressure_scale_pa=problem.pressure_scale_pa,
            )
            exact, ew = problem.fields(p, central_film_offset_m=entry["offset_m"])
            force = float(trapezoid(trapezoid(p, problem.x_m, axis=1), problem.y_m))
            force_error = abs(force / entry["target_load_n"] - 1)
            film_error = float(abs(h - exact).max())
            w_error = float(abs(w - ew).max())
            edges = np.r_[p[0], p[-1], p[:, 0], p[:, -1]]
            if (
                not admissible(actual, p, t, problem.pressure_scale_pa)
                or force_error > 1e-8
                or max(film_error, w_error) > 1e-18
                or np.any(edges != 0)
                or np.any(t[1:-1, 0] != 1)
                or h.min() <= problem.film_floor_m
            ):
                raise RuntimeError("saved force state failed original independent audit")
            history = json.loads((path.parent / "history.json").read_text())
            for correction in history:
                accepts = []
                for model in correction["models"]:
                    models += 1
                    switched += model["recovery"] > 0
                    fallback += model["used_bordered_fallback"]
                    maxlinear = max(maxlinear, model["linear_true_residual"])
                    if model["linear_true_residual"] > 5e-9:
                        raise RuntimeError("bad full correction")
                    for trial in model["trials"]:
                        if trial["accepted"]:
                            accepts.append(trial)
                            backtracked += trial["alpha"] < 1
                            if (
                                model["slope"] >= 0
                                or trial["merit"]
                                > correction["merit"] + 1e-4 * trial["alpha"] * model["slope"]
                            ):
                                raise RuntimeError("true directional Armijo failed")
                if len(accepts) != 1:
                    raise RuntimeError("invalid correction acceptance history")
            audits.append(
                {
                    "grid": name,
                    "state": entry["state_file"],
                    "state_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "audit": actual,
                    "force_n": force,
                    "relative_force_error": force_error,
                    "film_error_m": film_error,
                    "displacement_error_m": w_error,
                    "pressure_minimum_pa": float(p.min()),
                    "content_minimum": float(t.min()),
                    "content_maximum": float(t.max()),
                }
            )
    directory.mkdir(parents=True, exist_ok=False)
    fig, axes = figure(
        "Strongest accepted refined-grid state; numerical verification",
        f"(ny,nx)=({ny},{nx}); Q={force:.9g} N; "
        f"h0={last_entry['offset_m'] * 1e6:.9g} um; +x entrainment\n"
        f"R=15 mm, E_contact=113 GPa, Ue=2 m/s; finite flooded +/-3a reservoir\n"
        f"hmin={h.min() * 1e6:.9g} um; pmin={p.min():.5g} Pa; mesh/domain accuracy open",
        rows=2,
    )
    for ax, values, label in (
        (axes[0], p / 1e6, "Pressure [MPa]"),
        (axes[1], h * 1e6, "Total film [um]"),
    ):
        levels = np.linspace(float(values.min()), float(values.max()), 41)
        plot = ax.contourf(problem.x_m * 1e3, problem.y_m * 1e3, values, levels=levels)
        fig.colorbar(plot, ax=ax, label=label)
        ax.set_xlabel("x along entrainment [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_aspect("equal")
    save(fig, directory, "refined_fields")
    save_json(
        directory / "summary.json",
        {
            "status": "saved accepted states and histories verified",
            "runtime": provenance(),
            "input_summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
            "states": audits,
            "figure_manifest": ["refined_fields.png"],
            "linear_models": models,
            "switched_active_models": switched,
            "bordered_fallbacks": fallback,
            "accepted_backtracked_steps": backtracked,
            "maximum_true_linear_residual": maxlinear,
            "mesh_verified": False,
            "scope": (
                "original equations and force verification; mesh/domain/physical accuracy separate"
            ),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    main(args.input_directory, args.output_directory)
