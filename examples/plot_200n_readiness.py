"""Compare saved 200 N readiness probes, preserving their actual low forces."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid

from ehl_solver.benchmark_200n import Circular200N
from ehl_solver.elasticity_audit import direct_midpoint_displacement
from ehl_solver.property_reynolds import admissible, mass_audit
from examples.run_scientific_validation import save_json
from examples.scientific_figures import figure, save


def main(inputs, output):
    reports = [json.loads((p / "summary.json").read_text()) for p in inputs]
    if any(s["status"] == "incomplete" for s in reports):
        raise ValueError("complete readiness reports required")
    if any(s["case"] != reports[0]["case"] for s in reports):
        raise ValueError("physical cases must match")
    if any(s["bounds_in_hertz_radii"] != reports[0]["bounds_in_hertz_radii"] for s in reports):
        raise ValueError("physical domains must match for mesh comparison")
    audits = []
    case = Circular200N()
    if reports[0]["case"] != case.parameters():
        raise ValueError("this auditor requires the explicit registered 200 N case")
    for directory, report in zip(inputs, reports, strict=True):
        for probe in report["probes"]:
            if not probe["accepted"]:
                continue
            problem = case.problem(
                probe["offset_m"], ny=report["shape_ny_nx"][0], nx=report["shape_ny_nx"][1]
            )
            state_path = directory / f"offset_{probe['offset_m'] * 1e6:g}um" / "solution.npz"
            with np.load(state_path) as state:
                p, theta, h, w = (
                    state[k] for k in ("pressure_pa", "theta", "film_m", "displacement_m")
                )
                np.testing.assert_array_equal(state["x_m"], problem.x_m)
                np.testing.assert_array_equal(state["y_m"], problem.y_m)
                audit = mass_audit(
                    problem.x_m,
                    problem.y_m,
                    h,
                    p,
                    theta,
                    case.fluid(),
                    case.entrainment_m_s,
                    length_scale_m=problem.length_scale_m,
                    film_scale_m=problem.film_scale_m,
                    pressure_scale_pa=problem.pressure_scale_pa,
                )
                exact, exact_w = problem.fields(p)
                film_error = float(np.max(abs(exact - h)))
                displacement_error = float(np.max(abs(exact_w - w)))
                force = float(trapezoid(trapezoid(p, problem.x_m, axis=1), problem.y_m))
                peak = np.unravel_index(np.argmax(p), p.shape)
                targets = [[0.0, 0.0], [problem.x_m[peak[1]], problem.y_m[peak[0]]]]
                direct = direct_midpoint_displacement(
                    problem.x_m, problem.y_m, p, case.contact_modulus_pa, targets
                )
                stored = np.array([w[len(problem.y_m) // 2, len(problem.x_m) // 2], w[peak]])
                direct_error = float(np.max(abs(direct - stored)))
                if (
                    not admissible(audit, p, theta, problem.pressure_scale_pa)
                    or h.min() <= problem.film_floor_m
                    or max(film_error, displacement_error, direct_error) > 1e-18
                    or abs(force / probe["load_n"] - 1) > 1e-12
                ):
                    raise RuntimeError("saved probe failed independent field audit")
                audits.append(
                    {
                        "state": str(state_path),
                        "mass_audit": audit,
                        "film_error_m": film_error,
                        "displacement_error_m": displacement_error,
                        "direct_sample_error_m": direct_error,
                        "independent_load_n": force,
                    }
                )
    output.mkdir(parents=True, exist_ok=False)
    fig, axes = figure(
        "Circular 200 N case readiness", "Fixed physical reservoir; prescribed offsets", rows=2
    )
    for report in reports:
        probes = sorted((p for p in report["probes"] if p["accepted"]), key=lambda p: p["offset_m"])
        label = str(tuple(report["shape_ny_nx"])) + " (ny,nx)"
        axes[0].loglog(
            [p["offset_m"] * 1e6 for p in probes], [p["load_n"] for p in probes], "o-", label=label
        )
        axes[1].semilogx(
            [p["offset_m"] * 1e6 for p in probes],
            [p["peak_pressure_pa"] / 1e6 for p in probes],
            "o-",
            label=label,
        )
    axes[0].axhline(200, color="black", linestyle=":", label="requested 200 N")
    axes[0].set_ylabel("Integrated pressure load [N]")
    axes[1].set_ylabel("Sampled peak pressure [MPa]")
    for ax in axes:
        ax.set_xlabel("Prescribed positive offset h0 [um]")
        ax.legend()
    fig.suptitle(
        "Circular case readiness; R=15 mm, E_contact=113 GPa, Ue=2 m/s\n"
        "Assumed Roelands/Dowson-Higginson laws; finite +/-3a reservoir\n"
        "Accepted prescribed-offset roots do not establish 200 N closure"
    )
    save(fig, output, "readiness")
    comparison = []
    if len(reports) == 2:
        by_offset = [{p["offset_m"]: p for p in s["probes"] if p["accepted"]} for s in reports]
        for offset in sorted(by_offset[0].keys() & by_offset[1].keys()):
            a, b = (s[offset] for s in by_offset)
            comparison.append(
                {
                    "offset_m": offset,
                    "relative_load_change": b["load_n"] / a["load_n"] - 1,
                    "relative_peak_change": b["peak_pressure_pa"] / a["peak_pressure_pa"] - 1,
                    "relative_minimum_film_change": b["minimum_film_m"] / a["minimum_film_m"] - 1,
                }
            )
    save_json(
        output / "comparison.json",
        {
            "input_summary_sha256": {
                str(p / "summary.json"): hashlib.sha256(
                    (p / "summary.json").read_bytes()
                ).hexdigest()
                for p in inputs
            },
            "plot_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "figure_manifest": ["readiness.png"],
            "independent_saved_field_audits": audits,
            "mesh_changes": comparison,
            "target_achieved_in_all_reports": bool(np.all([s["target_achieved"] for s in reports])),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    main(args.inputs, args.output_directory)
