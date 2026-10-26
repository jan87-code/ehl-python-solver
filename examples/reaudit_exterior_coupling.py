"""Reaudit saved exterior roots against independently tightened uniform references.

Preserve the original rejected reference and all candidate data. No exterior
candidate seeds the independent Newton reference, and no tolerance is relaxed.
"""

import argparse
import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ehl_solver.cavitating_coupled import CavitatingElasticityProblem, CavitatingElasticitySolution
from ehl_solver.cavitating_newton import solve_cavitating_newton
from ehl_solver.rectilinear_cavitation import flux_audit
from examples.run_exterior_response import fields
from examples.run_farfield_budget import ETA, H0, P_REF, RX, RY, SPEED, L, accepted
from examples.run_scientific_validation import provenance, save_json
from examples.scientific_figures import figure, save


def main(source, directory):
    directory.mkdir(parents=True, exist_ok=False)
    raw = (source / "summary.json").read_bytes()
    original = json.loads(raw)
    if original.get("status") in ("incomplete", "failed") or not original.get("cases"):
        raise ValueError("reaudit requires completed candidates, including rejected references")
    report = {
        "status": "incomplete",
        "runtime": provenance(),
        "source_summary_sha256": hashlib.sha256(raw).hexdigest(),
        "source_runtime": original["runtime"],
        "references": [],
        "cases": [],
        "criteria": original["criteria"],
        "source_npz_sha256": {},
    }

    def checkpoint():
        save_json(directory / "summary.json", report)

    checkpoint()
    try:
        shutil.copy2(source / "summary.json", directory / "source_summary.json")
        for p in source.glob("*.npz"):
            shutil.copy2(p, directory / p.name)
            report["source_npz_sha256"][p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
        targets = {}
        for old in original["references"]:
            step = old["core_step"]
            f = np.load(source / f"reference_{step}.npz")
            x, y = f["x_m"], f["y_m"]
            p, t, h, w = f["pressure_pa"], f["theta"], f["film_m"], f["displacement_m"]
            problem = CavitatingElasticityProblem(x, y, H0, RX, RY, 1e9, L, H0, P_REF, 1)
            scale = dict(length_scale_m=L, film_scale_m=H0, pressure_scale_pa=P_REF)
            audit = flux_audit(x, y, h, p, t, ETA, SPEED, **scale)
            entry = {
                "step": step,
                "original_audit": audit,
                "original_accepted": accepted(audit, fields(x, y, h, p, t)),
                "original_history": old,
            }
            if not entry["original_accepted"]:
                # Warm only from the independent monolithic reference, never the exterior root.
                warm = CavitatingElasticitySolution(p, t, h, w, H0, 1, problem.load_n(p), ())
                tight = solve_cavitating_newton(
                    problem, initial_solution=warm, mass_tolerance=1e-10
                )
                s = tight.solution
                p, t, h, w = (
                    s.pressure_pa,
                    s.fractional_film_content,
                    s.film_thickness_m,
                    s.elastic_displacement_m,
                )
                entry["tightening_history"] = [asdict(row) for row in tight.history]
                np.savez_compressed(
                    directory / f"tight_reference_{step}.npz",
                    **fields(x, y, h, p, t),
                    displacement_m=w,
                )
            audit = flux_audit(x, y, h, p, t, ETA, SPEED, **scale)
            entry.update(
                audit=audit,
                accepted=accepted(audit, fields(x, y, h, p, t)),
                load_n=problem.load_n(p),
            )
            report["references"].append(entry)
            targets[step] = (problem, p, t, h)
            checkpoint()
        for old in original["cases"]:
            f = np.load(source / old["state_file"])
            problem, p, t, h = targets[old["step"]]
            cp, ct, ch = f["pressure_pa"], f["theta"], f["film_m"]
            exact, w = problem.fields(cp)
            audit = flux_audit(
                problem.x_m,
                problem.y_m,
                ch,
                cp,
                ct,
                ETA,
                SPEED,
                length_scale_m=L,
                film_scale_m=H0,
                pressure_scale_pa=P_REF,
            )
            entry = old | {
                "original_acceptance": old["accepted"],
                "audit": audit,
                "relative_pressure_error": float(np.linalg.norm(cp - p) / np.linalg.norm(p)),
                "relative_load_error": abs(problem.load_n(cp) / problem.load_n(p) - 1),
                "relative_film_error": float(np.linalg.norm(ch - h) / np.linalg.norm(h)),
                "maximum_reference_film_error_m": float(abs(ch - h).max()),
                "maximum_content_error": float(abs(ct - t).max()),
                "maximum_field_coupling_error_m": float(abs(ch - exact).max()),
                "maximum_displacement_error_m": float(abs(f["displacement_m"] - w).max()),
            }
            entry["accepted"] = bool(
                accepted(audit, fields(problem.x_m, problem.y_m, ch, cp, ct))
                and max(
                    entry["relative_pressure_error"],
                    entry["relative_load_error"],
                    entry["relative_film_error"],
                )
                <= 1e-8
                and entry["maximum_field_coupling_error_m"] <= 1e-18
                and all(r["accepted"] for r in report["references"])
            )
            report["cases"].append(entry)
            checkpoint()
        fig, axes = figure(
            "Exterior coupled roots against independently audited references",
            "Same finite reservoir +/-8L,+/-8Ly; h0=.1 um,E*=1 GPa\n"
            "Rejected fine monolithic state retained; tighter reference uses the same equations\n"
            "No candidate seeds a reference; original physical criteria unchanged",
        )
        for key in ("relative_pressure_error", "relative_load_error", "relative_film_error"):
            axes[0].plot(
                [f"{c['step']},cut{c['cut']}" for c in report["cases"]],
                [c[key] for c in report["cases"]],
                "o-",
                label=key,
            )
        axes[0].set_yscale("symlog", linthresh=1e-16)
        axes[0].axhline(1e-8, linestyle=":", color="black")
        axes[0].set(xlabel="Core dX and numerical cut", ylabel="Relative discrepancy")
        axes[0].legend(fontsize=8)
        save(fig, directory, "tight_reference_errors")
        report["figures"] = ["tight_reference_errors.png"]
        report["status"] = (
            "verified exterior coupling against audited references"
            if all(c["accepted"] for c in report["cases"])
            else "failed acceptance"
        )
        checkpoint()
    except Exception as exc:
        report["status"] = "failed"
        report["failure"] = repr(exc)
        checkpoint()
        raise
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    main(args.input_directory, args.output_directory)
