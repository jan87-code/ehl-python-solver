"""Load-controlled pressure-dependent JFO/elastic numerical verification, CPU only.

No experimental target is inferred: known-offset recovery uses an independent
complete nonlinear reference, followed by a separately imposed 0.1 N force.
"""

import argparse
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from ehl_solver.cavitating_coupled import CavitatingElasticityProblem
from ehl_solver.constitutive import IsothermalFluid
from ehl_solver.elasticity_audit import direct_midpoint_displacement
from ehl_solver.property_load import PropertyLoadError, solve_property_at_load
from ehl_solver.property_reynolds import admissible, mass_audit
from examples.property_monolithic_reference import small_monolithic_reference
from examples.run_farfield_budget import H0, LY, P_REF, RX, RY, L
from examples.run_scientific_validation import provenance, save_json
from examples.scientific_figures import figure, save


def make_problem(ny=9, nx=13):
    return CavitatingElasticityProblem(
        np.linspace(-4, 6, nx) * L,
        np.linspace(-4, 4, ny) * LY,
        1.1 * H0,
        RX,
        RY,
        1e11,
        L,
        H0,
        P_REF * 100,
        1,
    )


def save_state(path, problem, solution, fluid):
    rho, eta, _, _ = fluid.properties(solution.pressure_pa)
    np.savez_compressed(
        path,
        x_m=problem.x_m,
        y_m=problem.y_m,
        pressure_pa=solution.pressure_pa,
        theta=solution.fractional_film_content,
        film_m=solution.film_thickness_m,
        displacement_m=solution.elastic_displacement_m,
        density_kg_m3=rho,
        viscosity_pa_s=eta,
    )


def main(directory, include_refinement=False):
    directory.mkdir(parents=True, exist_ok=False)
    fluid = IsothermalFluid(0.04, 850)
    problem = make_problem()
    report = {
        "status": "incomplete",
        "runtime": provenance(),
        "fluid": asdict(fluid),
        "cases": [],
        "parameters": {
            "Rx_m": RX,
            "Ry_m": RY,
            "E_star_pa": 1e11,
            "Ue_m_s": 0.1,
            "L_m": L,
            "Ly_m": LY,
            "film_scale_m": H0,
            "pressure_scale_pa": P_REF * 100,
            "assigned_temperature_k": 313.15,
        },
        "criteria": {
            "relative_force": 1e-8,
            "relative_reference_pressure_offset": 1e-7,
            "mass_maximum": 1e-8,
            "relative_local_global_complementarity": 1e-10,
            "direct_displacement_m": 1e-18,
        },
        "scope": "Positive-offset finite-reservoir load control; physical/mesh validation open",
    }

    def checkpoint():
        save_json(directory / "summary.json", report)

    checkpoint()
    try:
        reference_problem = replace(problem, central_film_offset_m=H0)
        info, (rp, rt, rh, rw) = small_monolithic_reference(reference_problem, fluid, 0.1)
        audit = mass_audit(
            problem.x_m,
            problem.y_m,
            rh,
            rp,
            rt,
            fluid,
            0.1,
            length_scale_m=L,
            film_scale_m=H0,
            pressure_scale_pa=P_REF * 100,
        )
        if not admissible(audit, rp, rt, P_REF * 100):
            raise RuntimeError("independent reference failed original physical audit")
        q_reference = reference_problem.load_n(rp)
        report["reference"] = {"h0_m": H0, "load_n": q_reference, "status": info, "audit": audit}
        np.savez_compressed(
            directory / "known_reference.npz",
            x_m=problem.x_m,
            y_m=problem.y_m,
            pressure_pa=rp,
            theta=rt,
            film_m=rh,
            displacement_m=rw,
        )
        configurations = [
            ("known_offset", problem, q_reference, (0.8 * H0, 1.3 * H0)),
            ("imposed_force", problem, 0.1, (0.8 * H0, 1.6 * H0)),
            ("alternate_bracket", problem, 0.1, (H0, 1.5 * H0)),
        ]
        if include_refinement:
            configurations.append(("refined_force", make_problem(13, 17), 0.1, (H0, 2.2 * H0)))
        for name, case, target, bracket in configurations:
            case_dir = directory / name
            case_dir.mkdir()
            accepted = []

            def accepted_probe(e, s, case=case, case_dir=case_dir, accepted=accepted):
                index = len(accepted)
                accepted.append(asdict(e))
                save_state(case_dir / f"probe_{index:02d}.npz", case, s, fluid)
                save_json(case_dir / f"probe_{index:02d}_history.json", s.history)
                save_json(case_dir / "evaluations.json", accepted)

            result = solve_property_at_load(
                case, fluid, 0.1, target, h0_bracket_m=bracket, evaluation_callback=accepted_probe
            )
            s = result.coupled_solution
            save_state(case_dir / "solution.npz", case, s, fluid)
            peak = np.unravel_index(s.pressure_pa.argmax(), case.shape)
            point = [[float(case.x_m[peak[1]]), float(case.y_m[peak[0]])]]
            direct = direct_midpoint_displacement(case.x_m, case.y_m, s.pressure_pa, 1e11, point)[0]
            entry = {
                "name": name,
                "shape": list(case.shape),
                "target_load_n": target,
                "calculated_load_n": result.calculated_load_n,
                "relative_load_error": result.relative_load_error,
                "h0_m": result.central_film_offset_m,
                "minimum_film_m": float(s.film_thickness_m.min()),
                "peak_pressure_pa": float(s.pressure_pa.max()),
                "root_iterations": result.root_iterations,
                "evaluations": [asdict(e) for e in result.evaluations],
                "audit": s.audit,
                "direct_peak_displacement_error_m": float(
                    abs(direct - s.elastic_displacement_m[peak])
                ),
                "state_file": f"{name}/solution.npz",
                "bracket_m": bracket,
            }
            entry["accepted"] = bool(
                result.relative_load_error <= 1e-8
                and entry["direct_peak_displacement_error_m"] <= 1e-18
            )
            if name == "known_offset":
                entry["relative_h0_error"] = abs(result.central_film_offset_m / H0 - 1)
                entry["relative_pressure_error"] = float(
                    np.linalg.norm(s.pressure_pa - rp) / np.linalg.norm(rp)
                )
                entry["accepted"] = bool(
                    entry["accepted"]
                    and max(entry["relative_h0_error"], entry["relative_pressure_error"]) <= 1e-7
                )
            report["cases"].append(entry)
            checkpoint()
            print(
                name,
                len(result.evaluations),
                entry["h0_m"],
                entry["relative_load_error"],
                flush=True,
            )
        a, b = report["cases"][1:3]
        report["bracket_h0_relative_difference"] = abs(b["h0_m"] / a["h0_m"] - 1)
        # A real inner budget exhaustion must not produce a successful force probe.
        try:
            solve_property_at_load(
                problem,
                fluid,
                0.1,
                0.1,
                h0_bracket_m=(0.8 * H0, 1.6 * H0),
                maximum_iterations_per_stage=1,
            )
        except PropertyLoadError as exc:
            report["retained_failure"] = {
                "reason": str(exc),
                "accepted_evaluations": [asdict(e) for e in exc.evaluations],
                "failed_offset_m": exc.failed_offset_m,
                "coupled_history": getattr(exc.inner_failure, "history", ()),
            }
            if exc.failed_solution is not None:
                save_state(
                    directory / "failed_inner_candidate.npz", problem, exc.failed_solution, fluid
                )
        else:
            raise AssertionError("budget-one failure probe unexpectedly succeeded")
        if include_refinement:
            c = report["cases"][-1]
            report["fixed_load_h0_mesh_change"] = abs(c["h0_m"] / a["h0_m"] - 1)
            report["fixed_load_minimum_film_mesh_change"] = abs(
                c["minimum_film_m"] / a["minimum_film_m"] - 1
            )
            report["fixed_load_peak_pressure_mesh_change"] = abs(
                c["peak_pressure_pa"] / a["peak_pressure_pa"] - 1
            )
        render(directory, report)
        report["figures"] = [p.name for p in sorted(directory.glob("*.png"))]
        report["status"] = (
            "verified positive-offset load control; physical/mesh validation open"
            if all(c["accepted"] for c in report["cases"])
            and report["bracket_h0_relative_difference"] <= 1e-7
            else "failed acceptance"
        )
        checkpoint()
    except Exception as exc:
        report["status"] = "failed"
        report["failure"] = repr(exc)
        if isinstance(exc, PropertyLoadError):
            report["failed_evaluations"] = [asdict(e) for e in exc.evaluations]
            report["failed_offset_m"] = exc.failed_offset_m
            report["failed_coupled_history"] = getattr(exc.inner_failure, "history", ())
            if exc.failed_solution is not None:
                save_state(directory / "failed_candidate.npz", case, exc.failed_solution, fluid)
        checkpoint()
        raise
    return report


def render(directory, report):
    context = (
        "Pressure-dependent JFO + uniform elasticity; eta0=.04 Pa s,rho0=850 kg/m^3,z=.68\n"
        "Rx=10 mm,Ry=20 mm,E*=100 GPa,Ue=.1 m/s; fixed physical reservoir/scales, flooded west\n"
        "Positive-offset nested load root; no experimental or continuum accuracy claim"
    )
    fig, axes = figure("Audited force versus gap offset", context)
    for c in report["cases"]:
        e = sorted(c["evaluations"], key=lambda e: e["central_film_offset_m"])
        axes[0].plot(
            [a["central_film_offset_m"] * 1e9 for a in e],
            [a["calculated_load_n"] for a in e],
            "o-",
            label=c["name"],
        )
        axes[0].axhline(c["target_load_n"], linestyle=":", color="gray")
    axes[0].set(xlabel="Gap offset h0 [nm]", ylabel="Integrated pressure load [N]")
    axes[0].legend()
    save(fig, directory, "force_offset")
    fig, axes = figure("Independent force closure histories", context)
    for c in report["cases"]:
        axes[0].plot([e["relative_load_error"] for e in c["evaluations"]], "o-", label=c["name"])
    axes[0].set_yscale("symlog", linthresh=1e-12)
    axes[0].axhline(1e-8, color="black", linestyle=":")
    axes[0].set(
        xlabel="Distinct force evaluation", ylabel="Relative force error, exact zeros retained"
    )
    axes[0].legend()
    save(fig, directory, "load_closure")
    fig, axes = figure(
        "Known-offset recovery versus independent complete nonlinear reference", context, rows=2
    )
    r = np.load(directory / "known_reference.npz")
    f = np.load(directory / "known_offset/solution.npz")
    j = len(f["y_m"]) // 2
    for a, label in ((r, "Independent reference"), (f, "Load controlled")):
        axes[0].plot(a["x_m"] * 1e3, a["pressure_pa"][j] / 1e6, label=label)
        axes[1].plot(a["x_m"] * 1e3, a["film_m"][j] * 1e6, label=label)
    axes[0].set_ylabel("Pressure [MPa]")
    axes[0].legend()
    axes[1].set(xlabel="x [mm], entrainment +x", ylabel="Film [um]")
    axes[1].legend()
    save(fig, directory, "matched_fields")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=Path("results/property_load"))
    parser.add_argument(
        "--include-refinement",
        action="store_true",
        help="Add a bounded fixed-force mesh comparison",
    )
    args = parser.parse_args()
    main(args.output_directory, args.include_refinement)
