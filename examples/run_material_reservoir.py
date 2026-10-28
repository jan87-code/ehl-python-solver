"""Bounded fixed-spacing inlet/outlet/transverse studies at the registered 200 N.

Every case starts from the SAME audited source. Zero-pressure extensions are
unaccepted guesses; original full-material Newton repair and physical gates
are mandatory. Failures/partials remain rejected and are checkpointed.
"""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid

from ehl_solver.benchmark_200n import Circular200N
from ehl_solver.elasticity_audit import direct_midpoint_displacement
from ehl_solver.property_coupled import PropertyCoupledSolution
from ehl_solver.property_load_newton import PropertyLoadNewtonError, PropertyLoadSystem, verify_seed
from ehl_solver.property_reservoir import extend_property_guess, extended_reservoir_problem
from ehl_solver.property_transfer import repair_property_load_guess
from examples.audit_transport_wavelengths import load_root
from examples.run_load_directed_200n import finite_json
from examples.run_property_load_verification import save_state
from examples.run_scientific_validation import provenance, save_json
from examples.scientific_figures import figure, save


def audit_fields(problem, solution, offset, history):
    """Fresh SI force/field/source and original globalization checks."""
    c = Circular200N()
    system = PropertyLoadSystem(problem, c.fluid(), 2., 200., preconditioner="local")
    z = verify_seed(system, solution, offset)
    checked = system.solution(z).solution
    force = float(trapezoid(trapezoid(solution.pressure_pa, problem.x_m, axis=1), problem.y_m))
    if abs(force/200.-1) > 1e-8:
        raise RuntimeError("independent force closure failed")
    peak = np.unravel_index(np.argmax(solution.pressure_pa), problem.shape)
    points = [[0., 0.], [problem.x_m[peak[1]], problem.y_m[peak[0]]]]
    direct = direct_midpoint_displacement(problem.x_m, problem.y_m, solution.pressure_pa,
                                          c.contact_modulus_pa, points)
    center = (np.argmin(abs(problem.y_m)), np.argmin(abs(problem.x_m)))
    # Every extension keeps the original exact x=y=0 node.
    if problem.x_m[center[1]] != 0 or problem.y_m[center[0]] != 0:
        raise ValueError("registered centered source required")
    error = float(abs(direct-[solution.elastic_displacement_m[center],
                              solution.elastic_displacement_m[peak]]).max())
    if error > 1e-18:
        raise RuntimeError("independent full-source deformation failed")
    models, max_linear, shortened = 0, 0., 0
    for iteration in history:
        accepted = []
        for model in iteration["models"]:
            models += 1
            max_linear = max(max_linear, model["linear_true_residual"])
            if model["linear_true_residual"] > 5e-9:
                raise RuntimeError("original true correction gate failed")
            for trial in model["trials"]:
                if trial["accepted"]:
                    accepted.append(trial)
                    shortened += trial["alpha"] < 1
                    if (model["slope"] >= 0 or trial["merit"] >
                            iteration["merit"]+1e-4*trial["alpha"]*model["slope"]):
                        raise RuntimeError("original directional Armijo gate failed")
        if len(accepted) != 1:
            raise RuntimeError("exactly one accepted candidate required")
    minimum = np.unravel_index(np.argmin(solution.film_thickness_m), problem.shape)
    return {"force_n": force, "relative_force_error": abs(force/200.-1),
            "direct_displacement_error_m": error, "audit": checked.audit,
            "newton_corrections": len(history), "linear_models": models,
            "maximum_true_linear_residual": max_linear, "shortened_steps": shortened,
            "minimum_film_m": float(solution.film_thickness_m.min()),
            "minimum_location_xy_m": [float(problem.x_m[minimum[1]]),
                                      float(problem.y_m[minimum[0]])],
            "central_film_m": float(solution.film_thickness_m[center]),
            "peak_pressure_pa": float(solution.pressure_pa.max()), "offset_m": offset}


def main(source, directory, cells):
    if not isinstance(cells, int) or isinstance(cells, bool) or not 1 <= cells <= 8:
        raise ValueError("extension must be 1..8 original cells")
    original, z, seed, reference = load_root(source)
    problem = original.problem
    fluid = original.base.fluid
    offset = float(z[-1]*problem.film_scale_m)
    directory.mkdir(parents=True, exist_ok=False)
    baseline = audit_fields(problem, seed, offset, ())
    report = {"status": "incomplete", "case": Circular200N().parameters(), "runtime": provenance(),
              "source": reference, "baseline": baseline, "cases": [],
              "mesh_verified": False, "physical_domain_verified": False,
              "original_pressure_stencil_preserved": True, "cells": cells}

    def checkpoint():
        save_json(directory / "summary.json", finite_json(report))

    checkpoint()
    traces = [("baseline", problem.x_m, seed.pressure_pa[problem.shape[0]//2],
               seed.film_thickness_m[problem.shape[0]//2])]
    for name, counts in (("inlet", dict(west=cells)), ("outlet", dict(east=cells)),
                         ("transverse", dict(south=cells, north=cells))):
        target = extended_reservoir_problem(problem, **counts)
        path = directory / name
        path.mkdir()
        entry = {"name": name, "accepted": False, "extension_cells": counts,
                 "shape_ny_nx": list(target.shape),
                 "bounds_x_m": target.x_m[[0, -1]].tolist(),
                 "bounds_y_m": target.y_m[[0, -1]].tolist(),
                 "dx_m": target.dx_m, "dy_m": target.dy_m}
        report["cases"].append(entry)
        checkpoint()
        print(name, target.shape, "unaccepted extension; original bounded repair", flush=True)
        try:
            guess = extend_property_guess(problem, target, fluid, 2., seed=seed,
                                          seed_offset_m=offset)
            h, w = target.fields(guess.pressure_pa, central_film_offset_m=offset)
            np.savez_compressed(path / "unaccepted_guess.npz", x_m=target.x_m, y_m=target.y_m,
                                pressure_pa=guess.pressure_pa, theta=guess.fractional_film_content,
                                film_m=h, displacement_m=w)
            entry["initialization"] = guess.diagnostics
            checkpoint()
            root = repair_property_load_guess(target, fluid, 2., 200., guess)
        except (PropertyLoadNewtonError, ValueError) as error:
            if isinstance(error, PropertyLoadNewtonError):
                save_json(path / "failure.json", finite_json({
                    "reason": str(error), "history": error.history,
                    "diagnostics": error.diagnostics,
                }))
                if error.last_root is not None:
                    save_state(path / "failed_partial.npz", target, error.last_root.solution, fluid)
            entry["failure"] = str(error)
            checkpoint()
            print(name, "REJECTED", str(error), flush=True)
            continue
        s = root.solution
        save_state(path / "solution.npz", target, s, fluid)
        save_json(path / "history.json", finite_json(s.history))
        entry.update(audit_fields(target, s, root.central_film_offset_m, s.history))
        entry.update(accepted=True, state_file=f"{name}/solution.npz")
        entry["relative_minimum_film_change"] = entry["minimum_film_m"]/baseline["minimum_film_m"]-1
        entry["relative_central_film_change"] = entry["central_film_m"]/baseline["central_film_m"]-1
        entry["relative_peak_pressure_change"] = (
            entry["peak_pressure_pa"]/baseline["peak_pressure_pa"]-1
        )
        checkpoint()
        row = int(np.argmin(abs(target.y_m)))
        traces.append((name, target.x_m, s.pressure_pa[row], s.film_thickness_m[row]))
        print(name, "ACCEPTED", entry["newton_corrections"], "corrections; hmin",
              entry["minimum_film_m"], flush=True)
    if (hashlib.sha256(Path(reference["source"]).read_bytes()).hexdigest()
            != reference["source_sha256"]):
        raise RuntimeError("original source archive changed during independent cases")
    fig, axes = figure("200 N material JFO: separate fixed-spacing reservoir extensions",
                        "Same source/force; unchanged field bounds, stencil and repair budgets\n"
                        "Accepted roots only; raw centerlines; mesh/far-field accuracy open",
                        rows=2)
    for name, x, p, h in traces:
        axes[0].plot(x*1e3, p/1e6, "o-", label=name, markersize=2)
        axes[1].plot(x*1e3, h*1e6, "o-", label=name, markersize=2)
    axes[0].set_ylabel("Pressure [MPa]")
    axes[1].set_ylabel("Centerline film [um]")
    axes[1].set_ylim(0., .7)
    for ax in axes:
        ax.set_xlim(-.3, .3)
        ax.set_xlabel("x along entrainment [mm]")
        ax.legend()
    save(fig, directory, "raw_reservoir_centerlines")
    report["status"] = "bounded fixed-spacing reservoir assessment complete; accuracy open"
    checkpoint()


def audit_saved(source, directory):
    report = json.loads((source / "summary.json").read_text())
    if report["status"] == "incomplete" or report["case"] != Circular200N().parameters():
        raise ValueError("complete registered study required")
    directory.mkdir(parents=True, exist_ok=False)
    original_path = Path(report["source"]["source"])
    if hashlib.sha256(original_path.read_bytes()).hexdigest() != report["source"]["source_sha256"]:
        raise RuntimeError("reference source hash changed")
    with np.load(original_path) as a:
        old_x, old_y, old_p, old_h = [a[k].copy() for k in
                                     ("x_m", "y_m", "pressure_pa", "film_m")]
    rows, traces = [], []
    for entry in report["cases"]:
        if not entry["accepted"]:
            continue
        file = source / entry["state_file"]
        with np.load(file) as a:
            base = Circular200N().problem(.5e-6)
            p = replace(base, x_m=a["x_m"], y_m=a["y_m"])
            pressure, theta, h, w = [a[k].copy() for k in
                                     ("pressure_pa", "theta", "film_m", "displacement_m")]
        solution = PropertyCoupledSolution(pressure, theta, h, w, p.load_n(pressure), {}, (), 1.)
        history = json.loads((file.parent / "history.json").read_text())
        row = audit_fields(p, solution, entry["offset_m"], history)
        ix = int(np.flatnonzero(p.x_m == old_x[0])[0])
        iy = int(np.flatnonzero(p.y_m == old_y[0])[0])
        np.testing.assert_array_equal(p.x_m[ix:ix+old_x.size], old_x)
        np.testing.assert_array_equal(p.y_m[iy:iy+old_y.size], old_y)
        overlap = np.s_[iy:iy+old_y.size, ix:ix+old_x.size]
        row["shared_pressure_relative_l2_change"] = float(
            np.linalg.norm(pressure[overlap]-old_p)/np.linalg.norm(old_p)
        )
        row["shared_film_maximum_difference_m"] = float(abs(h[overlap]-old_h).max())
        center = int(np.argmin(abs(old_y)))
        traces.append((entry["name"], pressure[overlap][center]-old_p[center],
                       h[overlap][center]-old_h[center]))
        row.update(name=entry["name"], state_sha256=hashlib.sha256(file.read_bytes()).hexdigest())
        rows.append(row)
    fig, axes = figure(
        "200 N fixed-spacing reservoir: raw differences on identical original nodes",
        "Same audited source and original field/force gates; no resampling or filtering\n"
        "Finite extent sensitivity only; not mesh or infinite-domain validation", rows=2,
    )
    for name, dp, dh in traces:
        axes[0].plot(old_x*1e3, dp/1e6, "o-", label=name, markersize=2)
        axes[1].plot(old_x*1e3, dh*1e9, "o-", label=name, markersize=2)
    axes[0].set_ylabel("Pressure difference [MPa]")
    axes[1].set_ylabel("Total-film difference [nm]")
    for ax in axes:
        ax.set_xlabel("Original x along entrainment [mm]")
        ax.legend()
    save(fig, directory, "shared_node_differences")
    save_json(directory / "summary.json", {"status": "accepted saved reservoir roots audited",
                                           "runtime": provenance(), "cases": rows})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--cells", type=int, default=4)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.audit_only:
        audit_saved(args.input_directory, args.output_directory)
    else:
        main(args.input_directory, args.output_directory, args.cells)
