"""Resolved physical carriers and feasible one-sided fronts at audited 200 N roots.

Static response verification only: no production solve/stencil change, field
filtering, spectral condition estimate or temporal-stability claim.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid

from ehl_solver.benchmark_200n import Circular200N
from ehl_solver.property_coupled import PropertyCoupledSolution
from ehl_solver.property_load_newton import PropertyLoadSystem, verify_seed
from ehl_solver.transport_audit import mass_direction_components
from ehl_solver.transport_probes import (
    active_front_probes,
    minimum_directional,
    resolved_pressure_probes,
)
from examples.run_scientific_validation import provenance, save_json
from examples.scientific_figures import figure, save


def load_root(source, *, maximum_physical_nodes=20000):
    c = Circular200N()
    summary_file = source / "summary.json"
    summary = json.loads(summary_file.read_text())
    if summary["status"] == "incomplete" or summary["case"] != c.parameters():
        raise ValueError("complete registered source required")
    grid = summary["grids"]["refined"]
    entries = [e for e in grid["attempts"] if e["accepted"] and abs(e["load_n"]/200-1) <= 1e-8]
    if not entries:
        raise ValueError("accepted 200 N root required")
    entry = entries[-1]
    path = source / entry["state_file"]
    problem = c.problem(.5e-6, ny=grid["shape_ny_nx"][0], nx=grid["shape_ny_nx"][1],
                        maximum_physical_nodes=maximum_physical_nodes)
    with np.load(path) as a:
        np.testing.assert_array_equal(a["x_m"], problem.x_m)
        np.testing.assert_array_equal(a["y_m"], problem.y_m)
        p, t, h, w = [a[k].copy() for k in ("pressure_pa", "theta", "film_m", "displacement_m")]
    force = float(trapezoid(trapezoid(p, problem.x_m, axis=1), problem.y_m))
    if abs(force/200-1) > 1e-8:
        raise ValueError("original independent force gate failed")
    seed = PropertyCoupledSolution(p, t, h, w, force, {}, (), 1.)
    system = PropertyLoadSystem(problem, c.fluid(), 2., 200., preconditioner="local",
                                maximum_physical_nodes=maximum_physical_nodes)
    state = verify_seed(system, seed, entry["offset_m"])
    return system, state, seed, {
        "source": str(path), "force_n": force, "shape_ny_nx": list(p.shape),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "summary_sha256": hashlib.sha256(summary_file.read_bytes()).hexdigest(),
    }


def evaluate_probe(system, state, seed, probe, *, one_sided):
    n, problem = system.n, system.problem
    dp, dt = probe["pressure_pa"], probe["content"]
    v = np.r_[dp[1:-1, 1:-1].ravel()/problem.pressure_scale_pa,
              -dt[1:-1, 1:-1].ravel(), 0.]
    dh = system.film_direction(state, v)
    terms = mass_direction_components(
        problem.x_m, problem.y_m, seed.film_thickness_m, seed.pressure_pa,
        seed.fractional_film_content, system.base.fluid, 2., pressure_direction_pa=dp,
        content_direction=dt, film_direction_m=dh,
    )
    scale = problem.length_scale_m/system.base.flux_scale/system.base.mass_scale
    mass = sum(terms.values()).ravel()*scale
    comp = minimum_directional(state[:n], state[n:2*n], v[:n], v[n:2*n])
    load = float(trapezoid(trapezoid(dp, problem.x_m, axis=1), problem.y_m)/200.)
    directional = np.r_[mass, comp, load]
    selected = system.jacobian_vector(state, v)
    norm = float(np.linalg.norm(directional))
    mass_error = float(np.linalg.norm(selected[:n]-mass)/max(np.linalg.norm(mass), 1e-300))
    residual = system.residual(state)
    expected_slope = float(np.dot(residual, directional)/residual.size)
    slope_error = abs(system.merit_slope(state, residual, v)-expected_slope)
    slope_scale = max(float(np.linalg.norm(residual))*norm/residual.size, 1e-300)
    gap, change = state[:n]-state[n:2*n], v[:n]-v[n:2*n]
    if one_sided:
        switches = (gap*change < 0)
    else:
        switches = (gap != 0) & (change != 0)
    branch_limit = float(np.min(abs(gap[switches]/change[switches]))) if switches.any() else None
    base_step = min(.03, .1*branch_limit) if branch_limit is not None else .03
    differences = []

    def feasible(trial):
        system.sync(trial)
        P, t, h, _, _ = system.base.unpack(trial[:-1])
        return bool(P.min() >= -1e-10 and t.min() >= -1e-10
                    and t.max() <= 1+1e-10 and h.min() > problem.film_floor_m)

    for fraction in (1., .3, .1, .03, .01):
        step = base_step*fraction
        trial1 = state+step*v
        trial2 = state+(2*step if one_sided else -step)*v
        changed = not np.array_equal(trial1, state) and not np.array_equal(trial2, state)
        ok = feasible(trial1) and feasible(trial2)
        item = {"step": step, "bounds_feasible": ok, "both_trials_change_state": changed}
        if ok and changed:
            r1, r2 = system.residual(trial1), system.residual(trial2)
            fd = ((4*(r1-residual)-(r2-residual))/(2*step) if one_sided
                  else (r1-r2)/(2*step))
            item["relative_complete_directional_error"] = float(
                np.linalg.norm(fd-directional)/max(norm, 1e-300)
            )
        differences.append(item)
    best = min((d["relative_complete_directional_error"] for d in differences
                if "relative_complete_directional_error" in d), default=None)
    frozen = sum(terms[k] for k in ("pressure_diffusion", "density_diffusion",
                                   "viscosity_diffusion", "density_advection"))
    film = terms["film_diffusion"]+terms["film_advection"]
    row = {**probe["metadata"], "fd_scheme": "second_order_forward" if one_sided else "centered",
           "branch_switch_limit": branch_limit, "finite_differences": differences,
           "best_complete_directional_error": best,
           "mass_decomposition_relative_error": mass_error,
           "selected_vs_directional_complementarity_difference":
               float(np.linalg.norm(selected[n:2*n]-comp)),
           "force_direction_n": load*200., "merit_slope_scaled_error": slope_error/slope_scale,
           "state_direction_l2": float(np.linalg.norm(v)),
           "scaled_residual_gain": norm/max(float(np.linalg.norm(v)), 1e-300),
           "mass_gain": float(np.linalg.norm(mass))/max(float(np.linalg.norm(v)), 1e-300),
           "component_divergence_l2_kg_m2_s": {
               k: float(np.linalg.norm(t)) for k, t in terms.items()},
           "film_to_frozen_material_norm_ratio": float(np.linalg.norm(film))
               / max(float(np.linalg.norm(frozen)), 1e-300)}
    row["verified"] = bool(mass_error <= 1e-10 and abs(selected[-1]-load) <= 1e-12
                            and slope_error <= 1e-10*slope_scale
                            and best is not None and best <= 1e-6)
    arrays = {"dp_pa": dp, "dtheta": dt, "dh_m": dh,
              "directional_residual": directional, "selected_jvp": selected}
    return row, arrays


def main(sources, directory):
    directory.mkdir(parents=True, exist_ok=False)
    report = {"status": "incomplete", "runtime": provenance(), "grids": [],
              "production_stencil_changed": False, "stability_verified": False,
              "scope": "static supported carriers and bound-feasible front directions"}
    save_json(directory / "summary.json", report)
    for source in sources:
        system, z, seed, row = load_root(source)
        c = Circular200N()
        waves, omitted = resolved_pressure_probes(system.problem.x_m, system.problem.y_m,
                                                   seed.pressure_pa, c.hertz_radius_m)
        fronts, front_metadata = active_front_probes(seed.pressure_pa,
                                                     seed.fractional_film_content,
                                                     system.problem.pressure_scale_pa)
        for face in front_metadata["selected_faces"].values():
            for kind in ("full", "cavitated"):
                j, i = face[kind+"_yx"]
                face[kind+"_x_m"] = float(system.problem.x_m[i])
                face[kind+"_y_m"] = float(system.problem.y_m[j])
                face[kind+"_pressure_pa"] = float(seed.pressure_pa[j, i])
                face[kind+"_theta"] = float(seed.fractional_film_content[j, i])
        row.update(omitted_carriers=omitted, front_selection=front_metadata, probes={})
        report["grids"].append(row)
        name = "x".join(map(str, row["shape_ny_nx"]))
        arrays = {}
        for group, probes in (("wavelength", waves), ("front", fronts)):
            for label, probe in probes.items():
                result, fields = evaluate_probe(system, z, seed, probe, one_sided=group == "front")
                row["probes"][label] = {"group": group, **result}
                arrays.update({label+"_"+key: value for key, value in fields.items()})
                save_json(directory / "summary.json", report)
                print(name, label, result["verified"], result["best_complete_directional_error"],
                      flush=True)
                if not result["verified"]:
                    np.savez_compressed(directory / (name+"_partial.npz"), **arrays)
                    raise RuntimeError("directional diagnostic failed; checkpoint retained")
        np.savez_compressed(directory / (name+"_probes.npz"), **arrays)
    fig, axes = figure(
        "200 N accepted roots: fixed physical carriers and Nyquist references",
        "Lines: fixed carriers; crosses: Nyquist references; four cells per carrier minimum\n"
        "Scaled static mass gain; neither an eigenvalue nor a temporal-stability test", rows=2,
    )
    for grid, color in zip(report["grids"], ("C0", "C1", "C2", "C3"), strict=False):
        label = "x".join(map(str, grid["shape_ny_nx"]))
        for ax, axis in zip(axes, ("x", "y"), strict=True):
            for phase, style in (("cos", "o-"), ("sin", "s--")):
                rows = [r for r in grid["probes"].values()
                        if r.get("resolved_carrier") and r["axis"] == axis and r["phase"] == phase]
                rows.sort(key=lambda r: r["wavelength_over_a"])
                ax.loglog([r["wavelength_over_a"] for r in rows], [r["mass_gain"] for r in rows],
                          style, color=color, label=label+" "+phase)
            reference = grid["probes"]["nyquist_"+axis]
            ax.loglog(reference["wavelength_m"]/c.hertz_radius_m, reference["mass_gain"],
                      "x", color=color, markersize=9)
            ax.set_ylabel(axis+" carrier mass gain [scaled]")
            ax.set_xlabel("Nominal physical wavelength / dry Hertz radius a")
            ax.legend(fontsize=8)
    save(fig, directory, "carrier_responses")
    report["status"] = "directional carriers/front checks complete; stability/accuracy open"
    save_json(directory / "summary.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directories", nargs="+", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    main(args.input_directories, args.output_directory)
