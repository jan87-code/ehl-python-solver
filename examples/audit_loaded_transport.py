"""Bounded conservative material/elastic pressure-mode audit at accepted 200 N roots.

No production solve, smoothing, transport change, eigenvalue or temporal
stability claim. Every source must pass fresh field/force audits before use.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid

from ehl_solver.backend import as_float64, make_backend
from ehl_solver.benchmark_200n import Circular200N
from ehl_solver.elasticity import ElasticityOperator
from ehl_solver.property_coupled import PropertyCoupledSolution
from ehl_solver.property_load_newton import PropertyLoadSystem, verify_seed
from ehl_solver.transport_audit import (
    mass_direction_components,
    nyquist_pressure_projection,
    pressure_mode_directions,
)
from examples.run_scientific_validation import provenance, save_json
from examples.scientific_figures import figure, save


def audit_source(source, *, cuda_elasticity=False):
    case = Circular200N()
    fluid = case.fluid()
    summary_path = source / "summary.json"
    summary = json.loads(summary_path.read_text())
    if summary["case"] != case.parameters() or summary["status"] == "incomplete":
        raise ValueError("complete registered benchmark source required")
    grid = summary["grids"]["refined"]
    accepted = [r for r in grid["attempts"] if r["accepted"]
                and abs(r["load_n"]/200.-1) <= 1e-8]
    if not accepted:
        raise ValueError("an accepted force-closed 200 N root is required")
    record = accepted[-1]
    path = source / record["state_file"]
    ny, nx = grid["shape_ny_nx"]
    problem = case.problem(.5e-6, ny=ny, nx=nx)
    with np.load(path) as values:
        np.testing.assert_array_equal(values["x_m"], problem.x_m)
        np.testing.assert_array_equal(values["y_m"], problem.y_m)
        p, t, h, w = [values[k].copy() for k in
                      ("pressure_pa", "theta", "film_m", "displacement_m")]
    force = float(trapezoid(trapezoid(p, problem.x_m, axis=1), problem.y_m))
    if abs(force/200.-1) > 1e-8:
        raise ValueError("independently integrated force does not close 200 N")
    seed = PropertyCoupledSolution(p, t, h, w, force, {}, (), 1.)
    system = PropertyLoadSystem(problem, fluid, 2., 200., preconditioner="local")
    z = verify_seed(system, seed, record["offset_m"])
    n = system.n
    radius = case.hertz_radius_m
    modes = pressure_mode_directions(problem.x_m, problem.y_m, p, radius)
    row = {"shape_ny_nx": [ny, nx], "source": str(path),
           "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
           "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
           "force_n": force, "offset_m": record["offset_m"],
           "minimum_film_m": float(h.min()), "mass_scale": system.base.mass_scale,
           "pressure_projection": nyquist_pressure_projection(problem.x_m, problem.y_m, p, radius),
           "modes": {}}
    xx, yy = np.meshgrid(problem.x_m, problem.y_m)
    # Independent analytical dry Hertz support, not clipped solver pressure.
    dry = case.hertz_pressure_pa*np.sqrt(np.maximum(1-(xx**2+yy**2)/radius**2, 0))
    row["sampled_hertz_projection"] = nyquist_pressure_projection(
        problem.x_m, problem.y_m, dry, radius
    )
    gpu_op = None
    if cuda_elasticity:
        gpu = make_backend("torch", "cuda")
        gpu.xp.cuda.reset_peak_memory_stats()
        gpu_op = ElasticityOperator(problem.shape, problem.dx_m, problem.dy_m,
                                    case.contact_modulus_pa, gpu)
        device_w = gpu_op(as_float64(p, gpu)).cpu().numpy()
        error = float(abs(device_w-w).max())
        if error > 1e-18:
            raise RuntimeError("CUDA root deformation parity failed")
        row["cuda_elasticity"] = {
            "device": gpu.xp.cuda.get_device_name(), "dtype": "float64",
            "root_maximum_error_m": error, "directions": {},
            "scope": "elasticity parity only; production material/JFO solve remains CPU",
        }
    arrays = {}
    scale = problem.length_scale_m / system.base.flux_scale / system.base.mass_scale
    for name, dp in modes.items():
        v = np.r_[dp[1:-1, 1:-1].ravel()/problem.pressure_scale_pa, np.zeros(n+1)]
        dh = system.film_direction(z, v)
        if gpu_op is not None:
            device_dh = gpu_op(as_float64(dp, gpu)).cpu().numpy()
            error = float(abs(device_dh-dh).max())
            if error > 1e-18:
                raise RuntimeError("CUDA elastic direction parity failed")
            row["cuda_elasticity"]["directions"][name] = error
        terms = mass_direction_components(
            problem.x_m, problem.y_m, h, p, t, fluid, 2., pressure_direction_pa=dp,
            film_direction_m=dh, content_direction=np.zeros_like(t),
        )
        independent_mass = sum(terms.values()).ravel()*scale
        analytic = system.jacobian_vector(z, v)
        norm = float(np.linalg.norm(analytic))
        error = float(np.linalg.norm(analytic[:n]-independent_mass)/max(norm, 1e-300))
        force_direction = float(trapezoid(trapezoid(dp, problem.x_m, axis=1), problem.y_m))
        np.testing.assert_allclose(analytic[n:2*n], 0., atol=1e-14)
        if error > 1e-10 or abs(analytic[-1]-force_direction/200.) > 1e-12:
            raise RuntimeError("independent decomposed complete Jacobian check failed")
        differences = []
        for step in (.01, .001, .0001):
            plus, minus = z+step*v, z-step*v
            fields_plus = system.base.unpack(plus[:-1])
            fields_minus = system.base.unpack(minus[:-1])
            feasible = all(
                fields[2].min() > problem.film_floor_m
                and fields[0].min() >= -1e-10
                for fields in (fields_plus, fields_minus)
            )
            item = {"step": step, "both_signs_feasible": bool(feasible)}
            if feasible:
                fd = (system.residual(plus)-system.residual(minus))/(2*step)
                item["relative_complete_jvp_error"] = float(
                    np.linalg.norm(fd-analytic)/max(norm, 1e-300)
                )
            differences.append(item)
        best = min((r["relative_complete_jvp_error"] for r in differences
                    if r["both_signs_feasible"]), default=float("inf"))
        if not np.isfinite(best) or best > 1e-6:
            raise RuntimeError("independent residual-difference Jacobian check failed")
        component_norms = {k: float(np.linalg.norm(value)) for k, value in terms.items()}
        total_norm = float(np.linalg.norm(sum(terms.values())))
        frozen = sum(terms[k] for k in (
            "pressure_diffusion", "density_diffusion", "viscosity_diffusion", "density_advection"
        ))
        film = terms["film_diffusion"]+terms["film_advection"]
        row["modes"][name] = {
            "peak_direction_pa": float(abs(dp).max()), "force_direction_n": force_direction,
            "support_cells": int(np.count_nonzero(dp)),
            "full_domain_film_direction_peak_m": float(abs(dh).max()),
            "decomposition_relative_error": error, "residual_differences": differences,
            "best_complete_jvp_error": best,
            "component_divergence_l2_kg_m2_s": component_norms,
            "sum_divergence_l2_kg_m2_s": total_norm,
            "component_cancellation_factor": sum(component_norms.values())/max(total_norm, 1e-300),
            "full_to_frozen_film_norm_ratio": total_norm/max(np.linalg.norm(frozen), 1e-300),
            "film_to_frozen_film_norm_ratio": float(np.linalg.norm(film))
            / max(float(np.linalg.norm(frozen)), 1e-300),
        }
        arrays[name+"_dp_pa"], arrays[name+"_dh_m"] = dp, dh
        for component, value in terms.items():
            arrays[name+"_"+component+"_kg_m2_s"] = value
    if gpu_op is not None:
        row["cuda_elasticity"].update(
            peak_allocated_bytes=gpu.xp.cuda.max_memory_allocated(),
            peak_reserved_bytes=gpu.xp.cuda.max_memory_reserved(),
        )
    return row, arrays, (problem.x_m.copy(), p[ny//2].copy(), h[ny//2].copy())


def main(sources, directory, *, cuda_elasticity=False):
    directory.mkdir(parents=True, exist_ok=False)
    report = {"status": "incomplete", "runtime": provenance(), "grids": [],
              "scope": "static pressure-mode responses; no temporal/coupled stability claim",
              "production_stencil_changed": False, "mesh_verified": False}
    save_json(directory / "summary.json", report)
    traces = []
    for source in sources:
        row, arrays, trace = audit_source(source, cuda_elasticity=cuda_elasticity)
        name = "x".join(map(str, row["shape_ny_nx"]))
        if any(previous["shape_ny_nx"] == row["shape_ny_nx"] for previous in report["grids"]):
            raise ValueError("duplicate grid would overwrite diagnostic arrays")
        np.savez_compressed(directory / (name+"_directions.npz"), **arrays)
        report["grids"].append(row)
        traces.append((name, trace))
        save_json(directory / "summary.json", report)
        print(name, "independent pressure-mode checks passed", flush=True)
    fig, axes = figure(
        "Accepted 200 N roots: raw centerlines and independent directional transport audit",
        "Roelands/Dowson-Higginson, R=15 mm, E*=113 GPa; fixed finite +/-3a reservoir\n"
        "No filtering or stencil change; static Jacobian responses are not a stability proof",
        rows=3,
    )
    for name, (x, p, h) in traces:
        axes[0].plot(x*1e3, p/1e6, "o-", label=name, markersize=2)
        axes[1].plot(x*1e3, h*1e6, "o-", label=name, markersize=2)
        axes[2].plot(x*1e3, h*1e6, "o-", label=name, markersize=2)
    axes[0].set_ylabel("Pressure [MPa]")
    axes[1].set_ylabel("Total film [um]")
    axes[2].set_ylabel("Centerline film zoom [um]")
    for ax in axes:
        ax.set_xlabel("x along entrainment [mm]")
        ax.set_xlim(-.4, .4)
        ax.legend()
    for ax, half_width in ((axes[1], .4e-3), (axes[2], .25e-3)):
        visible = np.concatenate([h[abs(x) <= half_width]*1e6 for _, (x, _, h) in traces])
        padding = max(.02, .08*np.ptp(visible))
        ax.set_ylim(max(0., visible.min()-padding), visible.max()+padding)
        ax.set_xlim(-half_width*1e3, half_width*1e3)
    save(fig, directory, "raw_centerlines")
    report["status"] = "accepted-root transport checks complete; stability/accuracy open"
    save_json(directory / "summary.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directories", type=Path, nargs="+", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--cuda-elasticity-check", action="store_true")
    args = parser.parse_args()
    main(args.input_directories, args.output_directory,
         cuda_elasticity=args.cuda_elasticity_check)
