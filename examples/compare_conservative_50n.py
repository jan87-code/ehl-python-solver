"""Matching MATLAB-case JFO repair and explicit Hamrock–Dowson comparison.

SI fields on square (ny,nx) grids, +x entrainment, +/-4 Hertz radii. MATLAB P,H0
are an unaccepted initialization; theta=1 is a declared guess, never a root.
Integrated elasticity, arithmetic mobility and conservative pressure/content
equations differ from MATLAB's pressure-only formulation. Capacity transport
defaults to arithmetic faces; donor transport is an explicit experiment.
"""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.io import loadmat
from scipy.signal import convolve2d

from ehl_solver.benchmark_200n import Circular200N
from ehl_solver.constitutive import IsothermalFluid
from ehl_solver.donor_load import DonorLoadSystem, donor_mass_audit
from ehl_solver.elasticity import cell_integrated_kernel
from ehl_solver.film_correlations import hamrock_dowson_point_contact
from ehl_solver.property_coupled import PropertyCoupledSolution
from ehl_solver.property_load_newton import (
    PropertyLoadNewtonError,
    PropertyLoadSystem,
    _solve_property_load_state,
    verify_seed,
)
from ehl_solver.property_reynolds import admissible, mass_audit
from ehl_solver.property_transfer import prolongate_property_guess
from examples.audit_matlab_checkpoint import audit
from examples.run_load_directed_200n import finite_json
from examples.run_scientific_validation import provenance


def matched_fluid(alpha_fraction):
    """Exact MATLAB Roelands constants; alpha_fraction scales exponent z only."""
    if not np.isfinite(alpha_fraction) or not 0 < alpha_fraction <= 1:
        raise ValueError("alpha_fraction must be in (0,1]")
    z = alpha_fraction * 2.21e-8 * 1.98e8 / (np.log(0.04) + 9.67)
    return IsothermalFluid(
        0.04,
        850.0,
        viscosity_index=z,
        viscosity_reference_pa_s=np.exp(-9.67),
        roelands_pressure_pa=1.98e8,
    )


def film_comparison(film, alpha_fraction):
    centre = tuple(n // 2 for n in film.shape)
    reference = hamrock_dowson_point_contact(
        load_n=50.0,
        radius_x_m=0.015,
        contact_modulus_pa=113e9,
        viscosity_pa_s=0.04,
        entrainment_m_s=2.0,
        alpha_per_pa=alpha_fraction * 2.21e-8,
        ellipticity=1.0,
    )
    return {
        **asdict(reference),
        "sampled_minimum_m": float(np.min(film)),
        "sampled_central_m": float(film[centre]),
        "minimum_relative_difference": float(np.min(film) / reference.minimum_film_m - 1),
        "central_relative_difference": float(film[centre] / reference.central_film_m - 1),
        "alpha_definition": "ambient d(log eta)/dp of declared Roelands law",
        "correlation_agreement_is_not_mass_acceptance": True,
    }


def run(
    checkpoint,
    directory,
    *,
    alpha_fraction,
    selection,
    nodes=33,
    seed_source=None,
    transport="arithmetic",
):
    if transport not in ("arithmetic", "donor"):
        raise ValueError("transport must be arithmetic or donor")
    original_audit = audit(checkpoint)
    if not original_audit["passes_saved_matlab_gates"] or original_audit["load_n"] != 50:
        raise ValueError("Verified original 50 N checkpoint required")
    data = loadmat(checkpoint, simplify_cells=True)
    case, fluid = Circular200N(load_n=50.0), matched_fluid(alpha_fraction)
    # The problem's positive reference offset is distinct from the signed
    # load-controlled unknown; total-film feasibility is checked after packing.
    problem = case.problem(0.5e-6, ny=nodes, nx=nodes, extent=4.0)
    system_type = PropertyLoadSystem if transport == "arithmetic" else DonorLoadSystem
    audit_mass = mass_audit if transport == "arithmetic" else donor_mass_audit
    system = system_type(
        problem, fluid, 2.0, 50.0, preconditioner="local", elasticity_mode="cached"
    )
    guess = SimpleNamespace(
        pressure_pa=data["P"] * data["scale"]["pH"], fractional_film_content=np.ones((33, 33))
    )
    seed_info = None
    if seed_source is None:
        if nodes != 33:
            raise ValueError("MATLAB-only initialization requires N33")
        offset = data["H0"] * data["scale"]["H_scale"]
    else:
        previous = json.loads((seed_source / "summary.json").read_text())
        if (
            previous["status"] != "accepted_numerical_root"
            or previous["alpha_fraction"] != alpha_fraction
            or previous["fluid"] != asdict(fluid)
            or previous.get("transport_scheme", "arithmetic") != transport
        ):
            raise ValueError("Accepted same-physics 50 N source required")
        old_n = previous["shape_ny_nx"][0]
        old_problem = case.problem(0.5e-6, ny=old_n, nx=old_n, extent=4.0)
        path = seed_source / f"state_{len(previous['iterations']) - 1:03d}.npz"
        with np.load(path) as a:
            np.testing.assert_array_equal(a["x_m"], old_problem.x_m)
            np.testing.assert_array_equal(a["y_m"], old_problem.y_m)
            seed = PropertyCoupledSolution(
                a["pressure_pa"],
                a["theta"],
                a["film_m"],
                a["displacement_m"],
                old_problem.load_n(a["pressure_pa"]),
                {},
                (),
                1.0,
            )
            offset = float(a["offset_m"])
        if abs(seed.load_n / 50.0 - 1) > 1e-8:
            raise ValueError("Source force is not 50 N")
        if transport == "arithmetic":
            guess = prolongate_property_guess(
                old_problem,
                problem,
                fluid,
                2.0,
                seed=seed,
                seed_offset_m=offset,
                elasticity_mode="cached",
            )
        else:
            old_system = system_type(old_problem, fluid, 2.0, 50.0, elasticity_mode="cached")
            verify_seed(old_system, seed, offset)
            if nodes != 2 * (old_n - 1) + 1:
                raise ValueError("Experimental donor transfer requires nested factor two")
            yy, xx = np.meshgrid(problem.y_m, problem.x_m, indexing="ij")
            points = np.c_[yy.ravel(), xx.ravel()]

            def interpolate(a):
                return RegularGridInterpolator(
                    (old_problem.y_m, old_problem.x_m), a, bounds_error=True
                )(points).reshape(problem.shape)

            gp, gt = interpolate(seed.pressure_pa), interpolate(seed.fractional_film_content)
            if abs(problem.load_n(gp) / seed.load_n - 1) > 1e-12:
                raise ValueError("Donor transfer did not preserve force")
            guess = SimpleNamespace(
                pressure_pa=gp,
                fractional_film_content=gt,
                diagnostics={
                    "accepted": False,
                    "interpolation": "bilinear, no extrapolation",
                    "source_shape_ny_nx": list(old_problem.shape),
                    "target_shape_ny_nx": list(problem.shape),
                    "offset_adjustment_m": 0.0,
                },
            )
        seed_info = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "transfer": guess.diagnostics,
        }
    state = system.pack(guess, offset)
    directory.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "alpha_fraction": alpha_fraction,
        "selection": selection,
        "transport_scheme": transport,
        "source_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "original_checkpoint_audit": original_audit,
        "runtime": provenance(),
        "fluid": asdict(fluid),
        "shape_ny_nx": list(problem.shape),
        "target_load_n": 50.0,
        "entrainment_m_s": 2.0,
        "extent_hertz_radii": 4.0,
        "initial_theta": "uniform one; unaccepted guess"
        if seed_info is None
        else "nested pressure/content interpolation; unaccepted guess",
        "seed": seed_info,
        "equation_differences": [
            "fully cell-integrated elasticity",
            "pressure/content JFO mass equality",
            "arithmetic face mobility/capacity"
            if transport == "arithmetic"
            else "arithmetic face mobility; upstream rho*h*theta advection (first order)",
        ],
        "maximum_corrections": 30,
        "maximum_krylov_cycles": 10,
        "iterations": [],
        "physical_validation_complete": False,
    }

    def write():
        (directory / "summary.json").write_text(
            json.dumps(finite_json(report), indent=2, allow_nan=False)
        )

    def observe(iteration, raw, root):
        sol = root.solution
        index = len(report["iterations"])
        np.savez_compressed(
            directory / f"state_{index:03d}.npz",
            state_dimensionless=raw,
            x_m=problem.x_m,
            y_m=problem.y_m,
            pressure_pa=sol.pressure_pa,
            theta=sol.fractional_film_content,
            film_m=sol.film_thickness_m,
            displacement_m=sol.elastic_displacement_m,
            offset_m=root.central_film_offset_m,
        )
        entry = {
            "iteration": iteration,
            "load_n": sol.load_n,
            "audit": sol.audit,
            "film": film_comparison(sol.film_thickness_m, alpha_fraction),
        }
        report["iterations"].append(entry)
        write()

    write()
    try:
        root = _solve_property_load_state(
            system,
            state,
            30,
            10,
            1e-10,
            True,
            active_lookahead=True,
            lookahead_selection=selection,
            content_bound_pivots=True,
            state_observer=observe,
        )
        report["status"] = "accepted_numerical_root"
        history = root.solution.history
    except PropertyLoadNewtonError as exc:
        root, history = exc.last_root, exc.history
        report.update(status="rejected", reason=str(exc), linear_diagnostics=exc.diagnostics)
    if root is not None:
        sol = root.solution
        p, t, h = sol.pressure_pa, sol.fractional_film_content, sol.film_thickness_m
        direct = convolve2d(
            p,
            cell_integrated_kernel(
                problem.shape,
                problem.x_m[1] - problem.x_m[0],
                problem.y_m[1] - problem.y_m[0],
                113e9,
            ),
            mode="same",
        )
        xx, yy = np.meshgrid(problem.x_m, problem.y_m)
        independent_h = root.central_film_offset_m + (xx * xx + yy * yy) / (2 * 0.015) + direct
        independent_mass = audit_mass(
            problem.x_m,
            problem.y_m,
            independent_h,
            p,
            t,
            fluid,
            2.0,
            length_scale_m=problem.length_scale_m,
            film_scale_m=problem.film_scale_m,
            pressure_scale_pa=problem.pressure_scale_pa,
        )
        mass_ok = admissible(independent_mass, p, t, problem.pressure_scale_pa)
        force = problem.load_n(p)
        verified = mass_ok and abs(force / 50.0 - 1) <= 1e-8
        report["independent_audit"] = {
            "mass": independent_mass,
            "load_n": force,
            "accepted": bool(verified),
            "film_max_error_m": float(np.max(abs(independent_h - h))),
            "displacement_max_error_m": float(np.max(abs(direct - sol.elastic_displacement_m))),
        }
        report["film_comparison"] = film_comparison(independent_h, alpha_fraction)
        if verified != (report["status"] == "accepted_numerical_root"):
            raise RuntimeError("Independent numerical acceptance disagrees")
    report["history"] = history
    source = directory / "source"
    source.mkdir()
    report["source_hashes"] = {}
    for path in [
        Path(__file__),
        Path("src/ehl_solver/property_load_newton.py"),
        Path("src/ehl_solver/film_correlations.py"),
        Path("src/ehl_solver/donor_load.py"),
        Path("src/ehl_solver/transport_candidate.py"),
    ]:
        content = path.read_bytes()
        (source / path.name).write_bytes(content)
        report["source_hashes"][str(path)] = hashlib.sha256(content).hexdigest()
    write()
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("history", "iterations", "runtime", "original_checkpoint_audit")
            },
            indent=2,
            allow_nan=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--alpha", type=float, default=0.8978125)
    parser.add_argument("--selection", choices=("best", "legacy"), default="best")
    parser.add_argument("--nodes", type=int, choices=(33, 65, 129), default=33)
    parser.add_argument("--seed", type=Path)
    parser.add_argument("--transport", choices=("arithmetic", "donor"), default="arithmetic")
    args = parser.parse_args()
    run(
        args.checkpoint,
        args.output,
        alpha_fraction=args.alpha,
        selection=args.selection,
        nodes=args.nodes,
        seed_source=args.seed,
        transport=args.transport,
    )
