"""Nested fixed-reservoir force refinement: unaccepted interpolation, audited repair."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ehl_solver.benchmark_200n import Circular200N
from ehl_solver.elasticity_audit import direct_midpoint_displacement
from ehl_solver.property_coupled import PropertyCoupledSolution
from ehl_solver.property_load_newton import PropertyLoadNewtonError
from ehl_solver.property_transfer import prolongate_property_guess, repair_property_load_guess
from examples.run_load_directed_200n import finite_json
from examples.run_property_load_verification import save_state
from examples.run_scientific_validation import provenance, save_json
from examples.scientific_figures import figure, save


def main(
    source,
    directory,
    load_n,
    active_lookahead=True,
    refinement_factor=2,
    lookahead_selection="best",
    content_bound_pivots=True,
):
    if refinement_factor not in (2, 3):
        raise ValueError("refinement factor must be 2 or 3")
    old = json.loads((source / "summary.json").read_text())
    c = Circular200N()
    fluid = c.fluid()
    if old["case"] != c.parameters():
        raise ValueError("registered circular case required")
    grid = old["grids"]["refined"]
    ny, nx = grid["shape_ny_nx"]
    candidates = [
        e for e in grid["attempts"] if e["accepted"] and abs(e["load_n"] / load_n - 1) <= 1e-8
    ]
    if not candidates:
        raise ValueError("source must contain an independently force-closed requested load")
    entry = candidates[-1]
    source_path = source / entry["state_file"]
    coarse = c.problem(0.5e-6, ny=ny, nx=nx)
    target = c.problem(
        0.5e-6, ny=refinement_factor * (ny - 1) + 1, nx=refinement_factor * (nx - 1) + 1
    )
    with np.load(source_path) as a:
        np.testing.assert_array_equal(a["x_m"], coarse.x_m)
        np.testing.assert_array_equal(a["y_m"], coarse.y_m)
        seed = PropertyCoupledSolution(
            a["pressure_pa"],
            a["theta"],
            a["film_m"],
            a["displacement_m"],
            coarse.load_n(a["pressure_pa"]),
            {},
            (),
            1.0,
        )
    directory.mkdir(parents=True, exist_ok=False)
    output_grid = {
        "shape_ny_nx": list(target.shape),
        "input_npz": str(source_path),
        "input_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "attempts": [],
        "closed_loads": [],
        "force_200n_closed": False,
    }
    report = {
        "status": "incomplete",
        "runtime": provenance(),
        "case": c.parameters(),
        "target_n": load_n,
        "grids": {"refined": output_grid},
        "mesh_verified": False,
        "physical_domain_verified": False,
        "interpolated_guess_accepted": False,
        "active_lookahead": active_lookahead,
        "lookahead_selection": lookahead_selection,
        "content_bound_pivots": content_bound_pivots,
        "refinement_factor": refinement_factor,
        "matched_loads": [],
    }

    def checkpoint():
        save_json(directory / "summary.json", finite_json(report))

    checkpoint()
    guess = prolongate_property_guess(
        coarse, target, fluid, 2.0, seed=seed, seed_offset_m=entry["offset_m"]
    )
    h, w = target.fields(guess.pressure_pa, central_film_offset_m=guess.central_film_offset_m)
    np.savez_compressed(
        directory / "unaccepted_guess.npz",
        x_m=target.x_m,
        y_m=target.y_m,
        pressure_pa=guess.pressure_pa,
        theta=guess.fractional_film_content,
        film_m=h,
        displacement_m=w,
    )
    report["initialization"] = guess.diagnostics
    checkpoint()
    try:
        root = repair_property_load_guess(
            target,
            fluid,
            2.0,
            load_n,
            guess,
            active_lookahead=active_lookahead,
            lookahead_selection=lookahead_selection,
            content_bound_pivots=content_bound_pivots,
        )
    except PropertyLoadNewtonError as exc:
        save_json(
            directory / "failure.json",
            finite_json({"history": exc.history, "diagnostics": exc.diagnostics}),
        )
        if exc.last_root is not None:
            save_state(directory / "failed_partial.npz", target, exc.last_root.solution, fluid)
            report["failed_offset_m"] = exc.last_root.central_film_offset_m
        output_grid["stall"] = {"reason": str(exc), "target_n": load_n}
        report["status"] = "bounded refinement repair failed; no target root accepted"
        checkpoint()
        return
    s = root.solution
    save_state(directory / "solution.npz", target, s, fluid)
    save_json(directory / "history.json", finite_json(s.history))
    peak = np.unravel_index(np.argmax(s.pressure_pa), s.pressure_pa.shape)
    direct = direct_midpoint_displacement(
        target.x_m,
        target.y_m,
        s.pressure_pa,
        c.contact_modulus_pa,
        [[0.0, 0.0], [target.x_m[peak[1]], target.y_m[peak[0]]]],
    )
    direct_error = float(
        abs(
            direct
            - [
                s.elastic_displacement_m[target.shape[0] // 2, target.shape[1] // 2],
                s.elastic_displacement_m[peak],
            ]
        ).max()
    )
    if direct_error > 1e-18:
        raise RuntimeError("direct source check failed")
    record = {
        "accepted": True,
        "target_load_n": load_n,
        "load_n": s.load_n,
        "offset_m": root.central_film_offset_m,
        "minimum_film_m": float(s.film_thickness_m.min()),
        "central_film_m": float(s.film_thickness_m[target.shape[0] // 2, target.shape[1] // 2]),
        "peak_pressure_pa": float(s.pressure_pa.max()),
        "audit": s.audit,
        "direct_error_m": direct_error,
        "state_file": "solution.npz",
    }
    output_grid["attempts"].append(record)
    output_grid["closed_loads"].append(record)
    output_grid["force_200n_closed"] = load_n == 200.0
    report["matched_loads"] = [
        {
            "target_n": load_n,
            "coarse": entry,
            "refined": record,
            "relative_minimum_film_change": record["minimum_film_m"] / entry["minimum_film_m"] - 1,
            "relative_central_film_change": record["central_film_m"] / entry["central_film_m"] - 1,
            "relative_peak_pressure_change": record["peak_pressure_pa"] / entry["peak_pressure_pa"]
            - 1,
        }
    ]
    fig, axes = figure(
        "Matched-force nested refinement; raw centerlines",
        f"Q={load_n:g} N; same physical +/-3a reservoir, R=15 mm, E_contact=113 GPa, Ue=2 m/s\n"
        "Roelands / Dowson-Higginson; interpolation alone is never accepted",
        rows=2,
    )
    for p, solution, label in ((coarse, seed, str(coarse.shape)), (target, s, str(target.shape))):
        row = p.shape[0] // 2
        axes[0].plot(p.x_m * 1e3, solution.pressure_pa[row] / 1e6, "o-", label=label)
        axes[1].plot(p.x_m * 1e3, solution.film_thickness_m[row] * 1e6, "o-", label=label)
    axes[0].set_ylabel("Pressure [MPa]")
    axes[1].set_ylabel("Total film [um]")
    for ax in axes:
        ax.set_xlabel("x along entrainment [mm]")
        ax.legend()
        ax.set_xlim(-0.4, 0.4)
    save(fig, directory, "matched_centerlines")
    report["figure_manifest"] = ["matched_centerlines.png"]
    report["status"] = "force-closed nested refinement; physical/mesh accuracy open"
    checkpoint()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--load-n", type=float, default=200.0)
    parser.add_argument("--no-active-lookahead", action="store_true")
    parser.add_argument("--refinement-factor", type=int, choices=(2, 3), default=2)
    parser.add_argument("--lookahead-selection", choices=("best", "legacy"), default="best")
    parser.add_argument("--no-content-bound-pivots", action="store_true")
    args = parser.parse_args()
    main(
        args.input_directory,
        args.output_directory,
        args.load_n,
        not args.no_active_lookahead,
        args.refinement_factor,
        args.lookahead_selection,
        not args.no_content_bound_pivots,
    )
