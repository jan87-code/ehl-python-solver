"""Bounded registered mesh repair with replayable snapshots and unchanged gates.

Internal observer logs exact dimensionless P,D,h0/Href and independent SI
fields at each evaluated Newton state; interpolation is never an accepted root.
The observer cannot mutate numerical state and its failures are not solver
convergence failures. All output directories must be new. CPU float64 only.
"""

import argparse
import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.sparse.linalg import splu

from ehl_solver.benchmark_200n import Circular200N
from ehl_solver.property_load_newton import (
    PropertyLoadNewtonError,
    PropertyLoadSystem,
    PropertyStateObserverError,
    _solve_property_load_state,
)
from ehl_solver.property_transfer import _validate_property_guess_fields, prolongate_property_guess
from examples.audit_transport_wavelengths import load_root
from examples.benchmark_cached_material import peak_memory
from examples.run_load_directed_200n import finite_json
from examples.run_property_load_verification import save_state
from examples.run_scientific_validation import provenance, save_json


def main(source, directory, *, factor=2, iterations=30, maximum_physical_nodes=20000,
         field_factorization="full"):
    if factor not in (2, 3, 4) or type(iterations) is not int or not 0 <= iterations <= 30:
        raise ValueError("factor 2/3/4 and diagnostic budget 0..30 required")
    original, state, seed, info = load_root(source)
    case = Circular200N()
    ny, nx = original.problem.shape
    target = case.problem(.5e-6, ny=factor*(ny-1)+1, nx=factor*(nx-1)+1,
                           maximum_physical_nodes=maximum_physical_nodes)
    guess = prolongate_property_guess(original.problem, target, case.fluid(), 2., seed=seed,
                seed_offset_m=state[-1]*original.problem.film_scale_m, elasticity_mode="cached",
                maximum_physical_nodes=maximum_physical_nodes)
    _validate_property_guess_fields(target, guess)
    system = PropertyLoadSystem(target, case.fluid(), 2., 200., preconditioner="local",
                                elasticity_mode="cached",
                                maximum_physical_nodes=maximum_physical_nodes,
                                field_factorization=field_factorization)
    z = system.pack(guess, guess.central_film_offset_m)
    directory.mkdir(parents=True, exist_ok=False)
    report = {"status": "incomplete", "runtime": provenance(), "source": info,
              "case": case.parameters(), "target_n": 200., "shape_ny_nx": list(target.shape),
              "factor": factor, "maximum_iterations": iterations,
              "maximum_physical_nodes": maximum_physical_nodes,
              "field_factorization": field_factorization, "elasticity_mode": "cached",
              "initialization": guess.diagnostics, "snapshots": [],
              "state_scaling": "P=p/pref,D=1-theta,h0/Href; C-order interior (ny-2,nx-2)",
              "source_pressure_scale_pa": target.pressure_scale_pa,
              "source_film_scale_m": target.film_scale_m,
              "scope": "bounded diagnostic run; no new mesh/physical accuracy assertion"}

    def checkpoint():
        save_json(directory / "summary.json", finite_json(report))

    checkpoint()
    # Match the existing >20000-node resource/factor checks; smaller grids also
    # receive the same conservative preflight here. No budget/cap is increased.
    matrix = system.frozen_matrix(z)
    field = matrix[:2*system.n, :2*system.n]
    factorization = (system.reduced_field_factor(z, matrix) if field_factorization == "reduced"
                     else splu(field))
    rhs = np.random.default_rng(29).normal(size=2*system.n)
    error = float(np.linalg.norm(field@factorization.solve(rhs)-rhs)/np.linalg.norm(rhs))
    report["preflight"] = {"factor_true_relative_residual": error, "memory": peak_memory(),
                           "basis_61_bytes": 61*system.size*8,
                           "resident_limit_bytes": 2**30, "factor_residual_limit": 1e-8}
    checkpoint()
    if not np.isfinite(error) or error > 1e-8 or peak_memory()["peak_resident_bytes"] > 2**30:
        report["status"] = "resource/factor preflight rejected; no nonlinear result"
        checkpoint()
        return
    del matrix, field, factorization

    def observe(iteration, exact_state, root):
        path = directory / f"state_{iteration:03d}.npz"
        s = root.solution
        with path.open("xb") as stream:
            np.savez_compressed(stream, state_dimensionless=exact_state,
                                x_m=target.x_m, y_m=target.y_m,
                                pressure_pa=s.pressure_pa, theta=s.fractional_film_content,
                                film_m=s.film_thickness_m, displacement_m=s.elastic_displacement_m,
                                offset_m=root.central_film_offset_m)
        row = {"iteration": iteration, "state_file": path.name,
               "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "audit": s.audit, "load_n": s.load_n, "offset_m": root.central_film_offset_m,
               "minimum_film_m": float(s.film_thickness_m.min()),
               "classification": "evaluated iterate; root status determined only by solver gates"}
        report["snapshots"].append(row)
        checkpoint()
        print(iteration, "mass", s.audit["mass_maximum"], "force N", s.load_n, flush=True)

    start = perf_counter()
    try:
        root = _solve_property_load_state(system, z, iterations, 10, 1e-10, True,
                    active_lookahead=True, lookahead_selection="best", content_bound_pivots=True,
                    state_observer=observe)
    except PropertyLoadNewtonError as exc:
        report.update(status="bounded mesh repair rejected; NOT an accepted root", reason=str(exc),
                      diagnostics=exc.diagnostics)
        history = exc.history
        root = exc.last_root
    except PropertyStateObserverError as exc:
        report.update(status="diagnostic observer failed; no convergence classification",
                      reason=str(exc), observer_reason=str(exc.__cause__))
        checkpoint()
        raise
    else:
        report["status"] = "original gates passed; numerical root only, mesh accuracy open"
        history = root.solution.history
    report.update(elapsed_s=perf_counter()-start, memory=peak_memory())
    save_json(directory / "history.json", finite_json(history))
    if root is not None:
        save_state(directory / "terminal_state.npz", target, root.solution, case.fluid())
        report["terminal_offset_m"] = root.central_film_offset_m
        report["terminal_audit"] = root.solution.audit
        report["terminal_force_n"] = root.solution.load_n
    if (hashlib.sha256(Path(info["source"]).read_bytes()).hexdigest() != info["source_sha256"]
            or hashlib.sha256((source / "summary.json").read_bytes()).hexdigest()
            != info["summary_sha256"]):
        raise RuntimeError("source modified")
    checkpoint()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--refinement-factor", type=int, choices=(2, 3, 4), default=2)
    parser.add_argument("--maximum-iterations", type=int, choices=range(31), default=30)
    parser.add_argument("--maximum-physical-nodes", type=int, choices=(20000, 30000), default=20000)
    parser.add_argument("--field-factorization", choices=("full", "reduced"), default="full")
    args = parser.parse_args()
    main(args.input_directory, args.output_directory, factor=args.refinement_factor,
         iterations=args.maximum_iterations, maximum_physical_nodes=args.maximum_physical_nodes,
         field_factorization=args.field_factorization)
