"""CPU CG vs geometric V-cycle preconditioning; preserve all failed attempts.

Dimensionless frozen-film SPD diffusion only, not coupled material/JFO Newton.
Setup and warmed solve times are separate; independently measured true residual
is mandatory. Loaded mobility surrogates are NOT full material Jacobians.
"""

import argparse
import time
from pathlib import Path

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import LinearOperator, cg

from ehl_solver.multigrid import ReynoldsMultigrid
from examples.audit_transport_wavelengths import load_root
from examples.benchmark_cached_material import peak_memory
from examples.run_scientific_validation import provenance, save_json


def assess(film, dx, dy, *, maximum_iterations=1000):
    start = time.perf_counter()
    mg = ReynoldsMultigrid(film, dx, dy)
    setup = time.perf_counter()-start
    matrix = mg.matrix
    rhs = np.random.default_rng(57).normal(size=matrix.shape[0])
    rows = {}
    mg.apply(rhs)  # exclude first-cycle initialization from solve timings
    for name, inverse in (("cg", None), ("multigrid_cg", mg.aslinearoperator())):
        samples, counts, errors = [], [], []
        for _ in range(3):
            count = []
            start = time.perf_counter()
            answer, info = cg(matrix, rhs, M=inverse, rtol=1e-10, atol=0,
                              maxiter=maximum_iterations,
                              callback=lambda x, count=count: count.append(1))
            samples.append(time.perf_counter()-start)
            measured = float(np.linalg.norm(matrix@answer-rhs)/np.linalg.norm(rhs))
            counts.append(len(count))
            errors.append(measured)
            if info != 0 or measured > 1e-10*(1+1e-12):
                break
        rows[name] = {"accepted": info == 0 and max(errors) <= 1e-10*(1+1e-12),
                      "scipy_info": int(info), "iterations": counts,
                      "true_relative_residuals": errors, "seconds": samples,
                      "median_s": float(np.median(samples))}
    return {"shape_ny_nx": list(np.shape(film)), "setup_s": setup,
            "hierarchy": mg.diagnostics, "solves": rows,
            "solve_speedup": rows["cg"]["median_s"]/rows["multigrid_cg"]["median_s"],
            "speedup_meaningful": all(v["accepted"] for v in rows.values()),
            "memory": peak_memory()}


def known_pressure_check(film, dx, dy):
    """Forward error independent of residual; symmetric scaling controls units.

    This deliberately checks a pressure-like RHS A*p_exact, whose weak rows
    can be hidden by a global unscaled residual. Scaling is benchmark-only;
    no material Newton stopping rule is changed.
    """
    mg = ReynoldsMultigrid(film, dx, dy)
    y, x = np.meshgrid(np.linspace(0, 1, film.shape[0]),
                       np.linspace(0, 1, film.shape[1]), indexing="ij")
    exact = (np.sin(np.pi*x)*np.sin(np.pi*y))[1:-1, 1:-1].ravel()
    a = mg.matrix
    rhs = a@exact
    scale = 1/np.sqrt(a.diagonal())
    s = diags(scale)
    equilibrated = s@a@s
    inverse = LinearOperator(a.shape, matvec=lambda v: mg.apply(v/scale)/scale, dtype=np.float64)
    answer, info = cg(equilibrated, scale*rhs, M=inverse, rtol=1e-12, atol=0, maxiter=500)
    answer = scale*answer
    error = float(np.linalg.norm(answer-exact)/np.linalg.norm(exact))
    residual = float(np.linalg.norm(a@answer-rhs)/np.linalg.norm(rhs))
    return {"scipy_info": int(info), "relative_pressure_error": error,
            "original_true_relative_residual": residual,
            "passed": info == 0 and error < 1e-7 and residual < 1e-10}


def main(directory, sources):
    directory.mkdir(parents=True, exist_ok=False)
    report = {"status": "incomplete", "runtime": provenance(), "smooth": [], "loaded": [],
              "scope": "frozen-film SPD Reynolds only; no coupled Newton multigrid"}
    save_json(directory / "summary.json", report)
    for n in (33, 65, 129, 257, 513):
        x = np.linspace(0, 1, n)
        xx, yy = np.meshgrid(x, x)
        row = assess(1+.4*xx+.2*np.sin(np.pi*yy), x[1], x[1], maximum_iterations=3000)
        report["smooth"].append(row)
        save_json(directory / "summary.json", report)
        print(n, row["solves"], flush=True)
    for source in sources:
        system, _, seed, info = load_root(source, maximum_physical_nodes=30000)
        p = system.problem
        rho, eta, _, _ = system.base.fluid.ratios(seed.pressure_pa)
        mobility = rho*(seed.film_thickness_m/p.film_scale_m)**3/eta
        # Cube root only represents the prescribed positive mobility for the
        # existing SPD assembly; material/elastic derivative terms are absent.
        row = assess(np.cbrt(mobility), p.dx, p.dy, maximum_iterations=500)
        row.update(source=info, minimum_mobility=float(mobility.min()),
                   maximum_mobility=float(mobility.max()),
                   known_pressure_check=known_pressure_check(np.cbrt(mobility), p.dx, p.dy))
        report["loaded"].append(row)
        save_json(directory / "summary.json", report)
    report["status"] = "frozen-film multigrid assessment complete; inspect every acceptance flag"
    save_json(directory / "summary.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--input-directories", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    main(args.output_directory, args.input_directories)
