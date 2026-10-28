"""Optional MG inside signed full-film Newton: mathematical CPU benchmark only.

This does not implement multigrid for the 200 N material/JFO calculation.
Original Schur/GMRES, load coupling, both line searches and gates are retained.
"""

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ehl_solver.monolithic import (
    BorderedReynoldsElasticityProblem,
    compare_bordered_preconditioners,
    solve_monolithic_full_film_load,
)
from ehl_solver.reynolds import reynolds_residual
from examples.run_load_directed_200n import finite_json
from examples.run_monolithic_load_control_demo import (
    FILM_SCALE_M,
    LENGTH_SCALE_M,
    PRESSURE_SCALE_PA,
    REYNOLDS_PARAMETER,
    _integrated_load,
    _prescribed_pressure,
)
from examples.run_scientific_validation import provenance, save_json


def main(directory):
    directory.mkdir(parents=True, exist_ok=False)
    report = {"status": "incomplete", "runtime": provenance(), "grids": [],
              "scope": "signed constant-property mathematical reference; not material/JFO"}
    save_json(directory / "summary.json", report)
    for ny, nx in ((13, 17), (25, 33), (49, 65)):
        x, y = np.linspace(-.5e-3, .75e-3, nx), np.linspace(-.5e-3, .5e-3, ny)
        target = _integrated_load(_prescribed_pressure(x, y, 20e-6)*PRESSURE_SCALE_PA, x, y)
        initial = _prescribed_pressure(x, y, 18e-6)

        def problem(mode, x=x, y=y, target=target):
            return BorderedReynoldsElasticityProblem(
                x, y, .01, .02, 1e7, LENGTH_SCALE_M, FILM_SCALE_M, PRESSURE_SCALE_PA,
                REYNOLDS_PARAMETER, target, reynolds_residual_scale=100.,
                krylov_rtol=1e-8, krylov_max_iterations=300, preconditioner=mode,
            )

        p = problem("ilu")
        z = np.r_[initial[1:-1, 1:-1].ravel(), .9]
        corrections = compare_bordered_preconditioners(p, z, strategies=("ilu", "multigrid"))
        results, fields = {}, {}
        for mode in ("ilu", "multigrid"):
            def solve(mode=mode, factory=problem, initial=initial):
                return solve_monolithic_full_film_load(factory(mode), initial, 18e-6,
                                                       reynolds_tolerance=1e-9, load_tolerance=1e-9)

            solve()  # excluded warm-up
            samples = []
            for _ in range(3):
                start = time.perf_counter()
                s = solve()
                samples.append(time.perf_counter()-start)
                residual = reynolds_residual(s.pressure_dimensionless,
                                             s.film_thickness_m/FILM_SCALE_M,
                                             p.dx, p.dy, REYNOLDS_PARAMETER)
                mass = float(np.linalg.norm(residual)/np.sqrt(residual.size)/100.)
                load = abs(_integrated_load(s.pressure_pa, x, y)/target-1)
                if mass > 1e-9 or load > 1e-9 or np.min(s.film_thickness_m) <= p.film_floor_m:
                    raise RuntimeError("independent original signed full-film gates failed")
            fields[mode] = s
            results[mode] = {"seconds": samples, "median_s": float(np.median(samples)),
                             "newton_corrections": len(s.history),
                             "operator_evaluations": s.operator_evaluations,
                             "independent_reynolds_scaled": mass, "relative_force_error": load,
                             "history": [asdict(h) for h in s.history]}
        a, b = fields["ilu"], fields["multigrid"]
        error = float(np.linalg.norm(a.pressure_pa-b.pressure_pa)/np.linalg.norm(a.pressure_pa))
        if error > 1e-7 or abs(a.central_film_offset_m-b.central_film_offset_m) > 1e-12:
            raise RuntimeError("ILU/multigrid signed reference roots differ")
        report["grids"].append({"shape_ny_nx": [ny, nx], "solves": results,
                                "corrections": [asdict(c) for c in corrections],
                                "relative_pressure_difference": error,
                                "ilu_over_mg_runtime": results["ilu"]["median_s"]
                                                       /results["multigrid"]["median_s"]})
        save_json(directory / "summary.json", finite_json(report))
    report["status"] = "signed mathematical MG/ILU comparison independently checked"
    save_json(directory / "summary.json", finite_json(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    main(args.output_directory)
