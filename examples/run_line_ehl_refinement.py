"""Reproduce the stationary line-EHL/JFO uniform-grid study."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from ehl_solver.backend import make_backend
from ehl_solver.constitutive import (
    IsothermalFluid,
    roelands_index_from_ambient_slope,
)
from ehl_solver.film_correlations import dowson_higginson_line_minimum_film
from ehl_solver.geometry import LineContactGeometry
from ehl_solver.hertz import line_contact_hertz
from ehl_solver.line_cavitation import audit_variable_property_line_flux
from ehl_solver.line_contact import PlaneStrainLineElasticityOperator
from ehl_solver.line_ehl import (
    LineContactEHLCase,
    analyze_line_ehl_features,
    audit_line_elasticity_direct,
    solve_line_ehl_at_load,
    solve_line_ehl_at_offset,
)

LINE_LOAD_N_M = 1000.0
RADIUS_M = 0.01
E_STAR_PA = 113e9
ETA0_PA_S = 0.04
ALPHA_PER_PA = 2.21e-8
DENSITY0_KG_M3 = 850.0
ENTRAINMENT_M_S = 0.1
FILM_SCALE_M = 0.25e-6
DOMAIN_HALF_WIDTH_HERTZ = 20.0
DEFAULT_GRIDS = (257, 513, 1025, 2049, 4097)


def _case(grid_size: int):
    hertz = line_contact_hertz(LINE_LOAD_N_M, RADIUS_M, E_STAR_PA, 1.0)
    half_domain_m = DOMAIN_HALF_WIDTH_HERTZ * hertz.half_width_m
    hertz = line_contact_hertz(
        LINE_LOAD_N_M,
        RADIUS_M,
        E_STAR_PA,
        half_domain_m,
    )
    x_m = np.linspace(-half_domain_m, half_domain_m, grid_size)
    fluid = IsothermalFluid(
        ETA0_PA_S,
        DENSITY0_KG_M3,
        roelands_index_from_ambient_slope(ALPHA_PER_PA, ETA0_PA_S),
    )
    return (
        LineContactEHLCase(
            x_m=x_m,
            geometry=LineContactGeometry(RADIUS_M),
            line_load_n_m=LINE_LOAD_N_M,
            e_star_pa=E_STAR_PA,
            fluid=fluid,
            entrainment_speed_m_s=ENTRAINMENT_M_S,
            length_scale_m=hertz.half_width_m,
            film_scale_m=FILM_SCALE_M,
            pressure_scale_pa=hertz.peak_pressure_pa,
            elastic_reference_length_m=half_domain_m,
        ),
        hertz,
        half_domain_m,
    )


def _profile_change(previous, current, hertz_half_width_m):
    x_previous, p_previous, h_previous = previous
    x_current, p_current, h_current = current
    window = np.abs(x_previous) <= 5.0 * hertz_half_width_m
    p_interpolated = np.interp(x_previous[window], x_current, p_current)
    h_interpolated = np.interp(x_previous[window], x_current, h_current)
    pressure_error = np.linalg.norm(p_interpolated - p_previous[window]) / max(
        np.linalg.norm(p_interpolated), np.finfo(np.float64).tiny
    )
    film_error = np.linalg.norm(h_interpolated - h_previous[window]) / max(
        np.linalg.norm(h_interpolated), np.finfo(np.float64).tiny
    )
    return {
        "compared_nodes": int(np.count_nonzero(window)),
        "pressure_relative_l2": float(pressure_error),
        "film_relative_l2": float(film_error),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grids",
        default=",".join(str(value) for value in DEFAULT_GRIDS),
        help="comma-separated odd node counts on the unchanged +/-20b domain",
    )
    parser.add_argument(
        "--backend",
        choices=("torch", "numpy"),
        default="torch",
        help="default torch so the verified CUDA elasticity path is used",
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
        help="torch device; CUDA is required unless explicitly set to cpu",
    )
    parser.add_argument(
        "--compare-backends",
        action="store_true",
        help="time CPU NumPy and CUDA coupled corrections on the finest requested grid",
    )
    args = parser.parse_args()
    grids = tuple(int(value) for value in args.grids.split(","))
    if (
        len(grids) < 2
        or any(value < 33 or value % 2 == 0 for value in grids)
        or any(right <= left for left, right in zip(grids, grids[1:], strict=False))
    ):
        parser.error("grids must be at least two strictly increasing odd counts >=33")
    selected_backend = (
        make_backend("numpy")
        if args.backend == "numpy"
        else make_backend("torch", args.device)
    )
    study = []
    previous = None
    finest_case = finest_solution = None
    previous_grid = None
    coupling = {
        "compliance_factors": (0.0, 0.25, 0.5, 0.75, 1.0),
        "film_relaxation": 0.3,
        "pressure_tolerance": 1e-8,
        "film_tolerance": 1e-8,
        "mass_tolerance": 1e-8,
        "maximum_iterations_per_stage": 150,
        "maximum_property_iterations": 100,
    }

    for grid_size in grids:
        case, hertz, half_domain_m = _case(grid_size)
        started = time.perf_counter()
        root = solve_line_ehl_at_load(
            case,
            (0.1e-6, 1.0e-6),
            backend=selected_backend,
            load_tolerance=1e-5,
            coupling_options=coupling,
        )
        elapsed = time.perf_counter() - started
        solution = root.coupled_solution
        features = analyze_line_ehl_features(
            case.x_m,
            solution.pressure_pa,
            solution.film_thickness_m,
            hertz.half_width_m,
        )
        independent_mass = audit_variable_property_line_flux(
            case.x_m,
            solution.film_thickness_m,
            solution.pressure_pa,
            solution.fractional_film_content,
            case.fluid,
            case.entrainment_speed_m_s,
            length_scale_m=case.length_scale_m,
            film_scale_m=case.film_scale_m,
            pressure_scale_pa=case.pressure_scale_pa,
        )
        cpu_operator = PlaneStrainLineElasticityOperator(
            case.x_m,
            case.e_star_pa,
            case.elastic_reference_length_m,
            make_backend("numpy"),
        )
        direct_displacement = audit_line_elasticity_direct(
            solution.pressure_pa,
            cpu_operator,
        )
        elasticity_error = float(
            np.max(np.abs(direct_displacement - solution.elastic_displacement_m))
            / max(
                np.max(np.abs(direct_displacement)),
                np.finfo(np.float64).tiny,
            )
        )
        hertz_corr = dowson_higginson_line_minimum_film(
            line_load_n_m=case.line_load_n_m,
            radius_x_m=RADIUS_M,
            contact_modulus_pa=E_STAR_PA,
            viscosity_pa_s=ETA0_PA_S,
            entrainment_m_s=ENTRAINMENT_M_S,
            alpha_per_pa=ALPHA_PER_PA,
        )
        current = (
            case.x_m,
            solution.pressure_pa,
            solution.film_thickness_m,
        )
        record = {
            "grid_nodes": grid_size,
            "domain_half_width_over_hertz_b": half_domain_m / hertz.half_width_m,
            "backend": solution.backend,
            "elapsed_seconds": elapsed,
            "central_film_m": features.central_film_m,
            "minimum_film_m": features.minimum_film_m,
            "minimum_film_location_m": features.minimum_film_location_m,
            "correlation_minimum_film_m": hertz_corr.minimum_film_m,
            "peak_pressure_pa": features.peak_pressure_pa,
            "peak_pressure_location_m": features.peak_pressure_location_m,
            "outlet_constriction_film_m": features.outlet_constriction_film_m,
            "outlet_constriction_location_m": features.outlet_constriction_location_m,
            "petrusevich_spike_location_m": features.petrusevich_spike_location_m,
            "petrusevich_spike_width_m": features.petrusevich_spike_width_m,
            "petrusevich_spike_integrated_pressure_pa_m": (
                features.petrusevich_spike_integrated_pressure_pa_m
            ),
            "line_load_n_m": solution.calculated_line_load_n_m,
            "relative_line_load_error": root.relative_load_error,
            "mass_audit": independent_mass,
            "direct_elasticity_relative_error": elasticity_error,
            "nonlinear_iterations": len(solution.history),
            "load_root_trials": len(root.trials),
        }
        if previous is not None:
            record["change_from_previous"] = _profile_change(
                previous,
                current,
                hertz.half_width_m,
            )
            record["change_from_previous"]["previous_grid_nodes"] = previous_grid
        study.append(record)
        previous = current
        previous_grid = grid_size
        finest_case, finest_solution = case, solution

    backend_comparison = None
    if args.compare_backends:
        options = dict(coupling)
        options["initial_pressure_pa"] = finest_solution.pressure_pa
        timing = {}
        outputs = {}
        for name, backend in (
            ("numpy", make_backend("numpy")),
            ("cuda", make_backend("torch", "cuda")),
        ):
            start = time.perf_counter()
            corrected = solve_line_ehl_at_offset(
                finest_case,
                finest_solution.central_film_offset_m,
                backend=backend,
                **options,
            )
            timing[name] = time.perf_counter() - start
            outputs[name] = corrected
        numpy_root, cuda_root = outputs["numpy"], outputs["cuda"]
        backend_comparison = {
            "grid_nodes": grids[-1],
            "numpy_seconds": timing["numpy"],
            "cuda_seconds": timing["cuda"],
            "cuda_over_numpy_speedup": timing["numpy"] / timing["cuda"],
            "maximum_relative_pressure_difference": float(
                np.max(
                    np.abs(cuda_root.pressure_pa - numpy_root.pressure_pa)
                )
                / max(
                    np.max(np.abs(numpy_root.pressure_pa)),
                    np.finfo(np.float64).tiny,
                )
            ),
            "maximum_relative_film_difference": float(
                np.max(
                    np.abs(
                        cuda_root.film_thickness_m
                        - numpy_root.film_thickness_m
                    )
                )
                / max(
                    np.max(np.abs(numpy_root.film_thickness_m)),
                    np.finfo(np.float64).tiny,
                )
            ),
            "relative_load_difference": abs(
                cuda_root.calculated_line_load_n_m
                / numpy_root.calculated_line_load_n_m
                - 1.0
            ),
        }

    print(
        json.dumps(
            {
                "parameters": {
                    "line_load_n_m": LINE_LOAD_N_M,
                    "radius_m": RADIUS_M,
                    "e_star_pa": E_STAR_PA,
                    "eta0_pa_s": ETA0_PA_S,
                    "density0_kg_m3": DENSITY0_KG_M3,
                    "ambient_alpha_per_pa": ALPHA_PER_PA,
                    "entrainment_m_s": ENTRAINMENT_M_S,
                    "domain_half_width_hertz_b": DOMAIN_HALF_WIDTH_HERTZ,
                    "fluid_law": "existing Roelands viscosity and Dowson-Higginson density",
                    "film_correlation": (
                        "Dowson-Higginson line h_min proximity equation; "
                        "not an acceptance threshold"
                    ),
                },
                "grids": study,
                "backend_comparison": backend_comparison,
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
