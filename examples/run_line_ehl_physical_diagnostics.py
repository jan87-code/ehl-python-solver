"""Reproduce the accepted line-EHL case and a published dimensionless benchmark."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import next_fast_len

from ehl_solver.backend import make_backend
from ehl_solver.constitutive import IsothermalFluid, roelands_index_from_ambient_slope
from ehl_solver.film_correlations import dowson_higginson_line_minimum_film
from ehl_solver.geometry import LineContactGeometry
from ehl_solver.hertz import line_contact_hertz
from ehl_solver.line_cavitation import (
    audit_variable_property_line_flux,
    compute_line_mass_flux_components,
)
from ehl_solver.line_contact import PlaneStrainLineElasticityOperator
from ehl_solver.line_ehl import (
    LineContactEHLCase,
    analyze_line_ehl_features,
    audit_line_elasticity_direct,
    solve_line_ehl_at_load,
)

CURRENT_LINE_LOAD_N_M = 1000.0
CURRENT_RADIUS_M = 0.01
CURRENT_E_STAR_PA = 113e9
CURRENT_ETA0_PA_S = 0.04
CURRENT_RHO0_KG_M3 = 850.0
CURRENT_ALPHA_PER_PA = 2.21e-8
CURRENT_SPEED_M_S = 0.1
CURRENT_FILM_SCALE_M = 0.25e-6

VENNER_FIG5_X = np.array(
    [-1.4, -1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
)
VENNER_FIG5_PRESSURE = np.array(
    [0.0175, 0.0417, 0.2193, 0.5635, 0.8003, 0.9187, 0.9801, 0.9977, 0.9746,
     0.9099, 0.7894, 0.6183, 0.0307]
)
VENNER_FIG5_FILM_X = np.array(
    [-1.6, -1.4, -1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6,
     0.8, 1.0, 1.2, 1.4]
)
VENNER_FIG5_FILM = np.array(
    [0.5454, 0.3497, 0.1984, 0.0811, 0.0894, 0.0866, 0.0866, 0.0866, 0.0866,
     0.0866, 0.0872, 0.0888, 0.0904, 0.0811, 0.1743, 0.3618]
)
VENNER_X_UNCERTAINTY = 0.01
VENNER_PRESSURE_UNCERTAINTY = 0.006
VENNER_FILM_UNCERTAINTY = 0.003
VENNER_SPIKE_MINIMUM_X = 0.89
VENNER_SPIKE_MINIMUM = 0.583
VENNER_SPIKE_PEAK_X = 0.9035
VENNER_SPIKE_PEAK = 0.875


def _current_case(node_count: int):
    hertz = line_contact_hertz(
        CURRENT_LINE_LOAD_N_M,
        CURRENT_RADIUS_M,
        CURRENT_E_STAR_PA,
        1.0,
    )
    half_domain_m = 20.0 * hertz.half_width_m
    x_m = np.linspace(-half_domain_m, half_domain_m, node_count)
    fluid = IsothermalFluid(
        CURRENT_ETA0_PA_S,
        CURRENT_RHO0_KG_M3,
        roelands_index_from_ambient_slope(
            CURRENT_ALPHA_PER_PA,
            CURRENT_ETA0_PA_S,
        ),
    )
    case = LineContactEHLCase(
        x_m=x_m,
        geometry=LineContactGeometry(CURRENT_RADIUS_M),
        line_load_n_m=CURRENT_LINE_LOAD_N_M,
        e_star_pa=CURRENT_E_STAR_PA,
        fluid=fluid,
        entrainment_speed_m_s=CURRENT_SPEED_M_S,
        length_scale_m=hertz.half_width_m,
        film_scale_m=CURRENT_FILM_SCALE_M,
        pressure_scale_pa=hertz.peak_pressure_pa,
        elastic_reference_length_m=half_domain_m,
    )
    e_prime_pa = 2.0 * CURRENT_E_STAR_PA
    speed_parameter = CURRENT_ETA0_PA_S * CURRENT_SPEED_M_S / (
        e_prime_pa * CURRENT_RADIUS_M
    )
    load_parameter = CURRENT_LINE_LOAD_N_M / (e_prime_pa * CURRENT_RADIUS_M)
    material_parameter = CURRENT_ALPHA_PER_PA * e_prime_pa
    dimensionless = {
        "E_prime_pa": e_prime_pa,
        "U": speed_parameter,
        "G": material_parameter,
        "W": load_parameter,
        "M": load_parameter / math.sqrt(2.0 * speed_parameter),
        "L": material_parameter * (2.0 * speed_parameter) ** 0.25,
    }
    input_metadata = {
        "line_load_n_m": CURRENT_LINE_LOAD_N_M,
        "radius_m": CURRENT_RADIUS_M,
        "e_star_pa": CURRENT_E_STAR_PA,
        "eta0_pa_s": CURRENT_ETA0_PA_S,
        "rho0_kg_m3": CURRENT_RHO0_KG_M3,
        "alpha_per_pa": CURRENT_ALPHA_PER_PA,
        "entrainment_speed_m_s": CURRENT_SPEED_M_S,
        "domain_x_over_b": [-20.0, 20.0],
        "roelands_viscosity": "existing IsothermalFluid law",
        "density": "existing Dowson-Higginson law",
    }
    return case, hertz, dimensionless, input_metadata, (0.1e-6, 1.0e-6)


def _venner_case(node_count: int):
    """Map Venner & ten Napel's M=20, L=10 case to SI."""
    m_parameter = 20.0
    l_parameter = 10.0
    radius_m = 0.01
    e_prime_pa = 226e9
    e_star_pa = e_prime_pa / 2.0
    alpha_per_pa = 1.71e-8
    roelands_pressure_pa = 1.98e8
    viscosity_index = 0.68
    reference_viscosity_pa_s = math.exp(-9.67)
    eta0_pa_s = math.exp(
        alpha_per_pa * roelands_pressure_pa / viscosity_index - 9.67
    )
    material_parameter = alpha_per_pa * e_prime_pa
    speed_parameter = 0.5 * (l_parameter / material_parameter) ** 4
    load_parameter = m_parameter * math.sqrt(2.0 * speed_parameter)
    line_load_n_m = load_parameter * e_prime_pa * radius_m
    entrainment_speed_m_s = (
        speed_parameter * e_prime_pa * radius_m / eta0_pa_s
    )
    hertz = line_contact_hertz(
        line_load_n_m,
        radius_m,
        e_star_pa,
        1.0,
    )
    half_width_m = hertz.half_width_m
    film_scale_m = half_width_m**2 / radius_m
    x_m = np.linspace(-4.0 * half_width_m, 1.5 * half_width_m, node_count)
    fluid = IsothermalFluid(
        eta0_pa_s,
        850.0,
        viscosity_index,
        reference_viscosity_pa_s,
        roelands_pressure_pa,
    )
    case = LineContactEHLCase(
        x_m=x_m,
        geometry=LineContactGeometry(radius_m),
        line_load_n_m=line_load_n_m,
        e_star_pa=e_star_pa,
        fluid=fluid,
        entrainment_speed_m_s=entrainment_speed_m_s,
        length_scale_m=half_width_m,
        film_scale_m=film_scale_m,
        pressure_scale_pa=hertz.peak_pressure_pa,
        elastic_reference_length_m=0.1 * half_width_m,
    )
    actual_u = eta0_pa_s * entrainment_speed_m_s / (e_prime_pa * radius_m)
    actual_w = line_load_n_m / (e_prime_pa * radius_m)
    dimensionless = {
        "E_prime_pa": e_prime_pa,
        "U": actual_u,
        "G": alpha_per_pa * e_prime_pa,
        "W": actual_w,
        "M": actual_w / math.sqrt(2.0 * actual_u),
        "L": alpha_per_pa * e_prime_pa * (2.0 * actual_u) ** 0.25,
        "requested_M": m_parameter,
        "requested_L": l_parameter,
    }
    input_metadata = {
        "line_load_n_m": line_load_n_m,
        "radius_m": radius_m,
        "e_star_pa": e_star_pa,
        "E_prime_pa": e_prime_pa,
        "eta0_pa_s": eta0_pa_s,
        "viscosity_reference_pa_s": reference_viscosity_pa_s,
        "roelands_pressure_pa": roelands_pressure_pa,
        "viscosity_index": viscosity_index,
        "alpha_per_pa": alpha_per_pa,
        "entrainment_speed_m_s": entrainment_speed_m_s,
        "sum_surface_speed_m_s": 2.0 * entrainment_speed_m_s,
        "rho0_kg_m3": 850.0,
        "density_coefficients_per_pa": [0.6e-9, 1.7e-9],
        "domain_x_over_b": [-4.0, 1.5],
        "elastic_reference_length_over_b": 0.1,
        "elastic_datum_note": (
            "The paper uses b. This equivalent smaller datum shifts the "
            "additive h0 upward without changing the reconstructed physical film."
        ),
        "reference": "Venner and ten Napel (1990), Figures 5-6, Table 1",
        "reference_note": (
            "Figure captions and Table 1 give M=20, L=10; prose states M=30, "
            "an internal source inconsistency."
        ),
    }
    return case, hertz, dimensionless, input_metadata, (
        0.1 * film_scale_m,
        2.0 * film_scale_m,
    )


def _reference_profile_errors(
    source_x: np.ndarray,
    source_values: np.ndarray,
    solution_x: np.ndarray,
    solution_values: np.ndarray,
    solution_scale: float,
) -> dict[str, float]:
    finite = np.isfinite(source_x) & np.isfinite(source_values)
    difference = np.interp(
        source_x[finite],
        solution_x,
        solution_values,
    ) - source_values[finite]
    return {
        "sample_count": int(difference.size),
        "rmse_dimensionless": float(np.sqrt(np.mean(difference**2))),
        "bias_dimensionless": float(np.mean(difference)),
        "maximum_absolute_deviation_dimensionless": float(
            np.max(np.abs(difference))
        ),
        "rmse_physical": float(np.sqrt(np.mean(difference**2)) * solution_scale),
        "bias_physical": float(np.mean(difference) * solution_scale),
        "maximum_absolute_deviation_physical": float(
            np.max(np.abs(difference)) * solution_scale
        ),
    }


def _plot_diagnostics(
    case: LineContactEHLCase,
    hertz,
    solution,
    output_path: Path,
    benchmark: bool,
):
    x_over_b = case.x_m / hertz.half_width_m
    pressure_over_hertz = solution.pressure_pa / hertz.peak_pressure_pa
    film_reduced = (
        solution.film_thickness_m
        * case.geometry.radius_x_m
        / hertz.half_width_m**2
    )
    density, viscosity, _, _ = case.fluid.properties(solution.pressure_pa)
    flux = compute_line_mass_flux_components(
        case.x_m,
        solution.film_thickness_m,
        solution.pressure_pa,
        solution.fractional_film_content,
        case.fluid,
        case.entrainment_speed_m_s,
    )
    face_x_over_b = flux.face_x_m / hertz.half_width_m
    figure, axes = plt.subplots(3, 2, figsize=(13, 12), constrained_layout=True)
    axes[0, 0].plot(x_over_b, pressure_over_hertz, color="tab:blue", label="JFO solution")
    axes[0, 1].plot(x_over_b, film_reduced, color="tab:orange", label="JFO solution")
    if benchmark:
        axes[0, 0].errorbar(
            VENNER_FIG5_X,
            VENNER_FIG5_PRESSURE,
            xerr=VENNER_X_UNCERTAINTY,
            yerr=VENNER_PRESSURE_UNCERTAINTY,
            fmt="o",
            markersize=3,
            capsize=2,
            color="black",
            label="Venner & ten Napel, Fig. 5 digitization",
        )
        axes[0, 0].errorbar(
            [VENNER_SPIKE_MINIMUM_X, VENNER_SPIKE_PEAK_X],
            [VENNER_SPIKE_MINIMUM, VENNER_SPIKE_PEAK],
            xerr=0.0025,
            yerr=0.001,
            fmt="s",
            markersize=4,
            capsize=2,
            color="tab:red",
            label="Fig. 6 / Table 1 outlet extrema",
        )
        axes[0, 1].errorbar(
            VENNER_FIG5_FILM_X,
            VENNER_FIG5_FILM,
            xerr=VENNER_X_UNCERTAINTY,
            yerr=VENNER_FILM_UNCERTAINTY,
            fmt="o",
            markersize=3,
            capsize=2,
            color="black",
            label="Venner & ten Napel, Fig. 5 digitization",
        )

    axes[0, 0].set_ylabel("Pressure / Hertz pressure")
    axes[0, 1].set_ylabel("h R / b^2")
    axes[0, 0].set_title("Pressure")
    axes[0, 1].set_title("Film thickness")
    axes[1, 0].plot(x_over_b, solution.fractional_film_content)
    axes[1, 0].set_title("Fractional film content")
    axes[1, 0].set_ylabel("theta")
    axes[1, 1].plot(x_over_b, density / case.fluid.density_kg_m3)
    axes[1, 1].set_title("Density")
    axes[1, 1].set_ylabel("rho / rho0")
    axes[2, 0].plot(x_over_b, viscosity / case.fluid.viscosity_pa_s)
    axes[2, 0].set_title("Viscosity")
    axes[2, 0].set_ylabel("eta / eta0")
    axes[2, 1].plot(
        face_x_over_b,
        flux.pressure_driven_kg_m_s,
        label="Pressure-driven",
    )
    axes[2, 1].plot(
        face_x_over_b,
        flux.entrainment_kg_m_s,
        label="Entrainment",
    )
    axes[2, 1].plot(
        face_x_over_b,
        flux.total_kg_m_s,
        label="Total",
        linewidth=1.6,
    )
    axes[2, 1].set_title("Mass-flux decomposition")
    axes[2, 1].set_ylabel("kg/(m s)")
    axes[2, 1].legend()
    for axis in axes.flat:
        axis.set_xlabel("x / b")
        axis.grid(True, alpha=0.25)
        if axis in (axes[0, 0], axes[0, 1]) and benchmark:
            axis.legend(fontsize="small")
    figure.suptitle(
        f"Stationary line EHL/JFO diagnostics: {case.x_m.size} nodes",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return density, viscosity, flux


def _profile_csv(path: Path, case, hertz, solution, density, viscosity, flux):
    face_x = np.r_[flux.face_x_m, np.nan]
    fields = np.column_stack(
        (
            case.x_m,
            case.x_m / hertz.half_width_m,
            solution.pressure_pa,
            solution.pressure_pa / hertz.peak_pressure_pa,
            solution.film_thickness_m,
            solution.film_thickness_m
            * case.geometry.radius_x_m
            / hertz.half_width_m**2,
            solution.fractional_film_content,
            density,
            viscosity,
            face_x,
            np.r_[flux.pressure_driven_kg_m_s, np.nan],
            np.r_[flux.entrainment_kg_m_s, np.nan],
            np.r_[flux.total_kg_m_s, np.nan],
        )
    )
    np.savetxt(
        path,
        fields,
        delimiter=",",
        fmt="%.16e",
        header=(
            "x_m,x_over_b,pressure_pa,pressure_over_hertz,film_m,film_R_over_b2,"
            "theta,density_kg_m3,viscosity_pa_s,face_x_m,"
            "pressure_driven_kg_m_s,entrainment_kg_m_s,total_mass_flux_kg_m_s"
        ),
        comments="",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("current", "venner1990"), required=True)
    parser.add_argument("--nodes", type=int, default=1025)
    parser.add_argument("--backend", choices=("cuda", "numpy"), default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.nodes < 33 or args.nodes % 2 == 0 or args.nodes > 4097:
        parser.error("--nodes must be an odd integer between 33 and 4097")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"line_ehl_{args.case}_{args.nodes}"
    figure_path = output_dir / f"{stem}_diagnostics.png"
    metadata_path = output_dir / f"{stem}_metadata.json"
    profiles_path = output_dir / f"{stem}_profiles.csv"
    if any(path.exists() for path in (figure_path, metadata_path, profiles_path)):
        raise FileExistsError(f"refusing to overwrite existing outputs in {output_dir}")

    if args.case == "current":
        case, hertz, groups, inputs, h0_bracket = _current_case(args.nodes)
    else:
        case, hertz, groups, inputs, h0_bracket = _venner_case(args.nodes)
    selected_backend = (
        make_backend("torch", "cuda")
        if args.backend == "cuda"
        else make_backend("numpy")
    )
    start = time.perf_counter()
    root = solve_line_ehl_at_load(
        case,
        h0_bracket,
        backend=selected_backend,
        load_tolerance=1e-5,
        coupling_options={
            "compliance_factors": (0.0, 0.25, 0.5, 0.75, 1.0),
            "film_relaxation": 0.3,
            "pressure_tolerance": 1e-8,
            "film_tolerance": 1e-8,
            "mass_tolerance": 1e-8,
            "maximum_iterations_per_stage": 150,
            "maximum_property_iterations": 100,
        },
    )
    elapsed_seconds = time.perf_counter() - start
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
    reconstructed_film = (
        root.central_film_offset_m + case.geometry_m + direct_displacement
    )
    elasticity_difference = float(
        np.max(np.abs(direct_displacement - solution.elastic_displacement_m))
        / max(np.max(np.abs(direct_displacement)), np.finfo(np.float64).tiny)
    )
    film_reconstruction_maximum_m = float(
        np.max(np.abs(reconstructed_film - solution.film_thickness_m))
    )
    correlation = dowson_higginson_line_minimum_film(
        line_load_n_m=case.line_load_n_m,
        radius_x_m=case.geometry.radius_x_m,
        contact_modulus_pa=case.e_star_pa,
        viscosity_pa_s=case.fluid.viscosity_pa_s,
        entrainment_m_s=case.entrainment_speed_m_s,
        alpha_per_pa=(
            CURRENT_ALPHA_PER_PA if args.case == "current" else 1.71e-8
        ),
    )
    density, viscosity, flux = _plot_diagnostics(
        case,
        hertz,
        solution,
        figure_path,
        args.case == "venner1990",
    )
    _profile_csv(profiles_path, case, hertz, solution, density, viscosity, flux)

    comparisons = None
    if args.case == "venner1990":
        reduced_film = (
            solution.film_thickness_m
            * case.geometry.radius_x_m
            / hertz.half_width_m**2
        )
        comparisons = {
            "figure_5_bulk_pressure": _reference_profile_errors(
                VENNER_FIG5_X[3:12],
                VENNER_FIG5_PRESSURE[3:12],
                case.x_m / hertz.half_width_m,
                solution.pressure_pa / hertz.peak_pressure_pa,
                hertz.peak_pressure_pa,
            ),
            "figure_5_bulk_film": _reference_profile_errors(
                VENNER_FIG5_FILM_X[4:12],
                VENNER_FIG5_FILM[4:12],
                case.x_m / hertz.half_width_m,
                reduced_film,
                case.film_scale_m,
            ),
            "figure_5_outlet_film": _reference_profile_errors(
                VENNER_FIG5_FILM_X[12:],
                VENNER_FIG5_FILM[12:],
                case.x_m / hertz.half_width_m,
                reduced_film,
                case.film_scale_m,
            ),
            "figure_6_table_1_pre_spike_minimum": {
                "x_over_b_reported_or_digitized": VENNER_SPIKE_MINIMUM_X,
                "pressure_over_hertz_reported": VENNER_SPIKE_MINIMUM,
                "pressure_pa": VENNER_SPIKE_MINIMUM * hertz.peak_pressure_pa,
            },
            "figure_6_table_1_spike": {
                "x_over_b_digitized": VENNER_SPIKE_PEAK_X,
                "pressure_over_hertz_reported": VENNER_SPIKE_PEAK,
                "pressure_pa": VENNER_SPIKE_PEAK * hertz.peak_pressure_pa,
                "computed_spike_location_over_b": (
                    None
                    if features.petrusevich_spike_location_m is None
                    else features.petrusevich_spike_location_m / hertz.half_width_m
                ),
                "computed_spike_pressure_over_hertz": (
                    None
                    if features.petrusevich_spike_location_m is None
                    else float(
                        np.interp(
                            features.petrusevich_spike_location_m,
                            case.x_m,
                            solution.pressure_pa,
                        )
                        / hertz.peak_pressure_pa
                    )
                ),
            },
        }

    half_domain = max(abs(float(case.x_m[0])), abs(float(case.x_m[-1])))
    fft_size = next_fast_len(3 * case.x_m.size - 2)
    metadata = {
        "case": args.case,
        "nodes": args.nodes,
        "inputs_si": inputs,
        "dimensionless_groups": groups,
        "scales": {
            "hertz_half_width_m": hertz.half_width_m,
            "hertz_pressure_pa": hertz.peak_pressure_pa,
            "length_scale_m": case.length_scale_m,
            "film_scale_m": case.film_scale_m,
            "pressure_scale_pa": case.pressure_scale_pa,
            "elastic_reference_length_m": case.elastic_reference_length_m,
            "x_domain_m": [float(case.x_m[0]), float(case.x_m[-1])],
            "domain_half_width_over_b": half_domain / hertz.half_width_m,
        },
        "solver": {
            "elasticity_backend": solution.backend,
            "reynolds_active_set_backend": "CPU/SciPy sparse",
            "independent_audits": "CPU SI mass flux and direct plane-strain convolution",
            "elapsed_seconds": elapsed_seconds,
            "load_root_trials": len(root.trials),
            "nonlinear_iterations": len(solution.history),
            "fft_workspace_estimate_bytes": 64 * fft_size,
        },
        "root": {
            "h0_m": root.central_film_offset_m,
            "target_line_load_n_m": root.target_line_load_n_m,
            "calculated_line_load_n_m": root.calculated_line_load_n_m,
            "relative_load_error": root.relative_load_error,
            "features": vars(features),
            "minimum_film_correlation_m": correlation.minimum_film_m,
        },
        "independent_audits": {
            "si_mass_flux": independent_mass,
            "direct_vs_fft_elasticity_relative_error": elasticity_difference,
            "direct_film_reconstruction_maximum_m": film_reconstruction_maximum_m,
            "relative_total_mass_flux_range": float(
                np.ptp(flux.total_kg_m_s)
                / max(
                    np.max(np.abs(flux.total_kg_m_s)),
                    np.finfo(np.float64).tiny,
                )
            ),
        },
        "published_profile_comparison": comparisons,
        "digitization_uncertainty": (
            {
                "figure_5_x_over_b": VENNER_X_UNCERTAINTY,
                "figure_5_pressure_over_hertz": VENNER_PRESSURE_UNCERTAINTY,
                "figure_5_hR_over_b2": VENNER_FILM_UNCERTAINTY,
                "figure_6_peak_x_over_b": 0.0025,
                "figure_6_table_pressure_over_hertz": 0.001,
                "note": "Conservative scan/digitization bounds, not solver uncertainty.",
            }
            if args.case == "venner1990"
            else None
        ),
        "artifacts": {
            "diagnostic_figure": str(figure_path),
            "profile_csv": str(profiles_path),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(json.dumps(metadata, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
