"""Initial uniform compressible/piezoviscous elastic cases; bounded CPU verification."""

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ehl_solver.cavitating_coupled import CavitatingElasticityProblem
from ehl_solver.cavitating_newton import solve_cavitating_newton
from ehl_solver.constitutive import IsothermalFluid
from ehl_solver.elasticity_audit import direct_midpoint_displacement
from ehl_solver.property_coupled import PropertyCouplingError, solve_property_coupling
from ehl_solver.property_reynolds import admissible, mass_audit
from examples.property_monolithic_reference import small_monolithic_reference
from examples.run_farfield_budget import H0, LY, P_REF, RX, RY, L, fronts
from examples.run_scientific_validation import provenance, save_json
from examples.scientific_figures import figure, save


def parameters(x, y):
    return CavitatingElasticityProblem(x, y, H0, RX, RY, 1e11, L, H0, P_REF * 100, 1)


def archive(directory, name, problem, r, fluid):
    rho, eta, _, _ = fluid.properties(r.pressure_pa, strength=r.constitutive_strength)
    np.savez_compressed(
        directory / f"{name}.npz",
        x_m=problem.x_m,
        y_m=problem.y_m,
        pressure_pa=r.pressure_pa,
        theta=r.fractional_film_content,
        film_m=r.film_thickness_m,
        displacement_m=r.elastic_displacement_m,
        density_kg_m3=rho,
        viscosity_pa_s=eta,
    )
    j, i = np.unravel_index(r.pressure_pa.argmax(), r.pressure_pa.shape)
    points = [[0, 0], [float(problem.x_m[i]), float(problem.y_m[j])]]
    direct = direct_midpoint_displacement(problem.x_m, problem.y_m, r.pressure_pa, 1e11, points)
    # (0,0) is on both symmetric benchmark grids; small nonsymmetric grid is checked at peak only.
    peak_error = abs(direct[1] - r.elastic_displacement_m[j, i])
    q_rectangle = float(r.pressure_pa[1:-1, 1:-1].sum() * problem.dx_m * problem.dy_m)
    exact, _ = problem.fields(r.pressure_pa)
    entry = {
        "name": name,
        "shape": list(problem.shape),
        "state_file": f"{name}.npz",
        "x_limits_m": [float(problem.x_m[0]), float(problem.x_m[-1])],
        "y_limits_m": [float(problem.y_m[0]), float(problem.y_m[-1])],
        "load_n": r.load_n,
        "minimum_film_m": float(r.film_thickness_m.min()),
        "maximum_pressure_pa": float(r.pressure_pa.max()),
        "pressure_min_pa": float(r.pressure_pa.min()),
        "maximum_density_ratio": float(rho.max() / fluid.density_kg_m3),
        "maximum_viscosity_ratio": float(eta.max() / fluid.viscosity_pa_s),
        "direct_peak_displacement_error_m": float(peak_error),
        "independent_rectangular_load_error_n": abs(q_rectangle - r.load_n),
        "maximum_film_coupling_error_m": float(abs(exact - r.film_thickness_m).max()),
        "audit": r.audit,
        "history": r.history,
        "fronts": fronts(problem.x_m, r.fractional_film_content),
    }
    entry["load_quadrature_relative_error"] = entry["independent_rectangular_load_error_n"] / abs(
        r.load_n
    )
    entry["accepted"] = bool(
        admissible(r.audit, r.pressure_pa, r.fractional_film_content, problem.pressure_scale_pa)
        and entry["direct_peak_displacement_error_m"] <= 1e-18
        and entry["maximum_film_coupling_error_m"] <= 1e-18
        and entry["load_quadrature_relative_error"] <= 1e-10
    )
    return entry


def main(directory, include_fine=False):
    directory.mkdir(parents=True, exist_ok=False)
    fluid = IsothermalFluid(0.04, 850)
    report = {
        "status": "incomplete",
        "runtime": provenance(),
        "fluid": asdict(fluid),
        "cases": [],
        "parameters": {
            "h0_m": H0,
            "Rx_m": RX,
            "Ry_m": RY,
            "E_star_pa": 1e11,
            "Ue_m_s": 0.1,
            "temperature_k": 313.15,
            "L_m": L,
            "Ly_m": LY,
            "pressure_scale_pa": P_REF * 100,
        },
        "criteria": {
            "relative_matched_pressure_force": 1e-8,
            "mass_maximum": 1e-8,
            "relative_local_global_complementarity": 1e-10,
            "direct_displacement_m": 1e-18,
        },
        "scope": "Initial uniform finite-reservoir offset cases; classical-table validation open",
    }

    def checkpoint():
        save_json(directory / "summary.json", report)

    checkpoint()
    try:
        problem = parameters(np.linspace(-4, 6, 13) * L, np.linspace(-4, 4, 9) * LY)
        old = solve_cavitating_newton(problem).solution
        constant = solve_property_coupling(problem, fluid, 0.1, constitutive_strength=0)
        report["constant_limit"] = {
            "relative_pressure_error": float(
                np.linalg.norm(constant.pressure_pa - old.pressure_pa)
                / np.linalg.norm(old.pressure_pa)
            ),
            "relative_load_error": abs(constant.load_n / old.load_n - 1),
            "history": constant.history,
        }
        np.savez_compressed(
            directory / "constant_limit.npz",
            pressure_pa=constant.pressure_pa,
            reference_pressure_pa=old.pressure_pa,
            theta=constant.fractional_film_content,
            film_m=constant.film_thickness_m,
        )
        r = solve_property_coupling(problem, fluid, 0.1)
        reference, (p, t, h, w) = small_monolithic_reference(problem, fluid, 0.1)
        audit = mass_audit(
            problem.x_m,
            problem.y_m,
            h,
            p,
            t,
            fluid,
            0.1,
            length_scale_m=L,
            film_scale_m=H0,
            pressure_scale_pa=P_REF * 100,
        )
        entry = archive(directory, "matched_small", problem, r, fluid)
        entry["reference_status"] = reference
        entry["reference_audit"] = audit
        entry["relative_pressure_error"] = float(
            np.linalg.norm(r.pressure_pa - p) / np.linalg.norm(p)
        )
        entry["relative_load_error"] = abs(r.load_n / problem.load_n(p) - 1)
        entry["matched_reference_passed"] = bool(
            admissible(audit, p, t, P_REF * 100)
            and max(entry["relative_pressure_error"], entry["relative_load_error"]) <= 1e-8
        )
        report["matched"] = entry
        np.savez_compressed(
            directory / "monolithic_small.npz",
            x_m=problem.x_m,
            y_m=problem.y_m,
            pressure_pa=p,
            theta=t,
            film_m=h,
            displacement_m=w,
        )
        # Explicit bounded failure, preserving the actual candidate and both histories.
        try:
            solve_property_coupling(problem, fluid, 0.1, maximum_iterations_per_stage=1)
        except PropertyCouplingError as exc:
            report["bounded_failure"] = {"reason": str(exc), "history": exc.history}
            if exc.last_solution is not None:
                f = exc.last_solution
                np.savez_compressed(
                    directory / "failed_budget_one.npz",
                    pressure_pa=f.pressure_pa,
                    theta=f.fractional_film_content,
                    film_m=f.film_thickness_m,
                    displacement_m=f.elastic_displacement_m,
                )
        checkpoint()
        for n in (17, 33, 65) if include_fine else (17, 33):
            u = np.linspace(-8, 8, n)
            problem = parameters(u * L, u * LY)
            r = solve_property_coupling(problem, fluid, 0.1)
            entry = archive(directory, f"grid_{n}", problem, r, fluid)
            entry["constant_film_offset_m"] = H0
            report["cases"].append(entry)
            checkpoint()
            print(n, entry["load_n"], entry["maximum_pressure_pa"], len(r.history), flush=True)
        a, b = report["cases"][-2:]
        report["last_load_mesh_change"] = abs(b["load_n"] / a["load_n"] - 1)
        report["last_peak_mesh_change"] = abs(
            b["maximum_pressure_pa"] / a["maximum_pressure_pa"] - 1
        )
        report["mesh_target_met"] = bool(
            max(report["last_load_mesh_change"], report["last_peak_mesh_change"]) <= 0.01
        )
        render(directory, report)
        report["figures"] = [p.name for p in sorted(directory.glob("*.png"))]
        verified = (
            report["matched"]["matched_reference_passed"]
            and max(
                report["constant_limit"]["relative_pressure_error"],
                report["constant_limit"]["relative_load_error"],
            )
            <= 1e-8
            and all(c["accepted"] for c in report["cases"])
        )
        report["status"] = (
            "initial coupled verification complete; physical/grid validation open"
            if verified
            else "failed acceptance"
        )
        checkpoint()
    except Exception as exc:
        report["status"] = "failed"
        report["failure"] = repr(exc)
        if isinstance(exc, PropertyCouplingError):
            report["failed_history"] = exc.history
            if exc.last_solution is not None:
                f = exc.last_solution
                np.savez_compressed(
                    directory / "failed_candidate.npz",
                    pressure_pa=f.pressure_pa,
                    theta=f.fractional_film_content,
                    film_m=f.film_thickness_m,
                    displacement_m=f.elastic_displacement_m,
                )
            if getattr(exc.inner_failure, "last_solution", None) is not None:
                f = exc.inner_failure.last_solution
                report["inner_failed_history"] = exc.inner_failure.history
                np.savez_compressed(
                    directory / "inner_failed_candidate.npz",
                    pressure_pa=f.pressure_pa,
                    theta=f.fractional_film_content,
                )
        checkpoint()
        raise
    return report


def render(directory, report):
    context = (
        "Pressure-dependent density AND viscosity; h0=.1 um,E*=100 GPa,Ue=.1 m/s\n"
        "Rx=10 mm,Ry=20 mm,eta0=.04 Pa s,rho0=850 kg/m^3,z=.68,T=313.15 K\n"
        "Finite ambient reservoir; flooded west; full-source uniform FFT; prescribed offset"
    )
    fig, axes = figure("Coupled pressure and film under independent mesh changes", context, rows=2)
    for c in report["cases"]:
        f = np.load(directory / c["state_file"])
        j = len(f["y_m"]) // 2
        axes[0].plot(f["x_m"] * 1e3, f["pressure_pa"][j] / 1e6, label=str(c["shape"]))
        axes[1].plot(f["x_m"] * 1e3, f["film_m"][j] * 1e6, label=str(c["shape"]))
    axes[0].set_ylabel("Gauge pressure [MPa]")
    axes[0].legend()
    axes[1].set(xlabel="x [mm], entrainment +x", ylabel="Film [um]")
    axes[1].legend()
    save(fig, directory, "coupled_mesh_profiles")
    f = np.load(directory / report["cases"][-1]["state_file"])
    for key, scale, units in (
        ("pressure_pa", 1e-6, "MPa"),
        ("film_m", 1e6, "um"),
        ("displacement_m", 1e6, "um"),
        ("theta", 1, ""),
        ("density_kg_m3", 1, "kg/m^3"),
        ("viscosity_pa_s", 1, "Pa s"),
    ):
        data = f[key] * scale
        lo, hi = float(data.min()), float(data.max())
        if lo == hi:
            lo -= max(abs(lo), 1) * 1e-12
            hi += max(abs(hi), 1) * 1e-12
        fig, axes = figure(
            f"{key} [{units}]",
            context + f"\nGrid {data.shape}; raw range [{data.min():.8g},{data.max():.8g}]",
        )
        m = axes[0].contourf(f["x_m"] * 1e3, f["y_m"] * 1e3, data, levels=np.linspace(lo, hi, 25))
        fig.colorbar(m, ax=axes[0], label=units)
        axes[0].set(xlabel="x [mm], entrainment +x", ylabel="y [mm]")
        save(fig, directory, key)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=Path("results/property_coupling"))
    parser.add_argument(
        "--include-fine", action="store_true", help="Add bounded 65x65 coupled case"
    )
    args = parser.parse_args()
    main(args.output_directory, args.include_fine)
