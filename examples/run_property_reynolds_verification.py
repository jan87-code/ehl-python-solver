"""Independent pressure-driven reference and variable-property cavitating point grids."""

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid

from ehl_solver.constitutive import IsothermalFluid
from ehl_solver.property_reynolds import (
    PropertyConvergenceError,
    mass_fluxes,
    solve_property_reynolds,
)
from examples.property_channel_reference import pressure_driven_channel
from examples.run_farfield_budget import H0, LY, P_REF, RX, RY, L, fronts
from examples.run_scientific_validation import provenance, save_json
from examples.scientific_figures import figure, save


def main(directory):
    directory.mkdir(parents=True, exist_ok=False)
    fluid = IsothermalFluid(0.04, 850)
    report = {
        "status": "incomplete",
        "runtime": provenance(),
        "fluid": asdict(fluid),
        "channel": [],
        "point": [],
        "criteria": {
            "channel_pressure_relative": 1e-6,
            "channel_mass_flux_relative": 1e-5,
            "successive_point_load_peak_relative": 0.01,
        },
        "scope": "Prescribed-film compressible/piezoviscous JFO; finite reservoir; no elasticity",
    }

    def checkpoint():
        save_json(directory / "summary.json", report)

    checkpoint()
    try:
        for n in (17, 33, 65, 129):
            x = np.linspace(0, 0.001, n)
            y = np.linspace(-0.001, 0.001, 5)
            expected, q = pressure_driven_channel(x, 20e-6, fluid, 5e6, 1e6)
            h = np.full((5, n), 20e-6)
            r = solve_property_reynolds(
                x,
                y,
                h,
                fluid,
                0,
                length_scale_m=0.001,
                film_scale_m=20e-6,
                pressure_scale_pa=5e6,
                boundary_pressure_pa=np.broadcast_to(expected, h.shape),
            )
            flux, _ = mass_fluxes(x, y, h, r.pressure_pa, r.fractional_film_content, fluid, 0)
            report["channel"].append(
                {
                    "nx": n,
                    "pressure_relative_error": float(
                        np.linalg.norm(r.pressure_pa[2] - expected) / np.linalg.norm(expected)
                    ),
                    "center_mean_flux_relative_error": float(abs(flux[2].mean() / q - 1)),
                    "exact_mass_flux_kg_m_s": q,
                    "audit": r.audit,
                    "history": r.history,
                }
            )
            np.savez_compressed(
                directory / f"channel_{n}.npz",
                x_m=x,
                y_m=y,
                film_m=h,
                pressure_pa=r.pressure_pa,
                theta=r.fractional_film_content,
                reference_pressure_pa=expected,
            )
            checkpoint()
        for n in (17, 33, 65):
            u = np.linspace(-8, 8, n)
            x, y = u * L, u * LY
            h = H0 + x[None, :] ** 2 / (2 * RX) + y[:, None] ** 2 / (2 * RY)
            r = solve_property_reynolds(
                x,
                y,
                h,
                fluid,
                0.1,
                length_scale_m=L,
                film_scale_m=H0,
                pressure_scale_pa=P_REF * 100,
            )
            rho, eta, _, _ = fluid.properties(r.pressure_pa)
            entry = {
                "shape": list(h.shape),
                "step": 16 / (n - 1),
                "load_n": float(trapezoid(trapezoid(r.pressure_pa, x, axis=1), y)),
                "maximum_pressure_pa": float(r.pressure_pa.max()),
                "pressure_min_pa": float(r.pressure_pa.min()),
                "maximum_viscosity_ratio": float(eta.max() / fluid.viscosity_pa_s),
                "maximum_density_ratio": float(rho.max() / fluid.density_kg_m3),
                "audit": r.audit,
                "history": r.history,
                "fronts": fronts(x, r.fractional_film_content),
            }
            report["point"].append(entry)
            np.savez_compressed(
                directory / f"point_{n}.npz",
                x_m=x,
                y_m=y,
                film_m=h,
                pressure_pa=r.pressure_pa,
                theta=r.fractional_film_content,
                density_kg_m3=rho,
                viscosity_pa_s=eta,
            )
            checkpoint()
            print(n, entry["maximum_pressure_pa"], entry["audit"]["mass_maximum"], flush=True)
        render(directory, report)
        report["figures"] = [p.name for p in sorted(directory.glob("*.png"))]
        last = report["channel"][-1]
        report["channel_verified"] = bool(
            last["pressure_relative_error"] <= 1e-6
            and last["center_mean_flux_relative_error"] <= 1e-5
        )
        a, b = report["point"][-2:]
        report["last_point_load_change"] = abs(b["load_n"] / a["load_n"] - 1)
        report["last_point_peak_change"] = abs(
            b["maximum_pressure_pa"] / a["maximum_pressure_pa"] - 1
        )
        report["point_mesh_target_met"] = bool(
            max(report["last_point_load_change"], report["last_point_peak_change"]) <= 0.01
        )
        report["status"] = (
            "completed property-flow roots; independent channel verified; point mesh accuracy open"
        )
        checkpoint()
    except Exception as exc:
        report["status"] = "failed"
        report["failure"] = repr(exc)
        if isinstance(exc, PropertyConvergenceError):
            report["failed_history"] = exc.history
            if exc.last_solution is not None:
                np.savez_compressed(
                    directory / "failed_candidate.npz",
                    pressure_pa=exc.last_solution.pressure_pa,
                    theta=exc.last_solution.fractional_film_content,
                )
        checkpoint()
        raise
    return report


def render(directory, report):
    context = (
        "SI Roelands/Dowson-Higginson; eta0=.04 Pa s,rho0=850 kg/m^3,z=.68\n"
        "Prescribed rigid film; finite boundaries; no experimental validation"
    )
    fig, axes = figure("Independent pressure-driven Kirchhoff-reference convergence", context)
    c = report["channel"]
    axes[0].loglog(
        [a["nx"] for a in c], [a["pressure_relative_error"] for a in c], "o-", label="Pressure L2"
    )
    axes[0].loglog(
        [a["nx"] for a in c],
        [a["center_mean_flux_relative_error"] for a in c],
        "o-",
        label="Mass flux",
    )
    axes[0].set(xlabel="Streamwise nodes, fixed physical domain", ylabel="Relative error")
    axes[0].legend()
    save(fig, directory, "channel_reference")
    fig, axes = figure(
        "Variable-property cavitating point profiles",
        context + "\nh0=.1 um,Rx=10 mm,Ry=20 mm,Ue=.1 m/s; ambient pressure edges, flooded west",
        rows=2,
    )
    for n in (17, 33, 65):
        f = np.load(directory / f"point_{n}.npz")
        j = n // 2
        axes[0].plot(f["x_m"] * 1e3, f["pressure_pa"][j] / 1e6, label=f"{n}x{n}")
        axes[1].plot(f["x_m"] * 1e3, f["theta"][j], label=f"{n}x{n}")
    axes[0].set_ylabel("Gauge pressure [MPa]")
    axes[0].legend()
    axes[1].set(xlabel="x [mm], entrainment +x", ylabel="Fractional content theta")
    axes[1].legend()
    save(fig, directory, "point_profiles")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=Path("results/property_reynolds"))
    main(parser.parse_args().output_directory)
