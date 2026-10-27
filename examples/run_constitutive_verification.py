"""SI constitutive laws, analytical derivatives and reproducible illustrative curves."""

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ehl_solver.constitutive import IsothermalFluid
from examples.run_scientific_validation import provenance, save_json
from examples.scientific_figures import figure, save


def main(directory):
    directory.mkdir(parents=True, exist_ok=False)
    fluid = IsothermalFluid(0.04, 850)
    p = np.linspace(0, 1e9, 1001)
    rho, eta, drho, deta = fluid.properties(p)
    np.savez_compressed(
        directory / "properties_si.npz",
        pressure_pa=p,
        density_kg_m3=rho,
        viscosity_pa_s=eta,
        density_derivative_kg_m3_pa=drho,
        viscosity_derivative_pa_s_per_pa=deta,
    )
    report = {
        "runtime": provenance(),
        "fluid": asdict(fluid),
        "assigned_temperature_k": 313.15,
        "scope": "Illustrative fixed-temperature laws; no measured-fluid or coupled EHL validation",
        "sources": {
            "formulation": "https://ntrs.nasa.gov/api/citations/19910021217/downloads/19910021217.pdf",
            "equations": ["4-10", "4-19"],
            "printed_pages": [57, 70],
        },
        "density_at_1gpa_kg_m3": float(rho[-1]),
        "viscosity_at_1gpa_pa_s": float(eta[-1]),
        "origin_dlog_eta_per_pa": float(fluid.ratios(0)[3]),
        "status": "standalone laws; pressure-flow integration is subsequent",
    }
    context = (
        "Illustrative fluid at T=313.15 K; eta0=.04 Pa s,rho0=850 kg/m^3,z=.68\n"
        "NASA RP-1255 SI model; no specific-oil experimental validation"
    )
    fig, axes = figure("Pressure-dependent viscosity and density", context, rows=2)
    axes[0].semilogy(p / 1e9, eta)
    axes[0].set_ylabel("Viscosity [Pa s]")
    axes[1].plot(p / 1e9, rho)
    axes[1].set(xlabel="Gauge pressure [GPa]", ylabel="Liquid density [kg/m^3]")
    save(fig, directory, "constitutive_laws")
    report["figures"] = ["constitutive_laws.png"]
    save_json(directory / "summary.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=Path("results/constitutive"))
    main(parser.parse_args().output_directory)
