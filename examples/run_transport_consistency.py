"""Independent smooth SI transport checks; no coupled root or stability claim.

Arrays are (ny,nx), +x entrainment. Manufactured fields test flux divergence,
not JFO complementarity. Outputs contain no adjusted production solution.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from ehl_solver.constitutive import IsothermalFluid
from ehl_solver.property_reynolds import mass_flux_direction, mass_fluxes
from examples.run_scientific_validation import provenance


def smooth_case(nx, *, variable_content=False):
    """Compare conservative divergence with an analytic continuum mass source.

    L=1 mm, h_ref=2 um, p_ref=10 MPa, Ue=.1 m/s. The smooth fractional
    content case deliberately violates complementarity to isolate the upwind
    transport truncation error; it is not a cavitating solution.
    """
    length = 1e-3
    x = np.linspace(0, length, nx)
    y = np.linspace(0, length, (nx + 1) // 2)
    X, Y = np.meshgrid(x / length, y / length)
    wave = 2 * np.pi / length
    sx, cx, sy, cy = np.sin(2 * np.pi * X), np.cos(2 * np.pi * X), np.sin(
        2 * np.pi * Y
    ), np.cos(2 * np.pi * Y)
    h = 2e-6 * (1 + 0.1 * sx + 0.05 * cy)
    hx, hy = 2e-7 * wave * cx, -1e-7 * wave * sy
    p = 1e7 * (1 + 0.2 * cx * cy)
    px, py = -2e6 * wave * sx * cy, -2e6 * wave * cx * sy
    laplacian = -4e6 * wave**2 * cx * cy
    theta = 0.8 + 0.1 * sx if variable_content else np.ones_like(h)
    tx = 0.1 * wave * cx if variable_content else np.zeros_like(h)
    fluid, speed = IsothermalFluid(0.04, 850), 0.1
    rho, eta, drho, deta = fluid.properties(p)
    mobility = rho * h**3 / (12 * eta)
    mp = h**3 / 12 * (drho / eta - rho * deta / eta**2)
    mh = rho * h**2 / (4 * eta)
    exact = -(mp * (px**2 + py**2) + mh * (hx * px + hy * py) + mobility * laplacian)
    exact += speed * ((drho * px * h + rho * hx) * theta + rho * h * tx)
    qx, qy = mass_fluxes(x, y, h, p, theta, fluid, speed)
    discrete = np.diff(qx[1:-1], axis=1) / (x[1] - x[0])
    discrete += np.diff(qy[:, 1:-1], axis=0) / (y[1] - y[0])
    exact = exact[1:-1, 1:-1]
    error = np.linalg.norm(discrete - exact) / np.linalg.norm(exact)
    return {"shape_ny_nx": list(h.shape), "relative_l2_divergence_error": float(error)}


def alternating_pressure_probe():
    """Frozen-film linear density transport at a uniform full-film state.

    The centered capacity derivative annihilates alternating x pressure on
    interior cells. Diffusion remains nonzero. This diagnostic says nothing
    about stability of the complete elastic/material/complementarity Jacobian.
    """
    x, y = np.linspace(0, 1e-3, 17), np.linspace(0, 0.5e-3, 9)
    h = np.full((9, 17), 2e-6)
    p, theta = np.full_like(h, 1e7), np.ones_like(h)
    dp = np.broadcast_to(1e6 * (-1.0) ** np.arange(17), h.shape)
    zero = np.zeros_like(h)
    fluid = IsothermalFluid(0.04, 850)
    diffusion = mass_flux_direction(x, y, h, p, theta, zero, dp, zero, fluid, 0)
    total = mass_flux_direction(x, y, h, p, theta, zero, dp, zero, fluid, 0.1)

    def divergence(q):
        return np.diff(q[0][1:-1], axis=1) / (x[1] - x[0]) + np.diff(
            q[1][:, 1:-1], axis=0
        ) / (y[1] - y[0])

    diff = divergence(diffusion)
    adv = divergence(tuple(a - b for a, b in zip(total, diffusion, strict=True)))
    rho, eta, drho, _ = fluid.properties(p[0, 0])
    mobility = rho * h[0, 0] ** 3 / (12 * eta)
    expected = 4 * mobility * dp[1:-1, 1:-1] / (x[1] - x[0]) ** 2
    return {
        "density_advection_maximum_kg_m2_s": float(abs(adv).max()),
        "diffusion_maximum_kg_m2_s": float(abs(diff).max()),
        "relative_diffusion_symbol_error": float(np.linalg.norm(diff - expected)
                                                  / np.linalg.norm(expected)),
        "frozen_film_density_cell_peclet": float(
            0.1 * h[0, 0] * abs(drho) * (x[1] - x[0]) / (2 * mobility)
        ),
        "scope": "uniform frozen-film density tangent only; not coupled stability",
    }


def report():
    cases = {}
    for variable, name in ((False, "full_film"), (True, "smooth_variable_content")):
        rows = [smooth_case(n, variable_content=variable) for n in (17, 33, 65, 129)]
        for previous, current in zip(rows, rows[1:], strict=False):
            current["observed_order"] = float(np.log2(
                previous["relative_l2_divergence_error"] / current["relative_l2_divergence_error"]
            ))
        cases[name] = rows
    return {
        "status": "manufactured transport verification; no production root assessed",
        "divergence_units": "kg/(m^2 s)",
        "cases": cases,
        "alternating_pressure": alternating_pressure_probe(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    result = report()
    result["runtime"] = provenance()
    args.output_directory.mkdir(parents=True, exist_ok=False)
    (args.output_directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
