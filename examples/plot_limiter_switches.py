"""Show exact-plateau derivatives and loaded merit probes, not EHL convergence."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ehl_solver.constitutive import IsothermalFluid
from ehl_solver.limited_transport import (
    limited_flux_direction,
    limited_flux_directional_derivative,
    limited_fluxes,
)
from ehl_solver.prototype_cli import digest


def main(report, output):
    loaded = json.loads(report.read_text())
    output.mkdir(parents=True, exist_ok=False)
    x, y = np.linspace(0, 1e-3, 65), np.linspace(0, 1e-4, 5)
    h, p, t = (np.full((5, 65), value) for value in (1e-6, 0.0, 0.75))
    fluid = IsothermalFluid(0.04, 850)
    dt = np.broadcast_to(
        0.01 * (np.sin(3 * np.pi * x / x[-1]) + 0.4 * np.cos(7 * np.pi * x / x[-1])), h.shape
    )
    zero = np.zeros_like(h)
    true = limited_flux_directional_derivative(x, y, h, p, t, zero, zero, dt, fluid, 2.0)[0]
    selected = limited_flux_direction(x, y, h, p, t, zero, zero, dt, fluid, 2.0)[0]
    step = 1e-3
    base = limited_fluxes(x, y, h, p, t, fluid, 2.0)[0]
    finite = (limited_fluxes(x, y, h, p, t + step * dt, fluid, 2.0)[0] - base) / step
    norm = 2 * 850e-6
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    face = 0.5 * (x[1:] + x[:-1]) * 1e3
    axes[0].plot(face, true[2] / norm, label="True one-sided derivative")
    axes[0].plot(face, finite[2] / norm, ".", markersize=3, label="One-sided finite difference")
    axes[0].plot(face, selected[2] / norm, "--", label="Original fixed branch map")
    axes[0].set(
        xlabel="x [mm], +x entrainment",
        ylabel="Flux derivative / (U rho h)",
        title="Exact capacity plateau, prescribed fields",
    )
    axes[0].legend(fontsize=8)
    for row in loaded["rows"]:
        valid = [v for v in row["trials"] if v["merit"] is not None]
        if not valid:
            continue
        alpha = np.array([v["alpha"] for v in valid])
        value = np.array([v["merit"] / loaded["merit"] for v in valid])
        (line,) = axes[1].plot(
            alpha, 100 * (value - 1), "o-", label=f"{row['name']}, sign {row['sign']:g}"
        )
        axes[1].plot(
            alpha, 100 * 1e-4 * alpha * row["slope"] / loaded["merit"], ":", color=line.get_color()
        )
    axes[1].axhline(0, color="gray", linewidth=0.7)
    axes[1].set(
        xscale="log",
        xlabel="Trial amplitude alpha",
        ylabel="Merit change / base merit [%], signed log",
        title="200 N residual probes; dotted Armijo bounds",
    )
    axes[1].set_yscale("symlog", linthresh=0.01)
    axes[1].legend(fontsize=8)
    fig.suptitle("Derivative/globalization diagnostics only; no coupled candidate solve")
    fig.savefig(output / "switches.png", dpi=180)
    plt.close(fig)
    np.savez_compressed(
        output / "plateau.npz",
        x_m=x,
        y_m=y,
        true_derivative=true,
        legacy_selected_derivative=selected,
        one_sided_finite_difference=finite,
    )
    (output / "summary.json").write_text(
        json.dumps(
            dict(
                source_report=str(report),
                source_sha256=digest(report),
                exact_plateau_relative_error=float(
                    np.linalg.norm(finite - true) / np.linalg.norm(true)
                ),
                candidate_is_converged=False,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    main(args.report, args.output)
