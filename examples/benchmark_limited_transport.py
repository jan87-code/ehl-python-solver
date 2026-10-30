"""Isolated SI capacity reconstruction benchmarks; no candidate EHL solve."""

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ehl_solver.constitutive import IsothermalFluid
from ehl_solver.limited_transport import limited_fluxes
from ehl_solver.prototype_cli import digest
from ehl_solver.transport_candidate import donor_fluxes, face_divergence
from examples.benchmark_transport_candidate import smooth_case


def smooth_errors(n, varying_content=True):
    """Independent manufactured source, monotone in x; includes boundary cells.

    Constant material limit, linear SI pressure, exponential film and content.
    The source is not a complementary EHL solution. All norms include the
    west-adjacent interior column, testing the bounded boundary closure too.
    """
    length = 1e-3
    x, y = np.linspace(0, length, n), np.linspace(0, length, n)
    xx, yy = np.meshgrid(x / length, y / length)
    h = 1e-6 * np.exp(0.4 * xx) * (1 + 0.1 * yy + 0.05 * yy**2)
    hx = 0.4 * h / length
    hy = 1e-6 * np.exp(0.4 * xx) * (0.1 + 0.1 * yy) / length
    p = 1e7 + 1e7 * xx + 0.7e7 * yy
    theta = 0.65 + 0.1 * np.exp(0.3 * xx) if varying_content else np.ones_like(h)
    tx = 0.03 * np.exp(0.3 * xx) / length if varying_content else np.zeros_like(h)
    fluid, speed = IsothermalFluid(0.04, 850), 2.0
    exact = -850 / (4 * 0.04) * h**2 * (hx * 1e10 + hy * 0.7e10) + speed * 850 * (
        hx * theta + h * tx
    )
    exact = exact[1:-1, 1:-1]
    errors = {}
    for name, flux in [("donor", donor_fluxes), ("limited", limited_fluxes)]:
        numerical = face_divergence(x, y, *flux(x, y, h, p, theta, fluid, speed, strength=0.0))
        errors[name] = float(np.linalg.norm(numerical - exact) / np.linalg.norm(exact))
    return errors


def extrema_errors(n, varying_content=True):
    """Trigonometric manufactured source with extrema and limiter switches."""
    x, y, h, p, theta, fluid, speed, exact = smooth_case(n, varying_content=varying_content)
    return {
        name: float(
            np.linalg.norm(
                face_divergence(x, y, *flux(x, y, h, p, theta, fluid, speed, strength=0.0)) - exact
            )
            / np.linalg.norm(exact)
        )
        for name, flux in [("donor", donor_fluxes), ("limited", limited_fluxes)]
    }


def main(output):
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for category, error in [("monotone", smooth_errors), ("extrema", extrema_errors)]:
        for varying in (False, True):
            data = [dict(n=n, **error(n, varying)) for n in (33, 65, 129, 257)]
            for previous, current in zip(data[:-1], data[1:], strict=True):
                current["orders"] = {
                    s: float(np.log2(previous[s] / current[s])) for s in ("donor", "limited")
                }
            rows.append(dict(category=category, varying_content=varying, data=data))
    x, y = np.linspace(0, 1e-3, 129), np.linspace(0, 1e-4, 5)
    p = np.zeros((5, 129))
    h = np.full_like(p, 1e-6)
    theta = np.broadcast_to(np.where(x < 0.5e-3, 1.0, 0.2), p.shape).copy()
    fluid = IsothermalFluid(0.04, 850)
    q = limited_fluxes(x, y, h, p, theta, fluid, 2.0)[0] / (2 * 850e-6)
    report = dict(
        scope="isolated flux only; no coupled EHL validation",
        smooth=rows,
        sharp_front_minimum=float(q.min()),
        sharp_front_maximum=float(q.max()),
        source_sha256={},
    )
    for path in [
        Path(__file__),
        Path("src/ehl_solver/limited_transport.py"),
        Path("src/ehl_solver/transport_candidate.py"),
    ]:
        shutil.copyfile(path, output / path.name)
        report["source_sha256"][path.name] = digest(path)
    (output / "summary.json").write_text(json.dumps(report, indent=2))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for row in rows:
        ax = axes[0 if row["category"] == "monotone" else 1]
        for scheme in ("donor", "limited"):
            ax.loglog(
                [1 / (r["n"] - 1) for r in row["data"]],
                [r[scheme] for r in row["data"]],
                "o-",
                label=f"{scheme}, theta varies={row['varying_content']}",
            )
    for ax, title in zip(axes[:2], ("Monotone source", "Source with extrema"), strict=True):
        ax.set(xlabel="dx / domain length", ylabel="Relative divergence L2 error", title=title)
        ax.legend(fontsize=7)
    axes[2].step(x * 1e3, theta[2], where="mid", label="Nodal theta")
    axes[2].plot(
        0.5 * (x[1:] + x[:-1]) * 1e3, q[2], ".", label="Reconstructed face capacity / rho h"
    )
    axes[2].set(
        xlabel="x [mm], +x entrainment",
        ylabel="Content / normalized capacity",
        title="Prescribed sharp-front reconstruction",
    )
    axes[2].legend(fontsize=7)
    fig.savefig(output / "benchmark.png", dpi=180)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    main(parser.parse_args().output)
