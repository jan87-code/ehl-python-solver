"""Isolate finite inlet, outlet and transverse sensitivity at constant dx,dy.

All cases use matched SI geometry/material, reference offset and target force.
NumPy float64 on CPU; entrainment is +x and field array order is (ny,nx).
Each case starts independently, without interpolating another domain's root.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from examples.run_cavitating_domain import make_boundary_problem, run_problem


def boundary_cases():
    """Baseline plus two separate extensions of each boundary family."""
    yield "baseline", "baseline", 0, make_boundary_problem()
    for boundary in ("inlet", "outlet", "transverse"):
        for nodes in (4, 8):
            yield (
                f"{boundary}_{nodes}",
                boundary,
                nodes,
                make_boundary_problem(**{f"{boundary}_nodes": nodes}),
            )


def save_comparison(directory, cases):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    solved = [case for case in cases if case["target_reached"]]
    labels = [case["name"] for case in solved]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    axes[0].bar(labels, [case["final_h0_m"] * 1e6 for case in solved])
    axes[0].set_ylabel("Rigid-gap offset h0 [um] at Q=0.002 N")
    axes[1].plot(labels, [case["rigid_reference_force_n"] for case in solved], "o-", label="rigid")
    axes[1].plot(
        labels, [case["initial_reference_force_n"] for case in solved], "o-", label="elastic"
    )
    axes[1].set_ylabel("Force [N] at h0=20 um")
    axes[1].legend()
    for side in ("west", "east", "south", "north"):
        axes[2].plot(
            labels,
            [case["final_audit"][f"{side}_boundary_flux"] for case in solved],
            "o-",
            label=side,
        )
    axes[2].set_ylabel("Outward boundary flux [dimensionless]")
    axes[2].legend()
    for axis in axes:
        axis.tick_params(axis="x", rotation=55)
        axis.grid(alpha=0.3)
    fig.savefig(directory / "boundary_comparison.png", dpi=150)
    plt.close(fig)


def main(directory, *, plots=False):
    directory.mkdir(parents=True, exist_ok=False)
    report = {
        "method": "independent coupled references + adaptive positive-force continuation",
        "axis_order": "(ny,nx), entrainment +x",
        "inputs": {
            "target_load_n": 0.002,
            "reference_h0_m": 20e-6,
            "E_star_pa": 1e7,
            "Rx_m": 0.01,
            "Ry_m": 0.02,
            "L_m": 5e-4,
            "h_ref_m": 20e-6,
            "p_ref_pa": 21250,
            "Lambda": 12,
            "inlet_theta": 1,
        },
        "cases": [],
    }
    for name, boundary, nodes, problem in boundary_cases():
        entry = run_problem(problem, directory, name=name, plots=plots)
        entry.update({"extended_boundary": boundary, "added_nodes_per_edge": nodes})
        report["cases"].append(entry)
        if entry["target_reached"]:
            baseline = report["cases"][0]
            if baseline["target_reached"]:
                entry["h0_change_relative_to_baseline"] = (
                    entry["final_h0_m"] / baseline["final_h0_m"] - 1
                )
                with np.load(directory / "baseline_final.npz") as base:
                    # Compare pressure on the identical physical nodes only.
                    rows = slice(nodes, nodes + 13) if boundary == "transverse" else slice(0, 13)
                    cols = slice(nodes, nodes + 17) if boundary == "inlet" else slice(0, 17)
                    with np.load(directory / f"{name}_final.npz") as final:
                        entry["overlap_pressure_relative_l2_change"] = float(
                            np.linalg.norm(final["pressure_pa"][rows, cols] - base["pressure_pa"])
                            / np.linalg.norm(base["pressure_pa"])
                        )
        (directory / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            f"{name} {entry['shape']}: {entry['termination']}; "
            f"h0={entry.get('final_h0_m')}; trials={len(entry.get('trials', []))}",
            flush=True,
        )
    if plots:
        save_comparison(directory, report["cases"])
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory", type=Path, default=Path("results/cavitating_boundaries")
    )
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    report = main(args.output_directory, plots=args.plots)
    raise SystemExit(0 if all(case["target_reached"] for case in report["cases"]) else 1)
