"""Fixed finite point-reservoir mesh budget for a graded exterior response.

Uniform independent JFO references and graded response solves share physical
reservoir bounds. Core resolution and outer grading are separate controls.
Exterior deformation is diagnostic direct cell integration, never a nonuniform FFT.
"""

import argparse
import time
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid

from ehl_solver.backend import make_backend
from ehl_solver.cavitation import solve_cavitating_reynolds
from ehl_solver.elasticity import elastic_deformation
from ehl_solver.exterior import ExteriorSolution, exterior_source_budget, solve_exterior_response
from ehl_solver.rectilinear_cavitation import flux_audit
from examples.run_exterior_response import fields, interface_budget
from examples.run_farfield_budget import (
    ETA,
    H0,
    LY,
    P_REF,
    RX,
    RY,
    SPEED,
    L,
    accepted,
    fronts,
    graded_axis,
)
from examples.run_scientific_validation import provenance, save_json
from examples.scientific_figures import figure, save


def section_flux(x, y, h, p, theta):
    """Diagnostic at the SAME physical x=-L, y in [-Ly,Ly], m^3/s.

    Central pressure derivative and nodal h/theta, integrated by trapezoids.
    Not a replacement for the solver's independent conservative face audit.
    Every studied mesh includes these nodes and a locally uniform stencil.
    """
    i = int(np.argmin(abs(x + L)))
    jy = np.flatnonzero(abs(y) <= LY + 1e-18)
    if not np.isclose(x[i], -L, rtol=1e-12, atol=0) or not np.isclose(
        x[i + 1] - x[i], x[i] - x[i - 1]
    ):
        raise ValueError("section requires x=-L and a local uniform derivative stencil")
    gradient = (p[jy, i + 1] - p[jy, i - 1]) / (x[i + 1] - x[i - 1])
    c = SPEED * h[jy, i] * theta[jy, i]
    diff = -(h[jy, i] ** 3) * gradient / (12 * ETA)
    return {
        "couette_m3_s": float(trapezoid(c, y[jy])),
        "poiseuille_m3_s": float(trapezoid(diff, y[jy])),
        "net_m3_s": float(trapezoid(c + diff, y[jy])),
        "x_m": float(x[i]),
        "y_limits_m": [float(y[jy[0]]), float(y[jy[-1]])],
    }


def reference(step):
    units = np.linspace(-8, 8, round(16 / step) + 1)
    x, y = units * L, units * LY
    h = H0 + x[None, :] ** 2 / (2 * RX) + y[:, None] ** 2 / (2 * RY)
    t = time.perf_counter()
    result = solve_cavitating_reynolds(h / H0, step, step * LY / L, 1)
    p, theta = result.pressure * P_REF, result.fractional_film_content
    f = fields(x, y, h, p, theta)
    audit = flux_audit(
        x, y, h, p, theta, ETA, SPEED, length_scale_m=L, film_scale_m=H0, pressure_scale_pa=P_REF
    )
    deformation = elastic_deformation(p, step * L, step * LY, 1e9, make_backend("numpy"))
    j, i = len(y) // 2, len(x) // 2
    peak = np.unravel_index(p.argmax(), p.shape)
    entry = {
        "step": step,
        "shape": list(h.shape),
        "load_n": float(trapezoid(trapezoid(p, x, axis=1), y)),
        "pressure_max_pa": float(p.max()),
        "pressure_peak_xy_m": [float(x[peak[1]]), float(y[peak[0]])],
        "center_displacement_m": float(deformation[j, i]),
        "upstream_displacement_m": float(deformation[j, i - round(0.5 / step)]),
        "audit": audit,
        "fronts": fronts(x, theta),
        "section_flux": section_flux(x, y, h, p, theta),
        "elapsed_s": time.perf_counter() - t,
        "accepted": accepted(audit, f),
    }
    mask = (abs(x[None, :]) <= L) & (abs(y[:, None]) <= LY)
    entry["source_budget"] = exterior_source_budget(
        x, y, ExteriorSolution(p, theta, mask, audit, ()), 1e9, [[0, 0], [-0.5 * L, 0]]
    )
    return entry, f


def compare_core(f, ref):
    ix = np.flatnonzero(abs(f["x_m"]) <= 2 * L + 1e-18)
    iy = np.flatnonzero(abs(f["y_m"]) <= 2 * LY + 1e-18)
    rx = np.argmin(abs(ref["x_m"][:, None] - f["x_m"][ix]), axis=0)
    ry = np.argmin(abs(ref["y_m"][:, None] - f["y_m"][iy]), axis=0)
    np.testing.assert_allclose(ref["x_m"][rx], f["x_m"][ix], atol=1e-18, rtol=1e-13)
    np.testing.assert_allclose(ref["y_m"][ry], f["y_m"][iy], atol=1e-18, rtol=1e-13)
    target = ref["pressure_pa"][np.ix_(ry, rx)]
    return float(np.linalg.norm(f["pressure_pa"][np.ix_(iy, ix)] - target) / np.linalg.norm(target))


def compare_fronts(f, ref):
    """Bracket separation on identical physical core y rows; no interpolation."""
    rows = np.flatnonzero(abs(f["y_m"]) <= 2 * LY + 1e-18)
    ix = np.flatnonzero(abs(f["x_m"]) <= 2 * L + 1e-18)
    rx = np.flatnonzero(abs(ref["x_m"]) <= 2 * L + 1e-18)
    spacing = f["x_m"][ix[1]] - f["x_m"][ix[0]]
    maximum, mismatches = 0.0, 0

    def brackets(x, content):
        dry = content < 1 - 1e-8
        return [(bool(dry[k]), x[k], x[k + 1]) for k in np.flatnonzero(np.diff(dry))]

    for j in rows:
        rj = int(np.argmin(abs(ref["y_m"] - f["y_m"][j])))
        if not np.isclose(ref["y_m"][rj], f["y_m"][j], rtol=1e-13, atol=1e-18):
            raise ValueError("front comparison requires shared physical y rows")
        a = brackets(f["x_m"][ix], f["theta"][j, ix])
        b = brackets(ref["x_m"][rx], ref["theta"][rj, rx])
        if len(a) != len(b) or any(aa[0] != bb[0] for aa, bb in zip(a, b, strict=True)):
            mismatches += 1
            continue
        for aa, bb in zip(a, b, strict=True):
            maximum = max(maximum, max(0.0, max(aa[1], bb[1]) - min(aa[2], bb[2])))
    return {
        "shared_y_rows": len(rows),
        "front_count_or_kind_mismatches": mismatches,
        "maximum_bracket_separation_m": float(maximum),
        "maximum_separation_core_cells": float(maximum / spacing),
        "within_two_core_cells": bool(mismatches == 0 and maximum <= 2 * spacing),
    }


def render(directory, report):
    import matplotlib.pyplot as plt

    context = (
        "Fixed finite physical reservoir +/-8L,+/-8Ly; ambient pressure, west theta=1\n"
        "Rigid h0=.1 um, Rx=10 mm,Ry=20 mm, eta=.04 Pa s,Ue=.001 m/s\n"
        "Uniform [-2,2] core; geometric outer cells; direct diagnostic elasticity E*=1 GPa"
    )
    cases = report["cases"]
    refs = report["references"]
    fig, axes = figure("Separate core and outer-grid error budgets", context, rows=2)
    for g in (1.2, 1.1, 1.05):
        chosen = [c for c in cases if c["growth"] == g]
        for ax, key in zip(
            axes, ("relative_load_error", "relative_core_pressure_error"), strict=True
        ):
            ax.plot(
                [c["step"] for c in chosen],
                [c[key] * 100 for c in chosen],
                "o-",
                label=f"outer growth {g}",
            )
            ax.axhline(1, color="black", linestyle=":")
            ax.set_xscale("log")
            ax.legend(fontsize=8)
    axes[0].set_ylabel("Load error vs finest uniform [%]")
    axes[1].set(
        xlabel="Core dX (independent resolution control)",
        ylabel="Shared core pressure L2 error [%]",
    )
    save(fig, directory, "mesh_error")
    fig, axes = figure("Uniform reference refinement and graded load", context, rows=2)
    axes[0].plot(
        [r["step"] for r in refs],
        [r["load_n"] for r in refs],
        "ko-",
        label="Independent uniform reference",
    )
    for g in (1.2, 1.1, 1.05):
        chosen = [c for c in cases if c["growth"] == g]
        axes[0].plot(
            [c["step"] for c in chosen],
            [c["source_budget"]["total_load_n"] for c in chosen],
            "o-",
            label=f"growth {g}",
        )
    axes[0].set(xlabel="Core dX", ylabel="Full finite-reservoir load [N]")
    axes[0].legend(fontsize=8)
    axes[1].plot(
        [c["shape"][0] * c["shape"][1] for c in cases],
        [c["relative_load_error"] * 100 for c in cases],
        "o",
    )
    axes[1].set(xlabel="Total graded grid nodes", ylabel="Load error vs finest uniform [%]")
    save(fig, directory, "load_cost")
    fig, axes = figure("Exterior approximation: supply and elastic-source errors", context, rows=2)
    for g in (1.2, 1.1, 1.05):
        chosen = [c for c in cases if c["growth"] == g]
        axes[0].plot(
            [c["step"] for c in chosen],
            [c["relative_section_flux_error"] * 100 for c in chosen],
            "o-",
            label=f"growth {g}",
        )
        axes[1].plot(
            [c["step"] for c in chosen],
            [c["relative_center_displacement_error"] * 100 for c in chosen],
            "o-",
            label=f"growth {g}",
        )
    axes[0].set_ylabel("Fixed diagnostic section flow error [%]")
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel="Core dX", ylabel="Center displacement error [%]")
    axes[1].legend(fontsize=8)
    save(fig, directory, "supply_sources")
    best = [c for c in cases if c["step"] == 0.0625 and c["growth"] == 1.05][0]
    f = np.load(directory / best["state_file"])
    r = np.load(directory / refs[-1]["state_file"])
    fig, axes = figure(
        "Finest graded approximation versus independent uniform reference", context, rows=2
    )
    for a, label in ((r, "Uniform .0625"), (f, "Graded .0625, growth 1.05")):
        j = len(a["y_m"]) // 2
        axes[0].plot(a["x_m"] * 1e3, a["pressure_pa"][j] / 1e6, label=label)
        axes[1].plot(a["x_m"] * 1e3, a["theta"][j], label=label)
    axes[0].set_ylabel("Pressure [MPa]")
    axes[0].legend()
    axes[1].set(xlabel="x [mm], entrainment +x", ylabel="Fractional content theta")
    axes[1].legend()
    save(fig, directory, "matched_centerline")
    fig, axes = figure("Saturation fronts, without imposed interface pressure", context, rows=2)
    for ax, a, label in zip(axes, (r, f), ("Uniform reference", "Graded response"), strict=True):
        ax.contourf(a["x_m"] * 1e3, a["y_m"] * 1e3, a["theta"], levels=np.linspace(0, 1, 21))
        ax.set(xlabel="x [mm]", ylabel="y [mm]", title=label)
    save(fig, directory, "saturation")
    plt.close("all")


def main(directory):
    directory.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "incomplete",
        "runtime": provenance(),
        "references": [],
        "cases": [],
        "parameters": {
            "h0_m": H0,
            "Rx_m": RX,
            "Ry_m": RY,
            "eta_pa_s": ETA,
            "Ue_m_s": SPEED,
            "L_m": L,
            "Ly_m": LY,
            "pressure_scale_pa": P_REF,
            "physical_outer_label": 8,
            "uniform_core_label": 2,
            "condensation_cut_label": 1,
        },
        "criteria": {
            "physical_max_mass": 1e-8,
            "relative_local_global_complementarity": 1e-10,
            "relative_load_pressure_displacement_section_error": 0.01,
        },
        "scope": "Fixed finite-reservoir mesh/outer grading budget, rigid prescribed film only",
    }

    def checkpoint():
        save_json(directory / "summary.json", report)

    checkpoint()
    try:
        for step in (0.5, 0.25, 0.125, 0.0625):
            entry, f = reference(step)
            entry["state_file"] = f"uniform_{step}.npz"
            np.savez_compressed(directory / entry["state_file"], **f)
            report["references"].append(entry)
            checkpoint()
            print("reference", step, entry["load_n"], entry["audit"]["mass_maximum"], flush=True)
        target = report["references"][-1]
        ref = f
        for step in (0.25, 0.125, 0.0625):
            for growth in (1.2, 1.1, 1.05):
                units = graded_axis(8, 8, step, snap=False, growth=growth)
                x, y = units * L, units * LY
                h = H0 + x[None, :] ** 2 / (2 * RX) + y[:, None] ** 2 / (2 * RY)
                start = time.perf_counter()
                result = solve_exterior_response(
                    x,
                    y,
                    h,
                    ETA,
                    SPEED,
                    core_bounds_m=(-L, L, -LY, LY),
                    length_scale_m=L,
                    film_scale_m=H0,
                    pressure_scale_pa=P_REF,
                )
                f = fields(x, y, h, result.pressure_pa, result.fractional_film_content)
                budget = exterior_source_budget(x, y, result, 1e9, [[0, 0], [-0.5 * L, 0]])
                section = section_flux(x, y, h, result.pressure_pa, result.fractional_film_content)
                entry = {
                    "step": step,
                    "growth": growth,
                    "shape": list(h.shape),
                    "audit": result.audit,
                    "history": result.history,
                    "source_budget": budget,
                    "section_flux": section,
                    "fronts": fronts(x, result.fractional_film_content),
                    "front_comparison": compare_fronts(f, ref),
                    "pressure_max_pa": float(result.pressure_pa.max()),
                    "sampled_minimum_film_m": float(h.min()),
                    "relative_core_pressure_error": compare_core(f, ref),
                    "relative_load_error": abs(budget["total_load_n"] / target["load_n"] - 1),
                    "relative_peak_pressure_error": abs(
                        result.pressure_pa.max() / target["pressure_max_pa"] - 1
                    ),
                    "relative_center_displacement_error": abs(
                        budget["total_displacement_m"][0] / target["center_displacement_m"] - 1
                    ),
                    "relative_upstream_displacement_error": abs(
                        budget["total_displacement_m"][1] / target["upstream_displacement_m"] - 1
                    ),
                    "relative_section_flux_error": abs(
                        section["net_m3_s"] / target["section_flux"]["net_m3_s"] - 1
                    ),
                    "interface_budget": interface_budget(
                        x,
                        y,
                        h,
                        result.pressure_pa,
                        result.fractional_film_content,
                        result.retained_mask,
                    ),
                    "elapsed_s": time.perf_counter() - start,
                    "state_file": f"graded_{step}_{growth}.npz",
                }
                entry["physical_audit_passed"] = accepted(result.audit, f)
                entry["outer_control_faces_m"] = [
                    float((x[0] + x[1]) / 2),
                    float((x[-2] + x[-1]) / 2),
                    float((y[0] + y[1]) / 2),
                    float((y[-2] + y[-1]) / 2),
                ]
                if step == target["step"]:
                    entry["relative_exterior_source_load_error"] = abs(
                        budget["exterior_source_load_n"]
                        / target["source_budget"]["exterior_source_load_n"]
                        - 1
                    )
                    entry["relative_exterior_center_displacement_error"] = abs(
                        budget["exterior_displacement_m"][0]
                        / target["source_budget"]["exterior_displacement_m"][0]
                        - 1
                    )
                else:
                    # Different core cell widths give different source partitions.
                    entry["relative_exterior_source_load_error"] = None
                    entry["relative_exterior_center_displacement_error"] = None
                entry["reference_error_passed"] = (
                    all(
                        entry[key] <= 0.01
                        for key in (
                            "relative_load_error",
                            "relative_core_pressure_error",
                            "relative_peak_pressure_error",
                            "relative_center_displacement_error",
                            "relative_upstream_displacement_error",
                            "relative_section_flux_error",
                        )
                    )
                    and entry["front_comparison"]["within_two_core_cells"]
                )
                entry["reference_error_passed"] = bool(
                    entry["reference_error_passed"]
                    and all(
                        entry[k] is None or entry[k] <= 0.01
                        for k in (
                            "relative_exterior_source_load_error",
                            "relative_exterior_center_displacement_error",
                        )
                    )
                )
                np.savez_compressed(
                    directory / entry["state_file"], **f, retained_mask=result.retained_mask
                )
                report["cases"].append(entry)
                checkpoint()
                print(
                    "graded",
                    step,
                    growth,
                    entry["relative_load_error"],
                    entry["reference_error_passed"],
                    flush=True,
                )
        render(directory, report)
        report["figures"] = [p.name for p in sorted(directory.glob("*.png"))]
        report["status"] = (
            "mesh budget completed; coupled and infinite-domain validation remain open"
        )
        checkpoint()
    except Exception as exc:
        report["status"] = "failed"
        report["failure"] = repr(exc)
        checkpoint()
        raise
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=Path("results/exterior_mesh"))
    main(parser.parse_args().output_directory)
