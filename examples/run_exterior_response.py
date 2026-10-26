"""Finite point-reservoir exterior response verification; NumPy/SciPy CPU only.

One fixed physical reservoir, independent uniform JFO reference, varied inner
cuts. No exterior reference pressure is supplied to the response solve.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid

from ehl_solver.cavitation import solve_cavitating_reynolds
from ehl_solver.exterior import exterior_source_budget, solve_exterior_response
from ehl_solver.flux_diagnostics import face_components
from ehl_solver.rectilinear_cavitation import flux_audit, solve_rectilinear_cavitation
from examples.run_farfield_budget import ETA, H0, LY, P_REF, RX, RY, SPEED, L, accepted, fronts
from examples.run_scientific_validation import provenance, save_json
from examples.scientific_figures import figure, save


def interface_budget(x, y, h, p, theta, mask):
    """Outward SI flux around retained-node midpoint cells, not node rectangle."""
    c, px, py, residual = face_components(x, y, h, p, theta, ETA, SPEED)
    qx = c + px
    j, i = np.where(mask)
    j0, j1, i0, i1 = j.min(), j.max(), i.min(), i.max()
    wx, wy = (x[2:] - x[:-2]) / 2, (y[2:] - y[:-2]) / 2
    vx, vy = wx[i0 - 1 : i1], wy[j0 - 1 : j1]
    flows = [
        -qx[j0 : j1 + 1, i0 - 1] @ vy,
        qx[j0 : j1 + 1, i1] @ vy,
        -py[j0 - 1, i0 : i1 + 1] @ vx,
        py[j1, i0 : i1 + 1] @ vx,
    ]
    vol = residual[j0 - 1 : j1, i0 - 1 : i1] * vy[:, None] * vx
    return {
        "outward_volume_m3_s": [float(v) for v in flows],
        "net_volume_m3_s": float(sum(flows)),
        "summed_volume_defect_m3_s": float(vol.sum()),
        "relative_global_balance": float(abs(sum(flows)) / sum(abs(v) for v in flows)),
    }


def fields(x, y, h, p, theta):
    return dict(x_m=x, y_m=y, film_m=h, pressure_pa=p, theta=theta)


def render(directory, report):
    import matplotlib.pyplot as plt

    context = (
        "Rigid isoviscous finite point reservoir: h0=.1 um, Rx=10 mm, Ry=20 mm\n"
        "eta=.04 Pa s, Ue=.001 m/s; physical edges +/-8L,+/-8Ly ambient; west theta=1\n"
        "L=44.7214 um, Ly=63.2456 um; numerical cut is not a new reservoir"
    )
    fig, (ax,) = figure("Fixed reservoir: numerical cut independence", context)
    cuts = report["cases"]
    ax.plot(
        [c["cut_label"] for c in cuts],
        [c["source_budget"]["total_load_n"] for c in cuts],
        "o-",
        label="Exterior response, full load",
    )
    ax.axhline(
        report["reference_load_n"],
        label="Independent uniform reference",
        linestyle=":",
        color="black",
    )
    ax.plot(
        [c["cut_label"] for c in cuts],
        [c["zero_cut_load_n"] for c in cuts],
        "s--",
        label="New ambient reservoir at cut (different problem)",
    )
    ax.set(xlabel="Computational cut half-width / L (y / Ly)", ylabel="Load [N]")
    ax.legend()
    save(fig, directory, "cut_load")
    fig, axes = figure("Exterior sources retained in force and deformation", context, rows=2)
    for key in ("retained_source_load_n", "exterior_source_load_n"):
        axes[0].plot(
            [c["cut_label"] for c in cuts], [c["source_budget"][key] for c in cuts], "o-", label=key
        )
    for key in ("retained_displacement_m", "exterior_displacement_m", "total_displacement_m"):
        axes[1].plot(
            [c["cut_label"] for c in cuts],
            [c["source_budget"][key][0] * 1e9 for c in cuts],
            "o-",
            label=key,
        )
    axes[0].set_ylabel("Disjoint source-cell load [N]")
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel="Cut half-width / L", ylabel="Center displacement [nm], E*=1 GPa")
    axes[1].legend(fontsize=8)
    save(fig, directory, "source_budget")
    ref = np.load(directory / "reference.npz")
    fig, axes = figure("Reconstructed pressure and transported content", context, rows=2)
    for c in cuts:
        f = np.load(directory / c["state_file"])
        j = len(f["y_m"]) // 2
        axes[0].plot(f["x_m"] * 1e3, f["pressure_pa"][j] / 1e6, label=f"cut {c['cut_label']}")
        axes[1].plot(f["x_m"] * 1e3, f["theta"][j], label=f"cut {c['cut_label']}")
    axes[0].plot(
        ref["x_m"] * 1e3,
        ref["pressure_pa"][len(ref["y_m"]) // 2] / 1e6,
        "k:",
        label="Independent reference",
    )
    axes[0].set_ylabel("Pressure [MPa]")
    axes[0].legend()
    axes[1].set(xlabel="x [mm], entrainment +x", ylabel="Fractional content theta")
    axes[1].legend()
    save(fig, directory, "pressure_content")
    fig, (ax,) = figure("Cavitated exterior is solved, not prescribed dry", context)
    ax.contourf(ref["x_m"] * 1e3, ref["y_m"] * 1e3, ref["theta"], levels=np.linspace(0, 1, 21))
    for cut in (1, 2, 3):
        ax.plot(
            np.array([-cut, cut, cut, -cut, -cut]) * L * 1e3,
            np.array([-cut, -cut, cut, cut, -cut]) * LY * 1e3,
            label=f"cut {cut}",
        )
    ax.set(xlabel="x [mm], entrainment +x", ylabel="y [mm]")
    ax.legend()
    save(fig, directory, "exterior_content")
    plt.close("all")


def main(directory):
    directory.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "incomplete",
        "runtime": provenance(),
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
            "outer_label": 8,
            "step": 0.25,
        },
        "scope": "Exact finite-reservoir prescribed-film exterior elimination; no coupled solve",
        "criteria": {
            "mass_maximum": 1e-8,
            "relative_local_global_balance": 1e-10,
            "complementarity_admissibility": 1e-10,
            "relative_reference_load": 1e-8,
            "relative_reference_pressure": 1e-8,
        },
    }

    def checkpoint():
        save_json(directory / "summary.json", report)

    checkpoint()
    try:
        units = np.linspace(-8, 8, 65)
        x, y = units * L, units * LY
        h = H0 + x[None, :] ** 2 / (2 * RX) + y[:, None] ** 2 / (2 * RY)
        ref = solve_cavitating_reynolds(h / H0, 0.25, 0.25 * LY / L, 1)
        rp = ref.pressure * P_REF
        rt = ref.fractional_film_content
        np.savez_compressed(directory / "reference.npz", **fields(x, y, h, rp, rt))
        scales = dict(length_scale_m=L, film_scale_m=H0, pressure_scale_pa=P_REF)
        report["reference_audit"] = flux_audit(x, y, h, rp, rt, ETA, SPEED, **scales)
        report["reference_load_n"] = float(trapezoid(trapezoid(rp, x, axis=1), y))
        assert accepted(report["reference_audit"], fields(x, y, h, rp, rt))
        for cut in (1, 2, 3):
            result = solve_exterior_response(
                x,
                y,
                h,
                ETA,
                SPEED,
                core_bounds_m=(-cut * L, cut * L, -cut * LY, cut * LY),
                **scales,
            )
            f = fields(x, y, h, result.pressure_pa, result.fractional_film_content)
            state = f"cut_{cut}.npz"
            np.savez_compressed(directory / state, **f, retained_mask=result.retained_mask)
            budget = exterior_source_budget(x, y, result, 1e9, [[0, 0], [-0.5 * L, 0]])
            ix = np.flatnonzero(abs(x) <= cut * L)
            iy = np.flatnonzero(abs(y) <= cut * LY)
            local = solve_rectilinear_cavitation(
                x[ix], y[iy], h[np.ix_(iy, ix)], ETA, SPEED, **scales
            )
            entry = {
                "cut_label": cut,
                "state_file": state,
                "audit": result.audit,
                "history": result.history,
                "source_budget": budget,
                "interface_budget": interface_budget(
                    x,
                    y,
                    h,
                    result.pressure_pa,
                    result.fractional_film_content,
                    result.retained_mask,
                ),
                "pressure_relative_error": float(
                    np.linalg.norm(result.pressure_pa - rp) / np.linalg.norm(rp)
                ),
                "theta_maximum_error": float(abs(result.fractional_film_content - rt).max()),
                "relative_load_error": abs(budget["total_load_n"] / report["reference_load_n"] - 1),
                "fronts": fronts(x, result.fractional_film_content),
                "zero_cut_load_n": float(
                    trapezoid(trapezoid(local.pressure_pa, x[ix], axis=1), y[iy])
                ),
            }
            entry["accepted"] = (
                accepted(result.audit, f)
                and entry["pressure_relative_error"] <= 1e-8
                and entry["relative_load_error"] <= 1e-8
            )
            report["cases"].append(entry)
            checkpoint()
            print(
                cut,
                entry["accepted"],
                entry["pressure_relative_error"],
                result.audit["mass_maximum"],
                flush=True,
            )
        render(directory, report)
        report["figures"] = [p.name for p in sorted(directory.glob("*.png"))]
        report["status"] = (
            "verified finite-reservoir response"
            if all(c["accepted"] for c in report["cases"])
            else "failed acceptance"
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
    parser.add_argument("--output-directory", type=Path, default=Path("results/exterior_response"))
    main(parser.parse_args().output_directory)
