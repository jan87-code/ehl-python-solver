"""Conditional exterior film derivatives and all-inner elastic-source audit.

Uses the unchanged full uniform coupled solver as an independent reference.
No condensed-exterior nonlinear coupling is introduced by this experiment.
"""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid

from ehl_solver.backend import make_backend
from ehl_solver.elasticity import elastic_deformation
from ehl_solver.exterior import (
    differentiate_exterior_response,
    retained_source_displacements,
    solve_exterior_response,
)
from examples.run_exterior_response import fields
from examples.run_farfield_budget import ETA, H0, LY, P_REF, RX, RY, SPEED, L, accepted, solve_case
from examples.run_scientific_validation import provenance, save_json
from examples.scientific_figures import figure, save


def main(directory):
    directory.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "incomplete",
        "runtime": provenance(),
        "directions": [],
        "scope": (
            "Fixed-active-set prescribed-film derivative and frozen coupled-root audit; "
            "no new coupled solver"
        ),
        "criteria": {
            "max_mass": 1e-8,
            "relative_constraints_local_global": 1e-10,
            "relative_frozen_pressure": 1e-8,
            "relative_conditional_fd": 1e-5,
            "absolute_direct_fft_m": 1e-18,
        },
    }

    def checkpoint():
        save_json(directory / "summary.json", report)

    checkpoint()
    try:
        scales = dict(length_scale_m=L, film_scale_m=H0, pressure_scale_pa=P_REF)
        u = np.linspace(-8, 8, 33)
        x, y = u * L, u * LY
        h = H0 + x[None, :] ** 2 / (2 * RX) + y[:, None] ** 2 / (2 * RY)

        def solve(film):
            return solve_exterior_response(
                x, y, film, ETA, SPEED, core_bounds_m=(-2 * L, 2 * L, -2 * LY, 2 * LY), **scales
            )

        base = solve(h)
        np.savez_compressed(
            directory / "rigid_base.npz",
            **fields(x, y, h, base.pressure_pa, base.fractional_film_content),
            active_set=base.full_active_set,
        )
        for location in (-0.5, -4):
            dh = H0 * np.exp(-((x[None, :] / L - location) ** 2 + (y[:, None] / LY) ** 2) / 2)
            tangent = differentiate_exterior_response(x, y, h, dh, base, ETA, SPEED, **scales)
            entry = {
                "direction_center_x_over_L": location,
                "linear_info": tangent.linear_info,
                "finite_differences": [],
            }
            for eps in (1e-2, 3e-3, 1e-3, 3e-4):
                plus, minus = solve(h + eps * dh), solve(h - eps * dh)
                fixed = np.array_equal(
                    plus.full_active_set, base.full_active_set
                ) and np.array_equal(minus.full_active_set, base.full_active_set)
                dp = (plus.pressure_pa - minus.pressure_pa) / (2 * eps)
                dt = (plus.fractional_film_content - minus.fractional_film_content) / (2 * eps)
                item = {
                    "epsilon": eps,
                    "same_active_set": bool(fixed),
                    "relative_pressure_error": float(
                        np.linalg.norm(dp - tangent.pressure_direction_pa) / np.linalg.norm(dp)
                    ),
                    "relative_content_error": float(
                        np.linalg.norm(dt - tangent.content_direction) / np.linalg.norm(dt)
                    ),
                    "plus_audit": plus.audit,
                    "minus_audit": minus.audit,
                }
                analytic_load = float(
                    trapezoid(trapezoid(tangent.pressure_direction_pa, x, axis=1), y)
                )
                fd_load = float(trapezoid(trapezoid(dp, x, axis=1), y))
                item["analytic_load_direction_n"] = analytic_load
                item["relative_load_direction_error"] = abs(fd_load / analytic_load - 1)
                item["physical_audits_passed"] = bool(
                    accepted(
                        plus.audit,
                        fields(x, y, h + eps * dh, plus.pressure_pa, plus.fractional_film_content),
                    )
                    and accepted(
                        minus.audit,
                        fields(
                            x, y, h - eps * dh, minus.pressure_pa, minus.fractional_film_content
                        ),
                    )
                )
                entry["finite_differences"].append(item)
                np.savez_compressed(
                    directory / f"direction_{location}_{eps}.npz",
                    film_direction_m=dh,
                    analytic_pressure_direction_pa=tangent.pressure_direction_pa,
                    analytic_content_direction=tangent.content_direction,
                    plus_pressure_pa=plus.pressure_pa,
                    minus_pressure_pa=minus.pressure_pa,
                    plus_theta=plus.fractional_film_content,
                    minus_theta=minus.fractional_film_content,
                )
            # Signed tangent source response is linear, independently checked at ALL inner targets.
            direct = retained_source_displacements(
                x, y, replace(base, pressure_pa=tangent.pressure_direction_pa), 1e9
            )
            fft = elastic_deformation(
                tangent.pressure_direction_pa, x[1] - x[0], y[1] - y[0], 1e9, make_backend("numpy")
            )
            error = float(abs(direct["total_displacement_m"] - fft[base.retained_mask]).max())
            entry["all_inner_direct_fft_error_m"] = error
            entry["inner_targets"] = len(direct["target_points_m"])
            last = entry["finite_differences"][-1]
            entry["accepted"] = bool(
                last["same_active_set"]
                and max(last["relative_pressure_error"], last["relative_content_error"]) <= 1e-5
                and last["relative_load_direction_error"] <= 1e-5
                and all(c["physical_audits_passed"] for c in entry["finite_differences"])
                and error <= 1e-18
            )
            np.savez_compressed(
                directory / f"direction_sources_{location}.npz",
                **direct,
                fft_displacement_m=fft[base.retained_mask],
            )
            report["directions"].append(entry)
            checkpoint()
            print(
                "direction",
                location,
                last["relative_pressure_error"],
                last["relative_content_error"],
                error,
                flush=True,
            )
        reference, f = solve_case(
            "full_uniform_coupled", (8, 8, 8), 0.25, family="coupled", uniform=True, snap=False
        )
        report["coupled_reference"] = reference
        np.savez_compressed(directory / "coupled_reference.npz", **f)
        x, y, h = f["x_m"], f["y_m"], f["film_m"]
        result = solve_exterior_response(
            x, y, h, ETA, SPEED, core_bounds_m=(-2 * L, 2 * L, -2 * LY, 2 * LY), **scales
        )
        direct = retained_source_displacements(x, y, result, 1e9)
        error = float(
            abs(direct["total_displacement_m"] - f["displacement_m"][result.retained_mask]).max()
        )
        entry = {
            "audit": result.audit,
            "history": result.history,
            "inner_targets": len(direct["target_points_m"]),
            "relative_pressure_error": float(
                np.linalg.norm(result.pressure_pa - f["pressure_pa"])
                / np.linalg.norm(f["pressure_pa"])
            ),
            "maximum_content_error": float(abs(result.fractional_film_content - f["theta"]).max()),
            "maximum_direct_vs_coupled_fft_error_m": error,
            "maximum_source_partition_error_m": float(
                abs(
                    direct["retained_displacement_m"]
                    + direct["exterior_displacement_m"]
                    - direct["total_displacement_m"]
                ).max()
            ),
        }
        recovered_load = float(trapezoid(trapezoid(result.pressure_pa, x, axis=1), y))
        entry["relative_load_error"] = abs(recovered_load / reference["load_n"] - 1)
        entry["accepted"] = bool(
            reference["accepted"]
            and accepted(
                result.audit, fields(x, y, h, result.pressure_pa, result.fractional_film_content)
            )
            and entry["relative_pressure_error"] <= 1e-8
            and entry["relative_load_error"] <= 1e-8
            and error <= 1e-18
        )
        report["frozen_coupled_root"] = entry
        np.savez_compressed(
            directory / "frozen_response.npz",
            **fields(x, y, h, result.pressure_pa, result.fractional_film_content),
            active_set=result.full_active_set,
            **direct,
            reference_displacement_m=f["displacement_m"][result.retained_mask],
        )
        checkpoint()
        render(directory, report)
        report["figures"] = [p.name for p in sorted(directory.glob("*.png"))]
        report["status"] = (
            "verified preparatory coupling audits"
            if all(d["accepted"] for d in report["directions"]) and entry["accepted"]
            else "failed acceptance"
        )
        checkpoint()
        print("frozen root", entry["relative_pressure_error"], error, entry["accepted"], flush=True)
    except Exception as exc:
        report["status"] = "failed"
        report["failure"] = repr(exc)
        checkpoint()
        raise
    return report


def render(directory, report):
    context = (
        "Finite point reservoir +/-8L,+/-8Ly; h0=.1 um,Rx=10 mm,Ry=20 mm\n"
        "eta=.04 Pa s,Ue=.001 m/s; ambient outer p, flooded west; float64 CPU\n"
        "Conditional derivative at fixed wet/dry set; no new exterior nonlinear coupling"
    )
    fig, axes = figure(
        "Prescribed-film exterior derivative against independent nonlinear solves", context, rows=2
    )
    for d in report["directions"]:
        for ax, key in zip(
            axes, ("relative_pressure_error", "relative_content_error"), strict=True
        ):
            c = d["finite_differences"]
            ax.loglog(
                [a["epsilon"] for a in c],
                [a[key] for a in c],
                "o-",
                label=f"direction center {d['direction_center_x_over_L']}L",
            )
            ax.set_ylabel(key)
            ax.legend(fontsize=8)
    axes[-1].set_xlabel("Dimensionless perturbation amplitude epsilon")
    save(fig, directory, "film_derivative")
    f = np.load(directory / "frozen_response.npz")
    fig, axes = figure("Every inner target retains exterior elastic sources", context, rows=2)
    for key in ("total_displacement_m", "exterior_displacement_m"):
        axes[0].plot(f[key] * 1e9, label=key)
    axes[0].set_ylabel("Displacement [nm], E*=1 GPa")
    axes[0].legend()
    axes[1].plot((f["total_displacement_m"] - f["reference_displacement_m"]) * 1e18)
    axes[1].set(
        xlabel="Inner target index, (ny,nx) row-major", ylabel="Direct minus coupled FFT [1e-18 m]"
    )
    save(fig, directory, "all_inner_sources")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory", type=Path, default=Path("results/exterior_coupling_audit")
    )
    main(parser.parse_args().output_directory)
