"""Explain saved point-domain errors and verify bounded flux/boundary improvements.

CPU only. Reuses raw states without moving their reservoir boundaries.
The matched-boundary experiment is an independent rigid infinite-width
analytical limit; its boundary data are not transferred to point contact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad, trapezoid

from ehl_solver.flux_diagnostics import decompose_flux, nested_load_budget
from ehl_solver.parabolic_reference import parabolic_jfo_reference
from ehl_solver.rectilinear_cavitation import solve_rectilinear_cavitation
from examples.run_farfield_budget import H0, LY, P_REF, L, accepted
from examples.run_scientific_validation import load_fields, provenance, save_json
from examples.scientific_figures import figure, save


def matched_case(inlet, nx, *, mode="matched"):
    """SI original reservoir at -8L; finite windows and exact transverse data.

    A matched window has the original pressure at its inlet and sides.
    A finite reservoir instead has p=0 at its own inlet, with its own exact
    infinite-width side pressures. These are different physical problems.
    Only matched reconstruction includes the original upstream force tail.
    """
    h0, radius, eta, speed = 20e-6, 0.01, 0.04, 0.425
    length = np.sqrt(2 * radius * h0)
    x = np.linspace(inlet * length, 3 * length, nx)
    y = np.linspace(-4 * length, 4 * length, 9)
    original = parabolic_jfo_reference(np.r_[-8 * length, x], h0, radius, eta, speed)
    local = parabolic_jfo_reference(x, h0, radius, eta, speed)
    target = original.pressure_pa[1:]
    pressure_bc = target if mode == "matched" else local.pressure_pa
    h = np.broadcast_to(local.film_thickness_m, (len(y), len(x))).copy()
    boundary = np.broadcast_to(pressure_bc, h.shape)
    solution = solve_rectilinear_cavitation(
        x,
        y,
        h,
        eta,
        speed,
        length_scale_m=length,
        film_scale_m=h0,
        pressure_scale_pa=original.pressure_scale_pa,
        boundary_pressure_pa=boundary,
    )
    center = solution.pressure_pa[len(y) // 2]
    force = float(trapezoid(center, x))
    hc = h0 + original.rupture_location_m**2 / (2 * radius)

    def derivative(t):
        film = h0 + t * t / (2 * radius)
        return 12 * eta * speed * (film - hc) / film**3

    # Independent integration-by-parts tail integral; no sample-grid fit.
    tail = quad(lambda t: (x[0] - t) * derivative(t), -8 * length, x[0], epsabs=1e-9)[0]
    expected_window = original.force_per_width_n_m - tail
    reconstructed = force + tail if mode == "matched" else force
    entry = {
        "mode": mode,
        "nx": nx,
        "inlet_over_L": inlet,
        "shape": list(h.shape),
        "x_limits_m": [float(x[0]), float(x[-1])],
        "y_limits_m": [float(y[0]), float(y[-1])],
        "L_m": float(length),
        "h0_m": h0,
        "Rx_m": radius,
        "eta_pa_s": eta,
        "Ue_m_s": speed,
        "original_reservoir_x_m": float(-8 * length),
        "inlet_pressure_pa": float(pressure_bc[0]),
        "original_force_per_width_n_m": original.force_per_width_n_m,
        "window_force_per_width_n_m": force,
        "original_upstream_tail_n_m": tail,
        "original_window_force_n_m": expected_window,
        "represented_force_per_width_n_m": reconstructed,
        "relative_force_difference": abs(reconstructed / original.force_per_width_n_m - 1),
        "relative_pressure_difference": float(
            np.linalg.norm(center - target) / np.linalg.norm(target)
        ),
        "audit": solution.audit,
        "active_set_iterations": solution.iterations,
        "pressure_max_pa": float(solution.pressure_pa.max()),
        "theta_min": float(solution.fractional_film_content.min()),
    }
    entry["accepted"] = bool(
        solution.audit["mass_maximum"] <= 1e-8
        and solution.audit["relative_local_balance"] <= 1e-10
        and solution.audit["relative_global_balance"] <= 1e-10
        and solution.audit["maximum_complementarity"] <= 1e-10
        and solution.pressure_pa.min() >= -1e-10 * original.pressure_scale_pa
        and solution.fractional_film_content.min() >= -1e-10
        and solution.fractional_film_content.max() <= 1 + 1e-10
    )
    fields = {
        "x_m": x,
        "y_m": y,
        "film_m": h,
        "pressure_pa": solution.pressure_pa,
        "theta": solution.fractional_film_content,
        "reference_pressure_pa": target,
    }
    return entry, fields


def analyze_pair(first, second):
    d = nested_load_budget(
        first["x_m"],
        first["y_m"],
        first["pressure_pa"],
        second["x_m"],
        second["y_m"],
        second["pressure_pa"],
    )

    def core(f):
        ix = abs(f["x_m"]) <= 2 * L + 1e-18
        iy = abs(f["y_m"]) <= 2 * LY + 1e-18
        return float(
            trapezoid(
                trapezoid(f["pressure_pa"][np.ix_(iy, ix)], f["x_m"][ix], axis=1), f["y_m"][iy]
            )
        )

    d["inner_core_readjustment_n"] = core(second) - core(first)
    d["shared_exterior_readjustment_n"] = (
        d["shared_area_readjustment_n"] - d["inner_core_readjustment_n"]
    )
    return d


def matched_point_case(full):
    """Known finite point-reference trace; rigid prescribed film only.

    This verifies a subdomain boundary operator, not a self-contained outer
    model: boundary pressures and outside load are borrowed from the larger
    independently audited finite reference. No elastic tail is discarded.
    """
    ix = np.flatnonzero(abs(full["x_m"]) <= 8 * L + 1e-18)
    iy = np.flatnonzero(abs(full["y_m"]) <= 8 * LY + 1e-18)
    x, y = full["x_m"][ix], full["y_m"][iy]
    h = full["film_m"][np.ix_(iy, ix)]
    target = full["pressure_pa"][np.ix_(iy, ix)]
    options = dict(length_scale_m=L, film_scale_m=H0, pressure_scale_pa=P_REF)
    zero = solve_rectilinear_cavitation(x, y, h, 0.04, 0.001, **options)
    matched = solve_rectilinear_cavitation(
        x, y, h, 0.04, 0.001, boundary_pressure_pa=target, **options
    )

    def load(p, xx, yy):
        return float(trapezoid(trapezoid(p, xx, axis=1), yy))

    total = load(full["pressure_pa"], full["x_m"], full["y_m"])
    shared = load(target, x, y)
    tail = total - shared
    entry = {
        "full_shape": list(full["pressure_pa"].shape),
        "window_shape": list(h.shape),
        "full_x_limits_m": [float(full["x_m"][0]), float(full["x_m"][-1])],
        "window_x_limits_m": [float(x[0]), float(x[-1])],
        "reference_full_load_n": total,
        "known_outside_load_n": tail,
        "zero_boundary_window_load_n": load(zero.pressure_pa, x, y),
        "matched_window_load_n": load(matched.pressure_pa, x, y),
        "matched_reconstructed_load_n": load(matched.pressure_pa, x, y) + tail,
        "pressure_relative_difference": float(
            np.linalg.norm(matched.pressure_pa - target) / np.linalg.norm(target)
        ),
        "zero_boundary_audit": zero.audit,
        "matched_audit": matched.audit,
    }
    fields = {
        "x_m": x,
        "y_m": y,
        "film_m": h,
        "pressure_pa": matched.pressure_pa,
        "theta": matched.fractional_film_content,
        "reference_pressure_pa": target,
        "zero_boundary_pressure_pa": zero.pressure_pa,
    }
    return entry, fields


def render(directory, report):
    import matplotlib.pyplot as plt

    point = report["point_cases"]
    domain = [c for c in point if c["family"] == "all"]
    context = (
        "Saved isoviscous point states; h0=.1 um, Rx=.01 m, Ry=.02 m\n"
        "eta=.04 Pa s, Ue=.001 m/s; west content=1, zero edge pressure\n"
        "Outward boundary budgets and raw/rejected states retained; no new reservoir model imposed"
    )
    for name, keys, label in (
        (
            "inlet_components",
            ["inlet_couette_m3_s", "inlet_poiseuille_m3_s", "inlet_net_m3_s"],
            "Positive-inward inlet flow [m^3/s]",
        ),
        (
            "cancellation",
            ["inlet_cancellation_condition", "maximum_face_cancellation_condition"],
            "sum(abs(parts))/abs(net)",
        ),
        (
            "continuity",
            ["mass_maximum_float64", "mass_maximum_wider", "one_pressure_ulp_defect_change"],
            "Original dimensionless continuity defect / sensitivity",
        ),
    ):
        fig, (ax,) = figure(name.replace("_", " "), context)
        for key in keys:
            ax.plot(
                [c["extent_over_L"] for c in domain],
                [c["flux"][key] for c in domain],
                "o-",
                label=key,
            )
        if name == "continuity":
            ax.set_yscale("log")
            ax.axhline(1e-8, color="red", linestyle=":")
        ax.set(xlabel="Actual upstream extent / L", ylabel=label)
        ax.legend(fontsize=8)
        save(fig, directory, name)
    fig, (ax,) = figure("Point-domain load change: added area versus readjustment", context)
    labels = []
    bottom = np.zeros(len(report["load_budgets"]))
    for key in ("added_area_load_n", "shared_exterior_readjustment_n", "inner_core_readjustment_n"):
        vals = [d[key] for d in report["load_budgets"]]
        labels = [d["pair"] for d in report["load_budgets"]]
        labels = [
            label + (" [raw]" if not d["both_originally_accepted"] else "")
            for label, d in zip(labels, report["load_budgets"], strict=True)
        ]
        ax.barh(labels, vals, left=bottom, label=key)
        bottom += vals
    ax.tick_params(axis="y", labelsize=7)
    ax.set_xlabel("Load change [N]")
    ax.legend(fontsize=8)
    save(fig, directory, "load_change_accounting")
    fig, (ax,) = figure(
        "Independent original SI continuity after bounded defect correction", context
    )
    for c in report["refined_cases"]:
        if "history" not in c:
            continue
        values = [c["before"]["mass_maximum"]] + [t["trial"] for t in c["history"]]
        ax.semilogy(range(len(values)), values, "o-", label=c["name"])
    ax.axhline(1e-8, color="red", linestyle=":")
    ax.set(xlabel="Physical-flux defect-correction trial", ylabel="Original float64 SI max defect")
    ax.legend()
    save(fig, directory, "flux_refinement")
    matched = report["matched_cases"]
    reference = matched[0]["original_force_per_width_n_m"]
    context1 = (
        "Independent rigid infinite-width parabolic limit; hmin=20 um, Rx=.01 m\n"
        "eta=.04 Pa s, Ue=.425 m/s; original physical reservoir at -8L, outlet=3L\n"
        "Exact transverse boundary pressures; matched numerical inlet uses original pressure\n"
        "Finite zero-inlet pressure defines a different reservoir; "
        "no point-contact boundary adopted"
    )
    fig, (ax,) = figure(
        "Matched original reservoir versus a shortened physical reservoir", context1
    )
    for mode in ("matched", "finite_reservoir"):
        cases = [c for c in matched if c["mode"] == mode and c["nx"] == 129]
        ax.plot(
            [c["inlet_over_L"] for c in cases],
            [c["represented_force_per_width_n_m"] for c in cases],
            "o-",
            label=mode,
        )
    ax.axhline(reference, color="black", linestyle=":", label="Original exact force/width")
    ax.set(
        xlabel="Computational inlet x/L",
        ylabel="Force per width [N/m]; matched includes exact upstream tail",
    )
    ax.legend()
    save(fig, directory, "reservoir_boundary_comparison")
    fig, (ax,) = figure("Fixed matched domain mesh refinement", context1)
    cases = [c for c in matched if c["mode"] == "matched" and c["inlet_over_L"] == -1]
    cases.sort(key=lambda c: c["nx"])
    ax.loglog(
        [c["nx"] for c in cases],
        [c["relative_pressure_difference"] for c in cases],
        "o-",
        label="Pressure",
    )
    ax.loglog(
        [c["nx"] for c in cases],
        [c["relative_force_difference"] for c in cases],
        "o-",
        label="Reconstructed force/width",
    )
    ax.set(xlabel="Streamwise nodes", ylabel="Relative difference from original exact solution")
    ax.legend()
    save(fig, directory, "matched_mesh_refinement")
    for key, factor, ylabel in (
        ("pressure_pa", 1e-6, "Pressure [MPa]"),
        ("film_m", 1e6, "Film [um]"),
        ("theta", 1.0, "Fractional film content"),
    ):
        fig, (ax,) = figure("Matched-limit common-coordinate centerlines", context1)
        for c in matched:
            if c["nx"] != 129:
                continue
            f = load_fields(directory / c["state_file"])
            ax.plot(
                f["x_m"] * 1e3,
                f[key][len(f["y_m"]) // 2] * factor,
                label=f"{c['mode']} {c['inlet_over_L']}",
            )
        ax.set(
            xlim=(-0.5 * matched[0]["L_m"] * 1e3, 3 * matched[0]["L_m"] * 1e3),
            xlabel="x [mm], inlet to outlet",
            ylabel=ylabel,
        )
        ax.legend(fontsize=7)
        save(fig, directory, f"matched_{key}")
    # Raw saturation contours in one shared physical coordinate window.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    fig.suptitle(context1, fontsize=8)
    for ax, mode in zip(axes, ("matched", "finite_reservoir"), strict=True):
        c = next(
            c for c in matched if c["mode"] == mode and c["nx"] == 129 and c["inlet_over_L"] == -1
        )
        f = load_fields(directory / c["state_file"])
        im = ax.pcolormesh(
            f["x_m"] * 1e3, f["y_m"] * 1e3, f["theta"], shading="nearest", vmin=0, vmax=1
        )
        ax.set(title=mode, xlabel="x [mm], inlet to outlet", ylabel="y [mm]")
    fig.colorbar(im, ax=axes, label="Fractional film content")
    save(fig, directory, "matched_saturation_contours")
    fig, axes = figure(
        "Point-state extrema versus extent; this is not domain validation", context, rows=2
    )
    selected = [c for c in point if c["family"] == "all"]
    axes[0].plot(
        [c["extent_over_L"] for c in selected],
        [c["extrema"]["minimum_film"]["value_si"] * 1e6 for c in selected],
        "o-",
    )
    axes[0].set(xlabel="Upstream extent / L", ylabel="Rigid prescribed minimum film [um]")
    axes[1].plot(
        [c["extent_over_L"] for c in selected],
        [c["extrema"]["maximum_pressure"]["value_si"] / 1e6 for c in selected],
        "o-",
    )
    axes[1].set(xlabel="Upstream extent / L", ylabel="Sampled maximum pressure [MPa]")
    save(fig, directory, "point_extrema")
    for key, factor, label in (
        ("pressure_pa", 1e-6, "Pressure [MPa]"),
        ("film_m", 1e6, "Film [um]"),
    ):
        fig, (ax,) = figure(
            "Common physical point-contact window; no interpolation or smoothing", context
        )
        for name in ("all_64", "all_128", "coupled_extent_8", "coupled_extent_16"):
            f = load_fields(directory / f"{name}.npz")
            ax.plot(f["x_m"] * 1e3, f[key][len(f["y_m"]) // 2] * factor, ".-", label=name)
        ax.set(xlim=(-2 * L * 1e3, 2 * L * 1e3), xlabel="x [mm], inlet to outlet", ylabel=label)
        ax.legend()
        save(fig, directory, f"point_{key}")
    fig, (ax,) = figure("Coupling control on identical uniform grids and boundaries", context)
    for key in ("rigid_load_n", "coupled_load_n"):
        ax.plot((8, 16), [c[key] for c in report["coupling_controls"]], "o-", label=key)
    ax.set(xlabel="Uniform domain extent / natural length", ylabel="Load [N]")
    ax.legend()
    save(fig, directory, "coupling_control")
    fig, (ax,) = figure("Different geometries have different inlet cancellation", context)
    ax.semilogy(
        [c["extent_over_L"] for c in domain],
        [c["flux"]["inlet_cancellation_condition"] for c in domain],
        "o-",
        label="2-D point",
    )
    ax.semilogy(
        [-c["inlet_over_L"] for c in report["analytical_inlet_flux"]],
        [c["cancellation_condition"] for c in report["analytical_inlet_flux"]],
        "o-",
        label="Exact infinite-width rigid limit",
    )
    ax.set(xlabel="Upstream extent / natural length", ylabel="sum(abs(parts))/abs(net)")
    ax.legend()
    save(fig, directory, "analytical_cancellation")
    point_match = report["matched_point_reference"]
    f = load_fields(directory / "matched_point_reference.npz")
    fig, axes = figure(
        "Known finite point-reservoir trace: verified subdomain restriction",
        "Rigid point reference; hmin=.1 um at origin, Rx=.01 m, Ry=.02 m\n"
        "eta=.04 Pa s, Ue=.001 m/s; flooded west, ambient pressure on FULL outer edges\n"
        "Computational inner window instead receives the known nonzero pressure trace\n"
        "Full reference: 129x129, x=+/-16L, y=+/-16Ly; inner window=+/-8L,+/-8Ly\n"
        "Matched reconstruction includes a known outside load; "
        "no independent point far-field law claimed",
        rows=2,
    )
    axes[0].bar(
        ("Full reference", "Zero-edge inner only", "Matched inner + known tail"),
        (
            point_match["reference_full_load_n"],
            point_match["zero_boundary_window_load_n"],
            point_match["matched_reconstructed_load_n"],
        ),
    )
    axes[0].set_ylabel("Load [N]")
    for key in ("zero_boundary_pressure_pa", "pressure_pa", "reference_pressure_pa"):
        axes[1].plot(f["x_m"] * 1e3, f[key][len(f["y_m"]) // 2] / 1e6, label=key)
    axes[1].set(xlabel="x [mm], inlet to outlet", ylabel="Pressure [MPa]")
    axes[1].legend()
    save(fig, directory, "matched_point_boundary")


def main(input_directory, directory):
    directory.mkdir(parents=True, exist_ok=False)
    source = json.loads((input_directory / "summary.json").read_text())
    save_json(directory / "source_budget_summary.json", source)
    report = {
        "runtime": provenance(),
        "input_summary_sha256": hashlib.sha256(
            (input_directory / "summary.json").read_bytes()
        ).hexdigest(),
        "point_cases": [],
        "load_budgets": [],
        "refined_cases": [],
        "matched_cases": [],
        "coupling_controls": [],
        "analytical_inlet_flux": [],
        "scope": "flux/force accounting and numerical / 1-D matched-boundary experiments",
    }
    states = {}

    def checkpoint():
        save_json(directory / "summary.json", report)

    for c in source["cases"]:
        name = c["name"]
        p = input_directory / f"{name}.npz"
        f = load_fields(p)
        states[name] = f
        np.savez_compressed(directory / f"{name}.npz", **f)
        d = decompose_flux(
            f["x_m"],
            f["y_m"],
            f["film_m"],
            f["pressure_pa"],
            f["theta"],
            0.04,
            0.001,
            length_scale_m=L,
            film_scale_m=H0,
            pressure_scale_pa=P_REF,
        )
        report["point_cases"].append(
            {
                "name": name,
                "family": c["family"],
                "originally_accepted": c["accepted"],
                "extent_over_L": abs(c["x_limits_m"][0]) / L,
                "load_n": c["load_n"],
                "extrema": c["extrema"],
                "flux": d,
                "input_npz_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            }
        )
        checkpoint()
    for family in ("all", "coarse_all"):
        cases = [c for c in source["cases"] if c["family"] == family]
        for a, b in zip(cases, cases[1:], strict=False):
            d = analyze_pair(states[a["name"]], states[b["name"]])
            d["pair"] = a["name"] + " -> " + b["name"]
            d["both_originally_accepted"] = a["accepted"] and b["accepted"]
            report["load_budgets"].append(d)
    a, b = "coupled_extent_8", "coupled_extent_16"
    d = analyze_pair(states[a], states[b])
    d.update(
        pair=a + " -> " + b,
        both_originally_accepted=all(
            next(c for c in source["cases"] if c["name"] == name)["accepted"] for name in (a, b)
        ),
    )
    report["load_budgets"].append(d)
    checkpoint()
    for name in (a, b):
        f = states[name]
        x, y = f["x_m"], f["y_m"]
        h = H0 + x[None, :] ** 2 / 0.02 + y[:, None] ** 2 / 0.04
        rigid = solve_rectilinear_cavitation(
            x, y, h, 0.04, 0.001, length_scale_m=L, film_scale_m=H0, pressure_scale_pa=P_REF
        )
        rigid_fields = f | {
            "pressure_pa": rigid.pressure_pa,
            "theta": rigid.fractional_film_content,
            "film_m": h,
            "displacement_m": np.zeros_like(h),
        }
        rigid_load = float(trapezoid(trapezoid(rigid.pressure_pa, x, axis=1), y))
        original = next(c for c in source["cases"] if c["name"] == name)
        report["coupling_controls"].append(
            {
                "name": name,
                "rigid_load_n": rigid_load,
                "coupled_load_n": original["load_n"],
                "rigid_audit": rigid.audit,
                "rigid_accepted": accepted(rigid.audit, rigid_fields),
            }
        )
        states["rigid_" + name] = rigid_fields
        np.savez_compressed(directory / f"rigid_{name}.npz", **rigid_fields)
    d = analyze_pair(states["rigid_" + a], states["rigid_" + b])
    d.update(
        pair="rigid on coupled 8 -> 16",
        both_originally_accepted=all(c["rigid_accepted"] for c in report["coupling_controls"]),
    )
    report["load_budgets"].append(d)
    point_match, point_fields = matched_point_case(states["rigid_" + b])
    report["matched_point_reference"] = point_match
    np.savez_compressed(directory / "matched_point_reference.npz", **point_fields)
    checkpoint()
    for extent in (8, 16, 32, 64, 128, 256):
        ref = parabolic_jfo_reference(np.array([-extent * L, 0, 3 * L]), H0, 0.01, 0.04, 0.001)
        c = 0.001 * ref.film_thickness_m[0]
        p = -(ref.film_thickness_m[0] ** 3) / (12 * 0.04) * ref.pressure_gradient_pa_per_m[0]
        q = ref.volume_flux_per_width_m2_s
        report["analytical_inlet_flux"].append(
            {
                "inlet_over_L": -extent,
                "couette_per_width_m2_s": float(c),
                "poiseuille_per_width_m2_s": float(p),
                "exact_net_per_width_m2_s": float(q),
                "cancellation_condition": float((abs(c) + abs(p)) / q),
                "relative_float64_sum_error": float(abs(c + p - q) / q),
            }
        )
    checkpoint()
    wider = np.finfo(np.longdouble).nmant > np.finfo(float).nmant
    for name in ("all_128", "all_192", "all_256"):
        if not wider:
            report["refined_cases"].append(
                {"name": name, "status": "wider residual arithmetic unavailable"}
            )
            continue
        f = states[name]
        result = solve_rectilinear_cavitation(
            f["x_m"],
            f["y_m"],
            f["film_m"],
            0.04,
            0.001,
            length_scale_m=L,
            film_scale_m=H0,
            pressure_scale_pa=P_REF,
            flux_refinement_steps=3,
        )
        changed = f | {"pressure_pa": result.pressure_pa, "theta": result.fractional_film_content}
        old = next(c for c in source["cases"] if c["name"] == name)
        load = float(trapezoid(trapezoid(result.pressure_pa, f["x_m"], axis=1), f["y_m"]))
        report["refined_cases"].append(
            {
                "name": name,
                "before": old["audit"],
                "after": result.audit,
                "history": result.refinement_history,
                "accepted": accepted(result.audit, changed),
                "relative_load_change": load / old["load_n"] - 1,
                "state_dtype": str(result.pressure_pa.dtype),
            }
        )
        np.savez_compressed(directory / f"{name}_refined.npz", **changed)
        checkpoint()
    configurations = [
        (inlet, 129, mode) for inlet in (-0.5, -1, -2) for mode in ("matched", "finite_reservoir")
    ]
    configurations += [(-1, n, "matched") for n in (65, 257)]
    for inlet, n, mode in configurations:
        c, f = matched_case(inlet, n, mode=mode)
        c["state_file"] = f"{mode}_{inlet}_{n}.npz"
        report["matched_cases"].append(c)
        np.savez_compressed(directory / c["state_file"], **f)
        checkpoint()
        print(
            mode, inlet, n, c["relative_force_difference"], c["audit"]["mass_maximum"], flush=True
        )
    render(directory, report)
    report["status"] = "analysis completed; point-contact boundary/domain validation remains open"
    report["figures"] = [p.name for p in sorted(directory.glob("*.png"))]
    checkpoint()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", type=Path, default=Path("results/farfield_budget"))
    parser.add_argument("--output-directory", type=Path, default=Path("results/flux_conditioning"))
    args = parser.parse_args()
    main(args.input_directory, args.output_directory)
