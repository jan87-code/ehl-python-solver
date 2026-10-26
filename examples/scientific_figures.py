"""Annotated, unsmoothed scientific figures from portable SI benchmark states."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ehl_solver.visualization import interpolate_cross_section


def extrema(fields):
    """Return sampled extrema and their SI coordinates; never alter fields."""
    output = {}
    for name, key, operation in (
        ("minimum_film", "film_m", np.argmin),
        ("maximum_pressure", "pressure_pa", np.argmax),
    ):
        values = fields[key]
        j, i = np.unravel_index(operation(values), values.shape)
        output[name] = {
            "value_si": float(values[j, i]),
            "x_m": float(fields["x_m"][i]),
            "y_m": float(fields["y_m"][j]),
        }
    return output


def caption(fields, parameters, audit, *, status="verified discrete root"):
    """Every physical figure carries parameters, domain, extrema and status."""
    x, y = fields["x_m"], fields["y_m"]
    bounds = extrema(fields)
    low, high = bounds["minimum_film"], bounds["maximum_pressure"]
    return (
        f"{status}; isoviscous/incompressible JFO; entrainment +x (inlet left, outlet right)\n"
        f"Q={audit['load_n']:.8g} N; Ue={parameters['Ue_m_s']:g} m/s; "
        f"eta={parameters['eta_pa_s']:g} Pa s; E*={parameters['E_star_pa']:.6g} Pa; "
        f"Rx,Ry=({parameters['Rx_m']:g},{parameters['Ry_m']:g}) m\n"
        f"(ny,nx)=({len(y)},{len(x)}); x=[{x[0] * 1e3:.6g},{x[-1] * 1e3:.6g}] mm; "
        f"y=[{y[0] * 1e3:.6g},{y[-1] * 1e3:.6g}] mm\n"
        f"hmin={low['value_si'] * 1e6:.9g} um at "
        f"({low['x_m'] * 1e3:.6g},{low['y_m'] * 1e3:.6g}) mm; "
        f"pmax={high['value_si'] / 1e6:.9g} MPa at "
        f"({high['x_m'] * 1e3:.6g},{high['y_m'] * 1e3:.6g}) mm\n"
        f"mass max={audit['mass_maximum']:.3e}; net flux={audit['net_boundary_flux']:.3e}; "
        f"P(1-theta) max={audit['maximum_complementarity']:.3e}; "
        f"pmin={fields['pressure_pa'].min():.3e} Pa\n"
        f"force error={audit['load_n'] - parameters['target_load_n']:.3e} N; "
        f"film coupling error={audit['film_coupling_max_error_m']:.3e} m; "
        "finite-domain numerical verification; physical film-height validation open"
    )


def figure(title, context, *, rows=1):
    fig, axes = plt.subplots(
        rows,
        1,
        figsize=(11, 6 + 3 * (rows - 1) + 0.16 * ((title + context).count("\n") + 1)),
        constrained_layout=True,
        squeeze=False,
    )
    fig.suptitle(title + "\n" + context, fontsize=9)
    return fig, axes[:, 0]


def save(fig, directory, name):
    fig.savefig(directory / f"{name}.png", dpi=140)
    plt.close(fig)


def field_figures(directory, name, fields, parameters, audit):
    context = caption(fields, parameters, audit)
    x, y = fields["x_m"], fields["y_m"]
    specifications = (
        ("pressure", fields["pressure_pa"] / 1e6, "Pressure [MPa]"),
        ("film", fields["film_m"] * 1e6, "Film thickness [um]"),
        ("displacement", fields["displacement_m"] * 1e6, "Elastic gap increase [um]"),
        ("saturation", fields["theta"], "Fractional film content theta"),
    )
    for label, values, units in specifications:
        fig, (axis,) = figure(label.capitalize() + " contour", context)
        minimum, maximum = float(values.min()), float(values.max())
        if minimum == maximum:
            span = max(abs(minimum), 1.0) * np.finfo(float).eps * 128
            minimum, maximum = minimum - span, maximum + span
        artist = axis.contourf(x * 1e3, y * 1e3, values, levels=np.linspace(minimum, maximum, 41))
        fig.colorbar(artist, ax=axis, label=units)
        axis.set(xlabel="x [mm], inlet to outlet (+x)", ylabel="Transverse y [mm]")
        point = extrema(fields)["maximum_pressure" if label == "pressure" else "minimum_film"]
        marker_label = (
            "Sampled pressure maximum" if label == "pressure" else "Minimum-film location"
        )
        axis.plot(point["x_m"] * 1e3, point["y_m"] * 1e3, "rx", label=marker_label)
        axis.legend()
        save(fig, directory, f"{name}_{label}")
    for label, key, orientation, factor, units in (
        ("pressure_x", "pressure_pa", "x", 1e-6, "Pressure [MPa]"),
        ("film_x", "film_m", "x", 1e6, "Film thickness [um]"),
        ("film_y", "film_m", "y", 1e6, "Film thickness [um]"),
    ):
        coordinates, values = interpolate_cross_section(
            x, y, fields[key], axis=orientation, coordinate_m=0
        )
        fixed = "y" if orientation == "x" else "x"
        fig, (axis,) = figure(f"Centerline {label}; {fixed}=0 (linear interpolation)", context)
        axis.plot(coordinates * 1e3, values * factor, "o-", markersize=3)
        axis.set(xlabel=f"{orientation} [mm]", ylabel=units)
        axis.grid(alpha=0.3)
        save(fig, directory, f"{name}_{label}")
    fig, (axis,) = figure("Cavitation active region; theta < 1-1e-8", context)
    region = fields["theta"] < 1 - 1e-8
    artist = axis.pcolormesh(x * 1e3, y * 1e3, region, shading="nearest", vmin=0, vmax=1)
    fig.colorbar(artist, ax=axis, ticks=[0, 1], label="Reduced-content node (1=yes)")
    axis.set(xlabel="x [mm], inlet to outlet (+x)", ylabel="Transverse y [mm]")
    save(fig, directory, f"{name}_active_region")


def newton_figures(directory, name, fields, parameters, audit, history):
    context = caption(fields, parameters, audit)
    fig, axes = figure("Newton conservation, load and active-set diagnostics", context, rows=3)
    mass = [row["mass_maximum"] for row in history] + [audit["mass_maximum"]]
    force = [row["load_residual_n"] / parameters["target_load_n"] for row in history] + [
        (audit["load_n"] - parameters["target_load_n"]) / parameters["target_load_n"]
    ]
    axes[0].semilogy(range(len(mass)), mass, "o-")
    axes[0].axhline(1e-8, linestyle=":", color="grey", label="Mass tolerance")
    axes[0].set_ylabel("Original maximum mass residual")
    axes[0].legend()
    axes[1].plot(range(len(force)), force, "o-")
    axes[1].set_yscale("symlog", linthresh=1e-16)
    axes[1].set_ylabel("Signed force error / target (zero retained)")
    axes[2].plot([row["full_film_cells"] for row in history], "o-", label="Full-film derivative")
    axes[2].set(
        xlabel="Newton correction (final audit appended above)",
        ylabel="Interior full-film active cells",
    )
    for axis in axes:
        axis.grid(alpha=0.3)
    save(fig, directory, f"{name}_newton")
    fig, (axis,) = figure("Krylov convergence and independently measured true residuals", context)
    work = 0
    for row in history:
        for trace, measured in zip(
            row["krylov_histories"], row["linear_true_residuals"], strict=True
        ):
            axis.semilogy(range(work + 1, work + len(trace) + 1), trace, color="tab:blue")
            work += len(trace)
            axis.semilogy([work], [measured], "rx")
    axis.set(
        xlabel="Cumulative Krylov iteration",
        ylabel="Preconditioned history (blue) / true residual (red)",
    )
    save(fig, directory, f"{name}_krylov")


def grid_figure(directory, cases, fields, parameters):
    context = caption(fields, parameters, cases[-1]["audit"])
    details = "\n".join(
        f"{case['shape']}: h0={case['h0_m'] * 1e6:.8g} um, "
        f"hmin={case['audit']['minimum_film_m'] * 1e6:.8g} um, "
        f"pmax={case['peak_pressure_pa'] / 1e6:.6g} MPa, "
        f"mass={case['audit']['mass_maximum']:.2e}\n"
        f"hmin at ({case['sampled_extrema']['minimum_film']['x_m'] * 1e3:.6g},"
        f"{case['sampled_extrema']['minimum_film']['y_m'] * 1e3:.6g}) mm; "
        f"pmax at ({case['sampled_extrema']['maximum_pressure']['x_m'] * 1e3:.6g},"
        f"{case['sampled_extrema']['maximum_pressure']['y_m'] * 1e3:.6g}) mm; "
        f"force error={case['audit']['load_n'] - parameters['target_load_n']:.2e} N"
        for case in cases
    )
    fig, (axis,) = figure("Same-domain grid refinement; finest-state context\n" + details, context)
    axis.plot(
        [np.prod(case["shape"]) for case in cases], [case["h0_m"] * 1e6 for case in cases], "o-"
    )
    axis.set(xlabel="Total grid nodes (ny*nx)", ylabel="Rigid-gap offset h0 [um]")
    axis.grid(alpha=0.3)
    save(fig, directory, "grid_refinement")


def domain_figure(directory, cases, fields, parameters):
    context = caption(fields, parameters, cases[0]["final_audit"])
    details = "\n".join(
        f"{case['name']} {case['shape']}: x={np.array(case['x_limits_m']) * 1e3} mm, "
        f"y={np.array(case['y_limits_m']) * 1e3} mm, "
        f"mass={case['final_audit']['mass_maximum']:.2e}\n"
        f"hmin={case['sampled_extrema']['minimum_film']['value_si'] * 1e6:.6g} um at "
        f"({case['sampled_extrema']['minimum_film']['x_m'] * 1e3:.6g},"
        f"{case['sampled_extrema']['minimum_film']['y_m'] * 1e3:.6g}) mm; "
        f"pmax={case['peak_pressure_pa'] / 1e6:.6g} MPa at "
        f"({case['sampled_extrema']['maximum_pressure']['x_m'] * 1e3:.6g},"
        f"{case['sampled_extrema']['maximum_pressure']['y_m'] * 1e3:.6g}) mm"
        for case in cases
    )
    fig, (axis,) = figure(
        "Boundary sensitivity at fixed force; baseline-state context\n" + details, context
    )
    axis.bar([case["name"] for case in cases], [case["final_h0_m"] * 1e6 for case in cases])
    axis.tick_params(axis="x", rotation=45)
    axis.set(ylabel="Rigid-gap offset h0 [um]", xlabel="Extended boundary, original-cell count")
    save(fig, directory, "domain_sensitivity")
