"""Matplotlib plots for coupled Reynolds-elasticity solutions and histories."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import RegularGridInterpolator

if TYPE_CHECKING:
    from ehl_solver.coupled import CoupledReynoldsSolution, CoupledSnapshot
    from ehl_solver.load_control import FullFilmLoadSolution
    from ehl_solver.monolithic import (
        MonolithicIteration,
        MonolithicLoadContinuation,
        MonolithicLoadSolution,
        PreconditionerBenchmark,
    )


def _validated_plot_fields(
    x_m: ArrayLike,
    y_m: ArrayLike,
    pressure_pa: ArrayLike,
    film_thickness_m: ArrayLike,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    x = np.asarray(x_m, dtype=np.float64)
    y = np.asarray(y_m, dtype=np.float64)
    pressure = np.asarray(pressure_pa, dtype=np.float64)
    film = np.asarray(film_thickness_m, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or min(x.size, y.size) < 2:
        raise ValueError("x_m and y_m must be 1-D coordinate vectors")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("coordinates must contain only finite values")
    if np.any(np.diff(x) <= 0) or np.any(np.diff(y) <= 0):
        raise ValueError("coordinates must be strictly ascending")
    if pressure.shape != (y.size, x.size) or film.shape != pressure.shape:
        raise ValueError("pressure and film arrays must have shape (ny, nx)")
    if not np.all(np.isfinite(pressure)) or not np.all(np.isfinite(film)):
        raise ValueError("pressure and film must contain only finite values")
    return x, y, pressure, film


def interpolate_cross_section(
    x_m: ArrayLike,
    y_m: ArrayLike,
    field: ArrayLike,
    *,
    axis: str,
    coordinate_m: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Interpolate an x- or y-directed section through a `(ny, nx)` field.

    ``axis="x"`` returns values along x at the requested transverse y
    coordinate; ``axis="y"`` returns values along y at the requested x.
    Linear interpolation is performed between grid nodes as needed. Requests
    outside the coordinate domain raise ``ValueError`` rather than extrapolate.
    """
    x = np.asarray(x_m, dtype=np.float64)
    y = np.asarray(y_m, dtype=np.float64)
    values = np.asarray(field, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or values.shape != (y.size, x.size):
        raise ValueError("field must have shape (ny, nx) matching x_m and y_m")
    if axis not in ("x", "y"):
        raise ValueError("axis must be 'x' or 'y'")
    if not np.isfinite(coordinate_m):
        raise ValueError("coordinate_m must be finite")
    fixed_coordinates = y if axis == "x" else x
    if coordinate_m < fixed_coordinates[0] or coordinate_m > fixed_coordinates[-1]:
        raise ValueError("requested cross-section is outside the coordinate domain")
    interpolator = RegularGridInterpolator(
        (y, x), values, method="linear", bounds_error=True
    )
    if axis == "x":
        points = np.column_stack((np.full_like(x, coordinate_m), x))
        coordinates = x
    else:
        points = np.column_stack((y, np.full_like(y, coordinate_m)))
        coordinates = y
    return coordinates.copy(), np.asarray(interpolator(points), dtype=np.float64)


def plot_pressure_distribution(
    x_m: ArrayLike, y_m: ArrayLike, pressure_pa: ArrayLike
) -> tuple[Figure, Axes]:
    """Plot signed pressure in MPa with an explicitly labeled colorbar."""
    x = np.asarray(x_m, dtype=np.float64)
    y = np.asarray(y_m, dtype=np.float64)
    pressure = np.asarray(pressure_pa, dtype=np.float64)
    if pressure.shape != (y.size, x.size):
        raise ValueError("pressure_pa must have shape (ny, nx)")
    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    contour = ax.contourf(x * 1e3, y * 1e3, pressure / 1e6, levels=101, cmap="coolwarm")
    colorbar = fig.colorbar(contour, ax=ax)
    colorbar.set_label("Pressure [MPa]")
    ax.set_xlabel("Entrainment coordinate x [mm]")
    ax.set_ylabel("Transverse coordinate y [mm]")
    ax.set_title("Full-film pressure distribution")
    return fig, ax


def plot_film_thickness_distribution(
    x_m: ArrayLike, y_m: ArrayLike, film_thickness_m: ArrayLike
) -> tuple[Figure, Axes]:
    """Plot film thickness in micrometres and mark its minimum location."""
    x = np.asarray(x_m, dtype=np.float64)
    y = np.asarray(y_m, dtype=np.float64)
    film = np.asarray(film_thickness_m, dtype=np.float64)
    if film.shape != (y.size, x.size):
        raise ValueError("film_thickness_m must have shape (ny, nx)")
    minimum_j, minimum_i = np.unravel_index(np.argmin(film), film.shape)
    film_um = film * 1e6
    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    contour = ax.contourf(x * 1e3, y * 1e3, film_um, levels=101, cmap="viridis")
    colorbar = fig.colorbar(contour, ax=ax)
    colorbar.set_label("Film thickness [µm]")
    minimum_x_mm = x[minimum_i] * 1e3
    minimum_y_mm = y[minimum_j] * 1e3
    ax.scatter([minimum_x_mm], [minimum_y_mm], color="red", marker="x", s=70)
    ax.annotate(
        f"min={film_um[minimum_j, minimum_i]:.4g} µm\n"
        f"(x,y)=({minimum_x_mm:.4g}, {minimum_y_mm:.4g}) mm",
        (minimum_x_mm, minimum_y_mm),
        xytext=(8, 8),
        textcoords="offset points",
        color="red",
    )
    ax.set_xlabel("Entrainment coordinate x [mm]")
    ax.set_ylabel("Transverse coordinate y [mm]")
    ax.set_title("Film-thickness distribution")
    return fig, ax


def plot_field_cross_section(
    x_m: ArrayLike,
    y_m: ArrayLike,
    field: ArrayLike,
    *,
    axis: str,
    coordinate_m: float,
    quantity: str,
) -> tuple[Figure, Axes]:
    """Plot an interpolated pressure or film cross section with SI-based axes."""
    coordinates, section = interpolate_cross_section(
        x_m, y_m, field, axis=axis, coordinate_m=coordinate_m
    )
    if quantity == "pressure":
        plotted_values = section / 1e6
        y_label = "Pressure [MPa]"
        title_quantity = "pressure"
    elif quantity == "film":
        plotted_values = section * 1e6
        y_label = "Film thickness [µm]"
        title_quantity = "film thickness"
    else:
        raise ValueError("quantity must be 'pressure' or 'film'")
    coordinate_name = "y" if axis == "x" else "x"
    coordinate_label = (
        "Entrainment coordinate x [mm]"
        if axis == "x"
        else "Transverse coordinate y [mm]"
    )
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.plot(coordinates * 1e3, plotted_values, linewidth=1.8)
    if quantity == "pressure":
        ax.axhline(0.0, color="black", linewidth=0.9, linestyle="--", label="Zero pressure")
        ax.legend()
    ax.set_xlabel(coordinate_label)
    ax.set_ylabel(y_label)
    ax.set_title(
        f"{title_quantity.capitalize()} section at "
        f"{coordinate_name}={coordinate_m * 1e3:.4g} mm"
    )
    return fig, ax


def plot_convergence_history(
    history: tuple[Any, ...],
) -> tuple[Figure, NDArray[Any]]:
    """Plot nonlinear updates, Reynolds residual, extrema and relaxations."""
    if not history:
        raise ValueError("history must contain at least one accepted iteration")
    iterations = np.array([entry.global_iteration for entry in history])

    def positive_or_gap(values: ArrayLike) -> NDArray[np.float64]:
        data = np.asarray(values, dtype=np.float64)
        return np.where(data > 0.0, data, np.nan)

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True, constrained_layout=True)
    axes[0].semilogy(
        iterations,
        positive_or_gap([entry.pressure_update_relative for entry in history]),
        label="Relative pressure update",
    )
    axes[0].semilogy(
        iterations,
        positive_or_gap([entry.film_update_relative for entry in history]),
        label="Relative film update",
    )
    axes[0].semilogy(
        iterations,
        positive_or_gap([entry.reynolds_residual_relative for entry in history]),
        label="Relative Reynolds residual",
    )
    axes[0].set_ylabel("Normalized residual")
    axes[0].legend()
    axes[0].grid(True, which="both", alpha=0.25)

    axes[1].plot(
        iterations,
        [entry.minimum_film_thickness_m * 1e6 for entry in history],
        color="tab:green",
        label="Minimum film",
    )
    axes[1].set_ylabel("Minimum film [µm]", color="tab:green")
    pressure_axis = axes[1].twinx()
    pressure_axis.plot(
        iterations,
        [entry.maximum_pressure_pa / 1e6 for entry in history],
        color="tab:red",
        label="Maximum pressure",
    )
    pressure_axis.set_ylabel("Maximum pressure [MPa]", color="tab:red")
    axes[1].grid(True, alpha=0.25)

    axes[2].step(
        iterations,
        [entry.pressure_relaxation for entry in history],
        where="post",
        label="Pressure relaxation",
    )
    axes[2].step(
        iterations,
        [entry.film_relaxation for entry in history],
        where="post",
        label="Film relaxation",
    )
    axes[2].step(
        iterations,
        [entry.compliance_factor for entry in history],
        where="post",
        label="Compliance factor",
    )
    axes[2].set_xlabel("Accepted global outer iteration")
    axes[2].set_ylabel("Relaxation / continuation factor")
    axes[2].set_ylim(bottom=0)
    axes[2].legend()
    axes[2].grid(True, alpha=0.25)
    return fig, axes


def plot_load_control_history(
    solution: FullFilmLoadSolution,
) -> tuple[Figure, NDArray[Any]]:
    """Plot converged signed load versus h0 and relative error per evaluation."""
    if not solution.evaluations:
        raise ValueError("load solution must contain at least one evaluation")
    evaluations_by_offset = sorted(
        solution.evaluations, key=lambda evaluation: evaluation.central_film_offset_m
    )
    evaluation_indices = np.arange(1, len(solution.evaluations) + 1)
    relative_errors = np.asarray(
        [evaluation.relative_load_error for evaluation in solution.evaluations],
        dtype=np.float64,
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].plot(
        [item.central_film_offset_m * 1e6 for item in evaluations_by_offset],
        [item.calculated_signed_load_n for item in evaluations_by_offset],
        "o-",
        label="Converged signed load",
    )
    axes[0].axhline(
        solution.target_load_n, color="black", linestyle="--", label="Target signed load"
    )
    axes[0].axvline(
        solution.central_film_offset_m * 1e6,
        color="tab:red",
        linestyle=":",
        label="Root h0",
    )
    axes[0].set_xlabel("Central film offset h0 [µm]")
    axes[0].set_ylabel("Signed pressure integral [N]")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    positive_errors = np.where(relative_errors > 0.0, relative_errors, np.nan)
    axes[1].semilogy(evaluation_indices, positive_errors, "o-")
    axes[1].set_xlabel("Converged load-function evaluation")
    axes[1].set_ylabel("Relative signed-load error")
    axes[1].grid(True, which="both", alpha=0.25)
    fig.suptitle("Mathematical full-film load-control history")
    return fig, axes


def plot_snapshot(
    x_m: ArrayLike, y_m: ArrayLike, snapshot: CoupledSnapshot
) -> tuple[Figure, NDArray[Any]]:
    """Plot both signed pressure and film thickness for an accepted snapshot."""
    x = np.asarray(x_m, dtype=np.float64)
    y = np.asarray(y_m, dtype=np.float64)
    pressure = np.asarray(snapshot.pressure_pa, dtype=np.float64)
    film = np.asarray(snapshot.film_thickness_m, dtype=np.float64)
    if pressure.shape != (y.size, x.size) or film.shape != pressure.shape:
        raise ValueError("snapshot arrays must have shape (ny, nx)")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    pressure_plot = axes[0].contourf(
        x * 1e3, y * 1e3, pressure / 1e6, levels=81, cmap="coolwarm"
    )
    fig.colorbar(pressure_plot, ax=axes[0], label="Pressure [MPa]")
    axes[0].set_title("Pressure [MPa]")
    film_plot = axes[1].contourf(
        x * 1e3, y * 1e3, film * 1e6, levels=81, cmap="viridis"
    )
    fig.colorbar(film_plot, ax=axes[1], label="Film thickness [µm]")
    axes[1].set_title("Film thickness [µm]")
    for ax in axes:
        ax.set_xlabel("Entrainment coordinate x [mm]")
        ax.set_ylabel("Transverse coordinate y [mm]")
    fig.suptitle(
        f"Accepted iteration {snapshot.global_iteration}, "
        f"compliance={snapshot.compliance_factor:.2f}"
    )
    return fig, axes


def save_coupled_figures(
    x_m: ArrayLike,
    y_m: ArrayLike,
    solution: CoupledReynoldsSolution,
    *,
    case_name: str,
    results_directory: str | Path = "results",
    transverse_y_m: float = 0.0,
    entrainment_x_m: float = 0.0,
) -> tuple[Path, ...]:
    """Save maps, central/transverse sections, history and snapshots as PNGs.

    Figures are written beneath ``results_directory/case_name`` with
    descriptive filenames. Central sections use y=0 and x=0 by default;
    arbitrary requested sections are linearly interpolated between grid
    nodes. The output directory is created if needed. Figures are closed
    after writing.
    """
    x, y, pressure, film = _validated_plot_fields(
        x_m, y_m, solution.pressure_pa, solution.film_thickness_m
    )
    case = re.sub(r"[^A-Za-z0-9_.-]+", "_", case_name).strip("._")
    if not case:
        raise ValueError("case_name must contain at least one safe filename character")
    output = Path(results_directory) / case
    output.mkdir(parents=True, exist_ok=True)
    figures: list[tuple[str, Figure]] = [
        ("pressure_distribution.png", plot_pressure_distribution(x, y, pressure)[0]),
        (
            "film_thickness_distribution.png",
            plot_film_thickness_distribution(x, y, film)[0],
        ),
        (
            "pressure_cross_section_x.png",
            plot_field_cross_section(
                x, y, pressure, axis="x", coordinate_m=transverse_y_m, quantity="pressure"
            )[0],
        ),
        (
            "film_cross_section_x.png",
            plot_field_cross_section(
                x, y, film, axis="x", coordinate_m=transverse_y_m, quantity="film"
            )[0],
        ),
        (
            "pressure_cross_section_y.png",
            plot_field_cross_section(
                x, y, pressure, axis="y", coordinate_m=entrainment_x_m, quantity="pressure"
            )[0],
        ),
        (
            "film_cross_section_y.png",
            plot_field_cross_section(
                x, y, film, axis="y", coordinate_m=entrainment_x_m, quantity="film"
            )[0],
        ),
        ("convergence_history.png", plot_convergence_history(solution.history)[0]),
    ]
    for snapshot in solution.snapshots:
        figures.append(
            (
                f"snapshot_iter_{snapshot.global_iteration:04d}_"
                f"gamma_{snapshot.compliance_factor:.2f}.png",
                plot_snapshot(x, y, snapshot)[0],
            )
        )
    paths: list[Path] = []
    for filename, figure in figures:
        path = output / filename
        figure.savefig(path, dpi=160)
        plt.close(figure)
        paths.append(path)
    return tuple(paths)


def plot_monolithic_newton_history(
    history: tuple[MonolithicIteration, ...],
    *,
    failure_diagnostics: dict[str, object] | None = None,
) -> tuple[Figure, NDArray[Any]]:
    """Plot Newton residuals, globalization, Krylov work and Schur diagnostics."""
    if not history:
        raise ValueError("history must contain at least one Newton iteration")
    iterations = np.asarray([entry.iteration for entry in history])
    fig, axes = plt.subplots(5, 1, figsize=(10, 15), constrained_layout=True)
    axes[0].semilogy(
        iterations,
        [
            max(abs(entry.reynolds_residual_scaled), 1e-16)
            for entry in history
        ],
        "o-",
        label="Scaled Reynolds residual",
    )
    axes[0].semilogy(
        iterations,
        [max(abs(entry.load_residual_scaled), 1e-16) for entry in history],
        "s-",
        label="Scaled load residual",
    )
    axes[0].set_ylabel("Scaled residual")
    axes[0].legend()
    axes[0].grid(True, which="both", alpha=0.25)
    load_axis = axes[0].twinx()
    load_axis.semilogy(
        iterations,
        [max(abs(entry.load_residual_n), 1e-30) for entry in history],
        "^:",
        color="tab:purple",
        label="Load residual [N]",
    )
    load_axis.set_ylabel("Absolute load residual [N]")

    axes[1].plot(
        iterations,
        [entry.feasible_step_limit for entry in history],
        "o-",
        label="Feasible step limit",
    )
    axes[1].plot(
        iterations,
        [entry.accepted_step_length for entry in history],
        "s-",
        label="Accepted step length",
    )
    axes[1].set_ylabel("Newton step length")
    axes[1].set_ylim(bottom=0.0)
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)

    axes[2].step(
        iterations,
        [entry.krylov_iterations for entry in history],
        where="mid",
        label="Krylov iterations",
    )
    axes[2].set_ylabel("Krylov iterations")
    schur_axis = axes[2].twinx()
    schur_values = np.asarray(
        [entry.schur_condition_indicator for entry in history], dtype=np.float64
    )
    schur_axis.semilogy(
        iterations,
        np.where(schur_values > 0.0, schur_values, np.nan),
        "d--",
        color="tab:orange",
        label="Schur scale indicator",
    )
    schur_axis.set_ylabel("Schur scale indicator")
    axes[2].grid(True, alpha=0.25)

    axes[3].plot(
        iterations,
        [entry.minimum_film_thickness_m * 1e6 for entry in history],
        "o-",
        label="Minimum film [µm]",
    )
    pressure_axis = axes[3].twinx()
    pressure_axis.plot(
        iterations,
        [entry.minimum_pressure_pa / 1e6 for entry in history],
        "^-.",
        color="tab:purple",
        label="Minimum pressure",
    )
    pressure_axis.plot(
        iterations,
        [entry.maximum_pressure_pa / 1e6 for entry in history],
        "s--",
        color="tab:red",
        label="Maximum pressure [MPa]",
    )
    axes[3].set_ylabel("Minimum film [µm]")
    pressure_axis.set_ylabel("Pressure extrema [MPa]")
    film_handles, film_labels = axes[3].get_legend_handles_labels()
    pressure_handles, pressure_labels = pressure_axis.get_legend_handles_labels()
    axes[3].legend(
        film_handles + pressure_handles,
        film_labels + pressure_labels,
        loc="best",
        fontsize="small",
    )
    axes[3].set_xlabel("Newton iteration")
    axes[3].grid(True, alpha=0.25)
    axes[4].set_ylabel("Preconditioned GMRES residual")
    for entry in history:
        for diagnostic in entry.krylov_diagnostics:
            residual_history = np.asarray(
                diagnostic.preconditioned_residual_history, dtype=np.float64
            )
            if residual_history.size:
                valid_history = np.maximum(residual_history, np.finfo(float).tiny)
                label = f"{entry.iteration}: {diagnostic.label}"
                axes[4].semilogy(
                    np.arange(1, residual_history.size + 1),
                    valid_history,
                    label=label,
                    alpha=0.8,
                )
                axes[4].scatter(
                    [residual_history.size],
                    [max(diagnostic.true_relative_residual, np.finfo(float).tiny)],
                    marker="x",
                    color=axes[4].lines[-1].get_color(),
                    label=(
                        "True relative residual"
                        if not any(
                            collection.get_label() == "True relative residual"
                            for collection in axes[4].collections
                        )
                        else None
                    ),
                )
    axes[4].set_xlabel("Krylov iteration within each solve")
    axes[4].grid(True, which="both", alpha=0.25)
    if any(entry.krylov_diagnostics for entry in history):
        axes[4].legend(fontsize="x-small", ncol=2)
    fig.suptitle("Monolithic Newton convergence and linear-solve diagnostics")
    if failure_diagnostics:
        diagnostic_text = "\n".join(
            f"{key}: {value}"
            for key, value in failure_diagnostics.items()
        )
        fig.suptitle(
            "Monolithic Newton history (failed: "
            f"{failure_diagnostics.get('iteration', 'unknown')} iteration diagnostic)"
        )
        axes[0].text(
            0.01,
            0.02,
            diagnostic_text,
            transform=axes[0].transAxes,
            va="bottom",
            fontsize="small",
        )
    else:
        fig.suptitle("Monolithic Newton convergence history")
    return fig, axes


def plot_preconditioner_comparison(
    results: tuple[PreconditionerBenchmark, ...],
) -> tuple[Figure, NDArray[Any]]:
    """Compare bordered-solve timings, Krylov work and verified residuals."""
    if not results:
        raise ValueError("results must contain at least one preconditioner benchmark")
    labels = [item.preconditioner for item in results]
    positions = np.arange(len(results))
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    axes[0].bar(positions, [item.elapsed_seconds for item in results])
    axes[0].set_ylabel("Elapsed time [s]")
    axes[1].bar(positions, [item.krylov_iterations for item in results])
    axes[1].set_ylabel("Total Krylov iterations")
    axes[2].bar(
        positions,
        [
            max(item.maximum_true_relative_residual, np.finfo(float).tiny)
            for item in results
        ],
    )
    axes[2].set_yscale("log")
    axes[2].set_ylabel("Maximum independently evaluated true residual")
    for axis in axes:
        axis.set_xticks(positions, labels)
        axis.grid(True, axis="y", alpha=0.25)
    figure.suptitle("Frozen-film preconditioner comparison")
    return figure, axes


def plot_elastic_coupling_diagnostic(
    effective_modulus_pa: ArrayLike,
    deformation_to_rigid_gap_ratio: ArrayLike,
    *,
    converged: ArrayLike | None = None,
) -> tuple[Figure, Axes]:
    """Compare compliance strength using peak elastic gap change / rigid gap."""
    modulus = np.asarray(effective_modulus_pa, dtype=np.float64)
    ratio = np.asarray(deformation_to_rigid_gap_ratio, dtype=np.float64)
    if (
        modulus.ndim != 1
        or ratio.shape != modulus.shape
        or modulus.size == 0
        or np.any(~np.isfinite(modulus))
        or np.any(modulus <= 0.0)
        or np.any(~np.isfinite(ratio))
        or np.any(ratio < 0.0)
    ):
        raise ValueError("moduli and deformation ratios must be finite positive vectors")
    success = (
        np.ones(modulus.shape, dtype=bool)
        if converged is None
        else np.asarray(converged, dtype=bool)
    )
    if success.shape != modulus.shape:
        raise ValueError("converged must match the modulus vector shape")
    order = np.argsort(modulus)
    figure, axes = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    axes.plot(
        modulus[order][success[order]] / 1e6,
        ratio[order][success[order]],
        "o-",
        color="tab:green",
        label="Converged",
    )
    axes.scatter(
        modulus[order][~success[order]] / 1e6,
        ratio[order][~success[order]],
        marker="x",
        color="tab:red",
        label="Failed iterate",
        zorder=3,
    )
    axes.set_xscale("log")
    axes.set_xlabel("Effective elastic modulus E* [MPa]")
    axes.set_ylabel("Peak |elastic gap change| / minimum rigid gap")
    axes.grid(True, which="both", alpha=0.25)
    axes.legend()
    axes.set_title("Measured Reynolds-elasticity coupling strength")
    return figure, axes


def plot_monolithic_continuation_history(
    continuation: MonolithicLoadContinuation,
) -> tuple[Figure, NDArray[Any]]:
    """Plot attempted load steps and accepted h0/minimum-film branches."""
    steps = tuple(continuation.steps)
    if not steps:
        raise ValueError("continuation must contain at least one attempted load step")
    attempts = np.asarray([step.attempt for step in steps])
    loads = np.asarray([step.target_load_n for step in steps])
    accepted = np.asarray([step.accepted for step in steps], dtype=bool)
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), constrained_layout=True)
    axes[0].plot(
        attempts[accepted],
        loads[accepted],
        "o-",
        color="tab:green",
        label="Accepted load",
    )
    axes[0].scatter(
        attempts[~accepted],
        loads[~accepted],
        marker="x",
        color="tab:red",
        label="Rejected load",
        zorder=3,
    )
    axes[0].set_xlabel("Continuation attempt")
    axes[0].set_ylabel("Target signed load [N]")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(
        loads[accepted],
        [step.central_film_offset_m * 1e6 for step in steps if step.accepted],
        "o-",
        color="tab:blue",
    )
    axes[1].set_xlabel("Target signed load [N]")
    axes[1].set_ylabel("Central film offset h0 [µm]")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(
        loads[accepted],
        [step.minimum_film_thickness_m * 1e6 for step in steps if step.accepted],
        "o-",
        color="tab:purple",
    )
    axes[2].set_xlabel("Target signed load [N]")
    axes[2].set_ylabel("Minimum film thickness [µm]")
    axes[2].grid(True, alpha=0.25)
    fig.suptitle("Adaptive signed-load continuation history")
    return fig, axes


def save_monolithic_figures(
    x_m: ArrayLike,
    y_m: ArrayLike,
    solution: MonolithicLoadSolution,
    *,
    case_name: str,
    results_directory: str | Path = "results",
    continuation: MonolithicLoadContinuation | None = None,
    preconditioner_comparison: tuple[PreconditionerBenchmark, ...] | None = None,
    effective_modulus_pa: ArrayLike | None = None,
    deformation_to_rigid_gap_ratio: ArrayLike | None = None,
    coupling_converged: ArrayLike | None = None,
) -> tuple[Path, ...]:
    """Save final fields, center sections, Newton and optional load histories."""
    x, y, pressure, film = _validated_plot_fields(
        x_m, y_m, solution.pressure_pa, solution.film_thickness_m
    )
    case = re.sub(r"[^A-Za-z0-9_.-]+", "_", case_name).strip("._")
    if not case:
        raise ValueError("case_name must contain at least one safe filename character")
    output = Path(results_directory) / case
    output.mkdir(parents=True, exist_ok=True)
    figures: list[tuple[str, Figure]] = [
        ("pressure_distribution.png", plot_pressure_distribution(x, y, pressure)[0]),
        (
            "film_thickness_distribution.png",
            plot_film_thickness_distribution(x, y, film)[0],
        ),
        (
            "pressure_centerline.png",
            plot_field_cross_section(
                x, y, pressure, axis="x", coordinate_m=0.0, quantity="pressure"
            )[0],
        ),
        (
            "film_centerline.png",
            plot_field_cross_section(
                x, y, film, axis="x", coordinate_m=0.0, quantity="film"
            )[0],
        ),
        ("newton_history.png", plot_monolithic_newton_history(solution.history)[0]),
    ]
    if continuation is not None:
        figures.append(
            (
                "load_continuation_history.png",
                plot_monolithic_continuation_history(continuation)[0],
            )
        )
    if preconditioner_comparison is not None:
        figures.append(
            (
                "preconditioner_comparison.png",
                plot_preconditioner_comparison(preconditioner_comparison)[0],
            )
        )
    if (effective_modulus_pa is None) != (deformation_to_rigid_gap_ratio is None):
        raise ValueError(
            "effective_modulus_pa and deformation_to_rigid_gap_ratio must be "
            "provided together"
        )
    if effective_modulus_pa is not None and deformation_to_rigid_gap_ratio is not None:
        figures.append(
            (
                "elastic_coupling_diagnostic.png",
                plot_elastic_coupling_diagnostic(
                    effective_modulus_pa,
                    deformation_to_rigid_gap_ratio,
                    converged=coupling_converged,
                )[0],
            )
        )
    paths: list[Path] = []
    for filename, figure in figures:
        path = output / filename
        figure.savefig(path, dpi=160)
        plt.close(figure)
        paths.append(path)
    return tuple(paths)
