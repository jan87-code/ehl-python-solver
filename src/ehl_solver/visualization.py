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
