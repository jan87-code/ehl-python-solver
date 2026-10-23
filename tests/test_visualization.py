from dataclasses import replace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from ehl_solver.coupled import CoupledIteration, CoupledSnapshot
from ehl_solver.visualization import (
    interpolate_cross_section,
    plot_convergence_history,
    plot_field_cross_section,
    plot_film_thickness_distribution,
    plot_pressure_distribution,
    plot_snapshot,
    save_coupled_figures,
)


def _sample_fields():
    x = np.linspace(-1e-3, 1e-3, 5)
    y = np.linspace(-2e-3, 2e-3, 5)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    pressure_pa = (2.0 * xx + 3.0 * yy) * 1e6
    film_m = (20.0 + xx * 1e6 + yy * 2e6) * 1e-6
    return x, y, pressure_pa, film_m


def _sample_solution(x, y, pressure, film):
    history = tuple(
        CoupledIteration(
            global_iteration=iteration,
            compliance_factor=1.0,
            iteration=iteration,
            pressure_relaxation=0.6,
            film_relaxation=0.5,
            pressure_update_relative=0.1 / iteration,
            film_update_relative=0.08 / iteration,
            reynolds_residual_rms=0.01 / iteration,
            reynolds_residual_relative=0.02 / iteration,
            minimum_film_thickness_m=float(np.min(film)),
            minimum_pressure_pa=float(np.min(pressure)),
            maximum_pressure_pa=float(np.max(pressure)),
            linear_iterations=10,
        )
        for iteration in range(1, 4)
    )
    snapshot = CoupledSnapshot(
        global_iteration=2,
        compliance_factor=0.5,
        iteration=1,
        pressure_pa=pressure.copy(),
        film_thickness_m=film.copy(),
    )
    return type(
        "SampleSolution",
        (),
        {
            "pressure_pa": pressure,
            "film_thickness_m": film,
            "history": history,
            "snapshots": (snapshot,),
        },
    )()


def test_cross_sections_interpolate_between_grid_nodes():
    x, y, pressure, _ = _sample_fields()

    x_coordinates, x_section = interpolate_cross_section(
        x, y, pressure, axis="x", coordinate_m=0.5e-3
    )
    y_coordinates, y_section = interpolate_cross_section(
        x, y, pressure, axis="y", coordinate_m=-0.25e-3
    )

    np.testing.assert_allclose(x_section, (2 * x_coordinates + 3 * 0.5e-3) * 1e6)
    np.testing.assert_allclose(y_section, (2 * -0.25e-3 + 3 * y_coordinates) * 1e6)


def test_cross_section_rejects_out_of_bounds_coordinates():
    x, y, pressure, _ = _sample_fields()

    with pytest.raises(ValueError, match="outside"):
        interpolate_cross_section(x, y, pressure, axis="x", coordinate_m=3e-3)


def test_pressure_map_keeps_negative_values_and_labels_colorbar():
    x, y, pressure, _ = _sample_fields()
    figure, axes = plot_pressure_distribution(x, y, pressure)

    assert axes.collections[0].levels[0] < 0
    assert axes.collections[0].levels[-1] > 0
    assert axes.collections[0].levels[-1] == pytest.approx(pressure.max() / 1e6)
    assert figure.axes[-1].get_ylabel() == "Pressure [MPa]"
    assert axes.get_xlabel().endswith("[mm]")
    assert axes.get_ylabel().endswith("[mm]")
    plt.close(figure)


def test_film_map_marks_minimum_at_correct_location_and_units():
    x, y, _, film = _sample_fields()
    figure, axes = plot_film_thickness_distribution(x, y, film)
    marker = axes.collections[-1].get_offsets()[0]
    minimum_j, minimum_i = np.unravel_index(np.argmin(film), film.shape)

    assert figure.axes[-1].get_ylabel() == "Film thickness [µm]"
    np.testing.assert_allclose(marker, [x[minimum_i] * 1e3, y[minimum_j] * 1e3])
    assert f"min={film.min() * 1e6:.4g}" in axes.texts[0].get_text()
    plt.close(figure)


def test_pressure_cross_section_has_zero_reference_line():
    x, y, pressure, _ = _sample_fields()
    figure, axes = plot_field_cross_section(
        x, y, pressure, axis="x", coordinate_m=0.0, quantity="pressure"
    )

    zero_line = next(line for line in axes.lines if line.get_label() == "Zero pressure")
    np.testing.assert_allclose(axes.lines[0].get_xdata(), x * 1e3)
    np.testing.assert_allclose(axes.lines[0].get_ydata(), pressure[2, :] / 1e6)
    np.testing.assert_array_equal(zero_line.get_ydata(), [0.0, 0.0])
    assert axes.get_ylabel() == "Pressure [MPa]"
    plt.close(figure)


def test_convergence_plot_contains_all_metrics_and_relaxation_histories():
    x, y, pressure, film = _sample_fields()
    solution = _sample_solution(x, y, pressure, film)
    history = (replace(solution.history[0], film_update_relative=0.0), *solution.history[1:])
    figure, axes = plot_convergence_history(history)

    assert len(axes) == 3
    assert len(axes[0].lines) == 3
    assert len(axes[1].lines) == 1
    assert len(axes[2].lines) == 3
    assert np.isnan(axes[0].lines[1].get_ydata()[0])
    assert axes[0].get_ylim()[0] > 1e-6
    assert axes[0].get_ylim()[1] < 1.0
    assert axes[2].get_xlabel() == "Accepted global outer iteration"
    plt.close(figure)


def test_snapshot_contains_both_fields_with_correct_units():
    x, y, pressure, film = _sample_fields()
    solution = _sample_solution(x, y, pressure, film)
    figure, axes = plot_snapshot(x, y, solution.snapshots[0])

    assert len(axes) == 2
    assert any(ax.get_ylabel() == "Pressure [MPa]" for ax in figure.axes)
    assert any(ax.get_ylabel() == "Film thickness [µm]" for ax in figure.axes)
    plt.close(figure)


def test_report_saves_all_requested_pngs_with_safe_case_directory(tmp_path):
    x, y, pressure, film = _sample_fields()
    solution = _sample_solution(x, y, pressure, film)

    paths = save_coupled_figures(
        x,
        y,
        solution,
        case_name="../demo case",
        results_directory=tmp_path,
    )

    assert len(paths) == 8
    assert all(path.is_file() and path.stat().st_size > 1000 for path in paths)
    assert paths[0].parent == tmp_path / "demo_case"
    assert any("snapshot_iter_0002" in path.name for path in paths)
