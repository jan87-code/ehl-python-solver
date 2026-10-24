from types import SimpleNamespace

import matplotlib
import numpy as np

matplotlib.use("Agg")

from ehl_solver.visualization import (
    plot_monolithic_continuation_history,
    plot_monolithic_newton_history,
    save_monolithic_figures,
)


def _history_entry(iteration=1):
    return SimpleNamespace(
        iteration=iteration,
        reynolds_residual_scaled=2e-3,
        load_residual_scaled=-4e-3,
        load_residual_n=-8e-5,
        feasible_step_limit=0.8,
        accepted_step_length=0.4,
        krylov_iterations=7,
        krylov_diagnostics=(),
        schur_condition_indicator=0.3,
        minimum_film_thickness_m=19e-6,
        minimum_pressure_pa=-2e4,
        maximum_pressure_pa=5e4,
    )


def test_monolithic_plots_include_required_diagnostics_and_units(tmp_path):
    x = np.linspace(-1e-3, 1e-3, 5)
    y = np.linspace(-0.5e-3, 0.5e-3, 3)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    pressure = (1.0 - xx**2 / 1e-6) * 1e5
    film = (20.0 + xx**2 * 1e6 + yy**2 * 1e6) * 1e-6
    history = (_history_entry(), _history_entry(2))
    solution = SimpleNamespace(
        pressure_pa=pressure,
        film_thickness_m=film,
        history=history,
    )
    continuation = SimpleNamespace(
        steps=(
            SimpleNamespace(
                attempt=1,
                target_load_n=-0.2,
                accepted=True,
                central_film_offset_m=20e-6,
                minimum_film_thickness_m=19e-6,
            ),
            SimpleNamespace(
                attempt=2,
                target_load_n=-0.1,
                accepted=False,
                central_film_offset_m=20e-6,
                minimum_film_thickness_m=19e-6,
            ),
            SimpleNamespace(
                attempt=3,
                target_load_n=-0.05,
                accepted=True,
                central_film_offset_m=22e-6,
                minimum_film_thickness_m=20e-6,
            ),
        )
    )

    newton_figure, newton_axes = plot_monolithic_newton_history(history)
    assert "Newton" in newton_figure._suptitle.get_text()
    assert newton_axes[0].get_ylabel() == "Scaled residual"
    assert newton_axes[4].get_ylabel() == "Preconditioned GMRES residual"
    assert len(newton_axes[1].lines) == 2
    failed_figure, failed_axes = plot_monolithic_newton_history(
        history,
        failure_diagnostics={"iteration": 3, "load_residual_n": -2e-5},
    )
    assert "failed" in failed_figure._suptitle.get_text()
    assert "load_residual_n" in failed_axes[0].texts[0].get_text()

    continuation_figure, continuation_axes = plot_monolithic_continuation_history(
        continuation
    )
    assert continuation_axes[0].get_ylabel() == "Target signed load [N]"
    assert len(continuation_axes[0].collections) == 1

    paths = save_monolithic_figures(
        x,
        y,
        solution,
        case_name="newton-test",
        results_directory=tmp_path,
        continuation=continuation,
    )
    assert len(paths) == 6
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)
    assert "pressure_distribution.png" in {path.name for path in paths}

    import matplotlib.pyplot as plt

    plt.close(newton_figure)
    plt.close(failed_figure)
    plt.close(continuation_figure)
