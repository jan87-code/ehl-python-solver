"""Self-contained interactive visualizations for completed prototype runs."""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from ehl_solver.property_warm import WarmPropertyRoot
from ehl_solver.prototype import PointContactCase


def interactive_figure(case: PointContactCase, root: WarmPropertyRoot, status: str):
    if status not in ("accepted_numerical_root", "rejected"):
        raise ValueError(f"Unsupported run status for plotting: {status}")

    problem = case.problem()
    pressure = np.asarray(root.solution.pressure_pa, dtype=np.float64)
    film = np.asarray(root.solution.film_thickness_m, dtype=np.float64)
    expected_shape = (case.ny, case.nx)
    if pressure.shape != expected_shape or film.shape != expected_shape:
        raise ValueError(f"Surface fields must have shape {expected_shape}")
    if not np.isfinite(pressure).all() or not np.isfinite(film).all():
        raise ValueError("Surface fields must contain only finite values")

    figure = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "surface"}, {"type": "surface"}]],
        subplot_titles=("Pressure", "Film thickness"),
        horizontal_spacing=0.08,
    )
    x_mm = (problem.x_m * 1e3).tolist()
    y_mm = (problem.y_m * 1e3).tolist()
    figure.add_trace(
        go.Surface(
            x=x_mm,
            y=y_mm,
            z=(pressure * 1e-9).tolist(),
            colorbar={"title": "Pressure [GPa]", "x": 0.45, "len": 0.85},
            colorscale="Viridis",
            name="Pressure",
            hovertemplate=(
                "x=%{x:.6g} mm<br>y=%{y:.6g} mm"
                "<br>pressure=%{z:.6g} GPa<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Surface(
            x=x_mm,
            y=y_mm,
            z=(film * 1e9).tolist(),
            colorbar={"title": "Film thickness [nm]", "x": 1.0, "len": 0.85},
            colorscale="Viridis",
            name="Film thickness",
            hovertemplate=(
                "x=%{x:.6g} mm<br>y=%{y:.6g} mm"
                "<br>film=%{z:.6g} nm<extra></extra>"
            ),
        ),
        row=1,
        col=2,
    )
    for scene, z_title in (
        ("scene", "Pressure [GPa]"),
        ("scene2", "Film thickness [nm]"),
    ):
        figure.update_layout(
            **{
                scene: {
                    "xaxis_title": "x [mm], entrainment +x",
                    "yaxis_title": "y [mm]",
                    "zaxis_title": z_title,
                }
            }
        )
    figure.update_layout(
        title={
            "text": (
                f"{case.load_n:g} N, {case.ny}x{case.nx}, {case.transport}"
                f"<br><sup>Status: {status}; physical validation incomplete</sup>"
            ),
            "x": 0.5,
        },
        height=760,
        margin={"l": 10, "r": 10, "t": 90, "b": 10},
        meta={"run_status": status, "physical_validation_complete": False},
    )
    return figure


def write_interactive_plot(
    case: PointContactCase,
    root: WarmPropertyRoot,
    status: str,
    path: Path,
) -> None:
    figure = interactive_figure(case, root, status)
    html = pio.to_html(
        figure,
        full_html=True,
        include_plotlyjs=True,
        config={"responsive": True, "displaylogo": False},
    )
    Path(path).write_text(html, encoding="utf-8")
