import numpy as np

from ehl_solver.interactive_plot import interactive_figure, write_interactive_plot
from ehl_solver.prototype import PointContactCase, hertz_guess
from ehl_solver.prototype_cli import run_case


def hertz_root(case):
    system = case.system()
    state, _initialization = hertz_guess(case, system)
    return system.solution(state)


def test_surfaces_show_full_resolution_fields_with_units():
    case = PointContactCase()
    root = hertz_root(case)
    figure = interactive_figure(case, root, "rejected")

    assert len(figure.data) == 2
    np.testing.assert_array_equal(
        np.asarray(figure.data[0].z),
        root.solution.pressure_pa * 1e-9,
    )
    np.testing.assert_array_equal(
        np.asarray(figure.data[1].z),
        root.solution.film_thickness_m * 1e9,
    )
    assert figure.layout.scene.xaxis.title.text == "x [mm], entrainment +x"
    assert figure.layout.scene.yaxis.title.text == "y [mm]"
    assert figure.layout.scene.zaxis.title.text == "Pressure [GPa]"
    assert figure.layout.scene2.xaxis.title.text == "x [mm], entrainment +x"
    assert figure.layout.scene2.yaxis.title.text == "y [mm]"
    assert figure.layout.scene2.zaxis.title.text == "Film thickness [nm]"
    assert figure.layout.meta == {
        "run_status": "rejected",
        "physical_validation_complete": False,
    }


def test_html_embeds_plotly_for_offline_use(tmp_path):
    case = PointContactCase()
    root = hertz_root(case)
    output = tmp_path / "interactive_3d.html"
    write_interactive_plot(case, root, "accepted_numerical_root", output)

    html = output.read_text(encoding="utf-8")
    assert "Plotly.newPlot" in html
    assert "plotly.js v" in html.lower()
    assert "<script src=" not in html.lower()
    assert "Status: accepted_numerical_root" in html
    assert "physical validation incomplete" in html


def test_rejected_run_with_last_root_still_writes_interactive_plot(tmp_path):
    output = tmp_path / "rejected_with_root"
    report = run_case(PointContactCase(), output, maximum_iterations=0)

    assert report["status"] == "rejected"
    html = (output / "interactive_3d.html").read_text(encoding="utf-8")
    assert "Status: rejected" in html
    assert "physical validation incomplete" in html
