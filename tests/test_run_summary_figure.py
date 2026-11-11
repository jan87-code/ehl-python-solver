import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

from ehl_solver.visualization import plot_run_summary, write_run_summary


def _fields():
    x = np.array([-1e-3, 0.0, 1e-3])
    y = np.array([-2e-3, 1e-3, 3e-3])
    xx, yy = np.meshgrid(x, y)
    pressure = 100e6 + 2e10 * xx + 3e10 * yy
    film = 100e-9 + 4e-6 * xx + 5e-6 * yy
    return x, y, pressure, film


def test_run_summary_uses_full_range_and_interpolated_centerline():
    x, y, pressure, film = _fields()
    figure, profiles, metrics = plot_run_summary(
        x,
        y,
        pressure,
        film,
        load_n=15,
        formulation="donor-flux",
        status="accepted_numerical_root",
        force_audit={"force_relative_error": 1e-12},
        newton_corrections=7,
        backend="CPU",
        runtime_seconds=12.5,
    )
    try:
        assert figure._suptitle.get_text() == "15 N, 3x3, donor-flux, ACCEPTED"
        assert "FAILED" not in figure.texts[0].get_text()
        assert len(figure.axes[0].images) == 1
        image = figure.axes[0].images[0]
        assert image.get_cmap().name == "magma"
        assert np.min(image.get_array()) == np.min(pressure) * 1e-9
        assert np.max(image.get_array()) == np.max(pressure) * 1e-9
        np.testing.assert_allclose(profiles["pressure_centerline_pa"], 100e6 + 2e10 * x)
        np.testing.assert_allclose(profiles["film_centerline_m"], 100e-9 + 4e-6 * x)
        assert metrics["central_film_m"] == 100e-9
        assert metrics["minimum_film_xy_m"] == (-1e-3, -2e-3)
        assert figure.axes[2].get_xlim() == (-1.0, 1.0)
        assert figure.axes[3].get_xlim() == (-1.0, 1.0)
        assert figure.axes[3].get_ylabel() == "Film thickness [nm]"
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)


def test_film_display_cap_is_explicit_and_does_not_change_profile_or_field():
    x, y, pressure, film = _fields()
    film[1, 0] = 150e-9
    film[1, 2] = 1200e-9
    original_film = film.copy()
    figure, profiles, _ = plot_run_summary(
        x,
        y,
        pressure,
        film,
        load_n=15,
        formulation="donor-flux",
        status="accepted_numerical_root",
        film_display_range_nm=(300.0, 996.0),
    )
    try:
        image = figure.axes[1].images[0]
        assert image.get_clim() == (300.0, 996.0)
        assert np.min(image.get_array()) == np.min(film) * 1e9
        assert np.max(image.get_array()) == np.max(film) * 1e9
        assert "300-996 nm" in figure.axes[1].get_title()
        assert figure.axes[3].get_ylim() == (300.0, 996.0)
        assert "300-996 nm" in figure.axes[3].get_title()
        np.testing.assert_array_equal(film, original_film)
        expected_profile = np.array(
            [np.interp(0.0, y, film[:, i]) for i in range(x.size)]
        )
        np.testing.assert_allclose(
            profiles["film_centerline_m"], expected_profile, rtol=2e-16, atol=0.0
        )
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)


def test_automatic_film_cap_uses_contact_region_without_clipping_source_data():
    x = np.array([-1e-3, 0.0, 1e-3])
    y = np.array([-1e-3, 0.0, 1e-3])
    pressure = np.array(
        [[0.0, 0.0, 0.0], [0.0, 1e9, 0.0], [0.0, 0.0, 0.0]]
    )
    film = np.array(
        [[3000e-9, 3000e-9, 3000e-9], [3000e-9, 800e-9, 3000e-9],
         [3000e-9, 3000e-9, 3000e-9]]
    )
    original_film = film.copy()
    figure, profiles, metrics = plot_run_summary(
        x,
        y,
        pressure,
        film,
        load_n=15,
        formulation="donor-flux",
        status="accepted_numerical_root",
    )
    try:
        image = figure.axes[1].images[0]
        assert image.get_clim() == (800.0, 1000.0)
        assert "1000 nm" in figure.axes[1].get_title()
        assert metrics["film_display_cap_nm"] == 1000.0
        assert metrics["centerline_film_clipped_above_range"]
        assert figure.axes[3].get_ylim() == (800.0, 1000.0)
        assert np.max(image.get_array()) == 3000.0
        np.testing.assert_array_equal(film, original_film)
        assert profiles["film_centerline_m"][0] == 3000e-9
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)


def test_run_summary_contact_inset_shows_film_without_modifying_it():
    x = np.linspace(-0.4e-3, 0.4e-3, 33)
    y = x.copy()
    xx, yy = np.meshgrid(x, y)
    pressure = 1e9 * np.exp(-((xx / 0.05e-3) ** 2 + (yy / 0.05e-3) ** 2))
    film = 300e-9 + 1e-6 * ((xx / 0.2e-3) ** 2 + (yy / 0.2e-3) ** 2)
    original = film.copy()
    figure, _, metrics = plot_run_summary(
        x, y, pressure, film, load_n=15, formulation="donor-flux", status="accepted"
    )
    try:
        inset = figure.axes[1].child_axes[0]
        assert inset.get_title() == "Contact detail [mm]"
        assert inset.images[0].get_clim() == metrics["film_display_range_nm"]
        np.testing.assert_array_equal(film, original)
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)


def test_run_summary_marks_symmetric_global_max_and_centerline_downstream_peak():
    x = np.linspace(-2e-3, 2e-3, 9)
    y = np.linspace(-2e-3, 2e-3, 9)
    xx, yy = np.meshgrid(x, y)
    pressure = 30e6 * np.exp(-((xx / 1.5e-3) ** 2 + (yy / 1.5e-3) ** 2))
    pressure += 70e6 * np.exp(
        -((xx / 0.5e-3) ** 2 + ((yy - 0.5e-3) / 0.25e-3) ** 2)
    )
    pressure += 70e6 * np.exp(
        -((xx / 0.5e-3) ** 2 + ((yy + 0.5e-3) / 0.25e-3) ** 2)
    )
    pressure += 20e6 * np.exp(
        -(((xx - 1.5e-3) / 0.3e-3) ** 2 + (yy / 0.3e-3) ** 2)
    )
    film = 300e-9 + 1e-6 * (xx**2 + yy**2) / 1e-6
    figure, _, metrics = plot_run_summary(
        x,
        y,
        pressure,
        film,
        load_n=15,
        formulation="donor-flux",
        status="accepted_numerical_root",
    )
    try:
        assert len(metrics["maximum_pressure_locations_m"]) == 2
        assert metrics["centerline_outlet_peak"] is not None
        assert metrics["centerline_outlet_peak"]["x_m"] > 0.0
        assert len(figure.axes[0].collections) >= 2
        assert figure.axes[3].get_ylabel() == "Film thickness [nm]"
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)


def test_outlet_peak_selection_skips_near_center_local_maximum():
    x = np.array([-2e-3, 0.0, 1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3])
    y = np.array([-1e-3, 0.0, 1e-3])
    centerline_pressure = np.array(
        [0.0, 130.0, 80.0, 115.0, 50.0, 70.0, 110.0, 60.0, 0.0]
    ) * 1e6
    pressure = centerline_pressure[None, :] * np.exp(
        -((y[:, None] / 0.5e-3) ** 2)
    )
    xx, yy = np.meshgrid(x, y)
    film = 300e-9 + 1e-6 * (xx**2 + yy**2) / 1e-6
    figure, _, metrics = plot_run_summary(
        x,
        y,
        pressure,
        film,
        load_n=15,
        formulation="donor-flux",
        status="accepted_numerical_root",
    )
    try:
        assert metrics["centerline_outlet_peak"]["x_m"] == 5e-3
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)


def test_run_summary_writes_deterministic_png_and_raw_profile_csv(tmp_path):
    x, y, pressure, film = _fields()
    archive_path = tmp_path / "solution.npz"
    archive_path.touch()
    figure_path, profile_path = write_run_summary(
        x,
        y,
        pressure,
        film,
        directory=tmp_path,
        case_name="ciulli_15N",
        load_n=15,
        formulation="donor-flux",
        status="rejected",
        field_archive_path=archive_path,
    )
    assert figure_path.name == "ciulli_15N_15N_3x3_donor-flux_FAILED_summary.png"
    assert figure_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert profile_path.name.endswith("_profiles.csv")
    data = np.genfromtxt(profile_path, delimiter=",", names=True)
    np.testing.assert_allclose(data["pressure_centerline_pa"], 100e6 + 2e10 * x)
    np.testing.assert_allclose(data["film_centerline_m"], 100e-9 + 4e-6 * x)
    pressure_path = tmp_path / profile_path.name.replace("_profiles.csv", "_pressure_profile.csv")
    film_path = tmp_path / profile_path.name.replace("_profiles.csv", "_film_profile.csv")
    metadata_path = tmp_path / profile_path.name.replace("_profiles.csv", "_metadata.json")
    pressure_data = np.genfromtxt(pressure_path, delimiter=",", names=True)
    film_data = np.genfromtxt(film_path, delimiter=",", names=True)
    np.testing.assert_allclose(pressure_data["pressure_gpa"], (100e6 + 2e10 * x) * 1e-9)
    np.testing.assert_allclose(film_data["film_thickness_nm"], (100e-9 + 4e-6 * x) * 1e9)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["status_label"] == "FAILED / NOT AN ACCEPTED SOLUTION"
    assert Path(metadata["full_field_archive"]).name == archive_path.name
