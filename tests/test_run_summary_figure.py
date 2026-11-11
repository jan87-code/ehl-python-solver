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


def test_run_summary_writes_deterministic_png_and_raw_profile_csv(tmp_path):
    x, y, pressure, film = _fields()
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
    )
    assert figure_path.name == "ciulli_15N_15N_3x3_donor-flux_FAILED_summary.png"
    assert figure_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert profile_path.name.endswith("_profiles.csv")
    data = np.genfromtxt(profile_path, delimiter=",", names=True)
    np.testing.assert_allclose(data["pressure_centerline_pa"], 100e6 + 2e10 * x)
    np.testing.assert_allclose(data["film_centerline_m"], 100e-9 + 4e-6 * x)
