import numpy as np
import pytest

from examples.analyze_ciulli_outlet_refinement import (
    _global_film_minimum,
    feature_corridor_metrics,
)


def test_feature_corridor_separates_localized_pressure_and_film_changes():
    x_m = np.arange(-2, 3, dtype=np.float64) * 1e-6
    y_m = np.arange(-2, 3, dtype=np.float64) * 1e-6
    pressure_coarse = np.full((5, 5), 10e6)
    pressure_fine = pressure_coarse.copy()
    pressure_fine[:, 2] += 20e6
    film_coarse = np.full((5, 5), 500e-9)
    film_fine = film_coarse.copy()
    film_fine[:, 3] += 2e-9
    y_um = np.array([-2.0, 0.0, 2.0])
    tracks = [
        (y_um, np.zeros(3), y_um, np.ones(3)),
        (y_um, np.zeros(3), y_um, np.ones(3)),
    ]

    metrics = feature_corridor_metrics(
        x_m,
        y_m,
        pressure_coarse,
        pressure_fine,
        film_coarse,
        film_fine,
        tracks,
        half_width_um=0.25,
    )

    assert metrics["pressure_difference_energy_share_inside_percent"] == pytest.approx(
        100.0
    )
    assert metrics["film_difference_energy_share_inside_percent"] == pytest.approx(
        100.0
    )
    assert metrics["pressure_rms_outside_mpa"] == pytest.approx(0.0)
    assert metrics["film_rms_outside_nm"] == pytest.approx(0.0)


def test_feature_corridor_rejects_nonpositive_width():
    values = np.ones((3, 3))
    coordinates = np.arange(3, dtype=np.float64)
    tracks = [
        (coordinates, coordinates, coordinates, coordinates),
        (coordinates, coordinates, coordinates, coordinates),
    ]

    with pytest.raises(ValueError, match="positive corridor"):
        feature_corridor_metrics(
            coordinates,
            coordinates,
            values,
            values,
            values,
            values,
            tracks,
            half_width_um=0,
        )


def test_global_minimum_locations_include_roundoff_symmetric_tie():
    epsilon = np.finfo(np.float64).eps
    film = np.full((3, 3), 1e-6)
    film[1, 0] = 2e-7
    film[1, 2] = 2e-7 + 4 * epsilon * 2e-7
    run = {
        "film_m": film,
        "x_m": np.array([-1e-6, 0.0, 1e-6]),
        "y_m": np.array([-1e-6, 0.0, 1e-6]),
    }

    minimum = _global_film_minimum(run)

    assert minimum["locations_xy_um"] == [[-1.0, 0.0], [1.0, 0.0]]
