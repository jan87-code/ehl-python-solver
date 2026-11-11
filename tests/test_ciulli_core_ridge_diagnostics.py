import numpy as np
import pytest

from ehl_solver.prototype import PointContactCase
from examples.diagnose_ciulli_core_ridge import (
    _contiguous_threshold_width,
    _fixed_region_comparison,
    _manufactured_flux_order,
    _quadratic_peak,
    _symmetry_metrics,
)


def test_transverse_symmetry_metrics_detect_only_broken_reflection():
    axis = np.linspace(-2.0, 2.0, 5)
    xx, yy = np.meshgrid(axis, axis)
    pressure = 100.0 + xx**2 + yy**2
    fields = {
        "pressure_pa": pressure,
        "film_m": 1e-6 + pressure * 1e-9,
        "deformation_m": pressure * 1e-12,
        "theta": np.ones_like(pressure),
        "density_kg_m3": 800.0 + pressure,
        "viscosity_pa_s": 0.1 + pressure,
    }

    symmetric = _symmetry_metrics(fields)
    assert symmetric["pressure_pa"]["relative_l2"] == pytest.approx(0.0)

    fields["pressure_pa"][0, 0] += 1.0
    asymmetric = _symmetry_metrics(fields)
    assert asymmetric["pressure_pa"]["relative_l2"] > 0


def test_fixed_region_comparison_separates_ridge_and_complement():
    x = np.linspace(-100e-6, 60e-6, 33)
    y = np.linspace(-75e-6, 75e-6, 31)
    xx, yy = np.meshgrid(x, y)
    pressure = np.full_like(xx, 100e6)
    film = np.full_like(xx, 400e-9)
    ridge = (
        (xx >= 35e-6)
        & (xx <= 60e-6)
        & (np.abs(yy) >= 45e-6)
        & (np.abs(yy) <= 75e-6)
    )
    coarse = {"x_m": x, "y_m": y, "pressure_pa": pressure, "film_m": film}
    fine = {
        **coarse,
        "pressure_pa": pressure + ridge * 10e6,
        "film_m": film + ridge * 5e-9,
    }

    result = _fixed_region_comparison(coarse, fine)
    regions = result["regions"]
    assert 9.0 < regions["fixed_core_ridge"]["pressure_rms_mpa"] <= 10.0
    assert regions["main_excluding_fixed_core_ridge"]["pressure_rms_mpa"] < 1.0
    assert sum(
        regions[name]["pressure_difference_energy_share_of_main_percent"]
        for name in ("fixed_core_ridge", "main_excluding_fixed_core_ridge")
    ) == pytest.approx(100.0)


def test_manufactured_flux_orders_reflect_donor_and_diffusion_stencils():
    result = _manufactured_flux_order(PointContactCase(), grids=(33, 65, 129))
    for coarse in result["results"][:-1]:
        assert coarse["advection_observed_order"] == pytest.approx(1.0, abs=0.08)
        assert coarse["diffusion_observed_order"] == pytest.approx(2.0, abs=0.08)
        assert coarse["observed_order_to_next_grid"] == pytest.approx(1.0, abs=0.08)
    assert result["results"][-1]["smooth_full_film"]


def test_quadratic_peak_rejects_a_saddle_as_a_peak_estimate():
    x = np.linspace(30e-6, 65e-6, 15)
    y = np.linspace(40e-6, 80e-6, 17)
    xx, yy = np.meshgrid(x, y)
    pressure = 700e6 + 1e15 * (xx - 50e-6) ** 2 - 1e15 * (yy - 60e-6) ** 2
    result = _quadratic_peak(
        {"x_m": x, "y_m": y, "pressure_pa": pressure}, 50e-6, 60e-6
    )
    assert not result["valid"]
    assert result["reason"] == "3x3 quadratic fit is not a strict maximum"


def test_threshold_width_is_flagged_when_cut_by_fixed_band():
    coordinate = np.arange(0.0, 6.0)
    values = np.array([10.0, 10.0, 9.0, 4.0, 1.0, 0.0])
    result = _contiguous_threshold_width(
        coordinate, values, center_index=1, threshold_pa=8.0, bounds_m=(1.0, 4.0)
    )
    assert result["contiguous_cells"] == 1
    assert result["truncated_by_analysis_band"]
