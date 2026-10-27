"""Continuum truncation and independent uniform-state Fourier-symbol checks."""

import numpy as np

from examples.run_transport_consistency import alternating_pressure_probe, report, smooth_case


def test_full_film_mass_transport_is_second_order_against_continuum_source():
    rows = [smooth_case(n) for n in (33, 65, 129)]
    errors = [r["relative_l2_divergence_error"] for r in rows]
    assert rows[-1]["shape_ny_nx"] == [65, 129]
    assert all(3.7 < a / b < 4.2 for a, b in zip(errors, errors[1:], strict=False))
    assert errors[-1] < 6e-4


def test_variable_content_transport_has_expected_first_order_asymptote():
    # Smooth non-complementary fields isolate first-order theta upwinding.
    errors = [smooth_case(n, variable_content=True)["relative_l2_divergence_error"]
              for n in (65, 129, 257)]
    orders = np.log2(np.array(errors[:-1]) / errors[1:])
    assert 0.9 < orders[-1] < 1.1
    assert orders[-1] < orders[0]
    assert all(a > b for a, b in zip(errors, errors[1:], strict=False))


def test_alternating_pressure_has_diffusion_despite_centered_density_null_mode():
    result = alternating_pressure_probe()
    assert result["density_advection_maximum_kg_m2_s"] < 1e-12
    assert result["diffusion_maximum_kg_m2_s"] > 1
    assert result["relative_diffusion_symbol_error"] < 1e-13
    assert result["frozen_film_density_cell_peclet"] > 0


def test_report_retains_explicit_manufactured_scope_and_finite_evidence():
    result = report()
    assert "no production root" in result["status"]
    for rows in result["cases"].values():
        assert len(rows) == 4
        assert all(np.isfinite(r["relative_l2_divergence_error"]) for r in rows)
