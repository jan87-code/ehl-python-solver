import numpy as np
import pytest

from ehl_solver.cavitation import solve_cavitating_reynolds
from ehl_solver.flux_diagnostics import decompose_flux, face_components, nested_load_budget
from ehl_solver.rectilinear_cavitation import solve_rectilinear_cavitation
from examples.run_farfield_budget import H0, LY, P_REF, L, graded_axis


def test_nonuniform_constant_film_pressure_boundary_matches_exact_linear_solution():
    x = np.array([0, 0.1, 0.25, 0.5, 0.8, 1]) * 1e-3
    y = np.array([-0.5, -0.2, 0, 0.1, 0.5]) * 1e-3
    expected = np.broadcast_to(4000 - 2000 * x / x[-1], (len(y), len(x))).copy()
    film = np.full_like(expected, 2e-6)
    result = solve_rectilinear_cavitation(
        x,
        y,
        film,
        0.04,
        0.1,
        length_scale_m=0.001,
        film_scale_m=2e-6,
        pressure_scale_pa=1e5,
        boundary_pressure_pa=expected,
    )
    np.testing.assert_allclose(result.pressure_pa, expected, rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(result.fractional_film_content, 1, atol=1e-13)
    assert result.audit["mass_maximum"] < 1e-9
    c, px, py, r = face_components(
        x, y, film, result.pressure_pa, result.fractional_film_content, 0.04, 0.1
    )
    np.testing.assert_allclose(c, 0.1 * 2e-6, rtol=1e-14)
    np.testing.assert_allclose(px, (2e-6) ** 3 * 2e6 / (12 * 0.04), rtol=1e-10)
    np.testing.assert_allclose(py, 0, atol=1e-18)
    np.testing.assert_allclose(r, 0, atol=1e-12)


def test_flux_cancellation_recovers_known_opposing_contributions_and_volume_units():
    x, y = np.linspace(0, 0.001, 7), np.linspace(-0.0005, 0.0005, 5)
    h = np.full((5, 7), 2e-6)
    # Choose dp/dx so Poiseuille cancels 90% of the Couette flux.
    gradient = 0.9 * 0.1 * 12 * 0.04 / (2e-6) ** 2
    p = np.broadcast_to(gradient * x, (5, 7))
    d = decompose_flux(
        x,
        y,
        h,
        p,
        np.ones_like(h),
        0.04,
        0.1,
        length_scale_m=0.001,
        film_scale_m=2e-6,
        pressure_scale_pa=1e5,
    )
    assert d["inlet_poiseuille_m3_s"] == pytest.approx(-0.9 * d["inlet_couette_m3_s"])
    assert d["inlet_net_m3_s"] == pytest.approx(0.1 * d["inlet_couette_m3_s"])
    assert d["inlet_cancellation_condition"] == pytest.approx(19)
    assert d["maximum_face_cancellation_condition"] == pytest.approx(19)
    assert d["inlet_net_m3_s"] == pytest.approx(0.1 * 0.1 * 2e-6 * 3 * (y[1] - y[0]))


def test_exact_cancellation_is_reported_as_unbounded_without_nonfinite_json():
    x = y = np.arange(3, dtype=float)
    h = np.ones((3, 3))
    p = np.broadcast_to(12 * x, h.shape)
    d = decompose_flux(
        x,
        y,
        h,
        p,
        h,
        1,
        1,
        length_scale_m=1,
        film_scale_m=1,
        pressure_scale_pa=1,
    )
    assert d["inlet_net_m3_s"] == 0
    assert d["inlet_cancellation_condition"] is None
    assert d["maximum_face_cancellation_condition"] is None
    assert d["nonzero_opposing_parts_with_zero_net"] == 6


def test_nested_force_budget_matches_independent_constant_pressure_area_integrals():
    old = np.array([-1.0, 0, 1.0])
    new = np.arange(-2.0, 3.0)
    d = nested_load_budget(old, old, np.full((3, 3), 2.0), new, new, np.full((5, 5), 3.0))
    assert d["old_load_n"] == 8
    assert d["new_load_n"] == 48
    assert d["added_area_load_n"] == 36
    assert d["shared_area_readjustment_n"] == 4
    assert d["identity_defect_n"] == 0
    with pytest.raises(ValueError, match="nested"):
        nested_load_budget(old + 0.1, old, np.ones((3, 3)), new, new, np.ones((5, 5)))
    with pytest.raises(ValueError, match="unchanged cells"):
        nested_load_budget(np.array([-1.0, 1.0]), old, np.ones((3, 2)), new, new, np.ones((5, 5)))


@pytest.mark.skipif(
    np.finfo(np.longdouble).nmant <= np.finfo(float).nmant,
    reason="wider diagnostic arithmetic unavailable on this platform",
)
def test_physical_defect_correction_reduces_original_si_residual_without_changing_root():
    x = graded_axis(192, 192, 0.125) * L
    y = graded_axis(192, 192, 0.125) * LY
    h = H0 + x[None, :] ** 2 / 0.02 + y[:, None] ** 2 / 0.04
    options = dict(length_scale_m=L, film_scale_m=H0, pressure_scale_pa=P_REF)
    before = solve_rectilinear_cavitation(x, y, h, 0.04, 0.001, **options)
    after = solve_rectilinear_cavitation(x, y, h, 0.04, 0.001, flux_refinement_steps=3, **options)
    assert after.pressure_pa.dtype == np.float64
    assert after.audit["mass_maximum"] < before.audit["mass_maximum"]
    assert after.audit["mass_maximum"] < 1e-8
    np.testing.assert_allclose(after.pressure_pa, before.pressure_pa, rtol=1e-8, atol=1e-7)
    assert after.audit["relative_local_balance"] < 1e-10
    assert after.audit["relative_global_balance"] < 1e-10
    assert after.audit["maximum_complementarity"] < 1e-10
    assert after.fractional_film_content.min() >= 0
    assert after.fractional_film_content.max() <= 1 + 1e-10
    assert after.refinement_history
    assert all(t["trial"] < t["before"] for t in after.refinement_history if t["accepted"])


@pytest.mark.parametrize(
    "options",
    [
        {"flux_refinement_steps": 4},
        {"flux_refinement_steps": True},
        {"boundary_pressure_pa": np.full((5, 5), -1.0)},
        {"boundary_pressure_pa": np.ones((5, 5)), "inlet_content": 0.5},
    ],
)
def test_invalid_boundary_or_refinement_requests_raise(options):
    with pytest.raises(ValueError):
        solve_rectilinear_cavitation(
            np.arange(5),
            np.arange(5),
            np.ones((5, 5)),
            1.0,
            1.0,
            length_scale_m=1,
            film_scale_m=1,
            pressure_scale_pa=1,
            **options,
        )


def test_matched_boundary_mesh_refines_to_original_analytical_reservoir():
    from examples.run_flux_conditioning import matched_case

    cases = [matched_case(-1, n)[0] for n in (65, 129, 257)]
    assert all(c["accepted"] for c in cases)
    assert (
        cases[2]["relative_force_difference"]
        < cases[1]["relative_force_difference"]
        < cases[0]["relative_force_difference"]
    )
    assert (
        cases[2]["relative_pressure_difference"]
        < cases[1]["relative_pressure_difference"]
        < cases[0]["relative_pressure_difference"]
    )
    assert cases[-1]["relative_force_difference"] < 0.001
    assert cases[-1]["relative_pressure_difference"] < 0.001
    assert cases[-1]["inlet_pressure_pa"] > 0
    changed, _ = matched_case(-1, 129, mode="finite_reservoir")
    assert changed["accepted"]
    assert changed["inlet_pressure_pa"] == 0
    assert changed["relative_force_difference"] > 0.5  # Different physical reservoir.


def test_forced_wider_refinement_does_not_silently_use_unavailable_precision(monkeypatch):
    import ehl_solver.rectilinear_cavitation as module

    finfo = np.finfo
    monkeypatch.setattr(
        module.np, "finfo", lambda dtype: finfo(float) if dtype is np.longdouble else finfo(dtype)
    )
    with pytest.raises(RuntimeError, match="unavailable"):
        module.solve_rectilinear_cavitation(
            np.arange(5),
            np.arange(5),
            np.ones((5, 5)),
            1,
            1,
            length_scale_m=1,
            film_scale_m=1,
            pressure_scale_pa=1,
            flux_refinement_steps=1,
        )


def test_point_matched_trace_recovers_independent_uniform_reference_on_shared_domain():
    from examples.run_flux_conditioning import matched_point_case

    units = np.linspace(-16, 16, 129)
    x, y = units * L, units * LY
    h = H0 + x[None, :] ** 2 / 0.02 + y[:, None] ** 2 / 0.04
    reference = solve_cavitating_reynolds(h / H0, 0.25, 0.25 * LY / L, 1)
    state = {"x_m": x, "y_m": y, "film_m": h, "pressure_pa": reference.pressure * P_REF}
    entry, _ = matched_point_case(state)
    assert entry["pressure_relative_difference"] < 1e-9
    assert entry["matched_audit"]["mass_maximum"] < 1e-8
    assert entry["matched_reconstructed_load_n"] == pytest.approx(
        entry["reference_full_load_n"], rel=1e-10
    )
    assert entry["zero_boundary_window_load_n"] < 0.99 * entry["matched_window_load_n"]
