import numpy as np
import pytest

from ehl_solver.cavitating_newton import solve_cavitating_newton
from ehl_solver.verification import audit_cavitating_solution
from examples.run_cavitating_boundaries import boundary_cases
from examples.run_cavitating_domain import make_boundary_problem, make_domain_problem


def test_domain_expansions_preserve_spacing_and_existing_nodes():
    baseline = make_domain_problem(0)
    for extra in (4, 8):
        expanded = make_domain_problem(extra)
        assert expanded.dx_m == pytest.approx(baseline.dx_m)
        assert expanded.dy_m == pytest.approx(baseline.dy_m)
        np.testing.assert_allclose(expanded.x_m[extra:-extra], baseline.x_m, atol=1e-18)
        np.testing.assert_allclose(expanded.y_m[extra:-extra], baseline.y_m, atol=1e-18)
        assert expanded.pressure_scale_pa == baseline.pressure_scale_pa
        assert expanded.e_star_pa == baseline.e_star_pa
        assert expanded.inlet_fractional_film_content == baseline.inlet_fractional_film_content


@pytest.mark.parametrize("extra", [-1, 1.5, True])
def test_invalid_domain_extensions_are_rejected(extra):
    with pytest.raises(ValueError):
        make_domain_problem(extra)


def test_separate_boundary_study_preserves_overlap_nodes_and_all_physical_inputs():
    baseline = make_boundary_problem()
    cases = list(boundary_cases())
    assert len(cases) == 7
    assert len({name for name, *_ in cases}) == 7  # Same shapes must not overwrite files.
    for _, boundary, nodes, expanded in cases:
        west = nodes if boundary == "inlet" else 0
        south = nodes if boundary == "transverse" else 0
        np.testing.assert_array_equal(expanded.x_m[west : west + 17], baseline.x_m)
        np.testing.assert_array_equal(expanded.y_m[south : south + 13], baseline.y_m)
        assert expanded.dx_m == pytest.approx(baseline.dx_m, rel=1e-14)
        assert expanded.dy_m == pytest.approx(baseline.dy_m, rel=1e-14)
        assert expanded.shape == (
            13 + (2 * nodes if boundary == "transverse" else 0),
            17 + (nodes if boundary in ("inlet", "outlet") else 0),
        )
        for attr in (
            "central_film_offset_m",
            "radius_x_m",
            "radius_y_m",
            "e_star_pa",
            "length_scale_m",
            "film_scale_m",
            "pressure_scale_pa",
            "reynolds_parameter",
            "inlet_fractional_film_content",
            "film_floor_m",
        ):
            assert getattr(expanded, attr) == getattr(baseline, attr)


@pytest.mark.parametrize("boundary", ["inlet", "outlet", "transverse"])
@pytest.mark.parametrize("extra", [-1, True, 1.5])
def test_invalid_individual_extensions_are_rejected(boundary, extra):
    with pytest.raises(ValueError):
        make_boundary_problem(**{f"{boundary}_nodes": extra})


def test_extending_cavitated_outlet_preserves_coupled_force_closed_root():
    baseline = make_boundary_problem()
    expanded = make_boundary_problem(outlet_nodes=4)
    roots = [
        solve_cavitating_newton(problem, target_load_n=0.002).solution
        for problem in (baseline, expanded)
    ]
    for problem, root in zip((baseline, expanded), roots, strict=True):
        audit = audit_cavitating_solution(problem, root)
        assert audit.mass_maximum < 1e-8
        assert audit.load_n == pytest.approx(0.002, rel=1e-8)
        assert audit.maximum_complementarity < 1e-10
        assert audit.minimum_saturation >= 0
        assert audit.maximum_saturation <= 1 + 1e-10
        assert audit.film_coupling_max_error_m < 1e-18
    original, extended = roots
    np.testing.assert_allclose(
        extended.pressure_pa[:, :17], original.pressure_pa, rtol=1e-8, atol=1e-5
    )
    np.testing.assert_allclose(
        extended.film_thickness_m[:, :17], original.film_thickness_m, rtol=1e-10
    )
    # The old east boundary content was a copy, not an interior flow unknown.
    np.testing.assert_allclose(
        extended.fractional_film_content[1:-1, :16],
        original.fractional_film_content[1:-1, :16],
        atol=1e-9,
    )
    assert extended.central_film_offset_m == pytest.approx(
        original.central_film_offset_m, rel=1e-10
    )
    np.testing.assert_allclose(extended.pressure_pa[:, 16:], 0, atol=1e-6)
    h = extended.film_thickness_m / expanded.film_scale_m
    theta = extended.fractional_film_content
    convective_flux = 0.5 * (h[1:-1, 1:] + h[1:-1, :-1]) * theta[1:-1, :-1]
    np.testing.assert_allclose(np.diff(convective_flux[:, 16:], axis=1), 0, atol=1e-10)
