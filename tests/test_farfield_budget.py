import numpy as np
import pytest

from ehl_solver.backend import make_backend
from ehl_solver.cavitating_newton import CavitatingNewtonError
from ehl_solver.cavitation import solve_cavitating_reynolds
from ehl_solver.elasticity import elastic_deformation
from ehl_solver.elasticity_audit import direct_midpoint_displacement
from ehl_solver.parabolic_reference import parabolic_jfo_reference
from ehl_solver.rectilinear_cavitation import flux_audit, solve_rectilinear_cavitation
from examples.run_farfield_budget import graded_axis


@pytest.mark.parametrize("content", [1.0, 0.99, 0.9])
def test_uniform_rectilinear_limit_matches_independent_existing_jfo(content):
    x, y = np.linspace(0, 1, 21), np.linspace(-0.5, 0.5, 9)
    h = np.broadcast_to(1.4 - 0.4 * x + 0.8 * x**2, (len(y), len(x)))
    original = solve_cavitating_reynolds(
        h, x[1] - x[0], y[1] - y[0], 2, inlet_fractional_film_content=content
    )
    actual = solve_rectilinear_cavitation(
        x,
        y,
        h,
        1 / 12,
        2,
        length_scale_m=1,
        film_scale_m=1,
        pressure_scale_pa=1,
        inlet_content=content,
    )
    np.testing.assert_allclose(actual.pressure_pa, original.pressure, atol=1e-12)
    np.testing.assert_allclose(
        actual.fractional_film_content, original.fractional_film_content, atol=1e-12
    )
    assert actual.audit["mass_maximum"] < 1e-8
    assert actual.audit["relative_global_balance"] < 1e-12


def test_nonuniform_cavitation_converges_to_independent_analytic_rigid_limit():
    h0, radius, eta, speed = 20e-6, 0.01, 0.04, 0.425
    length = np.sqrt(2 * radius * h0)
    y = np.sinh(np.linspace(-2, 2, 17)) / np.sinh(2) * 8 * length
    errors = []
    for n in (33, 65, 129):
        x = np.sinh(np.linspace(-1, 1, n)) / np.sinh(1) * 3 * length
        reference = parabolic_jfo_reference(x, h0, radius, eta, speed)
        h = np.broadcast_to(reference.film_thickness_m, (len(y), n))
        actual = solve_rectilinear_cavitation(
            x,
            y,
            h,
            eta,
            speed,
            length_scale_m=length,
            film_scale_m=h0,
            pressure_scale_pa=reference.pressure_scale_pa,
        )
        errors.append(
            np.linalg.norm(actual.pressure_pa[len(y) // 2] - reference.pressure_pa)
            / np.linalg.norm(reference.pressure_pa)
        )
        assert actual.audit["mass_maximum"] < 1e-8
        assert actual.audit["maximum_complementarity"] < 1e-10
        assert actual.fractional_film_content.min() >= 0
        assert actual.fractional_film_content.max() <= 1 + 1e-10
    assert errors[2] < errors[1] < errors[0]
    assert errors[-1] < 0.02


def test_domain_expansion_preserves_core_and_every_existing_graded_node():
    base = graded_axis(16, 16, 0.125)
    for west, east in ((32, 16), (16, 32), (32, 32), (128, 128)):
        expanded = graded_axis(west, east, 0.125)
        for value in base:
            assert value in expanded
        core = expanded[abs(expanded) <= 2]
        np.testing.assert_allclose(np.diff(core), 0.125, atol=0, rtol=0)
    for step in (0.5, 0.25, 0.125):
        fixed = graded_axis(32, 32, step, snap=False)
        assert fixed[0] == -32
        assert fixed[-1] == 32


def test_zero_padding_embedding_and_spatial_displacement_keep_free_space_convention():
    rng = np.random.default_rng(17)
    p = np.zeros((7, 9))
    p[1:-1, 1:-1] = rng.uniform(0, 2e6, (5, 7))
    dx, dy, modulus = 2e-6, 3e-6, 1e9
    small = elastic_deformation(p, dx, dy, modulus, make_backend("numpy"))
    embedded = np.pad(p, ((4, 4), (6, 6)))
    large = elastic_deformation(embedded, dx, dy, modulus, make_backend("numpy"))
    np.testing.assert_allclose(large[4:11, 6:15], small, rtol=1e-12, atol=1e-21)
    x, y = (np.arange(9) - 4) * dx, (np.arange(7) - 3) * dy
    points = [[x[i], y[j]] for j, i in ((3, 4), (0, 0), (1, 6), (6, 8))]
    direct = direct_midpoint_displacement(x, y, p, modulus, points)
    np.testing.assert_allclose(
        direct, [small[j, i] for j, i in ((3, 4), (0, 0), (1, 6), (6, 8))], rtol=1e-12, atol=1e-21
    )


def test_nonuniform_si_volume_audit_has_correct_midpoint_control_widths():
    x, y = np.array([0, 1, 2, 4, 7]) * 1e-4, np.array([-3, -1, 0, 2, 5]) * 1e-4
    h, p, theta = np.full((5, 5), 2e-6), np.zeros((5, 5)), np.full((5, 5), 0.7)
    result = flux_audit(
        x, y, h, p, theta, 0.04, 0.1, length_scale_m=0.001, film_scale_m=2e-6, pressure_scale_pa=1e6
    )
    flow = 0.1 * 2e-6 * 0.7 * (y[-1] + y[-2] - y[0] - y[1]) / 2
    np.testing.assert_allclose(result["outward_volume_flow_m3_s"], [-flow, flow, 0, 0])
    assert result["mass_maximum"] == 0
    assert result["relative_global_balance"] == 0


@pytest.mark.parametrize(
    "changes",
    [{"x_m": [0, 1, 1]}, {"inlet_content": 1.1}, {"maximum_iterations": 0}, {"speed_m_s": 0}],
)
def test_rectilinear_invalid_inputs_raise(changes):
    arguments = dict(
        x_m=np.arange(5),
        y_m=np.arange(5),
        film_m=np.ones((5, 5)),
        viscosity_pa_s=1,
        speed_m_s=1,
        length_scale_m=1,
        film_scale_m=1,
        pressure_scale_pa=1,
    )
    with pytest.raises(ValueError):
        solve_rectilinear_cavitation(**(arguments | changes))


def test_active_set_budget_exhaustion_is_reported():
    x, y = np.linspace(-2, 3, 21), np.linspace(-2, 2, 17)
    h = 1 + x[None, :] ** 2 + y[:, None] ** 2
    with pytest.raises(RuntimeError, match="iteration limit"):
        solve_rectilinear_cavitation(
            x,
            y,
            h,
            1 / 12,
            1,
            length_scale_m=1,
            film_scale_m=1,
            pressure_scale_pa=1,
            maximum_iterations=1,
        )


def test_reference_recovery_retains_failed_candidate_and_original_iteration_limits(monkeypatch):
    import examples.run_farfield_budget as module

    original = module.solve_cavitating_newton
    calls = []

    def fail_once(problem, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            root = original(problem)
            raise CavitatingNewtonError("injected cold failure", root.history, root.solution)
        return original(problem, **kwargs)

    monkeypatch.setattr(module, "solve_cavitating_newton", fail_once)
    entry, fields = module.solve_case(
        "recovery", (4, 4, 4), 0.5, family="coupled", uniform=True, snap=False
    )
    assert entry["accepted"]
    assert entry["cold_start_failure"]["reason"] == "injected cold failure"
    assert entry["cold_start_failure"]["history"]
    assert entry["reference_history"]
    assert "failed_pressure_pa" in fields
    assert entry["direct_fft_displacement_error_m"] < 1e-18
    assert "initial_solution" in calls[1]
    assert all("maximum_iterations" not in kwargs for kwargs in calls)
