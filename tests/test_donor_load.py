import numpy as np
import pytest

from ehl_solver.cavitating_newton import solve_cavitating_correction
from ehl_solver.constitutive import IsothermalFluid
from ehl_solver.donor_load import DonorLoadSystem, donor_mass_audit
from ehl_solver.property_coupled import solve_property_coupling
from ehl_solver.property_load_newton import PropertyLoadSystem, _solve_property_load_state
from ehl_solver.property_reynolds import admissible
from ehl_solver.property_warm import PropertyWarmSystem
from ehl_solver.transport_candidate import donor_fluxes
from examples.run_property_load_verification import make_problem


@pytest.fixture(scope="module")
def case():
    problem, fluid = make_problem(7, 9), IsothermalFluid(0.04, 850.0)
    seed = solve_property_coupling(problem, fluid, 0.1)
    return problem, fluid, seed


def test_complete_donor_elastic_material_offset_jvp_and_bordered_correction(case):
    problem, fluid, seed = case
    system = DonorLoadSystem(problem, fluid, 0.1, seed.load_n * 1.05, preconditioner="local")
    z = system.pack(seed, problem.central_film_offset_m)
    smooth = z.copy()
    rng = np.random.default_rng(19)
    smooth[: system.n] = rng.uniform(0.001, 0.002, system.n)
    smooth[system.n : 2 * system.n] = rng.uniform(0.02, 0.05, system.n)
    direction = rng.normal(size=system.size) * 0.001
    epsilon = 1e-5
    fd = (
        system.residual(smooth + epsilon * direction)
        - system.residual(smooth - epsilon * direction)
    ) / (2 * epsilon)
    np.testing.assert_allclose(system.jacobian_vector(smooth, direction), fd, rtol=3e-6, atol=2e-9)
    matrix = np.column_stack([system.jacobian_vector(z, v) for v in np.eye(system.size)])
    expected = np.linalg.solve(matrix, -system.residual(z))
    for tolerance in (1e-10, 2.0):
        correction = solve_cavitating_correction(system, z, schur_tolerance=tolerance)
        assert correction.true_linear_residual <= 5e-9
        assert correction.used_bordered_fallback == (tolerance == 2.0)
        np.testing.assert_allclose(correction.direction, expected, rtol=2e-6, atol=1e-9)
    original = PropertyLoadSystem(problem, fluid, 0.1, seed.load_n)
    assert type(original.base) is PropertyWarmSystem


@pytest.mark.parametrize("mode", ["material", "local"])
def test_donor_preconditioner_matches_frozen_or_self_film_finite_differences(case, mode):
    problem, fluid, seed = case
    system = DonorLoadSystem(problem, fluid, 0.1, seed.load_n, preconditioner=mode)
    z = system.pack(seed, problem.central_film_offset_m)
    z[: system.n] = 0.002
    z[system.n : 2 * system.n] = 0.04
    p, t, h, _, _ = system.base.unpack(z[:-1])
    p = p * problem.pressure_scale_pa
    rng = np.random.default_rng(17)
    direction = rng.normal(size=system.size) * 0.001
    direction[-1] = 0
    dp, dt = np.zeros(problem.shape), np.zeros(problem.shape)
    dp[1:-1, 1:-1] = direction[: system.n].reshape(dp[1:-1, 1:-1].shape) * problem.pressure_scale_pa
    dt[1:-1, 1:-1] = -direction[system.n : 2 * system.n].reshape(dt[1:-1, 1:-1].shape)
    compliance = (
        problem.elastic_kernel[problem.shape[0] - 1, problem.shape[1] - 1]
        if mode == "local"
        else 0.0
    )
    dh = compliance * dp
    eps = 1e-5
    fluxes = []
    for sign in (1, -1):
        qx, qy = donor_fluxes(
            problem.x_m,
            problem.y_m,
            h + sign * eps * dh,
            p + sign * eps * dp,
            t + sign * eps * dt,
            fluid,
            0.1,
        )
        fluxes.append(
            system.base.divergence(qx, qy)
            * problem.length_scale_m
            / system.base.flux_scale
            / system.base.mass_scale
        )
    expected = (fluxes[0] - fluxes[1]).ravel() / (2 * eps)
    matrix = system.frozen_matrix(z)
    np.testing.assert_allclose((matrix @ direction)[: system.n], expected, rtol=3e-6, atol=2e-9)
    offset = np.zeros(system.size)
    offset[-1] = 1
    np.testing.assert_allclose(
        matrix @ offset, system.jacobian_vector(z, offset), rtol=1e-14, atol=1e-14
    )


def test_donor_audit_telescopes_on_nonuniform_grid_and_has_no_projection():
    x, y = np.array([0.0, 0.1, 0.3, 0.6]), np.array([0.0, 0.2, 0.5, 1.0, 1.2])
    rng = np.random.default_rng(25)
    h = 1e-6 * (1 + rng.random((5, 4)))
    p, theta = 1e7 * rng.random(h.shape), rng.random(h.shape)
    fluid = IsothermalFluid(0.04, 850.0)
    audit = donor_mass_audit(
        x, y, h, p, theta, fluid, 2.0, length_scale_m=0.1, film_scale_m=1e-6, pressure_scale_pa=1e7
    )
    assert audit["net_mass_flow_kg_s"] == pytest.approx(
        audit["summed_cell_mass_defect_kg_s"], abs=1e-18
    )
    qx, qy = donor_fluxes(x, y, h, p, theta, fluid, 2.0)
    wx, wy = 0.5 * (x[2:] - x[:-2]), 0.5 * (y[2:] - y[:-2])
    independent = (qx[1:-1, -1] - qx[1:-1, 0]) @ wy + (qy[-1, 1:-1] - qy[0, 1:-1]) @ wx
    assert audit["net_mass_flow_kg_s"] == pytest.approx(independent, abs=1e-18)


def test_real_donor_newton_repair_original_gates_and_histories(case):
    problem, fluid, seed = case
    system = DonorLoadSystem(problem, fluid, 0.1, seed.load_n * 1.05, preconditioner="local")
    z = system.pack(seed, problem.central_film_offset_m)
    result = _solve_property_load_state(
        system, z, 30, 10, 1e-10, True, active_lookahead=True, content_bound_pivots=True
    )
    s = result.solution
    assert admissible(s.audit, s.pressure_pa, s.fractional_film_content, problem.pressure_scale_pa)
    assert abs(s.load_n / (seed.load_n * 1.05) - 1) <= 1e-8
    assert s.film_thickness_m.min() > problem.film_floor_m
    for record in s.history:
        accepted = [(m, t) for m in record["models"] for t in m["trials"] if t["accepted"]]
        assert len(accepted) == 1
        m, t = accepted[0]
        assert m["slope"] < 0 and t["merit"] <= record["merit"] + 1e-4 * t["alpha"] * m["slope"]


def test_donor_and_reference_nyquist_capacity_response():
    # Benchmark before coupled-root comparisons: averaging erases a two-cell
    # full-film capacity signal; conservative donor transport does not.
    from ehl_solver.property_reynolds import mass_fluxes
    from ehl_solver.transport_candidate import face_divergence

    x, y = np.arange(33) * 1e-5, np.arange(5) * 1e-5
    p = np.zeros((5, 33))
    h = np.broadcast_to(1e-6 * (1 + 0.1 * (-1.0) ** np.arange(33)), p.shape)
    theta = np.ones_like(p)
    fluid = IsothermalFluid(0.04, 850.0)
    centered = face_divergence(x, y, *mass_fluxes(x, y, h, p, theta, fluid, 2.0))
    donor = face_divergence(x, y, *donor_fluxes(x, y, h, p, theta, fluid, 2.0))
    assert np.max(abs(centered)) < 1e-12
    np.testing.assert_allclose(abs(donor), 2 * 2 * 850.0 * 0.1e-6 / 1e-5, rtol=1e-14)


@pytest.mark.parametrize("mode", ["frozen", "material", "local"])
def test_donor_content_block_and_active_override(case, mode):
    problem, fluid, seed = case
    system = DonorLoadSystem(problem, fluid, 0.1, seed.load_n, preconditioner=mode)
    z = system.pack(seed, problem.central_film_offset_m)
    z[: system.n] = 0.002
    z[system.n : 2 * system.n] = 0.04
    rng = np.random.default_rng(32)
    v = np.r_[np.zeros(system.n), rng.normal(size=system.n) * 0.01, 0.0]
    # Content changes do not alter fluid/film coefficients, even for frozen
    # preconditioning. Boundary content directions are fixed by the supply.
    np.testing.assert_allclose(
        system.frozen_matrix(z) @ v, system.jacobian_vector(z, v), rtol=2e-12, atol=1e-12
    )
    selected = np.ones(system.n, dtype=bool)
    selected[::3] = False
    view = system.correction_view(z, selected)
    result = view.jacobian_vector(z, v)
    np.testing.assert_array_equal(
        result[system.n : 2 * system.n],
        np.where(selected, v[system.n : 2 * system.n], v[: system.n]),
    )
    dense = np.column_stack([view.jacobian_vector(z, e) for e in np.eye(system.size)])
    answer = solve_cavitating_correction(view, z)
    np.testing.assert_allclose(dense @ answer.direction, -view.residual(z), rtol=2e-7, atol=1e-10)
    assert answer.true_linear_residual <= 5e-9


def test_donor_root_zero_budget_rejection_and_strict_film_floor(case):
    from ehl_solver.property_load_newton import PropertyLoadNewtonError

    problem, fluid, seed = case
    system = DonorLoadSystem(problem, fluid, 0.1, seed.load_n * 1.05, preconditioner="local")
    z = system.pack(seed, problem.central_film_offset_m)
    with pytest.raises(PropertyLoadNewtonError) as exc:
        _solve_property_load_state(system, z, 0, 10, 1e-10, True)
    assert exc.value.last_root is not None
    assert "budget" in str(exc.value)
    invalid = z.copy()
    invalid[-1] = -1e6
    with pytest.raises(ValueError, match="film"):
        system.residual(invalid)
