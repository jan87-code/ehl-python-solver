import numpy as np
import pytest

from ehl_solver.cavitating_newton import solve_cavitating_correction
from ehl_solver.constitutive import IsothermalFluid
from ehl_solver.property_coupled import solve_property_coupling
from ehl_solver.property_load_newton import PropertyLoadSystem
from ehl_solver.property_reynolds import mass_fluxes
from examples.run_property_load_verification import make_problem


@pytest.mark.parametrize("mode", ["material", "local"])
def test_sparse_tangent_against_independent_si_fixed_film_differences(mode):
    problem = make_problem(7, 9)
    fluid = IsothermalFluid(0.04, 850)
    system = PropertyLoadSystem(problem, fluid, 0.1, 0.1, preconditioner=mode)
    rng = np.random.default_rng(574)
    z = np.r_[rng.uniform(0.1, 0.2, system.n), rng.uniform(0.01, 0.05, system.n), 1.0]
    system.sync(z)
    p, t, h, _, _ = system.base.unpack(z[:-1])
    p = p * problem.pressure_scale_pa
    direction = np.zeros(problem.shape)
    v = rng.normal(size=system.n) * 0.01
    direction[1:-1, 1:-1] = (
        v.reshape(problem.shape[0] - 2, problem.shape[1] - 2) * problem.pressure_scale_pa
    )
    film_direction = np.zeros_like(h)
    if mode == "local":
        film_direction = (
            problem.elastic_kernel[problem.shape[0] - 1, problem.shape[1] - 1] * direction
        )

    def evaluate(sign):
        eps = 1e-5
        qx, qy = mass_fluxes(
            problem.x_m,
            problem.y_m,
            h + sign * eps * film_direction,
            p + sign * eps * direction,
            t,
            fluid,
            0.1,
        )
        return (
            system.base.divergence(qx, qy).ravel()
            * problem.length_scale_m
            / system.base.flux_scale
            / 100
        )

    fd = (evaluate(1) - evaluate(-1)) / (2e-5)
    predicted = system.frozen_matrix(z)[: system.n, : system.n] @ v
    np.testing.assert_allclose(predicted, fd, rtol=2e-6, atol=1e-8)
    assert system.frozen_matrix(z).nnz < 30 * system.size


@pytest.fixture(scope="module")
def coupled_case():
    p = make_problem(7, 9)
    f = IsothermalFluid(0.04, 850)
    return p, f, solve_property_coupling(p, f, 0.1)


@pytest.mark.parametrize("mode", ["frozen", "material", "local"])
def test_preconditioner_preserves_complete_bordered_operator_and_correction(mode, coupled_case):
    p, f, seed = coupled_case
    target = 1.05 * seed.load_n
    system = PropertyLoadSystem(p, f, 0.1, target, preconditioner=mode)
    rng = np.random.default_rng(221)
    z = system.pack(seed, p.central_film_offset_m)
    reference = PropertyLoadSystem(p, f, 0.1, target)
    v = rng.normal(size=system.size) * 0.01
    np.testing.assert_array_equal(system.residual(z), reference.residual(z))
    np.testing.assert_array_equal(system.jacobian_vector(z, v), reference.jacobian_vector(z, v))
    matrix = np.column_stack([reference.jacobian_vector(z, v) for v in np.eye(system.size)])
    dense = np.linalg.solve(matrix, -reference.residual(z))
    correction = solve_cavitating_correction(system, z, maximum_krylov_cycles=10, gmres_restart=60)
    assert np.linalg.norm(correction.direction - dense) / np.linalg.norm(dense) < 1e-7
    assert correction.true_linear_residual <= 5e-9
