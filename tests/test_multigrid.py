import numpy as np
import pytest
from scipy.sparse.linalg import cg, spsolve

from ehl_solver.multigrid import ReynoldsMultigrid, linear_prolongation
from ehl_solver.reynolds import solve_reynolds_pressure
from examples.benchmark_multigrid import known_pressure_check


def test_prolongation_zero_boundaries_and_exact_linear_interpolation():
    p = linear_prolongation(9)
    coarse = np.array([1., 3., 2.])
    expected = np.interp(np.arange(9), np.arange(0, 9, 2), np.r_[0., coarse, 0.])
    np.testing.assert_array_equal(p@coarse, expected[1:-1])
    with pytest.raises(ValueError):
        linear_prolongation(8)


@pytest.mark.parametrize("shape", [(17, 25), (33, 49)])
def test_galerkin_identity_linear_symmetric_positive_cycle_and_direct_solution(shape):
    y, x = np.meshgrid(np.linspace(0, 1, shape[0]), np.linspace(0, 1, shape[1]), indexing="ij")
    mg = ReynoldsMultigrid(1+.4*x+.2*np.sin(np.pi*y), .1, .15)
    rng = np.random.default_rng(33)
    a, b = rng.normal(size=(2, mg.matrix.shape[0]))
    for fine, coarse in zip(mg.levels, mg.levels[1:], strict=False):
        p = fine["prolongation"]
        np.testing.assert_allclose(coarse["matrix"].toarray(),
                                   (p.T@fine["matrix"]@p).toarray(), rtol=1e-14, atol=1e-12)
    np.testing.assert_allclose(mg.apply(a+2*b), mg.apply(a)+2*mg.apply(b), rtol=2e-12, atol=1e-13)
    assert a@mg.apply(b) == pytest.approx(b@mg.apply(a), rel=1e-12, abs=1e-12)
    assert a@mg.apply(a) > 0
    answer, info = cg(mg.matrix, a, M=mg.aslinearoperator(), rtol=1e-11, atol=0, maxiter=100)
    assert info == 0
    assert np.linalg.norm(mg.matrix@answer-a)/np.linalg.norm(a) <= 1e-11
    np.testing.assert_allclose(answer, spsolve(mg.matrix, a), rtol=1e-9, atol=1e-12)
    assert mg.diagnostics["coarse_unknowns"] <= 64


def test_smooth_error_coarse_correction_and_mesh_independent_iterations():
    counts = []
    for n in (33, 65, 129):
        x = np.linspace(0, 1, n)
        xx, yy = np.meshgrid(x, x)
        mg = ReynoldsMultigrid(np.ones_like(xx), x[1], x[1])
        exact = (np.sin(np.pi*xx)*np.sin(np.pi*yy))[1:-1, 1:-1].ravel()
        rhs = mg.matrix@exact
        after = exact-mg.apply(rhs)
        assert after@(mg.matrix@after) < .1*(exact@(mg.matrix@exact))
        count = []
        # Mixed modes avoid the single-eigenmode one-step CG special case.
        rhs = np.random.default_rng(123).normal(size=exact.size)
        answer, info = cg(mg.matrix, rhs, M=mg.aslinearoperator(), rtol=1e-10, atol=0,
                          maxiter=100, callback=lambda a, count=count: count.append(1))
        assert info == 0
        assert np.linalg.norm(mg.matrix@answer-rhs)/np.linalg.norm(rhs) <= 1e-10
        counts.append(len(count))
    assert max(counts) <= 20
    assert max(counts)-min(counts) <= 5


def test_public_prescribed_film_solver_parity_and_failure():
    x = np.linspace(0, 1, 33)
    xx, yy = np.meshgrid(x, x)
    h = 1.4-.4*xx+.05*np.sin(np.pi*yy)
    reference = solve_reynolds_pressure(h, x[1], x[1], 2., rtol=1e-11)
    accelerated = solve_reynolds_pressure(h, x[1], x[1], 2., rtol=1e-11,
                                         preconditioner="multigrid")
    np.testing.assert_allclose(accelerated.pressure, reference.pressure, rtol=1e-8, atol=1e-11)
    assert accelerated.iterations < reference.iterations
    assert accelerated.algebraic_residual_relative_l2 <= 1e-11
    assert np.all(accelerated.pressure[[0, -1]] == 0)
    assert np.all(accelerated.pressure[:, [0, -1]] == 0)
    with pytest.raises(RuntimeError, match="did not converge"):
        solve_reynolds_pressure(h, x[1], x[1], 2., rtol=1e-13, max_iterations=1,
                                preconditioner="multigrid")


def test_unsupported_hierarchy_and_inputs_fail_explicitly():
    with pytest.raises(ValueError, match="cannot coarsen"):
        ReynoldsMultigrid(np.ones((32, 32)), .1, .1)
    with pytest.raises(ValueError):
        ReynoldsMultigrid(np.ones((9, 9)), .1, .1, smoothing_steps=0)
    mg = ReynoldsMultigrid(np.ones((9, 9)), .1, .1)
    with pytest.raises(ValueError):
        mg.apply(np.array([np.nan]))


def test_variable_coefficient_known_pressure_forward_error():
    x = np.linspace(0, 1, 33)
    xx, yy = np.meshgrid(x, x)
    film = np.exp(3*np.sin(np.pi*xx)*np.sin(np.pi*yy))
    result = known_pressure_check(film, x[1], x[1])
    assert result["passed"]


def test_analytic_manufactured_source_retains_second_order_grid_accuracy():
    errors = []
    for n in (17, 33, 65):
        x = np.linspace(0, 1, n)
        xx, yy = np.meshgrid(x, x)
        exact = np.sin(np.pi*xx)*np.sin(np.pi*yy)
        h = 1+.2*xx
        source = (.6*h**2*np.pi*np.cos(np.pi*xx)*np.sin(np.pi*yy)
                  - 2*np.pi**2*h**3*exact-.4)
        result = solve_reynolds_pressure(h, x[1], x[1], 2., source,
                                         preconditioner="multigrid", rtol=1e-12)
        errors.append(np.linalg.norm(result.pressure-exact)/np.linalg.norm(exact))
        assert result.algebraic_residual_relative_l2 <= 1e-12
        assert result.mass_balance_relative_error < 1e-10
    for old, new in zip(errors, errors[1:], strict=False):
        assert 1.95 < np.log2(old/new) < 2.05
