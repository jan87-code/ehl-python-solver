"""Independent reduced-block algebra and nonsymmetric fixed-cycle checks."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import diags
from scipy.sparse.linalg import gmres

from ehl_solver.constitutive import IsothermalFluid
from ehl_solver.property_coupled import solve_property_coupling
from ehl_solver.property_load_newton import PropertyLoadSystem
from ehl_solver.property_tangent_multigrid import (
    NonsymmetricMultigrid,
    PropertyTangentLU,
    PropertyTangentMultigrid,
)
from ehl_solver.reynolds import assemble_reynolds_system
from examples.benchmark_property_block import build_inverse, measure_solve
from examples.run_property_load_verification import make_problem


@pytest.mark.parametrize("shape", [(17, 25), (33, 49)])
def test_nonsymmetric_galerkin_scaled_identity_linearity_and_known_vector(shape):
    y, x = np.meshgrid(np.linspace(0, 1, shape[0]), np.linspace(0, 1, shape[1]), indexing="ij")
    diffusion, _ = assemble_reynolds_system(1+.2*x+.1*y, .1, .15, 0.)
    n = diffusion.shape[0]
    # Independent nonsymmetric drift, then different signed row/column units.
    drift = diags([np.full(n-1, -.2), np.full(n, .2)], [-1, 0])
    signs = np.where(np.arange(n) % 3 == 0, -1., 1.)
    matrix = diags(signs*np.geomspace(.01, 100., n))@(diffusion+drift)
    matrix = (matrix@diags(np.geomspace(100., .01, n))).tocsr()
    mg = NonsymmetricMultigrid(matrix, shape)
    scaled = diags(mg.row_scale)@matrix@diags(mg.column_scale)
    np.testing.assert_allclose(mg.levels[0]["matrix"].toarray(), scaled.toarray(), atol=1e-12)
    for fine, coarse in zip(mg.levels, mg.levels[1:], strict=False):
        prolong = fine["prolongation"]
        np.testing.assert_allclose(coarse["matrix"].toarray(),
                                   (prolong.T@fine["matrix"]@prolong).toarray(), atol=1e-12)
    rng = np.random.default_rng(42)
    a, b, expected = rng.normal(size=(3, n))
    np.testing.assert_allclose(mg.apply(a+2*b), mg.apply(a)+2*mg.apply(b),
                               rtol=1e-10, atol=1e-10)
    rhs = matrix@expected
    answer, info = gmres(matrix, rhs, M=mg.aslinearoperator(), rtol=1e-12, atol=0,
                         restart=60, maxiter=4)
    assert info == 0
    assert np.linalg.norm(matrix@answer-rhs)/np.linalg.norm(rhs) <= 5e-12
    assert np.linalg.norm(answer-expected)/np.linalg.norm(expected) < 1e-6
    assert mg.diagnostics["coarse_unknowns"] <= 64
    assert mg.diagnostics["factor_payload_bytes"] > 0


@pytest.fixture(scope="module")
def case():
    problem, fluid = make_problem(7, 9), IsothermalFluid(.04, 850.)
    seed = solve_property_coupling(problem, fluid, .1)
    return problem, fluid, seed


@pytest.mark.parametrize("inverse_type", [PropertyTangentMultigrid, PropertyTangentLU])
@pytest.mark.parametrize("mode", ["material", "local"])
@pytest.mark.parametrize("kind", ["full", "cavitated", "front", "islands"])
def test_exact_coarse_inverse_retains_every_mass_coupling_and_active_row(
        case, mode, kind, inverse_type):
    problem, fluid, seed = case
    base = PropertyLoadSystem(problem, fluid, .1, seed.load_n*1.05, preconditioner=mode)
    state = base.pack(seed, problem.central_film_offset_m)
    full = np.ones(base.n, dtype=bool)
    if kind == "cavitated":
        full[:] = False
    elif kind == "front":
        full.reshape(5, 7)[:, 4:] = False
    elif kind == "islands":
        full[::3] = False
    system = base.correction_view(state, full)
    snapshot = state.copy()
    block = inverse_type(system, state)
    dense = system.frozen_matrix(state)[:2*system.n, :2*system.n].toarray()
    # Assemble the mixed matrix by independently selecting original columns.
    columns = [j if full[j] else system.n+j for j in range(system.n)]
    np.testing.assert_array_equal(block.reduced.toarray(), dense[:system.n, columns])
    a, b = np.random.default_rng(12).normal(size=(2, 2*system.n))
    np.testing.assert_allclose(block.apply(a+2*b), block.apply(a)+2*block.apply(b),
                               rtol=1e-10, atol=1e-10)
    answer = block.apply(a)
    expected = np.linalg.solve(dense, a)
    np.testing.assert_allclose(answer, expected, rtol=1e-10, atol=1e-10)
    assert np.linalg.norm(dense@answer-a)/np.linalg.norm(a) <= 1e-11
    np.testing.assert_array_equal(answer[:system.n][~full], a[system.n:][~full])
    np.testing.assert_array_equal(answer[system.n:][full], a[system.n:][full])
    np.testing.assert_array_equal(state, snapshot)


@pytest.mark.parametrize("strategy", ["mg", "reduced"])
@pytest.mark.parametrize("mode", ["material", "local"])
def test_complete_border_inverse_equals_original_sparse_surrogate_and_true_jvp(
        case, mode, strategy):
    problem, fluid, seed = case
    system = PropertyLoadSystem(problem, fluid, .1, seed.load_n*1.05, preconditioner=mode,
                                elasticity_mode="cached")
    state = system.pack(seed, problem.central_film_offset_m)
    inverse, metadata = build_inverse(system, state, mode+"-"+strategy)
    rhs = np.random.default_rng(27).normal(size=system.size)
    expected = np.linalg.solve(system.frozen_matrix(state).toarray(), rhs)
    np.testing.assert_allclose(inverse@rhs, expected, rtol=1e-9, atol=1e-9)
    if strategy == "mg":
        assert metadata["hierarchy"]["coarse_unknowns"] <= 64
    else:
        assert metadata["reduced_unknowns"] == system.n
    dense = np.column_stack([system.jacobian_vector(state, e) for e in np.eye(system.size)])
    # Complete material/elastic/load operator, not the sparse surrogate.
    answer, row = measure_solve(system, state, inverse, -system.residual(state),
                                restart=60, cycles=4)
    assert row["verified_linear_result"] and row["true_relative_residual"] <= 5e-9
    direct = np.linalg.solve(dense, -system.residual(state))
    assert np.linalg.norm(answer-direct)/np.linalg.norm(direct) < 1e-7


def test_unsupported_shapes_singular_and_nonfinite_inputs_fail_explicitly():
    for shape, matrix in [((4, 4), np.eye(3)), ((2, 4), np.eye(1)),
                          ((3, 3), [[np.nan]]), ((3, 3), [[0.]])]:
        with pytest.raises(ValueError):
            NonsymmetricMultigrid(matrix, shape)
    with pytest.raises(ValueError, match="coarsen"):
        NonsymmetricMultigrid(np.eye(100), (12, 12))
    with pytest.raises(RuntimeError):
        NonsymmetricMultigrid(np.ones((9, 9)), (5, 5))
    with pytest.raises(ValueError):
        NonsymmetricMultigrid(np.eye(9), (5, 5), smoothing_steps=0)
    mg = NonsymmetricMultigrid(np.eye(9), (5, 5))
    for rhs in (np.zeros(8), np.full(9, np.inf)):
        with pytest.raises(ValueError, match="finite"):
            mg.apply(rhs)


def test_factor_failure_does_not_fall_back_to_fine_lu(monkeypatch):
    import ehl_solver.property_tangent_multigrid as module

    calls = []

    def fail(*args, **kwargs):
        raise RuntimeError("injected ILU failure")

    monkeypatch.setattr(module, "spilu", fail)
    monkeypatch.setattr(module, "splu", lambda matrix: calls.append(matrix.shape))
    with pytest.raises(RuntimeError, match="injected ILU"):
        NonsymmetricMultigrid(np.eye(81), (11, 11))
    assert calls == []


def test_nonfinite_cycle_and_invalid_tangent_mode_propagate(case):
    mg = NonsymmetricMultigrid(diags(np.ones(81)), (11, 11))
    mg.levels[0]["ilu"] = SimpleNamespace(solve=lambda rhs: np.full_like(rhs, np.nan))
    with pytest.raises(FloatingPointError):
        mg.apply(np.ones(81))
    problem, fluid, seed = case
    system = PropertyLoadSystem(problem, fluid, .1, seed.load_n)
    state = system.pack(seed, problem.central_film_offset_m)
    with pytest.raises(ValueError, match="material or local"):
        PropertyTangentMultigrid(system, state)


@pytest.mark.parametrize("inverse_type", [PropertyTangentMultigrid, PropertyTangentLU])
@pytest.mark.parametrize("kind", ["front", "islands"])
def test_multilevel_mixed_active_transport_known_vector_and_exact_constraints(kind, inverse_type):
    from scipy.sparse import bmat

    shape = (17, 25)
    n, nx = (shape[0]-2)*(shape[1]-2), shape[1]-2
    pressure, _ = assemble_reynolds_system(np.ones(shape), .1, .15, 0.)
    west = np.full(n-1, .5)
    west[np.arange(1, n) % nx == 0] = 0
    content = diags([west, np.full(n, -1.)], [-1, 0], format="csr")
    full = np.ones(n, dtype=bool)
    if kind == "front":
        full.reshape(shape[0]-2, nx)[:, nx//2:] = False
    else:
        full[::3] = False
    matrix = bmat([[pressure, content],
                   [diags((~full).astype(float)), diags(full.astype(float))]], format="csr")
    system = SimpleNamespace(n=n, size=2*n, preconditioner="local",
                             problem=SimpleNamespace(shape=shape),
                             full=lambda state: full, frozen_matrix=lambda state: matrix)
    block = inverse_type(system, None)
    known = np.random.default_rng(389).normal(size=2*n)
    rhs = matrix@known
    answer, info = gmres(matrix, rhs, M=block.aslinearoperator(), rtol=1e-12, atol=0,
                         restart=60, maxiter=4)
    assert info == 0
    assert np.linalg.norm(matrix@answer-rhs)/np.linalg.norm(rhs) < 5e-12
    assert np.linalg.norm(answer-known)/np.linalg.norm(known) < 1e-7
    np.testing.assert_array_equal(block.apply(rhs)[n:][full], rhs[n:][full])
    np.testing.assert_array_equal(block.apply(rhs)[:n][~full], rhs[n:][~full])


def test_material_zero_property_derivatives_recover_frozen_mass_columns(case):
    problem, _, seed = case
    fluid = IsothermalFluid(.04, 850., viscosity_index=0., density_a_per_pa=0.)
    frozen = PropertyLoadSystem(problem, fluid, .1, seed.load_n, preconditioner="frozen")
    material = PropertyLoadSystem(problem, fluid, .1, seed.load_n, preconditioner="material")
    state = material.pack(seed, problem.central_film_offset_m)
    block = PropertyTangentMultigrid(material, state)
    expected = frozen.frozen_matrix(state)[:frozen.n, :2*frozen.n].toarray()
    np.testing.assert_array_equal(block.pressure.toarray(), expected[:, :frozen.n])
    np.testing.assert_array_equal(block.content.toarray(), expected[:, frozen.n:])


@pytest.mark.parametrize("fill", [0., 5., float("nan"), True])
def test_explicit_ilu_fill_control_is_bounded(fill):
    with pytest.raises(ValueError, match="fill factor"):
        NonsymmetricMultigrid(np.eye(9), (5, 5), ilu_fill_factor=fill)


def test_diagnostic_mode_selection_preserves_legacy_defaults(tmp_path):
    from examples import benchmark_property_block as benchmark

    assert benchmark.DEFAULT_MODES == ("frozen", "material", "local", "multigrid")
    assert benchmark.preconditioner_mode("local-mg") == "local"
    assert benchmark.preconditioner_mode("material-mg") == "material"
    for modes in (("local-mg",), ("local", "local"), ("local", "bad")):
        with pytest.raises(ValueError, match="unique supported modes"):
            benchmark.main([], tmp_path / "should_not_exist", 1, 2, 1, modes)
    assert not (tmp_path / "should_not_exist").exists()


def test_failed_hierarchy_is_never_replaced_by_reduced_lu(case, monkeypatch):
    from examples import benchmark_property_block as benchmark

    problem, fluid, seed = case
    system = PropertyLoadSystem(problem, fluid, .1, seed.load_n, preconditioner="local")
    state = system.pack(seed, problem.central_film_offset_m)

    def fail(*args):
        raise RuntimeError("injected hierarchy failure")

    def forbidden(*args):
        raise AssertionError("silent reduced LU fallback")

    monkeypatch.setattr(benchmark, "PropertyTangentMultigrid", fail)
    monkeypatch.setattr(benchmark, "PropertyTangentLU", forbidden)
    with pytest.raises(RuntimeError, match="injected hierarchy"):
        build_inverse(system, state, "local-mg")
