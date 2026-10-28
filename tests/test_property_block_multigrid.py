"""Independent active-block algebra and complete material/elastic products."""

import numpy as np
import pytest
from scipy.sparse.linalg import LinearOperator, cg, gmres

from ehl_solver.constitutive import IsothermalFluid
from ehl_solver.multigrid import ReynoldsMultigrid
from ehl_solver.property_block_multigrid import PropertyBlockMultigrid
from ehl_solver.property_coupled import solve_property_coupling
from ehl_solver.property_load_newton import PropertyLoadSystem
from examples.run_property_load_verification import make_problem


@pytest.mark.parametrize("kind", ["full", "cavitated", "front", "islands"])
def test_masked_hierarchy_spd_galerkin_and_known_pressure(kind):
    shape = (17, 25)
    y, x = np.meshgrid(np.linspace(0, 1, shape[0]), np.linspace(0, 1, shape[1]), indexing="ij")
    active = np.ones((15, 23), dtype=bool)
    if kind == "cavitated":
        active[:] = False
    elif kind == "front":
        active[:, 12:] = False
    elif kind == "islands":
        active[::2, ::2] = False
    mg = ReynoldsMultigrid(1+.3*x+.1*y, .1, .15, pressure_active=active)
    matrix = mg.matrix.toarray()
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-12)
    assert np.linalg.eigvalsh(matrix)[0] > 0
    for fine, coarse in zip(mg.levels, mg.levels[1:], strict=False):
        prolong = fine["prolongation"]
        np.testing.assert_allclose(coarse["matrix"].toarray(),
                                   (prolong.T@fine["matrix"]@prolong).toarray(), atol=1e-12)
    expected = np.random.default_rng(23).normal(size=matrix.shape[0])*active.ravel()
    rhs = matrix@expected
    answer, info = cg(mg.matrix, rhs, M=mg.aslinearoperator(), rtol=1e-11, atol=0, maxiter=100)
    assert info == 0
    np.testing.assert_allclose(answer, expected, rtol=1e-8, atol=1e-9)
    assert np.linalg.norm(matrix@answer-rhs) <= 1e-10*max(np.linalg.norm(rhs), 1.)


@pytest.fixture(scope="module")
def case():
    problem = make_problem(9, 13)
    fluid = IsothermalFluid(.04, 850.)
    seed = solve_property_coupling(problem, fluid, .1)
    system = PropertyLoadSystem(problem, fluid, .1, seed.load_n*1.05)
    return system, system.pack(seed, problem.central_film_offset_m)


@pytest.mark.parametrize("kind", ["full", "cavitated", "front", "islands"])
def test_block_action_linearity_constraints_and_approximate_inverse(case, kind):
    base, z = case
    full = np.ones(base.n, dtype=bool)
    if kind == "cavitated":
        full[:] = False
    elif kind == "front":
        full.reshape(7, 11)[:, 6:] = False
    elif kind == "islands":
        full[::3] = False
    system = base.correction_view(z, full)
    block = PropertyBlockMultigrid(system, z)
    rng = np.random.default_rng(32)
    a, b = rng.normal(size=(2, 2*base.n))
    np.testing.assert_allclose(block.apply(a+2*b), block.apply(a)+2*block.apply(b),
                               rtol=1e-11, atol=1e-11)
    answer = block.apply(a)
    np.testing.assert_array_equal(answer[:base.n][~full], a[base.n:][~full])
    np.testing.assert_array_equal(answer[base.n:][full], a[base.n:][full])
    # Independent dense lower-block surrogate: omit D_cav -> mass_full coupling.
    matrix = system.frozen_matrix(z)[:2*base.n, :2*base.n].toarray()
    matrix[:base.n, base.n:] *= ~(full[:, None] & (~full)[None, :])
    expected = np.linalg.solve(matrix, a)
    # A V-cycle is an approximation; evaluate its pressure and mass defects,
    # while the cavitated mass and complementarity equations must be exact.
    defect = matrix@answer-a
    np.testing.assert_allclose(defect[base.n:], 0, atol=1e-12)
    np.testing.assert_allclose(defect[:base.n][~full], 0, atol=2e-12)
    if kind == "cavitated":
        np.testing.assert_allclose(answer, expected, rtol=1e-11, atol=1e-11)
    else:
        before = a[:base.n]-block.pressure@np.where(~full, a[base.n:], 0.)
        before -= block.content@np.where(full, a[base.n:], 0.)
        assert np.linalg.norm(defect[:base.n][full]) < np.linalg.norm(before[full])


def test_exact_coarse_block_inverse_matches_dense_surrogate(case):
    # Small physical grid makes the projected pressure solve exact Cholesky.
    problem = make_problem(7, 9)
    fluid = IsothermalFluid(.04, 850.)
    seed = solve_property_coupling(problem, fluid, .1)
    system = PropertyLoadSystem(problem, fluid, .1, seed.load_n)
    z = system.pack(seed, problem.central_film_offset_m)
    block = PropertyBlockMultigrid(system, z)
    full = block.full
    active_matrix = block.multigrid.matrix.toarray()[np.ix_(full, full)]
    np.testing.assert_allclose(active_matrix, block.pressure.toarray()[np.ix_(full, full)],
                               rtol=1e-13, atol=1e-13)
    matrix = system.frozen_matrix(z)[:2*system.n, :2*system.n].toarray()
    matrix[:system.n, system.n:] *= ~(full[:, None] & (~full)[None, :])
    rhs = np.random.default_rng(90).normal(size=2*system.n)
    np.testing.assert_allclose(block.apply(rhs), np.linalg.solve(matrix, rhs),
                               rtol=1e-10, atol=1e-10)


def test_complete_material_jvp_schur_and_dense_bordered_correction(case):
    system, z = case
    block = PropertyBlockMultigrid(system, z)
    size = 2*system.n
    # Independent complete dense analytic Jacobian, including force and h0.
    dense = np.column_stack([system.jacobian_vector(z, e) for e in np.eye(system.size)])
    direction = np.random.default_rng(44).normal(size=system.size)*1e-4
    smooth = z.copy()
    smooth[:system.n] = .001
    smooth[system.n:2*system.n] = .02
    step = 1e-5
    fd = (system.residual(smooth+step*direction)-system.residual(smooth-step*direction))/(2*step)
    np.testing.assert_allclose(system.jacobian_vector(smooth, direction), fd,
                               rtol=3e-6, atol=1e-9)
    residual = system.residual(z)
    operator = LinearOperator((size, size),
                              matvec=lambda v: system.jacobian_vector(z, np.r_[v, 0.])[:size])
    answers = []
    for rhs in (-residual[:size], dense[:size, -1]):
        answer, info = gmres(operator, rhs, M=block.aslinearoperator(), rtol=1e-11,
                             atol=0, restart=80, maxiter=20)
        assert info == 0
        assert np.linalg.norm(operator@answer-rhs) <= 5e-11*np.linalg.norm(rhs)
        answers.append(answer)
    pressure, response = answers
    load = dense[-1, :size]
    offset = (-residual[-1]-load@pressure)/(-load@response)
    reconstructed = np.r_[pressure-response*offset, offset]
    np.testing.assert_allclose(reconstructed, np.linalg.solve(dense, -residual),
                               rtol=1e-7, atol=1e-10)
    assert np.linalg.norm(dense@reconstructed+residual)/np.linalg.norm(residual) < 5e-9
    # Independently exercise full bordered GMRES, without a Schur solve of
    # the true operator. The approximate inverse retains the full force row.
    approximate_response = block.apply(dense[:size, -1])
    schur = -load@approximate_response
    assert abs(schur) > 1e-12

    def bordered_inverse(rhs):
        fields = block.apply(rhs[:size])
        dh0 = (rhs[-1]-load@fields)/schur
        return np.r_[fields-approximate_response*dh0, dh0]

    bordered = LinearOperator(dense.shape, matvec=lambda v: system.jacobian_vector(z, v))
    inverse = LinearOperator(dense.shape, matvec=bordered_inverse)
    answer, info = gmres(bordered, -residual, M=inverse, rtol=1e-11, atol=0,
                         restart=80, maxiter=20)
    assert info == 0
    assert np.linalg.norm(dense@answer+residual)/np.linalg.norm(residual) < 5e-9
    np.testing.assert_allclose(answer, reconstructed, rtol=1e-7, atol=1e-10)


def test_invalid_masks_rhs_and_unsupported_preconditioner(case):
    with pytest.raises(ValueError, match="boolean interior"):
        ReynoldsMultigrid(np.ones((9, 9)), .1, .1, pressure_active=np.ones((7, 7)))
    system, z = case
    block = PropertyBlockMultigrid(system, z)
    for rhs in (np.zeros(2*system.n-1), np.full(2*system.n, np.nan)):
        with pytest.raises(ValueError, match="finite"):
            block.apply(rhs)
    view = system.correction_view(z, system.full(z))
    view.preconditioner = "local"
    with pytest.raises(ValueError, match="frozen"):
        PropertyBlockMultigrid(view, z)


@pytest.mark.parametrize("failure", ["zero_diagonal", "upper_transport"])
def test_unsupported_transport_fails_explicitly(case, monkeypatch, failure):
    system, z = case
    # Force several cavitated unknowns; no physical state or flux is changed.
    view = system.correction_view(z, np.zeros(system.n, dtype=bool))
    matrix = view.frozen_matrix(z).tolil()
    if failure == "zero_diagonal":
        matrix[0, system.n] = 0.
    else:
        matrix[0, system.n+1] = 1.
    monkeypatch.setattr(view, "frozen_matrix", lambda state: matrix.tocsc())
    with pytest.raises(ValueError, match="transport"):
        PropertyBlockMultigrid(view, z)
