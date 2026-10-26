"""Independent dense/SciPy verification; optional torch CPU and CUDA execution."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.fft import dst
from scipy.integrate import trapezoid
from scipy.signal import convolve2d
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import gmres

from ehl_solver.backend import make_backend
from ehl_solver.device import DeviceFullFilmEvaluator
from ehl_solver.device_bordered import DeviceBorderedError, solve_device_bordered
from ehl_solver.device_krylov import device_gmres
from ehl_solver.device_newton import (
    DeviceContinuationError,
    DeviceNewtonError,
    continue_device_load,
    device_line_search,
    solve_device_full_film_load,
)
from ehl_solver.device_preconditioner import DeviceFrozenFilmPreconditioner, _dst1
from ehl_solver.elasticity import cell_integrated_kernel
from ehl_solver.monolithic import (
    BorderedReynoldsElasticityProblem,
    _armijo_line_search,
    continue_monolithic_load,
    solve_bordered_correction,
    solve_monolithic_full_film_load,
)
from ehl_solver.reynolds import assemble_reynolds_system
from examples.benchmark_device_krylov import case


@pytest.fixture(params=["cpu", "cuda"])
def torch_device(request):
    torch = pytest.importorskip("torch")
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("No CUDA device")
    return torch, request.param


@pytest.mark.parametrize("restart", [4, 30])
def test_nonsymmetric_restarted_dense_and_scipy(torch_device, restart):
    torch, device = torch_device
    rng = np.random.default_rng(309)
    a = rng.normal(size=(23, 23)) * .15 + np.diag(np.linspace(.5, 3, 23))
    b = rng.normal(size=23)
    ad, bd = (torch.as_tensor(t, dtype=torch.float64, device=device) for t in (a, b))
    inverse = 1 / torch.diag(ad)
    result = device_gmres(lambda x: ad @ x, bd, preconditioner=lambda x: inverse * x,
                          restart=restart, max_iterations=150, rtol=1e-11)
    assert result.converged, result.reason
    x = result.solution.cpu().numpy()
    reference, info = gmres(a, b, rtol=1e-11, restart=restart, maxiter=150)
    assert info == 0
    np.testing.assert_allclose(x, np.linalg.solve(a, b), rtol=1e-9, atol=2e-10)
    np.testing.assert_allclose(x, reference, rtol=1e-9, atol=2e-10)
    true = np.linalg.norm(a @ x - b)
    assert true / np.linalg.norm(b) < 1e-11
    assert result.true_residual_norm == pytest.approx(true, abs=2e-14)
    assert result.iterations <= 150


def test_exact_initial_zero_rhs_and_happy_breakdown(torch_device):
    torch, device = torch_device
    b = torch.arange(1., 8., dtype=torch.float64, device=device)
    def op(x):
        return 2 * x
    result = device_gmres(op, b, rtol=1e-13)
    assert result.converged and result.iterations == 1
    assert device_gmres(op, b, initial=b/2).iterations == 0
    assert device_gmres(op, torch.zeros_like(b)).converged
    absolute = device_gmres(op, b * 1e-20, rtol=0., atol=1e-15)
    assert absolute.converged and absolute.iterations == 0


def test_failures_and_true_residual_not_preconditioned_surrogate(torch_device):
    torch, device = torch_device
    b = torch.ones(9, dtype=torch.float64, device=device)
    a = torch.diag(torch.arange(1., 10., dtype=torch.float64, device=device))
    limited = device_gmres(lambda x: a @ x, b, max_iterations=2, restart=2, rtol=1e-13)
    assert not limited.converged and limited.reason == "iteration_limit"
    assert limited.iterations == 2
    for precond, reason in ((lambda x: x*0, "preconditioner_breakdown"),
                           (lambda x: x*float("nan"), "nonfinite_preconditioner")):
        result = device_gmres(lambda x: x, b, preconditioner=precond)
        assert not result.converged and result.reason == reason
    singular = device_gmres(lambda x: x*0, b)
    assert not singular.converged and singular.reason == "arnoldi_breakdown"
    tiny_scaled = device_gmres(lambda x: a @ x, b, preconditioner=lambda x: x*1e-20,
                               rtol=1e-12, max_iterations=20)
    assert tiny_scaled.converged
    assert tiny_scaled.true_relative_residual < 1e-12
    with pytest.raises(MemoryError):
        device_gmres(lambda x: x, b, memory_limit_bytes=1)
    with pytest.raises(ValueError):
        device_gmres(lambda x: x.float(), b)
    with pytest.raises(ValueError):
        device_gmres(lambda x: x, b, restart=0)


def test_resident_basis_no_exports_and_runtime_error_not_hidden(torch_device, monkeypatch):
    torch, device = torch_device
    b = torch.ones(5, dtype=torch.float64, device=device)

    def forbidden(*args, **kwargs):
        raise AssertionError("host export attempted")

    monkeypatch.setattr(torch.Tensor, "cpu", forbidden)
    monkeypatch.setattr(torch.Tensor, "numpy", forbidden)
    assert device_gmres(lambda x: x, b).converged

    def broken(x):
        raise RuntimeError("device calculation failed")

    with pytest.raises(RuntimeError, match="device calculation failed"):
        device_gmres(broken, b)


def problem_case(torch_device, compliance=1., e_star=1e8):
    torch, device = torch_device
    problem = BorderedReynoldsElasticityProblem(
        np.linspace(-.5e-3, .75e-3, 9), np.linspace(-.5e-3, .5e-3, 7),
        .01, .02, e_star, 5e-4, 20e-6, 21250., 12., -.001,
        compliance_factor=compliance, reynolds_residual_scale=100.,
    )
    rng = np.random.default_rng(390)
    z = np.r_[rng.normal(size=problem.pressure_size) * .02, 1.]
    evaluator = DeviceFullFilmEvaluator(problem, make_backend("torch", device))
    return problem, evaluator, z, torch.as_tensor(z, dtype=torch.float64, device=device)


def test_dst_and_frozen_matrix_and_exact_surrogate(torch_device):
    torch, device = torch_device
    problem, evaluator, z, zd = problem_case(torch_device)
    inverse = DeviceFrozenFilmPreconditioner(evaluator, zd)
    rng = np.random.default_rng(8)
    values = rng.normal(size=(5, 7))
    vd = torch.as_tensor(values, dtype=torch.float64, device=device)
    for axis in (0, 1):
        np.testing.assert_allclose(_dst1(vd, axis).cpu().numpy(), dst(values, type=1, axis=axis),
                                   rtol=3e-13, atol=2e-14)
    _, film, _ = problem.state_fields(z[:-1], z[-1])
    a, _ = assemble_reynolds_system(film/20e-6, problem.dx, problem.dy, 12.)
    a = -a / problem.reynolds_residual_scale
    b = values.ravel()
    np.testing.assert_allclose(inverse.diagonal.cpu().numpy(), a.diagonal(), rtol=3e-14)
    np.testing.assert_allclose(inverse.matrix_vector(vd.ravel()).cpu().numpy(), a @ b,
                               rtol=3e-13, atol=3e-12)
    # Construct the surrogate independently, without FFTs or face conductances.
    def laplacian(n, spacing):
        return diags([-np.ones(n-1), 2*np.ones(n), -np.ones(n-1)], [-1, 0, 1]) / spacing**2

    lap = kron(eye(5), laplacian(7, problem.dx)) + kron(laplacian(5, problem.dy), eye(7))
    s = diags((film[1:-1, 1:-1]/20e-6).ravel()**1.5)
    surrogate = (-s @ lap @ s / 100.).toarray()
    observed = inverse(vd.ravel()).cpu().numpy()
    np.testing.assert_allclose(observed, np.linalg.solve(surrogate, b), rtol=2e-12, atol=1e-13)
    jacobi = DeviceFrozenFilmPreconditioner(evaluator, zd, strategy="jacobi")
    np.testing.assert_allclose(jacobi(vd.ravel()).cpu().numpy(), b/a.diagonal(), rtol=3e-13)
    with pytest.raises(ValueError, match="ILU"):
        DeviceFrozenFilmPreconditioner(evaluator, zd, strategy="ilu")


@pytest.mark.parametrize("compliance,e_star", [(0., 1e8), (1., 1e8), (1., 1e7), (.15, 1e6)])
def test_bordered_dense_cpu_and_independent_directional_residual(torch_device, compliance, e_star):
    torch, device = torch_device
    problem, evaluator, z, zd = problem_case(torch_device, compliance, e_star)
    result = solve_device_bordered(evaluator, zd, rtol=1e-10, restart=40, max_iterations=160)
    dense = np.column_stack([problem.jacobian_vector(z, v) for v in np.eye(z.size)])
    expected = np.linalg.solve(dense, -problem.residual(z))
    observed = result.direction.cpu().numpy()
    np.testing.assert_allclose(observed, expected, rtol=2e-8, atol=2e-9)
    cpu = solve_bordered_correction(problem, z, krylov_rtol=1e-10)
    cpu_direction = np.r_[cpu.pressure_correction.ravel(), cpu.h0_correction_m/20e-6]
    np.testing.assert_allclose(observed, cpu_direction, rtol=3e-8, atol=3e-9)
    frozen_jvp = evaluator.linearize(zd)
    np.testing.assert_allclose(frozen_jvp(result.direction).cpu().numpy(),
                               dense @ observed, rtol=3e-12, atol=3e-12)
    step = 1e-5 / max(np.linalg.norm(observed), 1.)
    derivative = (
        problem.residual(z + step*observed) - problem.residual(z - step*observed)
    ) / (2*step)
    residual = problem.residual(z)
    assert np.linalg.norm(derivative + residual)/np.linalg.norm(residual) < 1e-7
    assert result.bordered_residual_relative < 1e-7
    assert all(record.converged and record.true_relative_residual < 1e-10
               for _, record in result.diagnostics)


def test_schur_recovery_krylov_failure_and_residency(torch_device, monkeypatch):
    torch, device = torch_device
    problem, evaluator, z, zd = problem_case(torch_device)
    fallback = solve_device_bordered(evaluator, zd, rtol=1e-10, schur_tolerance=2.)
    assert fallback.used_fallback
    assert fallback.diagnostics[-1][0] == "full_bordered_fallback"
    dense = np.column_stack([problem.jacobian_vector(z, v) for v in np.eye(z.size)])
    np.testing.assert_allclose(fallback.direction.cpu().numpy(),
                               np.linalg.solve(dense, -problem.residual(z)),
                               rtol=3e-8, atol=3e-9)
    with pytest.raises(DeviceBorderedError) as caught:
        solve_device_bordered(evaluator, zd, rtol=1e-12, max_iterations=1)
    assert len(caught.value.diagnostics) == 2
    assert all(not record.converged for _, record in caught.value.diagnostics)

    def forbidden(*args, **kwargs):
        raise AssertionError("host field export attempted")

    monkeypatch.setattr(torch.Tensor, "cpu", forbidden)
    monkeypatch.setattr(torch.Tensor, "numpy", forbidden)
    resident = solve_device_bordered(evaluator, zd, rtol=1e-9)
    assert resident.direction.device == zd.device


@pytest.mark.parametrize("compliance,e_star", [(0., 1e8), (1., 1e8), (1., 1e7), (.15, 1e6)])
def test_resident_newton_final_cpu_fields_and_independent_checks(torch_device, compliance, e_star):
    torch, device = torch_device
    problem, initial, state = case(9, compliance, e_star)
    evaluator = DeviceFullFilmEvaluator(problem, make_backend("torch", device))
    state_d = torch.as_tensor(state, dtype=torch.float64, device=device)
    gpu = solve_device_full_film_load(evaluator, state_d,
                                      reynolds_tolerance=1e-10, load_tolerance=1e-10)
    cpu = solve_monolithic_full_film_load(problem, initial, 18e-6,
                                         reynolds_tolerance=1e-10, load_tolerance=1e-10)
    final = gpu.state.cpu().numpy()
    residual = problem.residual(final)
    assert np.linalg.norm(residual[:-1])/np.sqrt(problem.pressure_size) <= 1e-10
    assert abs(residual[-1]) <= 1e-10
    np.testing.assert_allclose(gpu.pressure_dimensionless.cpu().numpy(),
                               cpu.pressure_dimensionless, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(gpu.film_thickness_m.cpu().numpy(),
                               cpu.film_thickness_m, rtol=1e-7, atol=1e-12)
    assert gpu.central_film_offset_m == pytest.approx(cpu.central_film_offset_m, rel=1e-7)
    assert abs(gpu.calculated_load_n - problem.target_load_n) <= (
        1e-7*problem.load_residual_scale_n
    )
    assert min(entry.minimum_film_thickness_m for entry in gpu.history) > problem.film_floor_m
    assert gpu.history[0].forcing_tolerance == .01
    assert all(entry.merit_directional_derivative < 0 for entry in gpu.history)
    assert all(entry.bordered_residual_relative <= .01 for entry in gpu.history)
    p = gpu.pressure_dimensionless.cpu().numpy()
    h_m = gpu.film_thickness_m.cpu().numpy()
    h = h_m/problem.film_scale_m
    qx = -.5*(h[:, 1:]**3+h[:, :-1]**3)*np.diff(p, axis=1)/problem.dx
    qx += .5*problem.reynolds_parameter*(h[:, 1:]+h[:, :-1])
    qy = -.5*(h[1:, :]**3+h[:-1, :]**3)*np.diff(p, axis=0)/problem.dy
    divergence = (np.diff(qx, axis=1)[1:-1]/problem.dx
                  + np.diff(qy, axis=0)[:, 1:-1]/problem.dy) / problem.reynolds_residual_scale
    assert np.linalg.norm(divergence)/np.sqrt(problem.pressure_size) <= 1e-10
    load_n = trapezoid(trapezoid(p*21250., problem.x, axis=1), problem.y)
    assert abs(load_n-problem.target_load_n) <= 1e-10*problem.load_residual_scale_n
    kernel = cell_integrated_kernel(p.shape, problem.dx_m, problem.dy_m, e_star)
    full = convolve2d(p*21250., kernel)
    direct_w = full[8:17, 8:17]
    np.testing.assert_allclose(h_m, gpu.central_film_offset_m + problem.rigid_geometry_m
                               + compliance*direct_w, rtol=2e-13, atol=1e-20)


def test_both_line_search_stages_match_cpu_and_exhaustion(torch_device):
    torch, device = torch_device
    problem, evaluator, z, zd = problem_case(torch_device)
    direction = np.zeros_like(z)
    direction[-1] = -2.
    dd = torch.as_tensor(direction, dtype=torch.float64, device=device)
    residual = problem.residual(z)
    merit = .5*(np.dot(residual[:-1], residual[:-1])/problem.pressure_size + residual[-1]**2)
    dr = problem.jacobian_vector(z, direction)
    slope = np.dot(residual[:-1], dr[:-1])/problem.pressure_size + residual[-1]*dr[-1]
    options = {"fraction_to_boundary": .99, "armijo_constant": 1e-4, "maximum_backtracks": 3}
    cpu = _armijo_line_search(problem, z, direction, merit, slope, **options)
    gpu = device_line_search(evaluator, zd, dd, merit, slope, **options)
    assert gpu.feasible_step_limit < 1.
    assert gpu.feasible_step_limit == pytest.approx(cpu.feasible_step_limit, rel=1e-12)
    assert gpu.rejected_trials == cpu.rejected_trials
    assert gpu.accepted_step_length == pytest.approx(cpu.accepted_step_length)
    # Deliberately impossible merit threshold exercises rejected Armijo trials.
    exhausted = device_line_search(evaluator, zd, dd, -1., -1., **options)
    assert exhausted.accepted_state is None
    assert exhausted.rejected_trials == 4
    assert exhausted.termination_reason == "backtracking_exhausted"


def test_newton_failure_diagnostics_and_no_field_exports(torch_device, monkeypatch):
    torch, device = torch_device
    problem, initial, z = case(9)
    evaluator = DeviceFullFilmEvaluator(problem, make_backend("torch", device))
    zd = torch.as_tensor(z, dtype=torch.float64, device=device)
    with pytest.raises(DeviceNewtonError) as caught:
        solve_device_full_film_load(evaluator, zd, max_krylov_iterations=1)
    assert caught.value.diagnostics["krylov_diagnostics"]
    assert caught.value.state.device == zd.device
    with pytest.raises(DeviceNewtonError, match="iteration limit"):
        solve_device_full_film_load(evaluator, zd, max_iterations=1)

    def forbidden(*args, **kwargs):
        raise AssertionError("host field export attempted")

    monkeypatch.setattr(torch.Tensor, "cpu", forbidden)
    monkeypatch.setattr(torch.Tensor, "numpy", forbidden)
    result = solve_device_full_film_load(evaluator, zd)
    assert result.state.device == zd.device


@pytest.mark.parametrize("factor", [.9, 1.1])
def test_continuation_cpu_parity_and_retry_reuses_accepted_state(torch_device, monkeypatch, factor):
    import ehl_solver.device_newton as module

    torch, device = torch_device
    problem, initial, z = case(9)
    evaluator = DeviceFullFilmEvaluator(problem, make_backend("torch", device))
    zd = torch.as_tensor(z, dtype=torch.float64, device=device)
    options = {"reynolds_tolerance": 1e-10, "load_tolerance": 1e-10}
    initial_gpu = solve_device_full_film_load(evaluator, zd, **options)
    initial_cpu = solve_monolithic_full_film_load(problem, initial, 18e-6, **options)
    original = module.solve_device_full_film_load
    calls = []

    def fail_once(evaluator, state, **kwargs):
        calls.append(state.clone())
        if len(calls) == 1:
            raise DeviceNewtonError("injected rejected load step", state + 100., (),
                                     {"test_failure": True})
        return original(evaluator, state, **kwargs)

    monkeypatch.setattr(module, "solve_device_full_film_load", fail_once)
    target = problem.target_load_n*factor
    gpu = continue_device_load(evaluator, initial_gpu, target, newton_options=options)
    assert not gpu.steps[0].accepted
    assert gpu.steps[0].failure_diagnostics
    assert torch.equal(calls[0], initial_gpu.state)
    assert torch.equal(calls[1], initial_gpu.state)
    assert gpu.steps[1].step_load_n == pytest.approx(gpu.steps[0].step_load_n/2)
    assert gpu.solution.target_load_n == target
    cpu = continue_monolithic_load(problem, initial_cpu, target, newton_options=options)
    np.testing.assert_allclose(gpu.solution.pressure_dimensionless.cpu().numpy(),
                               cpu.solution.pressure_dimensionless, rtol=1e-7, atol=1e-9)
    assert evaluator.target_load_n == target
    assert not continue_device_load(evaluator, gpu.solution, target).steps


def test_continuation_exhaustion_and_false_success(torch_device, monkeypatch):
    import ehl_solver.device_newton as module

    torch, device = torch_device
    problem, initial, z = case(9)
    evaluator = DeviceFullFilmEvaluator(problem, make_backend("torch", device))
    solution = solve_device_full_film_load(
        evaluator, torch.as_tensor(z, dtype=torch.float64, device=device)
    )
    target = evaluator.target_load_n

    def false_success(*args, **kwargs):
        return solution

    monkeypatch.setattr(module, "solve_device_full_film_load", false_success)
    with pytest.raises(DeviceContinuationError) as caught:
        continue_device_load(evaluator, solution, target*1.1, maximum_retries_per_step=0)
    assert caught.value.solution is solution
    assert not caught.value.steps[-1].accepted
    assert evaluator.target_load_n == target


def test_newton_tighter_linear_recovery_and_armijo_retry(torch_device, monkeypatch):
    import ehl_solver.device_newton as module

    torch, device = torch_device
    problem, initial, z = case(9)
    evaluator = DeviceFullFilmEvaluator(problem, make_backend("torch", device))
    zd = torch.as_tensor(z, dtype=torch.float64, device=device)
    original = module.solve_device_bordered
    tolerances = []

    def first_uphill(*args, **kwargs):
        tolerances.append(kwargs["rtol"])
        result = original(*args, **kwargs)
        return replace(result, direction=-result.direction) if len(tolerances) == 1 else result

    monkeypatch.setattr(module, "solve_device_bordered", first_uphill)
    result = solve_device_full_film_load(evaluator, zd)
    assert tolerances[1] < tolerances[0]
    assert result.history[0].merit_directional_derivative < 0
    monkeypatch.setattr(module, "solve_device_bordered", original)
    original_search = module.device_line_search
    calls = []

    def reject_once(*args, **kwargs):
        calls.append(True)
        result = original_search(*args, **kwargs)
        if len(calls) == 1:
            return replace(result, accepted_state=None, accepted_step_length=0.,
                           rejected_trials=1, termination_reason="injected_rejection")
        return result

    monkeypatch.setattr(module, "device_line_search", reject_once)
    result = solve_device_full_film_load(evaluator, zd)
    assert result.history[0].linear_refinement_attempts == 1
    assert result.history[0].rejected_line_search_trials >= 1
