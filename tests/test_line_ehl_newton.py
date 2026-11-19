import numpy as np
import pytest

import ehl_solver.device_krylov as device_krylov
from ehl_solver.backend import make_backend
from ehl_solver.constitutive import IsothermalFluid
from ehl_solver.geometry import LineContactGeometry
from ehl_solver.line_ehl import LineContactEHLCase
from ehl_solver.line_ehl_newton import (
    LineEHLNewtonSystem,
    _fraction_to_boundary,
    _solve_correction,
    _true_residual_tolerance,
)


def _case(node_count=33):
    x_m = np.linspace(-2.0e-4, 2.0e-4, node_count)
    fluid = IsothermalFluid(
        viscosity_pa_s=0.04,
        density_kg_m3=850.0,
        viscosity_index=0.68,
        viscosity_reference_pa_s=6.31e-5,
        roelands_pressure_pa=1.96e8,
    )
    return LineContactEHLCase(
        x_m=x_m,
        geometry=LineContactGeometry(0.01),
        line_load_n_m=1_000.0,
        e_star_pa=113.0e9,
        fluid=fluid,
        entrainment_speed_m_s=0.1,
        length_scale_m=1.0e-5,
        film_scale_m=0.25e-6,
        pressure_scale_pa=1.0e9,
        elastic_reference_length_m=2.0e-4,
    )


def _state(system):
    x = np.linspace(-1.0, 1.0, system.nx)
    pressure = 0.12 + 0.025 * np.exp(-5.0 * x**2)
    deficit = np.full(system.n, 0.015)
    return np.concatenate((pressure[1:-1], deficit, [2.0]))


def test_line_jfo_newton_jvp_matches_complete_directional_difference():
    pytest.importorskip("torch")
    system = LineEHLNewtonSystem(
        _case(), 1_000.0, backend=make_backend("torch", "cpu")
    )
    state = system.vector(_state(system))
    direction = system.vector(
        np.random.default_rng(42).normal(size=system.size) * 0.01
    )
    epsilon = 1.0e-6
    finite_difference = (
        system.residual(state + epsilon * direction)
        - system.residual(state - epsilon * direction)
    ) / (2.0 * epsilon)
    analytic = system.jacobian_vector(state, direction)
    relative_error = float(
        system.xp.linalg.vector_norm(analytic - finite_difference)
        / system.xp.linalg.vector_norm(finite_difference)
    )

    assert relative_error < 2.0e-7


def test_line_frozen_matrix_has_load_border_and_overflow_is_reported():
    torch = pytest.importorskip("torch")
    from ehl_solver.hybrid_jfo import CPUBlockSchurPreconditioner

    system = LineEHLNewtonSystem(
        _case(), 1_000.0, backend=make_backend("torch", "cpu")
    )
    state = _state(system)
    matrix = system.frozen_matrix(state).toarray()
    expected_load_derivative = (
        system.case.pressure_scale_pa
        * system.case.dx_m
        / system.target_load_n_m
    )

    assert matrix.shape == (system.size, system.size)
    np.testing.assert_allclose(
        matrix[-1, : system.n],
        expected_load_derivative,
        rtol=0.0,
        atol=0.0,
    )
    assert np.count_nonzero(matrix[-1, system.n :]) == 0
    sparse_matrix = system.frozen_matrix(state)
    inverse = CPUBlockSchurPreconditioner(
        sparse_matrix, system.n, "cpu"
    )
    rhs = torch.as_tensor(
        np.random.default_rng(2029).normal(size=system.size),
        dtype=torch.float64,
    )
    reconstructed = inverse(rhs).detach().cpu().numpy()
    relative_residual = np.linalg.norm(
        sparse_matrix @ reconstructed - rhs.numpy()
    ) / np.linalg.norm(rhs.numpy())
    assert relative_residual < 1e-9

    state[: system.n] = 1.0e6
    with pytest.raises(FloatingPointError, match=r"Roelands viscosity exponent.*p_max"):
        system.residual(state)


def test_line_newton_fraction_to_boundary_preserves_strict_film_positivity():
    pytest.importorskip("torch")
    system = LineEHLNewtonSystem(
        _case(), 1_000.0, backend=make_backend("torch", "cpu")
    )
    state = system.vector(_state(system))
    direction = system.xp.zeros_like(state)
    direction[-1] = -5.0
    step = _fraction_to_boundary(system, state, direction)

    assert 0.0 < step < 1.0
    assert system.feasible(state + step * direction)
    assert not system.feasible(state + 1.02 * step * direction)


def test_line_newton_accepts_tight_retry_with_passing_true_residual(
    monkeypatch,
):
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    system = LineEHLNewtonSystem(
        _case(node_count=9), 1_000.0, backend=make_backend("torch", "cpu")
    )
    state = system.vector(_state(system))
    residual = system.residual(state)
    basis = torch.eye(system.size, dtype=torch.float64, device=state.device)
    jacobian = np.column_stack(
        [
            system.jacobian_vector(state, basis[index]).detach().numpy()
            for index in range(system.size)
        ]
    )
    residual_numpy = residual.detach().numpy()
    exact_direction = np.linalg.solve(jacobian, -residual_numpy)
    desired_relative_residual = 5e-7
    linear_residual_error = np.zeros(system.size)
    linear_residual_error[0] = (
        desired_relative_residual * np.linalg.norm(residual_numpy)
    )
    direction = exact_direction + np.linalg.solve(
        jacobian, linear_residual_error
    )
    result = SimpleNamespace(
        solution=torch.as_tensor(direction, dtype=torch.float64),
        converged=False,
        reason="iteration_limit",
        iterations=600,
        restart_count=10,
    )
    monkeypatch.setattr(
        device_krylov, "device_gmres", lambda *_args, **_kwargs: result
    )

    _, returned_result, true_relative, _ = _solve_correction(
        system,
        state,
        residual,
        krylov_rtol=1e-10,
        true_residual_tolerance=_true_residual_tolerance(
            float(torch.linalg.vector_norm(residual)), 1e-8
        ),
        gmres_restart=60,
        maximum_gmres_iterations=600,
        memory_limit_bytes=2**20,
        progress_callback=None,
    )

    assert returned_result is result
    assert not returned_result.converged
    assert true_relative == pytest.approx(desired_relative_residual, rel=1e-5)
    assert true_relative <= _true_residual_tolerance(
        float(torch.linalg.vector_norm(residual)), 1e-8
    )


def test_line_jfo_newton_residual_and_jvp_match_cuda_and_cpu():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is optional for line-contact Newton parity")
    case = _case()
    cpu = LineEHLNewtonSystem(
        case, 1_000.0, backend=make_backend("torch", "cpu")
    )
    cuda = LineEHLNewtonSystem(
        case, 1_000.0, backend=make_backend("torch", "cuda")
    )
    state = _state(cpu)
    direction = np.random.default_rng(2029).normal(size=cpu.size) * 0.01

    residual_cpu = cpu.residual(cpu.vector(state)).detach().cpu().numpy()
    residual_cuda = cuda.residual(cuda.vector(state)).detach().cpu().numpy()
    jvp_cpu = cpu.jacobian_vector(
        cpu.vector(state), cpu.vector(direction)
    ).detach().cpu().numpy()
    jvp_cuda = cuda.jacobian_vector(
        cuda.vector(state), cuda.vector(direction)
    ).detach().cpu().numpy()

    np.testing.assert_allclose(residual_cuda, residual_cpu, rtol=2e-11, atol=2e-12)
    np.testing.assert_allclose(jvp_cuda, jvp_cpu, rtol=2e-10, atol=2e-11)
