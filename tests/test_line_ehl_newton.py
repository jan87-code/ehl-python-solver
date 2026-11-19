from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import ehl_solver.device_krylov as device_krylov
import ehl_solver.line_ehl_newton as line_ehl_newton
from ehl_solver.backend import make_backend
from ehl_solver.constitutive import IsothermalFluid
from ehl_solver.geometry import LineContactGeometry
from ehl_solver.line_ehl import LineContactEHLCase
from ehl_solver.line_ehl_newton import (
    LineEHLNewtonSystem,
    LineJFOKrylovError,
    _fraction_to_boundary,
    _solve_correction,
    _true_residual_tolerance,
    solve_line_ehl_newton,
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
    with pytest.raises(
        FloatingPointError,
        match=r"viscosity overflow risk at node \d+: p=",
    ) as error:
        system.residual(state)
    assert line_ehl_newton._failure_stage_from_error(error.value) == "viscosity"


def test_nonfinite_pressure_elasticity_and_film_report_first_grid_index(
    monkeypatch,
):
    torch = pytest.importorskip("torch")
    system = LineEHLNewtonSystem(
        _case(), 1_000.0, backend=make_backend("torch", "cpu")
    )
    state = _state(system)
    state[1] = np.finfo(np.float64).max
    with pytest.raises(
        FloatingPointError, match=r"pressure \[Pa\] at index \(2,\)"
    ):
        system._unpack(system.vector(state))

    state = _state(system)
    monkeypatch.setattr(
        system,
        "elasticity",
        lambda pressure: torch.full_like(pressure, torch.inf),
    )
    with pytest.raises(
        FloatingPointError,
        match=r"elasticity displacement \[m\] at index \(0,\)",
    ):
        system._unpack(system.vector(state))

    monkeypatch.undo()
    system = LineEHLNewtonSystem(
        _case(), 1_000.0, backend=make_backend("torch", "cpu")
    )
    state = _state(system)
    state[-1] = np.finfo(np.float64).max
    system.geometry_m = torch.full_like(
        system.geometry_m, np.finfo(np.float64).max
    )
    with pytest.raises(
        FloatingPointError, match=r"film thickness \[m\] at index \(0,\)"
    ):
        system._unpack(system.vector(state))


def test_nonfinite_density_reports_first_grid_index():
    torch = pytest.importorskip("torch")
    case = _case()
    fluid = replace(
        case.fluid,
        density_a_per_pa=1.0e308,
        density_b_per_pa=1.0e-9,
    )
    system = LineEHLNewtonSystem(
        replace(case, fluid=fluid),
        1_000.0,
        backend=make_backend("torch", "cpu"),
    )

    with pytest.raises(
        FloatingPointError, match=r"density ratio at index \(0,\)"
    ) as error:
        system._properties(torch.full((system.nx,), 1.0e9, dtype=torch.float64))
    assert line_ehl_newton._failure_stage_from_error(error.value) == "density"


def test_nonfinite_residual_jvp_and_line_search_are_stage_labeled(monkeypatch):
    torch = pytest.importorskip("torch")
    system = LineEHLNewtonSystem(
        _case(), 1_000.0, backend=make_backend("torch", "cpu")
    )
    state = system.vector(_state(system))
    monkeypatch.setattr(
        system,
        "_dimensionless_flux",
        lambda *_args: torch.full(
            (system.nx - 1,), torch.nan, dtype=torch.float64
        ),
    )
    with pytest.raises(FloatingPointError, match=r"mass residual at index"):
        system.residual(state)

    system = LineEHLNewtonSystem(
        _case(), 1_000.0, backend=make_backend("torch", "cpu")
    )
    direction = np.zeros(system.size)
    direction[0] = np.finfo(np.float64).max
    with pytest.raises(
        FloatingPointError,
        match=r"pressure JVP \[Pa\] at index",
    ):
        system.jacobian_vector(
            system.vector(_state(system)), system.vector(direction)
        )

    with pytest.raises(
        FloatingPointError,
        match=r"line-search pressure direction \[Pa\] at index",
    ) as error:
        _fraction_to_boundary(
            system,
            system.vector(_state(system)),
            system.vector(direction),
        )
    assert line_ehl_newton._failure_stage_from_error(error.value) == "line-search"


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


def test_line_newton_true_residual_forcing_bounds_remain_unchanged():
    assert _true_residual_tolerance(1.0e-2, 1.0e-8) == pytest.approx(1.0e-6)
    assert _true_residual_tolerance(1.0e-7, 1.0e-8) == pytest.approx(5.0e-8)


def test_line_newton_retains_gmres_history_and_cpu_preconditioner_costs(
    monkeypatch,
):
    torch = pytest.importorskip("torch")
    system = LineEHLNewtonSystem(
        _case(node_count=9), 1_000.0, backend=make_backend("torch", "cpu")
    )
    state = system.vector(_state(system))
    calls = 0

    def residual(_state_value):
        nonlocal calls
        calls += 1
        return torch.ones(system.size, dtype=torch.float64) if calls == 1 else torch.zeros(
            system.size, dtype=torch.float64
        )

    def metrics(_state_value, residual_value):
        norm = float(torch.linalg.vector_norm(residual_value))
        converged = norm == 0.0
        return {
            "residual_l2": norm,
            "mass_residual_linf": 0.0 if converged else 1.0,
            "load_residual_relative": 0.0 if converged else 1.0,
            "pressure_maximum_pa": 10.0,
            "pressure_maximum_location_m": 0.0,
            "central_film_m": 1.0e-6,
            "minimum_film_m": 0.5e-6,
            "maximum_viscosity_pa_s": 0.1,
            "maximum_density_kg_m3": 900.0,
        }

    gmres = SimpleNamespace(
        true_residual_history=(1.0, 0.01, 1.0e-9),
        estimated_preconditioned_history=(1.0, 0.1, 1.0e-8),
        converged=False,
        iterations=3,
        restart_count=1,
    )
    preconditioner = SimpleNamespace(
        used_bordered_fallback=False,
        schur_indicator=0.5,
        superlu_setup_seconds=0.02,
        superlu_solve_seconds=0.03,
    )
    monkeypatch.setattr(system, "residual", residual)
    monkeypatch.setattr(system, "metrics", metrics)
    monkeypatch.setattr(system, "feasibility_error", lambda _state_value: None)
    monkeypatch.setattr(
        system,
        "jacobian_vector",
        lambda _state_value, direction: -torch.ones_like(direction),
    )
    monkeypatch.setattr(
        system,
        "export_solution",
        lambda _state_value, history: SimpleNamespace(history=tuple(history)),
    )
    monkeypatch.setattr(line_ehl_newton, "_fraction_to_boundary", lambda *_args: 1.0)
    monkeypatch.setattr(
        line_ehl_newton,
        "_solve_correction",
        lambda *_args, **_kwargs: (
            torch.ones_like(state),
            gmres,
            1.0e-9,
            preconditioner,
        ),
    )

    solution = solve_line_ehl_newton(system, state)

    assert len(solution.history) == 1
    correction = solution.history[0]
    assert correction.gmres_true_residual_history == (1.0, 0.01, 1.0e-9)
    assert correction.gmres_estimated_preconditioned_history == (
        1.0,
        0.1,
        1.0e-8,
    )
    assert correction.superlu_setup_seconds == 0.02
    assert correction.superlu_solve_seconds == 0.03


@pytest.mark.parametrize(
    ("tolerance_factor", "reported_converged", "accepted"),
    ((0.5, False, True), (1.1, True, False)),
)
def test_line_newton_uses_independent_true_residual_gate(
    monkeypatch, tolerance_factor, reported_converged, accepted
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
    gate = _true_residual_tolerance(
        float(torch.linalg.vector_norm(residual)), 1e-8
    )
    desired_relative_residual = tolerance_factor * gate
    linear_residual_error = np.zeros(system.size)
    linear_residual_error[0] = (
        desired_relative_residual * np.linalg.norm(residual_numpy)
    )
    direction = exact_direction + np.linalg.solve(
        jacobian, linear_residual_error
    )
    result = SimpleNamespace(
        solution=torch.as_tensor(direction, dtype=torch.float64),
        converged=reported_converged,
        reason="converged" if reported_converged else "iteration_limit",
        iterations=600,
        restart_count=10,
        true_residual_history=(desired_relative_residual,),
        estimated_preconditioned_history=(desired_relative_residual,),
    )
    monkeypatch.setattr(
        device_krylov, "device_gmres", lambda *_args, **_kwargs: result
    )

    if accepted:
        _, returned_result, true_relative, _ = _solve_correction(
            system,
            state,
            residual,
            krylov_rtol=1e-10,
            true_residual_tolerance=gate,
            gmres_restart=60,
            maximum_gmres_iterations=600,
            memory_limit_bytes=2**20,
            progress_callback=None,
        )
        assert returned_result is result
        assert not returned_result.converged
        assert true_relative == pytest.approx(
            desired_relative_residual, rel=1e-5
        )
        assert true_relative <= gate
    else:
        with pytest.raises(LineJFOKrylovError) as error:
            _solve_correction(
                system,
                state,
                residual,
                krylov_rtol=1e-10,
                true_residual_tolerance=gate,
                gmres_restart=60,
                maximum_gmres_iterations=600,
                memory_limit_bytes=2**20,
                progress_callback=None,
            )
        assert (
            error.value.diagnostics[
                "independently_recomputed_true_relative_residual"
            ]
            > gate
        )
        assert error.value.diagnostics["gmres_reported_true_residual_history"]


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
