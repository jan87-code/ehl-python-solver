from dataclasses import replace

import numpy as np
import pytest

from ehl_solver.cavitating_newton import solve_cavitating_correction
from ehl_solver.constitutive import IsothermalFluid
from ehl_solver.property_coupled import solve_property_coupling
from ehl_solver.property_load_newton import (
    PropertyLoadNewtonError,
    PropertyLoadPathError,
    PropertyLoadSystem,
    continue_property_load,
    solve_property_load_newton,
)
from examples.run_property_load_verification import make_problem


@pytest.fixture(scope="module")
def case():
    p = make_problem(7, 9)
    f = IsothermalFluid(0.04, 850)
    s = solve_property_coupling(p, f, 0.1)
    return p, f, s


def test_complete_bordered_jvp_and_dense_correction(case):
    p, f, s = case
    system = PropertyLoadSystem(p, f, 0.1, s.load_n * 1.05)
    z = system.pack(s, p.central_film_offset_m)
    rng = np.random.default_rng(127)
    smooth = z.copy()
    smooth[: system.n] = rng.uniform(0.001, 0.002, system.n)
    smooth[system.n : 2 * system.n] = rng.uniform(0.02, 0.05, system.n)
    v = rng.normal(size=system.size) * 0.001
    eps = 1e-5
    fd = (system.residual(smooth + eps * v) - system.residual(smooth - eps * v)) / (2 * eps)
    np.testing.assert_allclose(system.jacobian_vector(smooth, v), fd, rtol=2e-6, atol=1e-9)
    matrix = np.column_stack([system.jacobian_vector(z, v) for v in np.eye(system.size)])
    expected = np.linalg.solve(matrix, -system.residual(z))
    for tolerance in (1e-10, 2.0):
        r = solve_cavitating_correction(system, z, schur_tolerance=tolerance)
        assert np.linalg.norm(r.direction - expected) / np.linalg.norm(expected) < 1e-7
        assert r.true_linear_residual <= 5e-9
        assert r.used_bordered_fallback == (tolerance == 2.0)
        assert r.krylov_histories
    full = ~system.full(z)
    view = system.correction_view(z, full)
    selected = np.column_stack([view.jacobian_vector(z, v) for v in np.eye(system.size)])
    d = np.linalg.solve(selected, -view.residual(z))
    np.testing.assert_allclose(view.jacobian_vector(z, d), -view.residual(z), atol=1e-9)


def test_load_recovery_force_gate_and_physical_linesearch(case):
    p, f, s = case
    result = solve_property_load_newton(
        p, f, 0.1, s.load_n * 1.05, seed=s, seed_offset_m=p.central_film_offset_m
    )
    assert abs(result.solution.load_n / (s.load_n * 1.05) - 1) <= 1e-8
    assert result.solution.audit["mass_maximum"] <= 1e-8
    for h in result.solution.history:
        accepted = [(m, t) for m in h["models"] for t in m["trials"] if t["accepted"]]
        assert len(accepted) == 1
        model, t = accepted[0]
        assert model["slope"] < 0
        assert t["merit"] <= h["merit"] + 1e-4 * t["alpha"] * model["slope"]
        assert model["linear_true_residual"] <= 5e-9
    with pytest.raises(PropertyLoadNewtonError) as error:
        solve_property_load_newton(
            p,
            f,
            0.1,
            2 * s.load_n,
            seed=s,
            seed_offset_m=p.central_film_offset_m,
            maximum_iterations=0,
        )
    assert error.value.last_root.solution.load_n == pytest.approx(s.load_n)
    with pytest.raises(ValueError, match="seed fails"):
        solve_property_load_newton(
            p,
            f,
            0.1,
            s.load_n,
            seed=replace(s, constitutive_strength=0),
            seed_offset_m=p.central_film_offset_m,
        )


def test_load_retry_rollback_and_bounded_failure(case, monkeypatch):
    import ehl_solver.property_load_newton as module

    p, f, s = case
    real = module.solve_property_load_newton
    calls = []

    def solve(*args, **kwargs):
        calls.append(kwargs["seed_offset_m"])
        if len(calls) == 1:
            raise PropertyLoadNewtonError("injected failure", (), None)
        return real(*args, **kwargs)

    monkeypatch.setattr(module, "solve_property_load_newton", solve)
    events = []
    r = continue_property_load(
        p,
        f,
        0.1,
        1.05 * s.load_n,
        seed=s,
        seed_offset_m=p.central_film_offset_m,
        initial_increment_n=0.05 * s.load_n,
        callback=lambda record, state: events.append(record["accepted"]),
    )
    assert calls[0] == calls[1] == p.central_film_offset_m
    assert events[0] is False and events[-1] is True
    assert abs(r.root.solution.load_n / (1.05 * s.load_n) - 1) <= 1e-8
    with pytest.raises(PropertyLoadPathError) as error:
        continue_property_load(
            p,
            f,
            0.1,
            1.5 * s.load_n,
            seed=s,
            seed_offset_m=p.central_film_offset_m,
            initial_increment_n=0.01 * s.load_n,
            maximum_attempts=1,
        )
    assert error.value.attempts and error.value.last_root.solution.audit["mass_maximum"] <= 1e-8


def test_film_direction_and_large_grid_no_dense_path(case):
    p, f, s = case
    system = PropertyLoadSystem(p, f, 0.1, s.load_n)
    z = system.pack(s, p.central_film_offset_m)
    v = np.zeros(system.size)
    v[-1] = -0.01
    step = 1e-3
    system.sync(z + step * v)
    plus = system.base.unpack((z + step * v)[:-1])[2]
    system.sync(z - step * v)
    minus = system.base.unpack((z - step * v)[:-1])[2]
    np.testing.assert_allclose(system.film_direction(z, v), (plus - minus) / (2 * step), rtol=1e-8)
    large = make_problem(33, 49)
    big = PropertyLoadSystem(large, f, 0.1, s.load_n)
    assert big.n > 400
    # Sparse frozen matrix includes the border; no dense column-stack or solve.
    fake = np.zeros(big.size)
    fake[-1] = large.central_film_offset_m / large.film_scale_m
    frozen = big.frozen_matrix(fake)
    assert frozen.shape == (big.size, big.size)
    assert frozen.nnz < 30 * big.size


def test_independent_known_offset_recovery(case):
    from examples.property_monolithic_reference import small_monolithic_reference

    p, f, s = case
    target_offset = 0.95 * p.central_film_offset_m
    _, (pressure, theta, h, w) = small_monolithic_reference(
        replace(p, central_film_offset_m=target_offset), f, 0.1
    )
    result = solve_property_load_newton(
        p, f, 0.1, p.load_n(pressure), seed=s, seed_offset_m=p.central_film_offset_m
    )
    assert abs(result.central_film_offset_m / target_offset - 1) < 1e-7
    assert np.linalg.norm(result.solution.pressure_pa - pressure) / np.linalg.norm(pressure) < 1e-7


def test_krylov_failure_preserves_complete_diagnostics(case, monkeypatch):
    import ehl_solver.cavitating_newton as linear_module

    p, f, s = case

    def failed_gmres(operator, rhs, **kwargs):
        kwargs["callback"](1.0)
        return np.zeros_like(rhs), 1

    monkeypatch.setattr(linear_module, "gmres", failed_gmres)
    with pytest.raises(PropertyLoadNewtonError) as error:
        solve_property_load_newton(
            p, f, 0.1, 1.1 * s.load_n, seed=s, seed_offset_m=p.central_film_offset_m
        )
    assert error.value.last_root is not None
    assert error.value.diagnostics["krylov_histories"]
    assert error.value.diagnostics["true_linear_residual"] > 5e-9
