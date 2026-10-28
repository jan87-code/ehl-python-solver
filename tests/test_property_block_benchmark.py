"""Independent benchmark algebra, failure classification and saved-source gates."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from ehl_solver.benchmark_200n import Circular200N
from ehl_solver.constitutive import IsothermalFluid
from ehl_solver.property_coupled import solve_property_coupling
from ehl_solver.property_load_newton import PropertyLoadSystem
from examples import benchmark_property_block as benchmark
from examples.run_property_load_verification import make_problem


@pytest.fixture(scope="module")
def case():
    problem, fluid = make_problem(7, 9), IsothermalFluid(.04, 850.)
    seed = solve_property_coupling(problem, fluid, .1)
    return problem, fluid, seed


def make_system(case, mode):
    problem, fluid, seed = case
    system = PropertyLoadSystem(problem, fluid, .1, seed.load_n*1.05,
                                preconditioner=mode, elasticity_mode="cached")
    return system, system.pack(seed, problem.central_film_offset_m)


@pytest.mark.parametrize("mode", ["frozen", "material", "local"])
def test_sparse_full_border_inverse_matches_independent_dense_solve(case, mode):
    system, state = make_system(case, mode)
    inverse, metadata = benchmark.build_inverse(system, state, mode)
    rhs = np.random.default_rng(198).normal(size=system.size)
    dense = system.frozen_matrix(state).toarray()
    answer = inverse@rhs
    np.testing.assert_allclose(answer, np.linalg.solve(dense, rhs), rtol=1e-10, atol=1e-9)
    assert np.linalg.norm(dense@answer-rhs)/np.linalg.norm(rhs) < 1e-10
    assert metadata["factor_payload_bytes"] > 0


def test_mg_border_inverse_linearity_exact_force_and_active_rows(case):
    system, state = make_system(case, "frozen")
    inverse, metadata = benchmark.build_inverse(system, state, "multigrid")
    a, b = np.random.default_rng(177).normal(size=(2, system.size))
    np.testing.assert_allclose(inverse@(a+2*b), inverse@a+2*(inverse@b),
                               rtol=1e-10, atol=1e-9)
    direction = inverse@a
    full = system.full(state)
    n = system.n
    np.testing.assert_allclose(np.where(full, direction[n:2*n], direction[:n]), a[n:2*n],
                               rtol=0, atol=1e-12)
    assert system.jacobian_vector(state, direction)[-1] == pytest.approx(a[-1], abs=1e-10)
    assert metadata["approximate_schur_indicator"] > 1e-10
    assert metadata["array_payload_bytes"] > 0
    assert metadata["hierarchy"]["coarse_unknowns"] <= 64


@pytest.mark.parametrize("mode", benchmark.MODES)
def test_true_complete_jvp_solve_against_dense_reference(case, mode):
    system, state = make_system(case, "frozen" if mode == "multigrid" else mode)
    inverse, _ = benchmark.build_inverse(system, state, mode)
    dense = np.column_stack([system.jacobian_vector(state, e) for e in np.eye(system.size)])
    known = np.random.default_rng(32).normal(size=system.size)
    for rhs in (dense@known, -system.residual(state)):
        answer, row = benchmark.measure_solve(system, state, inverse, rhs, restart=60, cycles=4)
        assert row["verified_linear_result"]
        assert row["true_relative_residual"] <= 5e-9
        assert len(row["preconditioned_residual_history"]) == row["iterations"]
        assert row["jvp_calls"] >= row["iterations"]
        expected = np.linalg.solve(dense, rhs)
        assert np.linalg.norm(answer-expected)/np.linalg.norm(expected) < 1e-6


@pytest.mark.parametrize("false_success", [True, False])
def test_report_requires_true_residual_and_gmres_success(case, monkeypatch, false_success):
    system, state = make_system(case, "local")
    inverse, _ = benchmark.build_inverse(system, state, "local")
    known = np.random.default_rng(72).normal(size=system.size)
    rhs = system.jacobian_vector(state, known)

    def injected(*args, **kwargs):
        kwargs["callback"](.5)
        return (np.zeros_like(known), 0) if false_success else (known.copy(), 1)

    monkeypatch.setattr(benchmark, "gmres", injected)
    _, row = benchmark.measure_solve(system, state, inverse, rhs, restart=2, cycles=1)
    assert not row["verified_linear_result"]
    assert row["iterations"] == 1
    assert row["preconditioned_residual_history"] == [.5]
    if false_success:
        assert row["true_relative_residual"] == pytest.approx(1.)
    else:
        assert row["true_relative_residual"] == 0.


def write_rejected_source(directory):
    case = Circular200N()
    problem = case.problem(.5e-6, ny=5, nx=5)
    pressure = np.zeros(problem.shape)
    film, displacement = problem.fields(pressure)
    np.savez_compressed(directory / "failed_partial.npz", x_m=problem.x_m, y_m=problem.y_m,
                        pressure_pa=pressure, theta=np.ones_like(pressure), film_m=film,
                        displacement_m=displacement)
    summary = {"case": case.parameters(), "status": "bounded refinement repair failed",
               "target_n": 200., "failed_offset_m": .5e-6,
               "grids": {"refined": {"shape_ny_nx": [5, 5], "attempts": []}}}
    (directory / "summary.json").write_text(json.dumps(summary))
    return summary


def test_rejected_state_is_not_promoted_or_modified(tmp_path):
    write_rejected_source(tmp_path)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    system, state, info = benchmark.load_source(tmp_path)
    assert info["classification"] == "rejected partial; NOT a root"
    assert info["force_n"] == 0
    assert system.target_load_n == 200
    assert np.linalg.norm(system.residual(state)) > 0
    for path in tmp_path.iterdir():
        assert before[path.name] == path.read_bytes()
    assert info["source_sha256"] == benchmark.digest(tmp_path / "failed_partial.npz")


def test_falsely_accepted_source_fails_independent_force_gate(tmp_path):
    summary = write_rejected_source(tmp_path)
    summary["status"] = "reported complete"
    summary["grids"]["refined"]["attempts"] = [
        {"accepted": True, "load_n": 200., "state_file": "failed_partial.npz", "offset_m": .5e-6}
    ]
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="force"):
        benchmark.load_source(tmp_path)


def test_saved_film_is_independently_recomputed(tmp_path):
    write_rejected_source(tmp_path)
    path = tmp_path / "failed_partial.npz"
    with np.load(path) as a:
        fields = {k: a[k].copy() for k in a.files}
    fields["film_m"] += 1e-8
    np.savez_compressed(path, **fields)
    with pytest.raises(AssertionError):
        benchmark.load_source(tmp_path)


def test_parent_retains_worker_errors_and_finishes_other_modes(tmp_path, monkeypatch):
    calls = []

    def worker(command, **kwargs):
        output = command[command.index("--output-directory")+1]
        mode = command[command.index("--worker-mode")+1]
        directory = type(tmp_path)(output)
        directory.mkdir()
        (directory / "summary.json").write_text(json.dumps({"status": "incomplete"}))
        calls.append(mode)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(benchmark.subprocess, "run", worker)
    monkeypatch.setattr(benchmark, "provenance", lambda: {})
    output = tmp_path / "comparison"
    with pytest.raises(RuntimeError, match="worker error"):
        benchmark.main([tmp_path / "source"], output, 1, 2, 1)
    assert calls == list(benchmark.MODES)
    report = json.loads((output / "summary.json").read_text())
    assert report["status"] == "worker errors; incomplete"
    assert all(row["worker_returncode"] == 1 for row in report["cases"][0]["modes"].values())
    with pytest.raises(FileExistsError):
        benchmark.main([], output, 1, 2, 1)


def test_snapshot_mismatch_is_rejected_and_unverified_directions_are_not_compared(tmp_path):
    source = {"source_sha256": "state", "summary_sha256": "summary"}
    modes = {mode: {"source": source.copy(), "state_sha256": "packed", "solves": {
        name: [{"verified_linear_result": False}] for name in ("newton", "known_vector")
    }} for mode in benchmark.MODES}
    row = {"modes": modes}
    benchmark.compare_saved_directions(row, tmp_path, 0)
    assert row["verified_direction_relative_differences_to_local"] == {}
    modes["multigrid"]["source"]["source_sha256"] = "changed"
    with pytest.raises(RuntimeError, match="different source"):
        benchmark.compare_saved_directions(row, tmp_path, 0)


def test_accepted_input_change_during_audit_is_detected(tmp_path, monkeypatch):
    summary = write_rejected_source(tmp_path)
    summary["grids"]["refined"]["attempts"] = [
        {"accepted": True, "load_n": 200., "state_file": "failed_partial.npz"}
    ]
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    path = tmp_path / "failed_partial.npz"

    def changed(source):
        path.write_bytes(b"changed during audit")
        return None, None, None, {"source_sha256": benchmark.digest(path)}

    monkeypatch.setattr(benchmark, "load_root", changed)
    with pytest.raises(RuntimeError, match="changed during audit"):
        benchmark.load_source(tmp_path)


def test_rejected_inverse_setup_is_saved_without_promoting_source(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    write_rejected_source(source)
    output = tmp_path / "worker"

    def fail(*args):
        raise RuntimeError("injected inverse failure")

    monkeypatch.setattr(benchmark, "build_inverse", fail)
    benchmark.worker(source, output, "multigrid", 1, 2, 1)
    row = json.loads((output / "summary.json").read_text())
    assert row["status"] == "inverse setup rejected"
    assert row["reason"] == "injected inverse failure"
    assert row["source"]["classification"] == "rejected partial; NOT a root"
    assert not (output / "linear_vectors.npz").exists()
