import json
from dataclasses import replace

import numpy as np
import pytest

from ehl_solver.prototype import (
    PointContactCase,
    audit_history,
    audit_root,
    hertz_guess,
    transfer_guess,
)
from ehl_solver.prototype_cli import load_run, main, run_case


@pytest.fixture(scope="module")
def solved(tmp_path_factory):
    folder = tmp_path_factory.mktemp("prototype") / "coarse"
    case = PointContactCase()
    report = run_case(case, folder, plots=False)
    assert report["status"] == "accepted_numerical_root"
    return folder, case, report


@pytest.mark.parametrize(
    "changes",
    [
        {"load_n": 0},
        {"viscosity_pa_s": float("nan")},
        {"ny": 10},
        {"nx": True},
        {"ny": 145, "nx": 145},
        {"transport": "silent_fallback"},
        {"domain_half_width_x_m": 1e-9},
        {"viscosity_reference_pa_s": 0.04},
    ],
)
def test_invalid_cases_raise(changes):
    with pytest.raises(ValueError):
        PointContactCase(**changes)


def test_hertz_initialization_is_positive_force_closed_and_unaccepted():
    case = PointContactCase()
    system = case.system()
    z, info = hertz_guess(case, system)
    root = system.solution(z)
    assert info["accepted"] is False
    assert root.solution.load_n == pytest.approx(case.load_n, rel=1e-14)
    assert root.solution.film_thickness_m.min() == pytest.approx(
        case.correlation().central_film_m, rel=1e-12
    )
    assert not audit_root(case, root)["numerically_accepted"]


def test_configuration_converts_physical_scalars_to_float64():
    case = PointContactCase(load_n=np.float32(50))
    assert type(case.load_n) is float
    assert case.problem().x_m.dtype == np.float64


def test_real_standalone_run_and_reaudited_portable_seed(solved):
    folder, case, report = solved
    assert report["independent_audit"]["numerically_accepted"]
    assert abs(report["independent_audit"]["force_relative_error"]) <= 1e-8
    assert report["history_audit"]["directional_armijo_verified"]
    assert not report["physical_validation_complete"]
    assert (folder / "report.md").exists()
    assert (folder / "source/ehl_solver/prototype.py").exists()
    new_case, root, _ = load_run(folder)
    assert case == new_case
    assert audit_root(case, root)["numerically_accepted"]
    with np.load(folder / "solution.npz", allow_pickle=False) as a:
        assert a["pressure_pa"].shape == (33, 49)
        assert a["state_dimensionless"].shape == (2 * 31 * 47 + 1,)
    assert len(report["iterations"]) == len(report["history"]) + 1


def test_nonnested_interpolation_preserves_offset_without_promoting_guess(solved):
    folder, case, _ = solved
    _, root, _ = load_run(folder)
    target = replace(case, ny=45, nx=67)
    system = target.system()
    state, info = transfer_guess(case, target, root, system)
    assert not info["accepted"]
    assert info["offset_adjustment_m"] == 0
    assert system.solution(state).central_film_offset_m == root.central_film_offset_m
    assert system.solution(state).solution.pressure_pa.shape == (45, 67)
    with pytest.raises(ValueError, match="physical parameters"):
        wrong = replace(case, transport="arithmetic")
        transfer_guess(case, wrong, root, wrong.system())
    with pytest.raises(ValueError, match="coarsen"):
        smaller = replace(case, ny=17, nx=25)
        transfer_guess(case, smaller, root, smaller.system())


def test_fixed_spacing_domain_embedding_and_misalignment_rejection(solved):
    folder, case, _ = solved
    _, root, _ = load_run(folder)
    target = replace(case, nx=65, domain_half_width_x_m=case.domain_half_width_x_m * 4 / 3)
    state, info = transfer_guess(case, target, root, target.system())
    assert not info["accepted"]
    new = target.system().solution(state).solution
    np.testing.assert_allclose(
        new.pressure_pa[:, 8:57], root.solution.pressure_pa, rtol=1e-14, atol=0
    )
    assert abs(info["initial_force_relative_error"]) < 1e-12
    bad = replace(case, nx=65, domain_half_width_x_m=case.domain_half_width_x_m * 1.2)
    with pytest.raises(ValueError, match="physical spacing"):
        transfer_guess(case, bad, root, bad.system())


def test_existing_output_is_preserved_and_state_tampering_rejected(solved, tmp_path):
    folder, case, _ = solved
    with pytest.raises(FileExistsError):
        run_case(case, folder, plots=False)
    # Copy only the report and deliberately invalid state: checksum fails first.
    target = tmp_path / "badseed"
    target.mkdir()
    (target / "summary.json").write_bytes((folder / "summary.json").read_bytes())
    (target / "solution.npz").write_bytes(b"invalid or replaced state")
    with pytest.raises(ValueError, match="checksum"):
        load_run(target)


def test_cli_rejection_has_nonzero_status_and_complete_diagnostics(tmp_path):
    output = tmp_path / "rejected"
    code = main(["--output", str(output), "--max-iterations", "0", "--no-plots"])
    assert code == 1
    report = json.loads((output / "summary.json").read_text())
    assert report["status"] == "rejected"
    assert not report["independent_audit"]["numerically_accepted"]
    assert (output / "solution.npz").exists() and (output / "report.md").exists()
    with pytest.raises(ValueError, match="accepted numerical root"):
        load_run(output)
    with pytest.raises(SystemExit) as exc:
        main(["--output", str(tmp_path / "too_many"), "--max-iterations", "31"])
    assert exc.value.code == 2


def test_history_audit_rejects_nan_and_false_acceptances():
    model = {
        "slope": -1.0,
        "linear_true_residual": 1e-10,
        "trials": [{"accepted": True, "merit": 0.5, "alpha": 1.0}],
    }
    history = [{"merit": 1.0, "models": [model]}]
    assert audit_history(history)["corrections"] == 1
    model["slope"] = float("nan")
    with pytest.raises(ValueError, match="Armijo"):
        audit_history(history)
    model["slope"] = -1.0
    model["linear_true_residual"] = float("nan")
    with pytest.raises(ValueError, match="true-linear"):
        audit_history(history)
