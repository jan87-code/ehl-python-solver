import json
from pathlib import Path

import numpy as np

from ehl_solver.limited_system import LimitedLoadProbe
from ehl_solver.prototype import PointContactCase
from examples.run_ciulli_donor_grid import (
    _persist_solver_record,
    _reconcile_persisted_history,
    _save_state,
    _seed_state,
)


def _checkpoint(path, case, state, probe):
    _save_state(path, probe, state, case)


def test_seed_loading_supports_exact_restart_and_bilinear_prolongation(tmp_path):
    source_case = PointContactCase(ny=5, nx=5)
    source_probe = LimitedLoadProbe(source_case, transport_fraction=0.0)
    source_state = np.r_[
        np.zeros(2 * source_probe.n),
        10e-6 / source_probe.system.problem.film_scale_m,
    ]
    checkpoint = tmp_path / "checkpoint.npz"
    _checkpoint(checkpoint, source_case, source_state, source_probe)
    with np.load(checkpoint, allow_pickle=False) as archive:
        for field in (
            "pressure_pa",
            "film_m",
            "deformation_m",
            "density_kg_m3",
            "viscosity_pa_s",
            "theta",
        ):
            assert field in archive
        np.testing.assert_allclose(
            archive["film_m"],
            archive["deformation_m"]
            + source_state[-1] * source_probe.system.problem.film_scale_m
            + source_probe.system.problem.rigid_geometry_m,
        )

    exact_state, exact_metadata = _seed_state(checkpoint, source_case, source_probe)
    np.testing.assert_array_equal(exact_state, source_state)
    assert exact_metadata["accepted_as_solution"] is False
    assert exact_metadata["interpolation"].startswith("none")

    target_case = PointContactCase(ny=9, nx=9)
    target_probe = LimitedLoadProbe(target_case, transport_fraction=0.0)
    prolonged, metadata = _seed_state(checkpoint, target_case, target_probe)
    assert prolonged.shape == (target_probe.size,)
    assert target_probe.feasible(prolonged)
    assert metadata["source_grid_ny_nx"] == [5, 5]
    assert metadata["target_grid_ny_nx"] == [9, 9]
    assert metadata["accepted_as_solution"] is False


def test_persistent_iteration_record_keeps_audits_trials_krylov_and_active_sets(tmp_path):
    record = {
        "iteration": 3,
        "merit": 1.5,
        "nonlinear_residual_l2": 2.0,
        "nonlinear_residual_linf": 1.0,
        "mass_residual_linf": 0.8,
        "complementarity_residual_linf": 0.4,
        "force_residual": 0.1,
        "audit": {"force_n": 15.0, "mass": {"relative_local_balance": 1e-8}},
        "models": [
            {
                "status": "linear_solved",
                "recovery": 0,
                "active_full": np.array([True, False, True]),
                "direction": np.array([1.0, -2.0, 0.5]),
                "guide": np.array([0.1, 0.2, 0.3]),
                "linear_true_residual": 1e-10,
                "linear_iterations": 4,
                "krylov_histories": ((0.2, 0.01, 1e-10),),
                "linear_solver_timings": {
                    "cpu_preconditioner_apply_seconds": 0.25
                },
                "trials": [
                    {"alpha": 0.1, "accepted": False, "reason": "Armijo rejected"},
                    {"alpha": 0.05, "accepted": False, "reason": "feasible Armijo candidate"},
                ],
            }
        ],
    }
    record_path, active = _persist_solver_record(tmp_path, record, None)
    saved = json.loads(Path(record_path).read_text())
    model = saved["models"][0]
    assert saved["force_residual"] == 0.1
    assert saved["audit"]["mass"]["relative_local_balance"] == 1e-8
    assert model["linear_iterations"] == 4
    assert model["linear_solver_timings"]["cpu_preconditioner_apply_seconds"] == 0.25
    assert model["krylov_histories"] == [[0.2, 0.01, 1e-10]]
    assert [trial["accepted"] for trial in model["trials"]] == [False, False]
    assert model["correction_norm_l2"] == np.linalg.norm([1.0, -2.0, 0.5])
    assert model["accepted_correction_norm_l2"] is None
    assert model["active_full_count"] == 2
    npz = np.load(saved["active_mask_archive"])
    unpacked = np.unpackbits(npz["model_00_active_full_packed"])[:3].astype(bool)
    np.testing.assert_array_equal(unpacked, [True, False, True])
    np.testing.assert_array_equal(active, [True, False, True])
    _reconcile_persisted_history(
        tmp_path,
        [{"iteration": 3, "models": [{"accepted_step": 0.05}]}],
    )
    saved = json.loads(Path(record_path).read_text())
    model = saved["models"][0]
    assert saved["accepted_recovery"] == 0
    assert saved["accepted_alpha"] == 0.05
    assert [trial["accepted"] for trial in model["trials"]] == [False, True]
    assert model["accepted_correction_norm_l2"] == 0.05 * np.linalg.norm(
        [1.0, -2.0, 0.5]
    )
