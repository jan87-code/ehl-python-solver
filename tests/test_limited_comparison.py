"""Audited path reading and numerical sampling, including non-nested grids."""

import json
import shutil
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

import ehl_solver
from ehl_solver.limited_comparison import compare_fields, file_sha256, load_limited_path
from ehl_solver.limited_system import LimitedLoadProbe
from ehl_solver.prototype import PointContactCase
from examples.compare_limited_paths import main
from examples.run_limited_transport_path import (
    _interpolate_limited_seed,
    _ProgressReporter,
    resource_preflight,
)


def save_fields(folder, case, state):
    probe = LimitedLoadProbe(case)
    p, theta, film = probe.fields(state)
    _, _, _, w, offset = probe.system.base.unpack(state[:-1])
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        folder / "state.npz",
        state_dimensionless=state,
        x_m=probe.system.problem.x_m,
        y_m=probe.system.problem.y_m,
        pressure_pa=p,
        theta=theta,
        film_m=film,
        displacement_m=w,
        offset_m=offset,
    )


@pytest.fixture(scope="module")
def saved(tmp_path_factory, accepted_mc_path):
    root = accepted_mc_path[0].result
    folder = tmp_path_factory.mktemp("limited_reader") / "accepted"
    save_fields(folder / "attempt_000", root.case, root.state)
    np.savez_compressed(folder / "last_accepted_state.npz", state_dimensionless=root.state)
    hashes = {}
    for path in Path(ehl_solver.__file__).parent.glob("*.py"):
        target = folder / "source/ehl_solver" / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        hashes[path.name] = file_sha256(path)
    driver = Path(__file__).parents[1] / "examples/run_limited_transport_path.py"
    shutil.copyfile(driver, folder / driver.name)
    report = dict(
        status="accepted_numerical_root",
        operator_transport="limited_capacity_target",
        transport_fraction=1.0,
        case=asdict(root.case),
        audit=root.audit,
        source_sha256=hashes,
        driver_sha256=file_sha256(driver),
        last_accepted_state_sha256=file_sha256(folder / "last_accepted_state.npz"),
        stages=[
            dict(
                status="accepted_stage",
                transport_fraction=1.0,
                state_sha256=file_sha256(folder / "attempt_000/state.npz"),
            )
        ],
    )
    (folder / "summary.json").write_text(json.dumps(report))
    return folder


def copy_run(saved, tmp_path):
    target = tmp_path / "portable"
    shutil.copytree(saved, target)
    return target


def rewrite_report(folder, update):
    path = folder / "summary.json"
    report = json.loads(path.read_text())
    update(report)
    path.write_text(json.dumps(report))


def test_portable_root_reaudited_and_fields_are_readonly(saved, tmp_path):
    loaded = load_limited_path(copy_run(saved, tmp_path))
    assert loaded.audit["numerically_accepted"] and loaded.audit["direct_elasticity_checked"]
    assert loaded.audit["full_limited_equations"] and loaded.audit["force_n"] == 50
    for field in (loaded.state, loaded.pressure_pa, loaded.theta, loaded.film_m):
        assert field.dtype == np.float64 and not field.flags.writeable
    with pytest.raises(ValueError):
        loaded.film_m[0, 0] = 0


@pytest.mark.parametrize(
    "changes",
    [
        {"status": "rejected_path"},
        {"transport_fraction": 0.999999},
        {"transport_fraction": True},
        {"operator_transport": "donor"},
    ],
)
def test_partial_or_incompatible_metadata_cannot_promote_a_root(saved, tmp_path, changes):
    folder = copy_run(saved, tmp_path)
    rewrite_report(folder, lambda report: report.update(changes))
    with pytest.raises(ValueError, match="Completed exact MC"):
        load_limited_path(folder)


def test_state_and_source_checksums_are_checked(saved, tmp_path):
    folder = copy_run(saved, tmp_path)
    state = folder / "last_accepted_state.npz"
    state.write_bytes(state.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Checksum"):
        load_limited_path(folder)
    shutil.copyfile(saved / "last_accepted_state.npz", state)
    source = folder / "source/ehl_solver/limited_transport.py"
    source.write_text("raise RuntimeError('archived code must never be executed')\n")
    with pytest.raises(ValueError, match="Checksum"):
        load_limited_path(folder)


def test_updated_checksum_cannot_hide_wrong_field_units_or_state(saved, tmp_path):
    folder = copy_run(saved, tmp_path)
    stage = folder / "attempt_000/state.npz"
    with np.load(stage) as data:
        fields = {k: data[k] for k in data.files}
    fields["pressure_pa"] = fields["pressure_pa"] / 1e6
    np.savez_compressed(stage, **fields)
    rewrite_report(folder, lambda r: r["stages"][-1].update(state_sha256=file_sha256(stage)))
    with pytest.raises(ValueError, match="ND/SI field.*pressure_pa"):
        load_limited_path(folder)


def test_consistent_state_and_field_tampering_still_fails_independent_force(saved, tmp_path):
    folder = copy_run(saved, tmp_path)
    original = load_limited_path(saved)
    state = original.state.copy()
    state[: (original.case.ny - 2) * (original.case.nx - 2)] *= 1.01
    save_fields(folder / "attempt_000", original.case, state)
    np.savez_compressed(folder / "last_accepted_state.npz", state_dimensionless=state)
    rewrite_report(
        folder,
        lambda r: r.update(
            last_accepted_state_sha256=file_sha256(folder / "last_accepted_state.npz"),
            stages=[
                dict(
                    status="accepted_stage",
                    transport_fraction=1.0,
                    state_sha256=file_sha256(folder / "attempt_000/state.npz"),
                )
            ],
        ),
    )
    with pytest.raises(ValueError, match="Independent exact MC"):
        load_limited_path(folder)


def test_computed_residuals_do_not_trust_saved_zero_values(saved, tmp_path):
    folder = copy_run(saved, tmp_path)
    rewrite_report(folder, lambda r: r["audit"]["mass"].update(mass_maximum=0))
    assert load_limited_path(folder).audit["mass"]["mass_maximum"] > 0


def mathematical_fields(case):
    x = np.linspace(-case.domain_half_width_x_m, case.domain_half_width_x_m, case.nx)
    y = np.linspace(-case.domain_half_width_y_m, case.domain_half_width_y_m, case.ny)
    xx, yy = np.meshgrid(x / case.hertz.radius_m, y / case.hertz.radius_m)
    # Bilinear SI test fields, deliberately not claimed as Reynolds/load roots.
    return 1e8 * (2 + 0.1 * xx + 0.1 * yy + 0.01 * xx * yy), 1e-6 * (2 + 0.1 * xx)


@pytest.mark.parametrize(
    "fine_shape,sampling", [((9, 13), "shared_nodes"), ((11, 17), "bilinear_fine_at_coarse_nodes")]
)
def test_nested_and_nonnested_sampling_of_exact_bilinear_si_fields(fine_shape, sampling):
    from ehl_solver.prototype import PointContactCase

    coarse = PointContactCase(ny=5, nx=7)
    fine = replace(coarse, ny=fine_shape[0], nx=fine_shape[1])
    cp, ch = mathematical_fields(coarse)
    fp, fh = mathematical_fields(fine)
    result = compare_fields(coarse, fine, cp, fp, ch, fh)
    assert result["sampling"] == sampling
    assert result["pressure_relative_l2_on_coarse_nodes"] < 1e-14
    assert result["pressure_relative_max_on_coarse_nodes"] < 1e-14
    assert result["exit_pressure_relative_l2_on_coarse_nodes"] < 1e-14
    assert result["exit_pressure_relative_max_on_coarse_nodes"] < 1e-14
    assert abs(result["minimum_film_relative_change"]) < 1e-14
    assert not result["numerical_acceptance_checked"] and not result["mesh_convergence_assessed"]


def test_changed_case_domain_and_coarsening_are_rejected():
    from ehl_solver.prototype import PointContactCase

    case = PointContactCase(ny=9, nx=13)
    p, h = mathematical_fields(case)
    for target in [
        replace(case, load_n=201),
        replace(case, ny=5),
        replace(case, domain_half_width_x_m=1.1 * case.domain_half_width_x_m),
    ]:
        with pytest.raises(ValueError):
            compare_fields(case, target, p, p, h, h)
    with pytest.raises(ValueError, match="positive SI film"):
        compare_fields(case, case, p, p, -h, h)
    with pytest.raises(ValueError, match="Nonzero"):
        compare_fields(case, case, p, np.zeros_like(p), h, h)


def test_large_grid_budget_is_explicit_and_does_not_change_physics():
    with pytest.raises(ValueError, match="20000-node budget"):
        PointContactCase(ny=173, nx=173)
    PointContactCase(ny=173, nx=173, maximum_physical_nodes=30000)
    normal = PointContactCase(ny=141, nx=141)
    for budget in (30000, 40000, 50000):
        larger_budget = replace(normal, maximum_physical_nodes=budget)
        assert larger_budget.physical_signature() == normal.physical_signature()
    PointContactCase(ny=199, nx=199, maximum_physical_nodes=40000)
    PointContactCase(ny=223, nx=223, maximum_physical_nodes=50000)
    for budget in (True, 30000.0, 60000):
        with pytest.raises(ValueError, match="maximum_physical_nodes"):
            PointContactCase(maximum_physical_nodes=budget)


def test_registered_anisotropic_50n_cases_refine_one_axis_at_fixed_domain():
    root = Path(__file__).parents[1]
    base = PointContactCase(**json.loads((root / "cases/circular_50n_199.json").read_text()))
    x_case = PointContactCase(
        **json.loads((root / "cases/circular_50n_199x223.json").read_text())
    )
    y_case = PointContactCase(
        **json.loads((root / "cases/circular_50n_223x199.json").read_text())
    )

    assert x_case.physical_signature() == base.physical_signature()
    assert y_case.physical_signature() == base.physical_signature()
    assert x_case.domain_half_width_x_m == base.domain_half_width_x_m
    assert x_case.domain_half_width_y_m == base.domain_half_width_y_m
    assert y_case.domain_half_width_x_m == base.domain_half_width_x_m
    assert y_case.domain_half_width_y_m == base.domain_half_width_y_m
    assert (x_case.ny, x_case.nx) == (base.ny, 223)
    assert (y_case.ny, y_case.nx) == (223, base.nx)
    assert x_case.nx * x_case.ny == y_case.nx * y_case.ny == 44377
    assert x_case.nx * x_case.ny <= x_case.maximum_physical_nodes
    assert y_case.nx * y_case.ny <= y_case.maximum_physical_nodes
    dx_base = 2 * base.domain_half_width_x_m / (base.nx - 1)
    dy_base = 2 * base.domain_half_width_y_m / (base.ny - 1)
    assert 2 * x_case.domain_half_width_x_m / (x_case.nx - 1) < dx_base
    assert 2 * x_case.domain_half_width_y_m / (x_case.ny - 1) == dy_base
    assert 2 * y_case.domain_half_width_y_m / (y_case.ny - 1) < dy_base
    assert 2 * y_case.domain_half_width_x_m / (y_case.nx - 1) == dx_base


def test_registered_nested_50n_convergence_cases_preserve_physics_and_double_intervals():
    root = Path(__file__).parents[1]
    base = PointContactCase(**json.loads((root / "cases/circular_50n_199.json").read_text()))
    grids = (27, 53, 105, 209)
    cases = [
        PointContactCase(**json.loads((root / f"cases/circular_50n_{n}.json").read_text()))
        for n in grids
    ]

    for n, case in zip(grids, cases, strict=True):
        assert (case.ny, case.nx) == (n, n)
        assert case.physical_signature() == base.physical_signature()
        assert case.domain_half_width_x_m == base.domain_half_width_x_m
        assert case.domain_half_width_y_m == base.domain_half_width_y_m
        assert n * n <= case.maximum_physical_nodes
    assert [case.maximum_physical_nodes for case in cases] == [20000, 20000, 20000, 50000]

    for coarse, fine in zip(cases[:-1], cases[1:], strict=True):
        assert (fine.nx - 1) == 2 * (coarse.nx - 1)
        assert (fine.ny - 1) == 2 * (coarse.ny - 1)
        assert 2 * fine.domain_half_width_x_m / (fine.nx - 1) * 2 == (
            2 * coarse.domain_half_width_x_m / (coarse.nx - 1)
        )
        assert 2 * fine.domain_half_width_y_m / (fine.ny - 1) * 2 == (
            2 * coarse.domain_half_width_y_m / (coarse.ny - 1)
        )


def test_progress_reporter_emits_periodic_solver_status(capsys):
    audit = {"mass": {"relative_local_balance": 2e-8}, "force_n": 50.0}
    with _ProgressReporter(0.01) as reporter:
        reporter.update(3, 0.999, 4, audit, "Newton correction")
        time.sleep(0.04)

    output = capsys.readouterr().out
    assert "[progress " in output
    assert "status=Newton correction" in output
    assert "attempt=4" in output
    assert "target_fraction=0.999" in output
    assert "Newton_iteration=4/30" in output
    assert "local_mass=2.000e-08" in output
    assert "force=50N" in output


def test_limited_seed_transfer_is_interpolated_unaccepted_and_resource_preflighted(saved):
    source = load_limited_path(saved)
    target = replace(
        source.case,
        ny=35,
        nx=37,
        maximum_physical_nodes=50000,
    )
    state, info = _interpolate_limited_seed(saved, target)
    probe = LimitedLoadProbe(target, transport_fraction=0.0)
    assert state.shape == (probe.size,)
    assert info["source_shape"] == [source.case.ny, source.case.nx]
    assert info["target_shape"] == [target.ny, target.nx]
    assert not info["accepted_for_target"] and probe.feasible(state)
    preflight = resource_preflight(target, state)
    assert preflight["physical_nodes"] == target.ny * target.nx
    assert preflight["passed"]
    assert preflight["known_vector_relative_residual"] < 1e-8


def test_comparison_cli_marks_audits_separately_from_convergence(saved, tmp_path):
    output = tmp_path / "comparison"
    assert main([str(saved), str(saved), "--output", str(output)]) == 0
    report = json.loads((output / "comparison.json").read_text())
    correlation = load_limited_path(saved).case.correlation()
    assert report["numerical_acceptance_checked"]
    assert (
        report["sampling"] == "shared_nodes"
        and report["pressure_relative_max_on_coarse_nodes"] == 0
    )
    assert report["hamrock_dowson_reference_m"] == {
        "minimum_film": correlation.minimum_film_m,
        "central_film": correlation.central_film_m,
    }
    assert report["minimum_film_coarse_relative_to_hd"] == pytest.approx(
        report["minimum_film_coarse_m"] / correlation.minimum_film_m - 1
    )
    assert report["minimum_film_fine_relative_to_hd"] == pytest.approx(
        report["minimum_film_fine_m"] / correlation.minimum_film_m - 1
    )
    assert report["central_film_coarse_relative_to_hd"] == pytest.approx(
        report["central_film_coarse_m"] / correlation.central_film_m - 1
    )
    assert report["central_film_fine_relative_to_hd"] == pytest.approx(
        report["central_film_fine_m"] / correlation.central_film_m - 1
    )
    assert not report["mesh_convergence_assessed"] and not report["physical_validation_complete"]
    text = (output / "report.md").read_text()
    assert "approximate fully flooded correlation references" in text
    before = (output / "comparison.json").read_bytes()
    with pytest.raises(SystemExit):
        main([str(saved), str(saved), "--output", str(output)])
    assert (output / "comparison.json").read_bytes() == before
