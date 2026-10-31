"""Audited path reading and numerical sampling, including non-nested grids."""

import json
import shutil
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

import ehl_solver
from ehl_solver.limited_comparison import compare_fields, file_sha256, load_limited_path
from ehl_solver.limited_system import LimitedLoadProbe
from examples.compare_limited_paths import main


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
