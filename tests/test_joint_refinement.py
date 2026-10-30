import json
from dataclasses import replace

import numpy as np
import pytest

from ehl_solver.donor_load import DonorLoadSystem
from ehl_solver.property_load_newton import PropertyLoadNewtonError
from ehl_solver.prototype import PointContactCase
from ehl_solver.prototype_cli import load_run, run_case
from examples.run_joint_refinement import independent_audit, joint_shape, run, sampled_minimum


def test_joint_grids_preserve_spacing_ratio_and_explicit_budget():
    for ny in (33, 65, 97, 113):
        n, m = joint_shape(ny)
        assert (n-1)/(m-1) == pytest.approx(2/3)
        assert n % 2 == m % 2 == 1
    with pytest.raises(ValueError, match='allowance'):
        joint_shape(129)
    assert joint_shape(129, 30000) == (129, 193)
    assert joint_shape(141, 30000) == (141, 211)
    for ny, budget in ((145, 30000), (129, 40000), (True, 20000), (99, 20000)):
        with pytest.raises(ValueError):
            joint_shape(ny, budget)


def test_quadratic_sampling_diagnostic_recovers_known_subgrid_minimum():
    x, y = np.arange(-2, 3)*1e-6, np.arange(-2, 3)*2e-6
    xx, yy = np.meshgrid(x/1e-6, y/2e-6)
    film = 100e-9 + 10e-9*(xx-.25)**2 + 20e-9*(yy+.2)**2
    unchanged = film.copy()
    audit = sampled_minimum(x, y, film)
    assert audit['index_yx'] == [2, 2]
    assert audit['fitted_minimum_m'] == pytest.approx(100e-9, abs=1e-22)
    np.testing.assert_allclose(audit['fitted_location_yx_m'], [-.4e-6, .25e-6], atol=1e-20)
    assert audit['fitted_minus_raw_m'] == pytest.approx(-1.425e-9, abs=1e-22)
    np.testing.assert_array_equal(film, unchanged)


def test_sampling_diagnostic_does_not_extrapolate_a_boundary_minimum():
    x = np.arange(5)*1e-6
    h = np.add.outer(x, x)
    audit = sampled_minimum(x, x, h)
    assert audit['fit_status'] == 'boundary minimum; no two-sided fit'
    assert 'fitted_minimum_m' not in audit
    with pytest.raises(ValueError, match='Finite SI'):
        sampled_minimum(x[::-1], x, h)
    h[2, 2] = np.nan
    with pytest.raises(ValueError, match='Finite SI'):
        sampled_minimum(x, x, h)


@pytest.fixture(scope='module')
def seed(tmp_path_factory):
    path = tmp_path_factory.mktemp('joint_seed')/'root'
    report = run_case(PointContactCase(), path, plots=False)
    assert report['status'] == 'accepted_numerical_root'
    return path


def test_real_joint_driver_reaudits_root_and_preserves_existing_outputs(
    seed, tmp_path, monkeypatch
):
    out = tmp_path/'joint'
    report = run(seed, out, 33)
    assert report['status'] == 'accepted_numerical_root'
    assert report['independent_audit']['numerically_accepted']
    assert not report['physical_validation_complete']
    assert report['maximum_corrections'] == 30
    assert report['initialization']['accepted'] is False
    assert (out/'run_joint_refinement.py').is_file()
    before = (out/'solution.npz').read_bytes()
    with pytest.raises(FileExistsError):
        run(seed, out, 33)
    assert (out/'solution.npz').read_bytes() == before

    def failed_krylov(system, state, *args, state_observer, **kwargs):
        raw = state.copy()
        raw[:-1] *= 1.001
        root = system.solution(raw)
        state_observer(0, raw, root)
        raise PropertyLoadNewtonError(
            'injected failed Krylov solve', [{'models': [], 'merit': 1.0}], root,
            {'linear_true_residual': 0.1}
        )

    monkeypatch.setattr('examples.run_joint_refinement._solve_property_load_state', failed_krylov)
    failed = tmp_path/'failed'
    with pytest.raises(ValueError, match='accepted trial'):
        run(seed, failed, 33)
    rejected = json.loads((failed/'summary.json').read_text())
    assert rejected['status'] == 'rejected'
    assert rejected['diagnostics']['linear_true_residual'] == 0.1
    assert rejected['history'] == [{'models': [], 'merit': 1.0}]
    assert (failed/'solution.npz').is_file()


def test_independent_joint_audit_detects_force_and_elastic_inconsistency(seed):
    case, root, _ = load_run(seed)
    system = DonorLoadSystem(case.problem(), case.fluid(), 2, 200,
                            preconditioner='local', elasticity_mode='cached')
    assert independent_audit(system, root)['numerically_accepted']
    damaged = replace(
        root, solution=replace(root.solution, pressure_pa=root.solution.pressure_pa*1.01)
    )
    assert not independent_audit(system, damaged)['numerically_accepted']
