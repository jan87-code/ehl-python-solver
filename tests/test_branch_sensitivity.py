import numpy as np
import pytest

from ehl_solver.monolithic import continue_monolithic_compliance_pseudo_arclength
from ehl_solver.verification import audit_full_film_solution
from examples.run_branch_sensitivity import CASES, Case, make_case, run_case


def test_mesh_and_domain_cases_share_load_without_retargeting_the_rigid_gap():
    baseline, initial = make_case(CASES[0])
    for case in CASES[1:]:
        problem, root = make_case(case, baseline.target_load_n)
        assert problem.target_load_n == baseline.target_load_n
        assert problem.load_residual_scale_n == baseline.load_residual_scale_n
        assert root.central_film_offset_m != initial.central_film_offset_m
        audit = audit_full_film_solution(problem, root)
        assert audit.reynolds_rms_scaled < 1e-8
        assert abs(audit.load_residual_scaled) < 1e-6
        if case.name == "expanded":
            assert problem.dx_m == pytest.approx(baseline.dx_m)
            assert problem.dy_m == pytest.approx(baseline.dy_m)
            np.testing.assert_allclose(problem.x[2:-2], baseline.x, atol=1e-18)
            np.testing.assert_allclose(problem.y[2:-2], baseline.y, atol=1e-18)



@pytest.mark.parametrize("target", [1.0, 0.1])
def test_normalized_compliance_metric_matches_modulus_branches_and_keeps_gamma_units(target):
    reference, initial = make_case(CASES[1])
    soft, soft_initial = make_case(Case("soft", 7, 9, e_star_pa=1e5), reference.target_load_n)
    branches = []
    for problem, root in ((reference, initial), (soft, soft_initial)):
        result = continue_monolithic_compliance_pseudo_arclength(
            problem, root, initial_step=0.03, maximum_step=0.05,
            maximum_steps=12, residual_tolerance=1e-8,
            compliance_reference_modulus_pa=1e6,
            target_compliance_factor=target * problem.e_star_pa / reference.e_star_pa,
        )
        audit = audit_full_film_solution(problem, result.solution)
        assert audit.reynolds_rms_scaled < 1.01e-8
        assert abs(audit.load_residual_scaled) < 1e-6
        branches.append(result.branch)
    a, b = branches
    assert a.turning_point_attempts == b.turning_point_attempts
    if target == 1.0:
        assert a.turning_point_attempts
        assert a.termination_reason == b.termination_reason == "maximum_steps"
    else:
        assert a.termination_reason == b.termination_reason == "target_parameter_reached"
        assert a.parameter == target
        assert b.parameter == target * soft.e_star_pa / reference.e_star_pa
    np.testing.assert_allclose([step.state for step in a.steps], [step.state for step in b.steps],
                               rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose([step.parameter for step in a.steps],
                               [10 * step.parameter for step in b.steps], rtol=1e-10)
    assert reference.compliance_factor == a.parameter
    assert soft.compliance_factor == b.parameter


@pytest.mark.parametrize("modulus", [0.0, -1.0, float("nan"), float("inf"), 1e308, 1e-308])
def test_normalized_compliance_metric_rejects_invalid_or_unrepresentable_values(modulus):
    problem, root = make_case(CASES[1])
    with pytest.raises(ValueError, match="compliance.*(modulus|metric)"):
        continue_monolithic_compliance_pseudo_arclength(
            problem, root, compliance_reference_modulus_pa=modulus
        )



def test_localized_coupled_turn_passes_independent_audit_and_restores_problem(tmp_path):
    case = CASES[1]
    problem, initial = make_case(case)
    report = run_case(
        case, problem, initial, tmp_path, localize_turn=True,
        compliance_reference_modulus_pa=1e6,
    )
    turn = report["turning_point"]
    assert turn["termination_reason"] == "localized"
    assert turn["bracket_width"] <= 1e-6
    assert abs(turn["normalized_parameter_tangent"]) <= 1e-6
    assert turn["best_audit"]["reynolds_rms_scaled"] < 1.01e-10
    assert abs(turn["best_audit"]["load_residual_scaled"]) < 1e-6
    assert turn["parameter"] >= report["maximum_sampled_compliance"]
    with np.load(tmp_path / f"{case.name}_states.npz") as archive:
        assert problem.compliance_factor == archive["parameters"][-1]
    assert (tmp_path / f"{case.name}_turning_point.json").is_file()
