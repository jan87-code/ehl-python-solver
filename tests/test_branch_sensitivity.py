import numpy as np
import pytest

from ehl_solver.verification import audit_full_film_solution
from examples.run_branch_sensitivity import CASES, make_case


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
