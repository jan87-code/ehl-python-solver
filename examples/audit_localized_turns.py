"""Audit saved localized roots without rerunning branch continuation.

Run as python -m examples.audit_localized_turns with one BLAS/OpenMP thread.
This checks numerical evidence of a nondegenerate discrete parameter maximum;
it does not certify a continuum fold or physical EHL instability.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ehl_solver.turning_point import audit_turning_point_conditions
from ehl_solver.verification import audit_full_film_solution
from examples.run_branch_sensitivity import (
    Case,
    _branch_callbacks,
    _json_finite,
    _require_verified,
    _solution_at,
    make_case,
)


def main(source_directory: Path, output_directory: Path):
    source = json.loads((source_directory / "summary.json").read_text())
    output_directory.mkdir(parents=True, exist_ok=False)
    thresholds = {
        "jacobian_relative_error_max": 2e-8,
        "smallest_to_next_singular_ratio_max": 1e-4,
        "next_to_largest_singular_ratio_min": 1e-8,
        "normalized_transversality_min": 1e-4,
        "normalized_curvature_projection_min": 1e-4,
        "relative_curvature_step_variation_max": 1e-3,
    }
    report = {"source_directory": str(source_directory), "thresholds": thresholds,
              "interpretation": "numerical discrete-root evidence, not a rigorous fold proof",
              "settings": {"curvature_steps": [1e-2, 3e-3, 1e-3],
                           "jacobian_difference_step": 1e-5, "maximum_state_size": 512},
              "cases": []}
    for summary in source["cases"]:
        case = Case(**summary["case"])
        checkpoint = json.loads((source_directory / f"{case.name}_turning_point.json").read_text())
        if checkpoint["termination_reason"] != "localized":
            raise ValueError(f"source root for {case.name} is not localized")
        point = SimpleNamespace(**{key: checkpoint["best"][key] for key in ("state", "parameter")})
        problem, initial = make_case(case, summary["target_load_n"])
        for key in ("length_scale_m", "film_scale_m", "pressure_scale_pa",
                    "reynolds_parameter", "reynolds_residual_scale"):
            if source["settings"].get(key) != getattr(problem, key):
                raise ValueError(f"saved model setting {key} differs from the benchmark factory")
        weights = np.ones(problem.pressure_size + 1)
        weights[:-1] = 1 / problem.pressure_size
        reference_modulus = source.get("compliance_reference_modulus_pa")
        parameter_weight = (1.0 if reference_modulus is None
                            else (reference_modulus / problem.e_star_pa)**2)
        residual, jacobian, derivative, feasible = _branch_callbacks(problem)
        original_gamma = problem.compliance_factor
        try:
            state_audit = audit_full_film_solution(
                problem, _solution_at(problem, initial, point), compliance_factor=point.parameter
            )
            _require_verified(state_audit, problem)
            if state_audit.reynolds_rms_scaled > 1.01e-10:
                raise ValueError("saved localized root fails its independent Reynolds tolerance")
            conditions = audit_turning_point_conditions(
                residual, jacobian, derivative, np.asarray(point.state), point.parameter,
                state_weights=weights, parameter_weight=parameter_weight, feasible=feasible,
            )
        finally:
            problem.compliance_factor = original_gamma
        sigma = conditions.singular_values
        checks = {
            "jacobian_difference": conditions.finite_difference_jacobian_relative_error
            < thresholds["jacobian_relative_error_max"],
            "isolated_small_singular_direction": conditions.smallest_to_next_singular_ratio
            < thresholds["smallest_to_next_singular_ratio_max"]
            and sigma[-2] / sigma[0] > thresholds["next_to_largest_singular_ratio_min"],
            "parameter_transversality": conditions.normalized_transversality
            > thresholds["normalized_transversality_min"],
            "nonzero_curvature": min(conditions.normalized_curvature_projections)
            > thresholds["normalized_curvature_projection_min"],
            "curvature_step_stability": conditions.relative_curvature_step_variation
            < thresholds["relative_curvature_step_variation_max"],
            "local_maximum_curvature": all(
                c < 0 for c in conditions.predicted_parameter_curvatures
            ),
        }
        report["cases"].append({
            "case": asdict(case), "gamma": point.parameter,
            "state_audit": asdict(state_audit), "conditions": asdict(conditions),
            "checks": checks, "supports_discrete_simple_maximum": all(checks.values()),
        })
        (output_directory / "summary.json").write_text(
            json.dumps(_json_finite(report), indent=2, allow_nan=False) + "\n"
        )
        print(f"{case.name}: "
              f"sigma_min/sigma_next={conditions.smallest_to_next_singular_ratio:.3e}; "
              f"transversality={conditions.normalized_transversality:.3e}; "
              f"c''={conditions.predicted_parameter_curvatures[-1]:.6g}; "
              f"all numerical checks={all(checks.values())}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    options = parser.parse_args()
    report = main(options.source_directory, options.output_directory)
    successful = all(case["supports_discrete_simple_maximum"] for case in report["cases"])
    raise SystemExit(0 if successful else 1)
