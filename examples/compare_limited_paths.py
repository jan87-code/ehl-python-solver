"""Re-audit two completed MC paths and report spatial changes with SI labels."""

import argparse
import json
from pathlib import Path

from ehl_solver.limited_comparison import (
    compare_fields,
    compare_nested_state_diagnostics,
    file_sha256,
    load_limited_path,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("coarse", type=Path)
    parser.add_argument("fine", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="New comparison directory")
    args = parser.parse_args(argv)
    try:
        coarse, fine = load_limited_path(args.coarse), load_limited_path(args.fine)
        report = compare_fields(
            coarse.case,
            fine.case,
            coarse.pressure_pa,
            fine.pressure_pa,
            coarse.film_m,
            fine.film_m,
        )
        nested_diagnostics = None
        if (
            fine.case.nx - 1 == 2 * (coarse.case.nx - 1)
            and fine.case.ny - 1 == 2 * (coarse.case.ny - 1)
        ):
            nested_diagnostics = compare_nested_state_diagnostics(coarse, fine)
        report.update(
            format="limited_spatial_comparison_v1",
            numerical_acceptance_checked=True,
            physical_validation_complete=False,
            nested_resolution_diagnostics=nested_diagnostics,
            coarse_audit=coarse.audit,
            fine_audit=fine.audit,
            source_runs={
                label: dict(directory=str(path), summary_sha256=file_sha256(path / "summary.json"))
                for label, path in [("coarse", args.coarse), ("fine", args.fine)]
            },
        )
        args.output.mkdir(parents=True, exist_ok=False)
    except (ValueError, TypeError, FileNotFoundError, FileExistsError) as exc:
        parser.error(str(exc))
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    text = f"""# Experimental MC spatial comparison

Both inputs pass independently recomputed exact MC and direct-elasticity audits.
This report measures change; it does not certify mesh convergence or physical accuracy.

- Array order: (ny,nx); mean entrainment: +x.
- Pressure: Pa; film and coordinates: m.
- Coarse/fine shapes: {report["coarse_shape"]} / {report["fine_shape"]}.
- Sampling: {report["sampling"]}.
- Relative pressure L2 on coarse nodes: {report["pressure_relative_l2_on_coarse_nodes"]:.6g}.
- Relative pressure maximum on coarse nodes: {report["pressure_relative_max_on_coarse_nodes"]:.6g}.
- Exit-region pressure L2/maximum differences:
  {report["exit_pressure_relative_l2_on_coarse_nodes"]:.6g} /
  {report["exit_pressure_relative_max_on_coarse_nodes"]:.6g}.
- Coarse/fine raw exit-region peak pressure [Pa]:
  {report["exit_pressure_coarse_peak_pa"]:.9g} / {report["exit_pressure_fine_peak_pa"]:.9g}.
- Raw minimum film, coarse/fine [m]:
  {report["minimum_film_coarse_m"]:.9g} / {report["minimum_film_fine_m"]:.9g}.
- Raw central film, coarse/fine [m]:
  {report["central_film_coarse_m"]:.9g} / {report["central_film_fine_m"]:.9g}.
- Hamrock–Dowson reference minimum/central film [m]:
  {report["hamrock_dowson_reference_m"]["minimum_film"]:.9g} /
  {report["hamrock_dowson_reference_m"]["central_film"]:.9g}.
- Minimum film relative to Hamrock–Dowson, coarse/fine:
  {report["minimum_film_coarse_relative_to_hd"]:+.3%} /
  {report["minimum_film_fine_relative_to_hd"]:+.3%}.
- Central film relative to Hamrock–Dowson, coarse/fine:
  {report["central_film_coarse_relative_to_hd"]:+.3%} /
  {report["central_film_fine_relative_to_hd"]:+.3%}.

L2 pressure is normalized by finer pressure sampled at coarse nodes. Maximum
pressure difference is normalized by the maximum of the full finer pressure.
Exit-region metrics use 0.5<=x/a<=1.5 and |y/a|<=1.2 on the same physical
domain; the regional maximum metric is normalized by the sampled finer regional
peak.
Film changes in comparison.json are relative to the coarse value. Hamrock–Dowson
values are approximate fully flooded correlation references, not acceptance
thresholds. Non-nested sampling includes interpolation error and must not be
presented as shared-node error.
See the JSON for source checksums, independent audits and minimum locations.
"""
    if nested_diagnostics is not None:
        region_lines = []
        region_labels = {
            "exit_inner_half_x_0.5_to_1.0a_abs_y_le_1.2a": (
                "Exit inner half (0.5<=x/a<1.0, |y/a|<=1.2)"
            ),
            "exit_outer_half_x_1.0_to_1.5a_abs_y_le_1.2a": (
                "Exit outer half (1.0<=x/a<=1.5, |y/a|<=1.2)"
            ),
            "exit_inner_center_x_0.5_to_1.0a_abs_y_le_0.5a": (
                "Exit inner center band (0.5<=x/a<1.0, |y/a|<=0.5)"
            ),
        }
        for key, region in nested_diagnostics["regions"].items():
            relative_l2 = region["relative_l2"]
            relative_text = (
                f"{relative_l2:.6g}"
                if relative_l2 is not None
                else "undefined (near-zero pressure)"
            )
            region_lines.append(
                f"- {region_labels[key]}: squared-error share "
                f"{region['total_squared_error_share']:.6g}; relative L2 {relative_text}; "
                f"cavitation-state flips {region['cavitation_state_flip_count']}."
            )
        maximum_front_shift = nested_diagnostics[
            "first_cavitated_sample_maximum_absolute_x_shift_over_a"
        ]
        mean_front_shift = nested_diagnostics[
            "first_cavitated_sample_mean_absolute_x_shift_over_a"
        ]
        front_shift_text = (
            f"{maximum_front_shift:.6g} a max / {mean_front_shift:.6g} a mean"
            if maximum_front_shift is not None
            else "not measured"
        )
        text += f"""
## Nested-grid localization

- Shared-node pressure relative L2: {nested_diagnostics["pressure_relative_l2"]:.6g}.
- Maximum shared-node pressure difference:
  {nested_diagnostics["pressure_maximum_absolute_difference_pa"] / 1e6:.6g} MPa at
  (x/a,y/a)=({nested_diagnostics["pressure_maximum_difference_location_x_over_a"]:.6g},
  {nested_diagnostics["pressure_maximum_difference_location_y_over_a"]:.6g}).
{chr(10).join(region_lines)}
- Fractional-content relative L2 / maximum absolute difference:
  {nested_diagnostics["fractional_content_relative_l2"]:.6g} /
  {nested_diagnostics["fractional_content_maximum_absolute_difference"]:.6g}.
- Cavitation-state flips at shared nodes
  ({nested_diagnostics["cavitation_state_threshold"]}):
  {nested_diagnostics["cavitation_state_flip_count"]}.
- First-cavitated-node x shift, rows with |y/a|<=1.2:
  {front_shift_text} across
  {nested_diagnostics["first_cavitated_sample_shift_rows_abs_y_le_1.2a"]} shared rows.

These are localization diagnostics, not convergence criteria. The existing
exit window is split into x and transverse bands so localized differences are
visible alongside the existing aggregate metric. Front shifts are first
cavitated sample locations on shared rows, not subcell front estimates.
"""
    (args.output / "report.md").write_text(text)
    print(f"Audited numerical roots; spatial convergence unassessed: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
