"""Flux-only elastic/material JVP probes on an accepted donor prototype seed.

A donor root is NOT a root of the limited scheme. This does not test a
bordered Newton correction or establish coupled-candidate convergence.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from ehl_solver.limited_transport import limited_flux_direction, limited_fluxes
from ehl_solver.prototype_cli import digest, load_run
from ehl_solver.transport_candidate import face_divergence


def limiter_signature(x, capacity):
    """Independent branch labels for diagnostic selection, not a derivative."""
    widths = np.diff(x)
    differences = np.diff(capacity, axis=1) / widths
    b, f = differences[:, :-1], differences[:, 1:]
    c = (widths[1:] * b + widths[:-1] * f) / (widths[:-1] + widths[1:])
    active = ((b > 0) & (f > 0)) | ((b < 0) & (f < 0))
    labels = np.zeros((capacity.shape[0], capacity.shape[1] - 1), dtype=int)
    labels[:, 1:] = np.where(active, 1 + np.argmin(abs(np.stack([c, 2 * b, 2 * f])), axis=0), 0)
    b, f = differences[:, 0], differences[:, 1]
    c = ((2 * widths[0] + widths[1]) * b - widths[0] * f) / (widths[0] + widths[1])
    active = ((b > 0) & (c > 0)) | ((b < 0) & (c < 0))
    labels[:, 0] = np.where(active, np.where(abs(c) <= 2 * abs(b), 1, 2), 0)
    return labels


def main(seed, output):
    case, root, _ = load_run(seed)
    if case.transport != "donor":
        raise ValueError("An independently audited donor prototype root is required")
    output.mkdir(parents=True, exist_ok=False)
    system = case.system()
    problem, fluid = system.problem, case.fluid()
    p, h, theta = (
        root.solution.pressure_pa,
        root.solution.film_thickness_m,
        root.solution.fractional_film_content,
    )
    state = system.pack(root.solution, root.central_film_offset_m)
    x, y = problem.x_m, problem.y_m
    yy, xx = np.meshgrid(y / case.hertz.radius_m, x / case.hertz.radius_m, indexing="ij")
    base = face_divergence(x, y, *limited_fluxes(x, y, h, p, theta, fluid, case.entrainment_m_s))
    wet = p > 1e-4 * problem.pressure_scale_pa
    smooth = wet[1:-1, 1:-1] & wet[1:-1, :-2] & wet[1:-1, 2:]
    # The west face reconstruction can use x=2 as well; omit reservoir-adjacent cells.
    smooth[:, 0] = False
    two_left = np.zeros_like(wet)
    two_left[:, 2:] = wet[:, :-2]
    smooth &= two_left[1:-1, 1:-1]
    rho, _, _, _ = fluid.properties(p)
    baseline_branches = limiter_signature(x, rho * h * theta)
    rows = []
    directions = [
        dict(name="pressure_elastic", dp=0.01 * p * np.sin(2 * xx + 0.4 * yy), dt=0.0, offset=0.0),
        dict(
            name="content",
            dp=np.zeros_like(p),
            dt=0.01 * theta * (1 - theta) * np.sin(xx + yy),
            offset=0.0,
        ),
        dict(name="offset_1nm", dp=np.zeros_like(p), dt=0.0, offset=1e-9),
    ]
    for probe in directions:
        dp = probe["dp"]
        dt = np.broadcast_to(probe["dt"], p.shape)
        v = np.r_[
            dp[1:-1, 1:-1].ravel() / problem.pressure_scale_pa,
            -dt[1:-1, 1:-1].ravel(),
            probe["offset"] / problem.film_scale_m,
        ]
        dh = system.film_direction(state, v)
        analytic = face_divergence(
            x,
            y,
            *limited_flux_direction(x, y, h, p, theta, dh, dp, dt, fluid, case.entrainment_m_s),
        )
        errors = []
        for step in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
            values = []
            stable_faces = np.ones_like(baseline_branches, dtype=bool)
            for sign in (-1, 1):
                hp, pp, tp = h + sign * step * dh, p + sign * step * dp, theta + sign * step * dt
                if hp.min() <= 0 or pp.min() < -1e-10 or tp.min() < -1e-10 or tp.max() > 1 + 1e-10:
                    raise ValueError(
                        "Loaded flux diagnostic perturbs outside physical field bounds"
                    )
                density, _, _, _ = fluid.properties(pp)
                stable_faces &= limiter_signature(x, density * hp * tp) == baseline_branches
                values.append(
                    face_divergence(
                        x, y, *limited_fluxes(x, y, hp, pp, tp, fluid, case.entrainment_m_s)
                    )
                )
            finite_difference = (values[1] - values[0]) / (2 * step)
            difference = finite_difference - analytic
            stable = stable_faces[1:-1, :-1] & stable_faces[1:-1, 1:]
            errors.append(
                dict(
                    step=step,
                    branch_stable_cells=int(stable.sum()),
                    branch_switching_cells=int(stable.size - stable.sum()),
                    branch_stable_relative_error=float(
                        np.linalg.norm(difference[stable])
                        / max(np.linalg.norm(analytic[stable]), 1e-30)
                    ),
                    global_relative_error=float(
                        np.linalg.norm(difference) / max(np.linalg.norm(analytic), 1e-30)
                    ),
                    full_film_relative_error=float(
                        np.linalg.norm(difference[smooth])
                        / max(np.linalg.norm(analytic[smooth]), 1e-30)
                    ),
                )
            )
        rows.append(
            dict(
                name=probe["name"],
                errors=errors,
                note="Limiter switches may lack a classical two-sided derivative",
            )
        )
    report = dict(
        scope=__doc__,
        seed=str(seed),
        seed_state_sha256=digest(seed / "solution.npz"),
        shape_ny_nx=list(p.shape),
        case=vars(case),
        candidate_is_converged=False,
        coupled_candidate_tested=False,
        candidate_scaled_mass_on_donor_root=float(
            abs(base).max() * problem.length_scale_m / system.base.flux_scale
        ),
        full_film_cells=int(smooth.sum()),
        probes=rows,
    )
    (output / "summary.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(output / "candidate_residual.npz", x_m=x, y_m=y, divergence_si=base)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    main(args.seed, args.output)
