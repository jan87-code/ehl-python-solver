"""Independent smooth-source and accepted-root diagnostics; no candidate solve.

Resource accounting reports sparse payloads, NOT total peak process memory.
"""

import argparse
import time
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import splu

from ehl_solver.benchmark_200n import Circular200N
from ehl_solver.constitutive import IsothermalFluid
from ehl_solver.elasticity import ElasticityOperator
from ehl_solver.property_reynolds import mass_fluxes
from ehl_solver.transport_candidate import donor_flux_direction, donor_fluxes, face_divergence
from ehl_solver.transport_probes import active_front_probes, resolved_pressure_probes
from examples.audit_transport_wavelengths import load_root
from examples.run_scientific_validation import provenance, save_json
from examples.scientific_figures import figure, save


def smooth_case(n, *, varying_content):
    """Manufactured SI source with independent analytic continuum derivatives.

    Constant material limit; p linear in x,y, h and theta smooth trigonometric.
    Prescribed fields with an added source, NOT complementarity/EHL roots.
    """
    length = 1e-3
    x, y = np.linspace(0., length, n), np.linspace(0., length, n)
    xx, yy = np.meshgrid(x, y)
    k, h0, ax, ay = 2*np.pi/length, 1e-6, 1e10, .7e10
    h = h0*(1+.1*np.sin(k*xx)+.08*np.cos(k*yy))
    hx, hy = .1*h0*k*np.cos(k*xx), -.08*h0*k*np.sin(k*yy)
    p = 1e7+ax*xx+ay*yy
    if varying_content:
        theta, tx = .8+.05*np.sin(k*xx+.3), .05*k*np.cos(k*xx+.3)
    else:
        theta, tx = np.ones_like(p), np.zeros_like(p)
    fluid, speed = IsothermalFluid(.04, 850.), 2.
    exact = (-fluid.density_kg_m3/(4*fluid.viscosity_pa_s)*h**2*(hx*ax+hy*ay)
             + speed*fluid.density_kg_m3*(hx*theta+h*tx))
    return x, y, h, p, theta, fluid, speed, exact[1:-1, 1:-1]


def smooth_errors(n, *, varying_content):
    x, y, h, p, t, f, speed, exact = smooth_case(n, varying_content=varying_content)
    errors = {}
    for name, flux in (("reference", mass_fluxes), ("donor", donor_fluxes)):
        numerical = face_divergence(x, y, *flux(x, y, h, p, t, f, speed, strength=0.))
        errors[name] = float(np.linalg.norm(numerical-exact)/np.linalg.norm(exact))
    return errors


def timed(call, repeats=5):
    call()
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        values.append(time.perf_counter()-start)
    return {"median_s": float(np.median(values)), "minimum_s": min(values), "samples_s": values}


def payload(matrix):
    return int(matrix.data.nbytes+matrix.indices.nbytes+matrix.indptr.nbytes)


def accepted_case(source, directory):
    system, state, seed, info = load_root(source)
    problem, f = system.problem, system.base.fluid
    x, y = problem.x_m, problem.y_m
    h, p, t = seed.film_thickness_m, seed.pressure_pa, seed.fractional_film_content
    flux_ref = mass_fluxes(x, y, h, p, t, f, 2.)
    flux_new = donor_fluxes(x, y, h, p, t, f, 2.)
    ref = face_divergence(x, y, *flux_ref)
    new = face_divergence(x, y, *flux_new)
    rho, _, _, _ = f.properties(p)
    dc = -np.diff(rho*h, axis=1)*t[:, :-1]  # -U/2 * diff(capacity) * theta_left; U=2
    difference = face_divergence(x, y, dc, np.zeros_like(flux_ref[1]))
    error = float(np.linalg.norm((new-ref)-difference)/np.linalg.norm(difference))
    if error > 1e-10:
        raise RuntimeError("independent candidate/reference flux identity failed")
    factor = problem.length_scale_m/system.base.flux_scale
    probes, _ = resolved_pressure_probes(x, y, p, Circular200N().hertz_radius_m)
    fronts, front_selection = active_front_probes(p, t, problem.pressure_scale_pa)
    probes.update(fronts)
    checks = []
    for name, probe in probes.items():
        dp, dt = probe["pressure_pa"], probe["content"]
        v = np.r_[dp[1:-1, 1:-1].ravel()/problem.pressure_scale_pa,
                  -dt[1:-1, 1:-1].ravel(), 0.]
        dh = system.film_direction(state, v)
        q = donor_flux_direction(x, y, h, p, t, dh, dp, dt, f, 2.)
        analytic = face_divergence(x, y, *q)
        errors = []
        for step in (1e-2, 1e-3, 1e-4):
            plus = donor_fluxes(x, y, h+step*dh, p+step*dp, t+step*dt, f, 2.)
            if name in fronts:
                for trial_step in (step, 2*step):
                    if (np.min(p+trial_step*dp) < -1e-10*problem.pressure_scale_pa
                            or np.min(t+trial_step*dt) < -1e-10
                            or np.max(t+trial_step*dt) > 1+1e-10
                            or np.min(h+trial_step*dh) <= problem.film_floor_m):
                        raise RuntimeError("candidate front diagnostic violates field bounds")
                twice = donor_fluxes(x, y, h+2*step*dh, p+2*step*dp, t+2*step*dt, f, 2.)
                fd = (4*(face_divergence(x, y, *plus)-new)
                      - (face_divergence(x, y, *twice)-new))/(2*step)
            else:
                minus = donor_fluxes(x, y, h-step*dh, p-step*dp, t-step*dt, f, 2.)
                fd = (face_divergence(x, y, *plus)-face_divergence(x, y, *minus))/(2*step)
            errors.append(float(np.linalg.norm(fd-analytic)/np.linalg.norm(analytic)))
        if min(errors) > 1e-6:
            raise RuntimeError("candidate loaded elastic direction failed")
        checks.append({"name": name, "one_sided": name in fronts, "relative_fd_errors": errors})
    start = time.perf_counter()
    matrix = system.frozen_matrix(state)
    assembly_s = time.perf_counter()-start
    rng = np.random.default_rng(20261028)
    known = rng.normal(size=system.size)
    factors = {}
    for name, a in (("pressure_content_block", matrix[:-1, :-1]),
                    ("full_bordered_fallback", matrix)):
        start = time.perf_counter()
        lu = splu(a)
        factor_s = time.perf_counter()-start
        rhs = a@known[:a.shape[0]]
        solved = lu.solve(rhs)
        linear_error = float(np.linalg.norm(a@solved-rhs)/np.linalg.norm(rhs))
        if linear_error > 1e-8:
            raise RuntimeError("preconditioner factor true residual failed")
        factors[name] = {
            "factor_s": factor_s, "solve_timing": timed(lambda lu=lu, rhs=rhs: lu.solve(rhs)),
            "true_relative_residual": linear_error,
            "lu_payload_bytes": payload(lu.L)+payload(lu.U),
            "lu_nnz": int(lu.L.nnz+lu.U.nnz),
        }
    storage = {
        "matrix_payload_bytes": payload(matrix),
        "krylov_81_float64_vectors_bytes": int(81*system.size*8),
        "material_krylov_61_float64_vectors_bytes": int(61*system.size*8),
        "dense_full_jacobian_bytes_not_allocated": int(system.size**2*8),
        "note": "payload lower bounds; exclude workspaces, Python, FFTs and allocator peaks",
    }
    full_shape = tuple(np.array(p.shape)+np.array(problem.elastic_kernel.shape)-1)
    spectrum = np.fft.rfft2(problem.elastic_kernel, s=full_shape)
    sy, sx = np.array(problem.elastic_kernel.shape)//2

    def cached_film_direction():
        full = np.fft.irfft2(np.fft.rfft2(p, s=full_shape)*spectrum, s=full_shape)
        return full[sy:sy+p.shape[0], sx:sx+p.shape[1]]

    displacement = cached_film_direction()
    cache_error = float(np.linalg.norm(displacement-seed.elastic_displacement_m)
                        /np.linalg.norm(seed.elastic_displacement_m))
    if cache_error > 1e-12:
        raise RuntimeError("cached-spectrum experiment differs from accepted elasticity")
    fast_operator = ElasticityOperator(
        tuple(p.shape), problem.dx_m, problem.dy_m, problem.e_star_pa
    )
    fast_error = float(np.linalg.norm(fast_operator(p)-seed.elastic_displacement_m)
                       /np.linalg.norm(seed.elastic_displacement_m))
    if fast_error > 1e-12:
        raise RuntimeError("existing FFT-friendly cached elasticity differs from accepted root")
    info.update(
        reference_scaled_mass_maximum=float(abs(ref).max()*factor),
        candidate_scaled_mass_maximum_on_reference_root=float(abs(new).max()*factor),
        relative_flux_identity_error=error,
        candidate_is_converged=False, elastic_direction_checks=checks,
        front_selection=front_selection,
        reference_residual_timing=timed(
            lambda: face_divergence(x, y, *mass_fluxes(x, y, h, p, t, f, 2.))),
        candidate_residual_timing=timed(
            lambda: face_divergence(x, y, *donor_fluxes(x, y, h, p, t, f, 2.))),
        existing_newton_jvp_timing=timed(lambda: system.jacobian_vector(state, known)),
        existing_residual_timing=timed(lambda: system.residual(state)),
        existing_film_direction_timing=timed(lambda: system.film_direction(state, known)),
        cached_spectrum_elasticity_timing=timed(cached_film_direction),
        cached_spectrum_relative_error=cache_error,
        existing_fast_operator_timing=timed(lambda: fast_operator(p)),
        existing_fast_operator_relative_error=fast_error,
        existing_fast_operator_workspace_estimate_bytes=fast_operator.estimated_workspace_bytes,
        existing_fast_operator_fft_shape=list(fast_operator.fft_shape),
        preconditioner_assembly_s=assembly_s, factors=factors, storage=storage,
    )
    np.savez_compressed(directory / f"diagnostic_{p.shape[0]}x{p.shape[1]}.npz",
                        x_m=x, y_m=y, reference_mass_si=ref, candidate_mass_si=new,
                        algebraic_difference_si=difference)
    return info


def main(sources, directory):
    directory.mkdir(parents=True, exist_ok=False)
    report = {"status": "incomplete", "runtime": provenance(), "smooth": [], "loaded": [],
              "case": Circular200N().parameters(),
              "scope": "isolated first-order candidate; no candidate nonlinear solve or cap change"}
    save_json(directory / "summary.json", report)
    for varying in (False, True):
        rows = []
        for n in (33, 65, 129, 257):
            rows.append({"n": n, **smooth_errors(n, varying_content=varying)})
        for scheme in ("reference", "donor"):
            for old, new in zip(rows, rows[1:], strict=False):
                new[scheme+"_observed_order"] = float(np.log2(old[scheme]/new[scheme]))
        report["smooth"].append({"varying_content": varying, "rows": rows})
    fig, axes = figure("Manufactured smooth-source transport consistency",
                       "Prescribed SI fields, constant material limit; not an EHL solution", rows=1)
    ax = np.atleast_1d(axes)[0]
    for case in report["smooth"]:
        for scheme in ("reference", "donor"):
            ax.loglog([1/(r["n"]-1) for r in case["rows"]],
                      [r[scheme] for r in case["rows"]], "o-",
                      label=f"{scheme}; varying content={case['varying_content']}")
    ax.set_xlabel("Grid spacing / domain length")
    ax.set_ylabel("Relative L2 divergence error")
    ax.legend()
    save(fig, directory, "smooth_consistency")
    for source in sources:
        report["loaded"].append(accepted_case(source, directory))
        save_json(directory / "summary.json", report)
    report["status"] = "independent candidate and reference resource checks passed"
    save_json(directory / "summary.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directories", nargs="+", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    main(args.input_directories, args.output_directory)
