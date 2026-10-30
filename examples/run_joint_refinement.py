"""Bounded 200 N donor joint refinement; SI, arrays (ny,nx), +x entrainment.

The default cap stays 20k; larger experiments require --maximum-physical-nodes
30000 and a <1 GiB sparse-factor/Krylov memory preflight. No equations change.
"""

import argparse
import json
import resource
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import convolve2d
from scipy.sparse.linalg import splu

from ehl_solver.benchmark_200n import Circular200N
from ehl_solver.donor_load import DonorLoadSystem, donor_mass_audit
from ehl_solver.elasticity import cell_integrated_kernel
from ehl_solver.property_load_newton import PropertyLoadNewtonError, _solve_property_load_state
from ehl_solver.property_reynolds import admissible
from ehl_solver.prototype import PointContactCase, audit_history
from ehl_solver.prototype_cli import clean_json, digest, load_run, save_state


def joint_shape(ny, maximum_physical_nodes=20000):
    """Keep dx/dy=2/3 on a square domain, with odd centered grid counts."""
    if type(ny) is not int or ny < 5 or ny % 4 != 1:
        raise ValueError("ny must be 4k+1, >=5")
    nx = 3 * (ny - 1) // 2 + 1
    if maximum_physical_nodes not in (20000, 30000) or ny * nx > maximum_physical_nodes:
        raise ValueError("Joint grid exceeds explicit 20k/30k physical-node allowance")
    return ny, nx


def sampled_minimum(x_m, y_m, film_m):
    """Raw SI minimum plus a local quadratic diagnostic; never alter a field.

    The fit is only a sampling diagnostic on the existing discrete solution,
    not a corrected physical prediction or a mesh-error estimate.
    """
    x, y, h = (np.asarray(v, dtype=np.float64) for v in (x_m, y_m, film_m))
    if (
        x.ndim != 1 or y.ndim != 1 or min(x.size, y.size) < 3
        or h.shape != (y.size, x.size)
        or not all(np.all(np.isfinite(v)) for v in (x, y, h))
        or np.any(np.diff(x) <= 0) or np.any(np.diff(y) <= 0)
    ):
        raise ValueError("Finite SI axes increasing and film array (ny,nx) required")
    iy, ix = np.unravel_index(np.argmin(h), h.shape)
    result = dict(raw_minimum_m=float(h[iy, ix]), index_yx=[int(iy), int(ix)],
                  location_yx_m=[float(y[iy]), float(x[ix])],
                  fit_status="boundary minimum; no two-sided fit")
    if iy in (0, y.size - 1) or ix in (0, x.size - 1):
        return result
    dx, dy = (x[ix+1] - x[ix-1]) / 2, (y[iy+1] - y[iy-1]) / 2
    u, v = np.meshgrid((x[ix-1:ix+2]-x[ix])/dx, (y[iy-1:iy+2]-y[iy])/dy)
    design = np.c_[np.ones(9), u.ravel(), v.ravel(),
                   u.ravel()**2 / 2, (u*v).ravel(), v.ravel()**2 / 2]
    coefficients = np.linalg.lstsq(design, (h[iy-1:iy+2, ix-1:ix+2]-h[iy, ix]).ravel(),
                                   rcond=None)[0]
    c, gx, gy, xx, xy, yy = coefficients
    curvature = np.array([[xx, xy], [xy, yy]])
    if np.min(np.linalg.eigvalsh(curvature)) <= 0:
        result['fit_status'] = 'nonpositive fitted curvature; no fitted minimum'
        return result
    offset = -np.linalg.solve(curvature, [gx, gy])
    if np.max(abs(offset)) > 1:
        result['fit_status'] = 'stationary point outside stencil; no fitted minimum'
        return result
    value = h[iy, ix] + c + np.dot([gx, gy], offset) + offset @ curvature @ offset / 2
    result.update(fit_status='local quadratic diagnostic only', fitted_minimum_m=float(value),
                  fitted_location_yx_m=[float(y[iy]+offset[1]*dy), float(x[ix]+offset[0]*dx)],
                  fitted_minus_raw_m=float(value-h[iy, ix]),
                  fit_rms_m=float(np.linalg.norm(design@coefficients -
                      (h[iy-1:iy+2, ix-1:ix+2]-h[iy, ix]).ravel())/3))
    return result


def independent_audit(system, root):
    problem, case = system.problem, Circular200N()
    s = root.solution
    p, theta = s.pressure_pa, s.fractional_film_content
    w = convolve2d(p, cell_integrated_kernel(problem.shape, problem.dx_m, problem.dy_m,
                                           case.contact_modulus_pa), mode='same')
    h = root.central_film_offset_m + problem.rigid_geometry_m + w
    mass = donor_mass_audit(problem.x_m, problem.y_m, h, p, theta, case.fluid(),
                           case.entrainment_m_s, length_scale_m=problem.length_scale_m,
                           film_scale_m=problem.film_scale_m,
                           pressure_scale_pa=problem.pressure_scale_pa)
    force = float(problem.load_n(p))
    error = float(np.max(abs(h-s.film_thickness_m)))
    displacement_error = float(np.max(abs(w-s.elastic_displacement_m)))
    feasible = bool(
        admissible(mass, p, theta, problem.pressure_scale_pa)
        and abs(force/case.load_n-1) <= 1e-8 and h.min() > problem.film_floor_m
        and np.all(np.r_[p[0], p[-1], p[:, 0], p[:, -1]] == 0)
        and np.all(theta[1:-1, 0] == 1)
        and max(error, displacement_error) <= 1e-10*problem.film_scale_m
    )
    return dict(numerically_accepted=feasible, mass=mass, force_n=force,
                force_relative_error=force/case.load_n-1,
                film_reconstruction_error_m=error,
                displacement_reconstruction_error_m=displacement_error,
                minimum=sampled_minimum(problem.x_m, problem.y_m, h),
                central_film_m=float(h[problem.shape[0]//2, problem.shape[1]//2]))


def run(seed, output, ny, maximum_physical_nodes=20000):
    shape = joint_shape(ny, maximum_physical_nodes)
    old_case, old_root, _ = load_run(seed)
    expected = PointContactCase()
    if old_case.physical_signature() != expected.physical_signature() or not np.allclose(
        [old_case.domain_half_width_x_m, old_case.domain_half_width_y_m],
        [expected.domain_half_width_x_m, expected.domain_half_width_y_m], rtol=1e-14, atol=0
    ):
        raise ValueError('Seed must be the same 200 N donor fluid and +/-3a domain')
    if any(new < old for new, old in zip(shape, (old_case.ny, old_case.nx), strict=True)):
        raise ValueError('Seed must not be finer than target')
    case = Circular200N()
    problem = case.problem(.5e-6, ny=shape[0], nx=shape[1], extent=3,
                           maximum_physical_nodes=maximum_physical_nodes)
    system = DonorLoadSystem(problem, case.fluid(), case.entrainment_m_s, case.load_n,
                             preconditioner='local', elasticity_mode='cached',
                             maximum_physical_nodes=maximum_physical_nodes)
    yy, xx = np.meshgrid(problem.y_m, problem.x_m, indexing='ij')
    points = np.c_[yy.ravel(), xx.ravel()]
    old = old_case.problem()

    def interpolate(a):
        return RegularGridInterpolator((old.y_m, old.x_m), a, bounds_error=True)(
            points).reshape(shape)

    p, theta = (interpolate(a) for a in
                (old_root.solution.pressure_pa, old_root.solution.fractional_film_content))
    state = system.pack(SimpleNamespace(pressure_pa=p, fractional_film_content=theta),
                        old_root.central_film_offset_m)
    initial = system.solution(state)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    source = out/'source/ehl_solver'
    source.mkdir(parents=True)
    hashes = {}
    for path in Path(__import__('ehl_solver').__file__).parent.glob('*.py'):
        shutil.copyfile(path, source/path.name)
        hashes[path.name] = digest(path)
    shutil.copyfile(__file__, out/'run_joint_refinement.py')
    report = dict(status='preflight', shape_ny_nx=shape, case=case.parameters(),
                  transport='donor', coordinates='(ny,nx), +x entrainment; SI',
                  dx_m=problem.dx_m, dy_m=problem.dy_m,
                  maximum_physical_nodes=maximum_physical_nodes,
                  default_prototype_cap_unchanged=20000,
                  maximum_corrections=30, maximum_krylov_cycles=10,
                  source_state_sha256=digest(Path(seed)/'solution.npz'),
                  source_sha256=hashes,
                  driver_sha256=digest(__file__),
                  initialization=dict(accepted=False, method='bilinear interpolation',
                      offset_adjustment_m=0, force_n=initial.solution.load_n),
                  iterations=[], physical_validation_complete=False)

    def write():
        (out/'summary.json').write_text(json.dumps(clean_json(report), indent=2, allow_nan=False))

    write()
    matrix = system.frozen_matrix(state)
    lu = splu(matrix)

    def payload(m):
        return m.data.nbytes+m.indices.nbytes+m.indptr.nbytes

    factor_bytes = payload(lu.L)+payload(lu.U)+lu.perm_c.nbytes+lu.perm_r.nbytes
    basis_bytes = 8*system.size*(60+3)*2
    estimate = (2*(payload(matrix)+factor_bytes+basis_bytes)
                + 2*system.base.elastic_operator.estimated_workspace_bytes + 160*np.prod(shape))
    known = np.random.default_rng(200193).normal(size=system.size)
    rhs = matrix@known
    residual = float(np.linalg.norm(matrix@lu.solve(rhs)-rhs)/np.linalg.norm(rhs))
    report['preflight'] = dict(estimated_peak_with_margin_bytes=int(estimate),
                              factor_bytes=factor_bytes, two_basis_bytes=basis_bytes,
                              known_vector_relative_residual=residual)
    write()
    if estimate >= 2**30 or residual >= 1e-8:
        report.update(status='rejected_preflight', reason='Memory or factor residual budget')
        write()
        return report
    del matrix, lu, known, rhs
    latest = {}

    def observe(iteration, raw, root):
        save_state(out/f'state_{iteration:03d}.npz', system, raw, root)
        latest['state'] = raw.copy()
        report['iterations'].append(dict(iteration=iteration, load_n=root.solution.load_n,
                                         audit=root.solution.audit))
        write()

    try:
        root = _solve_property_load_state(system, state, 30, 10, 1e-10, True,
                   active_lookahead=True, content_bound_pivots=True, state_observer=observe)
        report['status'] = 'accepted_numerical_root'
        history = root.solution.history
    except PropertyLoadNewtonError as exc:
        root, history = exc.last_root, exc.history
        report.update(status='rejected', reason=str(exc), diagnostics=exc.diagnostics)
    report['history'] = history
    write()  # Preserve rejected histories even if a subsequent audit raises.
    if root is not None:
        save_state(out/'solution.npz', system, latest['state'], root)
        report['state_sha256'] = digest(out/'solution.npz')
        report['independent_audit'] = independent_audit(system, root)
        write()
        report['history_audit'] = audit_history(history)
        if report['independent_audit']['numerically_accepted'] != (
            report['status'] == 'accepted_numerical_root'
        ):
            raise RuntimeError('Independent final acceptance differs from nonlinear solver')
    report['peak_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ny', type=int, default=113)
    parser.add_argument('--maximum-physical-nodes', type=int, choices=(20000, 30000), default=20000)
    args = parser.parse_args(argv)
    report = run(args.seed, args.output, args.ny, args.maximum_physical_nodes)
    print(json.dumps(clean_json({k: v for k, v in report.items()
                                if k not in ('history', 'iterations', 'source_sha256')}), indent=2))
    return 0 if report['status'] == 'accepted_numerical_root' else 1


if __name__ == '__main__':
    raise SystemExit(main())
