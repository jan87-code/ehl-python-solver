"""Re-audit and plot a fixed-ratio 200 N refinement study; never change fields."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from ehl_solver.benchmark_200n import Circular200N
from ehl_solver.donor_load import DonorLoadSystem
from ehl_solver.prototype import PointContactCase, audit_history
from ehl_solver.prototype_cli import clean_json, digest, load_run
from examples.run_joint_refinement import independent_audit, joint_shape, sampled_minimum


def read_run(folder, *, joint=False):
    folder = Path(folder)
    report = json.loads((folder/'summary.json').read_text())
    if joint:
        if report['status'] != 'accepted_numerical_root':
            raise ValueError('Joint run must be numerically accepted')
        shape = tuple(report['shape_ny_nx'])
        if joint_shape(shape[0], report['maximum_physical_nodes']) != shape:
            raise ValueError('Wrong joint grid ratio')
        if report['case'] != Circular200N().parameters() or report['transport'] != 'donor':
            raise ValueError('Wrong joint physical model')
        if digest(folder/'solution.npz') != report['state_sha256']:
            raise ValueError('State checksum mismatch')
        for name, checksum in report['source_sha256'].items():
            if digest(folder/'source/ehl_solver'/name) != checksum:
                raise ValueError('Source checksum mismatch')
        if digest(folder/'run_joint_refinement.py') != report['driver_sha256']:
            raise ValueError('Driver checksum mismatch')
        case = Circular200N()
        problem = case.problem(.5e-6, ny=shape[0], nx=shape[1], extent=3,
                               maximum_physical_nodes=report['maximum_physical_nodes'])
        system = DonorLoadSystem(problem, case.fluid(), 2, 200, preconditioner='local',
                                 elasticity_mode='cached',
                                 maximum_physical_nodes=report['maximum_physical_nodes'])
        with np.load(folder/'solution.npz', allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['x_m'], problem.x_m)
            np.testing.assert_array_equal(saved['y_m'], problem.y_m)
            root = system.solution(saved['state_dimensionless'])
            np.testing.assert_array_equal(root.solution.pressure_pa, saved['pressure_pa'])
            np.testing.assert_array_equal(root.solution.fractional_film_content, saved['theta'])
            np.testing.assert_array_equal(root.solution.film_thickness_m, saved['film_m'])
        audit = independent_audit(system, root)
    else:
        case, root, report = load_run(folder)
        expected = PointContactCase()
        if case.physical_signature() != expected.physical_signature() or not np.allclose(
            [case.domain_half_width_x_m, case.domain_half_width_y_m],
            [expected.domain_half_width_x_m, expected.domain_half_width_y_m], rtol=1e-14, atol=0
        ):
            raise ValueError('Wrong prototype physical model or domain')
        if joint_shape(case.ny) != (case.ny, case.nx):
            raise ValueError('Wrong prototype joint grid ratio')
        problem = case.problem()
        audit = dict(report['independent_audit'])
        audit['minimum'] = sampled_minimum(problem.x_m, problem.y_m, root.solution.film_thickness_m)
    if not audit['numerically_accepted']:
        raise ValueError('Independent numerical audit failed')
    history = audit_history(report['history'])
    return dict(folder=str(folder), shape_ny_nx=list(problem.shape),
                dx_m=problem.dx_m, dy_m=problem.dy_m, audit=audit, history_audit=history), dict(
                    x_m=problem.x_m, y_m=problem.y_m, pressure_pa=root.solution.pressure_pa,
                    film_m=root.solution.film_thickness_m)


def analyze(prototype, joint, output):
    runs = [read_run(p) for p in prototype] + [read_run(p, joint=True) for p in joint]
    runs.sort(key=lambda r: r[0]['shape_ny_nx'][0])
    if len(runs) < 2 or len({r[0]['shape_ny_nx'][0] for r in runs}) != len(runs):
        raise ValueError('At least two distinct joint meshes required')
    case = Circular200N()
    a = case.hertz_radius_m
    comparisons = []
    for (old, f), (new, g) in zip(runs[:-1], runs[1:], strict=True):
        yy, xx = np.meshgrid(f['y_m'], f['x_m'], indexing='ij')
        p = RegularGridInterpolator((g['y_m'], g['x_m']), g['pressure_pa'],
                                    bounds_error=True)(
                                        np.c_[yy.ravel(), xx.ravel()]).reshape(xx.shape)
        diff = p-f['pressure_pa']
        exit_mask = (xx >= .5*a) & (xx <= 1.5*a) & (abs(yy) <= 1.2*a)
        previous, current = old['audit'], new['audit']
        comparisons.append(dict(
            source=old['shape_ny_nx'], target=new['shape_ny_nx'],
            spacing_ratio=new['dx_m']/old['dx_m'],
            pressure_relative_l2=float(np.linalg.norm(diff)/np.linalg.norm(p)),
            exit_pressure_relative_l2=float(np.linalg.norm(diff[exit_mask])/
                                            np.linalg.norm(p[exit_mask])),
            minimum_relative_change=current['minimum']['raw_minimum_m']/
                                     previous['minimum']['raw_minimum_m']-1,
            central_relative_change=current['central_film_m']/previous['central_film_m']-1))
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    hd = PointContactCase().correlation()
    summary = dict(runs=[r[0] for r in runs], comparisons=comparisons,
        pressure_norm='finer pressure interpolated to coarse nodes; denominator finer norm',
        exit_region='0.5<=x/a<=1.5 and abs(y/a)<=1.2',
        minimum_diagnostic='raw sampled minimum; quadratic fit diagnostic never replaces field',
        hamrock_dowson_minimum_m=hd.minimum_film_m,
        hamrock_dowson_central_m=hd.central_film_m,
        physical_validation_complete=False, continuum_accuracy_established=False)
    (out/'study.json').write_text(json.dumps(clean_json(summary), indent=2, allow_nan=False))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for key, label, ref in [('minimum', 'Minimum', hd.minimum_film_m),
                            ('central', 'Central', hd.central_film_m)]:
        values = [(r[0]['audit']['minimum']['raw_minimum_m'] if key == 'minimum'
                   else r[0]['audit']['central_film_m'])*1e9 for r in runs]
        axes[0, 0].plot([r[0]['shape_ny_nx'][0] for r in runs], values, 'o-', label=label)
        axes[0, 0].axhline(ref*1e9, linestyle=':', label=f'HD {label.lower()}')
    axes[0, 0].set(xlabel='ny (nx=1.5(ny-1)+1)', ylabel='Film [nm]', title='Joint refinement')
    axes[0, 0].legend()
    for row, f in runs[-3:]:
        minimum = row['audit']['minimum']
        iy, ix = minimum['index_yx']
        # Plot both lobes along the x of the raw minimum, without interpolation.
        axes[0, 1].plot(f['y_m']/a, f['film_m'][:, ix]*1e9,
                        label=f"{row['shape_ny_nx']}, x/a={f['x_m'][ix]/a:.3f}")
        axes[1, 0].plot(f['x_m']/a, f['pressure_pa'][len(f['y_m'])//2]/1e9,
                        label=str(row['shape_ny_nx']))
    axes[0, 1].set(xlim=(-1.2, 1.2), ylim=(0, 800), xlabel='y/a', ylabel='Raw film [nm]',
                   title='Transverse profiles through each raw minimum')
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].set(xlim=(.5, 1.3), xlabel='x/a, +x entrainment', ylabel='Pressure [GPa]',
                   title='Centreline exit region')
    axes[1, 0].legend(fontsize=8)
    for row, _ in runs:
        minimum = row['audit']['minimum']
        y, x = minimum['location_yx_m']
        axes[1, 1].plot(x/a, abs(y)/a, 'o', label=str(row['shape_ny_nx']))
    axes[1, 1].set(xlabel='x_min/a', ylabel='abs(y_min)/a', title='Raw minimum location')
    axes[1, 1].legend(fontsize=8)
    fig.suptitle('200 N donor study: dx/dy=2/3; raw fields; accuracy remains under assessment')
    fig.savefig(out/'joint_refinement.png', dpi=180)
    plt.close(fig)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prototype', action='append', default=[], type=Path)
    parser.add_argument('--joint', action='append', default=[], type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    summary = analyze(args.prototype, args.joint, args.output)
    print(json.dumps(clean_json(summary['comparisons']), indent=2))


if __name__ == '__main__':
    main()
