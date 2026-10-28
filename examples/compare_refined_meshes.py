"""Matched-force mesh evidence; a three-point fit is not an accuracy certificate."""

import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from examples.audit_transport_wavelengths import load_root
from examples.run_scientific_validation import provenance, save_json
from examples.scientific_figures import figure, save


def apparent_order(spacings, values):
    """Fit Q(h)=Q0+C*h**p on three meshes only when monotone and resolvable."""
    h, q = np.asarray(spacings, dtype=float), np.asarray(values, dtype=float)
    if (h.shape != (3,) or q.shape != (3,) or not np.all(np.isfinite(np.r_[h, q]))
            or np.any(h <= 0) or np.any(np.diff(h) >= 0)):
        raise ValueError("three finite values and decreasing positive spacings required")
    d = np.diff(q)
    if d[0]*d[1] <= 0 or abs(d).min() <= 100*np.finfo(float).eps*max(abs(q).max(), 1e-300):
        return None
    ratio = d[0]/d[1]
    scaled = h/h[-1]

    def equation(p):
        return (scaled[0]**p-scaled[1]**p)/(scaled[1]**p-1)-ratio

    if equation(.05)*equation(8.) >= 0:
        return None
    order = brentq(equation, .05, 8., xtol=1e-12)
    c = (q[-2]-q[-1])/(scaled[-2]**order-1)
    return {"apparent_order": float(order), "three_point_extrapolation": float(q[-1]-c),
            "finest_estimated_difference": float(c),
            "qualified": False, "reason": "three grids only; extrema/fronts may move"}


def main(sources, directory):
    directory.mkdir(parents=True, exist_ok=False)
    rows, fields = [], []
    for source in sources:
        system, state, seed, info = load_root(source, maximum_physical_nodes=30000)
        problem = system.problem
        low = np.unravel_index(np.argmin(seed.film_thickness_m), problem.shape)
        peak = np.unravel_index(np.argmax(seed.pressure_pa), problem.shape)
        center = (problem.shape[0]//2, problem.shape[1]//2)
        row = {**info, "dx_m": problem.dx_m, "dy_m": problem.dy_m,
               "minimum_film_m": float(seed.film_thickness_m.min()),
               "central_film_m": float(seed.film_thickness_m[center]),
               "peak_pressure_pa": float(seed.pressure_pa.max()),
               "minimum_location_xy_m": [float(problem.x_m[low[1]]), float(problem.y_m[low[0]])],
               "peak_location_xy_m": [float(problem.x_m[peak[1]]), float(problem.y_m[peak[0]])],
               "offset_m": float(state[-1]*problem.film_scale_m)}
        rows.append(row)
        fields.append((problem, seed))
    if len(rows) < 3 or any(rows[i]["dx_m"] <= rows[i+1]["dx_m"] for i in range(len(rows)-1)):
        raise ValueError("at least three successively finer matched grids required")
    comparisons = []
    for index in range(1, len(rows)):
        old, new = rows[index-1], rows[index]
        p, a = fields[index-1]
        q, b = fields[index]

        def shared(old_axis, new_axis, length_scale=p.length_scale_m):
            pairs = [(i, int(np.argmin(abs(new_axis-v)))) for i, v in enumerate(old_axis)]
            return np.array([(i, j) for i, j in pairs
                             if abs(old_axis[i]-new_axis[j]) < 1e-12*length_scale])

        iy, ix = shared(p.y_m, q.y_m), shared(p.x_m, q.x_m)
        before = a.pressure_pa[np.ix_(iy[:, 0], ix[:, 0])]
        after = b.pressure_pa[np.ix_(iy[:, 1], ix[:, 1])]
        comparisons.append({
            "from": old["shape_ny_nx"], "to": new["shape_ny_nx"],
            "shared_shape_ny_nx": [len(iy), len(ix)],
            "shared_pressure_relative_l2_difference": float(np.linalg.norm(after-before)
                                                            /np.linalg.norm(before)),
            **{key+"_relative_change": new[key]/old[key]-1
               for key in ("minimum_film_m", "central_film_m", "peak_pressure_pa")},
        })
    fits = {key: apparent_order([r["dx_m"] for r in rows[-3:]], [r[key] for r in rows[-3:]])
            for key in ("minimum_film_m", "central_film_m", "peak_pressure_pa")}
    fig, axes = figure("200 N matched physical domain: raw mesh profiles",
                       "Same material laws/force gates; no smoothing; mesh accuracy remains open",
                       rows=2)
    for p, s in fields:
        j = p.shape[0]//2
        axes[0].plot(p.x_m*1e3, s.pressure_pa[j]/1e6, "o-", markersize=2,
                     label=str(p.shape))
        axes[1].plot(p.x_m*1e3, s.film_thickness_m[j]*1e6, "o-", markersize=2,
                     label=str(p.shape))
    for ax in axes:
        ax.set_xlabel("x along entrainment [mm]")
        ax.legend()
        ax.set_xlim(-.4, .4)
    axes[0].set_ylabel("Pressure [MPa]")
    axes[1].set_ylabel("Film [um]")
    visible_film = max(float(s.film_thickness_m[p.shape[0]//2, abs(p.x_m) <= .4e-3].max())
                       for p, s in fields)
    axes[1].set_ylim(0, visible_film*1e6*1.05)
    save(fig, directory, "raw_profiles")
    save_json(directory / "summary.json", {
        "status": "independent matched-force comparison; asymptotic accuracy not established",
        "runtime": provenance(), "grids": rows, "comparisons": comparisons,
        "provisional_three_point_fits": fits,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directories", type=Path, nargs="+", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    main(args.input_directories, args.output_directory)
