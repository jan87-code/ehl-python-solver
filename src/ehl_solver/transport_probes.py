"""Bounded static wavelength/front probes; production JFO equations are unchanged.

Public fields: pressure Pa, content dimensionless, axes/contact radius m,
array order (ny,nx), entrainment +x. Feasibility means Newton field bounds,
not that perturbed diagnostic states satisfy mass or complementarity.
"""

import numpy as np

from ehl_solver.transport_audit import pressure_mode_directions


def minimum_directional(a, b, da, db):
    """True one-sided derivative of min(a,b), using EXACT equality at ties.

    A selected generalized Jacobian at a tie need not equal this derivative
    for every direction. No near-zero value is rounded into a tie.
    """
    a, b, da, db = [np.asarray(v, dtype=np.float64) for v in (a, b, da, db)]
    if any(v.shape != a.shape or not np.all(np.isfinite(v)) for v in (b, da, db)):
        raise ValueError("finite same-shaped arguments required")
    if not np.all(np.isfinite(a)):
        raise ValueError("finite same-shaped arguments required")
    return np.where(a < b, da, np.where(b < a, db, np.minimum(da, db)))


def resolved_pressure_probes(x_m, y_m, pressure_pa, contact_radius_m, *,
                             wavelength_factors=(1., .5, .25), minimum_cells=4.,
                             amplitude_pa=1e6):
    """Windowed force-free sine/cosine carriers at fixed physical wavelengths.

    Carriers with fewer than minimum_cells per wavelength are explicitly
    omitted. Windowing and pressure support modify the spectrum; these are
    not eigenfunctions or convergence estimates. Two grid-relative Nyquist
    references are labelled separately, never called resolved carriers.
    """
    reference = pressure_mode_directions(x_m, y_m, pressure_pa, contact_radius_m,
                                         amplitude_pa=amplitude_pa)
    if (not np.isfinite(minimum_cells) or minimum_cells < 4
            or not wavelength_factors or any(not np.isfinite(f) or f <= 0
                                             for f in wavelength_factors)):
        raise ValueError("positive wavelength factors and minimum_cells >=4 required")
    if len(set(wavelength_factors)) != len(wavelength_factors):
        raise ValueError("duplicate wavelengths would overwrite probes")
    x, y, p = np.asarray(x_m), np.asarray(y_m), np.asarray(pressure_pa)
    xx, yy = np.meshgrid(x, y)
    window = np.exp(-2*((xx/contact_radius_m)**4+(yy/contact_radius_m)**4))
    window *= p > .1*p.max()
    window[[0, -1]] = 0
    window[:, [0, -1]] = 0
    probes, omitted = {}, []
    for axis, coords, spacing in (("x", xx, x[1]-x[0]), ("y", yy, y[1]-y[0])):
        probes["nyquist_"+axis] = {
            "pressure_pa": reference["alternating_"+axis], "content": np.zeros_like(p),
            "metadata": {"kind": "grid_relative_nyquist_reference", "axis": axis,
                         "wavelength_m": float(2*spacing), "cells_per_wavelength": 2.,
                         "resolved_carrier": False},
        }
        for factor in wavelength_factors:
            wavelength = factor*contact_radius_m
            cells = wavelength/spacing
            meta = {"kind": "fixed_physical_windowed_carrier", "axis": axis,
                    "wavelength_m": float(wavelength), "wavelength_over_a": factor,
                    "cells_per_wavelength": float(cells), "resolved_carrier": True}
            if cells < minimum_cells*(1-1e-12):
                omitted.append({**meta, "resolved_carrier": False,
                                "reason": "insufficient cells per physical wavelength"})
                continue
            for phase, trig in (("cos", np.cos), ("sin", np.sin)):
                pattern = trig(2*np.pi*coords/wavelength)
                direction = window*(pattern-np.sum(window*pattern)/window.sum())
                peak = abs(direction).max()
                if peak <= 64*np.finfo(float).eps:
                    omitted.append({**meta, "phase": phase, "reason": "degenerate support"})
                    continue
                name = f"physical_{axis}_{factor:g}a_{phase}"
                probes[name] = {"pressure_pa": amplitude_pa*direction/peak,
                                "content": np.zeros_like(p),
                                "metadata": {**meta, "phase": phase}}
    return probes, omitted


def copy_content_boundary_direction(direction):
    """Apply fixed west and copied east/north/south content-direction boundaries."""
    dt = np.asarray(direction, dtype=np.float64)
    if dt.ndim != 2 or min(dt.shape) < 3 or not np.all(np.isfinite(dt)):
        raise ValueError("finite 2-D content direction required")
    out = dt.copy()
    out[1:-1, 0] = 0
    out[1:-1, -1] = out[1:-1, -2]
    out[0], out[-1] = out[1], out[-2]
    return out


def active_front_probes(pressure_pa, theta, pressure_scale_pa, *, amplitude_pa=1e6,
                        content_amplitude=.01):
    """Representative +x rupture/reformation probes, nearest transverse center.

    Selection requires distinct full/cavitated margins; unresolved/near-tied
    cells are not relabelled. Each face gets force-free pressure transfer,
    content loss on its full cell, and content gain on its cavitated cell.
    An exact tie, if present, gets a separate feasible content-loss direction.
    """
    p, t = np.asarray(pressure_pa, dtype=np.float64), np.asarray(theta, dtype=np.float64)
    if (p.ndim != 2 or min(p.shape) < 3 or t.shape != p.shape
            or not np.all(np.isfinite(p)) or not np.all(np.isfinite(t))
            or not np.isfinite(pressure_scale_pa) or pressure_scale_pa <= 0
            or not np.isfinite(amplitude_pa) or amplitude_pa <= 0
            or not np.isfinite(content_amplitude) or not 0 < content_amplitude < 1
            or p.min() < -1e-10*pressure_scale_pa
            or t.min() < -1e-10 or t.max() > 1+1e-10):
        raise ValueError("original pressure/content bounds and positive SI scales required")
    P, D = p/pressure_scale_pa, 1-t
    full = (P > 1e-6) & (D < 1e-8)
    cav = (D > 1e-6) & (abs(P) < 1e-8)
    probes = {}
    metadata = {"exact_interior_ties": int(np.sum(P[1:-1, 1:-1] == D[1:-1, 1:-1])),
                "selected_faces": {}, "missing_faces": []}
    for kind, mask in (("rupture", full[:, :-1] & cav[:, 1:]),
                       ("reformation", cav[:, :-1] & full[:, 1:])):
        # Both cells must be unknown interior nodes.
        mask[[0, -1], :] = False
        mask[:, 0] = mask[:, -1] = False
        candidates = np.argwhere(mask)
        if not len(candidates):
            metadata["missing_faces"].append(kind)
            continue
        j, i = min(candidates.tolist(), key=lambda ji: (abs(ji[0]-p.shape[0]//2),
                                                       abs(ji[1]-p.shape[1]//2)))
        left, right = (j, i), (j, i+1)
        full_cell, cav_cell = (left, right) if kind == "rupture" else (right, left)
        metadata["selected_faces"][kind] = {"full_yx": full_cell, "cavitated_yx": cav_cell}
        amplitude = min(amplitude_pa, .25*p[full_cell])
        dp, dt = np.zeros_like(p), np.zeros_like(t)
        dp[full_cell], dp[cav_cell] = -amplitude, amplitude
        probes[kind+"_pressure_transfer"] = {
            "pressure_pa": dp, "content": dt,
            "metadata": {"kind": kind, "action": "force_free_pressure_transfer"},
        }
        for action, cell, value in (
            ("full_content_loss", full_cell, -content_amplitude),
            ("cavitated_content_gain", cav_cell, min(content_amplitude, .25*D[cav_cell])),
        ):
            dt = np.zeros_like(t)
            dt[cell] = value
            probes[kind+"_"+action] = {
                "pressure_pa": np.zeros_like(p), "content": copy_content_boundary_direction(dt),
                "metadata": {"kind": kind, "action": action, "cell_yx": cell},
            }
    ties = np.argwhere((P == D)[1:-1, 1:-1]) + 1
    if len(ties):
        j, i = min(ties.tolist(), key=lambda ji: (abs(ji[0]-p.shape[0]//2),
                                                 abs(ji[1]-p.shape[1]//2)))
        dt = np.zeros_like(t)
        dt[j, i] = -content_amplitude
        probes["exact_tie_content_loss"] = {
            "pressure_pa": np.zeros_like(p), "content": copy_content_boundary_direction(dt),
            "metadata": {"kind": "exact_tie", "cell_yx": [j, i]},
        }
    return probes, metadata
