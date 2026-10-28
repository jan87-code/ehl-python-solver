"""Explicit unaccepted fixed-spacing reservoir extension of an audited root.

Source values and SI geometry are preserved. Pressure is zero in added cells;
content uses edge-value extrapolation only as an INITIAL GUESS. No new mass
root or far-field boundary condition is inferred from this initialization.
"""

from dataclasses import replace

import numpy as np

from ehl_solver.property_load_newton import PropertyLoadSystem, verify_seed
from ehl_solver.property_transfer import PropertyLoadGuess


def extended_reservoir_problem(problem, *, west=0, east=0, south=0, north=0):
    """Extend by integer original cells, retaining exact original coordinates.

    The existing 20,000 physical-node guard is retained before kernel allocation.
    This is a finite-reservoir geometry change, not an infinite-domain boundary.
    """
    counts = (west, east, south, north)
    if any(not isinstance(n, int) or isinstance(n, bool) or not 0 <= n <= 8 for n in counts):
        raise ValueError("extension counts must be integers in 0..8")
    if not any(counts):
        raise ValueError("at least one boundary must extend")
    if (len(problem.x_m)+west+east)*(len(problem.y_m)+south+north) > 20000:
        raise ValueError("extended reservoir exceeds original 20000-node cap")

    def extend(axis, before, after):
        spacing = axis[1]-axis[0]
        return np.r_[axis[0]-spacing*np.arange(before, 0, -1), axis,
                     axis[-1]+spacing*np.arange(1, after+1)]

    return replace(problem, x_m=extend(problem.x_m, west, east),
                   y_m=extend(problem.y_m, south, north))


def extend_property_guess(source_problem, target_problem, fluid, speed_m_s, *, seed, seed_offset_m):
    """Audited source -> force-preserving unaccepted zero-pressure extension.

    All original grid nodes must be retained exactly at the same spacing.
    Added cells contribute zero initial pressure, not a truncated elastic
    source. Full target FFT reconstructs film including every source cell.
    Offset, pressure/content bounds and physical model are never adjusted.
    """
    for name in (
        "radius_x_m", "radius_y_m", "e_star_pa", "length_scale_m", "film_scale_m",
        "pressure_scale_pa", "reynolds_parameter", "inlet_fractional_film_content", "film_floor_m",
    ):
        if not np.isclose(getattr(source_problem, name), getattr(target_problem, name),
                           rtol=1e-12, atol=0):
            raise ValueError("extension requires identical geometry, supply and SI scales")
    padding = []
    slices = []
    for old, new in ((source_problem.y_m, target_problem.y_m),
                     (source_problem.x_m, target_problem.x_m)):
        starts = np.flatnonzero(new == old[0])
        if (len(starts) != 1 or not np.isclose(new[1]-new[0], old[1]-old[0], rtol=1e-10, atol=0)):
            raise ValueError("same-spacing exact-node extension required")
        start = int(starts[0])
        stop = start+len(old)
        if stop > len(new) or not np.array_equal(new[start:stop], old):
            raise ValueError("all original coordinates must be retained exactly")
        padding.append((start, len(new)-stop))
        slices.append(slice(start, stop))
    source = PropertyLoadSystem(source_problem, fluid, speed_m_s, 1.)
    source_state = verify_seed(source, seed, seed_offset_m)
    accepted = source.solution(source_state).solution
    target = PropertyLoadSystem(target_problem, fluid, speed_m_s, accepted.load_n)
    p = np.pad(seed.pressure_pa, padding, mode="constant", constant_values=0)
    theta = np.pad(seed.fractional_film_content, padding, mode="edge")
    guess = PropertyLoadGuess(p, theta, float(seed_offset_m), {})
    initial = target.solution(target.pack(guess, seed_offset_m)).solution
    overlap = tuple(slices)
    film_error = float(abs(initial.film_thickness_m[overlap]-accepted.film_thickness_m).max())
    force_error = abs(initial.load_n/accepted.load_n-1)
    if force_error > 1e-12 or film_error > 1e-18:
        raise ValueError("independent force/overlap-film extension audit failed")
    diagnostics = {
        "accepted": False, "source_load_n": accepted.load_n, "initial_load_n": initial.load_n,
        "relative_force_difference": force_error, "overlap_film_error_m": film_error,
        "source_shape_ny_nx": list(source_problem.shape),
        "target_shape_ny_nx": list(target_problem.shape),
        "padding_yx": padding, "offset_adjustment_m": 0., "initial_audit": initial.audit,
        "initialization": "zero added pressure; edge content extrapolation; no root implied",
    }
    return PropertyLoadGuess(initial.pressure_pa, initial.fractional_film_content,
                             float(seed_offset_m), diagnostics)
