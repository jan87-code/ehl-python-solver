"""Pressure-profile sampling and diagnostics on common physical coordinates."""

import numpy as np
from scipy.integrate import trapezoid


def _axis_and_profile(x_m, pressure_pa):
    x = np.asarray(x_m, dtype=np.float64)
    pressure = np.asarray(pressure_pa, dtype=np.float64)
    if (
        x.ndim != 1
        or pressure.shape != x.shape
        or x.size < 2
        or not np.all(np.isfinite(x))
        or not np.all(np.isfinite(pressure))
        or np.any(np.diff(x) <= 0)
    ):
        raise ValueError(
            "finite pressure profile and strictly increasing SI x coordinates required"
        )
    return x, pressure


def centerline_pressure_profile(x_m, y_m, pressure_pa):
    """Interpolate pressure at y=0 for each x; return x [m] and p [Pa]."""
    x = np.asarray(x_m, dtype=np.float64)
    y = np.asarray(y_m, dtype=np.float64)
    pressure = np.asarray(pressure_pa, dtype=np.float64)
    if (
        x.ndim != 1
        or y.ndim != 1
        or x.size < 2
        or y.size < 2
        or pressure.shape != (y.size, x.size)
        or not all(np.all(np.isfinite(a)) for a in (x, y, pressure))
        or np.any(np.diff(x) <= 0)
        or np.any(np.diff(y) <= 0)
        or y[0] > 0
        or y[-1] < 0
    ):
        raise ValueError("finite field on increasing SI axes spanning y=0 required")
    line = np.array([np.interp(0.0, y, pressure[:, i]) for i in range(x.size)])
    return x.copy(), line


def compare_pressure_profiles(
    reference_x_m,
    reference_pressure_pa,
    comparison_x_m,
    comparison_pressure_pa,
    *,
    window_m,
    sample_count=201,
):
    """Compare two centerline profiles over a requested physical x interval.

    The reported norms are descriptive only; no pass/fail tolerance is implied.
    """
    x_ref, p_ref = _axis_and_profile(reference_x_m, reference_pressure_pa)
    x_cmp, p_cmp = _axis_and_profile(comparison_x_m, comparison_pressure_pa)
    if (
        len(window_m) != 2
        or not all(np.isfinite(value) for value in window_m)
        or window_m[0] >= window_m[1]
        or window_m[0] < max(x_ref[0], x_cmp[0])
        or window_m[1] > min(x_ref[-1], x_cmp[-1])
        or type(sample_count) is not int
        or sample_count < 2
    ):
        raise ValueError("valid shared physical window and at least two samples required")
    x = np.linspace(window_m[0], window_m[1], sample_count)
    reference = np.interp(x, x_ref, p_ref)
    comparison = np.interp(x, x_cmp, p_cmp)
    difference = comparison - reference
    reference_norm = float(np.sqrt(trapezoid(reference**2, x)))
    difference_norm = float(np.sqrt(trapezoid(difference**2, x)))
    return {
        "x_m": x,
        "reference_pressure_pa": reference,
        "comparison_pressure_pa": comparison,
        "relative_l2": (
            difference_norm / reference_norm if reference_norm > 0 else None
        ),
        "maximum_absolute_difference_pa": float(np.max(np.abs(difference))),
    }


def centerline_peak_diagnostics(x_m, pressure_pa):
    """Report sampled maxima, adjacent pressures and positive-pressure extent."""
    x, pressure = _axis_and_profile(x_m, pressure_pa)
    peaks = [
        i
        for i in range(1, x.size - 1)
        if pressure[i] > pressure[i - 1] and pressure[i] >= pressure[i + 1]
    ]
    positive = np.flatnonzero(pressure > 0)
    maximum = int(np.argmax(pressure))
    return {
        "maximum": {
            "x_m": float(x[maximum]),
            "pressure_pa": float(pressure[maximum]),
        },
        "interior_local_maxima": [
            {
                "x_m": float(x[i]),
                "pressure_pa": float(pressure[i]),
                "left_adjacent_pressure_pa": float(pressure[i - 1]),
                "right_adjacent_pressure_pa": float(pressure[i + 1]),
            }
            for i in peaks
        ],
        "positive_pressure_extent_m": (
            [float(x[positive[0]]), float(x[positive[-1]])] if positive.size else None
        ),
        "grid_spacing_median_m": float(np.median(np.diff(x))),
    }


def outlet_spike_diagnostics(x_m, pressure_pa, *, window_m):
    """Measure a resolved interior peak using a shoulder-to-shoulder baseline.

    ``window_m`` must be chosen from the finest field being assessed and held
    fixed for comparisons. An absent or unresolved peak is reported explicitly.
    """
    x, pressure = _axis_and_profile(x_m, pressure_pa)
    if (
        len(window_m) != 2
        or not all(np.isfinite(value) for value in window_m)
        or window_m[0] >= window_m[1]
        or window_m[0] < x[0]
        or window_m[1] > x[-1]
    ):
        raise ValueError("spike window must be a finite physical interval within the profile")
    inside = (x > window_m[0]) & (x < window_m[1])
    window_x = np.r_[window_m[0], x[inside], window_m[1]]
    window_p = np.r_[
        np.interp(window_m[0], x, pressure),
        pressure[inside],
        np.interp(window_m[1], x, pressure),
    ]
    peak = int(np.argmax(window_p))
    result = {
        "identified": False,
        "window_m": [float(window_m[0]), float(window_m[1])],
        "shoulder_pressures_pa": [float(window_p[0]), float(window_p[-1])],
    }
    if peak == 0 or peak == window_x.size - 1:
        result["reason"] = "No interior peak in the supplied outlet window"
        return result
    if not (window_p[peak] > window_p[peak - 1] and window_p[peak] > window_p[peak + 1]):
        result["reason"] = "Outlet-window maximum is not a resolved strict local peak"
        return result
    baseline = np.interp(
        window_x[peak],
        [window_x[0], window_x[-1]],
        [window_p[0], window_p[-1]],
    )
    prominence = float(window_p[peak] - baseline)
    if prominence <= 0:
        result["reason"] = "Outlet peak has no positive prominence above shoulder baseline"
        return result
    relative = window_p - np.interp(
        window_x,
        [window_x[0], window_x[-1]],
        [window_p[0], window_p[-1]],
    )
    half_prominence = 0.5 * prominence
    left = peak
    while left > 0 and relative[left] > half_prominence:
        left -= 1
    right = peak
    while right < window_x.size - 1 and relative[right] > half_prominence:
        right += 1
    if relative[left] > half_prominence or relative[right] > half_prominence:
        result["reason"] = "Half-prominence crossing is outside the supplied window"
        return result

    def crossing(i0, i1):
        y0, y1 = relative[i0], relative[i1]
        if y1 == y0:
            return float(window_x[i0])
        return float(
            window_x[i0]
            + (half_prominence - y0) * (window_x[i1] - window_x[i0]) / (y1 - y0)
        )

    left_x = crossing(left, left + 1)
    right_x = crossing(right - 1, right)
    original_peak = int(np.argmin(np.abs(x - window_x[peak])))
    result.update(
        identified=True,
        peak_x_m=float(window_x[peak]),
        peak_pressure_pa=float(window_p[peak]),
        prominence_pa=prominence,
        left_adjacent_pressure_pa=float(pressure[max(0, original_peak - 1)]),
        right_adjacent_pressure_pa=float(pressure[min(x.size - 1, original_peak + 1)]),
        half_prominence_width_m=right_x - left_x,
        cells_across_width=(right_x - left_x) / float(np.median(np.diff(x))),
    )
    return result
