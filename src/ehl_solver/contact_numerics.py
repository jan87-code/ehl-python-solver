"""Backend-neutral constrained-vector operations shared by dry contact solvers."""

from __future__ import annotations

import numpy as np

from ehl_solver.backend import Backend


def project_nonnegative_fixed_sum(values, total: float, backend: Backend):
    """Euclidean projection onto nonnegative vectors with a fixed sum."""
    xp = backend.xp
    if backend.name == "numpy":
        ordered = np.sort(values)[::-1]
    else:
        ordered = xp.sort(values, descending=True).values
    cumulative = (
        xp.cumsum(ordered, dim=0) if backend.name == "torch" else np.cumsum(ordered)
    ) - total
    size = int(values.numel()) if backend.name == "torch" else values.size
    indices = (
        xp.arange(1, size + 1, dtype=values.dtype, device=backend.device)
        if backend.name == "torch"
        else np.arange(1, size + 1, dtype=np.float64)
    )
    candidates = ordered - cumulative / indices
    if backend.name == "numpy":
        active_count = int(np.flatnonzero(candidates > 0)[-1]) + 1
    else:
        active_count = int(xp.count_nonzero(candidates > 0).item())
    threshold = cumulative[active_count - 1] / active_count
    if backend.name == "torch":
        return xp.clamp_min(values - threshold, 0.0)
    return np.maximum(values - threshold, 0.0)


def positive_part(values, backend: Backend):
    if backend.name == "torch":
        return backend.xp.clamp_min(values, 0.0)
    return np.maximum(values, 0.0)
