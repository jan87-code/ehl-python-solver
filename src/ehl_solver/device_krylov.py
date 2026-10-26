"""Optional torch float64 restarted GMRES; only scalar diagnostics leave device.

The basis, Hessenberg matrix, Givens rotations and triangular solve remain on
the rhs device. Fixed left preconditioning and twice-applied classical
Gram-Schmidt (CGS2) support nonsymmetric real operators. max_iterations counts
Arnoldi steps, NOT restart cycles. No CPU or alternate-solver fallback occurs.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceGMRESResult:
    solution: object
    converged: bool
    reason: str
    iterations: int
    true_residual_norm: float
    true_relative_residual: float
    true_residual_history: tuple[float, ...]
    estimated_preconditioned_history: tuple[float, ...]


def device_gmres(
    matvec, rhs, *, preconditioner=None, initial=None, rtol=1e-8, atol=0.,
    restart=40, max_iterations=400, memory_limit_bytes=2**30,
):
    """Solve Ax=b with explicit termination and independent unpreconditioned checks.

    Inputs must already be real float64 torch vectors; operator/preconditioner
    outputs must retain their shape/dtype/device. Nonfinite operations produce
    a failed result. Programming/device runtime exceptions propagate. Workspace
    admission is conservative and includes two bases' worth of temporary room.
    """
    import torch

    if not isinstance(rhs, torch.Tensor) or rhs.dtype != torch.float64 or rhs.ndim != 1:
        raise ValueError("rhs must be a resident float64 torch vector")
    if rhs.numel() == 0 or not bool(torch.isfinite(rhs).all()):
        raise ValueError("rhs must be nonempty and finite")
    if any(not math.isfinite(t) or t < 0 for t in (rtol, atol)) or rtol > 1:
        raise ValueError("rtol and atol must be finite, nonnegative and rtol <= 1")
    if rtol == atol == 0:
        raise ValueError("at least one tolerance must be positive")
    if any(not isinstance(k, int) or isinstance(k, bool) or k < 1
           for k in (restart, max_iterations, memory_limit_bytes)):
        raise ValueError("restart, max_iterations and memory budget must be positive integers")
    n = rhs.numel()
    m = min(n, restart, max_iterations)
    estimate = 16 * n * (m + 5) + 32 * (m + 1)**2
    budget = memory_limit_bytes
    if rhs.is_cuda:
        budget = min(budget, torch.cuda.mem_get_info(rhs.device)[0] // 2)
    if estimate > budget:
        raise MemoryError("GMRES basis/workspace estimate exceeds device memory budget")
    x = torch.zeros_like(rhs) if initial is None else initial.clone()
    if x.shape != rhs.shape or x.dtype != rhs.dtype or x.device != rhs.device:
        raise ValueError("initial must match rhs shape, dtype and device")
    if not bool(torch.isfinite(x).all()):
        raise ValueError("initial must be finite")
    norm_b = float(torch.linalg.vector_norm(rhs))
    threshold = max(atol, rtol * norm_b)
    tiny = torch.finfo(rhs.dtype).tiny
    true_history, estimated_history = [], []
    iterations = 0

    def checked(call, vector):
        value = call(vector)
        if (not isinstance(value, torch.Tensor) or value.shape != rhs.shape
                or value.dtype != rhs.dtype or value.device != rhs.device):
            raise ValueError("operator must preserve vector shape, dtype and device")
        return value

    def result(reason, norm):
        return DeviceGMRESResult(
            x, reason == "converged", reason, iterations, norm,
            norm / max(norm_b, tiny), tuple(true_history), tuple(estimated_history),
        )

    def precondition(vector):
        return vector if preconditioner is None else checked(preconditioner, vector)

    residual = rhs - checked(matvec, x)
    norm = float(torch.linalg.vector_norm(residual))
    true_history.append(norm)
    if not math.isfinite(norm):
        return result("nonfinite_operator", norm)
    if norm <= threshold:
        return result("converged", norm)
    basis = rhs.new_zeros((n, m + 1))
    h = rhs.new_zeros((m + 1, m))
    rotations = torch.eye(m + 1, dtype=rhs.dtype, device=rhs.device)
    g = rhs.new_zeros(m + 1)
    while iterations < max_iterations:
        r = precondition(residual)
        beta = torch.linalg.vector_norm(r)
        if not math.isfinite(float(beta)):
            return result("nonfinite_preconditioner", norm)
        if float(beta) <= tiny:
            return result("preconditioner_breakdown", norm)
        basis[:, 0] = r / beta
        h.zero_()
        rotations.copy_(torch.eye(m + 1, dtype=rhs.dtype, device=rhs.device))
        g.zero_()
        g[0] = beta
        cycle_x = x
        for j in range(min(m, max_iterations - iterations)):
            w = precondition(checked(matvec, basis[:, j]))
            original_norm = float(torch.linalg.vector_norm(w))
            if not math.isfinite(original_norm):
                return result("nonfinite_operator_or_preconditioner", math.inf)
            q = basis[:, :j+1]
            # CGS2: reorthogonalize to suppress loss of basis orthogonality.
            coefficients = q.T @ w
            w = w - q @ coefficients
            second = q.T @ w
            w = w - q @ second
            h[:j+1, j] = coefficients + second
            subdiagonal = torch.linalg.vector_norm(w)
            h[j+1, j] = subdiagonal
            breakdown = float(subdiagonal) <= 32 * torch.finfo(rhs.dtype).eps * original_norm
            if not breakdown:
                basis[:, j+1] = w / subdiagonal
            # Apply accumulated rotations in one small device matvec, avoiding
            # O(restart**2) Python-dispatched scalar rotation kernels.
            h[:, j] = rotations @ h[:, j]
            denominator = torch.hypot(h[j, j], h[j+1, j])
            iterations += 1
            if float(denominator) <= tiny:
                return result("arnoldi_breakdown", norm)
            cosine = h[j, j] / denominator
            sine = h[j+1, j] / denominator
            row_a, row_b = rotations[j].clone(), rotations[j+1].clone()
            rotations[j] = cosine * row_a + sine * row_b
            rotations[j+1] = -sine * row_a + cosine * row_b
            h[j, j] = denominator
            h[j+1, j] = 0.
            g[j+1] = -sine * g[j]
            g[j] = cosine * g[j]
            estimated = abs(float(g[j+1]))
            estimated_history.append(estimated)
            # Estimate only triggers a true check, never establishes convergence.
            if (estimated <= rtol * float(beta) or breakdown or j == m - 1
                    or iterations == max_iterations):
                y = torch.linalg.solve_triangular(
                    h[:j+1, :j+1], g[:j+1, None], upper=True
                )[:, 0]
                x = cycle_x + basis[:, :j+1] @ y
                residual = rhs - checked(matvec, x)
                norm = float(torch.linalg.vector_norm(residual))
                true_history.append(norm)
                if not math.isfinite(norm):
                    return result("nonfinite_true_residual", norm)
                if norm <= threshold:
                    return result("converged", norm)
                if breakdown:
                    return result("arnoldi_breakdown", norm)
                # Restart from the independently recomputed residual.
                break
    return result("iteration_limit", norm)
