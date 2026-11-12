"""Hybrid donor/JFO correction: CUDA matrix-free products, CPU SuperLU inverse."""

import math
import time
from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import splu

from ehl_solver.backend import as_float64, make_backend
from ehl_solver.cavitating_newton import CavitatingLinearError
from ehl_solver.device_jfo import DeviceDonorJFO
from ehl_solver.device_krylov import device_gmres


class CPUBlockSchurPreconditioner:
    """Apply the existing frozen local CPU preconditioner to a device vector."""

    def __init__(self, matrix, n, device, *, schur_tolerance=1e-10):
        import torch

        if not math.isfinite(schur_tolerance) or schur_tolerance <= 0:
            raise ValueError("schur_tolerance must be finite and positive")
        started = time.perf_counter()
        self.n = n
        self.device = torch.empty((), device=device).device
        self.size = 2 * n + 1
        self.field_factor = None
        self.full_factor = None
        self.used_bordered_fallback = False
        self.schur_indicator = None
        self.field_setup_failures = []
        self.application_count = 0
        self.device_to_host_seconds = 0.0
        self.host_to_device_seconds = 0.0
        self.cpu_application_seconds = 0.0
        self.application_wall_seconds = 0.0
        self.synchronization_seconds = 0.0
        self.device_to_host_bytes = 0
        self.host_to_device_bytes = 0
        self._torch = torch

        field_matrix = matrix[:2 * n, :2 * n].tocsc()
        try:
            self.field_factor = splu(field_matrix)
            load_column = matrix[:2 * n, -1].toarray().ravel()
            self.load_row = matrix[-1, :2 * n].toarray().ravel()
            self.load_response = self.field_factor.solve(load_column)
            denominator = -float(self.load_row @ self.load_response)
            scale = float(
                np.linalg.norm(self.load_row) * np.linalg.norm(self.load_response)
            )
            self.schur_indicator = abs(denominator) / max(
                scale, np.finfo(np.float64).tiny
            )
            if (
                not math.isfinite(denominator)
                or self.schur_indicator <= schur_tolerance
            ):
                raise ArithmeticError("CPU bordered Schur complement is unsafe")
            self.load_denominator = denominator
        except (RuntimeError, ArithmeticError, ValueError) as exc:
            self.field_setup_failures.append(str(exc))
            self.field_factor = None
            self.full_factor = splu(matrix)
            self.used_bordered_fallback = True
            self.load_row = None
            self.load_response = None
            self.load_denominator = None
        self.setup_seconds = time.perf_counter() - started

    def _synchronize(self):
        if self.device.type == "cuda":
            started = time.perf_counter()
            self._torch.cuda.synchronize(self.device)
            self.synchronization_seconds += time.perf_counter() - started

    def __call__(self, rhs):
        torch = self._torch
        if (
            not isinstance(rhs, torch.Tensor)
            or rhs.shape != (self.size,)
            or rhs.dtype != torch.float64
            or rhs.device != self.device
        ):
            raise ValueError("bordered float64 rhs must match the preconditioner device")

        wall_started = time.perf_counter()
        self._synchronize()
        transfer_started = time.perf_counter()
        host_rhs = rhs.detach().cpu().numpy()
        self.device_to_host_seconds += time.perf_counter() - transfer_started
        self.device_to_host_bytes += int(host_rhs.nbytes)

        cpu_started = time.perf_counter()
        if self.used_bordered_fallback:
            host_solution = self.full_factor.solve(host_rhs)
        else:
            field_solution = self.field_factor.solve(host_rhs[:2 * self.n])
            h0_direction = (
                host_rhs[-1] - self.load_row @ field_solution
            ) / self.load_denominator
            field_solution -= self.load_response * h0_direction
            host_solution = np.r_[field_solution, h0_direction]
        self.cpu_application_seconds += time.perf_counter() - cpu_started
        if not np.all(np.isfinite(host_solution)):
            raise FloatingPointError("CPU preconditioner returned nonfinite values")

        self._synchronize()
        transfer_started = time.perf_counter()
        result = torch.as_tensor(
            host_solution, dtype=torch.float64, device=self.device
        )
        self._synchronize()
        self.host_to_device_seconds += time.perf_counter() - transfer_started
        self.host_to_device_bytes += int(host_solution.nbytes)
        self.application_wall_seconds += time.perf_counter() - wall_started
        self.application_count += 1
        return result


@dataclass(frozen=True)
class HybridJFOCorrection:
    direction: np.ndarray
    true_linear_residual: float
    linear_iterations: int
    used_bordered_fallback: bool
    schur_indicator: float | None
    linear_true_residuals: tuple[float, ...]
    krylov_histories: tuple[tuple[float, ...], ...]
    field_setup_failures: tuple[str, ...]
    timings: dict


class HybridJFOCorrectionSolver:
    """CUDA donor/JFO operator and resident GMRES with CPU SuperLU preconditioning.

    The outer nonlinear solver remains the existing CPU implementation. Only
    the pressure/content/load Krylov vectors and matrix-free residual/JVP live
    on the requested torch device.
    """

    def __init__(
        self,
        case,
        backend=None,
        *,
        workspace_limit_bytes=None,
        memory_limit_bytes=None,
    ):
        if workspace_limit_bytes is None and memory_limit_bytes is None:
            workspace_limit_bytes = 2**30
        elif workspace_limit_bytes is None:
            workspace_limit_bytes = memory_limit_bytes
        elif memory_limit_bytes is not None:
            raise ValueError("specify only one workspace limit")
        if type(workspace_limit_bytes) is not int or workspace_limit_bytes < 1:
            raise ValueError("workspace_limit_bytes must be a positive integer")
        self.backend = make_backend("torch", "cuda") if backend is None else backend
        if self.backend.name != "torch":
            raise ValueError("hybrid JFO correction requires a torch backend")
        initialization_started = time.perf_counter()
        self.evaluator = DeviceDonorJFO(
            case, self.backend, memory_limit_bytes=workspace_limit_bytes
        )
        self.initialization_seconds = time.perf_counter() - initialization_started
        self.workspace_limit_bytes = workspace_limit_bytes
        self.memory_limit_bytes = workspace_limit_bytes

    def __call__(
        self,
        model,
        state,
        *,
        krylov_rtol=1e-9,
        maximum_krylov_cycles=10,
        gmres_restart=60,
        schur_tolerance=1e-10,
    ):
        import torch

        if (
            model.probe.transport_fraction != 0
            or model.active_full is None
            or model.n != self.evaluator.n
            or model.size != self.evaluator.size
        ):
            raise ValueError("hybrid correction requires a donor endpoint active-set model")
        state = np.asarray(state, dtype=np.float64)
        model._check_state(state)
        if (
            not math.isfinite(krylov_rtol)
            or not 0 < krylov_rtol < 1
            or type(maximum_krylov_cycles) is not int
            or maximum_krylov_cycles < 1
            or type(gmres_restart) is not int
            or not 1 <= gmres_restart <= 60
            or not math.isfinite(schur_tolerance)
            or schur_tolerance <= 0
        ):
            raise ValueError("invalid hybrid GMRES tolerance or iteration budget")

        device = torch.empty((), device=self.backend.device).device
        synchronization_seconds = 0.0

        def synchronize():
            nonlocal synchronization_seconds
            if device.type == "cuda":
                started = time.perf_counter()
                torch.cuda.synchronize(device)
                synchronization_seconds += time.perf_counter() - started

        total_started = time.perf_counter()
        synchronize()
        transfer_started = time.perf_counter()
        z = as_float64(state, self.backend)
        guide = as_float64(model.guide, self.backend)
        active = torch.as_tensor(
            model.active_full, dtype=torch.bool, device=device
        )
        synchronize()
        state_transfer_seconds = time.perf_counter() - transfer_started
        state_transfer_bytes = int((z.numel() + guide.numel()) * z.element_size())
        active_transfer_bytes = int(active.numel())

        residual_started = time.perf_counter()
        residual = self.evaluator.linear_residual(z, active)
        synchronize()
        residual_seconds = time.perf_counter() - residual_started
        linearization_started = time.perf_counter()
        operator = self.evaluator.linearize(z, guide, active_full=active)
        synchronize()
        linearization_setup_seconds = time.perf_counter() - linearization_started

        cpu_validation_started = time.perf_counter()
        cpu_residual = model.residual(state)
        cpu_residual_seconds = time.perf_counter() - cpu_validation_started
        device_residual = residual.detach().cpu().numpy()
        residual_absolute_linf = float(np.max(np.abs(cpu_residual - device_residual)))
        transfer_started = time.perf_counter()
        right_hand_side = as_float64(-cpu_residual, self.backend)
        synchronize()
        right_hand_side_transfer_seconds = time.perf_counter() - transfer_started
        matrix_started = time.perf_counter()
        matrix = model.frozen_matrix(state)
        matrix_assembly_seconds = time.perf_counter() - matrix_started
        preconditioner = CPUBlockSchurPreconditioner(
            matrix, model.n, device, schur_tolerance=schur_tolerance
        )

        operator_seconds = 0.0
        operator_applications = 0

        def timed_operator(vector):
            nonlocal operator_seconds, operator_applications
            synchronize()
            started = time.perf_counter()
            result = operator(vector)
            synchronize()
            operator_seconds += time.perf_counter() - started
            operator_applications += 1
            return result

        synchronize()
        gmres_started = time.perf_counter()
        krylov = device_gmres(
            timed_operator,
            right_hand_side,
            preconditioner=preconditioner,
            rtol=krylov_rtol,
            restart=gmres_restart,
            max_iterations=maximum_krylov_cycles * gmres_restart,
            memory_limit_bytes=self.workspace_limit_bytes,
        )
        synchronize()
        gmres_seconds = time.perf_counter() - gmres_started

        true_residual = timed_operator(krylov.solution) - right_hand_side
        true_relative = float(
            torch.linalg.vector_norm(true_residual)
            / torch.linalg.vector_norm(right_hand_side).clamp_min(
                torch.finfo(torch.float64).tiny
            )
        )
        cuda_residual_true = true_residual + residual + right_hand_side
        cuda_residual_true_relative = float(
            torch.linalg.vector_norm(cuda_residual_true)
            / torch.linalg.vector_norm(residual).clamp_min(
                torch.finfo(torch.float64).tiny
            )
        )
        synchronize()
        output_started = time.perf_counter()
        direction = krylov.solution.detach().cpu().numpy().copy()
        synchronize()
        correction_transfer_seconds = time.perf_counter() - output_started
        cpu_true = model.jacobian_vector(state, direction) + cpu_residual
        cpu_true_relative = float(
            np.linalg.norm(cpu_true) / max(np.linalg.norm(cpu_residual), 1e-300)
        )
        cpu_true_residual_seconds = (
            time.perf_counter() - output_started - correction_transfer_seconds
        )
        timings = {
            "device_operator_initialization_seconds": self.initialization_seconds,
            "state_guide_mask_h2d_seconds": state_transfer_seconds,
            "state_guide_h2d_bytes": state_transfer_bytes,
            "active_mask_h2d_bytes": active_transfer_bytes,
            "cuda_residual_seconds": residual_seconds,
            "cuda_linearization_setup_seconds": linearization_setup_seconds,
            "cpu_residual_seconds": cpu_residual_seconds,
            "linear_rhs_h2d_seconds": right_hand_side_transfer_seconds,
            "linear_rhs_h2d_bytes": int(cpu_residual.nbytes),
            "cpu_matrix_assembly_seconds": matrix_assembly_seconds,
            "cpu_preconditioner_setup_seconds": preconditioner.setup_seconds,
            "cpu_preconditioner_applications": preconditioner.application_count,
            "cpu_preconditioner_apply_seconds":
                preconditioner.cpu_application_seconds,
            "d2h_seconds": preconditioner.device_to_host_seconds,
            "d2h_bytes": preconditioner.device_to_host_bytes,
            "h2d_preconditioner_seconds": preconditioner.host_to_device_seconds,
            "h2d_preconditioner_bytes": preconditioner.host_to_device_bytes,
            "cuda_jvp_seconds": operator_seconds,
            "cuda_operator_applications": operator_applications,
            "gmres_wall_seconds": gmres_seconds,
            "gmres_vector_and_internal_seconds": max(
                0.0,
                gmres_seconds
                - operator_seconds
                - preconditioner.application_wall_seconds,
            ),
            "explicit_synchronization_seconds":
                synchronization_seconds + preconditioner.synchronization_seconds,
            "correction_d2h_seconds": correction_transfer_seconds,
            "cpu_true_residual_seconds": cpu_true_residual_seconds,
            "cuda_residual_recomputed_relative_residual":
                cuda_residual_true_relative,
            "cpu_recomputed_true_relative_residual": cpu_true_relative,
            "cpu_residual_parity_linf": residual_absolute_linf,
            "total_correction_seconds": time.perf_counter() - total_started,
        }
        if (
            not krylov.converged
            or not math.isfinite(true_relative)
            or true_relative > 5 * krylov_rtol
            or not math.isfinite(cpu_true_relative)
            or cpu_true_relative > 5 * krylov_rtol
            or residual_absolute_linf > 1e-10
        ):
            raise CavitatingLinearError(
                "hybrid correction failed independently recomputed true-residual "
                f"gate (hybrid RHS={true_relative:.3e}, CPU={cpu_true_relative:.3e}, "
                f"CUDA-residual diagnostic={cuda_residual_true_relative:.3e}, "
                f"residual parity Linf={residual_absolute_linf:.3e}; {krylov.reason})",
                {
                    "true_linear_residual": true_relative,
                    "cuda_residual_true_relative": cuda_residual_true_relative,
                    "cpu_true_linear_residual": cpu_true_relative,
                    "linear_true_residuals": (true_relative,),
                    "linear_iterations": krylov.iterations,
                    "krylov_histories": (krylov.estimated_preconditioned_history,),
                    "field_factorization": "hybrid_cpu_superlu_schur",
                    "field_setup_failures": tuple(preconditioner.field_setup_failures),
                    **timings,
                },
            )

        return HybridJFOCorrection(
            direction=direction,
            true_linear_residual=true_relative,
            linear_iterations=krylov.iterations,
            used_bordered_fallback=preconditioner.used_bordered_fallback,
            schur_indicator=preconditioner.schur_indicator,
            linear_true_residuals=(true_relative,),
            krylov_histories=(krylov.estimated_preconditioned_history,),
            field_setup_failures=tuple(preconditioner.field_setup_failures),
            timings=timings,
        )
