"""Experimental GPU-resident local donor/JFO preconditioner candidate."""

import math

import numpy as np

from ehl_solver.multigrid import ReynoldsMultigrid


class _DevicePressureMultigrid:
    def __init__(self, mobility, active, dx, dy, device):
        import torch

        mobility_cpu = mobility.detach().cpu().numpy()
        active_cpu = active.detach().cpu().numpy().reshape(
            mobility_cpu.shape[0] - 2, mobility_cpu.shape[1] - 2
        )
        reference = ReynoldsMultigrid(
            np.cbrt(mobility_cpu),
            dx,
            dy,
            pressure_active=active_cpu,
        )
        self.levels = []
        self.active = active.clone()
        self.device = device
        self.estimated_bytes = 0
        for source in reference.levels:
            matrix = self._csr(source["matrix"], device)
            diagonal = torch.as_tensor(
                source["diagonal"], dtype=torch.float64, device=device
            )
            level = {
                "matrix": matrix,
                "diagonal": diagonal,
                "omega": source["omega"],
            }
            self.estimated_bytes += self._csr_bytes(source["matrix"])
            if "prolongation" in source:
                prolongation = source["prolongation"].tocsr()
                level["prolongation"] = self._csr(prolongation, device)
                level["restriction"] = self._csr(prolongation.T.tocsr(), device)
                self.estimated_bytes += (
                    self._csr_bytes(prolongation)
                    + self._csr_bytes(prolongation.T.tocsr())
                )
            else:
                dense = torch.as_tensor(
                    source["matrix"].toarray(), dtype=torch.float64, device=device
                )
                level["coarse_factor"] = torch.linalg.cholesky(dense)
                self.estimated_bytes += dense.numel() * dense.element_size()
            self.levels.append(level)
        self.smoothing_steps = reference.smoothing_steps

    @staticmethod
    def _csr_bytes(matrix):
        return int(matrix.nnz * 16 + (matrix.shape[0] + 1) * 8)

    @staticmethod
    def _csr(matrix, device):
        import torch

        matrix = matrix.tocsr()
        return torch.sparse_csr_tensor(
            torch.as_tensor(matrix.indptr, dtype=torch.int64, device=device),
            torch.as_tensor(matrix.indices, dtype=torch.int64, device=device),
            torch.as_tensor(matrix.data, dtype=torch.float64, device=device),
            size=matrix.shape,
        )

    def _apply_matrix(self, matrix, vector):
        import torch

        return torch.sparse.mm(matrix, vector[:, None])[:, 0]

    def _cycle(self, index, rhs):
        import torch

        level = self.levels[index]
        if "coarse_factor" in level:
            return torch.cholesky_solve(rhs[:, None], level["coarse_factor"])[:, 0]
        matrix, diagonal, omega = level["matrix"], level["diagonal"], level["omega"]
        solution = torch.zeros_like(rhs)
        for _ in range(self.smoothing_steps):
            solution += omega * (rhs - self._apply_matrix(matrix, solution)) / diagonal
        residual = rhs - self._apply_matrix(matrix, solution)
        coarse_rhs = self._apply_matrix(level["restriction"], residual)
        coarse = self._cycle(index + 1, coarse_rhs)
        solution += self._apply_matrix(level["prolongation"], coarse)
        for _ in range(self.smoothing_steps):
            solution += omega * (rhs - self._apply_matrix(matrix, solution)) / diagonal
        return solution

    def apply(self, rhs):
        import torch

        return torch.where(self.active, self._cycle(0, rhs), 0)


class ExperimentalDeviceDonorJFOPreconditioner:
    """Multigrid diffusion candidate with donor-content and load-border corrections.

    The frozen local mass action includes pressure-dependent material
    derivatives and cell-self elastic compliance. Long-range elasticity and
    variable-coefficient coupling are intentionally omitted from the inverse,
    not from the matrix-free Krylov operator. It is an experiment, not a
    production preconditioner: it fails the saved 257x257 checkpoints 0, 3,
    and 6 true-residual gates. Inputs and outputs are resident float64 vectors
    in the DeviceDonorJFO scaled `(P,D,h0/Href)` ordering.
    """

    def __init__(self, evaluator, state, active_full, *, memory_limit_bytes=2**30):
        import torch

        if evaluator.backend.name != "torch":
            raise ValueError("JFO preconditioner requires an explicit torch backend")
        z = evaluator._vector(state)
        if (
            not isinstance(active_full, torch.Tensor)
            or active_full.shape != (evaluator.n,)
            or active_full.dtype != torch.bool
            or active_full.device != z.device
        ):
            raise ValueError("active_full must be a same-device boolean interior mask")
        if not isinstance(memory_limit_bytes, int) or memory_limit_bytes < 1:
            raise ValueError("memory_limit_bytes must be a positive integer")
        self.evaluator = evaluator
        self.state = z
        self.active_full = active_full.clone()
        self.cavitated = ~self.active_full
        self.n = evaluator.n
        self.interior_shape = evaluator.interior_shape
        self.shape = evaluator.shape
        self.device = z.device
        self.dtype = z.dtype
        self.memory_limit_bytes = memory_limit_bytes
        self.application_count = 0

        p, theta, film, _ = evaluator.state_fields(z)
        rho, eta, drho, deta = evaluator.flux._properties(p)
        self.pressure = p
        self.rho = rho
        self.theta = theta
        self.film = film
        self.eta = eta
        self.mobility = rho * film**3 / (12 * eta)
        self.drho = drho
        self.deta = deta
        self.pressure_scale = evaluator.pressure_scale
        self.film_scale = evaluator.film_scale
        self.mass_factor = evaluator.mass_factor
        self.speed = evaluator.flux.speed
        self.dx, self.dy = evaluator.dx, evaluator.dy
        self.target_load = evaluator.target_load
        dx, dy = evaluator.dx, evaluator.dy
        e_star = evaluator.e_star_pa
        self.self_compliance = (
            2 * (dx * math.asinh(dy / dx) + dy * math.asinh(dx / dy))
            / (math.pi * e_star)
        )

        interior_mobility = self.mobility[1:-1, 1:-1]
        if bool((interior_mobility <= 0).any()):
            raise ValueError("frozen-film mobility must be positive")
        self.pressure_multigrid = _DevicePressureMultigrid(
            self.mobility, self.active_full, dx, dy, z.device
        )
        self.pressure_operator_scale = self.mass_factor * self.pressure_scale
        self.content_diagonal = (
            -self.mass_factor * self.speed * rho[1:-1, 1:-1]
            * film[1:-1, 1:-1] / dx
        )
        if bool(self.cavitated.any()) and (
            self.speed <= 0
            or bool((self.content_diagonal.reshape(-1)[self.cavitated] == 0).any())
        ):
            raise ArithmeticError(
                "donor-content diagonal is singular on cavitated cells"
            )
        self.load_gradient = torch.full(
            (self.n,),
            evaluator.pressure_scale * dx * dy / self.target_load,
            dtype=z.dtype,
            device=z.device,
        )
        self.estimated_bytes = int(
            8 * (12 * math.prod(self.shape) + 3 * self.n + math.prod(self.interior_shape))
            + self.pressure_multigrid.estimated_bytes
        )
        if self.estimated_bytes > memory_limit_bytes:
            raise MemoryError("JFO preconditioner workspace exceeds memory budget")

        offset = torch.zeros_like(z)
        offset[-1] = 1
        offset_column = self._mass_action(offset)
        self.load_response = self._field_inverse(
            torch.cat((offset_column, torch.zeros_like(offset_column)))
        )
        denominator = torch.dot(self.load_gradient, self.load_response[:self.n])
        self.load_denominator = float(denominator)
        scale = float(
            torch.linalg.vector_norm(self.load_gradient)
            * torch.linalg.vector_norm(self.load_response[:self.n])
        )
        if (
            not math.isfinite(self.load_denominator)
            or abs(self.load_denominator)
            <= 1e-12 * max(scale, torch.finfo(z.dtype).tiny)
        ):
            raise ArithmeticError("JFO approximate load Schur complement is singular")

    def _mass_action(self, direction):
        import torch

        evaluator = self.evaluator
        dp, dt = evaluator._expanded(direction, direction=True)
        dp_pa = dp
        dh = self.self_compliance * dp_pa + direction[-1] * self.film_scale
        dm = self.mobility * (
            (self.drho / self.rho - self.deta / self.eta) * dp_pa
            + 3 * dh / self.film
        )
        dc = self.drho * dp_pa * self.film + self.rho * dh
        qx = (
            -0.5 * (dm[:, 1:] + dm[:, :-1])
            * torch.diff(self.pressure, dim=1) / self.dx
            -0.5 * (self.mobility[:, 1:] + self.mobility[:, :-1])
            * torch.diff(dp_pa, dim=1) / self.dx
            + self.speed * (dc * self.theta + self.rho * self.film * dt)[:, :-1]
        )
        qy = (
            -0.5 * (dm[1:] + dm[:-1])
            * torch.diff(self.pressure, dim=0) / self.dy
            -0.5 * (self.mobility[1:] + self.mobility[:-1])
            * torch.diff(dp_pa, dim=0) / self.dy
        )
        return (
            torch.diff(qx[1:-1], dim=1) / self.dx
            + torch.diff(qy[:, 1:-1], dim=0) / self.dy
        ).reshape(-1) * self.mass_factor

    def _pressure_inverse(self, rhs):
        return (
            self.pressure_multigrid.apply(rhs) / self.pressure_operator_scale
        )

    def _field_inverse(self, rhs):
        import torch

        rhs_mass, rhs_complementarity = rhs[:self.n], rhs[self.n:2 * self.n]
        pressure = torch.where(
            self.cavitated, rhs_complementarity, torch.zeros_like(rhs_mass)
        )
        deficit = torch.where(self.active_full, rhs_complementarity, torch.zeros_like(rhs_mass))
        candidate = torch.cat((pressure, deficit, rhs.new_zeros(1)))
        remaining_mass = rhs_mass - self._mass_action(candidate)
        pressure = pressure + self._pressure_inverse(remaining_mass)
        candidate = torch.cat((pressure, deficit, rhs.new_zeros(1)))
        remaining_mass = rhs_mass - self._mass_action(candidate)
        cavitated = self.cavitated.reshape(self.interior_shape)
        remaining = remaining_mass.reshape(self.interior_shape)
        cumulative = torch.cumsum(torch.where(cavitated, remaining, 0), dim=1)
        starts = cavitated & ~torch.cat(
            (torch.zeros_like(cavitated[:, :1]), cavitated[:, :-1]), dim=1
        )
        columns = torch.arange(
            self.interior_shape[1], dtype=torch.int64, device=self.device
        )[None, :]
        latest_start = torch.cummax(
            torch.where(starts, columns + 1, 0), dim=1
        ).values
        previous_index = (latest_start - 2).clamp_min(0)
        previous_sum = torch.gather(cumulative, 1, previous_index.expand_as(cumulative))
        previous_sum = torch.where(latest_start > 1, previous_sum, 0)
        segment_sum = cumulative - previous_sum
        diagonal = self.content_diagonal
        cavitated_deficit = (segment_sum / diagonal).reshape(-1)
        deficit = torch.where(self.cavitated, cavitated_deficit, deficit)
        return torch.cat((pressure, deficit))

    def __call__(self, rhs):
        import torch

        self.application_count += 1
        if (
            not isinstance(rhs, torch.Tensor)
            or rhs.shape != (self.evaluator.size,)
            or rhs.dtype != self.dtype
            or rhs.device != self.device
            or not bool(torch.isfinite(rhs).all())
        ):
            raise ValueError("finite same-device float64 bordered rhs required")
        field = self._field_inverse(rhs[:2 * self.n])
        offset = (
            torch.dot(self.load_gradient, field[:self.n]) - rhs[-1]
        ) / self.load_denominator
        corrected = field - self.load_response * offset
        return torch.cat((corrected, offset.reshape(1)))

    def matrix_vector(self, direction):
        """Apply frozen local mass and selected complementarity rows."""
        import torch

        if (
            not isinstance(direction, torch.Tensor)
            or direction.shape != (2 * self.n,)
            or direction.dtype != self.dtype
            or direction.device != self.device
            or not bool(torch.isfinite(direction).all())
        ):
            raise ValueError("finite same-device float64 field direction required")
        mass = self._mass_action(torch.cat((direction, direction.new_zeros(1))))
        pressure, deficit = direction[:self.n], direction[self.n:]
        complementarity = torch.where(self.active_full, deficit, pressure)
        return torch.cat((mass, complementarity))
