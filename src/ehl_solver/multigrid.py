"""CPU geometric Galerkin V-cycles for frozen-film SPD Reynolds diffusion.

Dimensionless positive film (ny,nx), uniform dx,dy and homogeneous Dirichlet
pressure edges. This is not a multigrid solver for coupled JFO/elastic/load
equations. Fixed linear symmetric cycles can precondition CG on this SPD
operator. No nonlinear constraints or fields are clipped or rediscretized.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import coo_matrix, diags, kron
from scipy.sparse.linalg import LinearOperator

from ehl_solver.reynolds import assemble_reynolds_system


def linear_prolongation(n):
    """1-D linear interpolation of interior corrections, with zero edges.

    Fine physical n must be odd, with coarse physical (n+1)/2 nodes. Coarse
    boundary contributions are exactly zero, not renormalized into interior.
    """
    if type(n) is not int or n < 5 or n % 2 != 1:
        raise ValueError("odd physical fine axis >=5 required")
    nc = (n+1)//2
    rows, columns, values = [], [], []
    for i in range(1, n-1):
        left = i//2
        for j, weight in ((left, 1. if i % 2 == 0 else .5), (left+1, .5 if i % 2 else 0.)):
            if 0 < j < nc-1 and weight:
                rows.append(i-1)
                columns.append(j-1)
                values.append(weight)
    return coo_matrix((values, (rows, columns)), shape=(n-2, nc-2)).tocsr()


class ReynoldsMultigrid:
    """Bilinear P, R=P.T, Galerkin A_c=P.T A P, symmetric damped Jacobi.

    Pre/post smoothing counts are equal; the coarse solve is exact Cholesky
    on <=coarse_maximum unknowns. Per-level damping is bounded using an
    absolute row-sum spectral bound for D^-1 A. A cycle always starts at zero,
    so the inverse action stays fixed and linear. Setup rejects unsupported
    coarsening instead of silently factoring a large fine matrix.

    Optional boolean interior pressure_active imposes zero corrections on
    inactive pressure nodes through A_mask=F*A*F+(I-F), before Galerkin
    coarsening. Its active principal block remains SPD. This is a frozen
    linear surrogate, not a JFO complementarity or material Jacobian solve.
    Callers requiring exact inactive zeros must project the approximate
    cycle output; a finite V-cycle can couple them through coarse levels.
    """

    def __init__(self, film, dx, dy, *, smoothing_steps=2, omega=2/3, coarse_maximum=64,
                 pressure_active=None):
        if (type(smoothing_steps) is not int or not 1 <= smoothing_steps <= 8
                or type(coarse_maximum) is not int or not 1 <= coarse_maximum <= 256
                or not np.isfinite(omega) or not 0 < omega < 1):
            raise ValueError("bounded positive smoothing/coarse controls and 0<omega<1 required")
        shape = tuple(np.shape(film))
        if len(shape) == 2 and (shape[0]-2)*(shape[1]-2) > 300000:
            raise ValueError("multigrid foundation limited to 300000 interior unknowns")
        with np.errstate(over="raise", invalid="raise"):
            matrix, _ = assemble_reynolds_system(film, dx, dy, 0.)
        if pressure_active is not None:
            active = np.asarray(pressure_active)
            if active.shape != (shape[0]-2, shape[1]-2) or active.dtype != np.bool_:
                raise ValueError("pressure_active must be a boolean interior (ny-2,nx-2) mask")
            mask = diags(active.ravel().astype(float))
            matrix = (mask@matrix@mask + diags((~active).ravel().astype(float))).tocsr()
        self.matrix = matrix
        self.smoothing_steps = smoothing_steps
        self.levels = []
        while True:
            diagonal = matrix.diagonal()
            if not np.all(np.isfinite(matrix.data)) or np.any(diagonal <= 0):
                raise ValueError("finite positive Reynolds diagonal required")
            bound = float(np.max(np.asarray(abs(matrix).sum(axis=1)).ravel()/diagonal))
            damping = min(float(omega), 1.9/bound)
            level = {"matrix": matrix, "shape_ny_nx": shape, "diagonal": diagonal,
                     "omega": damping}
            self.levels.append(level)
            if matrix.shape[0] <= coarse_maximum:
                level["cholesky"] = cho_factor(matrix.toarray(), lower=True)
                break
            if len(self.levels) >= 12 or any(n < 5 or n % 2 == 0 for n in shape):
                raise ValueError("grid cannot coarsen to bounded exact coarse solve")
            prolong = kron(linear_prolongation(shape[0]), linear_prolongation(shape[1]),
                           format="csr")
            level["prolongation"] = prolong
            matrix = (prolong.T@matrix@prolong).tocsr()
            shape = tuple((n+1)//2 for n in shape)
        fine_nnz = self.matrix.nnz
        self.diagnostics = {
            "levels": len(self.levels),
            "shapes_ny_nx": [list(v["shape_ny_nx"]) for v in self.levels],
            "operator_complexity": sum(v["matrix"].nnz for v in self.levels)/fine_nnz,
            "grid_complexity": sum(v["matrix"].shape[0] for v in self.levels)/self.matrix.shape[0],
            "coarse_unknowns": matrix.shape[0],
            "omega_by_level": [v["omega"] for v in self.levels],
        }

    def _cycle(self, index, rhs):
        level = self.levels[index]
        if "cholesky" in level:
            return cho_solve(level["cholesky"], rhs)
        a, diagonal, omega = level["matrix"], level["diagonal"], level["omega"]
        x = np.zeros_like(rhs)
        for _ in range(self.smoothing_steps):
            x += omega*(rhs-a@x)/diagonal
        p = level["prolongation"]
        x += p@self._cycle(index+1, p.T@(rhs-a@x))
        for _ in range(self.smoothing_steps):
            x += omega*(rhs-a@x)/diagonal
        return x

    def apply(self, rhs):
        """One V-cycle inverse action; failed/nonfinite calculations propagate."""
        b = np.asarray(rhs, dtype=np.float64)
        if b.shape != (self.matrix.shape[0],) or not np.all(np.isfinite(b)):
            raise ValueError("finite interior RHS vector required")
        x = self._cycle(0, b)
        if not np.all(np.isfinite(x)):
            raise FloatingPointError("nonfinite multigrid cycle")
        return x

    def aslinearoperator(self):
        return LinearOperator(self.matrix.shape, matvec=self.apply, dtype=np.float64)
