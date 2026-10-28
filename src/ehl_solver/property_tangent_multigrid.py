"""Diagnostic nonsymmetric material/JFO multigrid, independent of production.

All physical problem inputs remain SI. Internal matrices/state corrections are
scaled dimensionless P,D,h0/H0, C-order (ny-2,nx-2); x follows entrainment.
No physical state, active set, constitutive law or nonlinear gate is changed.
"""

import numpy as np
from scipy.sparse import csr_matrix, diags, kron
from scipy.sparse.linalg import LinearOperator, spilu, splu

from ehl_solver.multigrid import linear_prolongation


def _factor_bytes(factor):
    return sum(a.data.nbytes+a.indices.nbytes+a.indptr.nbytes for a in (factor.L, factor.U)) + (
        factor.perm_r.nbytes+factor.perm_c.nbytes
    )


class NonsymmetricMultigrid:
    """Fixed linear Galerkin V-cycle with ILU defect correction and bounded LU.

    Takes a dimensionless sparse interior matrix and physical grid shape
    (ny,nx), including boundary nodes. Fine signed row/max-column scaling
    equilibrates the system without altering its solution. Bilinear P and
    R=P.T act on scaled corrections; no coarse physical state is inferred.
    Each noncoarse level uses two pre/post ILU defect-correction sweeps by
    default. ILU fill_factor is explicitly bounded in [1,4] (default 2), with
    drop_tol=1e-4 and NATURAL ordering;
    factor payload is measured rather than assumed to equal an ILU(0) bound.
    An exact sparse LU is allowed only at <=64 unknowns by default (<=256
    configurable). Every cycle starts from zero, so GMRES sees a fixed linear
    inverse. No SPD, contraction or mesh-independence guarantee is asserted.
    Setup/application failures propagate, without a sparse fine-LU fallback.
    """

    def __init__(self, matrix, shape_ny_nx, *, smoothing_steps=2, coarse_maximum=64,
                 ilu_fill_factor=2.):
        shape = tuple(shape_ny_nx)
        if (len(shape) != 2 or any(type(n) is not int or n < 3 for n in shape)
                or (shape[0]-2)*(shape[1]-2) > 300000):
            raise ValueError("physical (ny,nx) >=3 and <=300000 interior unknowns required")
        if (type(smoothing_steps) is not int or not 1 <= smoothing_steps <= 8
                or type(coarse_maximum) is not int or not 1 <= coarse_maximum <= 256):
            raise ValueError("bounded positive smoothing and coarse controls required")
        if (isinstance(ilu_fill_factor, bool) or not np.isfinite(ilu_fill_factor)
                or not 1 <= ilu_fill_factor <= 4):
            raise ValueError("ILU fill factor must be finite in [1,4]")
        matrix = csr_matrix(matrix, dtype=np.float64, copy=True)
        n = (shape[0]-2)*(shape[1]-2)
        if matrix.shape != (n, n) or not np.all(np.isfinite(matrix.data)):
            raise ValueError("finite square matrix matching interior grid required")
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        self.matrix = matrix
        diagonal = matrix.diagonal()
        if np.any(diagonal == 0):
            raise ValueError("nonzero fine diagonal required for signed equilibration")
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            row_max = abs(matrix).max(axis=1).toarray().ravel()
            self.row_scale = np.sign(diagonal)/row_max
            scaled = diags(self.row_scale)@matrix
            col_max = abs(scaled).max(axis=0).toarray().ravel()
            if np.any(col_max == 0):
                raise ValueError("nonzero columns required for equilibration")
            self.column_scale = 1/col_max
            matrix = (scaled@diags(self.column_scale)).tocsr()
        self.levels = []
        self.smoothing_steps = smoothing_steps
        while True:
            if not np.all(np.isfinite(matrix.data)):
                raise FloatingPointError("nonfinite Galerkin matrix")
            level = {"matrix": matrix, "shape_ny_nx": shape}
            self.levels.append(level)
            if matrix.shape[0] <= coarse_maximum:
                try:
                    level["lu"] = splu(matrix.tocsc())
                except RuntimeError as exc:
                    raise RuntimeError(
                        f"coarse LU failed at level {len(self.levels)-1}, "
                        f"{matrix.shape[0]} unknowns: {exc}"
                    ) from exc
                break
            if len(self.levels) >= 12 or any(n < 5 or n % 2 == 0 for n in shape):
                raise ValueError("grid cannot coarsen to bounded exact LU")
            try:
                level["ilu"] = spilu(matrix.tocsc(), drop_tol=1e-4, fill_factor=ilu_fill_factor,
                                      permc_spec="NATURAL", drop_rule="basic,area")
            except RuntimeError as exc:
                raise RuntimeError(
                    f"ILU failed at level {len(self.levels)-1}, {matrix.shape[0]} unknowns: {exc}"
                ) from exc
            prolong = kron(linear_prolongation(shape[0]), linear_prolongation(shape[1]),
                           format="csr")
            level["prolongation"] = prolong
            matrix = (prolong.T@matrix@prolong).tocsr()
            shape = tuple((n+1)//2 for n in shape)
        for level in self.levels:
            factor = level["lu"] if "lu" in level else level["ilu"]
            if not all(np.all(np.isfinite(a.data)) for a in (factor.L, factor.U)):
                raise FloatingPointError("nonfinite multigrid factors")
        self.diagnostics = {
            "levels": len(self.levels),
            "shapes_ny_nx": [list(v["shape_ny_nx"]) for v in self.levels],
            "coarse_unknowns": matrix.shape[0],
            "operator_complexity": sum(v["matrix"].nnz for v in self.levels)/self.matrix.nnz,
            "smoothing_steps": smoothing_steps,
            "smoother": "fixed ILU: drop_tol=1e-4, NATURAL, basic/area dropping",
            "ilu_fill_factor": float(ilu_fill_factor),
            "factor_payload_bytes": sum(_factor_bytes(v.get("lu", v.get("ilu")))
                                        for v in self.levels),
        }

    def _cycle(self, index, rhs):
        level = self.levels[index]
        if "lu" in level:
            return level["lu"].solve(rhs)
        matrix, ilu = level["matrix"], level["ilu"]
        answer = np.zeros_like(rhs)
        for _ in range(self.smoothing_steps):
            answer += ilu.solve(rhs-matrix@answer)
        prolong = level["prolongation"]
        answer += prolong@self._cycle(index+1, prolong.T@(rhs-matrix@answer))
        for _ in range(self.smoothing_steps):
            answer += ilu.solve(rhs-matrix@answer)
        return answer

    def apply(self, rhs):
        rhs = np.asarray(rhs, dtype=np.float64)
        if rhs.shape != (self.matrix.shape[0],) or not np.all(np.isfinite(rhs)):
            raise ValueError("finite interior RHS required")
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            answer = self.column_scale*self._cycle(0, self.row_scale*rhs)
        if not np.all(np.isfinite(answer)):
            raise FloatingPointError("nonfinite nonsymmetric multigrid cycle")
        return answer

    def aslinearoperator(self):
        return LinearOperator(self.matrix.shape, matvec=self.apply, dtype=np.float64)


class _PropertyTangentBlock:
    """Exact active-row elimination, with all frozen material/local mass terms.

    For full-film selector F and cavitated selector C=I-F, mass Jacobian blocks
    A=dR_mass/dP and B=dR_mass/dD give reduced K=A F+B C. Known dP_C and dD_F
    come from arbitrary complementarity RHS and their mass columns are removed
    before solving. The mixed correction u reconstructs dP_F=F u, dD_C=C u;
    active constraint rows therefore remain exact even with an approximate
    cycle. Unlike the frozen prototype, no pressure/content mass coupling is
    dropped. Material and cell-self elastic tangents enter A through the
    existing sparse assembly; full-source elasticity remains in the outer JVP.
    Neither coarse corrections nor ILU factors define physical coarse fields.
    """

    def __init__(self, system, state):
        if system.preconditioner not in ("material", "local"):
            raise ValueError("material or local tangent preconditioner required")
        matrix = system.frozen_matrix(state)
        self.n = system.n
        self.full = np.array(system.full(state), dtype=bool, copy=True)
        self.cavitated = ~self.full
        self.pressure = matrix[:self.n, :self.n].tocsr()
        self.content = matrix[:self.n, self.n:2*self.n].tocsr()
        self.reduced = (self.pressure@diags(self.full.astype(float))
                        + self.content@diags(self.cavitated.astype(float))).tocsr()

    def _solve_reduced(self, rhs):
        raise NotImplementedError

    def apply(self, rhs):
        rhs = np.asarray(rhs, dtype=np.float64)
        if rhs.shape != (2*self.n,) or not np.all(np.isfinite(rhs)):
            raise ValueError("finite mass/complementarity RHS of length 2*n required")
        pressure = np.where(self.cavitated, rhs[self.n:], 0.)
        deficit = np.where(self.full, rhs[self.n:], 0.)
        reduced_rhs = rhs[:self.n]-self.pressure@pressure-self.content@deficit
        mixed = self._solve_reduced(reduced_rhs)
        answer = np.r_[np.where(self.full, mixed, pressure),
                       np.where(self.cavitated, mixed, deficit)]
        if not np.all(np.isfinite(answer)):
            raise FloatingPointError("nonfinite tangent block inverse")
        return answer

    def aslinearoperator(self):
        return LinearOperator((2*self.n, 2*self.n), matvec=self.apply, dtype=np.float64)


class PropertyTangentMultigrid(_PropertyTangentBlock):
    """Diagnostic material/local mixed-correction hierarchy, no production selection."""

    def __init__(self, system, state, *, ilu_fill_factor=2.):
        super().__init__(system, state)
        self.multigrid = NonsymmetricMultigrid(self.reduced, system.problem.shape,
                                              ilu_fill_factor=ilu_fill_factor)

    def _solve_reduced(self, rhs):
        return self.multigrid.apply(rhs)


class PropertyTangentLU(_PropertyTangentBlock):
    """Exact sparse reduced-block control, not a multigrid fallback.

    Factors n mixed unknowns after exact complementarity elimination, compared
    with the original 2*n field matrix. The outer load Schur reconstruction
    can therefore be checked against the full 2*n+1 sparse bordered inverse.
    Explicit selection is required; a failed hierarchy never invokes this.
    """

    def __init__(self, system, state):
        super().__init__(system, state)
        self.factor = splu(self.reduced.tocsc())
        if not all(np.all(np.isfinite(a.data)) for a in (self.factor.L, self.factor.U)):
            raise FloatingPointError("nonfinite reduced LU factors")

    def _solve_reduced(self, rhs):
        return self.factor.solve(rhs)
