"""Diagnostic CPU active pressure/content block inverse; no production integration.

Uses the frozen material mobility and the existing conservative donor-content
stencil. The full elastic/material Jacobian and force border are deliberately
left to an outer nonsymmetric Krylov solve. No state is projected or clipped.
"""

import numpy as np
from scipy.sparse import triu
from scipy.sparse.linalg import LinearOperator, spsolve_triangular

from ehl_solver.multigrid import ReynoldsMultigrid


class PropertyBlockMultigrid:
    """Fixed linear lower-block inverse for a frozen active set.

    Input is an internal PropertyLoadSystem and its dimensionless P,D,h0/H0
    state (C-order interior arrays, x along entrainment, D=1-theta). Public
    physical problem fields retain SI units. Apply takes mass/comp RHS rows
    and returns dimensionless pressure/deficit corrections, without the load
    row. This prototype supports the frozen preconditioner only; production
    material/local tangent preconditioners and nonlinear paths are untouched.

    Complementarity fixes dD on full cells and dP on cavitated cells. After
    subtracting those known columns, a projected MG cycle approximates the
    full-cell pressure principal inverse. Cavitated-cell deficit corrections
    then use an exact triangular donor transport solve, including pressure
    coupling. The mass coupling from unknown cavitated deficits into full
    cells is omitted. Thus this inverse is nonsymmetric and is not suitable
    for CG. Original active constraints are satisfied exactly for any RHS.
    """

    def __init__(self, system, state):
        if system.preconditioner != "frozen":
            raise ValueError("diagnostic block MG requires the frozen preconditioner")
        matrix = system.frozen_matrix(state)
        self.n = system.n
        self.full = np.array(system.full(state), dtype=bool, copy=True)
        self.cavitated = ~self.full
        self.pressure = matrix[:self.n, :self.n].tocsr()
        self.content = matrix[:self.n, self.n:2*self.n].tocsr()
        self.transport = self.content[self.cavitated][:, self.cavitated].tocsr()
        if self.transport.shape[0]:
            upper = triu(self.transport, k=1)
            if upper.nnz and np.any(upper.data != 0):
                raise ValueError("lower triangular positive-x donor transport required")
            diagonal = self.transport.diagonal()
            if not np.all(np.isfinite(diagonal)) or np.any(diagonal == 0):
                raise ValueError("nonzero finite cavitated transport diagonal required")
        p, _, film, _, _ = system.base.unpack(state[:-1])
        density, viscosity, _, _ = system.base.fluid.ratios(
            p*system.problem.pressure_scale_pa
        )
        mobility = density*(film/system.problem.film_scale_m)**3/viscosity
        self.multigrid = ReynoldsMultigrid(
            np.cbrt(mobility/system.base.mass_scale), system.problem.dx, system.problem.dy,
            pressure_active=self.full.reshape(film.shape[0]-2, film.shape[1]-2),
        )

    def apply(self, rhs):
        """One fixed block inverse action; malformed/nonfinite RHS raises."""
        rhs = np.asarray(rhs, dtype=np.float64)
        if rhs.shape != (2*self.n,) or not np.all(np.isfinite(rhs)):
            raise ValueError("finite mass/complementarity RHS of length 2*n required")
        pressure, deficit = np.zeros(self.n), np.zeros(self.n)
        pressure[self.cavitated] = rhs[self.n:][self.cavitated]
        deficit[self.full] = rhs[self.n:][self.full]
        remaining = rhs[:self.n]-self.pressure@pressure-self.content@deficit
        pressure_rhs = np.where(self.full, remaining, 0.)
        pressure[self.full] = self.multigrid.apply(pressure_rhs)[self.full]
        if self.transport.shape[0]:
            content_rhs = (remaining-self.pressure@np.where(self.full, pressure, 0.))[
                self.cavitated
            ]
            deficit[self.cavitated] = spsolve_triangular(
                self.transport, content_rhs, lower=True
            )
        result = np.r_[pressure, deficit]
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("nonfinite block multigrid action")
        return result

    def aslinearoperator(self):
        return LinearOperator((2*self.n, 2*self.n), matvec=self.apply, dtype=np.float64)
