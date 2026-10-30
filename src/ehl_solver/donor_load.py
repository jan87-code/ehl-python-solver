"""Experimental CPU donor-capacity pressure/content/load Newton adapter.

SI fields (ny,nx), +x entrainment. Uses the existing tested conservative donor
flux and full-source elastic JVP. Production arithmetic-capacity default stays
unchanged. First-order transport diffusion is a discretization change, not a
pressure filter. All original Newton/admissibility/force gates are retained.
"""

from dataclasses import replace

import numpy as np
from scipy.sparse import coo_matrix

from ehl_solver.property_load_newton import PropertyLoadSystem
from ehl_solver.property_warm import PropertyWarmSystem
from ehl_solver.transport_candidate import donor_flux_direction, donor_fluxes


def donor_mass_audit(
    x, y, h, p, theta, fluid, speed, *, length_scale_m, film_scale_m, pressure_scale_pa
):
    """Independent SI face reconstruction; no call to the solver's donor flux.

    Flux per transverse length is kg/(m s); integrated boundary flow is kg/s.
    Shared faces telescope. No clipping, smoothing or modified boundaries.
    """
    rho, eta, _, _ = fluid.properties(p)
    mobility = rho * h**3 / (12 * eta)
    qx = -0.5 * (mobility[:, :-1] + mobility[:, 1:]) * np.diff(p, axis=1) / np.diff(x)
    qx += speed * rho[:, :-1] * h[:, :-1] * theta[:, :-1]
    qy = -0.5 * (mobility[:-1] + mobility[1:]) * np.diff(p, axis=0) / np.diff(y)[:, None]
    wx, wy = 0.5 * (x[2:] - x[:-2]), 0.5 * (y[2:] - y[:-2])
    residual = np.diff(qx[1:-1], axis=1) / wx + np.diff(qy[:, 1:-1], axis=0) / wy[:, None]
    volumes = residual * wx * wy[:, None]
    local = (abs(qx[1:-1, 1:]) + abs(qx[1:-1, :-1])) * wy[:, None] + (
        abs(qy[1:, 1:-1]) + abs(qy[:-1, 1:-1])
    ) * wx
    outward = np.array([-qx[1:-1, 0] @ wy, qx[1:-1, -1] @ wy, -qy[0, 1:-1] @ wx, qy[-1, 1:-1] @ wx])
    ref = (
        fluid.density_kg_m3
        * film_scale_m**3
        * pressure_scale_pa
        / (12 * fluid.viscosity_pa_s * length_scale_m)
    )
    return dict(
        mass_maximum=float(np.max(abs(residual)) * length_scale_m / ref),
        maximum_si_mass_divergence_kg_m2_s=float(np.max(abs(residual))),
        relative_local_balance=float(np.max(abs(volumes) / np.maximum(local, 1e-300))),
        relative_global_balance=float(abs(outward.sum()) / max(abs(outward).sum(), 1e-300)),
        outward_mass_flow_kg_s=outward.tolist(),
        net_mass_flow_kg_s=float(outward.sum()),
        summed_cell_mass_defect_kg_s=float(volumes.sum()),
        maximum_complementarity=float(np.max(abs(p / pressure_scale_pa * (1 - theta)))),
    )


class DonorWarmSystem(PropertyWarmSystem):
    """Isolated nonlinear donor operator; inherited geometry/boundaries unchanged."""

    def residual(self, state):
        p, t, h, _, _ = self.unpack(state)
        qx, qy = donor_fluxes(
            self.problem.x_m,
            self.problem.y_m,
            h,
            p * self.problem.pressure_scale_pa,
            t,
            self.fluid,
            self.speed,
        )
        mass = self.divergence(qx, qy) * self.problem.length_scale_m / self.flux_scale
        return np.r_[mass.ravel() / self.mass_scale, np.minimum(state[: self.n], state[self.n :])]

    def jacobian_vector(self, state, direction, *, offset_direction_m=0.0):
        v = np.asarray(direction, dtype=np.float64)
        if v.shape != (self.size,) or not np.all(np.isfinite(v)):
            raise ValueError("Finite donor P,D direction required")
        p, t, h, _, _ = self.unpack(state)
        dp, dt = np.zeros(self.problem.shape), np.zeros(self.problem.shape)
        dp[1:-1, 1:-1] = v[: self.n].reshape(dp[1:-1, 1:-1].shape) * self.problem.pressure_scale_pa
        dt[1:-1, 1:-1] = -v[self.n :].reshape(dt[1:-1, 1:-1].shape)
        dh = self.elastic_deformation(dp) + offset_direction_m
        qx, qy = donor_flux_direction(
            self.problem.x_m,
            self.problem.y_m,
            h,
            p * self.problem.pressure_scale_pa,
            t,
            dh,
            dp,
            dt,
            self.fluid,
            self.speed,
        )
        mass = self.divergence(qx, qy) * self.problem.length_scale_m / self.flux_scale
        full = state[: self.n] >= state[self.n :]
        return np.r_[mass.ravel() / self.mass_scale, np.where(full, v[self.n :], v[: self.n])]

    def audited_solution(self, state, history=()):
        solution = super().audited_solution(state, history)
        audit = donor_mass_audit(
            self.problem.x_m,
            self.problem.y_m,
            solution.film_thickness_m,
            solution.pressure_pa,
            solution.fractional_film_content,
            self.fluid,
            self.speed,
            length_scale_m=self.problem.length_scale_m,
            film_scale_m=self.problem.film_scale_m,
            pressure_scale_pa=self.problem.pressure_scale_pa,
        )
        return replace(solution, audit=audit)


class DonorLoadSystem(PropertyLoadSystem):
    """Experimental load system with donor-consistent sparse preconditioning."""

    warm_system_type = DonorWarmSystem

    def frozen_matrix(self, z):
        # Parent invokes our tangent_pressure and true offset JVP. Only its
        # frozen D advection block still needs the donor-minus-reference term.
        matrix = super().frozen_matrix(z)
        p, _t, h, _, _ = self.base.unpack(z[:-1])
        rho, _, _, _ = self.base.fluid.ratios(p * self.problem.pressure_scale_pa)
        c = rho * h / self.problem.film_scale_m
        k = np.arange(self.n).reshape(c.shape[0] - 2, c.shape[1] - 2)
        east = (
            0.5 * self.problem.reynolds_parameter * (c[1:-1, 1:-1] - c[1:-1, 2:]) / self.problem.dx
        )
        west = (
            0.5 * self.problem.reynolds_parameter * (c[1:-1, :-2] - c[1:-1, 1:-1]) / self.problem.dx
        )
        rows = np.r_[k.ravel(), k[:, 1:].ravel()]
        cols = np.r_[self.n + k.ravel(), self.n + k[:, :-1].ravel()]
        values = np.r_[-east.ravel(), west[:, 1:].ravel()] / self.base.mass_scale
        return (matrix + coo_matrix((values, (rows, cols)), shape=matrix.shape)).tocsc()

    def tangent_pressure(self, p, theta, h, r, v, dr, dv):
        # Add only the donor-minus-reference material/self-film advection
        # tangent. The existing diffusion tangent remains exactly unchanged.
        matrix = super().tangent_pressure(p, theta, h, r, v, dr, dv)
        dc = h * dr * self.problem.pressure_scale_pa
        if self.preconditioner == "local":
            ny, nx = self.problem.shape
            compliance = (
                self.problem.elastic_kernel[ny - 1, nx - 1]
                * self.problem.pressure_scale_pa
                / self.problem.film_scale_m
            )
            dc = dc + r * compliance
        ax = 0.5 * self.problem.reynolds_parameter * dc[:, :-1] * theta[:, :-1]
        bx = -0.5 * self.problem.reynolds_parameter * dc[:, 1:] * theta[:, :-1]
        k = np.arange(self.n).reshape(h.shape[0] - 2, h.shape[1] - 2)
        diagonal = (ax[1:-1, 1:] - bx[1:-1, :-1]) / self.problem.dx
        rows = np.r_[k.ravel(), k[:, :-1].ravel(), k[:, 1:].ravel()]
        cols = np.r_[k.ravel(), k[:, 1:].ravel(), k[:, :-1].ravel()]
        values = np.r_[
            diagonal.ravel(),
            bx[1:-1, 1:-1].ravel() / self.problem.dx,
            -ax[1:-1, 1:-1].ravel() / self.problem.dx,
        ]
        return matrix + coo_matrix((values, (rows, cols)), shape=matrix.shape).tocsr()
