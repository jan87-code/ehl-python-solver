"""Experimental limited-flux residual/linearization; no nonlinear EHL solver.

Public case inputs are SI; arrays (ny,nx), +x entrainment, CPU float64.
Unknowns P=p/pH,D=1-theta,h0/Href in C-order interiors; moment mode adds
alpha/(5e-4 rad). Mass residual uses Lref/(100*qref), complementarity is
min(P,D), force is relative SI load, and physical Mx is scaled by F*L/2.
The true action includes full-source elasticity; sparse preconditioning
freezes film or keeps only cell-self compliance. No default solver changes.
"""

import numpy as np
from scipy.sparse import bmat, coo_matrix, csr_matrix, diags, hstack, vstack

from ehl_solver.limited_transport import (
    _minmod_branch,
    _select_limiter_branches,
    _slope_candidates,
    freeze_limited_flux_jvp,
    limited_flux_direction,
    limited_flux_directional_derivative,
    limited_fluxes,
)
from ehl_solver.transport_candidate import donor_flux_direction, donor_fluxes

MOMENT_ANGLE_SCALE_RAD = 5e-4
MOMENT_MAX_ABS_TILT_RAD = 5e-4
MOMENT_MAX_ANGLE_STEP_RAD = 5e-5
MOMENT_ABSOLUTE_TOLERANCE_N_M = 2e-8
MOMENT_NORMALIZED_TOLERANCE = 2e-7


class LimitedLoadProbe:
    """Read-only mathematical residual and derivative adapter; not a solver."""

    def __init__(
        self,
        case,
        *,
        transport_fraction=1.0,
        workspace_limit_bytes=2**30,
        moment_target_n_m=None,
        angle_scale_rad=MOMENT_ANGLE_SCALE_RAD,
    ):
        if (
            isinstance(transport_fraction, bool)
            or not np.isfinite(transport_fraction)
            or not 0 <= transport_fraction <= 1
        ):
            raise ValueError("transport_fraction must be finite in [0,1]")
        if type(workspace_limit_bytes) is not int or workspace_limit_bytes < 1:
            raise ValueError("workspace_limit_bytes must be a positive integer")
        if moment_target_n_m is not None and (
            not hasattr(case, "axial_tilt_rad")
            or not hasattr(case, "half_length_m")
            or isinstance(moment_target_n_m, bool)
            or not isinstance(moment_target_n_m, (int, float, np.number))
            or not np.isfinite(moment_target_n_m)
            or isinstance(angle_scale_rad, bool)
            or not isinstance(angle_scale_rad, (int, float, np.number))
            or not np.isfinite(angle_scale_rad)
            or angle_scale_rad <= 0
        ):
            raise ValueError(
                "moment control requires a finite roller and finite moment/angle scale"
            )
        self.transport_fraction = float(transport_fraction)
        self.system = case.system(workspace_limit_bytes=workspace_limit_bytes)
        self.case = case
        self.n = self.system.n
        self.moment_target_n_m = (
            None if moment_target_n_m is None else float(moment_target_n_m)
        )
        self.angle_scale_rad = float(angle_scale_rad)
        self.moment_scale_n_m = (
            None if moment_target_n_m is None else case.load_n * case.half_length_m
        )
        self.moment_control = moment_target_n_m is not None
        self.size = self.system.size + int(self.moment_control)
        self._geometry_reference_m = self.system.problem.rigid_geometry_m.copy()
        self._geometry_zero_tilt_m = None
        self._y_geometry_m = None
        if self.moment_control:
            problem = self.system.problem
            self._y_geometry_m = np.broadcast_to(
                problem.y_m[:, None], problem.shape
            )
            self._geometry_zero_tilt_m = (
                self._geometry_reference_m
                - self._y_geometry_m * np.tan(case.axial_tilt_rad)
            )

    def base_state(self, state):
        state = np.asarray(state, dtype=np.float64)
        if state.shape != (self.size,) or not np.all(np.isfinite(state)):
            raise ValueError("finite full pressure/content/offset/angle state required")
        return state[:-1] if self.moment_control else state

    def angle_rad(self, state):
        if not self.moment_control:
            return float(self.case.axial_tilt_rad)
        state = np.asarray(state, dtype=np.float64)
        self.base_state(state)
        return float(state[-1] * self.angle_scale_rad)

    def offset_m(self, state):
        return float(self.base_state(state)[-1] * self.system.problem.film_scale_m)

    def rigid_geometry_m(self, state):
        if not self.moment_control:
            return self._geometry_reference_m
        angle = self.angle_rad(state)
        if not np.isfinite(angle) or abs(angle) > MOMENT_MAX_ABS_TILT_RAD:
            raise ValueError("moment-controlled tilt exceeds the qualified +/-0.5 mrad bound")
        return self._geometry_zero_tilt_m + self._y_geometry_m * np.tan(angle)

    def fields(self, state):
        base_state = self.base_state(state)
        self.system.problem.rigid_geometry_m[:] = self.rigid_geometry_m(state)
        self.system.sync(base_state)
        p, t, h, _, _ = self.system.base.unpack(base_state[:-1])
        return p * self.system.problem.pressure_scale_pa, t, h

    def directions(self, state, direction):
        state = np.asarray(state, dtype=np.float64)
        direction = np.asarray(direction, dtype=np.float64)
        if (
            state.shape != (self.size,)
            or direction.shape != (self.size,)
            or not np.all(np.isfinite(direction))
        ):
            raise ValueError("finite full bordered state and direction required")
        problem = self.system.problem
        dp, dt = np.zeros(problem.shape), np.zeros(problem.shape)
        dp[1:-1, 1:-1] = (
            direction[: self.n].reshape(dp[1:-1, 1:-1].shape) * problem.pressure_scale_pa
        )
        dt[1:-1, 1:-1] = -direction[self.n : 2 * self.n].reshape(dt[1:-1, 1:-1].shape)
        # Differentiate the same copied-content closure used by unpack().
        dt[1:-1, -1] = dt[1:-1, -2]
        dt[0], dt[-1] = dt[1], dt[-2]
        base_state = self.base_state(state)
        base_direction = direction[:-1] if self.moment_control else direction
        dh = self.system.film_direction(base_state, base_direction)
        if self.moment_control:
            angle = self.angle_rad(state)
            dh = dh + (
                self._y_geometry_m
                * (1.0 + np.tan(angle) ** 2)
                * self.angle_scale_rad
                * direction[-1]
            )
        return dh, dp, dt

    def physical_contact_moment_x_n_m(self, pressure_pa):
        problem = self.system.problem
        return float(
            -np.sum(pressure_pa * problem.y_m[:, None])
            * problem.dx_m
            * problem.dy_m
        )

    def mass(self, flux):
        base, problem = self.system.base, self.system.problem
        return (
            base.divergence(*flux) * problem.length_scale_m / base.flux_scale / base.mass_scale
        ).ravel()

    def residual(self, state):
        p, t, h = self.fields(state)
        problem = self.system.problem
        flux = limited_fluxes(
            problem.x_m, problem.y_m, h, p, t, self.case.fluid(), self.case.entrainment_m_s
        )
        if self.transport_fraction < 1:
            donor = donor_fluxes(
                problem.x_m, problem.y_m, h, p, t, self.case.fluid(), self.case.entrainment_m_s
            )
            flux = (donor[0] + self.transport_fraction * (flux[0] - donor[0]), donor[1])
        residual = np.r_[
            self.mass(flux),
            np.minimum(state[: self.n], state[self.n : 2 * self.n]),
            (problem.load_n(p) - self.case.load_n) / self.case.load_n,
        ]
        if self.moment_control:
            residual = np.r_[
                residual,
                (
                    self.physical_contact_moment_x_n_m(p) - self.moment_target_n_m
                )
                / self.moment_scale_n_m,
            ]
        return residual

    def derivative(self, state, direction, *, legacy=False):
        p, t, h = self.fields(state)
        dh, dp, dt = self.directions(state, direction)
        problem = self.system.problem
        function = limited_flux_direction if legacy else limited_flux_directional_derivative
        flux = function(
            problem.x_m,
            problem.y_m,
            h,
            p,
            t,
            dh,
            dp,
            dt,
            self.case.fluid(),
            self.case.entrainment_m_s,
        )
        if self.transport_fraction < 1:
            donor = donor_flux_direction(
                problem.x_m,
                problem.y_m,
                h,
                p,
                t,
                dh,
                dp,
                dt,
                self.case.fluid(),
                self.case.entrainment_m_s,
            )
            flux = (donor[0] + self.transport_fraction * (flux[0] - donor[0]), donor[1])
        pressure, deficit = state[: self.n], state[self.n : 2 * self.n]
        vp, vd = direction[: self.n], direction[self.n : 2 * self.n]
        complement = np.where(
            pressure < deficit, vp, np.where(deficit < pressure, vd, np.minimum(vp, vd))
        )
        result = np.r_[self.mass(flux), complement, problem.load_n(dp) / self.case.load_n]
        if self.moment_control:
            moment_direction = self.physical_contact_moment_x_n_m(dp)
            result = np.r_[result, moment_direction / self.moment_scale_n_m]
        return result

    def freeze(self, state, guide, *, branches=None):
        state = state.copy()
        p, t, h = self.fields(state)
        problem = self.system.problem
        dh, dp, dt = self.directions(state, guide)
        flux_action = freeze_limited_flux_jvp(
            problem.x_m,
            problem.y_m,
            h,
            p,
            t,
            dh,
            dp,
            dt,
            self.case.fluid(),
            self.case.entrainment_m_s,
            branches=branches,
        )
        pressure, deficit = state[: self.n], state[self.n : 2 * self.n]
        pressure_row = (pressure < deficit) | (
            (pressure == deficit) & (guide[: self.n] <= guide[self.n : 2 * self.n])
        )

        def action(direction):
            dh, dp, dt = self.directions(state, direction)
            complement = np.where(pressure_row, direction[: self.n], direction[self.n : 2 * self.n])
            if self.transport_fraction < 1:
                donor = donor_flux_direction(
                    problem.x_m,
                    problem.y_m,
                    h,
                    p,
                    t,
                    dh,
                    dp,
                    dt,
                    self.case.fluid(),
                    self.case.entrainment_m_s,
                )
                limited = flux_action(
                    dh,
                    dp,
                    dt,
                    donor_flux=donor if self.transport_fraction > 0 else None,
                )
                flux = (donor[0] + self.transport_fraction * (limited[0] - donor[0]), donor[1])
            else:
                flux = flux_action(dh, dp, dt)
            result = np.r_[
                self.mass(flux),
                complement,
                problem.load_n(dp) / self.case.load_n,
            ]
            if self.moment_control:
                result = np.r_[
                    result,
                    self.physical_contact_moment_x_n_m(dp) / self.moment_scale_n_m,
                ]
            return result

        return action

    def feasible(self, state):
        try:
            p, t, h = self.fields(state)
            feasible = bool(
                h.min() > self.system.problem.film_floor_m
                and p.min() >= -1e-10 * self.system.problem.pressure_scale_pa
                and t.min() >= -1e-10
                and t.max() <= 1 + 1e-10
            )
            if not feasible or not self.moment_control:
                return feasible
            if abs(self.angle_rad(state)) > MOMENT_MAX_ABS_TILT_RAD:
                return False
            if not np.isfinite(p).all() or p.max() <= 0.0:
                return False
            support_rows = np.flatnonzero(np.any(p > 0.01 * p.max(), axis=1))
            minimum_row = int(np.unravel_index(np.argmin(h), h.shape)[0])
            return bool(
                support_rows.size > 0
                and support_rows[0] > 1
                and support_rows[-1] < p.shape[0] - 2
                and 1 < minimum_row < p.shape[0] - 2
            )
        except ValueError:
            return False


def _difference_and_average(shape, coordinates, axis):
    indices = np.arange(np.prod(shape)).reshape(shape)
    left = indices[:, :-1] if axis == 1 else indices[:-1, :]
    right = indices[:, 1:] if axis == 1 else indices[1:, :]
    widths = np.diff(coordinates) if axis == 1 else np.diff(coordinates)[:, None]
    inverse = np.broadcast_to(1 / widths, left.shape).ravel()
    rows = np.arange(left.size)
    gradient = coo_matrix(
        (np.r_[-inverse, inverse], (np.r_[rows, rows], np.r_[left.ravel(), right.ravel()])),
        shape=(left.size, indices.size),
    ).tocsr()
    average = coo_matrix(
        (np.full(2 * left.size, 0.5), (np.r_[rows, rows], np.r_[left.ravel(), right.ravel()])),
        shape=gradient.shape,
    ).tocsr()
    return gradient, average


def _divergence(problem, axis):
    ny, nx = problem.shape
    cells = np.arange((ny - 2) * (nx - 2)).reshape(ny - 2, nx - 2)
    if axis == 1:
        faces = np.arange(ny * (nx - 1)).reshape(ny, nx - 1)
        left, right, width = faces[1:-1, :-1], faces[1:-1, 1:], problem.dx_m
    else:
        faces = np.arange((ny - 1) * nx).reshape(ny - 1, nx)
        left, right, width = faces[:-1, 1:-1], faces[1:, 1:-1], problem.dy_m
    return coo_matrix(
        (
            np.r_[-np.ones(cells.size), np.ones(cells.size)] / width,
            (np.r_[cells.ravel(), cells.ravel()], np.r_[left.ravel(), right.ravel()]),
        ),
        shape=(cells.size, faces.size),
    ).tocsr()


def _expansions(problem):
    ny, nx = problem.shape
    n = (ny - 2) * (nx - 2)
    full = np.arange(ny * nx).reshape(ny, nx)
    interior = np.arange(n).reshape(ny - 2, nx - 2)
    pressure = coo_matrix(
        (np.full(n, problem.pressure_scale_pa), (full[1:-1, 1:-1].ravel(), interior.ravel())),
        shape=(ny * nx, n),
    ).tocsr()
    mapping = np.full((ny, nx), -1)
    mapping[1:-1, 1:-1] = interior
    mapping[1:-1, -1] = mapping[1:-1, -2]
    mapping[0], mapping[-1] = mapping[1], mapping[-2]
    mask = mapping >= 0
    content = coo_matrix(
        (-np.ones(mask.sum()), (full[mask], mapping[mask])), shape=pressure.shape
    ).tocsr()
    return pressure, content


def _capacity_reconstruction_matrix(x, capacity, guide):
    """Sparse selected reconstruction, same guide as full nonlocal action."""
    west, interior = [
        _minmod_branch(v, g)
        for v, g in zip(_slope_candidates(x, capacity), _slope_candidates(x, guide), strict=True)
    ]
    ny, nx = capacity.shape
    widths = np.diff(x)
    rows, cols, data = [], [], []
    for j in range(ny):
        for i in range(nx - 1):
            weights = {i: 1.0}
            if i == 0:
                if west[j] == 0:
                    b = (2 * widths[0] + widths[1]) / (widths[0] + widths[1])
                    f = -widths[0] / (widths[0] + widths[1])
                    weights = {
                        0: 1 - 0.5 * b,
                        1: 0.5 * b - 0.5 * widths[0] * f / widths[1],
                        2: 0.5 * widths[0] * f / widths[1],
                    }
                elif west[j] == 1:
                    weights = {1: 1.0}
            else:
                branch = interior[j, i - 1]
                if branch == 0:
                    b = widths[i] / (widths[i - 1] + widths[i])
                    f = widths[i - 1] / (widths[i - 1] + widths[i])
                    factor = 0.5 * widths[i]
                    weights = {
                        i - 1: -factor * b / widths[i - 1],
                        i: 1 + factor * b / widths[i - 1] - factor * f / widths[i],
                        i + 1: factor * f / widths[i],
                    }
                elif branch == 1:
                    ratio = widths[i] / widths[i - 1]
                    weights = {i - 1: -ratio, i: 1 + ratio}
                elif branch == 2:
                    weights = {i + 1: 1.0}
            for k, value in weights.items():
                rows.append(j * (nx - 1) + i)
                cols.append(j * nx + k)
                data.append(value)
    return coo_matrix((data, (rows, cols)), shape=(ny * (nx - 1), ny * nx)).tocsr()


class LimitedLoadLinearization:
    """One immutable-state linear correction model; not a nonlinear solver.

    The fixed guide defines both analytic matrix-free products and the
    sparse preconditioner. Recreate the model to change state or guide.
    Material preconditioning freezes film; local preconditioning adds only
    cell-self elastic compliance. Offset/load and optional angle/moment
    borders remain exact.
    """

    def __init__(
        self,
        case,
        state,
        guide,
        *,
        preconditioner="material",
        active_full=None,
        transport_fraction=1.0,
        reference_state=None,
        workspace_limit_bytes=2**30,
        moment_target_n_m=None,
        angle_scale_rad=MOMENT_ANGLE_SCALE_RAD,
    ):
        if preconditioner not in ("material", "local"):
            raise ValueError("preconditioner must be material or local")
        self.probe = LimitedLoadProbe(
            case,
            transport_fraction=transport_fraction,
            workspace_limit_bytes=workspace_limit_bytes,
            moment_target_n_m=moment_target_n_m,
            angle_scale_rad=angle_scale_rad,
        )
        self.problem = self.probe.system.problem
        self.n, self.size = self.probe.n, self.probe.size
        self.target_load_n = case.load_n
        self.field_factorization = "full"
        self.state = np.asarray(state, dtype=np.float64).copy()
        self.reference_state = (
            self.state.copy()
            if reference_state is None
            else np.array(reference_state, dtype=np.float64, copy=True)
        )
        self.guide = np.asarray(guide, dtype=np.float64).copy()
        if self.guide.shape != (self.size,) or not np.all(np.isfinite(self.guide)):
            raise ValueError("finite full bordered guide required")
        self.state.setflags(write=False)
        self.guide.setflags(write=False)
        self.reference_state.setflags(write=False)
        self.probe.fields(self.state)
        self.limiter_branches = None
        if self.probe.transport_fraction > 0:
            p, theta, h = self.probe.fields(self.reference_state)
            rho, _, dr, _ = case.fluid().properties(p)
            dh, dp, dt = self.probe.directions(self.reference_state, self.guide)
            capacity = rho * h * theta
            guide_capacity = (dr * dp * h + rho * dh) * theta + rho * h * dt
            self.limiter_branches = _select_limiter_branches(
                self.problem.x_m, capacity, guide_capacity
            )
            for branch in self.limiter_branches:
                branch.setflags(write=False)
        self.action = self.probe.freeze(
            self.reference_state, self.guide, branches=self.limiter_branches
        )
        self.active_full = None
        if active_full is not None:
            active_full = np.asarray(active_full)
            if active_full.shape != (self.n,) or active_full.dtype != bool:
                raise ValueError("active_full must be a boolean interior mask")
            self.active_full = active_full.copy()
            self.active_full.setflags(write=False)
        self.preconditioner = preconditioner
        self._matrix = None

    def _check_state(self, state):
        if not np.array_equal(state, self.state):
            raise ValueError("Linear model state is frozen; recreate it for another state")

    def residual(self, state):
        self._check_state(state)
        result = self.probe.residual(self.state)
        if self.active_full is not None:
            result[self.n : 2 * self.n] = np.where(
                self.active_full, self.state[self.n : 2 * self.n], self.state[: self.n]
            )
        return result

    def jacobian_vector(self, state, direction):
        self._check_state(state)
        direction = np.asarray(direction, dtype=np.float64)
        if direction.shape != (self.size,) or not np.all(np.isfinite(direction)):
            raise ValueError("finite full bordered direction required")
        result = self.action(direction)
        if self.active_full is not None:
            result[self.n : 2 * self.n] = np.where(
                self.active_full, direction[self.n : 2 * self.n], direction[: self.n]
            )
        return result

    def merit_slope(self, state, residual, direction):
        self._check_state(state)
        return float(residual @ self.probe.derivative(self.state, direction) / self.size)

    def signature(self):
        """Identify the actual frozen model, avoiding duplicate recovery work."""
        branches = (
            ()
            if self.limiter_branches is None
            else tuple(branch.tobytes() for branch in self.limiter_branches)
        )
        if self.active_full is None:
            pressure, deficit = (
                self.reference_state[: self.n],
                self.reference_state[self.n : 2 * self.n],
            )
            full = (pressure > deficit) | (
                (pressure == deficit) & (self.guide[: self.n] > self.guide[self.n : 2 * self.n])
            )
        else:
            full = self.active_full
        return (
            self.preconditioner,
            self.probe.transport_fraction,
            self.reference_state.tobytes(),
            np.packbits(full).tobytes(),
            branches,
        )

    def frozen_matrix(self, state):
        self._check_state(state)
        if self._matrix is not None:
            return self._matrix.copy()
        problem, case = self.problem, self.probe.case
        p, theta, h = self.probe.fields(self.reference_state)
        rho, eta, dr, de = case.fluid().properties(p)
        mobility = rho * h**3 / (12 * eta)
        compliance = 0.0
        if self.preconditioner == "local":
            ny, nx = problem.shape
            compliance = problem.elastic_kernel[ny - 1, nx - 1]
        dm = mobility * (dr / rho - de / eta + 3 * compliance / h)
        dc = (dr * h + rho * compliance) * theta
        dhg, dpg, dtg = self.probe.directions(self.reference_state, self.guide)
        guide_capacity = (dr * dpg * h + rho * dhg) * theta + rho * h * dtg
        reconstruction = _capacity_reconstruction_matrix(
            problem.x_m, rho * h * theta, guide_capacity
        )
        fraction = self.probe.transport_fraction
        if fraction < 1:
            ny, nx = problem.shape
            indices = np.arange(ny * nx).reshape(ny, nx)[:, :-1].ravel()
            donor = coo_matrix(
                (np.ones(len(indices)), (np.arange(len(indices)), indices)),
                shape=reconstruction.shape,
            ).tocsr()
            reconstruction = donor + fraction * (reconstruction - donor)
        ep, et = _expansions(problem)
        gx, ax = _difference_and_average(problem.shape, problem.x_m, 1)
        gy, ay = _difference_and_average(problem.shape, problem.y_m, 0)
        dx, dy = _divergence(problem, 1), _divergence(problem, 0)
        pp = p.ravel()
        mass_p = dx @ (
            -diags(ax @ mobility.ravel()) @ gx @ ep
            - diags(gx @ pp) @ ax @ diags(dm.ravel()) @ ep
            + case.entrainment_m_s * reconstruction @ diags(dc.ravel()) @ ep
        ) + dy @ (
            -diags(ay @ mobility.ravel()) @ gy @ ep - diags(gy @ pp) @ ay @ diags(dm.ravel()) @ ep
        )
        mass_d = dx @ (case.entrainment_m_s * reconstruction @ diags((rho * h).ravel()) @ et)
        factor = (
            problem.length_scale_m
            / self.probe.system.base.flux_scale
            / self.probe.system.base.mass_scale
        )
        pressure, deficit = (
            self.reference_state[: self.n],
            self.reference_state[self.n : 2 * self.n],
        )
        pressure_row = (pressure < deficit) | (
            (pressure == deficit) & (self.guide[: self.n] <= self.guide[self.n : 2 * self.n])
        )
        if self.active_full is not None:
            pressure_row = ~self.active_full
        block = vstack(
            [
                hstack([mass_p, mass_d]) * factor,
                hstack([diags(pressure_row.astype(float)), diags((~pressure_row).astype(float))]),
            ],
            format="csc",
        )
        border_size = self.size - 2 * self.n
        columns = np.empty((2 * self.n, border_size), dtype=np.float64)
        for column_index in range(border_size):
            direction = np.zeros(self.size)
            direction[2 * self.n + column_index] = 1.0
            columns[:, column_index] = self.jacobian_vector(
                self.state, direction
            )[: 2 * self.n]
        rows = np.zeros((border_size, self.size), dtype=np.float64)
        rows[0, : self.n] = (
            problem.pressure_scale_pa * problem.dx_m * problem.dy_m / case.load_n
        )
        if self.probe.moment_control:
            moment_weights = (
                -problem.pressure_scale_pa
                * problem.y_m[1:-1, None]
                * problem.dx_m
                * problem.dy_m
                / self.probe.moment_scale_n_m
            )
            rows[1, : self.n] = np.broadcast_to(
                moment_weights, (problem.y_m.size - 2, problem.x_m.size - 2)
            ).ravel()
        self._matrix = bmat(
            [
                [block, csr_matrix(columns)],
                [
                    csr_matrix(rows[:, : 2 * self.n]),
                    csr_matrix(rows[:, 2 * self.n :]),
                ],
            ],
            format="csc",
        )
        return self._matrix.copy()
