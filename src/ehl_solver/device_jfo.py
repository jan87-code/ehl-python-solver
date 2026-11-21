"""Device-resident donor/JFO residual and bordered products, not a solver.

SI physical inputs; fields have shape (ny,nx), +x entrainment. Internal
state is C-order interior P=p/p_ref, D=1-theta, followed by h0/h_ref.
Moment mode appends alpha/(5e-4 rad); the transport fraction blends donor
and MC-limited x-face fluxes.
"""

import math

from ehl_solver.backend import as_float64, make_backend
from ehl_solver.device_donor import DeviceDonorFlux
from ehl_solver.device_limited_transport import DeviceLimitedMCFlux
from ehl_solver.elasticity import ElasticityOperator
from ehl_solver.limited_system import (
    MOMENT_ANGLE_SCALE_RAD,
    MOMENT_MAX_ABS_TILT_RAD,
)


class DeviceDonorJFO:
    """Evaluate donor/blended-MC JFO residuals and full-source state JVPs."""

    def __init__(
        self,
        case,
        backend=None,
        *,
        transport_fraction=0.0,
        memory_limit_bytes=2**30,
        moment_target_n_m=None,
        angle_scale_rad=MOMENT_ANGLE_SCALE_RAD,
    ):
        if case.transport != "donor":
            raise ValueError("device JFO requires donor transport")
        if (
            isinstance(transport_fraction, bool)
            or not isinstance(transport_fraction, (int, float))
            or not math.isfinite(transport_fraction)
            or not 0 <= transport_fraction <= 1
        ):
            raise ValueError("transport_fraction must be finite in [0,1]")
        if moment_target_n_m is not None and (
            not hasattr(case, "axial_tilt_rad")
            or not hasattr(case, "half_length_m")
            or isinstance(moment_target_n_m, bool)
            or not isinstance(moment_target_n_m, (int, float))
            or not math.isfinite(moment_target_n_m)
            or isinstance(angle_scale_rad, bool)
            or not isinstance(angle_scale_rad, (int, float))
            or not math.isfinite(angle_scale_rad)
            or angle_scale_rad <= 0
        ):
            raise ValueError(
                "moment control requires a finite roller and finite moment/angle scale"
            )
        self.backend = make_backend("torch", "cuda") if backend is None else backend
        if self.backend.name != "torch":
            raise ValueError("device JFO requires a torch backend")
        self.transport_fraction = float(transport_fraction)
        self.moment_control = moment_target_n_m is not None
        self.moment_target_n_m = (
            None if moment_target_n_m is None else float(moment_target_n_m)
        )
        self.angle_scale_rad = float(angle_scale_rad)
        problem = case.problem()
        self.shape = problem.shape
        self.interior_shape = (self.shape[0] - 2, self.shape[1] - 2)
        self.n = self.interior_shape[0] * self.interior_shape[1]
        self.size = 2 * self.n + 1 + int(self.moment_control)
        self.pressure_scale = problem.pressure_scale_pa
        self.film_scale = problem.film_scale_m
        self.e_star_pa = problem.e_star_pa
        self.film_floor = problem.film_floor_m
        self.inlet = problem.inlet_fractional_film_content
        self.dx, self.dy = problem.dx_m, problem.dy_m
        self.target_load = case.load_n
        self.moment_scale = (
            None if not self.moment_control else case.load_n * case.half_length_m
        )
        self.offset_index = -2 if self.moment_control else -1
        fluid = case.fluid()
        flux_scale = (
            fluid.density_kg_m3 * self.film_scale**3 * self.pressure_scale
            / (12 * fluid.viscosity_pa_s * problem.length_scale_m)
        )
        self.mass_factor = problem.length_scale_m / flux_scale / 100.0
        geometry = problem.rigid_geometry_m.copy()
        self.geometry = as_float64(geometry, self.backend)
        self.geometry_zero_tilt = None
        self.y_geometry = None
        if self.moment_control:
            import numpy as np

            y_geometry = np.broadcast_to(problem.y_m[:, None], problem.shape)
            geometry_zero_tilt = (
                geometry - y_geometry * math.tan(case.axial_tilt_rad)
            )
            self.geometry_zero_tilt = as_float64(
                geometry_zero_tilt, self.backend
            )
            self.y_geometry = as_float64(y_geometry, self.backend)
        self.flux = DeviceDonorFlux(
            problem.x_m, problem.y_m, fluid, case.entrainment_m_s, self.backend
        )
        self.limited_flux = DeviceLimitedMCFlux(self.flux)
        self.elasticity = ElasticityOperator(
            self.shape, self.dx, self.dy, problem.e_star_pa, self.backend,
            memory_limit_bytes=memory_limit_bytes,
        )

    def _vector(self, vector):
        import torch

        if (
            not isinstance(vector, torch.Tensor)
            or vector.shape != (self.size,)
            or vector.dtype != torch.float64
            or vector.device != self.geometry.device
            or not bool(torch.isfinite(vector).all())
        ):
            raise ValueError("finite same-device float64 bordered vector required")
        return vector

    def _expanded(self, vector, *, direction=False):
        import torch

        p = torch.zeros_like(self.geometry)
        theta = torch.zeros_like(self.geometry) if direction else torch.ones_like(self.geometry)
        p[1:-1, 1:-1] = vector[:self.n].reshape(self.interior_shape) * self.pressure_scale
        d = vector[self.n:2 * self.n].reshape(self.interior_shape)
        theta[1:-1, 1:-1] = -d if direction else 1 - d
        if not direction:
            theta[1:-1, 0] = self.inlet
        theta[1:-1, -1] = theta[1:-1, -2]
        theta[0], theta[-1] = theta[1], theta[-2]
        return p, theta

    def state_fields(self, state):
        """Return pressure [Pa], theta, film and deformation [m], all resident."""
        import torch

        z = self._vector(state)
        p, theta = self._expanded(z)
        w = self.elasticity(p)
        geometry = self.geometry
        if self.moment_control:
            angle = z[-1] * self.angle_scale_rad
            if bool((torch.abs(angle) > MOMENT_MAX_ABS_TILT_RAD).item()):
                raise ValueError("moment-controlled tilt exceeds the +/-0.5 mrad bound")
            geometry = self.geometry_zero_tilt + self.y_geometry * torch.tan(angle)
        film = z[self.offset_index] * self.film_scale + geometry + w
        if not bool(torch.isfinite(film).all()) or bool((film <= self.film_floor).any()):
            raise ValueError("film must be finite and strictly above film floor")
        return p, theta, film, w

    def _mass(self, flux):
        import torch

        qx, qy = flux
        return (
            torch.diff(qx[1:-1], dim=1) / self.dx
            + torch.diff(qy[:, 1:-1], dim=0) / self.dy
        ).reshape(-1) * self.mass_factor

    def _film_direction(self, state, direction, pressure_direction):
        import torch

        dh = (
            self.elasticity(pressure_direction)
            + direction[self.offset_index] * self.film_scale
        )
        if self.moment_control:
            angle = state[-1] * self.angle_scale_rad
            dh = dh + (
                self.y_geometry
                * (1.0 + torch.tan(angle) ** 2)
                * self.angle_scale_rad
                * direction[-1]
            )
        return dh

    def _join(self, mass, constraint, load, moment=None):
        import torch

        parts = (mass, constraint, load.reshape(1))
        if self.moment_control:
            if moment is None:
                raise ValueError("moment residual is required for a two-border device system")
            parts = (*parts, moment.reshape(1))
        result = torch.cat(parts)
        if not bool(torch.isfinite(result).all()):
            raise ValueError("nonfinite donor/JFO operator result")
        return result

    def residual(self, state):
        import torch

        z = self._vector(state)
        p, theta, h, _ = self.state_fields(z)
        mass = self._mass(
            self.limited_flux.fluxes(h, p, theta, self.transport_fraction)
        )
        constraint = torch.minimum(z[:self.n], z[self.n:2 * self.n])
        load = (p[1:-1, 1:-1].sum() * (self.dx * self.dy) - self.target_load) / self.target_load
        if self.moment_control:
            moment = (
                -torch.sum(p * self.y_geometry)
                * (self.dx * self.dy)
                - self.moment_target_n_m
            ) / self.moment_scale
        else:
            moment = None
        return self._join(mass, constraint, load, moment)

    def linear_residual(self, state, active_full):
        """Return the CPU Newton model residual with its selected contact rows."""
        import torch

        z = self._vector(state)
        if (
            not isinstance(active_full, torch.Tensor)
            or active_full.shape != (self.n,)
            or active_full.dtype != torch.bool
            or active_full.device != z.device
        ):
            raise ValueError("active_full must be a same-device boolean interior mask")
        result = self.residual(z)
        result[self.n:2 * self.n] = torch.where(
            active_full, z[self.n:2 * self.n], z[:self.n]
        )
        return result

    def _action(self, state, direction, fields, pressure_row, flux_action=None):
        import torch

        z = self._vector(state)
        v = self._vector(direction)
        p, theta, h, _ = fields
        dp, dt = self._expanded(v, direction=True)
        dh = self._film_direction(z, v, dp)
        donor = self.flux.direction(h, p, theta, dh, dp, dt)
        flux = (
            donor
            if flux_action is None
            else flux_action(dh, dp, dt, donor_flux=donor)
        )
        mass = self._mass(flux)
        constraint = torch.where(pressure_row, v[:self.n], v[self.n:2 * self.n])
        load = dp[1:-1, 1:-1].sum() * (self.dx * self.dy / self.target_load)
        if self.moment_control:
            moment = (
                -torch.sum(dp * self.y_geometry)
                * (self.dx * self.dy / self.moment_scale)
            )
        else:
            moment = None
        return self._join(mass, constraint, load, moment)

    def linearize(
        self, state, guide, *, active_full=None, limiter_branches=None
    ):
        """Freeze state and a selected generalized derivative for Krylov use."""
        import torch

        z = self._vector(state)
        g = self._vector(guide)
        fields = self.state_fields(z)
        if active_full is None:
            p, d = z[:self.n], z[self.n:2 * self.n]
            pressure_row = (p < d) | ((p == d) & (g[:self.n] <= g[self.n:2 * self.n]))
        else:
            if (
                not isinstance(active_full, torch.Tensor)
                or active_full.shape != (self.n,)
                or active_full.dtype != torch.bool
                or active_full.device != z.device
            ):
                raise ValueError("active_full must be a same-device boolean interior mask")
            pressure_row = ~active_full.clone()
        flux_action = None
        if self.transport_fraction > 0:
            p, theta, h, _ = fields
            dp_guide, dt_guide = self._expanded(g, direction=True)
            dh_guide = self._film_direction(z, g, dp_guide)
            flux_action = self.limited_flux.linearize(
                h,
                p,
                theta,
                dh_guide,
                dp_guide,
                dt_guide,
                self.transport_fraction,
                branch_indices=limiter_branches,
            )
        return lambda direction: self._action(
            z, direction, fields, pressure_row, flux_action
        )

    def directional_derivative(self, state, direction):
        """True one-sided min(P,D) derivative, including exact ties."""
        z, v = self._vector(state), self._vector(direction)
        p, d = z[:self.n], z[self.n:2 * self.n]
        pressure_row = (p < d) | ((p == d) & (v[:self.n] <= v[self.n:2 * self.n]))
        fields = self.state_fields(z)
        flux_action = None
        if self.transport_fraction > 0:
            pressure, theta, h, _ = fields
            dp, dt = self._expanded(v, direction=True)
            dh = self._film_direction(z, v, dp)
            flux_action = self.limited_flux.linearize(
                h, pressure, theta, dh, dp, dt, self.transport_fraction
            )
        return self._action(z, v, fields, pressure_row, flux_action)
