"""Device-resident donor/JFO residual and bordered products, not a solver.

SI physical inputs; fields have shape (ny,nx), +x entrainment. Internal
state is C-order interior P=p/p_ref, D=1-theta, followed by h0/h_ref.
This snapshots the full-strength donor endpoint of LimitedLoadProbe.
"""

from ehl_solver.backend import as_float64, make_backend
from ehl_solver.device_donor import DeviceDonorFlux
from ehl_solver.elasticity import ElasticityOperator


class DeviceDonorJFO:
    """Evaluate the donor endpoint and its full-source elastic state JVP."""

    def __init__(self, case, backend=None, *, memory_limit_bytes=2**30):
        if case.transport != "donor":
            raise ValueError("device JFO requires donor transport")
        self.backend = make_backend("torch", "cuda") if backend is None else backend
        if self.backend.name != "torch":
            raise ValueError("device JFO requires a torch backend")
        problem = case.problem()
        self.shape = problem.shape
        self.interior_shape = (self.shape[0] - 2, self.shape[1] - 2)
        self.n = self.interior_shape[0] * self.interior_shape[1]
        self.size = 2 * self.n + 1
        self.pressure_scale = problem.pressure_scale_pa
        self.film_scale = problem.film_scale_m
        self.film_floor = problem.film_floor_m
        self.inlet = problem.inlet_fractional_film_content
        self.dx, self.dy = problem.dx_m, problem.dy_m
        self.target_load = case.load_n
        fluid = case.fluid()
        flux_scale = (
            fluid.density_kg_m3 * self.film_scale**3 * self.pressure_scale
            / (12 * fluid.viscosity_pa_s * problem.length_scale_m)
        )
        self.mass_factor = problem.length_scale_m / flux_scale / 100.0
        self.geometry = as_float64(problem.rigid_geometry_m.copy(), self.backend)
        self.flux = DeviceDonorFlux(
            problem.x_m, problem.y_m, fluid, case.entrainment_m_s, self.backend
        )
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
        film = z[-1] * self.film_scale + self.geometry + w
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

    def _join(self, mass, constraint, load):
        import torch

        result = torch.cat((mass, constraint, load.reshape(1)))
        if not bool(torch.isfinite(result).all()):
            raise ValueError("nonfinite donor/JFO operator result")
        return result

    def residual(self, state):
        import torch

        z = self._vector(state)
        p, theta, h, _ = self.state_fields(z)
        mass = self._mass(self.flux.fluxes(h, p, theta))
        constraint = torch.minimum(z[:self.n], z[self.n:2 * self.n])
        load = (p[1:-1, 1:-1].sum() * (self.dx * self.dy) - self.target_load) / self.target_load
        return self._join(mass, constraint, load)

    def _action(self, direction, fields, pressure_row):
        import torch

        v = self._vector(direction)
        p, theta, h, _ = fields
        dp, dt = self._expanded(v, direction=True)
        dh = self.elasticity(dp) + v[-1] * self.film_scale
        mass = self._mass(self.flux.direction(h, p, theta, dh, dp, dt))
        constraint = torch.where(pressure_row, v[:self.n], v[self.n:2 * self.n])
        load = dp[1:-1, 1:-1].sum() * (self.dx * self.dy / self.target_load)
        return self._join(mass, constraint, load)

    def linearize(self, state, guide, *, active_full=None):
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
        return lambda direction: self._action(direction, fields, pressure_row)

    def directional_derivative(self, state, direction):
        """True one-sided min(P,D) derivative, including exact ties."""
        z, v = self._vector(state), self._vector(direction)
        p, d = z[:self.n], z[self.n:2 * self.n]
        pressure_row = (p < d) | ((p == d) & (v[:self.n] <= v[self.n:2 * self.n]))
        return self._action(v, self.state_fields(z), pressure_row)
