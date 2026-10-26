"""Device-resident signed full-film equations shared by audits and device solvers.

The optional resident GMRES/Newton path is in device_krylov/device_newton;
the NumPy/SciPy reference remains separate. No host-vector bridge into SciPy
inner iterations is provided.
"""

import math

from ehl_solver.backend import as_float64, make_backend
from ehl_solver.elasticity import ElasticityOperator
from ehl_solver.reynolds import reynolds_residual


class DeviceFullFilmEvaluator:
    """Snapshot a BorderedReynoldsElasticityProblem for explicit operator audits.

    State is interior dimensionless P in row-major order followed by h0/h_ref.
    The complete scaled Reynolds/load residual and analytic state/compliance
    JVP retain the reference's stencil, load quadrature and strict film floor.
    Fields and vectors remain on device; validity predicates synchronize scalars.
    Recreate the snapshot after changing the CPU problem (including load/gamma).
    """

    def __init__(self, problem, backend=None, *, memory_limit_bytes=2**30):
        self.backend = make_backend("torch", "cuda") if backend is None else backend
        self.shape = (problem.y.size, problem.x.size)
        self.interior_shape = problem.interior_shape
        self.pressure_size = problem.pressure_size
        for name in (
            "film_scale_m", "pressure_scale_pa", "compliance_factor", "film_floor_m",
            "dx", "dy", "reynolds_parameter", "reynolds_residual_scale",
            "target_load_n", "load_residual_scale_n",
            "dx_m", "dy_m",
        ):
            setattr(self, name, getattr(problem, name))
        self.geometry = as_float64(problem.rigid_geometry_m.copy(), self.backend)
        self.load_gradient = as_float64(problem.load_gradient().copy(), self.backend)
        self.elasticity = ElasticityOperator(
            self.shape, problem.dx_m, problem.dy_m, problem.e_star_pa, self.backend,
            memory_limit_bytes=memory_limit_bytes,
        )

    def _vector(self, values):
        vector = as_float64(values, self.backend)
        if tuple(vector.shape) != (self.pressure_size + 1,):
            raise ValueError("state/direction must have pressure_size + 1 entries")
        if not bool(self.backend.xp.isfinite(vector).all()):
            raise ValueError("state/direction must be finite")
        return vector

    def _pressure(self, interior):
        # Allocate through a resident tensor so dtype/device are inherited.
        p = self.backend.xp.zeros_like(self.geometry)
        p[1:-1, 1:-1] = interior.reshape(self.interior_shape)
        return p

    def state_fields(self, state):
        z = self._vector(state)
        p = self._pressure(z[:-1])
        w = self.elasticity(p * self.pressure_scale_pa)
        h = z[-1] * self.film_scale_m + self.geometry + self.compliance_factor * w
        if not bool(self.backend.xp.isfinite(h).all()) or bool((h <= self.film_floor_m).any()):
            raise ValueError("state film thickness must be strictly above film_floor_m")
        return p, h, w

    def _join(self, reynolds, load):
        xp = self.backend.xp
        parts = ((reynolds / self.reynolds_residual_scale).reshape(-1), load.reshape(1))
        result = xp.concatenate(parts) if self.backend.name == "numpy" else xp.cat(parts)
        if not bool(xp.isfinite(result).all()):
            raise ValueError("non-finite full-film operator result")
        return result

    def residual(self, state):
        p, h, _ = self.state_fields(state)
        r = reynolds_residual(
            p, h / self.film_scale_m, self.dx, self.dy, self.reynolds_parameter,
            backend=self.backend,
        )
        # Boundary pressure is zero; interior trapezoid weights are exactly one.
        load = (p[1:-1, 1:-1].reshape(-1) * self.load_gradient).sum()
        load = load - self.target_load_n / self.load_residual_scale_n
        return self._join(r, load)

    def jacobian_vector(self, state, direction, *, compliance_direction=0.0):
        """Complete analytic derivative, including elastic, h0 and force rows."""
        if not math.isfinite(compliance_direction):
            raise ValueError("compliance_direction must be finite")
        v = self._vector(direction)
        return self._jacobian_fields(v, self.state_fields(state), compliance_direction)

    def linearize(self, state):
        """Freeze state fields once for repeated analytic Krylov products."""
        fields = self.state_fields(state)

        def matvec(direction):
            return self._jacobian_fields(self._vector(direction), fields, 0.)

        return matvec

    def _jacobian_fields(self, v, fields, compliance_direction):
        p, film, w = fields
        dp = self._pressure(v[:-1])
        dw = self.elasticity(dp * self.pressure_scale_pa)
        h = film / self.film_scale_m
        dh = (
            v[-1] * self.film_scale_m + self.compliance_factor * dw
            + compliance_direction * w
        ) / self.film_scale_m
        h3, dh3 = h**3, 3.0 * h**2 * dh
        c = (slice(1, -1), slice(1, -1))
        result = self.backend.xp.zeros_like(p[c])
        for neighbor, spacing in (
            ((slice(1, -1), slice(2, None)), self.dx),
            ((slice(1, -1), slice(None, -2)), self.dx),
            ((slice(2, None), slice(1, -1)), self.dy),
            ((slice(None, -2), slice(1, -1)), self.dy),
        ):
            result += (
                .5 * (dh3[c] + dh3[neighbor]) * (p[neighbor] - p[c])
                + .5 * (h3[c] + h3[neighbor]) * (dp[neighbor] - dp[c])
            ) / spacing**2
        result -= self.reynolds_parameter * (dh[1:-1, 2:] - dh[1:-1, :-2]) / (2 * self.dx)
        return self._join(result, (v[:-1] * self.load_gradient).sum())
