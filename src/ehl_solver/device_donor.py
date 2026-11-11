"""Device-resident donor face fluxes and smooth flux JVP; not a JFO solver.

Fields are float64 torch tensors on one device, with SI units and shape
``(ny, nx)``. Only scalar validation predicates synchronize with the host.
"""

import math

from ehl_solver.backend import as_float64


class DeviceDonorFlux:
    """Torch implementation of the CPU donor flux for an isothermal fluid."""

    def __init__(self, x_m, y_m, fluid, speed_m_s, backend):
        import torch

        if backend.name != "torch":
            raise ValueError("device donor flux requires a torch backend")
        self.backend = backend
        self.x = as_float64(x_m, backend)
        self.y = as_float64(y_m, backend)
        if (
            self.x.ndim != 1
            or self.y.ndim != 1
            or min(self.x.numel(), self.y.numel()) < 3
            or not bool(torch.isfinite(self.x).all() and torch.isfinite(self.y).all())
            or bool((torch.diff(self.x) <= 0).any() or (torch.diff(self.y) <= 0).any())
            or not math.isfinite(speed_m_s)
            or speed_m_s < 0
        ):
            raise ValueError("finite ascending axes and nonnegative speed required")
        self.shape = (self.y.numel(), self.x.numel())
        self.dx = torch.diff(self.x)
        self.dy = torch.diff(self.y)[:, None]
        self.fluid = fluid
        self.speed = speed_m_s
        self.device = self.x.device

    def _fields(self, *fields):
        import torch

        for field in fields:
            if (
                not isinstance(field, torch.Tensor)
                or field.shape != self.shape
                or field.dtype != torch.float64
                or field.device != self.device
                or not bool(torch.isfinite(field).all())
            ):
                raise ValueError("fields must be finite same-device float64 tensors of grid shape")

    def _properties(self, pressure):
        import torch

        fluid = self.fluid
        if bool((pressure <= -0.5 * fluid.roelands_pressure_pa).any()):
            raise ValueError("invalid pressure domain")
        denominator = 1 + fluid.density_b_per_pa * pressure
        if bool((denominator <= 0).any()):
            raise ValueError("density pole reached")
        rho = fluid.density_kg_m3 * (
            1 + fluid.density_a_per_pa * pressure / denominator
        )
        dr = fluid.density_kg_m3 * fluid.density_a_per_pa / denominator**2
        logarithm = math.log(fluid.viscosity_pa_s / fluid.viscosity_reference_pa_s)
        t = torch.log1p(pressure / fluid.roelands_pressure_pa)
        eta = fluid.viscosity_pa_s * torch.exp(
            logarithm * torch.expm1(fluid.viscosity_index * t)
        )
        de = eta * logarithm * fluid.viscosity_index / fluid.roelands_pressure_pa * torch.exp(
            (fluid.viscosity_index - 1) * t
        )
        if bool((rho <= 0).any() or (eta <= 0).any()) or not all(
            bool(torch.isfinite(value).all()) for value in (rho, dr, eta, de)
        ):
            raise ValueError("nonpositive or nonfinite constitutive state")
        return rho, eta, dr, de

    @staticmethod
    def _checked_fluxes(qx, qy):
        import torch

        if not bool(torch.isfinite(qx).all() and torch.isfinite(qy).all()):
            raise ValueError("nonfinite donor flux")
        return qx, qy

    def fluxes(self, h, p, theta):
        """Return x/y face mass fluxes [kg/(m s)] with no field clipping."""
        self._fields(h, p, theta)
        if bool((h <= 0).any()):
            raise ValueError("film must be positive")
        rho, eta, _, _ = self._properties(p)
        mobility = rho * h**3 / (12 * eta)
        qx = -0.5 * (mobility[:, 1:] + mobility[:, :-1]) * (p[:, 1:] - p[:, :-1]) / self.dx
        qx += self.speed * (rho * h * theta)[:, :-1]
        qy = -0.5 * (mobility[1:] + mobility[:-1]) * (p[1:] - p[:-1]) / self.dy
        return self._checked_fluxes(qx, qy)

    def direction(self, h, p, theta, dh, dp, dt):
        """Return the analytic smooth face-flux JVP for supplied directions."""
        self._fields(h, p, theta, dh, dp, dt)
        if bool((h <= 0).any()):
            raise ValueError("film must be positive")
        rho, eta, dr, de = self._properties(p)
        mobility = rho * h**3 / (12 * eta)
        dm = mobility * ((dr / rho - de / eta) * dp + 3 * dh / h)
        dc = dr * dp * h + rho * dh
        qx = -0.5 * (dm[:, 1:] + dm[:, :-1]) * (p[:, 1:] - p[:, :-1]) / self.dx
        qx -= 0.5 * (mobility[:, 1:] + mobility[:, :-1]) * (
            dp[:, 1:] - dp[:, :-1]
        ) / self.dx
        qx += self.speed * (dc * theta + rho * h * dt)[:, :-1]
        qy = -0.5 * (dm[1:] + dm[:-1]) * (p[1:] - p[:-1]) / self.dy
        qy -= 0.5 * (mobility[1:] + mobility[:-1]) * (dp[1:] - dp[:-1]) / self.dy
        return self._checked_fluxes(qx, qy)
