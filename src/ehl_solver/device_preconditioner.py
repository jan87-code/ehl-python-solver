"""Explicit GPU frozen-film preconditioners; no emulation of sparse ILU."""

import math


def _dst1(values, axis):
    """Unnormalized DST-I via odd extension; equivalent to scipy.fft.dst(type=1)."""
    import torch

    n = values.shape[axis]
    shape = list(values.shape)
    shape[axis] = 1
    zero = values.new_zeros(shape)
    extended = torch.cat((zero, values, zero, -values.flip((axis,))), dim=axis)
    return -torch.fft.rfft(extended, dim=axis).imag.narrow(axis, 1, n)


class DeviceFrozenFilmPreconditioner:
    """Exact Jacobi or exact inverse of a mobility-scaled Laplacian surrogate.

    For k=H**3, S=diag(sqrt(k_interior)), L=positive Dirichlet -Laplacian,
    the laplacian strategy applies (-S L S / residual_scale)^-1 by DST-I.
    This is a declared approximation to the variable-face-conductance Reynolds
    block, not its exact inverse and not an ILU approximation. No elastic or
    entrainment derivative is included. Shapes are (ny,nx), spacings dimensionless.
    """

    def __init__(self, evaluator, state, *, strategy="laplacian"):
        import torch

        if evaluator.backend.name != "torch":
            raise ValueError("device preconditioner requires an explicit torch backend")
        if strategy not in ("laplacian", "jacobi", "none"):
            raise ValueError("strategy must be laplacian, jacobi or none; ILU is CPU-only")
        self.strategy = strategy
        _, film, _ = evaluator.state_fields(state)
        h = film / evaluator.film_scale_m
        self.k = h**3
        self.scale = evaluator.reynolds_residual_scale
        self.dx, self.dy = evaluator.dx, evaluator.dy
        self.shape = tuple(h.shape)
        c = self.k[1:-1, 1:-1]
        self.diagonal = -(
            .5 * (2*c + self.k[1:-1, 2:] + self.k[1:-1, :-2]) / self.dx**2
            + .5 * (2*c + self.k[2:, 1:-1] + self.k[:-2, 1:-1]) / self.dy**2
        ).reshape(-1) / self.scale
        if not bool(torch.isfinite(self.diagonal).all()) or bool((self.diagonal == 0).any()):
            raise ValueError("invalid frozen-film diagonal")
        self.inverse_sqrt_k = h[1:-1, 1:-1]**(-1.5)
        ny, nx = self.shape[0] - 2, self.shape[1] - 2
        j = torch.arange(1, ny+1, dtype=h.dtype, device=h.device)
        i = torch.arange(1, nx+1, dtype=h.dtype, device=h.device)
        self.eigenvalues = (
            4 * torch.sin(math.pi*j[:, None] / (2*(ny+1)))**2 / self.dy**2
            + 4 * torch.sin(math.pi*i[None, :] / (2*(nx+1)))**2 / self.dx**2
        )
        self.dst_normalization = 4 * (ny+1) * (nx+1)

    def __call__(self, rhs):
        if self.strategy == "none":
            return rhs
        if self.strategy == "jacobi":
            return rhs / self.diagonal
        field = rhs.reshape(self.inverse_sqrt_k.shape) * self.inverse_sqrt_k
        spectrum = _dst1(_dst1(field, 0), 1) / self.eigenvalues
        inverse = _dst1(_dst1(spectrum, 0), 1) / self.dst_normalization
        return (-self.scale * self.inverse_sqrt_k * inverse).reshape(-1)

    def matrix_vector(self, vector):
        """Apply the actual frozen-film face-flux matrix, for independent audits."""
        import torch

        p = torch.zeros_like(self.k)
        p[1:-1, 1:-1] = vector.reshape(self.inverse_sqrt_k.shape)
        k, c = self.k, self.k[1:-1, 1:-1]
        result = torch.zeros_like(c)
        for neighbor, spacing in (
            ((slice(1, -1), slice(2, None)), self.dx),
            ((slice(1, -1), slice(None, -2)), self.dx),
            ((slice(2, None), slice(1, -1)), self.dy),
            ((slice(None, -2), slice(1, -1)), self.dy),
        ):
            result += .5*(c + k[neighbor])*(p[neighbor] - p[1:-1, 1:-1]) / spacing**2
        return (result / self.scale).reshape(-1)
