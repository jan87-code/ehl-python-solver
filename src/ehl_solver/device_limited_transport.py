"""Device-resident MC-limited flux reconstruction and frozen JVP."""

import math


def _slope_candidates(x, capacity):
    import torch

    widths = torch.diff(x)
    differences = torch.diff(capacity, dim=1) / widths
    backward, forward = differences[:, :-1], differences[:, 1:]
    centered = (
        widths[1:] * backward + widths[:-1] * forward
    ) / (widths[:-1] + widths[1:])
    interior = torch.stack([centered, 2 * backward, 2 * forward])
    b, f = differences[:, 0], differences[:, 1]
    west = ((2 * widths[0] + widths[1]) * b - widths[0] * f) / (
        widths[0] + widths[1]
    )
    return torch.stack([west, 2 * b]), interior


def _minmod_branch(values, guide):
    import torch

    signs = torch.where(values != 0, torch.sign(values), torch.sign(guide))
    positive = torch.all(signs > 0, dim=0)
    negative = torch.all(signs < 0, dim=0)
    smallest = values == torch.amin(values, dim=0)
    largest = values == torch.amax(values, dim=0)
    increase = torch.argmin(
        torch.where(smallest, guide, torch.full_like(guide, math.inf)), dim=0
    )
    decrease = torch.argmax(
        torch.where(largest, guide, torch.full_like(guide, -math.inf)), dim=0
    )
    inactive = torch.full_like(increase, -1)
    return torch.where(positive, increase, torch.where(negative, decrease, inactive))


def _apply_branch(candidates, branch):
    import torch

    selected = torch.gather(candidates, 0, branch.clamp_min(0).unsqueeze(0)).squeeze(0)
    return torch.where(branch >= 0, selected, torch.zeros_like(selected))


def _reconstruct(x, capacity):
    import torch

    widths = torch.diff(x)
    west_candidates, interior_candidates = _slope_candidates(x, capacity)
    differences = torch.diff(capacity, dim=1) / widths
    backward, forward = differences[:, :-1], differences[:, 1:]
    monotone = ((backward > 0) & (forward > 0)) | (
        (backward < 0) & (forward < 0)
    )
    selected = torch.argmin(torch.abs(interior_candidates), dim=0)
    interior = torch.gather(
        interior_candidates, 0, selected.unsqueeze(0)
    ).squeeze(0)
    interior = torch.where(monotone, interior, torch.zeros_like(interior))

    b = differences[:, 0]
    west = west_candidates[0]
    west_active = ((west > 0) & (b > 0)) | ((west < 0) & (b < 0))
    west_centered = torch.abs(west) <= 2 * torch.abs(b)
    west_slope = torch.where(
        west_active,
        torch.where(west_centered, west, 2 * b),
        torch.zeros_like(west),
    )
    correction = torch.zeros_like(capacity[:, :-1])
    correction[:, 0] = 0.5 * widths[0] * west_slope
    correction[:, 1:] = 0.5 * widths[1:] * interior
    return capacity[:, :-1] + correction


class DeviceLimitedMCFlux:
    """Blend the donor flux with the MC-reconstructed x-face flux."""

    def __init__(self, donor_flux):
        self.donor_flux = donor_flux

    def fluxes(self, h, p, theta, transport_fraction):
        if transport_fraction == 0:
            return self.donor_flux.fluxes(h, p, theta)
        donor, rho = self.donor_flux._fluxes_with_density(h, p, theta)
        capacity = rho * h * theta
        reconstructed = _reconstruct(self.donor_flux.x, capacity)
        limited_x = donor[0] + self.donor_flux.speed * (
            reconstructed - capacity[:, :-1]
        )
        if transport_fraction == 1:
            return self.donor_flux._checked_fluxes(limited_x, donor[1])
        blended_x = donor[0] + transport_fraction * (limited_x - donor[0])
        return self.donor_flux._checked_fluxes(blended_x, donor[1])

    def linearize(
        self,
        h,
        p,
        theta,
        dh_guide,
        dp_guide,
        dt_guide,
        transport_fraction,
        *,
        branch_indices=None,
    ):
        if transport_fraction == 0:
            return None

        import torch

        rho, _, dr, _ = self.donor_flux._properties(p)
        capacity = rho * h * theta

        def capacity_direction(dh, dp, dt):
            return (dr * dp * h + rho * dh) * theta + rho * h * dt

        guide_capacity = capacity_direction(dh_guide, dp_guide, dt_guide)
        if branch_indices is None:
            base_candidates = _slope_candidates(self.donor_flux.x, capacity)
            guide_candidates = _slope_candidates(self.donor_flux.x, guide_capacity)
            west_branch, interior_branch = (
                _minmod_branch(base, guide)
                for base, guide in zip(
                    base_candidates, guide_candidates, strict=True
                )
            )
        else:
            expected_shapes = (
                (capacity.shape[0],),
                (capacity.shape[0], capacity.shape[1] - 2),
            )
            valid = isinstance(branch_indices, tuple) and len(branch_indices) == 2
            if valid:
                for branch, shape in zip(
                    branch_indices, expected_shapes, strict=True
                ):
                    if (
                        not isinstance(branch, torch.Tensor)
                        or branch.shape != shape
                        or branch.dtype != torch.int64
                        or branch.device != self.donor_flux.device
                        or bool(((branch < -1) | (branch > 2)).any())
                    ):
                        valid = False
                        break
            if not valid:
                raise ValueError("limiter branches must be valid same-device int64 arrays")
            west_branch, interior_branch = branch_indices
        widths = torch.diff(self.donor_flux.x)

        def action(dh, dp, dt, *, donor_flux):
            import torch

            direction = capacity_direction(dh, dp, dt)
            candidates = _slope_candidates(self.donor_flux.x, direction)
            west = _apply_branch(candidates[0], west_branch)
            interior = _apply_branch(candidates[1], interior_branch)
            correction = torch.zeros_like(direction[:, :-1])
            correction[:, 0] = 0.5 * widths[0] * west
            correction[:, 1:] = 0.5 * widths[1:] * interior
            limited_x = donor_flux[0] + self.donor_flux.speed * correction
            if transport_fraction == 1:
                return self.donor_flux._checked_fluxes(limited_x, donor_flux[1])
            blended_x = donor_flux[0] + transport_fraction * (
                limited_x - donor_flux[0]
            )
            if not bool(torch.isfinite(blended_x).all()):
                raise ValueError("nonfinite blended MC flux direction")
            return blended_x, donor_flux[1]

        return action
