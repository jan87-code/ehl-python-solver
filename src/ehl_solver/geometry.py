"""Undeformed contact geometry models with SI coordinates and heights."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Callable, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray


class ContactGeometry(Protocol):
    """Geometry interface; ``h_geom`` returns an ``(ny, nx)`` field in metres."""

    def h_geom(self, x_m: ArrayLike, y_m: ArrayLike) -> NDArray[np.float64]:
        """Evaluate undeformed separation on ascending 1-D coordinate axes."""


def _coordinates(x_m: ArrayLike, y_m: ArrayLike):
    x = np.asarray(x_m, dtype=np.float64)
    y = np.asarray(y_m, dtype=np.float64)
    for axis, name in ((x, "x_m"), (y, "y_m")):
        if (
            axis.ndim != 1
            or axis.size < 1
            or not np.all(np.isfinite(axis))
            or np.any(np.diff(axis) <= 0)
        ):
            raise ValueError(f"{name} must be a finite ascending 1-D SI coordinate axis")
    return np.meshgrid(x, y, indexing="xy")


def _positive_finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive finite SI")
    return float(value)


@dataclass(frozen=True)
class CircularPointContactGeometry:
    """Parabolic circular point-contact geometry."""

    radius_m: float

    def __post_init__(self):
        object.__setattr__(self, "radius_m", _positive_finite(self.radius_m, "radius_m"))

    def h_geom(self, x_m: ArrayLike, y_m: ArrayLike) -> NDArray[np.float64]:
        xx, yy = _coordinates(x_m, y_m)
        return xx**2 / (2.0 * self.radius_m) + yy**2 / (2.0 * self.radius_m)


@dataclass(frozen=True)
class EllipticalPointContactGeometry:
    """Parabolic point-contact geometry with independent principal radii."""

    radius_x_m: float
    radius_y_m: float

    def __post_init__(self):
        object.__setattr__(self, "radius_x_m", _positive_finite(self.radius_x_m, "radius_x_m"))
        object.__setattr__(self, "radius_y_m", _positive_finite(self.radius_y_m, "radius_y_m"))

    def h_geom(self, x_m: ArrayLike, y_m: ArrayLike) -> NDArray[np.float64]:
        xx, yy = _coordinates(x_m, y_m)
        return xx**2 / (2.0 * self.radius_x_m) + yy**2 / (2.0 * self.radius_y_m)


@dataclass(frozen=True)
class LineContactGeometry:
    """Classical line-contact parabola, invariant in the transverse direction."""

    radius_x_m: float

    def __post_init__(self):
        object.__setattr__(self, "radius_x_m", _positive_finite(self.radius_x_m, "radius_x_m"))

    def h_geom_x(self, x_m: ArrayLike) -> NDArray[np.float64]:
        """Evaluate the line-contact cross-section on an ascending x axis."""
        x = np.asarray(x_m, dtype=np.float64)
        if (
            x.ndim != 1
            or x.size < 1
            or not np.all(np.isfinite(x))
            or np.any(np.diff(x) <= 0)
        ):
            raise ValueError("x_m must be a finite ascending 1-D SI coordinate axis")
        return x**2 / (2.0 * self.radius_x_m)

    def h_geom(self, x_m: ArrayLike, y_m: ArrayLike) -> NDArray[np.float64]:
        xx, yy = _coordinates(x_m, y_m)
        return np.broadcast_to(self.h_geom_x(x_m)[None, :], yy.shape).copy()


@dataclass(frozen=True)
class ProfiledRollerGeometry:
    """Parabolic rolling-direction geometry plus an arbitrary axial profile ``g(y)``.

    The profile callable receives the transverse coordinate vector in metres and
    must return finite height values in metres, either scalar, shape ``(ny,)``,
    or an ``(ny, nx)`` field. Crown and tilt models are intentionally not
    supplied by this interface.
    """

    radius_x_m: float
    axial_profile_m: Callable[[NDArray[np.float64]], ArrayLike]

    def __post_init__(self):
        object.__setattr__(self, "radius_x_m", _positive_finite(self.radius_x_m, "radius_x_m"))
        if not callable(self.axial_profile_m):
            raise ValueError("axial_profile_m must be callable")

    def h_geom(self, x_m: ArrayLike, y_m: ArrayLike) -> NDArray[np.float64]:
        xx, yy = _coordinates(x_m, y_m)
        profile_values = self.axial_profile_m(np.asarray(y_m, dtype=np.float64))
        profile = np.asarray(profile_values, dtype=np.float64)
        if profile.ndim == 0:
            axial = np.full(yy.shape, float(profile), dtype=np.float64)
        elif profile.shape == (yy.shape[0],):
            axial = np.broadcast_to(profile[:, None], yy.shape)
        elif profile.shape == yy.shape:
            axial = profile
        else:
            raise ValueError("axial_profile_m must return scalar, (ny,), or (ny,nx) heights")
        if not np.all(np.isfinite(axial)):
            raise ValueError("axial_profile_m must return finite SI heights")
        return xx**2 / (2.0 * self.radius_x_m) + axial
