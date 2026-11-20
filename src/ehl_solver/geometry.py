"""Undeformed contact geometry models with SI coordinates and heights."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, tan
from typing import Callable, Mapping, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray


class ContactGeometry(Protocol):
    """Additive gap component; ``h_geom`` returns ``(ny,nx)`` metres."""

    def h_geom(
        self, x_m: ArrayLike, y_m: ArrayLike, *, time_s: float | None = None
    ) -> NDArray[np.float64]:
        """Evaluate a geometry/gap component on ascending SI coordinate axes."""


class AxialProfile(Protocol):
    """Callable axial profile ``g(y)`` returning heights in metres."""

    def __call__(self, y_m: NDArray[np.float64]) -> ArrayLike:
        """Evaluate profile heights for axial coordinates in metres."""


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


def _validate_time(time_s: float | None) -> None:
    if time_s is not None and (
        isinstance(time_s, bool) or not isfinite(time_s) or time_s < 0.0
    ):
        raise ValueError("time_s must be finite and nonnegative")


@dataclass(frozen=True)
class CircularPointContactGeometry:
    """Parabolic circular point-contact geometry."""

    radius_m: float

    def __post_init__(self):
        object.__setattr__(self, "radius_m", _positive_finite(self.radius_m, "radius_m"))

    def h_geom(
        self, x_m: ArrayLike, y_m: ArrayLike, *, time_s: float | None = None
    ) -> NDArray[np.float64]:
        _validate_time(time_s)
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

    def h_geom(
        self, x_m: ArrayLike, y_m: ArrayLike, *, time_s: float | None = None
    ) -> NDArray[np.float64]:
        _validate_time(time_s)
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

    def h_geom(
        self, x_m: ArrayLike, y_m: ArrayLike, *, time_s: float | None = None
    ) -> NDArray[np.float64]:
        _validate_time(time_s)
        _, yy = _coordinates(x_m, y_m)
        return np.broadcast_to(self.h_geom_x(x_m)[None, :], yy.shape).copy()


@dataclass(frozen=True)
class ZeroAxialProfile:
    """Cylindrical/no-crown axial profile ``g(y)=0``."""

    def __call__(self, y_m: NDArray[np.float64]) -> NDArray[np.float64]:
        y = np.asarray(y_m, dtype=np.float64)
        if y.ndim != 1 or not np.all(np.isfinite(y)):
            raise ValueError("y_m must be a finite 1-D axial coordinate array")
        return np.zeros_like(y)


@dataclass(frozen=True)
class ParabolicAxialProfile:
    """Parabolic axial relief ``g(y)=y**2/(2*radius_m)``."""

    radius_m: float

    def __post_init__(self):
        object.__setattr__(self, "radius_m", _positive_finite(self.radius_m, "radius_m"))

    def __call__(self, y_m: NDArray[np.float64]) -> NDArray[np.float64]:
        y = np.asarray(y_m, dtype=np.float64)
        if y.ndim != 1 or not np.all(np.isfinite(y)):
            raise ValueError("y_m must be a finite 1-D axial coordinate array")
        return y**2 / (2.0 * self.radius_m)


@dataclass(frozen=True)
class CircularAxialProfile:
    """Circular-arc axial relief ``R-sqrt(R**2-y**2)``."""

    radius_m: float

    def __post_init__(self):
        object.__setattr__(self, "radius_m", _positive_finite(self.radius_m, "radius_m"))

    def __call__(self, y_m: NDArray[np.float64]) -> NDArray[np.float64]:
        y = np.asarray(y_m, dtype=np.float64)
        if y.ndim != 1 or not np.all(np.isfinite(y)):
            raise ValueError("y_m must be a finite 1-D axial coordinate array")
        ratio = np.abs(y) / self.radius_m
        if np.any(ratio > 1.0):
            raise ValueError("axial coordinates must lie within the circular crown radius")
        return np.abs(y) * ratio / (1.0 + np.sqrt((1.0 - ratio) * (1.0 + ratio)))


@dataclass(frozen=True)
class LogarithmicAxialProfile:
    """Capped generic logarithmic relief on ``[-half_length_m, half_length_m]``."""

    half_length_m: float
    scale_m: float
    maximum_relief_m: float

    def __post_init__(self):
        for name in ("half_length_m", "scale_m", "maximum_relief_m"):
            object.__setattr__(self, name, _positive_finite(getattr(self, name), name))

    def __call__(self, y_m: NDArray[np.float64]) -> NDArray[np.float64]:
        y = np.asarray(y_m, dtype=np.float64)
        if y.ndim != 1 or not np.all(np.isfinite(y)):
            raise ValueError("y_m must be a finite 1-D axial coordinate array")
        absolute = np.abs(y)
        if np.any(absolute > self.half_length_m):
            raise ValueError("axial coordinates lie outside the logarithmic profile length")
        interior = absolute < self.half_length_m
        relief = np.full_like(y, self.maximum_relief_m)
        ratio = absolute[interior] / self.half_length_m
        log_distance = np.empty_like(ratio)
        near_center = ratio <= 0.5
        log_distance[near_center] = np.log1p(-ratio[near_center] ** 2)
        log_distance[~near_center] = np.log(
            (self.half_length_m - absolute[interior][~near_center]) / self.half_length_m
        ) + np.log1p(ratio[~near_center])
        relief[interior] = np.minimum(-self.scale_m * log_distance, self.maximum_relief_m)
        return relief


@dataclass(frozen=True)
class SmoothLogarithmicAxialProfile:
    """Logarithmic centre smoothly blended to finite relief at each end."""

    half_length_m: float
    scale_m: float
    maximum_relief_m: float
    blend_start_fraction: float

    def __post_init__(self):
        for name in ("half_length_m", "scale_m", "maximum_relief_m"):
            object.__setattr__(self, name, _positive_finite(getattr(self, name), name))
        start = self.blend_start_fraction
        if isinstance(start, bool) or not isfinite(start) or not 0.0 < start < 1.0:
            raise ValueError("blend_start_fraction must be strictly between zero and one")
        object.__setattr__(self, "blend_start_fraction", float(start))
        height = -self.scale_m * np.log1p(-(start**2))
        slope = 2.0 * self.scale_m * start / (1.0 - start**2)
        if height >= self.maximum_relief_m or (
            (1.0 - start) * slope > 3.0 * (self.maximum_relief_m - height)
        ):
            raise ValueError("blend parameters do not yield monotone bounded relief")

    def __call__(self, y_m: NDArray[np.float64]) -> NDArray[np.float64]:
        y = np.asarray(y_m, dtype=np.float64)
        if y.ndim != 1 or not np.all(np.isfinite(y)):
            raise ValueError("y_m must be a finite 1-D axial coordinate array")
        s = np.abs(y) / self.half_length_m
        if np.any(s > 1.0):
            raise ValueError("axial coordinates lie outside the logarithmic profile length")
        start = self.blend_start_fraction
        inner = s <= start
        relief = np.empty_like(s)
        relief[inner] = -self.scale_m * np.log1p(-(s[inner] ** 2))
        t = (s[~inner] - start) / (1.0 - start)
        height = -self.scale_m * np.log1p(-(start**2))
        slope = 2.0 * self.scale_m * start / (1.0 - start**2)
        relief[~inner] = (
            (2 * t**3 - 3 * t**2 + 1) * height
            + (t**3 - 2 * t**2 + t) * (1.0 - start) * slope
            + (-2 * t**3 + 3 * t**2) * self.maximum_relief_m
        )
        return relief


@dataclass(frozen=True)
class TabulatedAxialProfile:
    """Linearly interpolate measured or user-supplied ``(y,g(y))`` data.

    Evaluation outside the tabulated axial range is rejected; no implicit
    extrapolation or edge clamping is applied.
    """

    y_m: tuple[float, ...]
    height_m: tuple[float, ...]

    def __post_init__(self):
        y = np.asarray(self.y_m, dtype=np.float64)
        height = np.asarray(self.height_m, dtype=np.float64)
        if (
            y.ndim != 1
            or height.shape != y.shape
            or y.size < 2
            or not np.all(np.isfinite(y))
            or not np.all(np.isfinite(height))
            or np.any(np.diff(y) <= 0.0)
        ):
            raise ValueError(
                "tabulated y_m and height_m must be matching finite ascending "
                "1-D arrays with at least two points"
            )
        object.__setattr__(self, "y_m", tuple(float(value) for value in y))
        object.__setattr__(self, "height_m", tuple(float(value) for value in height))

    def __call__(self, y_m: NDArray[np.float64]) -> NDArray[np.float64]:
        y = np.asarray(y_m, dtype=np.float64)
        if y.ndim != 1 or not np.all(np.isfinite(y)):
            raise ValueError("y_m must be a finite 1-D axial coordinate array")
        if np.any(y < self.y_m[0]) or np.any(y > self.y_m[-1]):
            raise ValueError("requested axial coordinates lie outside the tabulated profile")
        return np.interp(y, self.y_m, self.height_m)


@dataclass(frozen=True)
class ProfiledRollerGeometry:
    """Parabolic rolling-direction geometry plus an arbitrary axial profile ``g(y)``.

    The profile receives axial coordinates in metres and must return finite
    heights in metres, either scalar, shape ``(ny,)``, or an ``(ny,nx)`` field.
    Crown shape and axial tilt remain separate components.
    """

    radius_x_m: float
    axial_profile_m: AxialProfile | Callable[[NDArray[np.float64]], ArrayLike]

    def __post_init__(self):
        object.__setattr__(self, "radius_x_m", _positive_finite(self.radius_x_m, "radius_x_m"))
        if not callable(self.axial_profile_m):
            raise ValueError("axial_profile_m must be callable")

    def h_geom(
        self, x_m: ArrayLike, y_m: ArrayLike, *, time_s: float | None = None
    ) -> NDArray[np.float64]:
        _validate_time(time_s)
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


@dataclass(frozen=True)
class AxialTiltGeometry:
    """Additive axial tilt ``h_tilt(y)=y*tan(angle_rad)`` in metres."""

    angle_rad: float

    def __post_init__(self):
        if isinstance(self.angle_rad, bool) or not isfinite(self.angle_rad):
            raise ValueError("angle_rad must be finite radians")
        object.__setattr__(self, "angle_rad", float(self.angle_rad))

    def h_geom(
        self, x_m: ArrayLike, y_m: ArrayLike, *, time_s: float | None = None
    ) -> NDArray[np.float64]:
        _validate_time(time_s)
        _, yy = _coordinates(x_m, y_m)
        return yy * tan(self.angle_rad)


@dataclass(frozen=True)
class CompositeContactGeometry:
    """Additive geometry/surface components evaluated at one optional time."""

    components: tuple[ContactGeometry, ...]

    def __post_init__(self):
        components = tuple(self.components)
        if not components or any(
            not callable(getattr(item, "h_geom", None)) for item in components
        ):
            raise ValueError("components must contain ContactGeometry instances")
        object.__setattr__(self, "components", components)

    def h_geom(
        self, x_m: ArrayLike, y_m: ArrayLike, *, time_s: float | None = None
    ) -> NDArray[np.float64]:
        _validate_time(time_s)
        x, y = np.asarray(x_m, dtype=np.float64), np.asarray(y_m, dtype=np.float64)
        shape = _coordinates(x, y)[0].shape
        total = np.zeros(shape, dtype=np.float64)
        for component in self.components:
            height = np.asarray(component.h_geom(x, y, time_s=time_s), dtype=np.float64)
            if height.shape != shape or not np.all(np.isfinite(height)):
                raise ValueError(
                    f"{type(component).__name__}.h_geom must return a finite field "
                    f"with shape {shape}"
                )
            total += height
        return total


def contact_geometry_from_config(config: Mapping[str, object]) -> ContactGeometry:
    """Build shared contact geometry from a case-input mapping.

    Supported ``type`` values are ``circular_point``, ``elliptical_point``,
    ``classical_line`` and ``profiled_roller``. A profile mapping can select
    ``cylindrical``, ``parabolic``, ``circular``, ``logarithmic``,
    ``smooth_logarithmic`` or
    ``tabulated``; Python callers may also provide a callable ``AxialProfile``.
    ``axial_tilt_rad`` is an independent additive component, never part of
    ``g(y)``.
    """
    if not isinstance(config, Mapping):
        raise ValueError("contact geometry configuration must be a mapping")
    values = dict(config)
    geometry_type = values.pop("type", None)
    tilt = values.pop("axial_tilt_rad", 0.0)
    if isinstance(tilt, bool) or not isinstance(tilt, (int, float)) or not isfinite(tilt):
        raise ValueError("axial_tilt_rad must be finite radians")

    if geometry_type == "circular_point":
        _reject_unknown_fields(values, {"radius_m"})
        geometry: ContactGeometry = CircularPointContactGeometry(
            _positive_config_value(values, "radius_m")
        )
    elif geometry_type == "elliptical_point":
        _reject_unknown_fields(values, {"radius_x_m", "radius_y_m"})
        geometry = EllipticalPointContactGeometry(
            _positive_config_value(values, "radius_x_m"),
            _positive_config_value(values, "radius_y_m"),
        )
    elif geometry_type == "classical_line":
        _reject_unknown_fields(values, {"radius_x_m"})
        geometry = LineContactGeometry(_positive_config_value(values, "radius_x_m"))
    elif geometry_type == "profiled_roller":
        _reject_unknown_fields(values, {"radius_x_m", "axial_profile"})
        profile_config = values.pop("axial_profile", None)
        geometry = ProfiledRollerGeometry(
            _positive_config_value(values, "radius_x_m"),
            _axial_profile_from_config(profile_config),
        )
    else:
        raise ValueError(f"unsupported contact geometry type: {geometry_type!r}")

    if tilt != 0.0:
        if geometry_type != "profiled_roller":
            raise ValueError("axial_tilt_rad is currently defined for profiled_roller geometry")
        return CompositeContactGeometry((geometry, AxialTiltGeometry(float(tilt))))
    return geometry


def _required_config_value(config: dict[str, object], name: str):
    try:
        return config.pop(name)
    except KeyError as error:
        raise ValueError(f"contact geometry requires {name}") from error


def _positive_config_value(config: dict[str, object], name: str) -> float:
    value = _required_config_value(config, name)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(f"{name} must be a positive finite length in metres")
    if value <= 0.0:
        raise ValueError(f"{name} must be a positive finite length in metres")
    return float(value)


def _reject_unknown_fields(config: dict[str, object], allowed: set[str]) -> None:
    if any(not isinstance(key, str) for key in config):
        raise ValueError("contact geometry configuration keys must be strings")
    unknown = config.keys() - allowed
    if unknown:
        raise ValueError(f"unsupported contact geometry fields: {sorted(unknown)}")


def _axial_profile_from_config(config: object) -> AxialProfile | Callable:
    if config is None:
        return ZeroAxialProfile()
    if callable(config):
        return config
    if not isinstance(config, Mapping):
        raise ValueError("axial_profile must be a profile mapping or callable")
    values = dict(config)
    profile_type = values.pop("type", None)
    if profile_type == "cylindrical":
        _reject_unknown_fields(values, set())
        return ZeroAxialProfile()
    if profile_type in {"parabolic", "circular"}:
        _reject_unknown_fields(values, {"radius_m"})
        radius = _positive_config_value(values, "radius_m")
        if profile_type == "parabolic":
            return ParabolicAxialProfile(radius)
        return CircularAxialProfile(radius)
    if profile_type == "logarithmic":
        _reject_unknown_fields(values, {"half_length_m", "scale_m", "maximum_relief_m"})
        return LogarithmicAxialProfile(
            _positive_config_value(values, "half_length_m"),
            _positive_config_value(values, "scale_m"),
            _positive_config_value(values, "maximum_relief_m"),
        )
    if profile_type == "smooth_logarithmic":
        _reject_unknown_fields(
            values, {"half_length_m", "scale_m", "maximum_relief_m", "blend_start_fraction"}
        )
        return SmoothLogarithmicAxialProfile(
            _positive_config_value(values, "half_length_m"),
            _positive_config_value(values, "scale_m"),
            _positive_config_value(values, "maximum_relief_m"),
            _positive_config_value(values, "blend_start_fraction"),
        )
    if profile_type == "tabulated":
        _reject_unknown_fields(values, {"y_m", "height_m"})
        y_m = _required_config_value(values, "y_m")
        height_m = _required_config_value(values, "height_m")
        try:
            return TabulatedAxialProfile(tuple(y_m), tuple(height_m))
        except TypeError as error:
            raise ValueError(
                "tabulated profile coordinates and heights must be sequences"
            ) from error
    raise ValueError(f"unsupported axial profile type: {profile_type!r}")
