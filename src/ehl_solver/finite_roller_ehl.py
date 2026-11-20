"""Stationary isothermal finite-roller JFO case adapter and audit utilities.

Public inputs use SI units; fields are ``(ny,nx)``, x follows entrainment,
and load is total force in N. The elastic half-space operator and donor/JFO
equations are the existing point-contact production formulation.
"""

from dataclasses import asdict, dataclass
from math import isfinite
from types import SimpleNamespace

import numpy as np
from scipy.integrate import trapezoid
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import find_peaks, peak_widths

from ehl_solver.backend import as_float64, make_backend
from ehl_solver.cavitating_coupled import CavitatingElasticityProblem
from ehl_solver.constitutive import IsothermalFluid, roelands_index_from_ambient_slope
from ehl_solver.device_jfo import DeviceDonorJFO
from ehl_solver.donor_load import DonorLoadSystem
from ehl_solver.film_correlations import FilmCorrelation, hamrock_dowson_point_contact
from ehl_solver.geometry import (
    AxialTiltGeometry,
    BlendedParabolicAxialProfile,
    CompositeContactGeometry,
    ProfiledRollerGeometry,
)
from ehl_solver.hertz import EllipticalHertzResult, elliptical_point_contact
from ehl_solver.limited_audit import audit_limited_state
from ehl_solver.limited_system import LimitedLoadProbe
from ehl_solver.transport_candidate import donor_fluxes


@dataclass(frozen=True)
class FiniteRollerEHLCase:
    """Finite profiled roller using the established donor/JFO load system."""

    load_n: float = 100.0
    radius_x_m: float = 0.01
    roller_length_m: float = 0.001
    end_relief_m: float = 2e-6
    end_zone_fraction: float = 0.1
    axial_tilt_rad: float = 0.0
    contact_modulus_pa: float = 113e9
    viscosity_pa_s: float = 0.04
    density_kg_m3: float = 850.0
    entrainment_m_s: float = 2.0
    alpha_per_pa: float = 2.21e-8
    roelands_pressure_pa: float = 1.96e8
    viscosity_reference_pa_s: float = 6.31e-5
    nx: int = 65
    ny: int = 129
    domain_half_width_x_m: float | None = None
    maximum_physical_nodes: int = 20000

    transport: str = "donor"

    def __post_init__(self):
        for name in (
            "load_n",
            "radius_x_m",
            "roller_length_m",
            "end_relief_m",
            "contact_modulus_pa",
            "viscosity_pa_s",
            "density_kg_m3",
            "entrainment_m_s",
            "alpha_per_pa",
            "roelands_pressure_pa",
            "viscosity_reference_pa_s",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive finite SI")
            object.__setattr__(self, name, float(value))
        if isinstance(self.axial_tilt_rad, bool) or not isfinite(self.axial_tilt_rad):
            raise ValueError("axial_tilt_rad must be finite radians")
        if (
            isinstance(self.end_zone_fraction, bool)
            or not isfinite(self.end_zone_fraction)
            or not 0 < self.end_zone_fraction <= 0.5
        ):
            raise ValueError("end_zone_fraction must lie in (0, 0.5]")
        if any(type(n) is not int or n < 5 or n % 2 == 0 for n in (self.nx, self.ny)):
            raise ValueError("Odd rectangular grids >=5 are required")
        supported_budgets = (20000, 30000, 40000, 50000, 70000, 100000, 300000)
        if type(self.maximum_physical_nodes) is not int or (
            self.maximum_physical_nodes not in supported_budgets
        ):
            raise ValueError(f"maximum_physical_nodes must be one of {supported_budgets}")
        if self.nx * self.ny > self.maximum_physical_nodes:
            raise ValueError("finite-roller grid exceeds its explicit physical-node budget")
        half_length = self.roller_length_m / 2.0
        if self.end_relief_m >= half_length:
            raise ValueError("end_relief_m must be smaller than the roller half-length")
        radius_y = half_length**2 / (2.0 * self.end_relief_m)
        hertz = elliptical_point_contact(
            self.load_n, self.radius_x_m, radius_y, self.contact_modulus_pa
        )
        domain = self.domain_half_width_x_m
        if domain is None:
            domain = 3.0 * hertz.semi_axis_x_m
        if (
            isinstance(domain, bool)
            or not isfinite(domain)
            or domain < 2.0 * hertz.semi_axis_x_m
        ):
            raise ValueError("domain_half_width_x_m must be at least two Hertz semi-axes")
        object.__setattr__(self, "domain_half_width_x_m", float(domain))
        if self.viscosity_pa_s <= self.viscosity_reference_pa_s:
            raise ValueError("Ambient viscosity must exceed its Roelands reference")
        self.fluid()
        self.correlation()

    @property
    def half_length_m(self) -> float:
        return self.roller_length_m / 2.0

    @property
    def radius_y_m(self) -> float:
        return self.half_length_m**2 / (2.0 * self.end_relief_m)

    @property
    def hertz(self) -> EllipticalHertzResult:
        return elliptical_point_contact(
            self.load_n, self.radius_x_m, self.radius_y_m, self.contact_modulus_pa
        )

    def fluid(self) -> IsothermalFluid:
        index = roelands_index_from_ambient_slope(
            self.alpha_per_pa,
            self.viscosity_pa_s,
            self.viscosity_reference_pa_s,
            self.roelands_pressure_pa,
        )
        return IsothermalFluid(
            self.viscosity_pa_s,
            self.density_kg_m3,
            viscosity_index=index,
            viscosity_reference_pa_s=self.viscosity_reference_pa_s,
            roelands_pressure_pa=self.roelands_pressure_pa,
        )

    def correlation(self) -> FilmCorrelation:
        return hamrock_dowson_point_contact(
            load_n=self.load_n,
            radius_x_m=self.radius_x_m,
            contact_modulus_pa=self.contact_modulus_pa,
            viscosity_pa_s=self.viscosity_pa_s,
            entrainment_m_s=self.entrainment_m_s,
            alpha_per_pa=self.alpha_per_pa,
            ellipticity=self.hertz.axis_ratio_y_over_x,
        )

    def geometry(self):
        profile = BlendedParabolicAxialProfile(
            self.half_length_m,
            self.end_relief_m,
            self.end_zone_fraction,
        )
        return CompositeContactGeometry(
            (
                ProfiledRollerGeometry(self.radius_x_m, profile),
                AxialTiltGeometry(self.axial_tilt_rad),
            )
        )

    def problem(self) -> CavitatingElasticityProblem:
        hertz = self.hertz
        correlation = self.correlation()
        length_scale = hertz.semi_axis_x_m
        pressure_scale = hertz.peak_pressure_pa
        film_scale = correlation.central_film_m
        reynolds_parameter = (
            12.0
            * self.viscosity_pa_s
            * self.entrainment_m_s
            * length_scale
            / (film_scale**2 * pressure_scale)
        )
        return CavitatingElasticityProblem(
            np.linspace(
                -self.domain_half_width_x_m,
                self.domain_half_width_x_m,
                self.nx,
            ),
            np.linspace(-self.half_length_m, self.half_length_m, self.ny),
            film_scale,
            self.radius_x_m,
            self.radius_y_m,
            self.contact_modulus_pa,
            length_scale,
            film_scale,
            pressure_scale,
            reynolds_parameter,
            geometry=self.geometry(),
        )

    def system(self, *, workspace_limit_bytes=2**30) -> DonorLoadSystem:
        return DonorLoadSystem(
            self.problem(),
            self.fluid(),
            self.entrainment_m_s,
            self.load_n,
            preconditioner="local",
            elasticity_mode="cached",
            maximum_physical_nodes=self.maximum_physical_nodes,
            workspace_limit_bytes=workspace_limit_bytes,
        )

    def physical_signature(self):
        return {
            key: value
            for key, value in asdict(self).items()
            if key
            not in (
                "nx",
                "ny",
                "domain_half_width_x_m",
                "maximum_physical_nodes",
            )
        }


def finite_roller_hertz_seed(case: FiniteRollerEHLCase, system=None) -> np.ndarray:
    """Return a strictly feasible, force-normalized seed; never an accepted root."""
    system = case.system() if system is None else system
    problem = system.problem
    xx, yy = np.meshgrid(problem.x_m, problem.y_m)
    hertz = case.hertz
    radial = (xx / hertz.semi_axis_x_m) ** 2 + (yy / hertz.semi_axis_y_m) ** 2
    pressure = hertz.peak_pressure_pa * np.sqrt(np.maximum(1.0 - radial, 0.0))
    pressure[[0, -1], :] = 0.0
    pressure[:, [0, -1]] = 0.0
    initial_load = problem.load_n(pressure)
    if not isfinite(initial_load) or initial_load <= 0:
        raise ValueError("Hertz seed has no finite positive integrated load")
    pressure *= case.load_n / initial_load
    deformation = system.base.elastic_deformation(pressure)
    offset = case.correlation().central_film_m - float(
        np.min(problem.rigid_geometry_m + deformation)
    )
    seed = SimpleNamespace(
        pressure_pa=pressure,
        fractional_film_content=np.ones(problem.shape, dtype=np.float64),
    )
    state = system.pack(seed, offset)
    probe = LimitedLoadProbe(case, transport_fraction=0.0)
    if not probe.feasible(state):
        raise ValueError("finite-roller Hertz seed does not satisfy strict film feasibility")
    return state


def transfer_finite_roller_seed(
    source_case: FiniteRollerEHLCase,
    source_state: np.ndarray,
    target_case: FiniteRollerEHLCase,
    target_system=None,
) -> np.ndarray:
    """Transfer an unaccepted/accepted same-physics state to tilt or a finer grid."""
    if _transfer_signature(source_case) != _transfer_signature(target_case):
        raise ValueError("finite-roller transfer requires identical physical parameters")
    source_probe = LimitedLoadProbe(source_case, transport_fraction=0.0)
    pressure, theta, _film = source_probe.fields(source_state)
    source_problem = source_probe.system.problem
    target_system = target_case.system() if target_system is None else target_system
    target_problem = target_system.problem
    if (
        target_problem.x_m[0] < source_problem.x_m[0]
        or target_problem.x_m[-1] > source_problem.x_m[-1]
        or target_problem.y_m[0] < source_problem.y_m[0]
        or target_problem.y_m[-1] > source_problem.y_m[-1]
    ):
        raise ValueError("finite-roller seed transfer does not extrapolate outside source axes")
    yy, xx = np.meshgrid(target_problem.y_m, target_problem.x_m, indexing="ij")
    points = np.column_stack((yy.ravel(), xx.ravel()))
    if pressure.shape == target_problem.shape and (
        np.array_equal(source_problem.x_m, target_problem.x_m)
        and np.array_equal(source_problem.y_m, target_problem.y_m)
    ):
        target_pressure = pressure.copy()
        target_theta = theta.copy()
    else:
        target_pressure = RegularGridInterpolator(
            (source_problem.y_m, source_problem.x_m),
            pressure,
            bounds_error=True,
        )(points).reshape(target_problem.shape)
        target_theta = RegularGridInterpolator(
            (source_problem.y_m, source_problem.x_m),
            theta,
            bounds_error=True,
        )(points).reshape(target_problem.shape)
    target_pressure[[0, -1], :] = 0.0
    target_pressure[:, [0, -1]] = 0.0
    transfer_load = target_problem.load_n(target_pressure)
    if not isfinite(transfer_load) or transfer_load <= 0:
        raise ValueError("transferred finite-roller seed has no positive load")
    target_pressure *= target_case.load_n / transfer_load
    offset = float(source_state[-1] * source_problem.film_scale_m)
    tilt_change = abs(np.tan(target_case.axial_tilt_rad) - np.tan(source_case.axial_tilt_rad))
    offset += target_case.half_length_m * tilt_change
    seed = SimpleNamespace(
        pressure_pa=target_pressure,
        fractional_film_content=target_theta,
    )
    state = target_system.pack(seed, offset)
    target_probe = LimitedLoadProbe(target_case, transport_fraction=0.0)
    if not target_probe.feasible(state):
        raise ValueError("transferred finite-roller seed is not strictly film feasible")
    return state


def _transfer_signature(case):
    signature = case.physical_signature()
    signature.pop("axial_tilt_rad")
    return signature


def mirror_finite_roller_seed(source_case, source_state, target_case, target_system=None):
    """Reflect a same-grid state in y for the opposite prescribed tilt."""
    if _transfer_signature(source_case) != _transfer_signature(target_case):
        raise ValueError("mirror seed requires identical physical parameters")
    if source_case.nx != target_case.nx or source_case.ny != target_case.ny:
        raise ValueError("mirror seed requires matching rectangular grids")
    if not np.isclose(
        source_case.axial_tilt_rad, -target_case.axial_tilt_rad, rtol=0.0, atol=1e-14
    ):
        raise ValueError("mirror seed requires equal and opposite prescribed tilts")
    source_probe = LimitedLoadProbe(source_case, transport_fraction=0.0)
    pressure, theta, _film = source_probe.fields(source_state)
    source_problem = source_probe.system.problem
    target_system = target_case.system() if target_system is None else target_system
    target_problem = target_system.problem
    if not (
        np.array_equal(source_problem.x_m, target_problem.x_m)
        and np.array_equal(source_problem.y_m, target_problem.y_m)
    ):
        raise ValueError("mirror seed requires identical coordinate axes")
    pressure = pressure[::-1].copy()
    theta = theta[::-1].copy()
    pressure[[0, -1], :] = 0.0
    pressure[:, [0, -1]] = 0.0
    load = target_problem.load_n(pressure)
    if not isfinite(load) or load <= 0:
        raise ValueError("mirrored finite-roller seed has no positive load")
    pressure *= target_case.load_n / load
    seed = SimpleNamespace(pressure_pa=pressure, fractional_film_content=theta)
    state = target_system.pack(seed, float(source_state[-1] * source_problem.film_scale_m))
    target_probe = LimitedLoadProbe(target_case, transport_fraction=0.0)
    if not target_probe.feasible(state):
        raise ValueError("mirrored finite-roller seed is not strictly film feasible")
    return state


def _integral(field, x_m, y_m) -> float:
    return float(trapezoid(trapezoid(field, x=x_m, axis=1), x=y_m))


def _axial_window_load(pressure, x_m, y_m, lower, upper) -> float:
    interior = (y_m > lower) & (y_m < upper)
    middle_y = y_m[interior]
    middle = pressure[interior]
    lower_pressure = np.array([np.interp(lower, y_m, pressure[:, i]) for i in range(len(x_m))])
    upper_pressure = np.array([np.interp(upper, y_m, pressure[:, i]) for i in range(len(x_m))])
    values = np.vstack((lower_pressure, middle, upper_pressure))
    coordinates = np.r_[lower, middle_y, upper]
    return float(trapezoid(trapezoid(values, x=x_m, axis=1), x=coordinates))


def finite_roller_metrics(case, probe, state, audit=None) -> tuple[dict, dict]:
    """Compute independently useful load, boundary, film, and flux diagnostics."""
    problem = probe.system.problem
    pressure, theta, film = probe.fields(state)
    audit = audit_limited_state(probe, state, direct=True) if audit is None else audit
    x_m, y_m = problem.x_m, problem.y_m
    peak_y, peak_x = np.unravel_index(np.argmax(pressure), pressure.shape)
    min_y, min_x = np.unravel_index(np.argmin(film), film.shape)
    center_y, center_x = len(y_m) // 2, len(x_m) // 2
    force = problem.load_n(pressure)
    pressure_x_moment = _integral(pressure * x_m[None, :], x_m, y_m)
    pressure_y_moment = _integral(pressure * y_m[:, None], x_m, y_m)
    pressure_film = pressure * film
    pressure_film_load = _integral(pressure_film, x_m, y_m)
    outer_loads = {}
    for fraction in (0.02, 0.05, 0.10):
        width = fraction * case.roller_length_m
        outer_loads[str(int(100 * fraction))] = [
            _axial_window_load(
                pressure, x_m, y_m, -case.half_length_m, -case.half_length_m + width
            ),
            _axial_window_load(
                pressure, x_m, y_m, case.half_length_m - width, case.half_length_m
            ),
        ]
    outermost_row_loads = [
        float(trapezoid(pressure[0], x=x_m) * problem.dy_m / 2.0),
        float(trapezoid(pressure[-1], x=x_m) * problem.dy_m / 2.0),
    ]
    pressure_threshold = max(1e-8 * float(pressure.max()), 1e3)
    pressurized = pressure > pressure_threshold
    active_rows = np.flatnonzero(np.any(pressurized, axis=1))
    active_columns = np.flatnonzero(np.any(pressurized, axis=0))
    if active_rows.size and active_columns.size:
        axial_clearance = min(
            y_m[active_rows[0]] + case.half_length_m,
            case.half_length_m - y_m[active_rows[-1]],
        )
        rolling_clearance = min(
            x_m[active_columns[0]] + case.domain_half_width_x_m,
            case.domain_half_width_x_m - x_m[active_columns[-1]],
        )
    else:
        axial_clearance = float("nan")
        rolling_clearance = float("nan")
    end_rows = max(1, int(np.ceil(0.1 * (len(y_m) - 1))))
    left_end = slice(0, end_rows + 1)
    right_end = slice(len(y_m) - end_rows - 1, len(y_m))
    h_centerline = film[center_y]
    outlet = np.flatnonzero(x_m > 0)
    outlet_minimum_index = int(outlet[np.argmin(h_centerline[outlet])])
    outlet_minima, _ = find_peaks(-h_centerline[outlet])
    outlet_width = None
    if outlet_minima.size:
        widths = peak_widths(-h_centerline[outlet], outlet_minima, rel_height=0.5)[0]
        selected = int(np.argmin(h_centerline[outlet][outlet_minima]))
        outlet_width = float(widths[selected])
        outlet_minimum_index = int(outlet[outlet_minima[selected]])
    centerline_pressure = pressure[center_y]
    downstream = np.flatnonzero(x_m > 0)
    outlet_pressure_peaks, outlet_pressure_properties = find_peaks(
        centerline_pressure[downstream],
        prominence=0.005 * float(pressure.max()),
    )
    outlet_pressure_spike = None
    if outlet_pressure_peaks.size:
        selected = int(np.argmax(centerline_pressure[downstream][outlet_pressure_peaks]))
        peak_index = int(downstream[outlet_pressure_peaks[selected]])
        width_cells = float(
            peak_widths(
                centerline_pressure[downstream],
                outlet_pressure_peaks,
                rel_height=0.5,
            )[0][selected]
        )
        outlet_pressure_spike = {
            "x_m": float(x_m[peak_index]),
            "pressure_pa": float(centerline_pressure[peak_index]),
            "prominence_pa": float(outlet_pressure_properties["prominences"][selected]),
            "half_prominence_width_cells": width_cells,
            "half_prominence_width_m": width_cells * problem.dx_m,
        }
    pressure_axial_profile = pressure[:, peak_x]
    axial_peak_width_cells = int(
        np.count_nonzero(pressure_axial_profile >= 0.5 * pressure[peak_y, peak_x])
    )
    end_pressure_peaks = {}
    end_pressure_support = {}
    for side, mask in (
        ("left", y_m <= -case.half_length_m + 0.1 * case.roller_length_m),
        ("right", y_m >= case.half_length_m - 0.1 * case.roller_length_m),
    ):
        end_field = pressure[mask]
        if end_field.size:
            local_y, local_x = np.unravel_index(np.argmax(end_field), end_field.shape)
            end_y = int(np.flatnonzero(mask)[local_y])
            end_x = int(local_x)
            peak = float(pressure[end_y, end_x])
            end_pressure_peaks[side] = {
                "peak_pa": peak,
                "coordinates_m": [float(x_m[end_x]), float(y_m[end_y])],
            }
        else:
            end_pressure_peaks[side] = None
        by_threshold = {}
        for fraction in (0.01, 0.05, 0.10):
            rows = np.flatnonzero(np.any(pressure > fraction * pressure.max(), axis=1))
            if rows.size:
                by_threshold[f"{fraction:g}_pmax"] = {
                    "threshold_pa": float(fraction * pressure.max()),
                    "left_clearance_m": float(y_m[rows[0]] + case.half_length_m),
                    "right_clearance_m": float(case.half_length_m - y_m[rows[-1]]),
                    "left_clearance_cells": float(
                        (y_m[rows[0]] + case.half_length_m) / problem.dy_m
                    ),
                    "right_clearance_cells": float(
                        (case.half_length_m - y_m[rows[-1]]) / problem.dy_m
                    ),
                }
            else:
                by_threshold[f"{fraction:g}_pmax"] = None
        end_pressure_support[side] = by_threshold
    boundary_adjacent_rows = [1, len(y_m) - 2]
    cavitated = (pressure <= 1e-8 * problem.pressure_scale_pa) & (theta < 1 - 1e-8)
    cavitated[0], cavitated[-1], cavitated[:, 0], cavitated[:, -1] = False, False, False, False
    cavitation_indices = np.argwhere(cavitated)
    mass_outward = audit["mass"]["outward_mass_flow_kg_s"]
    axial_net = float(mass_outward[2] + mass_outward[3])
    axial_scale = max(abs(mass_outward[2]) + abs(mass_outward[3]), 1e-300)
    qx, qy = donor_fluxes(
        x_m, y_m, film, pressure, theta, case.fluid(), case.entrainment_m_s
    )
    metrics = {
        "nx": case.nx,
        "ny": case.ny,
        "load_target_n": case.load_n,
        "load_integrated_n": force,
        "load_relative_error": force / case.load_n - 1.0,
        "h_c_m": float(film[center_y, center_x]),
        "h_min_m": float(film[min_y, min_x]),
        "h_min_coordinates_m": [float(x_m[min_x]), float(y_m[min_y])],
        "h_min_boundary_clearance_m": float(case.half_length_m - abs(y_m[min_y])),
        "centerline_h_min_m": float(h_centerline.min()),
        "end_10_percent_h_min_m": [
            float(film[left_end].min()),
            float(film[right_end].min()),
        ],
        "axial_end_row_h_min_m": [float(film[0].min()), float(film[-1].min())],
        "axial_end_row_theta_range": [
            [float(theta[0].min()), float(theta[0].max())],
            [float(theta[-1].min()), float(theta[-1].max())],
        ],
        "p_max_pa": float(pressure[peak_y, peak_x]),
        "p_max_coordinates_m": [float(x_m[peak_x]), float(y_m[peak_y])],
        "p_max_axial_half_height_width_cells": axial_peak_width_cells,
        "pressure_centroid_m": [
            float(pressure_x_moment / force),
            float(pressure_y_moment / force),
        ],
        "pressure_weighted_film_centroid_m": [
            _integral(pressure_film * x_m[None, :], x_m, y_m) / pressure_film_load,
            _integral(pressure_film * y_m[:, None], x_m, y_m) / pressure_film_load,
        ],
        "moment_about_x_n_m": pressure_y_moment,
        "moment_about_y_n_m": pressure_x_moment,
        "pressurized_support_threshold_pa": pressure_threshold,
        "pressurized_support_axial_clearance_m": float(axial_clearance),
        "pressurized_support_axial_clearance_cells": float(axial_clearance / problem.dy_m),
        "pressurized_support_rolling_clearance_m": float(rolling_clearance),
        "pressurized_support_clearance_by_peak_fraction": end_pressure_support,
        "outer_band_load_n_left_right": outer_loads,
        "outermost_axial_row_load_n": outermost_row_loads,
        "outermost_axial_row_load_fraction": [
            load / case.load_n for load in outermost_row_loads
        ],
        "outermost_axial_row_max_pressure_pa": [
            float(pressure[0].max()),
            float(pressure[-1].max()),
        ],
        "boundary_adjacent_row_max_pressure_pa": [
            float(pressure[boundary_adjacent_rows[0]].max()),
            float(pressure[boundary_adjacent_rows[1]].max()),
        ],
        "boundary_adjacent_row_load_n": [
            float(
                trapezoid(pressure[boundary_adjacent_rows[0]], x=x_m)
                * problem.dy_m
            ),
            float(
                trapezoid(pressure[boundary_adjacent_rows[1]], x=x_m)
                * problem.dy_m
            ),
        ],
        "outer_10_percent_pressure_peak": end_pressure_peaks,
        "outlet_centerline_pressure_spike": outlet_pressure_spike,
        "outlet_centerline_constriction": {
            "x_m": float(x_m[outlet_minimum_index]),
            "film_m": float(h_centerline[outlet_minimum_index]),
            "half_prominence_width_cells": outlet_width,
            "half_prominence_width_m": (
                None if outlet_width is None else outlet_width * problem.dx_m
            ),
        },
        "axial_peak_width_m": float(max(0, axial_peak_width_cells - 1) * problem.dy_m),
        "cavitated_interior_cell_count": int(len(cavitation_indices)),
        "cavitated_interior_axial_extent_m": (
            [float(y_m[cavitation_indices[:, 0].min()]), float(y_m[cavitation_indices[:, 0].max()])]
            if len(cavitation_indices)
            else None
        ),
        "cavitated_interior_rolling_extent_m": (
            [
                float(x_m[cavitation_indices[:, 1].min()]),
                float(x_m[cavitation_indices[:, 1].max()]),
            ]
            if len(cavitation_indices)
            else None
        ),
        "theta_min_max": [float(theta.min()), float(theta.max())],
        "boundary_pressure_max_pa": float(
            max(pressure[0].max(), pressure[-1].max(), pressure[:, 0].max(), pressure[:, -1].max())
        ),
        "axial_boundary_net_mass_flow_kg_s": axial_net,
        "relative_axial_boundary_flow_imbalance": abs(axial_net) / axial_scale,
        "audit": audit,
    }
    profiles = {
        "pressure_centerline_pa": pressure[center_y].copy(),
        "film_centerline_m": film[center_y].copy(),
        "theta_centerline": theta[center_y].copy(),
        "pressure_axial_pa": pressure[:, center_x].copy(),
        "film_axial_m": film[:, center_x].copy(),
        "theta_axial": theta[:, center_x].copy(),
        "qx_centerline_kg_m_s": qx[center_y].copy(),
        "qx_centerline_x_faces_m": (x_m[:-1] + x_m[1:]) / 2.0,
        "qy_centerline_kg_m_s": qy[:, center_x].copy(),
        "qy_centerline_y_faces_m": (y_m[:-1] + y_m[1:]) / 2.0,
        "pressure_pa": pressure,
        "film_m": film,
        "theta": theta,
    }
    return metrics, profiles


def verify_finite_roller_cuda_parity(case, state, probe=None) -> dict:
    """Compare CUDA donor/JFO fields and residual with the CPU reference."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for finite-roller production parity checks")
    backend = make_backend("torch", "cuda")
    evaluator = DeviceDonorJFO(case, backend, transport_fraction=0.0)
    probe = LimitedLoadProbe(case, transport_fraction=0.0) if probe is None else probe
    cpu_residual = probe.residual(state)
    state_device = as_float64(state, backend)
    torch.cuda.synchronize()
    residual_device = evaluator.residual(state_device)
    pressure_device, theta_device, film_device, deformation_device = evaluator.state_fields(
        state_device
    )
    torch.cuda.synchronize()
    pressure_cpu, theta_cpu, film_cpu = probe.fields(state)
    deformation_cpu = film_cpu - state[-1] * probe.system.problem.film_scale_m
    deformation_cpu -= probe.system.problem.rigid_geometry_m
    residual_difference = float(
        torch.max(
            torch.abs(residual_device - as_float64(cpu_residual, backend))
        ).detach().cpu()
    )
    pressure_difference = float(
        torch.max(torch.abs(pressure_device - as_float64(pressure_cpu, backend))).detach().cpu()
    )
    theta_difference = float(
        torch.max(torch.abs(theta_device - as_float64(theta_cpu, backend))).detach().cpu()
    )
    film_difference = float(
        torch.max(torch.abs(film_device - as_float64(film_cpu, backend))).detach().cpu()
    )
    deformation_difference = float(
        torch.max(
            torch.abs(deformation_device - as_float64(deformation_cpu, backend))
        ).detach().cpu()
    )
    tolerances = {
        "residual_linf": 1e-10,
        "pressure_linf_pa": 1e-10 * max(1.0, case.hertz.peak_pressure_pa),
        "theta_linf": 1e-12,
        "film_linf_m": 1e-10 * probe.system.problem.film_scale_m,
        "deformation_linf_m": 1e-10 * probe.system.problem.film_scale_m,
    }
    measured = {
        "residual_linf": residual_difference,
        "pressure_linf_pa": pressure_difference,
        "theta_linf": theta_difference,
        "film_linf_m": film_difference,
        "deformation_linf_m": deformation_difference,
    }
    passed = all(measured[name] <= tolerances[name] for name in tolerances)
    result = {
        "backend": "CUDA float64 device donor/JFO and FFT elasticity; CPU NumPy/SciPy reference",
        "passed": passed,
        "maximum_errors": measured,
        "tolerances": tolerances,
    }
    if not passed:
        raise RuntimeError(f"finite-roller CUDA/CPU parity gate failed: {result}")
    return result
