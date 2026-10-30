"""Bounded circular isothermal point-contact research prototype, CPU float64.

Public case/state quantities are SI; arrays (ny,nx), +x mean entrainment.
Donor transport is selected explicitly by the case, defaulting to the tested
candidate. The low-level arithmetic-capacity reference remains unchanged.
Numerical closure, correlations and spatial validation are distinct results.
"""

from dataclasses import asdict, dataclass
from math import isfinite
from types import SimpleNamespace

import numpy as np
from scipy.integrate import trapezoid
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import convolve2d

from ehl_solver.cavitating_coupled import CavitatingElasticityProblem
from ehl_solver.constitutive import IsothermalFluid
from ehl_solver.donor_load import DonorLoadSystem, donor_mass_audit
from ehl_solver.elasticity import cell_integrated_kernel
from ehl_solver.film_correlations import hamrock_dowson_point_contact
from ehl_solver.hertz import circular_point_contact
from ehl_solver.property_load_newton import PropertyLoadSystem, verify_seed
from ehl_solver.property_reynolds import admissible, mass_audit


@dataclass(frozen=True)
class PointContactCase:
    load_n: float = 200.0
    radius_m: float = 0.015
    contact_modulus_pa: float = 113e9
    viscosity_pa_s: float = 0.04
    density_kg_m3: float = 850.0
    entrainment_m_s: float = 2.0
    alpha_per_pa: float = 2.21e-8
    roelands_pressure_pa: float = 1.96e8
    viscosity_reference_pa_s: float = 6.31e-5
    ny: int = 33
    nx: int = 49
    domain_half_width_x_m: float | None = None
    domain_half_width_y_m: float | None = None
    transport: str = "donor"

    def __post_init__(self):
        for name in (
            "load_n",
            "radius_m",
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
        if (
            any(type(n) is not int or n < 5 or n % 2 == 0 for n in (self.ny, self.nx))
            or self.ny * self.nx > 20000
        ):
            raise ValueError("Odd grids >=5 and at most 20,000 physical nodes required")
        if self.transport not in ("donor", "arithmetic"):
            raise ValueError("transport must be donor or arithmetic")
        for name in ("domain_half_width_x_m", "domain_half_width_y_m"):
            value = getattr(self, name)
            if value is None:
                value = 3 * self.hertz.radius_m
            if isinstance(value, bool) or not isfinite(value) or value < 2 * self.hertz.radius_m:
                raise ValueError("Positive SI domain half-widths >=2 Hertz radii required")
            object.__setattr__(self, name, float(value))
        self.fluid()  # Validate supported Roelands index and reference properties.

    @property
    def hertz(self):
        return circular_point_contact(self.load_n, self.radius_m, self.contact_modulus_pa)

    def fluid(self):
        if self.viscosity_pa_s <= self.viscosity_reference_pa_s:
            raise ValueError("Ambient viscosity must exceed its Roelands reference")
        z = (
            self.alpha_per_pa
            * self.roelands_pressure_pa
            / np.log(self.viscosity_pa_s / self.viscosity_reference_pa_s)
        )
        return IsothermalFluid(
            self.viscosity_pa_s,
            self.density_kg_m3,
            viscosity_index=z,
            viscosity_reference_pa_s=self.viscosity_reference_pa_s,
            roelands_pressure_pa=self.roelands_pressure_pa,
        )

    def problem(self):
        a, h, p = self.hertz.radius_m, self.hertz.approach_m, self.hertz.peak_pressure_pa
        parameter = 12 * self.viscosity_pa_s * self.entrainment_m_s * a / (h * h * p)
        return CavitatingElasticityProblem(
            np.linspace(-self.domain_half_width_x_m, self.domain_half_width_x_m, self.nx),
            np.linspace(-self.domain_half_width_y_m, self.domain_half_width_y_m, self.ny),
            0.5e-6,
            self.radius_m,
            self.radius_m,
            self.contact_modulus_pa,
            a,
            h,
            p,
            parameter,
        )

    def system(self):
        factory = DonorLoadSystem if self.transport == "donor" else PropertyLoadSystem
        return factory(
            self.problem(),
            self.fluid(),
            self.entrainment_m_s,
            self.load_n,
            preconditioner="local",
            elasticity_mode="cached",
        )

    def correlation(self):
        return hamrock_dowson_point_contact(
            load_n=self.load_n,
            radius_x_m=self.radius_m,
            contact_modulus_pa=self.contact_modulus_pa,
            viscosity_pa_s=self.viscosity_pa_s,
            entrainment_m_s=self.entrainment_m_s,
            alpha_per_pa=self.alpha_per_pa,
        )

    def physical_signature(self):
        return {
            k: v
            for k, v in asdict(self).items()
            if k not in ("ny", "nx", "domain_half_width_x_m", "domain_half_width_y_m")
        }


def hertz_guess(case, system):
    """Unaccepted force-normalized Hertz P, theta=1 and explicit positive-gap guess.

    The initial offset makes the sampled minimum equal the HD central estimate.
    This constructs one starting guess; it never modifies Newton trial fields.
    """
    problem = system.problem
    x, y = np.meshgrid(problem.x_m, problem.y_m)
    p = case.hertz.peak_pressure_pa * np.sqrt(
        np.maximum(0.0, 1 - (x * x + y * y) / case.hertz.radius_m**2)
    )
    p *= case.load_n / problem.load_n(p)
    w = system.base.elastic_deformation(p)
    offset = case.correlation().central_film_m - float(np.min(problem.rigid_geometry_m + w))
    seed = SimpleNamespace(pressure_pa=p, fractional_film_content=np.ones(problem.shape))
    z = system.pack(seed, offset)
    system.solution(z)  # Strict total-film/property feasibility; no acceptance.
    return z, dict(
        kind="Hertz starting guess; not a solution",
        accepted=False,
        theta="uniform one",
        offset_m=offset,
        initial_force_n=problem.load_n(p),
    )


def transfer_guess(source_case, target_case, source_root, system):
    """Same-model spatial warm start; every target is explicitly unaccepted.

    Same bounds permit bilinear interpolation, including nonnested grids;
    expanded bounds require exact fixed-spacing embedding. Exterior embedding
    starts at p=0,theta=1. No extrapolation, field clipping or offset change.
    """
    if source_case.physical_signature() != target_case.physical_signature():
        raise ValueError("Seed must have identical physical parameters and transport")
    old = source_case.system()
    s = source_root.solution
    verify_seed(old, s, source_root.central_film_offset_m)
    if abs(old.problem.load_n(s.pressure_pa) / source_case.load_n - 1) > 1e-8:
        raise ValueError("Seed force gate failed")
    op, tp = old.problem, system.problem
    same_bounds = all(
        np.array_equal(a[[0, -1]], b[[0, -1]]) for a, b in ((op.x_m, tp.x_m), (op.y_m, tp.y_m))
    )
    if same_bounds:
        if target_case.ny < source_case.ny or target_case.nx < source_case.nx:
            raise ValueError("Spatial seed may refine, not coarsen")
        yy, xx = np.meshgrid(tp.y_m, tp.x_m, indexing="ij")
        points = np.c_[yy.ravel(), xx.ravel()]

        def interpolate(a):
            return RegularGridInterpolator((op.y_m, op.x_m), a, bounds_error=True)(points).reshape(
                tp.shape
            )

        p, t = interpolate(s.pressure_pa), interpolate(s.fractional_film_content)
        kind = "bilinear interpolation; no extrapolation"
    else:
        if (
            tp.x_m[0] > op.x_m[0]
            or tp.x_m[-1] < op.x_m[-1]
            or tp.y_m[0] > op.y_m[0]
            or tp.y_m[-1] < op.y_m[-1]
            or not np.isclose(tp.dx_m, op.dx_m, rtol=1e-12, atol=0)
            or not np.isclose(tp.dy_m, op.dy_m, rtol=1e-12, atol=0)
        ):
            raise ValueError("Domain expansion must preserve physical spacing and contain source")
        ix = int(round((op.x_m[0] - tp.x_m[0]) / tp.dx_m))
        iy = int(round((op.y_m[0] - tp.y_m[0]) / tp.dy_m))
        if not (
            np.allclose(tp.x_m[ix : ix + source_case.nx], op.x_m, rtol=1e-12, atol=1e-18)
            and np.allclose(tp.y_m[iy : iy + source_case.ny], op.y_m, rtol=1e-12, atol=1e-18)
        ):
            raise ValueError("Source nodes must align with expanded domain")
        p, t = np.zeros(tp.shape), np.ones(tp.shape)
        p[iy : iy + source_case.ny, ix : ix + source_case.nx] = s.pressure_pa
        t[iy : iy + source_case.ny, ix : ix + source_case.nx] = s.fractional_film_content
        kind = "fixed-spacing embedding; exterior p=0,theta=1"
    z = system.pack(
        SimpleNamespace(pressure_pa=p, fractional_film_content=t), source_root.central_film_offset_m
    )
    system.solution(z)
    return z, dict(
        kind=kind,
        accepted=False,
        offset_adjustment_m=0.0,
        source_shape_ny_nx=list(op.shape),
        target_shape_ny_nx=list(tp.shape),
        initial_force_n=tp.load_n(p),
        initial_force_relative_error=tp.load_n(p) / target_case.load_n - 1,
    )


def audit_root(case, root):
    """Direct elastic reconstruction, SI face fluxes, force and feasibility gates."""
    problem, fluid = case.problem(), case.fluid()
    s = root.solution
    p, t = s.pressure_pa, s.fractional_film_content
    kernel = cell_integrated_kernel(
        problem.shape, problem.dx_m, problem.dy_m, case.contact_modulus_pa
    )
    w = convolve2d(p, kernel, mode="same")
    h = root.central_film_offset_m + problem.rigid_geometry_m + w
    audit = donor_mass_audit if case.transport == "donor" else mass_audit
    mass = audit(
        problem.x_m,
        problem.y_m,
        h,
        p,
        t,
        fluid,
        case.entrainment_m_s,
        length_scale_m=problem.length_scale_m,
        film_scale_m=problem.film_scale_m,
        pressure_scale_pa=problem.pressure_scale_pa,
    )
    force = float(trapezoid(trapezoid(p, x=problem.x_m, axis=1), x=problem.y_m))
    film_error = float(np.max(abs(h - s.film_thickness_m)))
    displacement_error = float(np.max(abs(w - s.elastic_displacement_m)))
    boundaries = np.r_[p[0], p[-1], p[:, 0], p[:, -1]]
    numerical = bool(
        admissible(mass, p, t, problem.pressure_scale_pa)
        and abs(force / case.load_n - 1) <= 1e-8
        and np.min(h) > problem.film_floor_m
        and np.all(boundaries == 0)
        and np.all(t[1:-1, 0] == 1)
        and max(film_error, displacement_error) <= 1e-10 * problem.film_scale_m
    )
    ref = case.correlation()
    centre = (case.ny // 2, case.nx // 2)
    iy, ix = np.unravel_index(np.argmin(h), h.shape)
    return dict(
        numerically_accepted=numerical,
        mass=mass,
        force_n=force,
        force_relative_error=force / case.load_n - 1,
        film_reconstruction_error_m=film_error,
        displacement_reconstruction_error_m=displacement_error,
        minimum_film_m=float(np.min(h)),
        central_film_m=float(h[centre]),
        minimum_film_location_yx_m=[float(problem.y_m[iy]), float(problem.x_m[ix])],
        hamrock_dowson=asdict(ref),
        minimum_relative_correlation_difference=float(np.min(h) / ref.minimum_film_m - 1),
        central_relative_correlation_difference=float(h[centre] / ref.central_film_m - 1),
    )


def audit_history(history):
    maximum = 0.0
    for record in history:
        selected = [(m, t) for m in record["models"] for t in m["trials"] if t["accepted"]]
        if len(selected) != 1:
            raise ValueError("Exactly one accepted trial per correction required")
        model, trial = selected[0]
        if (
            not all(
                isfinite(v)
                for v in (model["slope"], trial["merit"], record["merit"], trial["alpha"])
            )
            or model["slope"] >= 0
            or trial["merit"] > record["merit"] + 1e-4 * trial["alpha"] * model["slope"]
        ):
            raise ValueError("Original directional Armijo gate failed")
        for model in record["models"]:
            if (
                not isfinite(model["linear_true_residual"])
                or not 0 <= model["linear_true_residual"] <= 5e-9
            ):
                raise ValueError("Original true-linear-residual gate failed")
            maximum = max(maximum, model["linear_true_residual"])
    return dict(
        corrections=len(history),
        maximum_true_linear_residual=maximum,
        directional_armijo_verified=True,
    )
