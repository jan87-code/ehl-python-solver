"""Small-grid analytic Newton reference for warm signed-offset JFO roots.

SI fields, full-source uniform FFT, pressure-dependent physical mass flux.
Dense Jacobian storage is deliberately limited to 400 interior nodes. This
is a separate bounded reference, not a change to the cold Picard solver.
"""

from dataclasses import dataclass

import numpy as np

from ehl_solver.backend import make_backend
from ehl_solver.cavitating_newton import CavitatingNewtonSystem
from ehl_solver.convolution import fft_convolve_same
from ehl_solver.elasticity import ElasticityOperator
from ehl_solver.property_coupled import PropertyCoupledSolution
from ehl_solver.property_reynolds import admissible, mass_audit, mass_fluxes


class PropertyWarmSystem(CavitatingNewtonSystem):
    def __init__(self, problem, fluid, speed_m_s, offset_m, *, maximum_interior_nodes=400,
                 elasticity_mode="reference"):
        super().__init__(problem)
        if elasticity_mode not in ("reference", "cached"):
            raise ValueError("elasticity_mode must be reference or cached (CPU only)")
        if maximum_interior_nodes not in (400, 20000, 30000) or self.n > maximum_interior_nodes:
            raise ValueError(f"warm system limited to {maximum_interior_nodes} interior nodes")
        if not np.isfinite(offset_m) or not np.isfinite(speed_m_s) or speed_m_s <= 0:
            raise ValueError("finite signed offset and positive SI speed required")
        parameter = (
            12
            * fluid.viscosity_pa_s
            * speed_m_s
            * problem.length_scale_m
            / (problem.film_scale_m**2 * problem.pressure_scale_pa)
        )
        if not np.isclose(parameter, problem.reynolds_parameter, rtol=1e-12, atol=0):
            raise ValueError("fluid and speed must reproduce fixed problem Lambda")
        self.fluid, self.speed, self.offset = fluid, speed_m_s, float(offset_m)
        self.elasticity_mode = elasticity_mode
        self.elastic_operator = (
            ElasticityOperator(problem.shape, problem.dx_m, problem.dy_m, problem.e_star_pa)
            if elasticity_mode == "cached" else None
        )
        self.flux_scale = (
            fluid.density_kg_m3
            * problem.film_scale_m**3
            * problem.pressure_scale_pa
            / (12 * fluid.viscosity_pa_s * problem.length_scale_m)
        )

    def unpack(self, state):
        z = np.asarray(state, dtype=float)
        if z.shape != (self.size,) or not np.all(np.isfinite(z)):
            raise ValueError("finite P,D state required")
        problem = self.problem
        p, theta = np.zeros(problem.shape), np.ones(problem.shape)
        p[1:-1, 1:-1] = z[: self.n].reshape(p[1:-1, 1:-1].shape)
        theta[1:-1, 1:-1] = 1 - z[self.n :].reshape(p[1:-1, 1:-1].shape)
        theta[1:-1, 0] = problem.inlet_fractional_film_content
        theta[1:-1, -1] = theta[1:-1, -2]
        theta[0], theta[-1] = theta[1], theta[-2]
        if self.elastic_operator is None:
            h, w = problem.fields(p * problem.pressure_scale_pa, central_film_offset_m=self.offset)
        else:
            w = self.elastic_deformation(p * problem.pressure_scale_pa)
            h = self.offset + problem.rigid_geometry_m + w
            if not np.all(np.isfinite(h)) or np.any(h <= problem.film_floor_m):
                raise ValueError("coupled film is nonfinite or at/below film_floor_m")
        return p, theta, h, w, self.offset

    def elastic_deformation(self, pressure_pa):
        """Pa -> gap increase m, same kernel; optional per-system CPU spectrum.

        Cached padding is FFT-friendly and still computes the full linear
        convolution. Failures propagate; reference mode remains the default.
        """
        if self.elastic_operator is not None:
            return self.elastic_operator(pressure_pa)
        return fft_convolve_same(pressure_pa, self.problem.elastic_kernel, make_backend("numpy"))

    def divergence(self, qx, qy):
        return (
            np.diff(qx[1:-1], axis=1) / self.problem.dx_m
            + np.diff(qy[:, 1:-1], axis=0) / self.problem.dy_m
        )

    def residual(self, state):
        p, t, h, _, _ = self.unpack(state)
        qx, qy = mass_fluxes(
            self.problem.x_m,
            self.problem.y_m,
            h,
            p * self.problem.pressure_scale_pa,
            t,
            self.fluid,
            self.speed,
        )
        mass = self.divergence(qx, qy) * self.problem.length_scale_m / self.flux_scale
        return np.r_[mass.ravel() / self.mass_scale, np.minimum(state[: self.n], state[self.n :])]

    def jacobian_vector(self, state, direction, *, offset_direction_m=0.0):
        v = np.asarray(direction, dtype=float)
        if v.shape != (self.size,) or not np.all(np.isfinite(v)):
            raise ValueError("finite P,D direction required")
        p, t, h, _, _ = self.unpack(state)
        problem = self.problem
        p = p * problem.pressure_scale_pa
        dp, dt = np.zeros(problem.shape), np.zeros(problem.shape)
        dp[1:-1, 1:-1] = v[: self.n].reshape(dp[1:-1, 1:-1].shape) * problem.pressure_scale_pa
        dt[1:-1, 1:-1] = -v[self.n :].reshape(dt[1:-1, 1:-1].shape)
        dh = self.elastic_deformation(dp) + offset_direction_m
        rho, eta, dr, de = self.fluid.properties(p)
        dr, de = dr * dp, de * dp
        m = rho * h**3 / (12 * eta)
        dm = (dr * h**3 + 3 * rho * h * h * dh) / (12 * eta) - m * de / eta
        c, dc = rho * h, dr * h + rho * dh
        qx = (
            -(dm[:, 1:] + dm[:, :-1]) / 2 * np.diff(p, axis=1) / problem.dx_m
            - (m[:, 1:] + m[:, :-1]) / 2 * np.diff(dp, axis=1) / problem.dx_m
            + self.speed
            * ((dc[:, 1:] + dc[:, :-1]) / 2 * t[:, :-1] + (c[:, 1:] + c[:, :-1]) / 2 * dt[:, :-1])
        )
        qy = (
            -(dm[1:] + dm[:-1]) / 2 * np.diff(p, axis=0) / problem.dy_m
            - (m[1:] + m[:-1]) / 2 * np.diff(dp, axis=0) / problem.dy_m
        )
        mass = self.divergence(qx, qy) * problem.length_scale_m / self.flux_scale
        full = state[: self.n] >= state[self.n :]
        return np.r_[mass.ravel() / self.mass_scale, np.where(full, v[self.n :], v[: self.n])]

    def audited_solution(self, state, history=()):
        p, t, h, w, _ = self.unpack(state)
        p = p * self.problem.pressure_scale_pa
        audit = mass_audit(
            self.problem.x_m,
            self.problem.y_m,
            h,
            p,
            t,
            self.fluid,
            self.speed,
            length_scale_m=self.problem.length_scale_m,
            film_scale_m=self.problem.film_scale_m,
            pressure_scale_pa=self.problem.pressure_scale_pa,
        )
        return PropertyCoupledSolution(
            p, t, h, w, self.problem.load_n(p), audit, tuple(history), 1.0
        )


@dataclass(frozen=True)
class WarmPropertyRoot:
    central_film_offset_m: float
    solution: PropertyCoupledSolution


class WarmPropertyError(RuntimeError):
    def __init__(self, message, history, last_solution, inner_failure=None):
        super().__init__(message)
        self.history, self.last_solution = tuple(history), last_solution
        self.inner_failure = inner_failure


def solve_warm_property_offset(
    problem, fluid, speed_m_s, offset_m, *, seed, seed_offset_m, maximum_iterations=30
):
    """Warm full-material/full-compliance root, <=30 analytic Newton corrections.

    Seed must pass fresh original SI mass/constraint and full-source film
    audits at its declared offset. New offset may be signed; its *total*
    film must be feasible under seed pressure. Dense linear solve, verified
    JVP correction and true directional Armijo (including min ties), <=24
    backtracks per correction. Intermediate states need not conserve mass;
    only the final independently audited state can be an accepted root.
    """
    if (
        not isinstance(maximum_iterations, int)
        or isinstance(maximum_iterations, bool)
        or not 0 <= maximum_iterations <= 30
    ):
        raise ValueError("Newton correction budget must lie in 0..30")
    original = PropertyWarmSystem(problem, fluid, speed_m_s, seed_offset_m)
    z = original.pack(seed)
    checked = original.audited_solution(z)
    if (
        seed.constitutive_strength != 1
        or np.any(
            np.r_[
                seed.pressure_pa[0],
                seed.pressure_pa[-1],
                seed.pressure_pa[:, 0],
                seed.pressure_pa[:, -1],
            ]
            != 0
        )
        or not np.allclose(
            seed.fractional_film_content, checked.fractional_film_content, rtol=0, atol=1e-10
        )
        or not admissible(
            checked.audit,
            checked.pressure_pa,
            checked.fractional_film_content,
            problem.pressure_scale_pa,
        )
        or not np.allclose(
            checked.film_thickness_m,
            seed.film_thickness_m,
            rtol=0,
            atol=1e-10 * problem.film_scale_m,
        )
        or not np.allclose(
            checked.elastic_displacement_m,
            seed.elastic_displacement_m,
            rtol=0,
            atol=1e-10 * problem.film_scale_m,
        )
    ):
        raise ValueError("warm seed failed original physical/film audit")
    system = PropertyWarmSystem(problem, fluid, speed_m_s, offset_m)
    history, last = [], None
    try:
        for iteration in range(maximum_iterations + 1):
            last = system.audited_solution(z, history)
            if admissible(
                last.audit,
                last.pressure_pa,
                last.fractional_film_content,
                problem.pressure_scale_pa,
            ):
                return WarmPropertyRoot(float(offset_m), last)
            if iteration == maximum_iterations:
                raise RuntimeError("warm Newton correction budget exhausted")
            residual = system.residual(z)
            # At most 800x800 float64: explicit bounded verification reference.
            matrix = np.column_stack([system.jacobian_vector(z, v) for v in np.eye(system.size)])
            rows = np.max(abs(matrix), axis=1)
            if np.any(rows == 0):
                raise RuntimeError("zero Newton row")
            equilibrated = matrix / rows[:, None]
            columns = np.max(abs(equilibrated), axis=0)
            if np.any(columns == 0):
                raise RuntimeError("zero Newton column")
            direction = np.linalg.solve(equilibrated / columns, -residual / rows) / columns
            error = float(
                np.linalg.norm(system.jacobian_vector(z, direction) + residual)
                / np.linalg.norm(residual)
            )
            if not np.isfinite(error) or error > 5e-9:
                raise RuntimeError("true Newton correction residual failed")
            slope = system.merit_slope(z, residual, direction)
            merit = float(np.dot(residual, residual) / residual.size / 2)
            if not slope < 0:
                raise RuntimeError("Newton direction is not a true merit descent")
            trials = []
            for backtrack in range(24):
                alpha = 2.0 ** (-backtrack)
                trial = z + alpha * direction
                accepted = False
                try:
                    if np.min(trial) < -1e-10 or np.max(trial[system.n :]) > 1 + 1e-10:
                        raise ValueError("pressure/content infeasible")
                    r = system.residual(trial)
                    value = float(np.dot(r, r) / r.size / 2)
                    accepted = value <= merit + 1e-4 * alpha * slope
                    reason = "Armijo" if not accepted else "accepted"
                except (ValueError, FloatingPointError) as exc:
                    reason = str(exc)
                trials.append({"alpha": alpha, "accepted": accepted, "reason": reason})
                if accepted:
                    break
            history.append(
                {
                    "iteration": iteration,
                    "audit": last.audit,
                    "true_linear_residual": error,
                    "merit": merit,
                    "slope": slope,
                    "trials": trials,
                }
            )
            if not accepted:
                raise RuntimeError("warm Newton line search exhausted")
            z = trial
    except (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        raise WarmPropertyError(str(exc), history, last, exc) from exc


class WarmContinuationError(RuntimeError):
    def __init__(self, message, attempts, last_root):
        super().__init__(message)
        self.attempts, self.last_root = tuple(attempts), last_root


def continue_warm_property_offset(
    problem,
    fluid,
    speed_m_s,
    target_offset_m,
    *,
    seed,
    seed_offset_m,
    initial_step_m,
    minimum_step_m=1e-12,
    maximum_attempts=60,
    callback=None,
):
    """Bounded signed-offset steps; retry from the last audited root, never a failure.

    Each proposed offset step is at most half the last minimum total film.
    Failed Newton trials halve the step; accepted trials grow it by 1.5.
    Callback(record, root_or_error) streams all accepted and rejected states.
    Only scalar attempt summaries are held in this continuation's ledger.
    This follows offset, not imposed load; every resulting force is measured.
    """
    if (
        not np.isfinite(target_offset_m)
        or not all(np.isfinite(v) and v > 0 for v in (initial_step_m, minimum_step_m))
        or not isinstance(maximum_attempts, int)
        or isinstance(maximum_attempts, bool)
        or not 1 <= maximum_attempts <= 60
    ):
        raise ValueError("invalid bounded offset continuation controls")
    if callback is not None and not callable(callback):
        raise ValueError("callback must be callable")
    current = solve_warm_property_offset(
        problem, fluid, speed_m_s, seed_offset_m, seed=seed, seed_offset_m=seed_offset_m
    )
    attempts = []
    step = initial_step_m
    for _ in range(maximum_attempts):
        remaining = target_offset_m - current.central_film_offset_m
        if remaining == 0:
            return current, tuple(attempts)
        amount = min(abs(remaining), step, 0.5 * float(current.solution.film_thickness_m.min()))
        if amount < minimum_step_m and amount < abs(remaining):
            raise WarmContinuationError("minimum offset step reached", attempts, current)
        target = (
            target_offset_m
            if amount == abs(remaining)
            else current.central_film_offset_m + np.sign(remaining) * amount
        )
        record = {
            "from_offset_m": current.central_film_offset_m,
            "trial_offset_m": float(target),
            "step_m": float(amount),
            "accepted": False,
        }
        try:
            root = solve_warm_property_offset(
                problem,
                fluid,
                speed_m_s,
                target,
                seed=current.solution,
                seed_offset_m=current.central_film_offset_m,
            )
        except WarmPropertyError as exc:
            record["reason"] = str(exc)
            attempts.append(record)
            if callback is not None:
                callback(record, exc)
            step = amount / 2
            continue
        record.update(
            accepted=True,
            load_n=root.solution.load_n,
            minimum_film_m=float(root.solution.film_thickness_m.min()),
            newton_corrections=len(root.solution.history),
            audit=root.solution.audit,
        )
        attempts.append(record)
        if callback is not None:
            callback(record, root)
        current = root
        step = amount * 1.5
    if current.central_film_offset_m == target_offset_m:
        return current, tuple(attempts)
    raise WarmContinuationError("offset continuation attempt budget exhausted", attempts, current)
