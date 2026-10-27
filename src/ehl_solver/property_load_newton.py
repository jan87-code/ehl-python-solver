"""Matrix-free bordered full-material JFO force Newton and bounded load steps.

SI interfaces; (ny,nx), +x entrainment. Full-source FFT in every analytic
product, sparse frozen-property/film preconditioning; no dense Jacobian.
"""

from copy import copy
from dataclasses import dataclass

import numpy as np
from scipy.sparse import bmat, coo_matrix, csr_matrix, diags, hstack, vstack

from ehl_solver.backend import make_backend
from ehl_solver.cavitating_newton import (
    CavitatingLinearError,
    CavitatingNewtonSystem,
    solve_cavitating_correction,
)
from ehl_solver.convolution import fft_convolve_same
from ehl_solver.property_reynolds import _frozen_system, admissible
from ehl_solver.property_warm import PropertyWarmSystem, WarmPropertyRoot


class PropertyLoadSystem:
    merit_slope = CavitatingNewtonSystem.merit_slope

    def __init__(self, problem, fluid, speed_m_s, target_load_n, *, preconditioner="frozen"):
        if preconditioner not in ("frozen", "material", "local"):
            raise ValueError("preconditioner must be frozen, material or local")
        self.preconditioner = preconditioner
        if not np.isfinite(target_load_n) or target_load_n <= 0:
            raise ValueError("positive finite SI target load required")
        if np.prod(problem.shape) > 20000:
            raise ValueError("physical node budget exceeds 20000")
        self.problem = problem
        self.base = PropertyWarmSystem(
            problem, fluid, speed_m_s, problem.central_film_offset_m, maximum_interior_nodes=20000
        )
        self.n = self.base.n
        self.size = 2 * self.n + 1
        self.target_load_n = target_load_n
        self.active_override = None
        self.linear_mode = False

    def pack(self, seed, offset):
        return np.r_[self.base.pack(seed), offset / self.problem.film_scale_m]

    def sync(self, z):
        if np.shape(z) != (self.size,) or not np.all(np.isfinite(z)):
            raise ValueError("finite P,D,h0/href state required")
        self.base.offset = float(z[-1] * self.problem.film_scale_m)

    def residual(self, z):
        self.sync(z)
        r = self.base.residual(z[:-1])
        if self.linear_mode:
            full = self.full(z)
            r[self.n :] = np.where(full, z[self.n : 2 * self.n], z[: self.n])
        load = self.problem.load_n(self.base.unpack(z[:-1])[0] * self.problem.pressure_scale_pa)
        return np.r_[r, (load - self.target_load_n) / self.target_load_n]

    def full(self, z):
        return (
            z[: self.n] >= z[self.n : 2 * self.n]
            if self.active_override is None
            else self.active_override
        )

    def jacobian_vector(self, z, v):
        self.sync(z)
        if np.shape(v) != (self.size,) or not np.all(np.isfinite(v)):
            raise ValueError("finite bordered direction required")
        r = self.base.jacobian_vector(
            z[:-1], v[:-1], offset_direction_m=v[-1] * self.problem.film_scale_m
        )
        if self.active_override is not None:
            r[self.n :] = np.where(self.full(z), v[self.n : 2 * self.n], v[: self.n])
        load = (
            self.problem.pressure_scale_pa
            * self.problem.dx_m
            * self.problem.dy_m
            * np.sum(v[: self.n])
        )
        return np.r_[r, load / self.target_load_n]

    def frozen_matrix(self, z):
        self.sync(z)
        p, theta, film, _, _ = self.base.unpack(z[:-1])
        r, v, dr, dv = self.base.fluid.ratios(p * self.problem.pressure_scale_pa)
        h = film / self.problem.film_scale_m
        mass, _ = _frozen_system(
            self.problem.x_m / self.problem.length_scale_m,
            self.problem.y_m / self.problem.length_scale_m,
            r * h**3 / v,
            r * h,
            self.problem.reynolds_parameter,
            np.zeros(self.problem.shape),
            self.problem.inlet_fractional_film_content,
        )
        # The frozen assembly uses theta; Newton unknowns use D=1-theta.
        mass = mass @ diags(np.r_[np.ones(self.n), -np.ones(self.n)])
        if self.preconditioner != "frozen":
            pressure_part = self.tangent_pressure(p, theta, h, r, v, dr, dv)
            mass = mass + hstack([pressure_part, csr_matrix((self.n, self.n))])
        full = self.full(z)
        block = vstack(
            [
                mass / self.base.mass_scale,
                hstack([diags((~full).astype(float)), diags(full.astype(float))]),
            ],
            format="csc",
        )
        d = np.zeros(self.size)
        d[-1] = 1.0
        column = self.jacobian_vector(z, d)[:-1]
        row = np.r_[
            np.full(
                self.n,
                self.problem.pressure_scale_pa
                * self.problem.dx_m
                * self.problem.dy_m
                / self.target_load_n,
            ),
            np.zeros(self.n),
        ]
        return bmat(
            [[block, csr_matrix(column[:, None])], [csr_matrix(row[None, :]), csr_matrix((1, 1))]],
            format="csc",
        )

    def tangent_pressure(self, p, theta, h, r, v, dr, dv):
        """Sparse extra pressure action at frozen film, optionally cell-self film.

        All coefficients here are preconditioning only. The true operator
        retains full-source elastic response and the original pressure law.
        """
        pref = self.problem.pressure_scale_pa
        dm = h**3 / v * (dr - r * dv / v) * pref
        dc = h * dr * pref
        if self.preconditioner == "local":
            ny, nx = self.problem.shape
            compliance = (
                self.problem.elastic_kernel[ny - 1, nx - 1] * pref / self.problem.film_scale_m
            )
            dm = dm + 3 * r * h**2 / v * compliance
            dc = dc + r * compliance
        dx, dy = self.problem.dx, self.problem.dy
        gx, gy = np.diff(p, axis=1) / dx, np.diff(p, axis=0) / dy
        lam = self.problem.reynolds_parameter
        ax = -0.5 * dm[:, :-1] * gx + 0.5 * lam * dc[:, :-1] * theta[:, :-1]
        bx = -0.5 * dm[:, 1:] * gx + 0.5 * lam * dc[:, 1:] * theta[:, :-1]
        ay = -0.5 * dm[:-1] * gy
        by = -0.5 * dm[1:] * gy
        ny, nx = h.shape
        k = np.arange(self.n).reshape(ny - 2, nx - 2)
        center = (ax[1:-1, 1:] - bx[1:-1, :-1]) / dx + (ay[1:, 1:-1] - by[:-1, 1:-1]) / dy
        rows = [k.ravel()]
        cols = [k.ravel()]
        values = [center.ravel()]
        for row, col, val in (
            (k[:, :-1], k[:, 1:], bx[1:-1, 1:-1] / dx),
            (k[:, 1:], k[:, :-1], -ax[1:-1, 1:-1] / dx),
            (k[:-1], k[1:], by[1:-1, 1:-1] / dy),
            (k[1:], k[:-1], -ay[1:-1, 1:-1] / dy),
        ):
            rows.append(row.ravel())
            cols.append(col.ravel())
            values.append(val.ravel())
        return coo_matrix(
            (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
            shape=(self.n, self.n),
        ).tocsr()

    def solution(self, z, history=()):
        self.sync(z)
        return WarmPropertyRoot(
            float(self.base.offset), self.base.audited_solution(z[:-1], history)
        )

    def film_direction(self, z, direction):
        self.sync(z)
        dp = np.zeros(self.problem.shape)
        dp[1:-1, 1:-1] = (
            direction[: self.n].reshape(dp[1:-1, 1:-1].shape) * self.problem.pressure_scale_pa
        )
        return (
            fft_convolve_same(dp, self.problem.elastic_kernel, make_backend("numpy"))
            + direction[-1] * self.problem.film_scale_m
        )

    def correction_view(self, z, full):
        other = copy(self)
        other.base = copy(self.base)
        other.active_override = np.array(full, copy=True)
        other.linear_mode = True
        return other


class PropertyLoadNewtonError(RuntimeError):
    def __init__(self, message, history, last_root, diagnostics=None):
        super().__init__(message)
        self.history = tuple(history)
        self.last_root = last_root
        self.diagnostics = diagnostics or {}


def verify_seed(system, seed, offset):
    z = system.pack(seed, offset)
    s = system.solution(z).solution
    if (
        seed.constitutive_strength != 1
        or not admissible(
            s.audit, s.pressure_pa, s.fractional_film_content, system.problem.pressure_scale_pa
        )
        or not np.allclose(
            s.film_thickness_m,
            seed.film_thickness_m,
            rtol=0,
            atol=1e-10 * system.problem.film_scale_m,
        )
        or not np.allclose(
            s.elastic_displacement_m,
            seed.elastic_displacement_m,
            rtol=0,
            atol=1e-10 * system.problem.film_scale_m,
        )
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
            s.fractional_film_content, seed.fractional_film_content, rtol=0, atol=1e-10
        )
    ):
        raise ValueError("seed fails independent original physical/film/boundary audit")
    return z


def solve_property_load_newton(
    problem,
    fluid,
    speed_m_s,
    target_load_n,
    *,
    seed,
    seed_offset_m,
    maximum_iterations=30,
    maximum_krylov_cycles=10,
    schur_tolerance=1e-10,
    active_recovery=True,
    preconditioner="frozen",
):
    """Bordered force solve; true-JVP-checked Schur/GMRES and bounded globalization.

    Recovery solves selected active constraints P_new=0 or D_new=0, checking
    the complete selected-system correction. It is an active-set Newton
    model, not a claim that a switched row is the old min derivative.
    Merit/slope always use original min equations, including directional ties.
    No projection of pressure, content or film is performed.
    """
    if (
        not isinstance(maximum_iterations, int)
        or isinstance(maximum_iterations, bool)
        or not 0 <= maximum_iterations <= 30
        or not isinstance(maximum_krylov_cycles, int)
        or isinstance(maximum_krylov_cycles, bool)
        or not 1 <= maximum_krylov_cycles <= 10
    ):
        raise ValueError("Newton/Krylov budgets must lie in 0..30 and 1..10")
    system = PropertyLoadSystem(
        problem, fluid, speed_m_s, target_load_n, preconditioner=preconditioner
    )
    z = verify_seed(system, seed, seed_offset_m)
    return _solve_property_load_state(
        system, z, maximum_iterations, maximum_krylov_cycles, schur_tolerance, active_recovery
    )


def _solve_property_load_state(
    system,
    z,
    maximum_iterations,
    maximum_krylov_cycles,
    schur_tolerance,
    active_recovery,
    *,
    active_lookahead=False,
):
    """Shared Newton engine; callers distinguish audited roots from initial guesses."""
    problem = system.problem
    target_load_n = system.target_load_n
    z = np.array(z, dtype=float, copy=True)
    history = []
    last = None
    diagnostics = {}
    try:
        for iteration in range(maximum_iterations + 1):
            last = system.solution(z, history)
            s = last.solution
            if (
                admissible(
                    s.audit, s.pressure_pa, s.fractional_film_content, problem.pressure_scale_pa
                )
                and abs(s.load_n - target_load_n) / target_load_n <= 1e-8
            ):
                return last
            if iteration == maximum_iterations:
                raise RuntimeError("load Newton correction budget exhausted")
            residual = system.residual(z)
            merit = float(residual @ residual / (2 * system.size))
            record = {
                "iteration": iteration,
                "audit": s.audit,
                "load_n": s.load_n,
                "offset_m": last.central_film_offset_m,
                "merit": merit,
                "models": [],
            }
            history.append(record)
            full = system.full(z).copy()
            seen = set()
            accepted = False
            backup = None
            for recovery in range(8 if active_recovery else 1):
                key = np.packbits(full).tobytes()
                if key in seen:
                    break
                seen.add(key)
                view = system if recovery == 0 else system.correction_view(z, full)
                try:
                    linear = solve_cavitating_correction(
                        view,
                        z,
                        maximum_krylov_cycles=maximum_krylov_cycles,
                        gmres_restart=60,
                        schur_tolerance=schur_tolerance,
                    )
                except CavitatingLinearError as exc:
                    if backup is None:
                        raise
                    record.setdefault("active_recovery_failures", []).append(
                        {"model": recovery, "reason": str(exc), "diagnostics": exc.diagnostics}
                    )
                    break
                d = linear.direction
                # Slope uses actual min derivative, not switched active equations.
                slope = system.merit_slope(z, residual, d)
                model = {
                    "recovery": recovery,
                    "active_changes": int(np.count_nonzero(full != system.full(z))),
                    "linear_true_residual": linear.true_linear_residual,
                    "linear_iterations": linear.linear_iterations,
                    "krylov_histories": linear.krylov_histories,
                    "used_bordered_fallback": linear.used_bordered_fallback,
                    "schur_indicator": linear.schur_indicator,
                    "slope": slope,
                    "trials": [],
                }
                record["models"].append(model)
                predicted = z + d
                proposed = full.copy()
                proposed[predicted[: system.n] < -1e-10] = False
                proposed[predicted[system.n : 2 * system.n] < -1e-10] = True
                novel = np.packbits(proposed).tobytes() not in seen
                if slope < 0:
                    # First try the full step; next start at a safe affine bound.
                    lower = z[: 2 * system.n] + 1e-10
                    bound = 1.0
                    falling = d[: 2 * system.n] < -lower
                    if np.any(falling):
                        bound = min(
                            bound,
                            0.99 * float(np.min(lower[falling] / -d[: 2 * system.n][falling])),
                        )
                    rising = d[system.n : 2 * system.n] > (1 + 1e-10 - z[system.n : 2 * system.n])
                    if np.any(rising):
                        bound = min(
                            bound,
                            0.99
                            * float(
                                np.min(
                                    (1 + 1e-10 - z[system.n : 2 * system.n][rising])
                                    / d[system.n : 2 * system.n][rising]
                                )
                            ),
                        )
                    system.sync(z)
                    film = system.base.unpack(z[:-1])[2]
                    dh = system.film_direction(z, d)
                    closing = dh < -(film - problem.film_floor_m)
                    if np.any(closing):
                        bound = min(
                            bound,
                            0.99
                            * float(np.min((film[closing] - problem.film_floor_m) / -dh[closing])),
                        )
                    model["fraction_to_boundary"] = float(bound)
                    for backtrack in range(24):
                        alpha = (
                            1.0 if backtrack == 0 else max(0.0, bound) * 2.0 ** (-(backtrack - 1))
                        )
                        trial = z + alpha * d
                        value = None
                        try:
                            if (
                                alpha <= 0
                                or np.min(trial[: 2 * system.n]) < -1e-10
                                or np.max(trial[system.n : 2 * system.n]) > 1 + 1e-10
                            ):
                                raise ValueError("pressure/content infeasible")
                            r = system.residual(trial)
                            value = float(r @ r / (2 * system.size))
                            accepted = value <= merit + 1e-4 * alpha * slope
                            reason = "accepted" if accepted else "Armijo"
                        except (ValueError, FloatingPointError) as exc:
                            reason = str(exc)
                        model["trials"].append(
                            {"alpha": alpha, "merit": value, "reason": reason, "accepted": accepted}
                        )
                        if accepted:
                            if active_lookahead and alpha < 0.1 and recovery < 7 and novel:
                                candidate = model["trials"][-1]
                                if backup is None or candidate["merit"] < backup[1]["merit"]:
                                    backup = (model, candidate, trial.copy())
                                candidate["accepted"] = False
                                candidate["reason"] = (
                                    "deferred feasible Armijo step for active lookahead"
                                )
                                accepted = False
                            break
                if accepted:
                    break
                # Pivot only violating lower complementarity constraints in the
                # full linear proposal; unchanged merit still governs acceptance.
                predicted = z + d
                updated = full.copy()
                updated[predicted[: system.n] < -1e-10] = False
                updated[predicted[system.n : 2 * system.n] < -1e-10] = True
                full = updated
            if not accepted and backup is not None:
                _, candidate, trial = backup
                candidate["accepted"] = True
                candidate["reason"] = "accepted safeguarded fallback after active lookahead"
                accepted = True
            if not accepted:
                raise RuntimeError("all feasible Armijo/active recovery models exhausted")
            z = trial
    except (RuntimeError, ValueError, FloatingPointError) as exc:
        diagnostics = getattr(exc, "diagnostics", {})
        raise PropertyLoadNewtonError(str(exc), history, last, diagnostics) from exc


@dataclass(frozen=True)
class PropertyLoadPath:
    root: WarmPropertyRoot
    attempts: tuple


class PropertyLoadPathError(RuntimeError):
    def __init__(self, message, attempts, last_root):
        super().__init__(message)
        self.attempts, self.last_root = tuple(attempts), last_root


def continue_property_load(
    problem,
    fluid,
    speed_m_s,
    target_load_n,
    *,
    seed,
    seed_offset_m,
    initial_increment_n,
    minimum_increment_n=1e-5,
    maximum_attempts=32,
    callback=None,
    preconditioner="frozen",
):
    """Adaptive SI-force steps; rejected solves never become continuation seeds."""
    if (
        not all(
            np.isfinite(v) and v > 0
            for v in (target_load_n, initial_increment_n, minimum_increment_n)
        )
        or not isinstance(maximum_attempts, int)
        or isinstance(maximum_attempts, bool)
        or not 1 <= maximum_attempts <= 60
        or (callback is not None and not callable(callback))
    ):
        raise ValueError("invalid bounded load continuation controls")
    system = PropertyLoadSystem(
        problem, fluid, speed_m_s, target_load_n, preconditioner=preconditioner
    )
    z = verify_seed(system, seed, seed_offset_m)
    current = system.solution(z)
    attempts = []
    step = initial_increment_n
    for _ in range(maximum_attempts):
        remaining = target_load_n - current.solution.load_n
        if abs(remaining) / target_load_n <= 1e-8:
            return PropertyLoadPath(current, tuple(attempts))
        amount = min(abs(remaining), step)
        if amount < minimum_increment_n and amount < abs(remaining):
            raise PropertyLoadPathError("minimum force increment reached", attempts, current)
        trial_load = (
            target_load_n
            if amount == abs(remaining)
            else current.solution.load_n + np.sign(remaining) * amount
        )
        record = {
            "from_load_n": current.solution.load_n,
            "target_load_n": float(trial_load),
            "accepted": False,
        }
        try:
            root = solve_property_load_newton(
                problem,
                fluid,
                speed_m_s,
                trial_load,
                seed=current.solution,
                seed_offset_m=current.central_film_offset_m,
                preconditioner=preconditioner,
            )
        except PropertyLoadNewtonError as exc:
            record["reason"] = str(exc)
            attempts.append(record)
            if callback is not None:
                callback(record, exc)
            step = amount / 2
            continue
        record.update(
            accepted=True,
            load_n=root.solution.load_n,
            offset_m=root.central_film_offset_m,
            audit=root.solution.audit,
        )
        attempts.append(record)
        if callback is not None:
            callback(record, root)
        current = root
        step = amount * 1.5
    if abs(current.solution.load_n - target_load_n) / target_load_n <= 1e-8:
        return PropertyLoadPath(current, tuple(attempts))
    raise PropertyLoadPathError("load continuation attempt budget exhausted", attempts, current)
