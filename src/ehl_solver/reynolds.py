"""Dimensionless residual for steady isoviscous-incompressible Reynolds flow."""

import math
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import cg, spsolve

from ehl_solver.backend import Backend, as_float64, make_backend


def _validated_film(
    film: ArrayLike,
    dx: float,
    dy: float,
    reynolds_parameter: float,
    source: ArrayLike | None,
    backend: Backend | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    backend = make_backend() if backend is None else backend
    xp = backend.xp
    h = as_float64(film, backend)
    if h.ndim != 2:
        raise ValueError("film must be a 2-D array with shape (ny, nx)")
    if min(h.shape) < 3:
        raise ValueError("film must have at least 3 cells per axis")
    if not bool(xp.isfinite(h).all()) or bool((h <= 0).any()):
        raise ValueError("film must contain only finite, positive values")
    if not math.isfinite(dx) or dx <= 0 or not math.isfinite(dy) or dy <= 0:
        raise ValueError("dx and dy must be finite and positive")
    if not math.isfinite(reynolds_parameter) or reynolds_parameter < 0:
        raise ValueError("reynolds_parameter must be finite and nonnegative")
    if source is None:
        manufactured_source = xp.zeros_like(h)
    else:
        manufactured_source = as_float64(source, backend)
        if manufactured_source.shape != h.shape:
            raise ValueError("source must have the same shape as film")
        if not bool(xp.isfinite(manufactured_source).all()):
            raise ValueError("source must contain only finite values")
    return h, manufactured_source


@dataclass(frozen=True)
class ReynoldsSolution:
    """Full-film pressure and independent linear and global-flux diagnostics.

    ``pressure`` is dimensionless and has the same ``(ny, nx)`` shape as the
    prescribed film, including zero-valued Dirichlet boundary nodes.
    """

    pressure: NDArray[np.float64]
    iterations: int
    algebraic_residual_l2: float
    algebraic_residual_inf: float
    algebraic_residual_relative_l2: float
    mass_balance_absolute_error: float
    mass_balance_relative_error: float


@dataclass(frozen=True)
class ReynoldsDiagnostics:
    """Physical pressure/load and discrete conservation diagnostics."""

    minimum_pressure_pa: float
    maximum_pressure_pa: float
    integrated_load_n: float
    normalized_algebraic_residual: float
    maximum_local_conservation_error: float
    maximum_local_conservation_error_relative: float
    west_boundary_flux: float
    east_boundary_flux: float
    south_boundary_flux: float
    north_boundary_flux: float
    net_boundary_flux: float


@dataclass(frozen=True)
class ReynoldsLinearSolverBenchmark:
    """Timing and residual comparison for one sparse linear solver."""

    method: str
    elapsed_seconds: float
    residual_l2: float
    relative_residual_l2: float
    pressure: NDArray[np.float64]
    iterations: int | None


def parabolic_point_contact_film(
    x_m: ArrayLike,
    y_m: ArrayLike,
    minimum_film_m: float,
    radius_x_m: float,
    radius_y_m: float,
    *,
    backend: Backend | None = None,
) -> NDArray[np.float64]:
    """Return the rigid parabolic point-contact film thickness in metres.

    ``x_m`` and ``y_m`` are same-shaped coordinate arrays in metres, with x
    along entrainment and y transverse. The geometry is
    ``h = h_min + x**2/(2*Rx) + y**2/(2*Ry)``. It represents prescribed rigid
    surfaces only; no pressure-induced deformation is included.
    """
    backend = make_backend() if backend is None else backend
    xp = backend.xp
    x = as_float64(x_m, backend)
    y = as_float64(y_m, backend)
    parameters = (minimum_film_m, radius_x_m, radius_y_m)
    if x.ndim != 2 or y.shape != x.shape:
        raise ValueError("x_m and y_m must be same-shaped 2-D coordinate arrays")
    if not bool(xp.isfinite(x).all()) or not bool(xp.isfinite(y).all()):
        raise ValueError("x_m and y_m must contain only finite coordinates")
    if not all(math.isfinite(value) and value > 0 for value in parameters):
        raise ValueError("minimum film and radii must be finite and positive")
    return (
        minimum_film_m
        + x**2 / (2.0 * radius_x_m)
        + y**2 / (2.0 * radius_y_m)
    )


def dimensionless_reynolds_parameter(
    viscosity_pa_s: float,
    surface_speed_1_m_s: float,
    surface_speed_2_m_s: float,
    length_scale_m: float,
    film_scale_m: float,
    pressure_scale_pa: float,
) -> float:
    """Compute Lambda using mean entrainment speed ``(u1 + u2)/2``.

    The returned dimensionless coefficient is
    ``12*eta0*Ue*L/(h_ref**2*p_ref)``, where ``Ue=(u1+u2)/2``.
    Equivalently, when speed is defined as the sum ``u1+u2``, the coefficient
    is ``6*eta0*(u1+u2)*L/(h_ref**2*p_ref)``. Positive speeds point in x.
    """
    values = (
        viscosity_pa_s,
        surface_speed_1_m_s,
        surface_speed_2_m_s,
        length_scale_m,
        film_scale_m,
        pressure_scale_pa,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Reynolds dimensional parameters must be finite")
    if viscosity_pa_s <= 0 or length_scale_m <= 0 or film_scale_m <= 0 or pressure_scale_pa <= 0:
        raise ValueError("viscosity and length, film, pressure scales must be positive")
    mean_speed = 0.5 * (surface_speed_1_m_s + surface_speed_2_m_s)
    if mean_speed < 0:
        raise ValueError("mean entrainment speed must be nonnegative")
    return (
        12.0
        * viscosity_pa_s
        * mean_speed
        * length_scale_m
        / (film_scale_m**2 * pressure_scale_pa)
    )


def assemble_reynolds_system(
    film: ArrayLike,
    dx: float,
    dy: float,
    reynolds_parameter: float,
    source: ArrayLike | None = None,
) -> tuple[csr_matrix, NDArray[np.float64]]:
    """Assemble the interior sparse system with homogeneous Dirichlet edges.

    The unknown vector follows row-major interior order: x varies fastest and
    the boundary pressure is fixed to zero on all four edges. The returned
    matrix is the positive-definite ``-div(H**3 grad(P))`` operator, and the
    right-hand side is ``-Lambda*dH/dX - S``. ``dx`` and ``dy`` are
    dimensionless spacings, and all values and coefficients use float64.

    The face conductance is the arithmetic mean of neighboring ``H**3``
    values, matching :func:`reynolds_residual`. The artificial ``source`` is
    useful for manufactured-solution tests; it is dimensionless and is not an
    additional physical model.
    """
    h, manufactured_source = _validated_film(
        film, dx, dy, reynolds_parameter, source
    )
    ny, nx = h.shape
    interior_nx = nx - 2
    interior_ny = ny - 2
    film_cubed = h**3
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    rhs = np.empty((interior_ny, interior_nx), dtype=np.float64)

    def append_entry(row: int, column: int, value: float) -> None:
        rows.append(row)
        columns.append(column)
        values.append(value)

    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            row = (j - 1) * interior_nx + (i - 1)
            center = film_cubed[j, i]
            coefficients = {
                "west": 0.5 * (center + film_cubed[j, i - 1]) / dx**2,
                "east": 0.5 * (center + film_cubed[j, i + 1]) / dx**2,
                "south": 0.5 * (center + film_cubed[j - 1, i]) / dy**2,
                "north": 0.5 * (center + film_cubed[j + 1, i]) / dy**2,
            }
            diagonal = sum(coefficients.values())
            append_entry(row, row, diagonal)
            for direction, neighbor_i, neighbor_j in (
                ("west", i - 1, j),
                ("east", i + 1, j),
                ("south", i, j - 1),
                ("north", i, j + 1),
            ):
                if 0 < neighbor_i < nx - 1 and 0 < neighbor_j < ny - 1:
                    neighbor = (neighbor_j - 1) * interior_nx + (neighbor_i - 1)
                    append_entry(row, neighbor, -coefficients[direction])

            film_derivative_x = (h[j, i + 1] - h[j, i - 1]) / (2 * dx)
            rhs[j - 1, i - 1] = (
                -reynolds_parameter * film_derivative_x
                - manufactured_source[j, i]
            )

    matrix = coo_matrix(
        (values, (rows, columns)),
        shape=(interior_nx * interior_ny, interior_nx * interior_ny),
        dtype=np.float64,
    ).tocsr()
    return matrix, rhs.ravel()


def reynolds_residual(
    pressure: ArrayLike,
    film: ArrayLike,
    dx: float,
    dy: float,
    reynolds_parameter: float,
    source: ArrayLike | None = None,
    *,
    backend: Backend | None = None,
) -> NDArray[np.float64]:
    """Return the conservative Reynolds-equation residual on interior cells.

    ``pressure`` and ``film`` are dimensionless arrays with shape ``(ny, nx)``;
    x is the entrainment direction and y is transverse. ``dx`` and ``dy`` are
    dimensionless grid spacings. All input cells, including boundary cells,
    contribute to the stencil, but residuals are returned only for the
    ``(ny - 2, nx - 2)`` interior cells. The residual function itself imposes
    no boundary condition.

    The equation is ``div(H**3 grad(P)) - Lambda*dH/dX - S = 0``, where
    ``Lambda = 12*eta0*Ue*L/(h_ref**2*p_ref)`` and
    ``Ue = (u1 + u2)/2`` is the mean surface velocity in x. ``source`` is an
    optional dimensionless manufactured source ``S``.

    Face values of ``H**3`` use arithmetic interpolation, pressure gradients
    use adjacent grid values, and divergence uses opposing face fluxes.
    The returned residual is dimensionless.

    An explicit torch backend returns a float64 tensor on its selected device.
    Field validation synchronizes scalar predicates; arrays are never copied
    back to the host. This ports the centered full-film stencil, not JFO upwinding.
    """
    backend = make_backend() if backend is None else backend
    xp = backend.xp
    p = as_float64(pressure, backend)
    h, manufactured_source = _validated_film(
        film, dx, dy, reynolds_parameter, source, backend
    )
    if p.ndim != 2:
        raise ValueError("pressure must be a 2-D array with shape (ny, nx)")
    if p.shape != h.shape:
        raise ValueError("pressure and film must have the same shape")
    if not bool(xp.isfinite(p).all()):
        raise ValueError("pressure must contain only finite values")

    interior = (slice(1, -1), slice(1, -1))
    film_cubed = h**3
    pressure_center = p[interior]
    film_center = film_cubed[interior]

    flux_east = (
        0.5 * (film_center + film_cubed[1:-1, 2:])
        * (p[1:-1, 2:] - pressure_center)
        / dx
    )
    flux_west = (
        0.5 * (film_center + film_cubed[1:-1, :-2])
        * (pressure_center - p[1:-1, :-2])
        / dx
    )
    flux_north = (
        0.5 * (film_center + film_cubed[2:, 1:-1])
        * (p[2:, 1:-1] - pressure_center)
        / dy
    )
    flux_south = (
        0.5 * (film_center + film_cubed[:-2, 1:-1])
        * (pressure_center - p[:-2, 1:-1])
        / dy
    )
    residual = (flux_east - flux_west) / dx + (flux_north - flux_south) / dy
    residual -= reynolds_parameter * (h[1:-1, 2:] - h[1:-1, :-2]) / (2 * dx)
    residual -= manufactured_source[interior]
    return residual


def solve_reynolds_pressure(
    film: ArrayLike,
    dx: float,
    dy: float,
    reynolds_parameter: float,
    source: ArrayLike | None = None,
    *,
    rtol: float = 1e-10,
    atol: float = 0.0,
    max_iterations: int | None = None,
    initial_pressure: ArrayLike | None = None,
    preconditioner: str | None = None,
) -> ReynoldsSolution:
    """Solve the unconstrained dimensionless full-film Reynolds problem.

    The prescribed film ``H`` is positive and fixed. The solver assumes a
    steady, isoviscous, incompressible, Newtonian lubricant, uses the existing
    mean-surface-speed convention in ``Lambda``, and imposes homogeneous
    Dirichlet pressure ``P=0`` on the rectangular domain edges. The uniform
    dimensionless grid spacings ``dx`` and ``dy`` and all arrays are in
    float64. Pressure is solved without clipping or other cavitation
    constraints; negative pressure, if produced by a full-film solution, is
    preserved.

    The sparse SPD system is solved by conjugate gradients. ``rtol`` and
    ``atol`` are passed to SciPy's solver for the algebraic convergence
    criterion ``||A p - b|| <= max(rtol*||b||, atol)``. Failure to converge
    raises ``RuntimeError`` rather than returning a success-shaped result.
    Optional ``preconditioner="multigrid"`` uses a symmetric geometric
    Galerkin V-cycle and independently verifies the true residual. Unsupported
    grid hierarchies raise explicitly; default unpreconditioned CG is preserved.
    The result reports algebraic residual norms and a separately computed
    global flux balance. No elasticity, load control, pressure-dependent
    properties, GPU acceleration, or cavitation model is included.
    """
    if preconditioner not in (None, "multigrid"):
        raise ValueError("preconditioner must be None or multigrid")
    if not math.isfinite(rtol) or rtol < 0 or not math.isfinite(atol) or atol < 0:
        raise ValueError("rtol and atol must be finite and nonnegative")
    if rtol == 0 and atol == 0:
        raise ValueError("at least one of rtol or atol must be positive")
    if max_iterations is not None and (
        not isinstance(max_iterations, int)
        or isinstance(max_iterations, bool)
        or max_iterations < 1
    ):
        raise ValueError("max_iterations must be a positive integer or None")

    h, manufactured_source = _validated_film(
        film, dx, dy, reynolds_parameter, source
    )
    initial_guess = None
    if initial_pressure is not None:
        initial_guess = np.asarray(initial_pressure, dtype=np.float64)
        if initial_guess.shape != h.shape or not np.all(np.isfinite(initial_guess)):
            raise ValueError("initial_pressure must be finite and have the same shape as film")
        if (
            np.any(initial_guess[0, :] != 0.0)
            or np.any(initial_guess[-1, :] != 0.0)
            or np.any(initial_guess[:, 0] != 0.0)
            or np.any(initial_guess[:, -1] != 0.0)
        ):
            raise ValueError("initial_pressure must satisfy the homogeneous Dirichlet boundary")
    matrix, rhs = assemble_reynolds_system(
        h, dx, dy, reynolds_parameter, manufactured_source
    )
    iterations = 0
    inverse = None
    if preconditioner == "multigrid":
        from ehl_solver.multigrid import ReynoldsMultigrid

        inverse = ReynoldsMultigrid(h, dx, dy).aslinearoperator()

    def count_iteration(iterate: NDArray[np.float64]) -> None:
        nonlocal iterations
        if not np.all(np.isfinite(iterate)):
            raise RuntimeError("Reynolds conjugate-gradient iteration became non-finite")
        iterations += 1

    interior_pressure, info = cg(
        matrix,
        rhs,
        x0=None if initial_guess is None else initial_guess[1:-1, 1:-1].ravel(),
        rtol=rtol,
        atol=atol,
        maxiter=max_iterations,
        callback=count_iteration,
        M=inverse,
    )
    if info != 0:
        reason = f"did not converge within {info} iterations" if info > 0 else "illegal input"
        raise RuntimeError(f"Reynolds conjugate-gradient solve {reason}")

    pressure = np.zeros(h.shape, dtype=np.float64)
    pressure[1:-1, 1:-1] = interior_pressure.reshape(h.shape[0] - 2, h.shape[1] - 2)
    algebraic_residual = matrix @ interior_pressure - rhs
    residual_l2 = float(np.linalg.norm(algebraic_residual))
    residual_inf = float(np.linalg.norm(algebraic_residual, ord=np.inf))
    rhs_l2 = float(np.linalg.norm(rhs))
    relative_residual_l2 = residual_l2 / max(rhs_l2, np.finfo(np.float64).tiny)
    if preconditioner == "multigrid" and residual_l2 > max(rtol*rhs_l2, atol)*(1+1e-12):
        raise RuntimeError("multigrid-CG failed independent true residual verification")
    mass_absolute, mass_relative = _global_mass_balance(
        pressure, h, dx, dy, reynolds_parameter, manufactured_source
    )

    return ReynoldsSolution(
        pressure=pressure,
        iterations=iterations,
        algebraic_residual_l2=residual_l2,
        algebraic_residual_inf=residual_inf,
        algebraic_residual_relative_l2=relative_residual_l2,
        mass_balance_absolute_error=mass_absolute,
        mass_balance_relative_error=mass_relative,
    )


def reynolds_diagnostics(
    solution: ReynoldsSolution,
    film: ArrayLike,
    dx: float,
    dy: float,
    reynolds_parameter: float,
    *,
    pressure_scale_pa: float,
    length_scale_m: float,
    source: ArrayLike | None = None,
) -> ReynoldsDiagnostics:
    """Report dimensional pressure/load and independent conservation checks.

    ``solution.pressure`` and ``film`` are dimensionless nodal arrays;
    ``pressure_scale_pa`` maps pressure to Pa, and ``length_scale_m`` maps
    both dimensionless grid coordinates to metres. Integrated load [N] uses
    tensor-product trapezoidal integration, including zero-pressure edges.
    Boundary fluxes are outward integrals of the dimensionless flux
    ``(-H**3 grad(P) + Lambda*H*e_x)`` over dimensionless edge lengths.

    Local conservation errors are control-volume-integrated balances for
    ``div(q) + S = 0``. Their units follow the dimensionless flux; absolute
    and locally normalized maxima are both reported. The algebraic residual
    is the relative L2 norm for the sparse system.
    """
    if not math.isfinite(pressure_scale_pa) or pressure_scale_pa <= 0:
        raise ValueError("pressure_scale_pa must be finite and positive")
    if not math.isfinite(length_scale_m) or length_scale_m <= 0:
        raise ValueError("length_scale_m must be finite and positive")
    h, manufactured_source = _validated_film(
        film, dx, dy, reynolds_parameter, source
    )
    pressure = np.asarray(solution.pressure, dtype=np.float64)
    if pressure.shape != h.shape or not np.all(np.isfinite(pressure)):
        raise ValueError("solution pressure must be finite and have the film shape")

    dimensional_pressure = pressure * pressure_scale_pa
    integrated_x = np.sum(
        0.5 * (dimensional_pressure[:, :-1] + dimensional_pressure[:, 1:]),
        axis=1,
    ) * dx * length_scale_m
    load_n = float(
        np.sum(0.5 * (integrated_x[:-1] + integrated_x[1:])) * dy * length_scale_m
    )
    local_absolute, local_relative = _local_conservation_errors(
        pressure, h, dx, dy, reynolds_parameter, manufactured_source
    )
    boundary_fluxes = _outward_boundary_fluxes(
        pressure, h, dx, dy, reynolds_parameter
    )
    return ReynoldsDiagnostics(
        minimum_pressure_pa=float(np.min(dimensional_pressure)),
        maximum_pressure_pa=float(np.max(dimensional_pressure)),
        integrated_load_n=load_n,
        normalized_algebraic_residual=solution.algebraic_residual_relative_l2,
        maximum_local_conservation_error=float(np.max(local_absolute)),
        maximum_local_conservation_error_relative=float(np.max(local_relative)),
        west_boundary_flux=boundary_fluxes[0],
        east_boundary_flux=boundary_fluxes[1],
        south_boundary_flux=boundary_fluxes[2],
        north_boundary_flux=boundary_fluxes[3],
        net_boundary_flux=float(sum(boundary_fluxes)),
    )


def compare_reynolds_linear_solvers(
    film: ArrayLike,
    dx: float,
    dy: float,
    reynolds_parameter: float,
    source: ArrayLike | None = None,
    *,
    rtol: float = 1e-10,
    atol: float = 0.0,
    max_iterations: int | None = None,
) -> tuple[ReynoldsLinearSolverBenchmark, ReynoldsLinearSolverBenchmark]:
    """Time SciPy direct sparse and conjugate-gradient solves on one system.

    Assembly and diagnostics are excluded from ``elapsed_seconds``; timing
    covers only the ``spsolve`` or ``cg`` call. Both measured residuals use the
    same assembled system and are returned for comparability. Timings are
    environment-dependent and should be recorded as benchmark observations,
    never used as pass/fail thresholds.
    """
    if not math.isfinite(rtol) or rtol < 0 or not math.isfinite(atol) or atol < 0:
        raise ValueError("rtol and atol must be finite and nonnegative")
    if rtol == 0 and atol == 0:
        raise ValueError("at least one of rtol or atol must be positive")
    if max_iterations is not None and (
        not isinstance(max_iterations, int)
        or isinstance(max_iterations, bool)
        or max_iterations < 1
    ):
        raise ValueError("max_iterations must be a positive integer or None")

    h, manufactured_source = _validated_film(
        film, dx, dy, reynolds_parameter, source
    )
    matrix, rhs = assemble_reynolds_system(
        h, dx, dy, reynolds_parameter, manufactured_source
    )

    start = perf_counter()
    direct_pressure = spsolve(matrix, rhs)
    direct_seconds = perf_counter() - start
    direct_result = _linear_solver_result(
        "direct", matrix, rhs, direct_pressure, direct_seconds, None
    )

    iterations = 0

    def count_iteration(iterate: NDArray[np.float64]) -> None:
        nonlocal iterations
        if not np.all(np.isfinite(iterate)):
            raise RuntimeError("Reynolds conjugate-gradient iteration became non-finite")
        iterations += 1

    start = perf_counter()
    iterative_pressure, info = cg(
        matrix,
        rhs,
        rtol=rtol,
        atol=atol,
        maxiter=max_iterations,
        callback=count_iteration,
    )
    iterative_seconds = perf_counter() - start
    if info != 0:
        reason = f"did not converge within {info} iterations" if info > 0 else "illegal input"
        raise RuntimeError(f"Reynolds conjugate-gradient benchmark {reason}")
    iterative_result = _linear_solver_result(
        "iterative", matrix, rhs, iterative_pressure, iterative_seconds, iterations
    )
    return direct_result, iterative_result


def _linear_solver_result(
    method: str,
    matrix: csr_matrix,
    rhs: NDArray[np.float64],
    pressure: NDArray[np.float64],
    elapsed_seconds: float,
    iterations: int | None,
) -> ReynoldsLinearSolverBenchmark:
    if not np.all(np.isfinite(pressure)):
        raise RuntimeError(f"Reynolds {method} solver returned non-finite pressure")
    residual = matrix @ pressure - rhs
    residual_l2 = float(np.linalg.norm(residual))
    rhs_l2 = float(np.linalg.norm(rhs))
    relative_residual = residual_l2 / max(rhs_l2, np.finfo(np.float64).tiny)
    return ReynoldsLinearSolverBenchmark(
        method=method,
        elapsed_seconds=elapsed_seconds,
        residual_l2=residual_l2,
        relative_residual_l2=relative_residual,
        pressure=pressure,
        iterations=iterations,
    )


def _outward_boundary_fluxes(
    pressure: NDArray[np.float64],
    film: NDArray[np.float64],
    dx: float,
    dy: float,
    reynolds_parameter: float,
) -> tuple[float, float, float, float]:
    film_cubed = film**3
    west_qx = (
        -0.5 * (film_cubed[1:-1, 0] + film_cubed[1:-1, 1])
        * (pressure[1:-1, 1] - pressure[1:-1, 0])
        / dx
        + reynolds_parameter * 0.5 * (film[1:-1, 0] + film[1:-1, 1])
    )
    east_qx = (
        -0.5 * (film_cubed[1:-1, -2] + film_cubed[1:-1, -1])
        * (pressure[1:-1, -1] - pressure[1:-1, -2])
        / dx
        + reynolds_parameter * 0.5 * (film[1:-1, -2] + film[1:-1, -1])
    )
    south_qy = (
        -0.5 * (film_cubed[0, 1:-1] + film_cubed[1, 1:-1])
        * (pressure[1, 1:-1] - pressure[0, 1:-1])
        / dy
    )
    north_qy = (
        -0.5 * (film_cubed[-2, 1:-1] + film_cubed[-1, 1:-1])
        * (pressure[-1, 1:-1] - pressure[-2, 1:-1])
        / dy
    )
    return (
        -float(np.sum(west_qx)) * dy,
        float(np.sum(east_qx)) * dy,
        -float(np.sum(south_qy)) * dx,
        float(np.sum(north_qy)) * dx,
    )


def _local_conservation_errors(
    pressure: NDArray[np.float64],
    film: NDArray[np.float64],
    dx: float,
    dy: float,
    reynolds_parameter: float,
    source: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    film_cubed = film**3
    center = film[1:-1, 1:-1]
    flux_east = (
        -0.5 * (film_cubed[1:-1, 1:-1] + film_cubed[1:-1, 2:])
        * (pressure[1:-1, 2:] - pressure[1:-1, 1:-1])
        / dx
        + reynolds_parameter * 0.5 * (center + film[1:-1, 2:])
    )
    flux_west = (
        -0.5 * (film_cubed[1:-1, 1:-1] + film_cubed[1:-1, :-2])
        * (pressure[1:-1, 1:-1] - pressure[1:-1, :-2])
        / dx
        + reynolds_parameter * 0.5 * (center + film[1:-1, :-2])
    )
    flux_north = (
        -0.5 * (film_cubed[1:-1, 1:-1] + film_cubed[2:, 1:-1])
        * (pressure[2:, 1:-1] - pressure[1:-1, 1:-1])
        / dy
    )
    flux_south = (
        -0.5 * (film_cubed[1:-1, 1:-1] + film_cubed[:-2, 1:-1])
        * (pressure[1:-1, 1:-1] - pressure[:-2, 1:-1])
        / dy
    )
    source_term = source[1:-1, 1:-1]
    integrated_error = (
        (flux_east - flux_west) * dy
        + (flux_north - flux_south) * dx
        + source_term * dx * dy
    )
    local_scale = (
        (np.abs(flux_east) + np.abs(flux_west)) * dy
        + (np.abs(flux_north) + np.abs(flux_south)) * dx
        + np.abs(source_term) * dx * dy
    )
    relative_error = np.abs(integrated_error) / np.maximum(
        local_scale, np.finfo(np.float64).tiny
    )
    return np.abs(integrated_error), relative_error


def _global_mass_balance(
    pressure: NDArray[np.float64],
    film: NDArray[np.float64],
    dx: float,
    dy: float,
    reynolds_parameter: float,
    source: NDArray[np.float64],
) -> tuple[float, float]:
    """Integrate outward flux and source over interior control volumes."""
    film_cubed = film**3
    west_flux = (
        0.5 * (film_cubed[1:-1, 0] + film_cubed[1:-1, 1])
        * (pressure[1:-1, 1] - pressure[1:-1, 0])
        / dx
        - reynolds_parameter * 0.5 * (film[1:-1, 0] + film[1:-1, 1])
    )
    east_flux = (
        0.5 * (film_cubed[1:-1, -2] + film_cubed[1:-1, -1])
        * (pressure[1:-1, -1] - pressure[1:-1, -2])
        / dx
        - reynolds_parameter * 0.5 * (film[1:-1, -2] + film[1:-1, -1])
    )
    south_flux = (
        0.5 * (film_cubed[0, 1:-1] + film_cubed[1, 1:-1])
        * (pressure[1, 1:-1] - pressure[0, 1:-1])
        / dy
    )
    north_flux = (
        0.5 * (film_cubed[-2, 1:-1] + film_cubed[-1, 1:-1])
        * (pressure[-1, 1:-1] - pressure[-2, 1:-1])
        / dy
    )
    outward_flux = (
        -float(np.sum(west_flux)) * dy
        + float(np.sum(east_flux)) * dy
        - float(np.sum(south_flux)) * dx
        + float(np.sum(north_flux)) * dx
    )
    integrated_source = float(np.sum(source[1:-1, 1:-1])) * dx * dy
    absolute_error = outward_flux - integrated_source
    scale = (
        float(np.sum(np.abs(west_flux)) + np.sum(np.abs(east_flux))) * dy
        + float(np.sum(np.abs(south_flux)) + np.sum(np.abs(north_flux))) * dx
        + float(np.sum(np.abs(source[1:-1, 1:-1]))) * dx * dy
    )
    relative_error = abs(absolute_error) / max(scale, np.finfo(np.float64).tiny)
    return absolute_error, relative_error
