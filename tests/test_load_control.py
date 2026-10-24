import matplotlib
import numpy as np
import pytest
from scipy.integrate import trapezoid

matplotlib.use("Agg")

from ehl_solver.coupled import solve_coupled_reynolds_elasticity
from ehl_solver.load_control import solve_full_film_at_load
from ehl_solver.reynolds import dimensionless_reynolds_parameter
from ehl_solver.visualization import plot_load_control_history

VISCOSITY_PA_S = 0.04
SPEED_1_M_S = 0.85
LENGTH_SCALE_M = 5e-4
FILM_SCALE_M = 20e-6
PRESSURE_SCALE_PA = (
    VISCOSITY_PA_S * SPEED_1_M_S * 0.5 * LENGTH_SCALE_M / FILM_SCALE_M**2
)
REYNOLDS_PARAMETER = dimensionless_reynolds_parameter(
    VISCOSITY_PA_S,
    SPEED_1_M_S,
    0.0,
    LENGTH_SCALE_M,
    FILM_SCALE_M,
    PRESSURE_SCALE_PA,
)


def _grid():
    return (
        np.linspace(-0.5e-3, 0.75e-3, 17),
        np.linspace(-0.5e-3, 0.5e-3, 13),
    )


def _coupling_options():
    return {
        "continuation_factors": (0.0,),
        "pressure_relaxation": 1.0,
        "film_relaxation": 1.0,
        "pressure_tolerance": 1e-8,
        "film_tolerance": 1e-8,
        "reynolds_tolerance": 1e-10,
        "inner_rtol": 1e-12,
        "inner_atol": 1e-14,
    }


def _known_load(h0_m):
    x, y = _grid()
    state = solve_coupled_reynolds_elasticity(
        x,
        y,
        h0_m,
        0.01,
        0.02,
        1e8,
        LENGTH_SCALE_M,
        FILM_SCALE_M,
        PRESSURE_SCALE_PA,
        REYNOLDS_PARAMETER,
        **_coupling_options(),
    )
    return float(trapezoid(trapezoid(state.pressure_pa, x, axis=1), y))


def test_signed_full_film_load_root_recovers_known_offset_and_load():
    known_h0 = 20e-6
    target_load = _known_load(known_h0)
    x, y = _grid()
    result = solve_full_film_at_load(
        x,
        y,
        target_load,
        0.01,
        0.02,
        1e8,
        LENGTH_SCALE_M,
        FILM_SCALE_M,
        PRESSURE_SCALE_PA,
        REYNOLDS_PARAMETER,
        h0_bracket_m=(15e-6, 25e-6),
        load_tolerance=1e-6,
        h0_tolerance_m=1e-12,
        coupling_options=_coupling_options(),
    )

    assert result.central_film_offset_m == pytest.approx(known_h0, abs=2e-10)
    assert result.calculated_signed_load_n == pytest.approx(target_load, rel=1e-6)
    assert result.relative_load_error <= 1e-6
    assert len(result.evaluations) >= 5
    assert all(item.nonlinear_iterations > 0 for item in result.evaluations)
    assert result.coupled_solution.pressure_pa.min() < 0.0
    figure, axes = plot_load_control_history(result)
    assert axes[0].get_xlabel() == "Central film offset h0 [µm]"
    assert axes[0].get_ylabel() == "Signed pressure integral [N]"
    assert axes[1].get_ylabel() == "Relative signed-load error"
    figure.canvas.draw()


def test_load_controller_rejects_unbracketed_or_nonmonotone_targets():
    x, y = _grid()
    common = (
        x,
        y,
        -100.0,
        0.01,
        0.02,
        1e8,
        LENGTH_SCALE_M,
        FILM_SCALE_M,
        PRESSURE_SCALE_PA,
        REYNOLDS_PARAMETER,
    )
    with pytest.raises(ValueError, match="does not bracket"):
        solve_full_film_at_load(
            *common,
            h0_bracket_m=(15e-6, 25e-6),
            coupling_options=_coupling_options(),
        )


def test_load_root_does_not_accept_unconverged_coupled_states():
    x, y = _grid()
    with pytest.raises(RuntimeError, match="failed at compliance_factor"):
        solve_full_film_at_load(
            x,
            y,
            -0.003,
            0.01,
            0.02,
            1e8,
            LENGTH_SCALE_M,
            FILM_SCALE_M,
            PRESSURE_SCALE_PA,
            REYNOLDS_PARAMETER,
            h0_bracket_m=(15e-6, 25e-6),
            coupling_options={"max_iterations_per_stage": 1},
        )
