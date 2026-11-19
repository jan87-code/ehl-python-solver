import numpy as np
import pytest

from ehl_solver.backend import make_backend
from ehl_solver.constitutive import (
    IsothermalFluid,
    roelands_index_from_ambient_slope,
)
from ehl_solver.geometry import LineContactGeometry
from ehl_solver.hertz import line_contact_hertz
from ehl_solver.line_contact import PlaneStrainLineElasticityOperator
from ehl_solver.line_ehl import (
    LineContactEHLCase,
    LineEHLLoadConvergenceError,
    analyze_line_ehl_features,
    audit_line_elasticity_direct,
    line_contact_ehl_case_from_config,
    solve_line_ehl_at_load,
    solve_line_ehl_at_offset,
)

LINE_LOAD_N_M = 1000.0
RADIUS_M = 0.01
E_STAR_PA = 113e9
ETA0_PA_S = 0.04
SPEED_M_S = 0.1
HERTZ = line_contact_hertz(LINE_LOAD_N_M, RADIUS_M, E_STAR_PA, 1e-3)
HALF_DOMAIN_M = 20 * HERTZ.half_width_m
FILM_SCALE_M = 0.25e-6
FLUID = IsothermalFluid(
    ETA0_PA_S,
    850.0,
    roelands_index_from_ambient_slope(2.21e-8, ETA0_PA_S),
)


def _case(grid_size=65):
    x_m = np.linspace(-HALF_DOMAIN_M, HALF_DOMAIN_M, grid_size)
    return LineContactEHLCase(
        x_m=x_m,
        geometry=LineContactGeometry(RADIUS_M),
        line_load_n_m=LINE_LOAD_N_M,
        e_star_pa=E_STAR_PA,
        fluid=FLUID,
        entrainment_speed_m_s=SPEED_M_S,
        length_scale_m=HERTZ.half_width_m,
        film_scale_m=FILM_SCALE_M,
        pressure_scale_pa=HERTZ.peak_pressure_pa,
        elastic_reference_length_m=HALF_DOMAIN_M,
    )


def _coupling_options():
    return {
        "compliance_factors": (0.0, 1.0),
        "maximum_iterations_per_stage": 100,
        "film_relaxation": 0.3,
    }


def test_line_ehl_config_uses_explicit_n_per_m_and_rejects_total_force():
    config = {
        "type": "classical_line_ehl",
        "x_m": np.linspace(-1e-4, 1e-4, 9),
        "geometry": {"type": "classical_line", "radius_x_m": RADIUS_M},
        "line_load_n_m": LINE_LOAD_N_M,
        "e_star_pa": E_STAR_PA,
        "fluid": {"viscosity_pa_s": ETA0_PA_S, "density_kg_m3": 850.0},
        "entrainment_speed_m_s": SPEED_M_S,
        "length_scale_m": 1e-5,
        "film_scale_m": FILM_SCALE_M,
        "pressure_scale_pa": HERTZ.peak_pressure_pa,
        "elastic_reference_length_m": 1e-4,
    }

    case = line_contact_ehl_case_from_config(config)
    assert case.line_load_n_m == LINE_LOAD_N_M
    assert not hasattr(case, "load_n")

    invalid = dict(config)
    invalid["load_n"] = LINE_LOAD_N_M
    with pytest.raises(ValueError, match="line_load_n_m"):
        line_contact_ehl_case_from_config(invalid)

    invalid = dict(config)
    invalid["geometry"] = {"type": "circular_point", "radius_m": RADIUS_M}
    with pytest.raises(ValueError, match="classical_line"):
        line_contact_ehl_case_from_config(invalid)


def test_coupled_line_ehl_closes_n_per_m_and_passes_independent_audits():
    case = _case()
    root = solve_line_ehl_at_load(
        case,
        (0.1e-6, 1.0e-6),
        load_tolerance=1e-6,
        coupling_options=_coupling_options(),
    )
    solution = root.coupled_solution
    direct_operator = PlaneStrainLineElasticityOperator(
        case.x_m,
        case.e_star_pa,
        case.elastic_reference_length_m,
    )
    direct_displacement = audit_line_elasticity_direct(
        solution.pressure_pa,
        direct_operator,
    )
    reconstructed_film = (
        root.central_film_offset_m
        + case.geometry_m
        + direct_displacement
    )

    assert root.target_line_load_n_m == LINE_LOAD_N_M
    assert root.relative_load_error < 1e-6
    assert solution.calculated_line_load_n_m == pytest.approx(
        LINE_LOAD_N_M, rel=1e-6
    )
    assert solution.audit["mass_maximum"] < 1e-8
    assert solution.audit["relative_local_balance"] < 1e-10
    assert solution.audit["relative_global_balance"] < 1e-10
    assert solution.audit["maximum_complementarity"] < 1e-10
    assert solution.audit["boundary_pressure_maximum_pa"] == 0.0
    assert solution.audit["minimum_pressure_pa"] >= 0.0
    assert solution.audit["minimum_fractional_film_content"] >= 0.0
    assert solution.audit["maximum_fractional_film_content"] <= 1.0
    assert np.all(solution.film_thickness_m > 0.0)
    np.testing.assert_allclose(
        solution.elastic_displacement_m,
        direct_displacement,
        rtol=2e-12,
        atol=1e-22,
    )
    np.testing.assert_allclose(
        solution.film_thickness_m,
        reconstructed_film,
        rtol=0.0,
        atol=1e-20,
    )
    assert solution.film_thickness_m[case.x_m.size // 2] > 0.0


def test_rejected_load_bracket_preserves_converged_h0_trials():
    with pytest.raises(LineEHLLoadConvergenceError) as error:
        solve_line_ehl_at_load(
            _case(),
            (2.0e-6, 3.0e-6),
            coupling_options=_coupling_options(),
        )

    assert len(error.value.trials) == 3
    assert error.value.last_solution is not None


def test_symmetric_pressure_has_symmetric_plane_strain_deformation():
    case = _case()
    pressure = 5e7 * np.exp(-((case.x_m / HERTZ.half_width_m) ** 2))
    operator = PlaneStrainLineElasticityOperator(
        case.x_m,
        case.e_star_pa,
        case.elastic_reference_length_m,
    )

    displacement = operator(pressure)
    direct = audit_line_elasticity_direct(pressure, operator)

    np.testing.assert_allclose(displacement, direct, rtol=2e-12, atol=1e-22)
    np.testing.assert_allclose(displacement, displacement[::-1], rtol=2e-12, atol=1e-22)


def test_line_ehl_features_report_bulk_and_optional_outlet_spike():
    x = np.linspace(-10.0, 10.0, 401)
    pressure = np.exp(-(x / 2.0) ** 2) + 0.4 * np.exp(-((x - 4.0) / 0.3) ** 2)
    film = 1.0 + 0.02 * x**2 + 0.1 * np.exp(-((x - 4.0) / 0.3) ** 2)

    features = analyze_line_ehl_features(x, pressure, film, 1.0)

    assert features.central_film_m == pytest.approx(film[200])
    assert features.minimum_film_m == pytest.approx(film.min())
    assert features.petrusevich_spike_location_m == pytest.approx(4.0, abs=0.05)
    assert features.petrusevich_spike_width_m is not None
    assert features.petrusevich_spike_integrated_pressure_pa_m is not None


def test_line_ehl_spike_detector_includes_published_x_over_b_below_one():
    x = np.linspace(-2.0, 1.5, 351)
    pressure = np.exp(-(x / 0.55) ** 2) + 0.4 * np.exp(
        -((x - 0.9) / 0.04) ** 2
    )
    film = np.ones_like(x)

    features = analyze_line_ehl_features(x, pressure, film, 1.0)

    assert features.petrusevich_spike_location_m == pytest.approx(0.9)
    assert features.petrusevich_spike_width_m is not None
    assert features.petrusevich_spike_integrated_pressure_pa_m is not None


def test_cuda_line_ehl_matches_numpy_when_available():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("No CUDA or ROCm GPU available")
    case = _case()
    cpu = solve_line_ehl_at_offset(
        case,
        0.17e-6,
        backend=make_backend("numpy"),
        **_coupling_options(),
    )
    cuda = solve_line_ehl_at_offset(
        case,
        0.17e-6,
        backend=make_backend("torch", "cuda"),
        **_coupling_options(),
    )

    assert cuda.backend.startswith(("NVIDIA CUDA", "AMD ROCm"))
    np.testing.assert_allclose(cuda.pressure_pa, cpu.pressure_pa, rtol=2e-9, atol=1e-8)
    np.testing.assert_allclose(
        cuda.film_thickness_m, cpu.film_thickness_m, rtol=2e-9, atol=1e-18
    )
    assert cuda.calculated_line_load_n_m == pytest.approx(
        cpu.calculated_line_load_n_m, rel=2e-9
    )
