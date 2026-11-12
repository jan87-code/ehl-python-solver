"""Compare bounded Ciulli donor roots under alternative constitutive assumptions."""

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import scipy

from ehl_solver.donor_load import DonorLoadSystem
from ehl_solver.limited_audit import audit_limited_state
from ehl_solver.limited_newton import LimitedNewtonError, solve_limited_load
from ehl_solver.limited_system import LimitedLoadProbe
from ehl_solver.prototype import PointContactCase


class _BarusFluid:
    def __init__(self, case, *, constant_density):
        density_reference = case.fluid()
        self.viscosity_pa_s = case.viscosity_pa_s
        self.density_kg_m3 = case.density_kg_m3
        self.alpha_per_pa = case.alpha_per_pa
        self.density_a_per_pa = (
            0.0 if constant_density else density_reference.density_a_per_pa
        )
        self.density_b_per_pa = density_reference.density_b_per_pa
        self.roelands_pressure_pa = case.roelands_pressure_pa
        self.constant_density = constant_density

    def ratios(self, pressure_pa, *, strength=1.0):
        pressure = np.asarray(pressure_pa, dtype=np.float64)
        if (
            not np.all(np.isfinite(pressure))
            or np.any(pressure <= -0.5 * self.roelands_pressure_pa)
            or not np.isfinite(strength)
            or not 0 <= strength <= 1
        ):
            raise ValueError("Invalid Barus sensitivity pressure or strength")
        denominator = 1 + self.density_b_per_pa * pressure
        if np.any(denominator <= 0):
            raise ValueError("Density pole reached in Barus sensitivity model")
        density = 1 + strength * self.density_a_per_pa * pressure / denominator
        density_derivative = strength * self.density_a_per_pa / denominator**2
        with np.errstate(over="raise", invalid="raise"):
            viscosity = np.exp(strength * self.alpha_per_pa * pressure)
        viscosity_derivative = viscosity * strength * self.alpha_per_pa
        return density, viscosity, density_derivative, viscosity_derivative

    def properties(self, pressure_pa, *, strength=1.0):
        density, viscosity, density_derivative, viscosity_derivative = self.ratios(
            pressure_pa, strength=strength
        )
        return (
            self.density_kg_m3 * density,
            self.viscosity_pa_s * viscosity,
            self.density_kg_m3 * density_derivative,
            self.viscosity_pa_s * viscosity_derivative,
        )


class _ConstitutiveCase:
    def __init__(self, base, fluid):
        self._base = base
        self._fluid = fluid

    def __getattr__(self, name):
        return getattr(self._base, name)

    def fluid(self):
        return self._fluid

    def system(self, *, workspace_limit_bytes=2**30):
        return DonorLoadSystem(
            self._base.problem(),
            self._fluid,
            self._base.entrainment_m_s,
            self._base.load_n,
            preconditioner="local",
            elasticity_mode="cached",
            maximum_physical_nodes=self._base.maximum_physical_nodes,
            workspace_limit_bytes=workspace_limit_bytes,
        )


def _profile_rmse(fields, profile_csv):
    with Path(profile_csv).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = {}
    for direction in ("longitudinal", "transverse"):
        selected = [row for row in rows if row["direction"] == direction]
        coordinates = np.array([float(row["coordinate_um"]) for row in selected]) * 1e-6
        observed = np.array([float(row["film_thickness_um"]) for row in selected])
        if direction == "longitudinal":
            profile = np.array(
                [
                    np.interp(0.0, fields["y_m"], fields["film_m"][:, i])
                    for i in range(len(fields["x_m"]))
                ]
            )
            predicted = np.interp(coordinates, fields["x_m"], profile) * 1e6
        else:
            profile = np.array(
                [
                    np.interp(0.0, fields["x_m"], fields["film_m"][j])
                    for j in range(len(fields["y_m"]))
                ]
            )
            predicted = np.interp(coordinates, fields["y_m"], profile) * 1e6
        error = predicted - observed
        result[direction] = {
            "bias_nm": float(np.mean(error) * 1e3),
            "rmse_nm": float(np.sqrt(np.mean(error**2)) * 1e3),
            "maximum_absolute_error_nm": float(np.max(abs(error)) * 1e3),
        }
    return result


def _field_differences(fields, baseline):
    pressure_difference = fields["pressure_pa"] - baseline["pressure_pa"]
    pressure_mask = (fields["pressure_pa"] > 1e3) | (baseline["pressure_pa"] > 1e3)
    film_difference = fields["film_m"] - baseline["film_m"]
    return {
        "pressure_relative_l2": float(
            np.linalg.norm(pressure_difference[pressure_mask])
            / np.linalg.norm(baseline["pressure_pa"][pressure_mask])
        ),
        "pressure_rms_mpa": float(
            np.sqrt(np.mean(pressure_difference[pressure_mask] ** 2)) / 1e6
        ),
        "pressure_linf_mpa": float(
            np.max(abs(pressure_difference[pressure_mask])) / 1e6
        ),
        "film_relative_l2": float(
            np.linalg.norm(film_difference) / np.linalg.norm(baseline["film_m"])
        ),
        "film_rms_nm": float(np.sqrt(np.mean(film_difference**2)) * 1e9),
        "film_linf_nm": float(np.max(abs(film_difference)) * 1e9),
    }


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(case_path, baseline_directory, output_directory, profile_csv):
    started = time.perf_counter()
    requested_case = PointContactCase(**json.loads(Path(case_path).read_text()))
    baseline_directory = Path(baseline_directory)
    baseline_summary = json.loads((baseline_directory / "summary.json").read_text())
    if (
        baseline_summary.get("status") != "accepted_numerical_root"
        or baseline_summary.get("transport_fraction") != 0
        or baseline_summary.get("audit", {}).get("numerically_accepted") is not True
        or baseline_summary.get("state_sha256")
        != _digest(baseline_directory / "state.npz")
        or baseline_summary.get("case_sha256") != _digest(case_path)
    ):
        raise ValueError("Baseline must be an intact, audited donor-flux root")
    base = PointContactCase(**baseline_summary["case"])
    if base.physical_signature() != requested_case.physical_signature():
        raise ValueError("Baseline physical inputs do not match the requested Ciulli case")
    with np.load(baseline_directory / "state.npz", allow_pickle=False) as archive:
        initial_state = archive["state_dimensionless"].copy()
        baseline_fields = {
            name: archive[name].copy()
            for name in ("pressure_pa", "theta", "film_m", "x_m", "y_m")
        }
    expected_size = 2 * (base.nx - 2) * (base.ny - 2) + 1
    if (
        initial_state.ndim != 1
        or initial_state.size != expected_size
        or baseline_fields["film_m"].shape != (base.ny, base.nx)
        or baseline_fields["pressure_pa"].shape != (base.ny, base.nx)
        or baseline_fields["theta"].shape != (base.ny, base.nx)
    ):
        raise ValueError("Malformed baseline state")
    if output_directory.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_directory}")
    output_directory.mkdir(parents=True)

    baseline_fluid = base.fluid()
    variants = {
        "roelands_dh": (baseline_fluid, base),
        "roelands_constant_density": (
            replace(baseline_fluid, density_a_per_pa=0.0),
            None,
        ),
        "barus_dh": (_BarusFluid(base, constant_density=False), None),
        "barus_constant_density": (_BarusFluid(base, constant_density=True), None),
    }
    report = {
        "equation": "donor-flux JFO/Elrod-Adams",
        "grid_ny_nx": [base.ny, base.nx],
        "domain_half_width_mm": [
            base.domain_half_width_x_m * 1e3,
            base.domain_half_width_y_m * 1e3,
        ],
        "transport_fraction": 0.0,
        "baseline": str(baseline_directory),
        "case_sha256": _digest(case_path),
        "profile_sha256": _digest(profile_csv),
        "baseline_state_sha256": _digest(baseline_directory / "state.npz"),
        "runtime": {
            "python": sys.version,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
            "git_revision": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip(),
        },
        "source_sha256": {
            path.name: _digest(path)
            for path in (
                Path(__file__),
                Path(__import__("ehl_solver.donor_load", fromlist=["__file__"]).__file__),
                Path(__import__("ehl_solver.cavitating_newton", fromlist=["__file__"]).__file__),
                Path(__import__("ehl_solver.elasticity", fromlist=["__file__"]).__file__),
                Path(__import__("ehl_solver.limited_audit", fromlist=["__file__"]).__file__),
                Path(__import__("ehl_solver.limited_newton", fromlist=["__file__"]).__file__),
                Path(__import__("ehl_solver.limited_system", fromlist=["__file__"]).__file__),
                Path(__import__("ehl_solver.property_load_newton", fromlist=["__file__"]).__file__),
                Path(__import__("ehl_solver.property_warm", fromlist=["__file__"]).__file__),
                Path(__import__("ehl_solver.prototype", fromlist=["__file__"]).__file__),
                Path(__import__("ehl_solver.constitutive", fromlist=["__file__"]).__file__),
                Path(__import__("ehl_solver.transport_candidate", fromlist=["__file__"]).__file__),
            )
        },
        "solver_controls": {
            "maximum_iterations": 30,
            "maximum_krylov_cycles": 10,
            "transport_fraction": 0.0,
            "local_balance_tolerance": 1e-10,
        },
        "variants": {},
    }
    for name, (fluid, case) in variants.items():
        if case is None:
            case = _ConstitutiveCase(base, fluid)
        if name == "roelands_dh":
            fields = baseline_fields
            audit = baseline_summary["audit"]
            state = initial_state
            history = ()
        else:
            try:
                root = solve_limited_load(
                    case,
                    initial_state=initial_state,
                    transport_fraction=0.0,
                )
                probe = LimitedLoadProbe(case, transport_fraction=0.0)
                audit = audit_limited_state(probe, root.state, direct=True)
                if not audit["numerically_accepted"]:
                    raise RuntimeError("Independent direct endpoint audit rejected root")
                pressure, theta, film = probe.fields(root.state)
                state = root.state
                history = root.history
                fields = {
                    "pressure_pa": pressure,
                    "theta": theta,
                    "film_m": film,
                    "x_m": probe.system.problem.x_m,
                    "y_m": probe.system.problem.y_m,
                }
                with (output_directory / f"{name}.npz").open("wb") as stream:
                    np.savez_compressed(stream, state_dimensionless=state, **fields)
            except LimitedNewtonError as exc:
                probe = LimitedLoadProbe(case, transport_fraction=0.0)
                pressure, theta, film = probe.fields(exc.state)
                candidate_path = output_directory / f"{name}_candidate_unaccepted.npz"
                with candidate_path.open("wb") as stream:
                    np.savez_compressed(
                        stream,
                        state_dimensionless=exc.state,
                        pressure_pa=pressure,
                        theta=theta,
                        film_m=film,
                        x_m=probe.system.problem.x_m,
                        y_m=probe.system.problem.y_m,
                    )
                report["variants"][name] = {
                    "status": "rejected",
                    "failure_reason": str(exc),
                    "audit": exc.audit,
                    "history_records": len(exc.history),
                    "candidate_sha256": _digest(candidate_path),
                }
                continue
        pmax = float(np.max(fields["pressure_pa"]))
        center = len(fields["x_m"]) // 2
        report["variants"][name] = {
            "status": "accepted_numerical_root",
            "fluid": {
                "viscosity_model": "Barus" if name.startswith("barus") else "Roelands",
                "density_model": (
                    "constant"
                    if name.endswith("constant_density")
                    else "Dowson-Higginson"
                ),
                "viscosity_slope_per_pa": base.alpha_per_pa,
                "density_a_per_pa": getattr(fluid, "density_a_per_pa", None),
                "density_b_per_pa": getattr(fluid, "density_b_per_pa", None),
            },
            "audit": audit,
            "h_min_nm": float(np.min(fields["film_m"]) * 1e9),
            "h_c_nm": float(fields["film_m"][center, center] * 1e9),
            "p_max_mpa": pmax / 1e6,
            "profile_comparison": _profile_rmse(fields, profile_csv),
            "field_difference_from_baseline": _field_differences(
                fields, baseline_fields
            ),
            "history_records": len(history),
            "state_sha256": (
                _digest(output_directory / f"{name}.npz")
                if name != "roelands_dh"
                else report["baseline_state_sha256"]
            ),
        }
    report["elapsed_seconds"] = time.perf_counter() - started
    path = output_directory / "summary.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, default=Path("cases/ciulli_15n_donor.json"))
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument(
        "--profiles",
        type=Path,
        default=Path("cases/ciulli_2008_figure13_s0_profiles.csv"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run(args.case, args.baseline, args.output, args.profiles)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
