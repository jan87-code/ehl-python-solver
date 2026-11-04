import json

from ehl_solver import prototype_cli
from ehl_solver.property_load_newton import PropertyLoadNewtonError


def test_failure_without_valid_root_writes_report(tmp_path, monkeypatch):
    def fail_before_valid_root(*args, **kwargs):
        raise PropertyLoadNewtonError(
            "Newton failed before a valid root",
            history=[],
            last_root=None,
            diagnostics={"iteration": 0},
        )

    monkeypatch.setattr(prototype_cli, "_solve_property_load_state", fail_before_valid_root)

    output = tmp_path / "rejected"
    assert prototype_cli.main(["--output", str(output), "--no-plots"]) == 1

    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "rejected"
    assert summary["failure_reason"] == "Newton failed before a valid root"
    report = (output / "report.md").read_text()
    assert "No valid numerical root was produced" in report
    assert "Newton failed before a valid root" in report
    assert not (output / "solution.npz").exists()
    assert not (output / "fields.png").exists()
    assert not (output / "interactive_3d.html").exists()
