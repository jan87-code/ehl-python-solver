"""Shared independently solved MC path; no ignored or MATLAB artifacts required."""

import json
from pathlib import Path

import pytest

from ehl_solver.limited_continuation import continue_limited_transport
from ehl_solver.prototype import PointContactCase


@pytest.fixture(scope="session")
def accepted_mc_path():
    case = PointContactCase(
        **json.loads((Path(__file__).parents[1] / "cases/circular_50n.json").read_text())
    )
    stages = []
    path = continue_limited_transport(case, stage_observer=lambda i, c, t, r: stages.append(r))
    return path, stages
