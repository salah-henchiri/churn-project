"""Consistency of dvc.yaml with the code and with params.yaml.

A stage that forgets one of its dependencies is the most common DVC mistake:
the stage is then not re-run when that file changes, and results go stale.
"""

from pathlib import Path

import pytest
import yaml

PIPELINE = yaml.safe_load(Path("dvc.yaml").read_text())["stages"]
PARAMS = yaml.safe_load(Path("params.yaml").read_text())


def lookup(params: dict, dotted: str):
    value = params
    for key in dotted.split("."):
        value = value[key]
    return value


@pytest.mark.parametrize("stage", sorted(PIPELINE))
def test_declared_params_exist(stage):
    for name in PIPELINE[stage].get("params", []):
        lookup(PARAMS, name)  # raises KeyError if the parameter does not exist


@pytest.mark.parametrize("stage", sorted(PIPELINE))
def test_stage_depends_on_its_own_module(stage):
    deps = PIPELINE[stage]["deps"]
    assert f"src/churn_project/{stage}.py" in deps


def test_stages_are_chained():
    outs = {
        o if isinstance(o, str) else next(iter(o))
        for s in PIPELINE.values()
        for o in s.get("outs", [])
    }
    assert "data/prepared" in outs
    assert "models/model.skops" in outs
    assert "models/model.skops" in PIPELINE["evaluate"]["deps"]
