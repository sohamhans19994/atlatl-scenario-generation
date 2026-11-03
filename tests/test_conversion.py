"""Maps in the generator's text format become scenarios the engine accepts."""

import pytest

from scenario_generation.atlatl_backend import AtlatlBackend
from scenario_generation.utils import HexMapValidator, str_to_scenario

from conftest import load_example_tasks

EXAMPLES = load_example_tasks()


@pytest.mark.parametrize(
    "task", EXAMPLES, ids=[f"example{i + 1}" for i in range(len(EXAMPLES))]
)
def test_example_map_validates_converts_and_runs(task, tmp_path):
    validation = HexMapValidator.validate_map_text(task["Map String"])
    assert validation["ok"], validation["errors"]

    scenario = str_to_scenario(task["Map String"])
    assert scenario["map"]["hexes"]
    assert scenario["units"]

    replays = AtlatlBackend(timeout=300).run(scenario, tmp_path)
    assert replays.blue.stat().st_size > 0
    assert replays.red.stat().st_size > 0
