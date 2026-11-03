"""The generator's backend can drive the engine on a stock scenario."""

import json

from scenario_generation.atlatl_backend import AtlatlBackend
from scenario_generation.scenario_evaluation.utils import (
    convert_replay_to_json,
    get_replay_description,
    split_replay_reps,
)

from conftest import ENGINE_SCENARIOS


def test_engine_runs_stock_scenario(tmp_path):
    scenario = json.loads((ENGINE_SCENARIOS / "Scenario5Chokepoint.scn").read_text())

    replays = AtlatlBackend(timeout=300).run(scenario, tmp_path)

    assert replays.blue.exists() and replays.red.exists()
    blue_reps = split_replay_reps(convert_replay_to_json(replays.blue))
    red_reps = split_replay_reps(convert_replay_to_json(replays.red))
    assert len(blue_reps) >= 1 and len(red_reps) >= 1

    description, _ = get_replay_description(blue_reps[0], red_reps[0])
    assert "Phase" in description
