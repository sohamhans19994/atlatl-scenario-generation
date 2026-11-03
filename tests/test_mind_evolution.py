"""Full Mind Evolution loop with a fake LLM and the real engine."""

import json
import random

import pytest

from scenario_generation.llm_client import LLMConfig
from scenario_generation.mind_evolution.mind_evolution import MindEvolution

TINY_CONFIG = {
    "N_generations": 2,
    "N_islands": 2,
    "N_convs": 1,
    "N_seqs": 2,
    "N_reset_interval": 100,
    "N_reset": 1,
    "N_top": 1,
    "N_parents": 1,
    "Pr_no_parents": 0.0,
    "N_emigrate": 1,
}


def make_experiment(llm, objective, output_dir):
    llm_config = LLMConfig(provider="openai", model="fake")
    return MindEvolution(
        dict(TINY_CONFIG),
        objective,
        llm=llm,
        llm_config=llm_config,
        structured_llm=llm,
        structured_llm_config=llm_config.for_structured_output(),
        output_dir=output_dir,
        run_desc="test",
    )


def test_evolution_runs_to_completion(fake_llm_factory, objective, tmp_path):
    random.seed(0)
    llm = fake_llm_factory(answers=[True, False, True, False])
    experiment = make_experiment(llm, objective, tmp_path)

    experiment.evolve()
    experiment.save_evolution_results()

    run_dir = experiment.logging.run_dir
    for name in (
        "evolution_graph.json",
        "island_evolutions.json",
        "evolution_results.json",
        "llm_results.jsonl",
        "llm_config.json",
    ):
        assert (run_dir / name).exists(), name

    assert len(experiment.islands) == TINY_CONFIG["N_islands"]
    assert all(experiment.islands)
    for island in experiment.islands:
        for candidate in island:
            assert candidate["score"] == pytest.approx(0.5)

    graph = json.loads((run_dir / "evolution_graph.json").read_text())
    assert len(graph["generations"]) == TINY_CONFIG["N_generations"]
    # 2 islands x (1 conv + 1 RCC) at init, then 2 islands x 1 conv x 2 seqs.
    assert llm.calls["generate"] == 8
    assert llm.calls["evaluate"] == 8


def test_evolution_stops_after_three_perfect_candidates(
    fake_llm_factory, objective, tmp_path
):
    random.seed(0)
    llm = fake_llm_factory(answers=[True, True, True, True])
    experiment = make_experiment(llm, objective, tmp_path)

    with pytest.raises(SystemExit) as stopped:
        experiment.evolve()

    assert stopped.value.code == 0
    assert experiment.num_perfect_solutions == 3
    assert llm.calls["evaluate"] == 3
    assert "Max number of perfect solutions reached" in (
        experiment.logging.run_dir / "events.log"
    ).read_text()
