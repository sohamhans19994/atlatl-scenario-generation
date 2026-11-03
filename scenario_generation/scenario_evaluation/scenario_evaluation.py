"""Evaluate generated scenarios using an installed AtlAtl backend and an LLM."""

from __future__ import annotations

import tempfile
from importlib.resources import files
from pathlib import Path
from typing import Any

from jinja2 import Template

from scenario_generation.atlatl_backend import AtlatlBackend
from scenario_generation.run_logging import RunLogger
from scenario_generation.scenario_evaluation.utils import (
    convert_replay_to_json,
    get_initial_state_board,
    get_queries_description,
    get_replay_description,
    split_replay_reps,
    validator_replay_analysis,
)
from scenario_generation.utils import safe_llm_complete, str_to_scenario


class ScenarioEvaluation:
    def __init__(
        self,
        logging: RunLogger | None = None,
        *,
        llm: Any,
        backend: AtlatlBackend | None = None,
    ):
        prompt_path = files("scenario_generation.scenario_evaluation").joinpath(
            "prompts/replay_evaluator.txt"
        )
        self.replay_evaluator_prompt = Template(prompt_path.read_text(encoding="utf-8"))
        self.llm = llm
        self.logging = logging
        self.backend = backend or AtlatlBackend()

    def evaluate_generated_map(
        self,
        generated_map_str: str,
        evaluation_objective: dict[str, Any],
        map_id: str | None = None,
    ) -> tuple[dict[str, Any], float]:
        if self.logging is not None and map_id is None:
            raise ValueError("map_id is required when evaluation logging is enabled")

        scenario = str_to_scenario(generated_map_str)
        temp_root = self.logging.run_dir if self.logging else None
        with tempfile.TemporaryDirectory(
            prefix="atlatl-evaluation-", dir=temp_root
        ) as temporary_directory:
            replay_paths = self.backend.run(scenario, Path(temporary_directory))
            if self.logging is not None:
                self.logging.log_replay_runs(
                    map_id, replay_paths.blue, replay_paths.red
                )
            result = self.evaluate_replay(
                replay_paths.blue,
                replay_paths.red,
                evaluation_objective,
                map_id=map_id,
            )

        if not result["per_replay_analysis"]:
            raise RuntimeError("AtlAtl produced no replay repetitions to evaluate")
        first_result = result["per_replay_analysis"][0]
        if first_result is None:
            raise RuntimeError("The LLM replay evaluator did not return a valid result")
        return first_result, self.evaluation_scorer(
            first_result, evaluation_objective
        )

    @staticmethod
    def evaluation_scorer(
        evaluation_result: dict[str, Any],
        evaluation_objective: dict[str, Any],
    ) -> float:
        queries = evaluation_objective["Queries"]
        answers = {answer["qid"]: answer["answer"] for answer in evaluation_result["answers"]}
        total_score = 0.0
        maximum_score = 0.0

        for index, query in enumerate(queries, start=1):
            if query["Answerdatatype"] == "bool":
                maximum_score += 1
                if answers.get(index) is True:
                    total_score += 1
                continue

            if "Maxscore" not in query:
                raise ValueError(f"Maxscore is missing for query {query['Query']!r}")
            query_maximum = float(query["Maxscore"])
            maximum_score += query_maximum
            if answers.get(index) == query.get("Answer"):
                total_score += query_maximum

        if maximum_score == 0:
            raise ValueError("At least one scored evaluation query is required")
        return total_score / maximum_score

    def evaluate_replay(
        self,
        blue_replay_path: str | Path,
        red_replay_path: str | Path,
        evaluation_objective: dict[str, Any],
        *,
        map_id: str | None = None,
    ) -> dict[str, Any]:
        blue_repetitions = split_replay_reps(
            convert_replay_to_json(blue_replay_path)
        )
        red_repetitions = split_replay_reps(
            convert_replay_to_json(red_replay_path)
        )

        analyses: list[dict[str, Any] | None] = []
        llm_outputs: list[dict[str, Any]] = []
        for blue_rep, red_rep in zip(blue_repetitions, red_repetitions):
            analysis, llm_output = self.evaluate_replay_rep(
                blue_rep, red_rep, evaluation_objective
            )
            analyses.append(analysis)
            llm_outputs.append(llm_output)

        result = {"per_replay_analysis": analyses}
        if self.logging is not None:
            if map_id is None:
                raise ValueError("map_id is required when evaluation logging is enabled")
            self.logging.log_evaluation_calls(map_id, llm_outputs, result)
        return result

    def evaluate_replay_rep(
        self,
        blue_replay_rep: list[dict[str, Any]],
        red_replay_rep: list[dict[str, Any]],
        evaluation_objective: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        replay_description, grouped_replay = get_replay_description(
            blue_replay_rep, red_replay_rep
        )
        initial_board = get_initial_state_board(grouped_replay)
        queries = evaluation_objective["Queries"]
        prompt = self.replay_evaluator_prompt.render(
            hex_map=initial_board,
            replay_description=replay_description,
            queries=get_queries_description(queries),
            learning_objective=(
                f"Learning Objective: {evaluation_objective['Learning Objective']}\n"
                f"Strategy Expected: {evaluation_objective['Strategy Expected']}"
            ),
        )
        result = safe_llm_complete(
            prompt,
            llm=self.llm,
            validator=validator_replay_analysis,
            queries=queries,
        )
        if result["ok"]:
            return result["validation_result"]["response_json"], result
        return None, result


# Preserve the original method names for existing notebooks.
ScenarioEvaluation.evaluateReplay = ScenarioEvaluation.evaluate_replay
ScenarioEvaluation.evaluateReplayRep = ScenarioEvaluation.evaluate_replay_rep
