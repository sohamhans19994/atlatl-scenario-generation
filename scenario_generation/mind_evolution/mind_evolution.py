"""Mind Evolution driver for LLM-generated AtlAtl scenarios.

The evolutionary search is a domain-specific adaptation of Mind Evolution from
Lee et al., "Evolving Deeper LLM Thinking" (arXiv:2501.09891, 2025).
"""

from __future__ import annotations

import argparse
import json
import random
from copy import deepcopy
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Template

from scenario_generation.llm_client import LLMConfig, create_llm
from scenario_generation.mind_evolution.utils import (
    get_examples_prompt,
    softmax,
    validator_RCC_prompt,
    validator_initial_prompt,
)
from scenario_generation.run_logging import EventType, RunLogger
from scenario_generation.scenario_evaluation.scenario_evaluation import (
    ScenarioEvaluation,
)
from scenario_generation.utils import safe_llm_complete


class MindEvolution:
    def __init__(
        self,
        config: dict[str, Any],
        user_requirements: dict[str, Any],
        *,
        llm: Any,
        llm_config: LLMConfig,
        structured_llm: Any,
        structured_llm_config: LLMConfig,
        output_dir: str | Path,
        run_desc: str | None = None,
    ):
        self.llm = llm
        self.llm_config = llm_config
        self.structured_llm = structured_llm
        self.structured_llm_config = structured_llm_config
        self.config = config
        self.user_requirements = user_requirements
        self.islands: list[list[dict[str, Any]]] = []
        self._initialize_prompt_templates()
        self.logging = RunLogger(
            user_requirements,
            config,
            output_dir,
            description=run_desc,
        )
        (self.logging.run_dir / "llm_config.json").write_text(
            json.dumps(
                {
                    "scenario_generation": llm_config.public_metadata(),
                    "structured_output": {
                        **structured_llm_config.public_metadata(),
                        "tasks": [
                            "example_selection",
                            "replay_evaluation",
                        ],
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        self.scenario_evaluation = ScenarioEvaluation(
            logging=self.logging,
            llm=self.structured_llm,
        )
        self.results_path = self.logging.run_dir / "island_evolutions.json"
        self.temp_log_path = self.logging.run_dir / "incremental_results.json"
        self.result_generations: list[Any] = []
        self.temp_log: list[Any] = []
        self.num_perfect_solutions = 0
        self.max_num_perfect_solutions = 3

    def stop_evolution(self, reason: str) -> None:
        self.logging.log_evolution_graph_event(
            EventType.STOP_EVOLUTION, reason=reason
        )
        self.logging.save_evolution_graph()
        self.logging.simple_log(f"Evolution stopped: {reason}")
        print(f"Evolution stopped: {reason}")
        raise SystemExit(0)

    def _initialize_prompt_templates(self) -> None:
        package = files("scenario_generation.mind_evolution")
        prompts = package.joinpath("prompts")
        self.general_instructions = Template(
            prompts.joinpath("general_instructions.txt").read_text(encoding="utf-8")
        )
        self.output_instructions = Template(
            prompts.joinpath("output_instructions.txt").read_text(encoding="utf-8")
        )
        self.rcc_instructions = Template(
            prompts.joinpath("RCC.txt").read_text(encoding="utf-8")
        )
        self.critic_strategy = Template(
            prompts.joinpath("critic_strategy.txt").read_text(encoding="utf-8")
        )
        self.parent_solutions = Template(
            prompts.joinpath("parent_solutions.txt").read_text(encoding="utf-8")
        )

    def _record_candidate(self, candidate: dict[str, Any]) -> None:
        self.temp_log.append(candidate)
        self.temp_log_path.write_text(
            json.dumps(self.temp_log, indent=2, default=str), encoding="utf-8"
        )
        if candidate["score"] == 1:
            self.num_perfect_solutions += 1
        if self.num_perfect_solutions >= self.max_num_perfect_solutions:
            self.stop_evolution("Max number of perfect solutions reached")

    def initialize_population(self) -> list[dict[str, Any]]:
        population: list[dict[str, Any]] = []
        for _ in range(self.config["N_convs"]):
            initial_prompt = (
                self.general_instructions.render(
                    examples_prompt=get_examples_prompt(
                        self.user_requirements,
                        llm=self.structured_llm,
                        num_examples=2,
                    ),
                    learning_objective=self.user_requirements[
                        "Learning Objective"
                    ],
                    strategy_expected=self.user_requirements["Strategy Expected"],
                )
                + "\n## Output Instructions\n"
                + self.output_instructions.render()
            )
            result = safe_llm_complete(
                initial_prompt,
                validator=validator_initial_prompt,
                llm=self.llm,
                max_tries=6,
                logging=self.logging,
            )
            call_id = RunLogger.generate_llm_call_id()
            if not result["ok"]:
                continue

            generated_map = result["validation_result"]["response_json"][
                "Generated Map"
            ]
            evaluation, score = self.scenario_evaluation.evaluate_generated_map(
                generated_map, self.user_requirements, map_id=call_id
            )
            candidate = {
                "id": call_id,
                "generated_map_str": generated_map,
                "evaluation_result": evaluation,
                "score": score,
            }
            population.append(candidate)
            self.logging.log_llm_call(
                initial_prompt,
                result,
                call_id,
                generated_map,
                call_type="conv",
            )
            self._record_candidate(candidate)
            if self.config["N_seqs"] > 1:
                population.extend(self.RCC([candidate]))
        return population

    def RCC(
        self, parents: list[dict[str, Any]], seq_start: int = 1
    ) -> list[dict[str, Any]]:
        if not parents:
            return []

        general_prompt = self.general_instructions.render(
            learning_objective=self.user_requirements["Learning Objective"],
            strategy_expected=self.user_requirements["Strategy Expected"],
        )
        output_prompt = self.output_instructions.render()
        sequence_solutions: list[dict[str, Any]] = []
        questions = [query["Query"] for query in self.user_requirements["Queries"]]

        for sequence in range(seq_start, self.config["N_seqs"]):
            parent_prompt = self.parent_solutions.render(
                parents=parents,
                questions=questions,
            )
            rcc_prompt = self.rcc_instructions.render(
                parents=parent_prompt,
                critic_strategy=self.critic_strategy.render(),
                output_instructions=output_prompt,
            )
            prompt = "\n".join([general_prompt, rcc_prompt])
            result = safe_llm_complete(
                prompt,
                validator=validator_RCC_prompt,
                llm=self.llm,
                max_tries=6,
                logging=self.logging,
            )
            call_id = RunLogger.generate_llm_call_id()
            if not result["ok"]:
                continue

            generated_map = result["validation_result"]["response_json"][
                "Generated Map"
            ]
            evaluation, score = self.scenario_evaluation.evaluate_generated_map(
                generated_map, self.user_requirements, map_id=call_id
            )
            candidate = {
                "id": call_id,
                "generated_map_str": generated_map,
                "evaluation_result": evaluation,
                "score": score,
            }
            sequence_solutions.append(candidate)
            self.logging.log_llm_call(
                prompt,
                result,
                call_id,
                generated_map,
                call_type="conv" if sequence == 0 else "rcc",
                parents=parents,
            )
            self._record_candidate(candidate)
            parents = [candidate]
        return sequence_solutions

    def boltzmann_selection(self, island_idx: int) -> list[dict[str, Any]]:
        population = self.islands[island_idx]
        if not population:
            raise RuntimeError(f"Island {island_idx} has no candidates to select")
        if random.random() < self.config["Pr_no_parents"]:
            return []
        weights = softmax([candidate["score"] for candidate in population])
        parent_count = random.randint(1, self.config["N_parents"])
        return random.choices(population, weights=weights, k=parent_count)

    def migration(self) -> None:
        island_count = len(self.islands)
        if island_count != self.config["N_islands"]:
            raise RuntimeError(
                f"Expected {self.config['N_islands']} islands, found {island_count}"
            )
        migrated_islands = [[] for _ in range(island_count)]
        for island_idx, population in enumerate(self.islands):
            selected = sorted(
                population, key=lambda candidate: candidate["score"], reverse=True
            )[: self.config["N_emigrate"]]
            destination = (island_idx + 1) % island_count
            for candidate in selected:
                migration_copy = deepcopy(candidate)
                migration_copy["id"] = RunLogger.generate_llm_call_id()
                self.logging.log_evolution_graph_event(
                    EventType.MIGRATION,
                    original_id=candidate["id"],
                    migration_id=migration_copy["id"],
                    old_island=island_idx,
                    new_island=destination,
                )
                migrated_islands[destination].append(migration_copy)

        for island_idx, migrants in enumerate(migrated_islands):
            self.islands[island_idx].extend(migrants)

    def island_new_generation(self, island_idx: int) -> None:
        for _ in range(self.config["N_convs"]):
            parents = self.boltzmann_selection(island_idx)
            self.islands[island_idx].extend(self.RCC(parents, seq_start=0))

    def get_island_avg_score(self) -> list[float]:
        return [
            sum(candidate["score"] for candidate in island) / len(island)
            for island in self.islands
        ]

    def islands_reset(self) -> None:
        all_candidates = [
            candidate for island in self.islands for candidate in island
        ]
        top_performers = sorted(
            all_candidates, key=lambda candidate: candidate["score"], reverse=True
        )[: self.config["N_top"]]
        scored_islands = sorted(
            enumerate(self.get_island_avg_score()), key=lambda item: item[1]
        )
        islands_to_remove = scored_islands[: self.config["N_reset"]]
        new_ids_by_island = {
            island_idx: [] for island_idx in range(len(self.islands))
        }

        for island_idx, _ in reversed(islands_to_remove):
            replacement = []
            for candidate in top_performers:
                new_candidate = deepcopy(candidate)
                new_candidate["id"] = RunLogger.generate_llm_call_id()
                new_ids_by_island[island_idx].append(new_candidate["id"])
                replacement.append(new_candidate)
            self.islands[island_idx] = replacement

        self.logging.log_evolution_graph_event(
            EventType.ISLAND_RESET,
            removed_islands=[index for index, _ in islands_to_remove],
            top_performers_new_ids_per_new_island=new_ids_by_island,
            top_performers_old_ids=[
                candidate["id"] for candidate in top_performers
            ],
        )

    def evolve(self) -> None:
        print("Initializing population...")
        self.logging.log_evolution_graph_event(EventType.NEW_GENERATION)
        for island_idx in range(self.config["N_islands"]):
            self.logging.log_evolution_graph_event(
                EventType.FOCUS_ISLAND, island_number=island_idx
            )
            population = self.initialize_population()
            if not population:
                raise RuntimeError(
                    f"Could not generate any valid candidates for island {island_idx}"
                )
            self.islands.append(population)
        self.migration()

        for generation in range(1, self.config["N_generations"]):
            self.logging.log_evolution_graph_event(EventType.NEW_GENERATION)
            print(f"Starting Generation {generation}...")
            self.result_generations.append(deepcopy(self.islands))
            self.results_path.write_text(
                json.dumps(self.result_generations, indent=2, default=str),
                encoding="utf-8",
            )
            for island_idx, _ in enumerate(self.islands):
                self.logging.log_evolution_graph_event(
                    EventType.FOCUS_ISLAND, island_number=island_idx
                )
                print(f"Starting Island {island_idx}...")
                self.island_new_generation(island_idx)
                print(f"Island {island_idx} new generation complete")
            self.migration()
            self.logging.simple_log(
                f"Generation {generation} complete; average scores: "
                f"{self.get_island_avg_score()}"
            )
            if (generation + 1) % self.config["N_reset_interval"] == 0:
                self.logging.simple_log("Resetting islands...")
                self.islands_reset()

        self.logging.save_evolution_graph()

    def save_evolution_results(self) -> None:
        (self.logging.run_dir / "evolution_results.json").write_text(
            json.dumps(self.islands, indent=2, default=str), encoding="utf-8"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and evolve scenarios using an installed AtlAtl backend."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--objective", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--run-desc")
    parser.add_argument(
        "--llm-provider",
        choices=("openai", "llama-cpp"),
        default="openai",
    )
    parser.add_argument(
        "--model",
        help="Model name, or llama.cpp model ID from /v1/models.",
    )
    parser.add_argument(
        "--llm-base-url",
        help="OpenAI-compatible base URL, such as http://127.0.0.1:8080/v1.",
    )
    parser.add_argument("--context-window", type=int, default=32_768)
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument(
        "--structured-max-output-tokens",
        type=int,
        default=4_096,
        help=(
            "Output limit for non-reasoning JSON tasks such as example "
            "selection and replay evaluation (default: 4096)."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--llm-timeout", type=float, default=500)
    parser.add_argument(
        "--thinking",
        choices=("off", "high", "max"),
        default="off",
        help=(
            "llama.cpp reasoning mode for scenario generation (default: off). "
            "The output-token limit covers both reasoning and the final answer. "
            "Structured selection and evaluation calls always disable reasoning."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    model = args.model or ("gpt-5" if args.llm_provider == "openai" else None)
    if model is None:
        raise SystemExit("--model is required when --llm-provider=llama-cpp.")
    llm_config = LLMConfig(
        provider=args.llm_provider,
        model=model,
        base_url=args.llm_base_url,
        context_window=args.context_window,
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
        timeout=args.llm_timeout,
        thinking=args.thinking,
    )
    try:
        llm = create_llm(llm_config)
        structured_llm_config = llm_config.for_structured_output(
            max_output_tokens=args.structured_max_output_tokens
        )
        structured_llm = create_llm(structured_llm_config)
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    config_document = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    objective = yaml.safe_load(args.objective.read_text(encoding="utf-8"))
    experiment = MindEvolution(
        config_document["evolution"],
        objective,
        llm=llm,
        llm_config=llm_config,
        structured_llm=structured_llm,
        structured_llm_config=structured_llm_config,
        output_dir=args.output_dir,
        run_desc=args.run_desc,
    )
    experiment.evolve()
    experiment.save_evolution_results()
    print(f"Run artifacts: {experiment.logging.run_dir}")


if __name__ == "__main__":
    main()
