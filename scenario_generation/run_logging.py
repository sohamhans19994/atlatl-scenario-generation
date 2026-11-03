"""Run logging and evolution-lineage recording."""

from __future__ import annotations

import json
import shutil
import time
import uuid
from enum import Enum
from pathlib import Path
from typing import Any


class EventType(Enum):
    NEW_GENERATION = "new_generation"
    FOCUS_ISLAND = "focus_island"
    NEW_CONV = "new_conv"
    NEW_RCC = "new_rcc"
    MIGRATION = "migration"
    ISLAND_RESET = "island_reset"
    STOP_EVOLUTION = "stop_evolution"


class RunLogger:
    def __init__(
        self,
        user_requirements: dict[str, Any],
        config: dict[str, Any],
        output_root: str | Path,
        description: str | None = None,
    ):
        suffix = f"_{description}" if description else ""
        self.run_id = time.strftime("%Y%m%d%H%M%S") + suffix
        self.run_dir = Path(output_root).resolve() / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)

        self.llm_results_path = self.run_dir / "llm_results.jsonl"
        self.evaluation_results_path = self.run_dir / "evaluation_results.jsonl"
        self.evolution_graph_path = self.run_dir / "evolution_graph.json"
        self.events_path = self.run_dir / "events.log"
        self.evolution_graph: dict[str, Any] = {"generations": []}
        self.current_metadata = {"generation_number": 0, "island_number": 0}
        self.current_island_data: list[dict[str, Any]] = []
        self.current_conv_data: dict[str, Any] | None = None
        self.config = config

        self._write_json("objective.json", user_requirements)
        self._write_json("config.json", config)

    def _write_json(self, filename: str, value: Any) -> None:
        (self.run_dir / filename).write_text(
            json.dumps(value, indent=2, default=str), encoding="utf-8"
        )

    @staticmethod
    def generate_llm_call_id() -> str:
        return str(uuid.uuid4())

    def log_replay_runs(
        self, map_id: str, blue_replay_path: Path, red_replay_path: Path
    ) -> None:
        replay_dir = self.run_dir / "replays" / f"map_{map_id}"
        replay_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(blue_replay_path, replay_dir / "bluereplay.js")
        shutil.copy2(red_replay_path, replay_dir / "redreplay.js")

    def log_evaluation_calls(
        self, map_id: str, llm_outputs: list[Any], evaluation_result: Any
    ) -> None:
        record = {
            "id": map_id,
            "llm_outputs": llm_outputs,
            "evaluation_result": evaluation_result,
        }
        with self.evaluation_results_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, default=str) + "\n")

    def log_llm_call(
        self,
        prompt: str,
        llm_result: Any,
        call_id: str,
        generated_map_str: str,
        *,
        call_type: str | None = None,
        parents: list[dict[str, Any]] | None = None,
    ) -> None:
        if call_type == "conv":
            self.log_evolution_graph_event(EventType.NEW_CONV, id=call_id)
        elif call_type == "rcc":
            self.log_evolution_graph_event(EventType.NEW_RCC, id=call_id)
        elif call_type is not None:
            raise ValueError("call_type must be 'conv', 'rcc', or None")

        record = {
            "id": call_id,
            "prompt": prompt,
            "llm_result": llm_result,
            "generated_map_str": generated_map_str,
            "parents": [parent["id"] for parent in parents or []],
        }
        with self.llm_results_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, default=str) + "\n")

    def save_evolution_graph(self) -> None:
        self._flush_current_conversation()
        self._write_json("evolution_graph.json", self.evolution_graph)

    def simple_log(self, message: str) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(f"{timestamp} | {message}\n")

    def _flush_current_conversation(self) -> None:
        if not self.evolution_graph["generations"]:
            return
        if self.current_conv_data is not None:
            self.current_island_data.append(self.current_conv_data)
            self.current_conv_data = None
        if self.current_island_data:
            island_number = self.current_metadata["island_number"]
            self.evolution_graph["generations"][-1]["islands"][
                island_number
            ].extend(self.current_island_data)
            self.current_island_data = []

    def log_evolution_graph_event(self, event_type: EventType, **kwargs: Any) -> None:
        if event_type == EventType.NEW_GENERATION:
            self._flush_current_conversation()
            if self.evolution_graph["generations"]:
                self._write_json("evolution_graph.json", self.evolution_graph)
            self.current_metadata["generation_number"] += 1
            self.current_metadata["island_number"] = 0
            self.evolution_graph["generations"].append(
                {"islands": [[] for _ in range(self.config["N_islands"])]}
            )
            self.simple_log(
                f"New Generation {self.current_metadata['generation_number']}"
            )
            return

        if event_type == EventType.FOCUS_ISLAND:
            island_number = kwargs["island_number"]
            self._flush_current_conversation()
            self.current_metadata["island_number"] = island_number
            self.simple_log(f"Focus Island {island_number}")
            return

        if event_type == EventType.NEW_CONV:
            if self.current_conv_data is not None:
                self.current_island_data.append(self.current_conv_data)
            call_id = kwargs["id"]
            self.current_conv_data = {"id": call_id, "rccs": []}
            self.simple_log(f"New Conv {call_id}")
            return

        if event_type == EventType.NEW_RCC:
            if self.current_conv_data is None:
                raise ValueError("A conversation must exist before an RCC event")
            rcc_id = kwargs["id"]
            self.current_conv_data["rccs"].append({"id": rcc_id})
            self.simple_log(f"New RCC {rcc_id}")
            return

        if event_type == EventType.MIGRATION:
            original_id = kwargs["original_id"]
            migration_id = kwargs["migration_id"]
            old_island = kwargs["old_island"]
            new_island = kwargs["new_island"]
            self.evolution_graph["generations"][-1]["islands"][new_island].append(
                {
                    "id": migration_id,
                    "parent_uuids": [original_id],
                    "rccs": [],
                    "notes": f"Migrated from {old_island}",
                }
            )
            self.simple_log(
                f"Migration {original_id} to {migration_id} "
                f"from {old_island} to {new_island}"
            )
            return

        if event_type == EventType.ISLAND_RESET:
            removed_islands = kwargs["removed_islands"]
            old_ids = kwargs["top_performers_old_ids"]
            new_ids_by_island = kwargs["top_performers_new_ids_per_new_island"]
            for island in removed_islands:
                self.evolution_graph["generations"][-1]["islands"][island] = [
                    {
                        "id": new_id,
                        "parent_uuids": [old_ids[index]],
                        "rccs": [],
                        "notes": "Added as a top performer during island reset",
                    }
                    for index, new_id in enumerate(new_ids_by_island[island])
                ]
            self.simple_log(
                f"Island Reset: removed {removed_islands}; "
                f"top performers {old_ids}"
            )
            return

        if event_type == EventType.STOP_EVOLUTION:
            self.simple_log(f"Evolution stopped: {kwargs.get('reason', 'Unknown')}")


# Preserve the original class name for callers using the research code directly.
Logging = RunLogger
