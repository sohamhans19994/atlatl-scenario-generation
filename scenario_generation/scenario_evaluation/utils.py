"""Replay parsing, narration, and evaluator-response validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scenario_generation.str_utils import _extract_first_json_dict, _type_matches
from scenario_generation.utils import observation_to_str


def split_replay_reps(replay: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    repetitions: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for entry in replay:
        if entry["type"] == "parameters" and current:
            repetitions.append(current)
            current = []
        current.append(entry)
    if current:
        repetitions.append(current)
    return repetitions


def phase_grouping(
    blue_replay_rep: list[dict[str, Any]],
    red_replay_rep: list[dict[str, Any]],
) -> dict[str, Any]:
    replay = {"parameters": blue_replay_rep[0]["parameters"], "replayPhases": []}

    def first_non_setup(entries: list[dict[str, Any]]) -> int:
        for index, entry in enumerate(entries[1:], start=1):
            if entry["type"] != "observation":
                continue
            if not entry["observation"]["status"]["setupMode"]:
                return index
        return len(entries)

    blue_phases: list[dict[str, Any]] = []
    current = {"turn": "blue", "phaseCount": 0, "observations": [], "actions": []}
    on_blue_turn = True
    for entry in blue_replay_rep[first_non_setup(blue_replay_rep) :]:
        if entry["type"] == "observation":
            is_blue = entry["observation"]["status"]["onMove"] == "blue"
            if is_blue and not on_blue_turn:
                on_blue_turn = True
                current = {
                    "turn": "blue",
                    "phaseCount": entry["observation"]["status"]["phaseCount"],
                    "observations": [],
                    "actions": [],
                }
            elif not is_blue and on_blue_turn:
                if current["observations"]:
                    blue_phases.append(current)
                on_blue_turn = False
            if on_blue_turn:
                current["observations"].append(entry)
        elif entry["type"] == "action" and on_blue_turn:
            current["actions"].append(entry)

    red_phases: list[dict[str, Any]] = []
    current_red: dict[str, Any] | None = None
    on_red_turn = False
    for entry in red_replay_rep[first_non_setup(red_replay_rep) :]:
        if entry["type"] == "observation":
            is_red = entry["observation"]["status"]["onMove"] == "red"
            if is_red and not on_red_turn:
                on_red_turn = True
                current_red = {
                    "turn": "red",
                    "phaseCount": entry["observation"]["status"]["phaseCount"],
                    "observations": [],
                    "actions": [],
                }
            elif not is_red and on_red_turn:
                if current_red and current_red["observations"]:
                    red_phases.append(current_red)
                on_red_turn = False
            if on_red_turn and current_red is not None:
                current_red["observations"].append(entry)
        elif entry["type"] == "action" and on_red_turn and current_red is not None:
            current_red["actions"].append(entry)

    phases = blue_phases[:]
    for red_phase in red_phases:
        insertion_index = min(red_phase["phaseCount"], len(phases))
        phases.insert(insertion_index, red_phase)
    replay["replayPhases"] = phases
    return replay


def _unit_index(observation: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        unit["uniqueId"]: unit
        for unit in observation.get("units", [])
        if unit.get("uniqueId")
    }


def _terrain_index(parameters: dict[str, Any]) -> dict[str, str]:
    return {
        f"hex-{item['x_offset']}-{item['y_offset']}": item["terrain"]
        for item in parameters["map"]["hexes"]
    }


def get_phase_descriptions(
    replay: dict[str, Any],
) -> tuple[list[str], list[dict[str, Any]]]:
    action_descriptions: list[str] = []
    observation_strings: list[dict[str, Any]] = []
    terrain_by_hex = _terrain_index(replay["parameters"])
    phases = replay["replayPhases"]

    for phase_index, phase in enumerate(phases):
        if not phase["observations"]:
            continue
        turn = phase["turn"]
        first_observation = phase["observations"][0]["observation"]
        observation_strings.append(
            {
                "Phase": phase_index,
                "Turn": turn,
                "Observation": observation_to_str(
                    replay["parameters"], first_observation
                ),
            }
        )
        initial_units = _unit_index(first_observation)
        descriptions: list[str] = []

        for action_index, entry in enumerate(phase["actions"]):
            action = entry["action"]
            action_type = action["type"]
            if action_type == "pass":
                descriptions.append(f"No additional actions taken by {turn.title()}")
                break

            if action_type == "move":
                mover = initial_units.get(action["mover"], {})
                source = mover.get("hex", "an unknown hex")
                destination = action["destination"]
                descriptions.append(
                    f"{turn.title()} moved unit {action['mover']} from {source} "
                    f"in terrain {terrain_by_hex.get(source, 'unknown')} to "
                    f"{destination} in terrain "
                    f"{terrain_by_hex.get(destination, 'unknown')}"
                )
                continue

            if action_type == "fire":
                target_id = action["target"]
                original_target = initial_units.get(target_id, {})
                original_strength = original_target.get("currentStrength", 0)
                observation_index = min(
                    action_index + 1, len(phase["observations"]) - 1
                )
                next_observation = phase["observations"][observation_index][
                    "observation"
                ]
                next_target = _unit_index(next_observation).get(target_id)
                if next_target is None or next_target.get("ineffective", False):
                    outcome = f"{target_id} to become ineffective"
                else:
                    damage = original_strength - next_target.get(
                        "currentStrength", original_strength
                    )
                    outcome = f"it to lose {damage} strength"
                descriptions.append(
                    f"{action['source']} fired at {target_id}, causing {outcome}"
                )

        action_descriptions.append("\n".join(descriptions))

    return action_descriptions, observation_strings


def get_initial_state_board(replay: dict[str, Any]) -> str:
    first_phase = replay["replayPhases"][0]
    first_observation = first_phase["observations"][0]["observation"]
    return observation_to_str(
        replay["parameters"], first_observation, include_on_move=False
    )


def get_replay_description(
    blue_replay_rep: list[dict[str, Any]],
    red_replay_rep: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    grouped = phase_grouping(blue_replay_rep, red_replay_rep)
    action_descriptions, observations = get_phase_descriptions(grouped)
    sections = []
    for phase, actions in zip(observations, action_descriptions):
        sections.append(
            f"Phase {phase['Phase']}:\n"
            f"Current state:\n{phase['Observation']}\n"
            f"{phase['Turn']}'s turn, it takes the following actions:\n"
            f"{actions}\n"
        )
    return "\n".join(sections), grouped


def get_queries_description(queries: list[dict[str, Any]]) -> str:
    descriptions: list[str] = []
    for index, query in enumerate(queries, start=1):
        description = f"Question {index}: {query['Query']}"
        if query.get("Initialstatebased"):
            description += (
                " Answer only from the initial board state. Consider potential "
                "movement patterns rather than limiting the answer to the replay."
            )
        description += f" Answer must be of type {query['Answerdatatype']}."
        descriptions.append(description)
    return "\n\n".join(descriptions)


def evaluator_json_validator(
    queries: list[dict[str, Any]], parsed: Any
) -> tuple[bool, list[str]]:
    if not isinstance(parsed, dict):
        return False, ["Output is not a JSON object"]

    errors: list[str] = []
    required_top_keys = {"initial_map_analysis", "replay_analysis", "answers"}
    required_answer_keys = {"qid", "question_analysis", "answer", "reasoning"}

    if set(parsed) != required_top_keys:
        missing = required_top_keys - set(parsed)
        extra = set(parsed) - required_top_keys
        if missing:
            errors.append(f"Missing top-level keys: {sorted(missing)}")
        if extra:
            errors.append(f"Extra top-level keys: {sorted(extra)}")
    for key in ("initial_map_analysis", "replay_analysis"):
        if not isinstance(parsed.get(key), str):
            errors.append(f"{key!r} must be a string")

    answers = parsed.get("answers")
    if not isinstance(answers, list):
        errors.append("'answers' must be an array")
        return False, errors
    if len(answers) != len(queries):
        errors.append(f"Expected {len(queries)} answers, got {len(answers)}")

    for index, answer in enumerate(answers[: len(queries)]):
        if not isinstance(answer, dict):
            errors.append(f"Answer {index + 1} is not an object")
            continue
        if set(answer) != required_answer_keys:
            errors.append(
                f"Answer {index + 1} must contain exactly "
                f"{sorted(required_answer_keys)}"
            )
        if answer.get("qid") != index + 1:
            errors.append(f"Answer {index + 1} has the wrong qid")
        if not isinstance(answer.get("question_analysis"), str):
            errors.append(f"Answer {index + 1} question_analysis must be a string")
        if not isinstance(answer.get("reasoning"), str):
            errors.append(f"Answer {index + 1} reasoning must be a string")
        expected_type = str(queries[index].get("Answerdatatype", "string"))
        if "answer" in answer and not _type_matches(expected_type, answer["answer"]):
            errors.append(
                f"Answer {index + 1} does not match type {expected_type!r}"
            )

    return not errors, errors


def validator_replay_analysis(
    response_text: str, **kwargs: Any
) -> dict[str, Any]:
    json_text = _extract_first_json_dict(response_text)
    if json_text is None:
        return {
            "ok": False,
            "response_json": None,
            "errors": ["No JSON object found in output"],
        }
    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError as exc:
        return {
            "ok": False,
            "response_json": None,
            "errors": [f"Invalid JSON: {exc}"],
        }

    queries = kwargs.get("queries")
    if queries is None:
        raise ValueError("queries are required for replay validation")
    ok, errors = evaluator_json_validator(queries, parsed)
    return {
        "ok": ok,
        "response_json": parsed if ok else None,
        "errors": errors,
    }


def convert_replay_to_json(replay_js_path: str | Path) -> list[dict[str, Any]]:
    source = Path(replay_js_path).read_text(encoding="utf-8")
    start = source.find("[")
    end = source.rfind("]")
    if start < 0 or end <= start:
        raise ValueError(f"{replay_js_path} is not an AtlAtl replay file")

    replay: list[dict[str, Any]] = []
    for raw_line in source[start + 1 : end].splitlines():
        line = raw_line.strip().removesuffix(",").strip()
        if not line:
            continue
        if line.startswith("'") and line.endswith("'"):
            line = line[1:-1]
        replay.append(json.loads(line))
    return replay


# Compatibility aliases retained for research notebooks.
splitReplayReps = split_replay_reps
phaseGrouping = phase_grouping
getPhaseDescriptions = get_phase_descriptions
getInitialStateBoard = get_initial_state_board
getReplayDescription = get_replay_description
getQueriesDescription = get_queries_description
