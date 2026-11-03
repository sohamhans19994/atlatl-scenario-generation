"""Prompt preparation and validation for Mind Evolution."""

from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np

from scenario_generation.str_utils import _extract_first_json_dict
from scenario_generation.utils import HexMapValidator, safe_llm_complete


@lru_cache(maxsize=1)
def load_examples(path: str | Path | None = None) -> list[dict[str, Any]]:
    if path is None:
        resource = files("scenario_generation.mind_evolution").joinpath(
            "examples/examples.json"
        )
        return json.loads(resource.read_text(encoding="utf-8"))
    return json.loads(Path(path).read_text(encoding="utf-8"))


def get_examples_prompt(
    query: dict[str, Any], llm: Any, num_examples: int = 2
) -> str:
    chosen_ids = get_top_k_examples(query, llm=llm, k=num_examples)
    examples = load_examples()
    sections = [
        "## Examples",
        (
            "Here are scenario maps generated for similar learning objectives "
            "and expected strategies."
        ),
    ]
    for example_id in chosen_ids:
        example = examples[example_id - 1]
        sections.extend(
            [
                f"Learning Objective: {example['Learning Objective']}",
                f"Strategy Expected: {example['Strategy Expected']}",
                f"Scenario Map:\n{example['Map String']}",
                (
                    "Map Analysis Description:\n"
                    f"{example['Evaluation']['Analysis']['initial_map_analysis']}"
                ),
            ]
        )
    return "\n".join(sections)


def get_top_k_examples(
    query: dict[str, Any], llm: Any, k: int = 2
) -> list[int]:
    examples = load_examples()
    examples_text = "\n".join(
        (
            f"{index})\n"
            f"Learning Objective: {example['Learning Objective']}\n"
            f"Strategy Expected: {example['Strategy Expected']}"
        )
        for index, example in enumerate(examples, start=1)
    )
    prompt = f"""
Given the following user objectives:
Learning Objective: {query["Learning Objective"]}
Strategy Expected: {query["Strategy Expected"]}

Choose {k} items from the list below that are most similar to the objectives.
Reason briefly, then return their one-based indices inside a JSON object:
{{"top_k_indices": [1, 4]}}

Previous items:
{examples_text}
"""

    def validate_examples(response_text: str, **_: Any) -> dict[str, Any]:
        json_text = _extract_first_json_dict(response_text)
        if json_text is None:
            return {
                "ok": False,
                "response_json": None,
                "errors": ["No JSON object found in output"],
            }
        try:
            parsed = json.loads(json_text)
            indices = parsed["top_k_indices"]
            if not isinstance(indices, list) or len(indices) != k:
                raise ValueError(f"top_k_indices must contain exactly {k} items")
            if len(set(indices)) != k:
                raise ValueError("top_k_indices must not contain duplicates")
            if not all(
                isinstance(index, int) and 1 <= index <= len(examples)
                for index in indices
            ):
                raise ValueError("top_k_indices contains an out-of-range index")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return {"ok": False, "response_json": None, "errors": [str(exc)]}
        return {"ok": True, "response_json": parsed, "errors": []}

    result = safe_llm_complete(
        prompt,
        llm=llm,
        validator=validate_examples,
    )
    if not result["ok"]:
        raise RuntimeError(f"Could not select prompt examples: {result['errors']}")
    return result["validation_result"]["response_json"]["top_k_indices"]


def _validate_generated_map_response(response_text: str) -> dict[str, Any]:
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
        return {"ok": False, "response_json": None, "errors": [str(exc)]}

    generated_map = parsed.get("Generated Map")
    if not isinstance(generated_map, str):
        return {
            "ok": False,
            "response_json": None,
            "errors": ["Generated Map must be present and must be a string"],
        }
    validation = HexMapValidator.validate_map_text(generated_map)
    if not validation["ok"]:
        return {
            "ok": False,
            "response_json": None,
            "errors": validation["errors"],
        }
    return {
        "ok": True,
        "response_json": parsed,
        "string_validation_result": validation,
        "errors": [],
    }


def validator_initial_prompt(response_text: str, **_: Any) -> dict[str, Any]:
    return _validate_generated_map_response(response_text)


def validator_RCC_prompt(response_text: str, **_: Any) -> dict[str, Any]:
    return _validate_generated_map_response(response_text)


def softmax(values: list[float]) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    shifted = vector - np.max(vector)
    weights = np.exp(shifted)
    return weights / np.sum(weights)
