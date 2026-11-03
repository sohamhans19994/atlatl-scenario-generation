"""Shared fixtures for integration tests against a sibling atlatl_ICT checkout.

All tests are offline: LLM calls go to ``FakeLLM``, which returns canned
responses shaped to pass the generator's own validators. The engine is real.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import yaml

from scenario_generation.atlatl_backend import find_atlatl_root

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPECTED_ENGINE = (REPO_ROOT.parent / "atlatl_ICT").resolve()
ENGINE = find_atlatl_root()
EXAMPLES_DIR = REPO_ROOT / "examples"
ENGINE_SCENARIOS = ENGINE / "server" / "scenarios"


@pytest.fixture(autouse=True)
def naval_combat(monkeypatch):
    # Generated maps use destroyers, which need AtlAtl's naval combat model.
    # The backend forwards os.environ to the engine subprocess.
    monkeypatch.setenv("ATLATL_COMBAT", "naval")


def load_example_tasks() -> list[dict]:
    from scenario_generation.mind_evolution.utils import load_examples

    return load_examples()


@pytest.fixture
def example_tasks() -> list[dict]:
    return load_example_tasks()


@pytest.fixture
def objective() -> dict:
    return yaml.safe_load((EXAMPLES_DIR / "objective.yaml").read_text())


@dataclass
class FakeResponse:
    text: str


@dataclass
class FakeLLM:
    """Routes prompts to canned replies by markers in the prompt templates.

    - example selection prompt contains ``top_k_indices``
    - replay evaluator prompt contains ``question_analysis``
    - anything else is a generation / RCC prompt
    """

    generated_map: str
    answers: list[bool] = field(default_factory=lambda: [True, False, True, False])
    calls: dict[str, int] = field(
        default_factory=lambda: {"select": 0, "evaluate": 0, "generate": 0}
    )

    def complete(self, prompt: str) -> FakeResponse:
        if "top_k_indices" in prompt:
            self.calls["select"] += 1
            return FakeResponse('Reasoning.\n{"top_k_indices": [1, 2]}')
        if "question_analysis" in prompt:
            self.calls["evaluate"] += 1
            reply = {
                "initial_map_analysis": "fake map analysis",
                "replay_analysis": "fake replay analysis",
                "answers": [
                    {
                        "qid": index,
                        "question_analysis": "fake",
                        "answer": answer,
                        "reasoning": "fake",
                    }
                    for index, answer in enumerate(self.answers, start=1)
                ],
            }
            return FakeResponse(f"```json\n{json.dumps(reply)}\n```")
        self.calls["generate"] += 1
        reply = {"Generated Map": self.generated_map}
        return FakeResponse(
            f"Reasoning about the map.\n## Generated Scenario Map\n"
            f"```json\n{json.dumps(reply)}\n```"
        )


@pytest.fixture
def fake_llm_factory(example_tasks):
    def make(answers: list[bool] | None = None) -> FakeLLM:
        llm = FakeLLM(generated_map=example_tasks[0]["Map String"])
        if answers is not None:
            llm.answers = answers
        return llm

    return make
