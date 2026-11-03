"""Generated map -> engine replay -> (fake) LLM evaluation -> score."""

import pytest

from scenario_generation.scenario_evaluation.scenario_evaluation import (
    ScenarioEvaluation,
)


def test_evaluate_generated_map_end_to_end(fake_llm_factory, example_tasks, objective):
    llm = fake_llm_factory(answers=[True, False, True, False])
    evaluator = ScenarioEvaluation(llm=llm)

    result, score = evaluator.evaluate_generated_map(
        example_tasks[0]["Map String"], objective
    )

    assert llm.calls["evaluate"] == 1
    assert [answer["answer"] for answer in result["answers"]] == [
        True,
        False,
        True,
        False,
    ]
    assert score == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("answers", "expected"),
    [([True] * 4, 1.0), ([False] * 4, 0.0), ([True, True, False, False], 0.5)],
)
def test_evaluation_scorer(objective, answers, expected):
    result = {
        "answers": [
            {"qid": index, "answer": answer}
            for index, answer in enumerate(answers, start=1)
        ]
    }
    assert ScenarioEvaluation.evaluation_scorer(result, objective) == pytest.approx(
        expected
    )
