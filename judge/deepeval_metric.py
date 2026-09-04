"""The calibrated judge, exposed as DeepEval metrics.

One metric per rubric criterion. All criteria come from a single judge call per
answer — DeepEval measures metrics independently, so the raw scores are cached by
answer hash to keep a four-criterion gate at one API call per test case.
"""
import hashlib
import os

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from judge import rubric as rubric_mod
from judge.run_judge import client, score_item

_CACHE = {}


def _key(test_case, rubric_version, model, variant):
    raw = f"{rubric_version}|{model}|{variant}|{test_case.input}|{test_case.actual_output}"
    return hashlib.sha256(raw.encode()).hexdigest()


def judge_all(test_case, rubric_version, model, variant="reason_first"):
    key = _key(test_case, rubric_version, model, variant)
    if key not in _CACHE:
        rub = rubric_mod.load(rubric_version)
        item = {
            "id": key[:12],
            "question": test_case.input,
            "context": "\n".join(test_case.retrieval_context or test_case.context or []),
            "answer": test_case.actual_output,
        }
        scores, reasoning, _ = score_item(client(), model, rub, item, variant)
        _CACHE[key] = (scores, reasoning, rub["scale"])
    return _CACHE[key]


class RubricCriterion(BaseMetric):
    """Score is normalised to 0-1 so thresholds read the same whatever the rubric scale is."""

    _required_params = [LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT]

    def __init__(self, criterion, threshold=0.5, rubric_version="v2", model=None,
                 variant="reason_first", strict_mode=False):
        self.criterion = criterion
        self.threshold = threshold
        self.rubric_version = rubric_version
        self.model = model or os.environ.get("JUDGE_MODEL", "claude-opus-5")
        self.variant = variant
        self.strict_mode = strict_mode
        self.async_mode = False
        self.include_reason = True
        self.evaluation_model = self.model

    @property
    def __name__(self):
        return f"{self.criterion} ({self.rubric_version})"

    def measure(self, test_case: LLMTestCase, *args, **kwargs):
        scores, reasoning, scale = judge_all(test_case, self.rubric_version, self.model, self.variant)
        self.raw_score = scores[self.criterion]
        self.score = (self.raw_score - min(scale)) / (max(scale) - min(scale))
        self.reason = reasoning.get(self.criterion, "")
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs):
        return self.measure(test_case)

    def is_successful(self):
        return self.success
