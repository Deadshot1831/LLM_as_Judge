"""CI gate. Runs on every pull request; fails the build on a quality regression.

  pytest tests/test_eval_gate.py -v          # plain pytest
  deepeval test run tests/test_eval_gate.py  # same metrics, DeepEval's reporting

Needs ANTHROPIC_API_KEY. Skips itself without one so the fast unit tests still run.
"""
import os

import pytest

pytest.importorskip("deepeval")

from judge import gate, rubric as rubric_mod  # noqa: E402

pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"), reason="no ANTHROPIC_API_KEY"
)


@pytest.fixture(scope="module")
def measured():
    baseline = gate.load_baseline()
    cases = gate.build_cases()
    normalised, per_case = gate.measure(cases, baseline["rubric_version"])
    current = gate.means(normalised)
    gate.record(baseline["rubric_version"], current)
    return baseline, current, per_case, normalised


def test_no_criterion_regresses(measured):
    baseline, current, _, _ = measured
    bad = gate.regressions(current, baseline)
    report = "\n".join(
        f"  {r['criterion']}: {r['baseline']:.3f} -> {r['current']:.3f} "
        f"(dropped {r['drop']:.3f}, tolerance {r['tolerance']:.3f})" for r in bad)
    assert not bad, f"quality regression against baseline.json:\n{report}"


def test_safety_never_drops(measured):
    baseline, current, _, _ = measured
    if "safety" not in current:
        pytest.skip("this rubric declares no safety criterion")
    assert current["safety"] >= baseline["means"]["safety"], (
        f"safety dropped {baseline['means']['safety']:.3f} -> {current['safety']:.3f}; "
        "this criterion has zero tolerance")


def test_no_individual_safety_failure(measured):
    baseline, _, per_case, _ = measured
    if not all("safety" in s for s in per_case.values()):
        pytest.skip("this rubric declares no safety criterion")
    scale = rubric_mod.load(baseline["rubric_version"])["scale"]
    failures = {name: s["safety"] for name, s in per_case.items() if s["safety"] == min(scale)}
    assert not failures, f"answers scored the lowest safety point: {failures}"


def test_baseline_matches_current_rubric(measured):
    baseline, _, _, _ = measured
    assert baseline["rubric_version"] in rubric_mod.versions(), (
        "baseline.json references a rubric that no longer exists")
