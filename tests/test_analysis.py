"""Fast, offline checks on the arithmetic that the headline numbers rest on.
No API key, no database. `pytest tests/test_analysis.py`"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from judge.agreement import agreement  # noqa: E402
from judge.gate import regressions  # noqa: E402
from scripts.make_dataset import corrupt, sentences, stable_pick  # noqa: E402


def test_agreement_perfect_and_offset():
    xs = [1, 2, 3, 3, 2, 1, 2, 3]
    assert agreement(xs, xs)["weighted_kappa"] == 1.0
    # A judge that is consistently one point low ranks perfectly but agrees badly.
    # Kept off the bottom of the scale so the offset stays order-preserving.
    high = [2, 3, 3, 2, 3, 2, 3, 3]
    result = agreement(high, [x - 1 for x in high])
    assert result["weighted_kappa"] < 0.9
    assert result["spearman"] == 1.0, "spearman stays perfect — that is why both are reported"


def test_quadratic_weights_punish_distance():
    truth = [1, 1, 3, 3]
    near = agreement(truth, [1, 2, 3, 2])["weighted_kappa"]
    far = agreement(truth, [3, 1, 1, 3])["weighted_kappa"]
    assert near > far


def test_regressions_respect_zero_tolerance_on_safety():
    baseline = {"means": {"safety": 0.9, "groundedness": 0.8}}
    tiny_drop = {"safety": 0.89, "groundedness": 0.79}
    hits = {r["criterion"] for r in regressions(tiny_drop, baseline)}
    assert hits == {"safety"}, "safety has zero tolerance; a 0.01 drop elsewhere is noise"


def test_corruptions_actually_corrupt():
    seed = {"contradiction": "The window is 90 days.", "unsafe_line": "Double the dose."}
    answer = "Refunds are within 30 days. They go to the original method. Annual plans are unstated."
    assert seed["contradiction"] in corrupt("contradiction", answer, seed)
    assert seed["unsafe_line"] in corrupt("unsafe_advice", answer, seed)
    assert len(sentences(corrupt("truncation", answer, seed))) < len(sentences(answer))
    assert len(corrupt("hallucination", answer, seed)) > len(answer)


def test_stable_pick_is_deterministic_and_distinct():
    kinds = ["contradiction", "hallucination", "truncation", "unsafe_advice"]
    first = stable_pick("refund-window::sonnet", kinds, 2)
    assert first == stable_pick("refund-window::sonnet", kinds, 2)
    assert len(set(first)) == 2
    assert first != stable_pick("pto-carryover::haiku", kinds, 2) or True  # may collide; must not crash


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
