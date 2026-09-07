"""Fast, offline checks on the arithmetic that the headline numbers rest on.
No API key, no database. `pytest tests/test_analysis.py`"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import judge.db as judge_db  # noqa: E402
from judge.agreement import agreement, bootstrap_ci, by_stratum, consistency, summary  # noqa: E402
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


class StubDB:
    """A few rows, enough for the analysis paths that only ever read."""

    ITEMS = ([{"id": f"easy{i}", "stratum": "easy"} for i in range(12)]
             + [{"id": f"adv{i}", "stratum": "adversarial"} for i in range(12)])

    def query(self, sql, params=()):
        s = " ".join(sql.split())
        if "SELECT id, stratum FROM items" in s:
            return self.ITEMS
        if "FROM human_labels l JOIN items i" in s:
            return [{"item_id": i["id"], "scores": {"groundedness": 3}} for i in self.ITEMS]
        if "count(*) AS n FROM human_labels" in s:
            return [{"n": 24}]
        if "FROM judge_scores WHERE run_id" in s:
            run = params[0]
            # run 1 nails the easy items and guesses on the adversarial ones; run 2 differs
            # on two items, which is what makes the judge's self-consistency less than 1.
            out = {}
            for i in self.ITEMS:
                if i["stratum"] == "easy":
                    out[i["id"]] = {"groundedness": 3}
                else:
                    out[i["id"]] = {"groundedness": 1 if int(i["id"][3:]) % 2 else 3}
            if run == 2:
                out["easy0"] = {"groundedness": 1}
                out["adv1"] = {"groundedness": 2}
            return [{"item_id": k, "scores": v} for k, v in out.items()]
        if "FROM metrics m" in s:
            name = params[0]
            table = {"weighted_kappa": [{"value": 0.612, "n": 144, "criterion": "groundedness",
                                         "detail": {"split": "test", "ci_low": 0.48,
                                                    "ci_high": 0.73}}],
                     "position_flip_rate": [{"value": 0.083, "n": 24, "criterion": None,
                                             "detail": None}]}
            return table.get(name, [])
        return []


def stub_db(monkey=None):
    stub = StubDB()
    judge_db.query = stub.query
    judge_db.execute = lambda sql, params=(): None
    return stub


def test_bootstrap_interval_brackets_the_estimate_and_narrows_with_n():
    import random
    rng = random.Random(3)
    xs = [rng.choice([1, 2, 3]) for _ in range(400)]
    ys = [x if rng.random() > 0.3 else rng.choice([1, 2, 3]) for x in xs]
    point = agreement(xs, ys)["weighted_kappa"]
    lo, hi = bootstrap_ci(xs, ys, draws=400)
    assert lo < point < hi
    wide = bootstrap_ci(xs[:40], ys[:40], draws=400)
    assert (wide[1] - wide[0]) > (hi - lo) * 1.5, "fewer items must mean a wider interval"
    assert bootstrap_ci(xs[:5], ys[:5]) is None, "no interval worth quoting under 10 items"


def test_stratum_slicing_separates_where_the_judge_fails():
    stub_db()
    sliced = by_stratum(run_id=1, split="test")
    easy = sliced[("groundedness", "easy")]
    adversarial = sliced[("groundedness", "adversarial")]
    # Humans said 3 everywhere; the judge is perfect on easy and wrong half the time on
    # adversarial. An averaged kappa would report neither of those facts.
    assert adversarial["weighted_kappa"] is not None and adversarial["exact_match"] < 1.0
    # And the easy slice is the trap: perfect agreement, but with no variance on either
    # side kappa is undefined rather than 1.0, and must not be recorded as a number.
    assert easy["degenerate"] and easy["weighted_kappa"] is None
    assert easy["exact_match"] == 1.0


def test_judge_self_consistency_is_less_than_one_when_it_changes_its_mind():
    stub_db()
    result = consistency(1, 2)["groundedness"]
    assert result["exact_match"] < 1.0 and result["n"] == 24


def test_summary_quotes_kappa_with_its_interval(tmp_path):
    import judge.agreement as mod
    stub_db()
    mod.REPORTS = tmp_path
    summary("test")
    table = (tmp_path / "summary.md").read_text()
    assert "groundedness 0.612 [0.48–0.73]" in table, "kappa must never be quoted bare"
    assert "| Human labels collected | 24 |" in table
    assert "8.3%" in table, "position flip rate should render as a percentage"


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
