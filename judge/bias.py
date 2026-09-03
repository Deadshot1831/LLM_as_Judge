"""The three biases every LLM judge has and almost nobody measures.

  python -m judge.bias --all --rubric v2

position   : same pair, both orders. Flip rate above ~10% means pairwise verdicts
             from this setup are not trustworthy.
length     : quality held constant, characters tripled. If the score moves, the
             rubric is rewarding volume.
self-pref  : two judges from different families score the same pairs. If each
             prefers its own family's answer, neither number is clean.
"""
import argparse
import concurrent.futures as futures
import itertools
import os

from scipy.stats import spearmanr

from judge import db, rubric as rubric_mod
from judge.run_judge import client, judge_pair


def base_pairs():
    """One pair per question: two models' unmodified answers to the same question."""
    rows = db.query("SELECT * FROM items WHERE id LIKE '%%::base' ORDER BY id")
    by_question = {}
    for row in rows:
        by_question.setdefault(row["question_id"], []).append(row)
    return [(a, b) for group in by_question.values() for a, b in itertools.combinations(group, 2)]


def position(api, model, rub, run_id):
    pairs = base_pairs()
    if not pairs:
        raise SystemExit("no competing answers — the dataset needs two models per question")

    def both_orders(pair):
        a, b = pair
        fwd = judge_pair(api, model, rub, a["question"], a["context"], a["answer"], b["answer"])
        rev = judge_pair(api, model, rub, a["question"], a["context"], b["answer"], a["answer"])
        return a, b, fwd, rev

    flips, ties, verdicts = 0, 0, []
    with futures.ThreadPoolExecutor(6) as pool:
        for a, b, fwd, rev in pool.map(both_orders, pairs):
            # fwd names a position; translate both verdicts to the item that won.
            fwd_item = {"a": a["id"], "b": b["id"]}.get(fwd)
            rev_item = {"a": b["id"], "b": a["id"]}.get(rev)
            if fwd_item is None or rev_item is None:
                ties += 1
            elif fwd_item != rev_item:
                flips += 1
            verdicts.append({"question_id": a["question_id"], "forward": fwd, "reverse": rev})
            db.execute(
                """INSERT INTO judge_pairwise (run_id, question_id, first_item, second_item, winner)
                   VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING""",
                (run_id, a["question_id"], a["id"], b["id"], fwd),
            )

    decided = len(verdicts) - ties
    rate = flips / decided if decided else 0.0
    db.record_metric(run_id, "bias", "position_flip_rate", rate, n=decided,
                     detail={"flips": flips, "ties": ties, "pairs": len(verdicts)})
    print(f"position bias: flip rate {rate:.1%} over {decided} decided pairs "
          f"({flips} flips, {ties} ties) — {'FAIL, above 10%' if rate > 0.10 else 'within 10%'}")
    return rate


def length(run_id):
    """Padded items carry identical content to their base twin, at roughly 3x the length."""
    rows = db.query(
        """SELECT i.id, i.answer, s.scores FROM judge_scores s JOIN items i ON i.id = s.item_id
           WHERE s.run_id = %s AND (i.id LIKE '%%::padded' OR i.id LIKE '%%::base')""",
        (run_id,),
    )
    if not rows:
        raise SystemExit("run has no base/padded items — score the bias split too "
                         "(`--splits dev,test,bias`)")
    criteria = sorted({c for r in rows for c in r["scores"]})
    lengths = [len(r["answer"]) for r in rows]
    means = [sum(r["scores"][c] for c in criteria) / len(criteria) for r in rows]
    rho = spearmanr(lengths, means).statistic
    db.record_metric(run_id, "bias", "length_spearman", rho, n=len(rows))
    print(f"length bias: spearman(chars, mean score) = {rho:+.3f} over {len(rows)} items")

    by_id = {r["id"]: r for r in rows}
    for criterion in criteria:
        deltas = [by_id[i]["scores"][criterion] - by_id[i.replace("::padded", "::base")]["scores"][criterion]
                  for i in by_id if i.endswith("::padded") and i.replace("::padded", "::base") in by_id]
        if not deltas:
            continue
        delta = sum(deltas) / len(deltas)
        db.record_metric(run_id, "bias", "length_delta", delta, criterion=criterion, n=len(deltas))
        print(f"  {criterion:<16} padded minus base: {delta:+.2f}")
    return rho


def self_preference(api, rub, run_id, judges):
    """Each judge scores the same pairs; count how often it picks its own family's answer."""
    pairs = base_pairs()
    results = {}
    for judge_model in judges:
        family = judge_model.split("-")[1] if judge_model.startswith("claude") else judge_model
        own, decided = 0, 0

        def verdict(pair):
            a, b = pair
            return a, b, judge_pair(api, judge_model, rub, a["question"], a["context"],
                                    a["answer"], b["answer"])

        with futures.ThreadPoolExecutor(6) as pool:
            for a, b, winner in pool.map(verdict, pairs):
                chosen = {"a": a, "b": b}.get(winner)
                if chosen is None:
                    continue
                other = b if chosen is a else a
                if (family in chosen["model"]) != (family in other["model"]):
                    decided += 1
                    own += family in chosen["model"]
        rate = own / decided if decided else None
        results[judge_model] = rate
        if rate is None:
            print(f"self-preference: {judge_model} — no pairs where exactly one answer is its own family")
            continue
        db.record_metric(run_id, "bias", "self_preference_rate", rate, n=decided,
                         detail={"judge": judge_model, "family": family})
        print(f"self-preference: {judge_model} picked its own family {rate:.1%} of {decided} pairs "
              f"(50% is neutral)")
    clean = [r for r in results.values() if r is not None]
    if len(clean) == 2:
        gap = abs(clean[0] - clean[1])
        db.record_metric(run_id, "bias", "self_preference_gap", gap)
        print(f"  swap gap between judges: {gap:.1%}")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rubric", default="v2")
    p.add_argument("--all", action="store_true")
    p.add_argument("--position", action="store_true")
    p.add_argument("--length", action="store_true")
    p.add_argument("--self-preference", action="store_true")
    p.add_argument("--score-run", type=int, help="run id whose scores the length test reads (default: latest)")
    p.add_argument("--judges", default=None, help="comma separated, for the self-preference swap")
    a = p.parse_args()

    rub = rubric_mod.load(a.rubric)
    model = os.environ.get("JUDGE_MODEL", "claude-opus-5")
    api = client()
    run_id = db.start_run(f"bias / {a.rubric}", rub["version"], model, "bias")

    if a.all or a.position:
        position(api, model, rub, run_id)
    if a.all or a.length:
        length(a.score_run or db.latest_run_id())
    if a.all or a.self_preference:
        judges = (a.judges or f"{model},claude-sonnet-5").split(",")
        self_preference(api, rub, run_id, [j.strip() for j in judges])
