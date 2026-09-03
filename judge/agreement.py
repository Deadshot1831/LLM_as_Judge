"""Judge-to-human agreement, and the human's own ceiling.

  python -m judge.agreement --run 3            # report agreement for a judge run
  python -m judge.agreement --self-agreement   # intra-rater ceiling, no judge needed

Weighted Cohen's kappa (quadratic weights) is the headline because the scores are
ordinal — a 3-vs-1 disagreement should cost more than 3-vs-2. Spearman is reported
alongside it because kappa punishes a judge that is consistently offset by one point
even when its ranking is perfect, and you want to see both failure modes separately.
"""
import argparse
import pathlib
import statistics

from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

from judge import db, rubric as rubric_mod

REPORTS = pathlib.Path(__file__).resolve().parent.parent / "reports"


def human_consensus(split=None, pass_no=1):
    """{item_id: {criterion: score}} — median across labelers, so a third labeler breaks ties."""
    sql = """SELECT l.item_id, l.scores FROM human_labels l JOIN items i ON i.id = l.item_id
             WHERE l.pass_no = %s"""
    params = [pass_no]
    if split:
        sql += " AND i.split = %s"
        params.append(split)
    per_item = {}
    for row in db.query(sql, tuple(params)):
        per_item.setdefault(row["item_id"], []).append(row["scores"])
    return {
        item: {c: round(statistics.median([s[c] for s in labels if c in s]))
               for c in {k for s in labels for k in s}}
        for item, labels in per_item.items()
    }


def judge_scores(run_id):
    return {r["item_id"]: r["scores"]
            for r in db.query("SELECT item_id, scores FROM judge_scores WHERE run_id = %s", (run_id,))}


def paired(a, b, criterion):
    items = sorted(set(a) & set(b))
    xs = [a[i][criterion] for i in items if criterion in a[i] and criterion in b[i]]
    ys = [b[i][criterion] for i in items if criterion in a[i] and criterion in b[i]]
    return xs, ys


def agreement(xs, ys):
    if len(xs) < 2:
        return None
    kappa = cohen_kappa_score(xs, ys, weights="quadratic")
    # Spearman is undefined when either side is constant; report it as None rather than nan.
    rho = spearmanr(xs, ys).statistic if len(set(xs)) > 1 and len(set(ys)) > 1 else None
    return {"weighted_kappa": kappa, "spearman": rho, "n": len(xs),
            "exact_match": sum(x == y for x, y in zip(xs, ys)) / len(xs)}


def self_agreement():
    """The ceiling: the same labeler, same items, two different days."""
    first, second = human_consensus(pass_no=1), human_consensus(pass_no=2)
    rows = db.query("SELECT DISTINCT labeler_id FROM human_labels WHERE pass_no = 2")
    out = {}
    for criterion in all_criteria():
        xs, ys = paired(first, second, criterion)
        result = agreement(xs, ys)
        if result:
            out[criterion] = result
    if not out:
        print("no second-pass labels yet — label 20 items again on a different day")
    return out, [r["labeler_id"] for r in rows]


def all_criteria():
    rows = db.query("SELECT scores FROM human_labels LIMIT 50") or db.query("SELECT scores FROM judge_scores LIMIT 1")
    return sorted({k for r in rows for k in r["scores"]})


def disagreements(run_id, split="dev", top=20):
    """The 20 biggest gaps. Read them one by one — the rubric is usually the thing that is wrong."""
    human, judge = human_consensus(split=split), judge_scores(run_id)
    rows = []
    meta = {r["id"]: r for r in db.query("SELECT * FROM items")}
    reasoning = {r["item_id"]: r["reasoning"]
                 for r in db.query("SELECT item_id, reasoning FROM judge_scores WHERE run_id = %s", (run_id,))}
    for item_id in set(human) & set(judge):
        for criterion, h in human[item_id].items():
            j = judge[item_id].get(criterion)
            if j is None:
                continue
            rows.append({"item_id": item_id, "criterion": criterion, "human": h, "judge": j,
                         "gap": abs(h - j), "item": meta[item_id],
                         "why": (reasoning.get(item_id) or {}).get(criterion, "")})
    return sorted(rows, key=lambda r: -r["gap"])[:top]


def write_disagreement_report(run_id, split, top):
    rows = disagreements(run_id, split, top)
    REPORTS.mkdir(exist_ok=True)
    path = REPORTS / f"disagreements_run{run_id}_{split}.md"
    lines = [f"# Largest judge/human disagreements — run {run_id}, {split} split\n",
             "Read each one and ask: is the rubric ambiguous here, or is the judge wrong?",
             "In most cases it is the rubric. Fix the wording, add an anchor, bump the version.\n"]
    for r in rows:
        item = r["item"]
        lines += [
            f"\n## {r['item_id']} — {r['criterion']}: human {r['human']}, judge {r['judge']} (gap {r['gap']})",
            f"- stratum: {item['stratum']}  defect: {item['defect'] or 'none'}",
            f"- question: {item['question']}",
            f"- answer: {item['answer']}",
            f"- judge said: {r['why']}",
            "- [ ] rubric ambiguous  - [ ] judge wrong  - [ ] human wrong",
        ]
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path} ({len(rows)} disagreements)")


def report(run_id, split):
    human, judge = human_consensus(split=split), judge_scores(run_id)
    overlap = set(human) & set(judge)
    if not overlap:
        raise SystemExit(f"no labelled items in the {split} split are covered by run {run_id}")
    print(f"\nrun {run_id} vs humans on {split} ({len(overlap)} items)")
    print(f"{'criterion':<16}{'kappa_w':>10}{'spearman':>10}{'exact':>8}{'n':>6}")
    for criterion in sorted({c for v in human.values() for c in v}):
        xs, ys = paired(human, judge, criterion)
        result = agreement(xs, ys)
        if not result:
            continue
        rho = f"{result['spearman']:.3f}" if result["spearman"] is not None else "n/a"
        print(f"{criterion:<16}{result['weighted_kappa']:>10.3f}{rho:>10}"
              f"{result['exact_match']:>8.2f}{result['n']:>6}")
        db.record_metric(run_id, "agreement", "weighted_kappa", result["weighted_kappa"],
                         criterion=criterion, n=result["n"], detail={"split": split})
        if result["spearman"] is not None:
            db.record_metric(run_id, "agreement", "spearman", result["spearman"],
                             criterion=criterion, n=result["n"], detail={"split": split})
        db.record_metric(run_id, "agreement", "exact_match", result["exact_match"],
                         criterion=criterion, n=result["n"], detail={"split": split})

    ceiling, labelers = self_agreement()
    if ceiling:
        print(f"\nhuman self-agreement (the ceiling) — labelers: {', '.join(labelers)}")
        for criterion, result in ceiling.items():
            print(f"{criterion:<16}{result['weighted_kappa']:>10.3f}{'':>10}"
                  f"{result['exact_match']:>8.2f}{result['n']:>6}")
            db.record_metric(run_id, "agreement", "self_kappa", result["weighted_kappa"],
                             criterion=criterion, n=result["n"])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=int, help="judge run id (default: latest)")
    p.add_argument("--split", default="test", help="report agreement on test; tune the rubric on dev")
    p.add_argument("--disagreements", type=int, default=0, metavar="N", help="write the top N disagreements")
    p.add_argument("--self-agreement", action="store_true")
    a = p.parse_args()
    if a.self_agreement:
        result, who = self_agreement()
        for c, r in result.items():
            print(f"{c:<16}kappa_w={r['weighted_kappa']:.3f}  exact={r['exact_match']:.2f}  n={r['n']}")
    else:
        run_id = a.run or db.latest_run_id()
        report(run_id, a.split)
        if a.disagreements:
            write_disagreement_report(run_id, "dev", a.disagreements)
