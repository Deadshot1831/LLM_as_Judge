"""Judge-to-human agreement, and the human's own ceiling.

  python -m judge.agreement --run 3            # report agreement for a judge run
  python -m judge.agreement --self-agreement   # intra-rater ceiling, no judge needed
  python -m judge.agreement --consistency 3 4  # the judge's own test-retest ceiling
  python -m judge.agreement --summary          # fill the README's results table

Weighted Cohen's kappa (quadratic weights) is the headline because the scores are
ordinal — a 3-vs-1 disagreement should cost more than 3-vs-2. Spearman is reported
alongside it because kappa punishes a judge that is consistently offset by one point
even when its ranking is perfect, and you want to see both failure modes separately.
"""
import argparse
import datetime as dt
import pathlib
import random
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


def bootstrap_ci(xs, ys, draws=1000, alpha=0.05, seed=0):
    """Percentile bootstrap over item pairs.

    A kappa of 0.61 on 144 items is not the same claim as a kappa of 0.61 on 1,440, and
    "v2 beat v1" is only a finding if the intervals do not overlap. Reporting the point
    estimate alone is the overclaiming this project exists to argue against.
    """
    if len(xs) < 10:
        return None
    rng = random.Random(seed)
    n, values = len(xs), []
    for _ in range(draws):
        idx = [rng.randrange(n) for _ in range(n)]
        a = [xs[i] for i in idx]
        b = [ys[i] for i in idx]
        if len(set(a)) < 2 and len(set(b)) < 2:
            continue                      # a degenerate resample has no kappa to speak of
        value = cohen_kappa_score(a, b, weights="quadratic")
        if value == value:                # skip nan draws, which carry no information
            values.append(value)
    if len(values) < draws // 2:
        return None
    values.sort()
    lo = values[int(alpha / 2 * len(values))]
    hi = values[min(int((1 - alpha / 2) * len(values)), len(values) - 1)]
    return lo, hi


def agreement(xs, ys, ci=False):
    """Kappa and Spearman, or None where they are genuinely undefined.

    Chance-corrected agreement needs variance to correct against: if both sides scored
    every item a 3, sklearn returns nan and it is right to. Reporting that as 1.0 would
    claim perfect agreement on a criterion that has never once discriminated between two
    answers, and storing the raw nan puts it silently on a chart. Both are worse than
    saying undefined and quoting the exact-match rate instead.
    """
    if len(xs) < 2:
        return None
    kappa = cohen_kappa_score(xs, ys, weights="quadratic")
    if kappa != kappa:                      # nan: no variance to correct against
        kappa = None
    # Spearman is undefined when either side is constant; report it as None rather than nan.
    rho = spearmanr(xs, ys).statistic if len(set(xs)) > 1 and len(set(ys)) > 1 else None
    out = {"weighted_kappa": kappa, "spearman": rho, "n": len(xs),
           "degenerate": kappa is None,
           "exact_match": sum(x == y for x, y in zip(xs, ys)) / len(xs)}
    if ci:
        out["ci"] = bootstrap_ci(xs, ys) if kappa is not None else None
    return out


def strata():
    return {r["id"]: r["stratum"] for r in db.query("SELECT id, stratum FROM items")}


def by_stratum(run_id, split, min_n=10):
    """Where the judge actually fails.

    A single averaged kappa hides the shape that matters: agreement is usually fine on easy
    items and falls apart on adversarial ones. Averaging those together produces a number
    that describes neither.
    """
    human, judged = human_consensus(split=split), judge_scores(run_id)
    label = strata()
    out = {}
    for criterion in sorted({c for v in human.values() for c in v}):
        items = [i for i in sorted(set(human) & set(judged))
                 if criterion in human[i] and criterion in judged[i]]
        for stratum in sorted({label[i] for i in items}):
            picked = [i for i in items if label[i] == stratum]
            if len(picked) < min_n:
                continue
            result = agreement([human[i][criterion] for i in picked],
                               [judged[i][criterion] for i in picked])
            if result:
                out[(criterion, stratum)] = result
    return out


def consistency(run_a, run_b):
    """The judge's own ceiling: same items, same rubric, two runs at temperature 0.

    Humans get a self-agreement number and so should the judge. A judge that disagrees with
    itself cannot agree with anyone else, and the gap between this and the human ceiling is
    how much headroom rubric work still has.
    """
    a, b = judge_scores(run_a), judge_scores(run_b)
    shared = set(a) & set(b)
    if not shared:
        raise SystemExit(f"runs {run_a} and {run_b} share no items")
    return {c: agreement(*paired(a, b, c), ci=True)
            for c in sorted({k for i in shared for k in a[i]})}


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


def report(run_id, split, ci=True):
    human, judge = human_consensus(split=split), judge_scores(run_id)
    overlap = set(human) & set(judge)
    if not overlap:
        raise SystemExit(f"no labelled items in the {split} split are covered by run {run_id}")
    print(f"\nrun {run_id} vs humans on {split} ({len(overlap)} items)")
    print(f"{'criterion':<16}{'kappa_w':>10}{'95% CI':>18}{'spearman':>10}{'exact':>8}{'n':>6}")
    for criterion in sorted({c for v in human.values() for c in v}):
        xs, ys = paired(human, judge, criterion)
        result = agreement(xs, ys, ci=ci)
        if not result:
            continue
        rho = f"{result['spearman']:.3f}" if result["spearman"] is not None else "n/a"
        bounds = result.get("ci")
        span = f"[{bounds[0]:.3f}, {bounds[1]:.3f}]" if bounds else "—"
        kappa = f"{result['weighted_kappa']:.3f}" if not result["degenerate"] else "undef"
        print(f"{criterion:<16}{kappa:>10}{span:>18}{rho:>10}"
              f"{result['exact_match']:>8.2f}{result['n']:>6}")
        if result["degenerate"]:
            print(f"{'':<16}  ^ no variance on either side — nothing for kappa to correct "
                  f"against; exact match is the number to read")
        else:
            db.record_metric(run_id, "agreement", "weighted_kappa", result["weighted_kappa"],
                             criterion=criterion, n=result["n"],
                             detail={"split": split,
                                     "ci_low": bounds[0] if bounds else None,
                                     "ci_high": bounds[1] if bounds else None})
        if result["spearman"] is not None:
            db.record_metric(run_id, "agreement", "spearman", result["spearman"],
                             criterion=criterion, n=result["n"], detail={"split": split})
        db.record_metric(run_id, "agreement", "exact_match", result["exact_match"],
                         criterion=criterion, n=result["n"], detail={"split": split})

    sliced = by_stratum(run_id, split)
    if sliced:
        print(f"\nby stratum — a single average describes none of these")
        print(f"{'criterion':<16}{'stratum':<14}{'kappa_w':>10}{'n':>6}")
        for (criterion, stratum), result in sliced.items():
            kappa = f"{result['weighted_kappa']:.3f}" if not result["degenerate"] else "undef"
            print(f"{criterion:<16}{stratum:<14}{kappa:>10}{result['n']:>6}")
            if not result["degenerate"]:
                db.record_metric(run_id, "agreement", "weighted_kappa_by_stratum",
                                 result["weighted_kappa"], criterion=criterion, n=result["n"],
                                 detail={"split": split, "stratum": stratum})

    ceiling, labelers = self_agreement()
    if ceiling:
        print(f"\nhuman self-agreement (the ceiling) — labelers: {', '.join(labelers)}")
        for criterion, result in ceiling.items():
            kappa = f"{result['weighted_kappa']:.3f}" if not result["degenerate"] else "undef"
            print(f"{criterion:<16}{kappa:>10}{'':>18}{'':>10}"
                  f"{result['exact_match']:>8.2f}{result['n']:>6}")
            if not result["degenerate"]:
                db.record_metric(run_id, "agreement", "self_kappa", result["weighted_kappa"],
                                 criterion=criterion, n=result["n"])


def summary(split="test"):
    """Write reports/summary.md — the results table, filled from the database.

    Transcribing numbers into a README by hand is how a number ends up meaning something
    different from what was measured. This regenerates the table instead.
    """
    def latest(name, **match):
        rows = db.query(
            """SELECT m.value, m.n, m.criterion, m.detail FROM metrics m
               WHERE m.name = %s ORDER BY m.id DESC""", (name,))
        return [r for r in rows
                if all((r["detail"] or {}).get(k) == v for k, v in match.items())]

    def fmt(rows, form="{:.3f}", scale=1.0):
        if not rows:
            return "—"
        picked, seen = [], set()
        for r in rows:                      # newest row per criterion
            if r["criterion"] not in seen:
                seen.add(r["criterion"])
                picked.append(r)
        if len(picked) == 1 and picked[0]["criterion"] is None:
            return form.format(picked[0]["value"] * scale)
        return " · ".join(f"{r['criterion']} {form.format(r['value'] * scale)}"
                          for r in sorted(picked, key=lambda r: r["criterion"] or ""))

    def fmt_with_ci(rows):
        """kappa is quoted with its interval or not at all — the interval is the claim."""
        if not rows:
            return "—"
        seen, parts = set(), []
        for r in rows:
            if r["criterion"] in seen:
                continue
            seen.add(r["criterion"])
            d = r["detail"] or {}
            span = (f" [{d['ci_low']:.2f}–{d['ci_high']:.2f}]"
                    if d.get("ci_low") is not None else "")
            parts.append((r["criterion"] or "", f"{r['criterion']} {r['value']:.3f}{span}"))
        return " · ".join(text for _, text in sorted(parts))

    labels = db.query("SELECT count(*) AS n FROM human_labels WHERE pass_no = 1")[0]["n"]
    rows = [
        ("Human labels collected", str(labels), "`make label`"),
        (f"Judge↔human weighted κ [95% CI], {split} split",
         fmt_with_ci(latest("weighted_kappa", split=split)), "`make agreement`"),
        (f"Spearman ρ, {split} split", fmt(latest("spearman", split=split)), "`make agreement`"),
        ("Human self-agreement κ — **the ceiling**", fmt(latest("self_kappa")),
         "`python -m judge.agreement --self-agreement`"),
        ("Judge self-consistency κ", fmt(latest("judge_consistency")),
         "`python -m judge.agreement --consistency A B`"),
        ("Position-bias flip rate", fmt(latest("position_flip_rate"), "{:.1f}%", 100), "`make bias`"),
        ("Length-bias ρ(chars, score)", fmt(latest("length_spearman"), "{:+.3f}"), "`make bias`"),
        ("Self-preference rate (50% = neutral)",
         fmt(latest("self_preference_rate"), "{:.1f}%", 100), "`make bias`"),
    ]
    REPORTS.mkdir(exist_ok=True)
    path = REPORTS / "summary.md"
    lines = ["| Metric | Value | Produced by |", "|---|---|---|"]
    lines += [f"| {a} | {b} | {c} |" for a, b, c in rows]
    lines.append("")
    lines.append(f"_Generated {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M UTC} from "
                 f"{labels} first-pass labels. Agreement is reported on the {split} split; "
                 "the rubric is iterated on dev._")
    path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {path} — paste it over the Results table in the README")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=int, help="judge run id (default: latest)")
    p.add_argument("--split", default="test", help="report agreement on test; tune the rubric on dev")
    p.add_argument("--disagreements", type=int, default=0, metavar="N", help="write the top N disagreements")
    p.add_argument("--self-agreement", action="store_true")
    p.add_argument("--consistency", nargs=2, type=int, metavar=("RUN_A", "RUN_B"),
                   help="two judge runs over the same items: the judge's own ceiling")
    p.add_argument("--summary", action="store_true", help="write reports/summary.md")
    p.add_argument("--no-ci", action="store_true", help="skip the bootstrap, which is the slow part")
    a = p.parse_args()
    if a.summary:
        summary(a.split)
    elif a.consistency:
        run_a, run_b = a.consistency
        print(f"judge self-consistency: run {run_a} vs run {run_b}")
        for c, r in consistency(run_a, run_b).items():
            bounds = r.get("ci")
            span = f"  95% CI [{bounds[0]:.3f}, {bounds[1]:.3f}]" if bounds else ""
            kappa = f"{r['weighted_kappa']:.3f}" if not r["degenerate"] else "undef"
            print(f"{c:<16}kappa_w={kappa}  exact={r['exact_match']:.2f}  n={r['n']}{span}")
            if not r["degenerate"]:
                db.record_metric(run_b, "agreement", "judge_consistency", r["weighted_kappa"],
                                 criterion=c, n=r["n"], detail={"vs_run": run_a})
    elif a.self_agreement:
        result, who = self_agreement()
        for c, r in result.items():
            kappa = f"{r['weighted_kappa']:.3f}" if not r["degenerate"] else "undef"
            print(f"{c:<16}kappa_w={kappa}  exact={r['exact_match']:.2f}  n={r['n']}")
    else:
        run_id = a.run or db.latest_run_id()
        report(run_id, a.split, ci=not a.no_ci)
        if a.disagreements:
            write_disagreement_report(run_id, "dev", a.disagreements)
