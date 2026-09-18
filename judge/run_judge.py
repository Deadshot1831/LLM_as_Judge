"""The judge: score items against a rubric, store every run in Postgres.

  python -m judge.run_judge --rubric v2 --label "v2 + anchors"
  python -m judge.run_judge --rubric v2 --variant score_first   # the A/B for phase 1

`reason_first` and `score_first` differ only in the order of keys in the JSON the
model emits. Because tokens are generated in order, that ordering decides whether
the score is conditioned on the reasoning or the reasoning is a post-hoc story.
Which one actually agrees with humans is measured, not assumed.
"""
import argparse
import concurrent.futures as futures
import json
import os
import re
import time

from anthropic import Anthropic
from dotenv import load_dotenv
from psycopg.types.json import Jsonb

from judge import db, rubric as rubric_mod

load_dotenv()

SYSTEM = """You are a strict evaluator of retrieval-augmented answers. You apply the rubric
exactly as written, including its anchor examples. You do not reward length, confidence,
or fluency — only what the rubric names."""

TEMPLATE = """{rubric}

Evaluate the ANSWER below against every criterion.

QUESTION:
{question}

RETRIEVED CONTEXT (the only material the answer was allowed to use):
{context}

ANSWER:
{answer}

Reply with a single JSON object and nothing else, with keys in exactly this order:
{shape}

`reasoning` maps each criterion to one or two sentences citing the specific span or
sentence that decided the score. `scores` maps each criterion to an integer in {scale}."""

SHAPES = {
    "reason_first": '{"reasoning": {<criterion>: <string>}, "scores": {<criterion>: <int>}}',
    "score_first": '{"scores": {<criterion>: <int>}, "reasoning": {<criterion>: <string>}}',
}

PAIRWISE = """{rubric}

Two answers to the same question are shown. Decide which better satisfies the rubric overall.

QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

ANSWER A:
{a}

ANSWER B:
{b}

Reply with a single JSON object and nothing else:
{{"reasoning": <string>, "winner": "A" | "B" | "tie"}}"""


def client():
    return Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def parse_json(text):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def score_item(api, model, rubric, item, variant):
    prompt = TEMPLATE.format(
        rubric=rubric_mod.render(rubric), question=item["question"], context=item["context"],
        answer=item["answer"], shape=SHAPES[variant], scale=rubric["scale"],
    )
    start = time.time()
    msg = api.messages.create(
        model=model, max_tokens=1200, temperature=0,
        system=SYSTEM, messages=[{"role": "user", "content": prompt}],
    )
    out = parse_json(msg.content[0].text)
    names = rubric_mod.criteria(rubric)
    scores = {c: int(out["scores"][c]) for c in names}
    bad = [c for c, s in scores.items() if s not in rubric["scale"]]
    if bad:
        raise ValueError(f"{item['id']}: off-scale scores for {bad}")
    usage = (msg.usage.input_tokens, msg.usage.output_tokens)
    return scores, out.get("reasoning", {}), int((time.time() - start) * 1000), usage


def judge_pair(api, model, rubric, question, context, answer_a, answer_b):
    msg = api.messages.create(
        model=model, max_tokens=600, temperature=0, system=SYSTEM,
        messages=[{"role": "user", "content": PAIRWISE.format(
            rubric=rubric_mod.render(rubric), question=question, context=context,
            a=answer_a, b=answer_b)}],
    )
    return parse_json(msg.content[0].text)["winner"].strip().lower()


def fetch_items(splits, limit=None):
    sql = "SELECT * FROM items WHERE split = ANY(%s) ORDER BY id"
    if limit:
        sql += f" LIMIT {int(limit)}"
    return db.query(sql, (list(splits),))


def spend(run_id, tokens_in, tokens_out, calls):
    """Record what the run cost.

    A gate that runs on every pull request has a bill, and "is this affordable at our PR
    volume" is a question the numbers should answer. Prices are read from the environment
    rather than hardcoded — a rate baked into source goes stale silently and then the
    dashboard is quoting a number that was true last year.
    """
    db.record_metric(run_id, "cost", "input_tokens", tokens_in, n=calls)
    db.record_metric(run_id, "cost", "output_tokens", tokens_out, n=calls)
    price_in = os.environ.get("JUDGE_PRICE_IN_PER_MTOK")
    price_out = os.environ.get("JUDGE_PRICE_OUT_PER_MTOK")
    line = (f"{calls} calls · {tokens_in:,} in / {tokens_out:,} out tokens")
    if price_in and price_out:
        usd = tokens_in / 1e6 * float(price_in) + tokens_out / 1e6 * float(price_out)
        db.record_metric(run_id, "cost", "usd", usd, n=calls)
        line += f" · ${usd:.3f} (${usd / calls:.4f} per item)"
    else:
        line += " · set JUDGE_PRICE_IN_PER_MTOK and JUDGE_PRICE_OUT_PER_MTOK for a dollar figure"
    print(line)


def run(args):
    rub = rubric_mod.load(args.rubric)
    rubric_mod.register(rub)
    items = fetch_items(args.splits.split(","), args.limit)
    if not items:
        raise SystemExit("no items — run scripts/make_dataset.py first")

    model = args.model or os.environ.get("JUDGE_MODEL", "claude-opus-5")
    run_id = db.start_run(args.label or f"{args.rubric} / {args.variant}", rub["version"], model, args.variant)
    api = client()

    def work(item):
        for attempt in range(3):
            try:
                return item, score_item(api, model, rub, item, args.variant)
            except Exception as exc:  # ponytail: flat retry, add backoff if 429s show up
                if attempt == 2:
                    print(f"  !! {item['id']}: {exc}")
                    return item, None
                time.sleep(2 * (attempt + 1))

    done, tokens_in, tokens_out = 0, 0, 0
    with futures.ThreadPoolExecutor(args.concurrency) as pool, db.connect() as conn:
        for item, result in pool.map(work, items):
            if result is None:
                continue
            scores, reasoning, latency, usage = result
            tokens_in += usage[0]
            tokens_out += usage[1]
            conn.execute(
                """INSERT INTO judge_scores (run_id, item_id, scores, reasoning, latency_ms)
                   VALUES (%s, %s, %s, %s, %s)
                   ON CONFLICT (run_id, item_id) DO UPDATE SET scores = EXCLUDED.scores""",
                (run_id, item["id"], Jsonb(scores), Jsonb(reasoning), latency),
            )
            done += 1
            if done % 25 == 0:
                conn.commit()
                print(f"  scored {done}/{len(items)}")
        conn.commit()

    # Mean score per criterion is what CI gates on.
    for criterion in rubric_mod.criteria(rub):
        row = db.query(
            "SELECT avg((scores->>%s)::int) AS m, count(*) AS n FROM judge_scores WHERE run_id = %s",
            (criterion, run_id),
        )[0]
        db.record_metric(run_id, "score", "mean", row["m"], criterion=criterion, n=row["n"])

    if done:
        spend(run_id, tokens_in, tokens_out, done)
    print(f"run {run_id}: scored {done}/{len(items)} items with {model} ({args.variant})")
    return run_id


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rubric", default="v2")
    p.add_argument("--variant", default="reason_first", choices=list(SHAPES))
    p.add_argument("--model")
    p.add_argument("--label")
    p.add_argument("--splits", default="dev,test", help="comma separated: dev, test, bias")
    p.add_argument("--limit", type=int)
    p.add_argument("--concurrency", type=int, default=8)
    run(p.parse_args())
