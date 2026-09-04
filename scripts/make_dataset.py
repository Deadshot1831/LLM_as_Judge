"""Build the evaluation set: real model answers plus deliberately broken variants.

  python scripts/make_dataset.py            # generate answers with the Claude API
  python scripts/make_dataset.py --offline  # no API; derive arms from reference answers

Stratification is the point: a calibration set with no failures has no variance.
Every item lands in a split — `dev` for rubric tuning, `test` for the agreement
number you report. Splitting by seed keeps variants of one question on one side.
"""
import argparse
import hashlib
import json
import os
import pathlib
import re

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus.yaml"
OUT = ROOT / "data" / "items.jsonl"

PREAMBLE = (
    "That's a great question, and I'm happy to help you work through it. "
    "Let me carefully review the retrieved context to make sure I give you an accurate picture. "
    "Having looked at the material provided, here is what I found. "
)
CLOSER = (
    " I hope this helps clarify things for you. Please don't hesitate to reach out if you have "
    "any other questions about this or anything else — I'm always happy to assist further."
)
HALLUCINATION = (
    " According to section 4.2 of the same document, this was revised in March 2019 after an "
    "internal audit found a 47% non-compliance rate."
)


def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def stable_pick(key, options, k):
    """Deterministic per-item choice, so re-running the builder does not reshuffle the set."""
    h = int(hashlib.sha256(key.encode()).hexdigest(), 16)
    picked, pool = [], list(options)
    while pool and len(picked) < k:
        picked.append(pool.pop(h % len(pool)))
        h //= max(len(pool), 1) + 1
    return picked


def corrupt(kind, answer, seed):
    if kind == "contradiction":
        return f"{answer.strip()} {seed['contradiction'].strip()}"
    if kind == "hallucination":
        return answer.strip() + HALLUCINATION
    if kind == "truncation":
        sents = sentences(answer)
        keep = sents[: max(1, int(len(sents) * 0.55))]
        return " ".join(keep).rsplit(" ", 3)[0]  # cut mid-sentence
    if kind == "unsafe_advice":
        return f"{answer.strip()} {seed['unsafe_line'].strip()}"
    raise ValueError(kind)


def generate(models, seeds):
    """Real outputs from the application under evaluation — same prompt CI gates on."""
    from anthropic import Anthropic

    from judge.answerer import answer

    api = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    for seed in seeds:
        for model in models:
            yield seed, model, answer(seed["question"], seed["context"], model=model, api=api)


def generate_offline(seeds):
    """Two arms without an API: the reference answer, and one with its last sentence dropped."""
    for seed in seeds:
        ref = " ".join(sentences(seed["reference_answer"]))
        yield seed, "reference", ref
        weak = sentences(ref)
        yield seed, "reference-weak", " ".join(weak[:-1]) if len(weak) > 1 else weak[0]


def build(args):
    seeds = yaml.safe_load(CORPUS.read_text())
    models = [m.strip() for m in os.environ.get(
        "GENERATOR_MODELS", "claude-sonnet-5,claude-haiku-4-5-20251001").split(",")]
    source = generate_offline(seeds) if args.offline else generate(models, seeds)

    items = []
    for seed, model, answer in source:
        short = model.split("-")[1] if model.startswith("claude") else model
        split = "test" if int(hashlib.sha256(seed["id"].encode()).hexdigest(), 16) % 2 else "dev"
        base = f"{seed['id']}::{short}"

        def add(suffix, text, stratum, defect=None, item_split=split):
            items.append({
                "id": f"{base}::{suffix}", "question_id": seed["id"], "question": seed["question"],
                "context": seed["context"].strip(), "answer": text.strip(), "model": model,
                "stratum": stratum, "defect": defect, "split": item_split,
            })

        add("base", answer, seed["stratum"])

        kinds = ["contradiction", "hallucination", "truncation"]
        if seed.get("unsafe_line"):
            kinds.append("unsafe_advice")
        for kind in stable_pick(base, kinds, 2):
            add(kind, corrupt(kind, answer, seed), "broken", kind)

        # Length probe: identical content, ~3x the characters. Judged, not hand-labelled.
        add("padded", PREAMBLE + answer.strip() + CLOSER, seed["stratum"], "padding", "bias")

    OUT.write_text("\n".join(json.dumps(i) for i in items) + "\n")
    print(f"wrote {len(items)} items to {OUT}")
    for key in ("stratum", "split"):
        counts = {}
        for i in items:
            counts[i[key]] = counts.get(i[key], 0) + 1
        print(f"  {key}: {counts}")

    if not args.no_db:
        load()


def load():
    from judge import db

    items = [json.loads(line) for line in OUT.read_text().splitlines() if line.strip()]
    with db.connect() as conn:
        for i in items:
            conn.execute(
                """INSERT INTO items (id, question_id, question, context, answer, model, stratum, defect, split)
                   VALUES (%(id)s, %(question_id)s, %(question)s, %(context)s, %(answer)s,
                           %(model)s, %(stratum)s, %(defect)s, %(split)s)
                   ON CONFLICT (id) DO UPDATE SET answer = EXCLUDED.answer, split = EXCLUDED.split""",
                i,
            )
        conn.commit()
    print(f"loaded {len(items)} items into postgres")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--offline", action="store_true", help="no API calls; derive arms from reference answers")
    p.add_argument("--no-db", action="store_true", help="write jsonl only")
    p.add_argument("--load-only", action="store_true", help="load an existing items.jsonl into postgres")
    a = p.parse_args()
    load() if a.load_only else build(a)
