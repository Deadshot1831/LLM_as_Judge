# LLM-as-Judge with Human Calibration

An LLM judge for RAG answers, calibrated against hand labels, measured for its three
known biases, and wired into CI so a pull request that weakens answer quality fails the build.

The point is not that a judge produces scores. The point is that these scores have a
**measured relationship to human judgement**, a **known ceiling**, and **quantified biases**.

## Headline numbers

Fill these by running the pipeline below; they are deliberately blank rather than
plausible, because a judge number with no provenance is exactly what this project exists
to argue against.

| | value | how |
|---|---|---|
| Human labels collected | — | `make label` |
| Judge↔human weighted Cohen κ (test split) | — | `make agreement` |
| Spearman ρ (test split) | — | `make agreement` |
| Human self-agreement κ (the ceiling) | — | `python -m judge.agreement --self-agreement` |
| Position-bias flip rate | — | `make bias` |
| Length-bias Spearman(chars, score) | — | `make bias` |
| Self-preference rate (50% = neutral) | — | `make bias` |

Agreement is reported on the **test** split only. The rubric is iterated on the **dev**
split. Tuning wording against the same examples you report agreement on is leakage, and
it is the quietest way to make this whole project meaningless.

## Setup

```bash
make install          # pip install -e ".[ci]"
cp .env.example .env  # ANTHROPIC_API_KEY, DATABASE_URL
make db               # docker compose up + apply schema.sql
```

## Phase 1 — the rubric and the judge

Four criteria, each on a **3-point ordinal scale with every point defined in words**:
`groundedness`, `completeness`, `directness`, `safety`. A 1–10 scale produces noise you
spend a week chasing; three points you can define is worth more than ten you cannot.

- [`rubrics/v1.yaml`](rubrics/v1.yaml) — the thin first draft. Frozen: it is the baseline of the improvement curve.
- [`rubrics/v2.yaml`](rubrics/v2.yaml) — sharpened wording plus an anchor example per score point.

The judge returns a single JSON object. `reason_first` emits `reasoning` before `scores`;
`score_first` reverses them. Because tokens are generated in order, that ordering decides
whether the score is conditioned on the reasoning or the reasoning is a post-hoc story.
Run both and compare κ — measure it rather than assume it:

```bash
python -m judge.run_judge --rubric v2 --variant reason_first
python -m judge.run_judge --rubric v2 --variant score_first
```

## Phase 2 — the gold standard

```bash
make dataset          # real answers from two Claude models + injected defects
make dataset-offline  # same shape, no API key, derived from reference answers
make label            # Streamlit labeling UI
```

192 items, stratified so the set actually has variance:

| stratum | n | what it is |
|---|---|---|
| easy | 32 | context answers the question cleanly |
| hard | 40 | needs inference, or the context is partial |
| adversarial | 24 | false premise, unanswerable, or contradictory sources |
| broken | 96 | deliberate defects: contradiction, hallucinated specific, truncation, unsafe advice |

Defects are injected programmatically ([`scripts/make_dataset.py`](scripts/make_dataset.py)),
so the broken stratum has ground truth about *what* is wrong with it. 48 further items form a
`bias` split — padded twins of real answers, identical content at roughly 3× the length — which
are judged but never hand-labelled.

The labeling UI hides the stratum and the injected defect. A labeler told an answer is
broken will find it broken.

**Label 20 items a second time, on a different day.** `pass 2` in the sidebar re-serves them
without showing what you said the first time, and refuses items labelled less than 12 hours
ago — same-day re-labelling measures memory, not consistency. That number is your ceiling:
the judge cannot be more consistent than the humans defining correct.

Everything lands in Postgres with labeler id and timestamp ([`schema.sql`](schema.sql)).

## Phase 3 — measure agreement, then close the gaps

```bash
make agreement   # weighted kappa + spearman, and writes the top 20 disagreements
```

Weighted Cohen's κ (quadratic) is the headline because the scale is ordinal — a 3-vs-1
disagreement should cost more than a 3-vs-2. Spearman is reported beside it because κ
punishes a judge that is consistently one point low even when its ranking is perfect, and
those are different problems with different fixes. [`tests/test_analysis.py`](tests/test_analysis.py)
pins that distinction so the two numbers cannot quietly collapse into one.

`--disagreements 20` writes `reports/disagreements_run<N>_dev.md`: the 20 largest gaps with
the judge's own reasoning and a three-box checklist per case — *rubric ambiguous / judge
wrong / human wrong*. Read them one at a time. In nearly every case the rubric is ambiguous
rather than the model being wrong, which is the single most useful lesson here.

Then sharpen the wording, add anchors, bump the version, re-run, re-measure. The κ per
rubric version is charted in the dashboard against the human ceiling — that improvement
curve is the deliverable, not any single κ.

## Phase 4 — the three biases

```bash
make bias
```

- **Position** — every pair judged in both orders. Flip rate above ~10% means pairwise
  verdicts from this setup are not trustworthy yet.
- **Length** — quality held constant, characters tripled by padding that adds no content.
  If the score rises, the rubric is rewarding volume. Reported as Spearman(chars, mean
  score) plus the per-criterion padded-minus-base delta, so you can see *which* criterion leaks.
- **Self-preference** — the same pairs judged by two model families, then swapped. If each
  judge prefers its own family's answers, neither verdict is clean.

Caveat worth stating out loud: with `--offline` there is only one answer family, so the
self-preference number is degenerate. It needs two genuinely different answer-producing
families to mean anything, and `GENERATOR_MODELS` is where you set them.

## Phase 5 — gate releases on calibrated scores

The application under evaluation is [`prompts/answerer.md`](prompts/answerer.md), deliberately
a single file so that weakening it is a visible one-line diff.

```bash
make baseline   # writes baseline.json from the current prompt — review this diff
make gate       # pytest tests/test_eval_gate.py
```

The judge is exposed as DeepEval metrics ([`judge/deepeval_metric.py`](judge/deepeval_metric.py)),
one per criterion, normalised to 0–1 and cached by answer hash so a four-criterion gate costs
one API call per case. [`.github/workflows/eval.yml`](.github/workflows/eval.yml) runs it on
every pull request against a Postgres service.

Explicit thresholds ([`judge/gate.py`](judge/gate.py)):

- no criterion may drop more than **0.05** normalised (one tenth of a rubric point)
- **`safety` may not drop at all**, and no single answer may score the lowest safety point

Every run is stored in Postgres, so quality has a trend line rather than an anecdote:

```bash
make dashboard   # agreement curve vs human ceiling, bias numbers, score drift
```

### The demo

```bash
git checkout -b weaken-the-prompt
# delete the "no preamble" and the high-stakes-qualification rules
sed -i '' '/No preamble/,+1d;/medical, legal or financial/,+1d' prompts/answerer.md
git commit -am "simplify the answerer prompt" && gh pr create
```

CI answers the fixed eval set with the weakened prompt, scores it with the calibrated judge,
and fails on the `directness` and `safety` drop.

## Layout

```
rubrics/         versioned rubrics; v1 is frozen as the baseline of the curve
prompts/         the answerer prompt — the thing CI actually gates
data/corpus.yaml 24 seed questions with retrieved context and reference answers
scripts/         dataset builder, baseline refresher
judge/           run_judge · agreement · bias · gate · deepeval_metric · db
app/             label.py (Streamlit labeler) · dashboard.py (trends)
tests/           test_analysis.py (offline, no key) · test_eval_gate.py (the CI gate)
```

## Known gaps

- The headline table is empty until someone labels the set. That is the honest state: the
  machinery is built and tested, the human labels are human work.
- `judge/db.py`, the labeling UI and the dashboard are written against Postgres but have not
  been exercised against a live server here — no Docker daemon and no local Postgres on the
  build machine. `make db && make dataset-offline` is the one command that confirms them.
- Self-preference needs two answer-producing model families to be meaningful (see Phase 4).

---

**Resume line:** Built an LLM-as-judge evaluation pipeline calibrated against N human labels
at weighted κ X; quantified position, length and self-preference bias, and gated releases on
it in CI.
