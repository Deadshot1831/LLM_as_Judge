<h1 align="center">LLM-as-Judge with Human Calibration</h1>

<p align="center">
  An LLM judge for RAG answers — calibrated against hand labels, measured for its three
  known biases, and wired into CI so a pull request that weakens answer quality fails the build.
</p>

<p align="center">
  <a href="https://github.com/Deadshot1831/LLM_as_Judge/actions/workflows/eval.yml">
    <img alt="eval gate" src="https://github.com/Deadshot1831/LLM_as_Judge/actions/workflows/eval.yml/badge.svg"></a>
  <img alt="python" src="https://img.shields.io/badge/python-3.11%2B-blue">
  <img alt="postgres" src="https://img.shields.io/badge/postgres-17-336791">
  <img alt="judge" src="https://img.shields.io/badge/judge-claude--opus--5-b8734f">
  <img alt="deepeval" src="https://img.shields.io/badge/eval-DeepEval-6f42c1">
</p>

---

Anyone can point an LLM at an output and get a number back. The number is worthless until
you know three things about it:

1. **Does it agree with a human?** Measured here as weighted Cohen's κ against hand labels.
2. **How good could it possibly be?** Bounded by the same humans re-labelling on a
   different day. The judge cannot be more consistent than the people defining correct.
3. **Which way does it cheat?** Judges prefer the first option shown, longer answers, and
   their own model family. All three are measured, not assumed.

Everything below produces those numbers on a real task, then uses them to gate a release.

## Contents

[What it's for](#what-its-for) · [Results](#results) · [How it works](#how-it-works) ·
[Quickstart](#quickstart) · [The five phases](#the-five-phases) ·
[Design decisions](#design-decisions) · [Repo layout](#repo-layout) ·
[Configuration](#configuration) · [Commands](#commands) · [Known gaps](#known-gaps)

## What it's for

Evaluation is the part of shipping an LLM feature that teams postpone until something
breaks in front of a customer. Each row below is a job this repo already does, not a
roadmap item.

| Use it to | How |
|---|---|
| **Stop prompt edits from silently degrading quality** | The CI gate scores a fixed eval set on every pull request and fails the build on a regression. The [demo](#the-demo) opens a PR that weakens the prompt and lets CI reject it. |
| **Decide a model swap on evidence** | `python -m judge.run_judge --model <candidate>` scores the same set with a different model. Compare mean scores per criterion — and because κ is known, you also know how much to trust the comparison. |
| **Tune a RAG pipeline end to end** | Chunk size, reranker, retriever swaps show up as movement in `groundedness` and `completeness`. Recall@k tells you the passage was retrieved; this tells you the answer got better. |
| **Trust a pairwise leaderboard** | The position-bias flip rate says whether your A-vs-B verdicts are measuring quality or measuring which one you showed first. Above ~10%, they are measuring order. |
| **Retrofit a judge nobody believes** | Most teams already have an LLM scoring outputs with no human baseline, which makes every number unfalsifiable. Label 150–200 outputs, compute κ, read the 20 largest gaps. The rubric is usually the thing that is wrong. |
| **Sharpen human review guidelines** | The disagreement triage improves the wording that *human* reviewers work from, not only the judge's prompt. Ambiguity that confuses the model was confusing your reviewers first. |
| **Watch for drift in production** | Sample live traffic into the `items` table under its own split and run `--splits prod` on a schedule. Every run is stored, so the dashboard shows a trend line rather than an anecdote. |
| **Show the work** | A κ against human labels, a stated ceiling, and three measured bias numbers is a rarer artifact than years of experience. Almost nobody has quantified any of them. |

### Pointing it at your own task

Almost nothing outside these three files knows that the task is RAG answers. Summarisation,
support replies, code review comments, extraction with justifications — same machinery.

| Swap | For |
|---|---|
| [`data/corpus.yaml`](data/corpus.yaml) | Your inputs. Keep the keys — `id`, `question`, `context`, `reference_answer`, `stratum`, plus `contradiction` and `unsafe_line` on high-stakes items. Leave `context` empty for tasks with no retrieval. |
| [`rubrics/`](rubrics/) | Your criteria. Keep them ordinal, keep every point defined in words, add anchors. Any number of criteria on a 3- to 5-point scale: the names and the scale flow through the judge, the labeler's buttons and keyboard shortcuts, the gate and the dashboard on their own. |
| [`prompts/answerer.md`](prompts/answerer.md) | The prompt under evaluation — whatever produces the outputs you want to gate. |

The schema, the agreement maths, the bias tests, the labeling UI and the CI gate are all
driven by whatever criteria your rubric declares; the `safety` gate simply stands down if
your rubric has no such criterion. `make dataset-offline && make test` confirms a swap
before you spend a token on it.

The one coupling left: `judge/bias.py` finds competing answers through the `::base` and
`::padded` item ids that `scripts/make_dataset.py` produces, so a hand-built dataset needs
to keep that naming for the position and length tests to find their pairs.

## Results

Blank on purpose. These are the numbers the project exists to produce, and filling them
with plausible values would be exactly the failure it argues against. `make report`
regenerates this table from the database into `reports/summary.md`, so the numbers here are
never transcribed by hand.

| Metric | Value | Produced by |
|---|---|---|
| Human labels collected | — | `make label` |
| Judge↔human weighted κ **[95% CI]**, test split | — | `make agreement` |
| Spearman ρ, test split | — | `make agreement` |
| Human self-agreement κ — **the ceiling** | — | `python -m judge.agreement --self-agreement` |
| Judge self-consistency κ — its own ceiling | — | `python -m judge.agreement --consistency A B` |
| κ by stratum — easy vs adversarial | — | `make agreement` |
| κ improvement, rubric v1 → v2 | — | `make dashboard` |
| Position-bias flip rate | — | `make bias` |
| Length-bias ρ(chars, score) | — | `make bias` |
| Self-preference rate (50% = neutral) | — | `make bias` |

**κ is quoted with its bootstrap interval or not at all.** At 144 labelled items the 95%
interval on κ is roughly ±0.13 wide, which means a v1→v2 improvement smaller than about
0.15 is not distinguishable from resampling noise. If your improvement curve moves less
than that, the honest conclusion is *label more*, not *the rubric got better*.

Agreement is reported on the **test** split only; the rubric is iterated on **dev**.
Tuning wording against the same examples you report agreement on is leakage, and it is the
quietest way to make a project like this meaningless.

## How it works

```mermaid
flowchart TB
    subgraph build ["1 · build the evaluation set"]
        corpus["data/corpus.yaml<br/>24 seed questions + retrieved context"]
        answerer["prompts/answerer.md<br/>the application under evaluation"]
        defects["deterministic defects<br/>contradiction · hallucination<br/>truncation · unsafe advice"]
        items[("items — 192<br/>dev · test · bias")]
        corpus --> answerer --> items
        corpus --> defects --> items
    end

    subgraph measure ["2 · measure the judge against humans"]
        labeler["Streamlit labeler<br/>keyboard-driven, 2 passes"]
        judge["judge/run_judge.py<br/>rubric v1 → v2"]
        humans[("human_labels")]
        scores[("judge_scores")]
        agree["weighted κ + Spearman ρ<br/>vs human self-agreement ceiling"]
        items --> labeler --> humans --> agree
        items --> judge --> scores --> agree
        agree -->|"read the 20 largest gaps"| sharpen["sharpen rubric,<br/>add anchors, bump version"]
        sharpen -.->|"re-run, re-measure"| judge
    end

    subgraph ship ["3 · use it"]
        bias["position · length · self-preference"]
        gate["CI gate vs baseline.json<br/>safety has zero tolerance"]
        dash["Streamlit dashboard<br/>trend, not anecdote"]
        scores --> bias
        judge --> gate
        agree --> dash
        bias --> dash
        gate --> dash
    end
```

Every run — judge, bias, gate — is a row in Postgres, so quality has a trend line.

## Quickstart

```bash
make install                 # pip install -e ".[ci]"
cp .env.example .env         # ANTHROPIC_API_KEY, DATABASE_URL
make db                      # docker compose up + apply schema.sql
make doctor                  # preflight: what is ready, what is missing
make dataset                 # real answers from two Claude models + injected defects
make label                   # Streamlit labeling UI  →  localhost:8501
make judge && make agreement # score the set, then compare to your labels
```

**`make doctor` before anything else.** This pipeline depends on a database, an API key, a
built dataset, a rubric and a baseline, and each fails differently. The preflight checks them
in order and prints a list, so a missing key does not surface as a psycopg traceback three
commands later. It reads only; `python -m judge.doctor --live` adds one cheap API call to
prove the key and model work. It also catches the quiet ones — the answerer prompt having
changed since the baseline was cut, a rubric point with no definition, the judge and the
answerer being set to the same model.

<details>
<summary><b>No Docker? Postgres two other ways</b></summary><br>

The compose file is a convenience, not a requirement — anything reachable over
`DATABASE_URL` works.

```bash
# macOS, native
brew install postgresql@17 && brew services start postgresql@17
createdb judge && psql judge -c "CREATE USER judge WITH PASSWORD 'judge' SUPERUSER"
export DATABASE_URL=postgresql://judge:judge@localhost:5432/judge   # note 5432, not 5433
python -m judge.db          # apply the schema

# or hosted — Neon, Supabase, RDS: paste the connection string
export DATABASE_URL='postgresql://user:pass@host/db?sslmode=require'
python -m judge.db
```

`make doctor` confirms the connection and the schema either way.
</details>

No API key? `make dataset-offline` builds the same 192-item set from the reference answers
in the corpus, so the labeler, the dashboard and every analysis path run without spending
a token. `make test` runs the offline suite — arithmetic and both Streamlit apps — with
neither a key nor a database.

## The five phases

### 1 · A rubric worth agreeing about

Four criteria on a **3-point ordinal scale with every point defined in words**:
`groundedness`, `completeness`, `directness`, `safety`. A 1–10 scale produces noise you
will spend a week chasing; three points you can actually define are worth more than ten
you cannot.

| File | Role |
|---|---|
| [`rubrics/v1.yaml`](rubrics/v1.yaml) | The thin first draft. **Frozen** — it is the baseline of the improvement curve. |
| [`rubrics/v2.yaml`](rubrics/v2.yaml) | Sharpened wording plus an anchor example per score point. |

The judge returns one JSON object. `reason_first` emits `reasoning` before `scores`;
`score_first` reverses them. Because tokens are generated in order, that ordering decides
whether the score is conditioned on the reasoning or the reasoning is a post-hoc story for
a score already committed to. Run both and compare κ — measure it rather than assume it:

```bash
python -m judge.run_judge --rubric v2 --variant reason_first
python -m judge.run_judge --rubric v2 --variant score_first
```

### 2 · A gold standard with actual variance

```bash
make dataset            # two Claude models answer the corpus, then defects are injected
make dataset-offline    # same shape, no API key
```

192 items, stratified so the set has failures in it — a calibration set with no failures
has no variance and teaches the judge nothing:

| Stratum | n | What it is |
|---|---|---|
| easy | 32 | Context answers the question cleanly |
| hard | 40 | Needs inference, or the context is only partial |
| adversarial | 24 | False premise, unanswerable, or sources that contradict each other |
| broken | 96 | Injected defects with known ground truth |

144 of those are labelable (`dev` + `test`). The remaining 48 form a `bias` split: padded
twins of real answers, identical content at roughly 3× the length, judged but never
hand-labelled — they exist to make the length test possible.

Defects are injected programmatically in [`scripts/make_dataset.py`](scripts/make_dataset.py),
so the broken stratum has ground truth about *what* is wrong with each item, not just that
something is.

**The labeling UI** ([`app/label.py`](app/label.py)) is built for the grind — 144 items ×
4 criteria is a lot of clicking if you let it be:

- **Keyboard-first.** Digits run across the criteria in reading order — with the shipped
  4×3 rubric that is `1`–`9` for the first three criteria and `⌘/Ctrl+1`–`3` for safety —
  then `Backspace` revises the last item and `Esc` parks one for later. The legend at the
  bottom of the screen is generated from whatever rubric is loaded. Digits rather than letters,
  so a stray keystroke while writing a note is less likely to score something — and when it
  does, the highlighted button shows it rather than hiding it.
- **No save button.** The last criterion saves and advances. Turn auto-advance off in the
  sidebar when you want to write a note first.
- **The rubric is on the buttons.** Each button carries its score point's wording; the ⓘ
  popover holds the full definition and the anchor examples. Nothing needs scrolling.
- **↩ back** reopens the item you just saved with your scores pre-filled, because the answer
  to *"wait, was that a 2?"* should not be *"too late"*.
- **🚩 rubric unclear** prefixes your note with `[rubric-unclear]`. An item you could not
  decide is worth more than one you guessed on — those flags are Phase 3's raw material.
- Context and answer sit side by side, with a live pace estimate that turns the remaining
  pile into a number of minutes.

It hides the stratum and the injected defect throughout: a labeler told an answer is broken
will find it broken.

**Label 20 items a second time, on a different day.** Pass 2 in the sidebar re-serves them
without showing what you said the first time, and refuses anything labelled less than 12
hours ago — same-day re-labelling measures memory, not consistency. That number is your
ceiling.

Everything lands in Postgres with labeler id and timestamp ([`schema.sql`](schema.sql)).

### 3 · Measure agreement, then close the gaps

```bash
make agreement    # weighted κ + ρ, and writes the 20 largest disagreements
```

Weighted Cohen's κ (quadratic) is the headline because the scale is ordinal — a 3-vs-1
disagreement should cost more than a 3-vs-2. Spearman is reported beside it because κ
punishes a judge that is consistently one point low even when its ranking is perfect, and
those are different problems with different fixes. High ρ with low κ means recalibrate the
offset; low both means fix the rubric.
[`tests/test_analysis.py`](tests/test_analysis.py) pins that distinction so the two numbers
cannot quietly collapse into one.

`--disagreements 20` writes `reports/disagreements_run<N>_dev.md`: the largest gaps with the
judge's own reasoning and a three-box checklist per case — *rubric ambiguous / judge wrong /
human wrong*. The same triage is in the browser under **Disagreements** in the dashboard:
pick a row and the question, context, answer, the judge's reasoning and the labeler's note
come up side by side with the two scores.

Read them one at a time. In nearly every case the rubric is ambiguous rather than the model
being wrong, which is the single most useful lesson in this project. Then sharpen the
wording, add anchors, bump the version, re-run, re-measure. **The κ-per-rubric-version curve
is the deliverable, not any single κ.**

Three things stop that curve from being read too generously:

- **A bootstrap interval on every κ.** 1,000 resamples over item pairs, quoted beside the
  point estimate. Without it, "v2 beat v1" is an eyeball comparison of two noisy numbers.
- **κ sliced by stratum.** A single average hides the shape that matters — agreement is
  usually fine on easy items and falls apart on adversarial ones, and the mean of those two
  describes neither. Slices under 10 items are not reported at all.
- **The judge's own ceiling.** Humans get a self-agreement number, so the judge gets one
  too: run it twice over the same items at temperature 0 and compare the runs.

  ```bash
  make judge && make judge                        # two runs, same rubric, same items
  python -m judge.agreement --consistency 3 4     # kappa between them
  ```

  A judge that disagrees with itself cannot agree with anyone else, and the distance between
  this and the human ceiling is how much headroom rubric work actually has.

One trap worth naming, because it is silent: when humans and judge both score every item the
same value, κ is **undefined**, not 1.0 — there is no variance for chance-correction to
correct against. `agreement()` returns `None` there and the number is never recorded or
charted, because a criterion that has never once discriminated between two answers should not
be able to display a perfect score.

### 4 · Test the judge for its known biases

```bash
make bias
```

| Bias | Method | Line to worry about |
|---|---|---|
| **Position** | Every pair judged in both orders | Flip rate above ~10% means pairwise verdicts from this setup are not trustworthy yet |
| **Length** | Quality held constant, characters tripled by padding that adds no content | If the score moves, the rubric is rewarding volume |
| **Self-preference** | The same pairs judged by two model families, then swapped | If each judge prefers its own family, neither verdict is clean |

Length is reported as ρ(chars, mean score) *plus* the per-criterion padded-minus-base delta,
so you can see **which** criterion leaks rather than only that something does.

One caveat stated out loud: with `--offline` there is a single answer family, so the
self-preference number is degenerate. It needs two genuinely different answer-producing
families to mean anything — `GENERATOR_MODELS` is where you set them.

### 5 · Gate releases on calibrated scores

The application under evaluation is [`prompts/answerer.md`](prompts/answerer.md) —
deliberately one file, so weakening it is a visible one-line diff.

```bash
make baseline   # writes baseline.json from the current prompt — review this diff
make gate       # pytest tests/test_eval_gate.py
```

The judge is exposed as DeepEval metrics ([`judge/deepeval_metric.py`](judge/deepeval_metric.py)),
one per criterion, normalised to 0–1 and cached by answer hash so a four-criterion gate
costs one API call per test case. [`.github/workflows/eval.yml`](.github/workflows/eval.yml)
runs it on every pull request against a Postgres service.

Explicit thresholds ([`judge/gate.py`](judge/gate.py)):

- no criterion may drop more than **0.05** normalised — one tenth of a rubric point
- **`safety` may not drop at all**, and no single answer may score the lowest safety point

Every judge run records its token counts, and its dollar cost when the two price variables
are set. A gate that runs on every pull request has a bill, and "can we afford this at our PR
volume" should be answerable from the dashboard rather than from the invoice.

The dashboard turns the stored runs into trend lines:

```bash
make dashboard
```

| Tab | What it answers |
|---|---|
| **Agreement** | The improvement curve against the human ceiling, κ beside ρ, and a human-vs-judge confusion matrix per criterion — offset and noise look different at a glance |
| **Bias** | The three numbers against their pass/fail lines, plus which criterion leaks length |
| **Disagreements** | Click a gap, read the case, decide whether the rubric or the judge is wrong |
| **Runs & progress** | Score drift across runs, how much of the set is labelled, and what the runs cost |

#### The demo

```bash
git checkout -b weaken-the-prompt
sed -i '' '/No preamble/,+1d;/medical, legal or financial/,+1d' prompts/answerer.md
git commit -am "simplify the answerer prompt" && gh pr create
```

CI answers the fixed eval set with the weakened prompt, scores it with the calibrated judge,
and fails on the `directness` and `safety` drop.

## Design decisions

<details>
<summary><b>Why a 3-point scale and not 1–10</b></summary><br>

Every point has to be defined in words, and a labeler has to apply that definition
consistently across two sittings. Ten points cannot be defined distinctly, so labelers drift
between them and the judge inherits the noise. Three points that mean something beat ten
that do not — and the anchors in `rubrics/v2.yaml` are what make even three stick.
</details>

<details>
<summary><b>Why the dev/test split exists</b></summary><br>

The obvious way to run this project is to iterate the rubric until κ looks good, then report
that κ. That number is fitted to the examples you tuned on and will not survive contact with
new data. Seeds hash to `dev` or `test`; the disagreement report only ever pulls from `dev`,
and the headline is only ever reported on `test`.
</details>

<details>
<summary><b>Why reasoning-before-score is a measurement, not an assumption</b></summary><br>

"Chain of thought improves consistency" is repeated everywhere and tested almost nowhere. The
two prompt variants differ only in JSON key order, which is enough to control whether the
score is generated before or after the reasoning. Both are stored as separate runs, so the
claim is answerable from your own data.
</details>

<details>
<summary><b>Why the judge never scores its own answers by default</b></summary><br>

`GENERATOR_MODELS` produces the answers and `JUDGE_MODEL` scores them, and the defaults keep
them different. Self-preference is then measured explicitly rather than being quietly built
into every number in the repo.
</details>

<details>
<summary><b>Why defects are injected programmatically</b></summary><br>

A hand-written "bad answer" is bad in a way you already believe in. Deterministic corruption
of a real model answer — appending a contradiction, cutting it mid-sentence, padding it with
preamble — produces failures with known ground truth about *what* is wrong, which is what
makes the broken stratum diagnostic instead of decorative.
</details>

## Repo layout

```
rubrics/            versioned rubrics; v1 frozen as the baseline of the improvement curve
prompts/answerer.md the application under evaluation — the file CI actually gates
data/corpus.yaml    24 seed questions with retrieved context and reference answers
data/items.jsonl    the built evaluation set (192 items, committed so the repo runs offline)
scripts/            make_dataset.py · update_baseline.py
judge/
  answerer.py       the RAG answerer under test
  rubric.py         load, render and version rubrics
  run_judge.py      the judge: pointwise scoring and pairwise verdicts
  agreement.py      weighted κ, Spearman, self-agreement, disagreement reports
  bias.py           position, length and self-preference tests
  gate.py           thresholds and baseline comparison
  deepeval_metric.py the judge as DeepEval metrics
  db.py             Postgres access, plain SQL
app/
  label.py          keyboard-driven labeling UI
  dashboard.py      agreement curve, bias numbers, disagreement triage
tests/
  test_analysis.py  offline — the arithmetic the headline numbers rest on
  test_ui_smoke.py  offline — renders both Streamlit apps against an in-memory database
  test_eval_gate.py the CI gate itself
schema.sql          items · human_labels · judge_runs · judge_scores · judge_pairwise · metrics
```

## Configuration

`.env`, from [`.env.example`](.env.example):

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Required for judging and for generating answers |
| `DATABASE_URL` | `postgresql://judge:judge@localhost:5433/judge` | Matches `docker-compose.yml` |
| `JUDGE_MODEL` | `claude-opus-5` | The judge |
| `ANSWERER_MODEL` | `claude-sonnet-5` | The application under evaluation |
| `GENERATOR_MODELS` | `claude-sonnet-5,claude-haiku-4-5-20251001` | Competing answer families, for pairwise and self-preference |
| `JUDGE_PRICE_IN_PER_MTOK` | unset | Optional. Set with the next row and every run records its dollar cost |
| `JUDGE_PRICE_OUT_PER_MTOK` | unset | Left unset, runs record token counts only — a price hardcoded in source goes stale silently |

## Commands

| Command | What it does |
|---|---|
| `make install` | `pip install -e ".[ci]"` |
| `make doctor` | Preflight — database, key, data, rubrics, runs, baseline drift |
| `make db` | Start Postgres and apply `schema.sql` |
| `make dataset` / `make dataset-offline` | Build the 192-item evaluation set, with or without the API |
| `make label` | Labeling UI |
| `make judge` | Score every split with rubric v2 |
| `make agreement` | κ, ρ, the human ceiling, and the top-20 disagreement report |
| `make bias` | All three bias tests |
| `make report` | Regenerate the results table from the database into `reports/summary.md` |
| `make baseline` | Refresh `baseline.json` from the current prompt |
| `make gate` | Run the CI gate locally |
| `make dashboard` | Trends and triage |
| `make test` | Offline suite — no API key, no database |

## Known gaps

- **The results table is empty until someone labels the set.** That is the honest state: the
  machinery is built and tested, the human labels are human work.
- **Not yet run against a live Postgres.** Both Streamlit apps render headlessly in
  `tests/test_ui_smoke.py` against an in-memory stand-in, and every embedded query is
  syntax-checked, but no server was available on the build machine.
  `make db && make dataset-offline` is the one command that confirms it.
- **Self-preference needs two answer families** to be meaningful — see Phase 4.
- **Single labeler by default.** The schema supports several (`labeler_id` is on every row,
  and consensus is a median), but inter-rater agreement between *different* people is a
  stronger ceiling than one person's self-agreement, and it needs a second person.

---

**Resume line:** Built an LLM-as-judge evaluation pipeline calibrated against N human labels
at weighted κ X; quantified position, length and self-preference bias, and gated releases on
it in CI.
