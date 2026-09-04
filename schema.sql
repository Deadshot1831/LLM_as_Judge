-- Idempotent. Applied by docker-compose initdb and by `python -m judge.db init`.

CREATE TABLE IF NOT EXISTS items (
    id          TEXT PRIMARY KEY,
    question_id TEXT NOT NULL,          -- items sharing a question_id are competing answers
    question    TEXT NOT NULL,
    context     TEXT NOT NULL,          -- retrieved passages the answer had to use
    answer      TEXT NOT NULL,
    model       TEXT NOT NULL,          -- which model produced the answer
    stratum     TEXT NOT NULL,          -- easy | hard | adversarial | broken
    defect      TEXT,                   -- known injected defect, NULL if none
    split       TEXT NOT NULL DEFAULT 'dev',  -- dev = rubric tuning, test = reported agreement
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS rubric_versions (
    version    TEXT PRIMARY KEY,
    body       TEXT NOT NULL,           -- the rubric yaml, verbatim
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One row per labeling event: a labeler scoring one item once.
-- pass_no = 2 is the deliberate re-label used for intra-rater reliability.
CREATE TABLE IF NOT EXISTS human_labels (
    id         BIGSERIAL PRIMARY KEY,
    item_id    TEXT NOT NULL REFERENCES items(id),
    labeler_id TEXT NOT NULL,
    pass_no    INT  NOT NULL DEFAULT 1,
    scores     JSONB NOT NULL,          -- {criterion: int}
    note       TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (item_id, labeler_id, pass_no)
);

CREATE TABLE IF NOT EXISTS judge_runs (
    id             BIGSERIAL PRIMARY KEY,
    label          TEXT NOT NULL,       -- human-readable, e.g. "rubric v3 + anchors"
    rubric_version TEXT NOT NULL,
    judge_model    TEXT NOT NULL,
    prompt_variant TEXT NOT NULL DEFAULT 'reason_first',  -- reason_first | score_first
    git_sha        TEXT,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS judge_scores (
    run_id     BIGINT NOT NULL REFERENCES judge_runs(id) ON DELETE CASCADE,
    item_id    TEXT   NOT NULL REFERENCES items(id),
    scores     JSONB  NOT NULL,
    reasoning  JSONB,
    latency_ms INT,
    PRIMARY KEY (run_id, item_id)
);

-- Pairwise verdicts, used for position bias and self-preference.
CREATE TABLE IF NOT EXISTS judge_pairwise (
    id          BIGSERIAL PRIMARY KEY,
    run_id      BIGINT NOT NULL REFERENCES judge_runs(id) ON DELETE CASCADE,
    question_id TEXT   NOT NULL,
    first_item  TEXT   NOT NULL REFERENCES items(id),
    second_item TEXT   NOT NULL REFERENCES items(id),
    winner      TEXT   NOT NULL,        -- first | second | tie
    UNIQUE (run_id, first_item, second_item)
);

-- Flat results table so the dashboard can chart anything over time.
CREATE TABLE IF NOT EXISTS metrics (
    id        BIGSERIAL PRIMARY KEY,
    run_id    BIGINT NOT NULL REFERENCES judge_runs(id) ON DELETE CASCADE,
    family    TEXT NOT NULL,            -- agreement | bias | score
    name      TEXT NOT NULL,            -- e.g. weighted_kappa, position_flip_rate
    criterion TEXT,                     -- NULL when the metric is not per-criterion
    value     DOUBLE PRECISION NOT NULL,
    n         INT,
    detail    JSONB
);
CREATE INDEX IF NOT EXISTS metrics_lookup ON metrics (family, name, criterion);
