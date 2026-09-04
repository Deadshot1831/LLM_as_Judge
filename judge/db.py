"""Postgres access. Plain SQL, no ORM."""
import os
import pathlib
import subprocess

import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row

load_dotenv()

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_URL = "postgresql://judge:judge@localhost:5433/judge"


def connect():
    return psycopg.connect(os.environ.get("DATABASE_URL", DEFAULT_URL), row_factory=dict_row)


def query(sql, params=()):
    with connect() as conn:
        return conn.execute(sql, params).fetchall()


def execute(sql, params=()):
    with connect() as conn:
        cur = conn.execute(sql, params)
        conn.commit()
        return cur


def init():
    """Apply schema.sql. Idempotent — every statement is IF NOT EXISTS."""
    with connect() as conn:
        conn.execute((ROOT / "schema.sql").read_text())
        conn.commit()
    print("schema applied")


def git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return None


def start_run(label, rubric_version, judge_model, prompt_variant="reason_first"):
    with connect() as conn:
        run_id = conn.execute(
            """INSERT INTO judge_runs (label, rubric_version, judge_model, prompt_variant, git_sha)
               VALUES (%s, %s, %s, %s, %s) RETURNING id""",
            (label, rubric_version, judge_model, prompt_variant, git_sha()),
        ).fetchone()["id"]
        conn.commit()
        return run_id


def record_metric(run_id, family, name, value, criterion=None, n=None, detail=None):
    from psycopg.types.json import Jsonb

    execute(
        """INSERT INTO metrics (run_id, family, name, criterion, value, n, detail)
           VALUES (%s, %s, %s, %s, %s, %s, %s)""",
        (run_id, family, name, criterion, float(value), n, Jsonb(detail) if detail else None),
    )


def latest_run_id(prompt_variant="reason_first"):
    rows = query(
        "SELECT id FROM judge_runs WHERE prompt_variant = %s ORDER BY id DESC LIMIT 1",
        (prompt_variant,),
    )
    if not rows:
        raise SystemExit("no judge runs yet — run `python -m judge.run_judge` first")
    return rows[0]["id"]


if __name__ == "__main__":
    init()
