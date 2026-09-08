"""Preflight. `make doctor`

This pipeline depends on a database, an API key, a built dataset, a rubric and a
baseline, and each one fails differently. Checking them in order turns a psycopg
traceback three commands in into a list of what is actually missing.

Reads only. `--live` adds one cheap API call to prove the key and the model work.
"""
import argparse
import hashlib
import json
import os

from dotenv import load_dotenv

from judge import db, rubric as rubric_mod
from judge.gate import BASELINE, ROOT

load_dotenv()

OK, WARN, FAIL = "  ok  ", " warn ", " FAIL "


class Report:
    def __init__(self):
        self.failures = 0

    def line(self, status, check, detail=""):
        self.failures += status == FAIL
        print(f"[{status}] {check:<26}{detail}")

    def section(self, title):
        print(f"\n{title}")


def check_env(report):
    report.section("environment")
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        report.line(FAIL, "ANTHROPIC_API_KEY", "unset — copy .env.example to .env")
    elif not key.startswith("sk-"):
        report.line(WARN, "ANTHROPIC_API_KEY", "set, but does not look like an Anthropic key")
    else:
        report.line(OK, "ANTHROPIC_API_KEY", f"set (…{key[-4:]})")
    for name, default in [("JUDGE_MODEL", "claude-opus-5"), ("ANSWERER_MODEL", "claude-sonnet-5")]:
        report.line(OK, name, os.environ.get(name, f"{default} (default)"))
    judge_model = os.environ.get("JUDGE_MODEL", "claude-opus-5")
    answerer = os.environ.get("ANSWERER_MODEL", "claude-sonnet-5")
    if judge_model == answerer:
        report.line(WARN, "judge vs answerer",
                    "same model judging its own output — self-preference is baked in")


def check_database(report):
    report.section("database")
    url = os.environ.get("DATABASE_URL", db.DEFAULT_URL)
    try:
        with db.connect() as conn:
            version = conn.execute("SHOW server_version").fetchone()["server_version"]
        report.line(OK, "connection", f"postgres {version} at {url.rsplit('@', 1)[-1]}")
    except Exception as exc:
        report.line(FAIL, "connection", f"{type(exc).__name__}: {str(exc).strip().splitlines()[0]}")
        print("         start one with `make db`, or point DATABASE_URL at a hosted instance")
        return False

    tables = {r["table_name"] for r in db.query(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")}
    expected = {"items", "human_labels", "judge_runs", "judge_scores", "judge_pairwise",
                "rubric_versions", "metrics"}
    missing = expected - tables
    if missing:
        report.line(FAIL, "schema", f"missing {sorted(missing)} — run `python -m judge.db`")
        return False
    report.line(OK, "schema", f"{len(expected)} tables present")
    return True


def check_data(report):
    report.section("data")
    splits = {r["split"]: r["n"] for r in db.query(
        "SELECT split, count(*) AS n FROM items GROUP BY split")}
    if not splits:
        report.line(FAIL, "items", "none loaded — run `make dataset` or `make dataset-offline`")
        return
    report.line(OK, "items", " · ".join(f"{k} {v}" for k, v in sorted(splits.items())))

    strata = {r["stratum"]: r["n"] for r in db.query(
        "SELECT stratum, count(*) AS n FROM items GROUP BY stratum")}
    if strata.get("broken", 0) == 0:
        report.line(WARN, "strata", "no broken items — a set with no failures has no variance")
    else:
        report.line(OK, "strata", " · ".join(f"{k} {v}" for k, v in sorted(strata.items())))

    labelable = splits.get("dev", 0) + splits.get("test", 0)
    first = db.query("SELECT count(*) AS n FROM human_labels WHERE pass_no = 1")[0]["n"]
    second = db.query("SELECT count(*) AS n FROM human_labels WHERE pass_no = 2")[0]["n"]
    if not first:
        report.line(WARN, "human labels", f"none yet — `make label` ({labelable} to do)")
    else:
        report.line(OK, "human labels", f"{first} of {labelable} first-pass")
    if second < 20:
        report.line(WARN, "re-labels", f"{second} of 20 — without these there is no ceiling, "
                                       "and no ceiling means no way to read a kappa")
    else:
        report.line(OK, "re-labels", f"{second} second-pass")

    labelers = db.query("SELECT count(DISTINCT labeler_id) AS n FROM human_labels")[0]["n"]
    if labelers == 1:
        report.line(WARN, "labelers", "one — self-agreement is a weaker ceiling than "
                                      "agreement between two people")


def check_rubrics(report):
    report.section("rubrics")
    for version in rubric_mod.versions():
        try:
            rub = rubric_mod.load(version)
        except Exception as exc:
            report.line(FAIL, version, f"does not parse: {exc}")
            continue
        names = rubric_mod.criteria(rub)
        anchored = sum(bool(c.get("anchors")) for c in rub["criteria"])
        detail = f"{len(names)} criteria on a {len(rub['scale'])}-point scale, {anchored} anchored"
        report.line(OK if anchored or version == "v1" else WARN, version, detail)
        undefined = [c["name"] for c in rub["criteria"]
                     if set(c["points"]) != set(rub["scale"])]
        if undefined:
            report.line(FAIL, f"{version} points", f"{undefined} do not define every scale point")


def check_runs(report):
    report.section("runs")
    runs = db.query("SELECT count(*) AS n FROM judge_runs")[0]["n"]
    if not runs:
        report.line(WARN, "judge runs", "none — `make judge`")
    else:
        latest = db.query(
            """SELECT r.id, r.label, r.rubric_version, count(s.item_id) AS scored
               FROM judge_runs r LEFT JOIN judge_scores s ON s.run_id = r.id
               GROUP BY r.id ORDER BY r.id DESC LIMIT 1""")[0]
        report.line(OK, "judge runs", f"{runs}; latest #{latest['id']} "
                                      f"({latest['label']}, {latest['scored']} items)")
    variants = {r["prompt_variant"] for r in db.query("SELECT DISTINCT prompt_variant FROM judge_runs")}
    if runs and not {"reason_first", "score_first"} <= variants:
        report.line(WARN, "prompt variants", "only one order run — the reason-before-score claim "
                                             "is still an assumption, not a measurement")


def check_baseline(report):
    report.section("release gate")
    if not BASELINE.exists():
        report.line(WARN, "baseline.json", "absent — `make baseline` before the gate can run")
        return
    baseline = json.loads(BASELINE.read_text())
    report.line(OK, "baseline.json", f"{baseline['n_cases']} cases, rubric "
                                     f"{baseline['rubric_version']}, cut {baseline['generated_at']}")
    if baseline["rubric_version"] not in rubric_mod.versions():
        report.line(FAIL, "baseline rubric", f"{baseline['rubric_version']} no longer exists")
    current = hashlib.sha256((ROOT / "prompts" / "answerer.md").read_bytes()).hexdigest()[:12]
    if baseline.get("answerer_prompt_sha") != current:
        report.line(WARN, "prompt drift", "answerer.md has changed since the baseline was cut — "
                                          "the gate is measuring against a stale reference")
    else:
        report.line(OK, "prompt drift", "answerer.md matches the baseline")


def check_live(report):
    report.section("live api")
    try:
        from anthropic import Anthropic
        model = os.environ.get("JUDGE_MODEL", "claude-opus-5")
        msg = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"]).messages.create(
            model=model, max_tokens=8, messages=[{"role": "user", "content": "Reply with: ok"}])
        report.line(OK, model, f"reachable ({msg.usage.input_tokens} in / "
                               f"{msg.usage.output_tokens} out tokens)")
    except Exception as exc:
        report.line(FAIL, "api call", f"{type(exc).__name__}: {str(exc).strip().splitlines()[0]}")


def main(live=False):
    report = Report()
    print("preflight — what is ready and what is not\n" + "=" * 60)
    check_env(report)
    check_rubrics(report)
    if check_database(report):
        check_data(report)
        check_runs(report)
    check_baseline(report)
    if live:
        check_live(report)
    print("=" * 60)
    if report.failures:
        print(f"{report.failures} blocking problem(s). Warnings are things still to do, "
              "not things broken.")
    else:
        print("nothing blocking. Warnings are things still to do, not things broken.")
    return 1 if report.failures else 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--live", action="store_true", help="spend one cheap API call to prove the key")
    raise SystemExit(main(p.parse_args().live))
