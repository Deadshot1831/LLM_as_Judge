"""Refresh baseline.json from the current prompt. Run deliberately, and review the diff:
this is the number every future pull request is measured against.

  python scripts/update_baseline.py --rubric v2
"""
import argparse
import datetime as dt
import json
import os

from judge import gate

p = argparse.ArgumentParser()
p.add_argument("--rubric", default="v2")
a = p.parse_args()

cases = gate.build_cases()
normalised, _ = gate.measure(cases, a.rubric)
current = gate.means(normalised)
gate.record(a.rubric, current, label="baseline refresh")

gate.BASELINE.write_text(json.dumps({
    "rubric_version": a.rubric,
    "judge_model": os.environ.get("JUDGE_MODEL", "claude-opus-5"),
    "answerer_model": os.environ.get("ANSWERER_MODEL", "claude-sonnet-5"),
    "answerer_prompt_sha": __import__("hashlib").sha256(
        (gate.ROOT / "prompts" / "answerer.md").read_bytes()).hexdigest()[:12],
    "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    "n_cases": len(cases),
    "means": {k: round(v, 4) for k, v in current.items()},
}, indent=2) + "\n")
print(json.dumps(current, indent=2))
print(f"wrote {gate.BASELINE}")
