"""Renders both Streamlit apps headlessly against an in-memory stand-in for Postgres.

Catches the failure mode that a compile check cannot: a widget called with arguments
this Streamlit does not accept, or a code path that only runs once there is data.
No database, no API key. `pytest tests/test_ui_smoke.py`
"""
import datetime as dt
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from streamlit.testing.v1 import AppTest  # noqa: E402

from judge import db  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
NOW = dt.datetime(2026, 9, 1, tzinfo=dt.timezone.utc)

ITEMS = [
    {"id": "refund-window::reference::base", "question_id": "refund-window", "split": "test",
     "question": "How long for a refund?", "context": "Refunds within 30 days of invoice.",
     "answer": "Refunds may be requested within 30 days of the invoice date.",
     "model": "reference", "stratum": "easy", "defect": None},
    {"id": "refund-window::reference::contradiction", "question_id": "refund-window", "split": "test",
     "question": "How long for a refund?", "context": "Refunds within 30 days of invoice.",
     "answer": "Refunds within 30 days. Annual plans carry a 90-day window.",
     "model": "reference", "stratum": "broken", "defect": "contradiction"},
    {"id": "pto-carryover::reference::base", "question_id": "pto-carryover", "split": "dev",
     "question": "How many days carry over?", "context": "Up to 5 days, expiring 31 March.",
     "answer": "Up to 5 unused days carry over and expire on 31 March.",
     "model": "reference", "stratum": "easy", "defect": None},
]
CRITERIA = ["groundedness", "completeness", "directness", "safety"]
LABELS = [
    {"item_id": ITEMS[0]["id"], "labeler_id": "yy", "pass_no": 1, "created_at": NOW,
     "scores": {"groundedness": 3, "completeness": 3, "directness": 3, "safety": 3},
     "note": "clean"},
    {"item_id": ITEMS[1]["id"], "labeler_id": "yy", "pass_no": 1, "created_at": NOW,
     "scores": {"groundedness": 1, "completeness": 2, "directness": 3, "safety": 3},
     "note": "[rubric-unclear] is a contradiction also a completeness failure?"},
    {"item_id": ITEMS[0]["id"], "labeler_id": "yy", "pass_no": 2, "created_at": NOW,
     "scores": {"groundedness": 3, "completeness": 2, "directness": 3, "safety": 3}, "note": None},
]
JUDGED = {
    ITEMS[0]["id"]: {"groundedness": 3, "completeness": 3, "directness": 3, "safety": 3},
    ITEMS[1]["id"]: {"groundedness": 3, "completeness": 2, "directness": 2, "safety": 3},
}
WRITES = []


def fake_query(sql, params=()):
    s = " ".join(sql.split())
    if "FROM items i" in s and "human_labels l1" in s:                 # pass 2 queue
        done2 = {l["item_id"] for l in LABELS if l["pass_no"] == 2}
        return [i for i in ITEMS
                if any(l["item_id"] == i["id"] and l["pass_no"] == 1 for l in LABELS)
                and i["id"] not in done2]
    if "FROM items i" in s and "NOT EXISTS" in s:                      # pass 1 queue
        done1 = {l["item_id"] for l in LABELS if l["pass_no"] == 1}
        return [i for i in ITEMS if i["split"] in (params[0] or []) and i["id"] not in done1]
    if s.startswith("SELECT * FROM items WHERE id"):
        return [i for i in ITEMS if i["id"] == params[0]]
    if s.startswith("SELECT * FROM items"):
        return ITEMS
    if "count(*) AS n FROM items" in s:
        return [{"n": len([i for i in ITEMS if i["split"] in ("dev", "test")])}]
    if "count(*) AS n FROM human_labels" in s:
        return [{"n": len([l for l in LABELS
                           if l["labeler_id"] == params[0] and l["pass_no"] == params[1]])}]
    if s.startswith("SELECT created_at FROM human_labels"):
        return [{"created_at": l["created_at"]} for l in LABELS]
    if s.startswith("SELECT scores, note FROM human_labels"):
        return [{"scores": l["scores"], "note": l["note"]} for l in LABELS
                if l["item_id"] == params[0] and l["pass_no"] == params[2]]
    if "SELECT labeler_id, pass_no, note" in s:
        return [{"labeler_id": l["labeler_id"], "pass_no": l["pass_no"], "note": l["note"]}
                for l in LABELS if l["item_id"] == params[0] and l["note"]]
    if "GROUP BY labeler_id, pass_no" in s:
        return [{"labeler_id": "yy", "pass_no": 1, "labels": 2, "last_label": NOW},
                {"labeler_id": "yy", "pass_no": 2, "labels": 1, "last_label": NOW}]
    if "FROM human_labels l JOIN items i" in s:
        rows = [{"item_id": l["item_id"], "scores": l["scores"]} for l in LABELS
                if l["pass_no"] == params[0]]
        if len(params) > 1:
            keep = {i["id"] for i in ITEMS if i["split"] == params[1]}
            rows = [r for r in rows if r["item_id"] in keep]
        return rows
    if "SELECT scores FROM human_labels" in s:
        return [{"scores": l["scores"]} for l in LABELS]
    if "FROM judge_runs r LEFT JOIN judge_scores" in s:
        return [{"id": 1, "label": "v2 / reason_first", "rubric_version": "v2",
                 "judge_model": "claude-opus-5", "prompt_variant": "reason_first",
                 "created_at": NOW, "scored": len(JUDGED)}]
    if "FROM metrics m JOIN judge_runs r" in s:
        family = params[0]
        rows = []
        if family == "agreement":
            for c in CRITERIA:
                rows.append({"name": "weighted_kappa", "criterion": c, "value": 0.71, "n": 2})
                rows.append({"name": "spearman", "criterion": c, "value": 0.80, "n": 2})
                rows.append({"name": "self_kappa", "criterion": c, "value": 0.86, "n": 1})
        elif family == "bias":
            rows = [{"name": "position_flip_rate", "criterion": None, "value": 0.14, "n": 24},
                    {"name": "length_spearman", "criterion": None, "value": 0.31, "n": 48},
                    {"name": "self_preference_rate", "criterion": None, "value": 0.58, "n": 20}]
            rows += [{"name": "length_delta", "criterion": c, "value": 0.2, "n": 24} for c in CRITERIA]
        else:
            rows = [{"name": "mean", "criterion": c, "value": 2.6, "n": 2} for c in CRITERIA]
        for r in rows:
            r.update({"run_id": 1, "label": "v2 / reason_first", "rubric_version": "v2",
                      "prompt_variant": "reason_first", "created_at": NOW, "detail": None})
        return rows
    if "FROM judge_scores WHERE run_id" in s and "reasoning" in s:
        return [{"item_id": k, "reasoning": {c: "cited the 30-day span" for c in CRITERIA}}
                for k in JUDGED]
    if "FROM judge_scores WHERE run_id" in s:
        return [{"item_id": k, "scores": v} for k, v in JUDGED.items()]
    if "SELECT id FROM judge_runs" in s:
        return [{"id": 1}]
    return []


def fake_execute(sql, params=()):
    WRITES.append((" ".join(sql.split()), params))


@pytest.fixture(autouse=True)
def stub_db(monkeypatch):
    WRITES.clear()
    monkeypatch.setattr(db, "query", fake_query)
    monkeypatch.setattr(db, "execute", fake_execute)
    yield


def score_all(at, point):
    """Click `point` for each criterion in turn — the last click completes the item."""
    for criterion in CRITERIA:
        key = next(b.key for b in at.button if b.key and f"|{criterion}|btn{point}" in b.key)
        at = at.button(key=key).click().run()
    return at


def run(app, **session):
    at = AppTest.from_file(str(ROOT / "app" / app), default_timeout=30)
    for k, v in session.items():
        at.session_state[k] = v
    return at.run()


def test_labeler_asks_for_an_id_before_showing_anything():
    at = run("label.py")
    assert not at.exception
    assert any("labeler id" in str(i.value) for i in at.info)


def test_labeling_screen_renders_a_full_scoring_grid():
    at = run("label.py", labeler="yy")
    assert not at.exception, at.exception
    labels = [b.label for b in at.button]
    # one button per criterion per scale point, plus back / skip
    assert sum(l[0].isdigit() for l in labels) == len(CRITERIA) * 3
    assert any("back" in l for l in labels) and any("skip" in l for l in labels)
    assert any("keys:" in c.value for c in at.caption), "keyboard legend missing"


def test_scoring_every_criterion_saves_and_advances():
    at = run("label.py", labeler="yy")
    shown = [c.value for c in at.caption if "::" in c.value]
    at = score_all(at, 3)
    assert not at.exception
    inserts = [w for w in WRITES if "INSERT INTO human_labels" in w[0]]
    assert len(inserts) == 1, f"expected one save after the last criterion, got {len(inserts)}"
    item_id, labeler, pass_no, scores, note = inserts[0][1]
    assert labeler == "yy" and pass_no == 1
    assert set(scores.obj) == set(CRITERIA) and set(scores.obj.values()) == {3}
    assert shown, "the item id caption should identify what was being scored"


def test_flag_prefixes_the_note_for_rubric_triage():
    at = run("label.py", labeler="yy")
    flag = next(t.key for t in at.toggle if t.key and t.key.endswith("|flag"))
    at = at.toggle(key=flag).set_value(True).run()
    at = score_all(at, 1)
    note = [w for w in WRITES if "INSERT INTO human_labels" in w[0]][0][1][4]
    assert note.startswith("[rubric-unclear]")


def test_manual_mode_shows_an_explicit_save_button():
    at = run("label.py", labeler="yy")
    auto = next(t for t in at.toggle if t.label == "auto-advance")
    at = auto.set_value(False).run()
    at = score_all(at, 2)
    assert not [w for w in WRITES if "INSERT INTO human_labels" in w[0]], "must not autosave"
    assert any("save and next" in b.label for b in at.button)


def test_a_five_by_five_rubric_still_gets_every_button(monkeypatch):
    """The spec allows a 3- to 5-point scale, and nothing stops a task needing five
    criteria. Both used to be silently truncated by a hardcoded shortcut table."""
    from judge import rubric as rubric_mod

    wide = {
        "version": "wide", "scale": [1, 2, 3, 4, 5], "_raw": "",
        "criteria": [{"name": f"c{i}", "question": f"question {i}",
                      "points": {p: f"point {p}" for p in range(1, 6)}} for i in range(5)],
    }
    monkeypatch.setattr(rubric_mod, "load", lambda v: wide)
    monkeypatch.setattr(rubric_mod, "versions", lambda: ["wide"])
    at = run("label.py", labeler="yy")
    assert not at.exception, at.exception
    score_buttons = [b for b in at.button if b.key and "|btn" in b.key]
    assert len(score_buttons) == 25, f"expected 5 criteria x 5 points, got {len(score_buttons)}"
    assert {b.key.rsplit("|btn", 1)[1] for b in score_buttons} == {"1", "2", "3", "4", "5"}


def test_dashboard_renders_every_tab():
    import streamlit as st
    st.cache_data.clear()
    at = run("dashboard.py")
    assert not at.exception, at.exception
    assert len(at.tabs) == 4
    headline = " ".join(m.label for m in at.metric)
    assert "position flip rate" in headline and "self-preference" in headline
    assert any("Improvement curve" in s.value for s in at.subheader)


def test_dashboard_survives_an_empty_database(monkeypatch):
    import streamlit as st
    st.cache_data.clear()
    monkeypatch.setattr(db, "query", lambda sql, params=(): [])
    at = run("dashboard.py")
    assert not at.exception
    assert any("No judge runs yet" in w.value for w in at.warning)
