"""Human labeling UI. `streamlit run app/label.py`

Built for the grind: 192 items x 4 criteria. One click per criterion, no save
button, and the rubric wording is one hover away instead of one scroll away.

Deliberately hides the stratum and the injected defect: a labeler who is told an
answer is broken will find it broken. Pass 2 re-serves items you have already
labelled, without showing what you said the first time — that is what makes the
self-agreement number mean anything.
"""
import statistics
import time

import streamlit as st
from psycopg.types.json import Jsonb

from judge import db, rubric as rubric_mod

st.set_page_config(page_title="Label", layout="wide", page_icon="✍️")
st.markdown("""
<style>
  .block-container {padding-top: 2.2rem; padding-bottom: 1rem;}
  div[data-testid="stMetricValue"] {font-size: 1.4rem;}
  .answer-box {background: rgba(255,196,0,.10); border-left: 4px solid #f0a800;
               padding: .9rem 1.1rem; border-radius: 6px; font-size: 1.02rem; line-height: 1.55;}
  .ctx-box {background: rgba(70,140,255,.08); border-left: 4px solid #4b8dff;
            padding: .8rem 1rem; border-radius: 6px; max-height: 15rem; overflow-y: auto;
            font-size: .92rem; line-height: 1.5;}
</style>""", unsafe_allow_html=True)

st.session_state.setdefault("history", [])
st.session_state.setdefault("skipped", [])
st.session_state.setdefault("editing", None)

with st.sidebar:
    st.header("Labeler")
    labeler = st.text_input("labeler id", value=st.session_state.get("labeler", ""),
                            placeholder="your initials")
    pass_no = st.radio("pass", [1, 2], horizontal=True,
                       help="pass 2 re-labels items you did more than 12 hours ago, without "
                            "showing your earlier scores. That number is the judge's ceiling.")
    version = st.selectbox("rubric", rubric_mod.versions(), index=len(rubric_mod.versions()) - 1)
    splits = st.multiselect("splits", ["dev", "test"], default=["dev", "test"],
                            help="dev is for tuning the rubric, test is what you report")
    st.divider()
    auto = st.toggle("auto-advance", value=True,
                     help="save and jump to the next item as soon as all criteria are scored. "
                          "Turn it off when you want to write a note first.")
    show_help = st.toggle("show rubric wording inline", value=True)
    if st.session_state.skipped:
        st.caption(f"{len(st.session_state.skipped)} skipped this session")
        if st.button("un-skip all", use_container_width=True):
            st.session_state.skipped = []
            st.rerun()

if not labeler:
    st.title("✍️ Labeling")
    st.info("Enter a labeler id in the sidebar to start.")
    st.stop()
st.session_state["labeler"] = labeler
rubric = rubric_mod.load(version)
criteria = rubric["criteria"]
scale = rubric["scale"]


def queue():
    skipped = st.session_state.skipped or [""]
    if pass_no == 1:
        return db.query(
            """SELECT i.* FROM items i
               WHERE i.split = ANY(%s) AND NOT (i.id = ANY(%s))
                 AND NOT EXISTS (SELECT 1 FROM human_labels l
                                 WHERE l.item_id = i.id AND l.labeler_id = %s AND l.pass_no = 1)
               ORDER BY md5(i.id)""",
            (splits, skipped, labeler),
        )
    return db.query(
        """SELECT i.* FROM items i
           JOIN human_labels l1 ON l1.item_id = i.id AND l1.labeler_id = %s AND l1.pass_no = 1
           WHERE NOT (i.id = ANY(%s))
             AND NOT EXISTS (SELECT 1 FROM human_labels l2
                             WHERE l2.item_id = i.id AND l2.labeler_id = %s AND l2.pass_no = 2)
             AND l1.created_at < now() - interval '12 hours'
           ORDER BY md5(i.id || 'pass2')""",
        (labeler, skipped, labeler),
    )


def save(item_id, scores, note):
    db.execute(
        """INSERT INTO human_labels (item_id, labeler_id, pass_no, scores, note)
           VALUES (%s, %s, %s, %s, %s)
           ON CONFLICT (item_id, labeler_id, pass_no) DO UPDATE
             SET scores = EXCLUDED.scores, note = EXCLUDED.note, created_at = now()""",
        (item_id, labeler, pass_no, Jsonb(scores), note or None),
    )
    if item_id not in st.session_state.history:
        st.session_state.history.append(item_id)


def pace():
    """Median seconds per label over your last 20, so the remaining pile has a number on it."""
    rows = db.query(
        """SELECT created_at FROM human_labels WHERE labeler_id = %s AND pass_no = %s
           ORDER BY created_at DESC LIMIT 20""", (labeler, pass_no))
    stamps = sorted(r["created_at"] for r in rows)
    gaps = [(b - a).total_seconds() for a, b in zip(stamps, stamps[1:]) if (b - a).total_seconds() < 600]
    return statistics.median(gaps) if len(gaps) >= 3 else None


todo = queue()
done = db.query("SELECT count(*) AS n FROM human_labels WHERE labeler_id = %s AND pass_no = %s",
                (labeler, pass_no))[0]["n"]

editing = st.session_state.editing
if editing:
    rows = db.query("SELECT * FROM items WHERE id = %s", (editing,))
    prior = db.query(
        "SELECT scores, note FROM human_labels WHERE item_id = %s AND labeler_id = %s AND pass_no = %s",
        (editing, labeler, pass_no))
    item = rows[0]
    prefill = prior[0]["scores"] if prior else {}
    prefill_note = (prior[0]["note"] if prior else "") or ""
elif todo:
    item = todo[0]
    prefill, prefill_note = {}, ""
else:
    st.title("✍️ Labeling")
    st.success(f"Nothing left in pass {pass_no}. You have **{done}** labels.")
    if pass_no == 2:
        st.caption("Pass 2 only serves items you labelled more than 12 hours ago — "
                   "same-day re-labelling measures memory, not consistency.")
    if st.session_state.history and st.button("↩ revise my last label"):
        st.session_state.editing = st.session_state.history[-1]
        st.rerun()
    st.stop()

# ── header: progress, pace, navigation ────────────────────────────────────────
remaining = len(todo)
total = done + remaining
bar, m1, m2, nav = st.columns([5, 1.3, 1.3, 2])
with bar:
    st.progress(done / total if total else 1.0)
    st.caption(f"**{done}** labelled · **{remaining}** left in pass {pass_no}"
               + ("  ·  ✏️ revising an earlier item" if editing else ""))
seconds = pace()
m1.metric("done", done)
m2.metric("pace", f"{seconds:.0f}s" if seconds else "—",
          help="median seconds per label over your last 20"
               + (f" · about {remaining * seconds / 60:.0f} min left" if seconds else ""))
with nav:
    back, skip = st.columns(2)
    if back.button("↩ back", use_container_width=True, shortcut="Backspace",
                   disabled=not st.session_state.history,
                   help="revise the label you just saved"):
        st.session_state.editing = st.session_state.history[-1]
        st.rerun()
    if skip.button("skip →", use_container_width=True, shortcut="Esc", disabled=bool(editing),
                   help="park this one for later in the session"):
        st.session_state.skipped.append(item["id"])
        st.rerun()

# ── the item ──────────────────────────────────────────────────────────────────
st.subheader(item["question"])
ctx, ans = st.columns([1, 1])
with ctx:
    st.caption("RETRIEVED CONTEXT — the only material the answer was allowed to use")
    st.markdown(f'<div class="ctx-box">{item["context"]}</div>', unsafe_allow_html=True)
with ans:
    st.caption("ANSWER — score this")
    st.markdown(f'<div class="answer-box">{item["answer"]}</div>', unsafe_allow_html=True)
    st.caption(f"`{item['id']}`  ·  {len(item['answer'])} chars")

st.divider()

# ── scoring ───────────────────────────────────────────────────────────────────
key = f"{item['id']}|{pass_no}|{version}"


def widget_key(name):
    return f"{key}|{name}"


def current_scores():
    return {c["name"]: st.session_state.get(widget_key(c["name"])) for c in criteria}


for c in criteria:                      # seed the editing view with what was saved before
    if widget_key(c["name"]) not in st.session_state and c["name"] in prefill:
        st.session_state[widget_key(c["name"])] = prefill[c["name"]]


def composed_note():
    text = (st.session_state.get(f"{key}|note") or "").strip()
    if st.session_state.get(f"{key}|flag") and not text.startswith("[rubric-unclear]"):
        return f"[rubric-unclear] {text}".strip()
    return text


# Home-row keys, one block of three per criterion. Digits rather than letters so a
# stray keystroke while writing a note is less likely to score something — and when it
# does, the highlighted button shows it rather than hiding it.
KEYS = [["1", "2", "3"], ["4", "5", "6"], ["7", "8", "9"], ["Mod+1", "Mod+2", "Mod+3"]]

def set_score(name, point):
    """A callback, not a return value: it runs before the rerun, so the highlighted
    button and the auto-advance are correct on the very next frame."""
    st.session_state[widget_key(name)] = point
    scores = current_scores()
    if auto and not editing and all(v is not None for v in scores.values()):
        save(item["id"], scores, composed_note())


cols = st.columns(len(criteria))
for col, c, keyrow in zip(cols, criteria, KEYS):
    with col:
        head, info = st.columns([4, 1])
        head.markdown(f"**{c['name']}**")
        with info.popover("ⓘ", use_container_width=True):
            st.markdown(f"**{c['name']}** — {c['question']}")
            for point, text in sorted(c["points"].items()):
                st.markdown(f"**{point}** — {text}")
            for point, anchor in sorted(c.get("anchors", {}).items()):
                st.caption(f"anchor {point}: “{anchor['answer']}” — {anchor['why']}")

        chosen = st.session_state.get(widget_key(c["name"]))
        for point, shortcut in zip(scale, keyrow):
            summary = c["points"][point].split(". ")[0].rstrip(".")
            st.button(f"{point} — {summary}", key=f"{widget_key(c['name'])}|btn{point}",
                      shortcut=shortcut, use_container_width=True,
                      type="primary" if chosen == point else "secondary",
                      help=c["points"][point], on_click=set_score, args=(c["name"], point))

st.caption("keys: **1–9** score the first three criteria · **⌘/Ctrl+1–3** safety · "
           "**Backspace** revise the last one · **Esc** skip")

note_col, flag_col = st.columns([4, 1])
note_col.text_input(
    "note", key=f"{key}|note", value=prefill_note, label_visibility="collapsed",
    placeholder="what decided the call, or what the rubric failed to tell you")
flag_col.toggle("🚩 rubric unclear", key=f"{key}|flag",
                            help="flags this item for rubric triage — the most valuable thing "
                                 "you can leave behind")

scores = current_scores()
note = composed_note()
complete = all(v is not None for v in scores.values())

if editing:
    a, b = st.columns([1, 4])
    if a.button("💾 update", type="primary", disabled=not complete, use_container_width=True):
        save(item["id"], scores, note)
        st.session_state.editing = None
        st.rerun()
    if b.button("cancel", use_container_width=False):
        st.session_state.editing = None
        st.rerun()
elif not auto:
    if st.button("save and next →", type="primary", shortcut="Enter", disabled=not complete):
        save(item["id"], scores, note)
        st.rerun()
elif not complete:
    st.caption(f"{sum(v is not None for v in scores.values())}/{len(criteria)} scored — "
               "saves and advances on the last one")
