"""Human labeling UI. `streamlit run app/label.py`

Deliberately hides the stratum and the injected defect: a labeler who is told an
answer is broken will find it broken. Pass 2 re-serves items you have already
labelled, without showing what you said the first time — that is what makes the
self-agreement number mean anything.
"""
import hashlib

import streamlit as st
from psycopg.types.json import Jsonb

from judge import db, rubric as rubric_mod

st.set_page_config(page_title="Label", layout="wide")

with st.sidebar:
    st.header("Labeler")
    labeler = st.text_input("labeler id", value=st.session_state.get("labeler", ""))
    pass_no = st.radio("pass", [1, 2], help="pass 2 re-labels items you have already done, "
                                            "on a different day, to measure your own consistency")
    version = st.selectbox("rubric", rubric_mod.versions(), index=len(rubric_mod.versions()) - 1)
    splits = st.multiselect("splits", ["dev", "test"], default=["dev", "test"])

if not labeler:
    st.info("Enter a labeler id in the sidebar to start.")
    st.stop()
st.session_state["labeler"] = labeler
rubric = rubric_mod.load(version)

if pass_no == 1:
    todo = db.query(
        """SELECT i.* FROM items i
           WHERE i.split = ANY(%s)
             AND NOT EXISTS (SELECT 1 FROM human_labels l
                             WHERE l.item_id = i.id AND l.labeler_id = %s AND l.pass_no = 1)
           ORDER BY md5(i.id)""",
        (splits, labeler),
    )
else:
    todo = db.query(
        """SELECT i.* FROM items i
           JOIN human_labels l1 ON l1.item_id = i.id AND l1.labeler_id = %s AND l1.pass_no = 1
           WHERE NOT EXISTS (SELECT 1 FROM human_labels l2
                             WHERE l2.item_id = i.id AND l2.labeler_id = %s AND l2.pass_no = 2)
             AND l1.created_at < now() - interval '12 hours'
           ORDER BY md5(i.id || 'pass2')""",
        (labeler, labeler),
    )

done = db.query(
    "SELECT count(*) AS n FROM human_labels WHERE labeler_id = %s AND pass_no = %s", (labeler, pass_no)
)[0]["n"]

if not todo:
    st.success(f"Nothing left in pass {pass_no}. You have {done} labels.")
    if pass_no == 2:
        st.caption("Pass 2 only serves items you labelled more than 12 hours ago — "
                   "same-day re-labelling measures memory, not consistency.")
    st.stop()

item = todo[0]
st.progress(done / (done + len(todo)), text=f"{done} labelled, {len(todo)} remaining in pass {pass_no}")

left, right = st.columns([3, 2])
with left:
    st.subheader("Question")
    st.write(item["question"])
    st.subheader("Retrieved context")
    st.info(item["context"])
    st.subheader("Answer")
    st.warning(item["answer"])
    st.caption(f"item {item['id']}")

with right:
    st.subheader("Rubric")
    for c in rubric["criteria"]:
        with st.expander(c["name"], expanded=False):
            st.write(c["question"])
            for point, text in sorted(c["points"].items()):
                st.markdown(f"**{point}** — {text}")
            for point, anchor in sorted(c.get("anchors", {}).items()):
                st.caption(f"anchor {point}: “{anchor['answer']}” — {anchor['why']}")

st.divider()
key = hashlib.md5(f"{item['id']}{pass_no}".encode()).hexdigest()[:8]
scores = {}
cols = st.columns(len(rubric["criteria"]))
for col, c in zip(cols, rubric["criteria"]):
    with col:
        scores[c["name"]] = st.radio(
            c["name"], rubric["scale"], index=None, horizontal=True, key=f"{key}-{c['name']}",
            format_func=lambda s, c=c: f"{s} — {c['points'][s][:60]}",
        )
note = st.text_area("note — what decided the call, or what the rubric failed to tell you",
                    key=f"{key}-note")

if st.button("Save and next", type="primary", disabled=any(v is None for v in scores.values())):
    db.execute(
        """INSERT INTO human_labels (item_id, labeler_id, pass_no, scores, note)
           VALUES (%s, %s, %s, %s, %s)
           ON CONFLICT (item_id, labeler_id, pass_no) DO UPDATE
             SET scores = EXCLUDED.scores, note = EXCLUDED.note""",
        (item["id"], labeler, pass_no, Jsonb(scores), note or None),
    )
    st.rerun()
