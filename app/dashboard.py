"""Quality trend lines. `streamlit run app/dashboard.py`

Every judge run is a row in Postgres, so quality has a trend rather than an anecdote.
"""
import pandas as pd
import streamlit as st

from judge import db

st.set_page_config(page_title="Judge dashboard", layout="wide")
st.title("LLM-as-judge — calibration and drift")


def frame(family, name):
    rows = db.query(
        """SELECT m.run_id, r.label, r.rubric_version, r.judge_model, r.prompt_variant,
                  r.created_at, m.criterion, m.value, m.n, m.detail
           FROM metrics m JOIN judge_runs r ON r.id = m.run_id
           WHERE m.family = %s AND m.name = %s ORDER BY m.run_id""",
        (family, name),
    )
    return pd.DataFrame(rows)


agreement = frame("agreement", "weighted_kappa")
st.header("Judge-to-human agreement")
if agreement.empty:
    st.info("No agreement metrics yet. Label some items, then run `python -m judge.agreement`.")
else:
    ceiling = frame("agreement", "self_kappa")
    st.caption("The improvement curve. Each point is a rubric iteration; the dashed reference is "
               "human self-agreement, which the judge cannot exceed.")
    pivot = agreement.pivot_table(index="run_id", columns="criterion", values="value")
    if not ceiling.empty:
        for criterion, value in ceiling.groupby("criterion")["value"].last().items():
            pivot[f"{criterion} (human ceiling)"] = value
    st.line_chart(pivot)
    st.dataframe(agreement[["run_id", "label", "rubric_version", "criterion", "value", "n"]],
                 use_container_width=True, hide_index=True)

st.header("Measured biases")
bias = db.query(
    """SELECT r.created_at, r.rubric_version, m.name, m.criterion, m.value, m.n, m.detail
       FROM metrics m JOIN judge_runs r ON r.id = m.run_id
       WHERE m.family = 'bias' ORDER BY m.id DESC""")
if not bias:
    st.info("No bias runs yet. Run `python -m judge.bias --all`.")
else:
    latest = {}
    for row in bias:
        latest.setdefault((row["name"], row["criterion"], str(row["detail"])), row)
    cols = st.columns(3)
    headline = ["position_flip_rate", "length_spearman", "self_preference_rate"]
    for col, name in zip(cols, headline):
        picks = [r for (n, _, _), r in latest.items() if n == name]
        with col:
            if not picks:
                st.metric(name, "—")
            else:
                value = picks[0]["value"]
                st.metric(name, f"{value:+.3f}" if "spearman" in name else f"{value:.1%}",
                          help=f"n={picks[0]['n']}")
    st.dataframe(pd.DataFrame(bias), use_container_width=True, hide_index=True)

st.header("Mean score per criterion (what CI gates on)")
scores = frame("score", "mean")
if scores.empty:
    st.info("No judge runs yet.")
else:
    st.line_chart(scores.pivot_table(index="run_id", columns="criterion", values="value"))

st.header("Labeling progress")
progress = db.query(
    """SELECT labeler_id, pass_no, count(*) AS labels FROM human_labels
       GROUP BY labeler_id, pass_no ORDER BY labeler_id, pass_no""")
total = db.query("SELECT count(*) AS n FROM items WHERE split IN ('dev','test')")[0]["n"]
st.caption(f"{total} labelable items in the dev+test splits")
st.dataframe(pd.DataFrame(progress), use_container_width=True, hide_index=True)
