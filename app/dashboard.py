"""Quality trends and disagreement triage. `streamlit run app/dashboard.py`

Every judge run is a row in Postgres, so quality has a trend rather than an anecdote.
The Disagreements tab is the Phase 3 workflow in the browser: pick the largest gap,
read the judge's reasoning against the labeler's note, decide whether the rubric or
the judge is the thing that is wrong.
"""
import pandas as pd
import streamlit as st

from judge import agreement as ag, db, rubric as rubric_mod

st.set_page_config(page_title="Judge dashboard", layout="wide", page_icon="⚖️")
st.markdown("<style>.block-container{padding-top:2.2rem;}</style>", unsafe_allow_html=True)

POSITION_LIMIT = 0.10          # flip rate above this makes pairwise verdicts untrustworthy
LENGTH_LIMIT = 0.20            # |spearman(chars, score)| above this is the rubric rewarding volume
SELF_PREF_LIMIT = 0.60         # 50% is neutral


@st.cache_data(ttl=30)
def runs():
    return pd.DataFrame(db.query(
        """SELECT r.id, r.label, r.rubric_version, r.judge_model, r.prompt_variant,
                  r.created_at, count(s.item_id) AS scored
           FROM judge_runs r LEFT JOIN judge_scores s ON s.run_id = r.id
           GROUP BY r.id ORDER BY r.id"""))


@st.cache_data(ttl=30)
def metrics(family):
    return pd.DataFrame(db.query(
        """SELECT m.run_id, r.label, r.rubric_version, r.prompt_variant, r.created_at,
                  m.name, m.criterion, m.value, m.n, m.detail
           FROM metrics m JOIN judge_runs r ON r.id = m.run_id
           WHERE m.family = %s ORDER BY m.run_id""", (family,)))


all_runs = runs()
if all_runs.empty:
    st.title("⚖️ LLM-as-judge")
    st.warning("No judge runs yet. `make dataset-offline && make judge` to populate this.")
    st.stop()

with st.sidebar:
    st.header("Filters")
    versions = sorted(all_runs["rubric_version"].unique())
    pick_versions = st.multiselect("rubric version", versions, default=versions)
    variants = sorted(all_runs["prompt_variant"].unique())
    pick_variants = st.multiselect("prompt variant", variants, default=variants)
    split = st.radio("split", ["test", "dev"], horizontal=True,
                     help="report agreement on test; tune the rubric on dev")
    st.divider()
    st.caption(f"{len(all_runs)} runs · latest #{int(all_runs['id'].max())}")
    if st.button("refresh", use_container_width=True, shortcut="U"):
        st.cache_data.clear()
        st.rerun()

visible = all_runs[all_runs["rubric_version"].isin(pick_versions)
                   & all_runs["prompt_variant"].isin(pick_variants)]
run_ids = set(visible["id"])

st.title("⚖️ LLM-as-judge — calibration and drift")
tab_agree, tab_bias, tab_gaps, tab_runs = st.tabs(
    ["Agreement", "Bias", "Disagreements", "Runs & progress"])

# ── agreement ─────────────────────────────────────────────────────────────────
with tab_agree:
    kappa = metrics("agreement")
    kappa = kappa[kappa["run_id"].isin(run_ids)] if not kappa.empty else kappa
    if kappa.empty:
        st.info("No agreement metrics yet. Label some items, then `make agreement`.")
    else:
        weighted = kappa[kappa["name"] == "weighted_kappa"]
        ceiling = kappa[kappa["name"] == "self_kappa"]
        latest = weighted[weighted["run_id"] == weighted["run_id"].max()]

        cols = st.columns(max(len(latest), 1))
        for col, (_, row) in zip(cols, latest.iterrows()):
            ceil_row = ceiling[ceiling["criterion"] == row["criterion"]]
            gap = row["value"] - ceil_row["value"].iloc[-1] if not ceil_row.empty else None
            col.metric(row["criterion"], f"κ {row['value']:.3f}",
                       delta=f"{gap:+.3f} vs human ceiling" if gap is not None else None,
                       help=f"n={row['n']} on the {split} split")

        st.subheader("Improvement curve")
        st.caption("Each point is a rubric iteration. The judge cannot beat the human ceiling — "
                   "if a line crosses it, the ceiling is measured on too few re-labels.")
        curve = weighted.pivot_table(index="run_id", columns="criterion", values="value")
        for criterion, value in ceiling.groupby("criterion")["value"].last().items():
            curve[f"{criterion} · ceiling"] = value
        st.line_chart(curve, height=340)

        with st.expander("Spearman ρ beside κ — they fail differently"):
            st.caption("κ punishes a judge that is consistently one point low; ρ does not. "
                       "High ρ with low κ means recalibrate the offset, not the rubric.")
            rho = kappa[kappa["name"] == "spearman"]
            if rho.empty:
                st.write("no Spearman rows yet")
            else:
                st.line_chart(rho.pivot_table(index="run_id", columns="criterion", values="value"))

        st.subheader("Where the judge and the humans actually differ")
        pick_run = st.selectbox("run", sorted(run_ids, reverse=True),
                                format_func=lambda r: f"#{r} · "
                                f"{visible.set_index('id').loc[r, 'label']}")
        human, judged = ag.human_consensus(split=split), ag.judge_scores(pick_run)
        if not (set(human) & set(judged)):
            st.info(f"Run #{pick_run} covers no labelled items in the {split} split.")
        else:
            grid = st.columns(len(rubric_mod.criteria(rubric_mod.load(pick_versions[-1]))))
            for col, criterion in zip(grid, sorted({c for v in human.values() for c in v})):
                xs, ys = ag.paired(human, judged, criterion)
                table = pd.crosstab(pd.Series(xs, name="human"), pd.Series(ys, name="judge"))
                table.columns = [f"judge {c}" for c in table.columns]
                with col:
                    st.caption(f"**{criterion}** — rows human, cols judge")
                    st.dataframe(table, use_container_width=True, column_config={
                        c: st.column_config.ProgressColumn(c, min_value=0, format="%d",
                                                           max_value=int(table.values.max()))
                        for c in table.columns})

# ── bias ──────────────────────────────────────────────────────────────────────
with tab_bias:
    bias = metrics("bias")
    if bias.empty:
        st.info("No bias runs yet. `make bias`.")
    else:
        latest_of = lambda name: bias[bias["name"] == name].tail(1)
        pos, length, pref = latest_of("position_flip_rate"), latest_of("length_spearman"), \
            latest_of("self_preference_rate")
        c1, c2, c3 = st.columns(3)
        if not pos.empty:
            v = pos["value"].iloc[0]
            c1.metric("position flip rate", f"{v:.1%}",
                      delta=f"{'over' if v > POSITION_LIMIT else 'within'} the 10% line",
                      delta_color="inverse" if v > POSITION_LIMIT else "normal",
                      help="same pair judged in both orders; how often the verdict flips")
        if not length.empty:
            v = length["value"].iloc[0]
            c2.metric("length bias ρ", f"{v:+.3f}",
                      delta="rubric rewards volume" if abs(v) > LENGTH_LIMIT else "length-neutral",
                      delta_color="inverse" if abs(v) > LENGTH_LIMIT else "normal",
                      help="quality held constant, characters tripled by padding")
        if not pref.empty:
            v = pref["value"].iloc[0]
            c3.metric("self-preference", f"{v:.1%}",
                      delta=f"{v - 0.5:+.1%} vs neutral",
                      delta_color="inverse" if v > SELF_PREF_LIMIT else "normal",
                      help="how often a judge picks its own model family; 50% is neutral")

        st.subheader("Which criterion leaks length")
        deltas = bias[bias["name"] == "length_delta"]
        if deltas.empty:
            st.caption("no per-criterion length deltas yet")
        else:
            st.bar_chart(deltas.set_index("criterion")["value"], height=260,
                         y_label="padded minus base (rubric points)")
        with st.expander("every bias measurement, newest first"):
            st.dataframe(bias.sort_values("run_id", ascending=False)
                         [["run_id", "created_at", "rubric_version", "name", "criterion", "value", "n"]],
                         use_container_width=True, hide_index=True)

# ── disagreements ─────────────────────────────────────────────────────────────
with tab_gaps:
    st.caption("Sorted by gap. Select a row to read it. In nearly every case the rubric is "
               "ambiguous rather than the model being wrong — that is the lesson of this phase.")
    gap_run = st.selectbox("run", sorted(run_ids, reverse=True), key="gap_run",
                           format_func=lambda r: f"#{r} · {visible.set_index('id').loc[r, 'label']}")
    top = st.slider("how many", 5, 50, 20, step=5)
    rows = ag.disagreements(gap_run, split=split, top=top)
    if not rows:
        st.info(f"Nothing to compare — run #{gap_run} has no labelled items in the {split} split.")
    else:
        table = pd.DataFrame([{"item": r["item_id"], "criterion": r["criterion"],
                               "human": r["human"], "judge": r["judge"], "gap": r["gap"],
                               "stratum": r["item"]["stratum"], "defect": r["item"]["defect"] or "—"}
                              for r in rows])
        picked = st.dataframe(table, use_container_width=True, hide_index=True,
                              on_select="rerun", selection_mode="single-row",
                              column_config={"gap": st.column_config.ProgressColumn(
                                  "gap", min_value=0, max_value=2, format="%d")})
        index = (picked.selection.rows or [0])[0]
        row = rows[index]
        item = row["item"]
        st.divider()
        left, right = st.columns([3, 2])
        with left:
            st.markdown(f"**{item['question']}**")
            st.caption("retrieved context")
            st.info(item["context"])
            st.caption("answer")
            st.warning(item["answer"])
        with right:
            a, b = st.columns(2)
            a.metric("human", row["human"])
            b.metric("judge", row["judge"], delta=row["judge"] - row["human"])
            st.caption(f"criterion: **{row['criterion']}** · stratum {item['stratum']} · "
                       f"defect {item['defect'] or 'none'}")
            st.caption("the judge's reasoning")
            st.write(row["why"] or "_none recorded_")
            notes = db.query(
                """SELECT labeler_id, pass_no, note FROM human_labels
                   WHERE item_id = %s AND note IS NOT NULL""", (row["item_id"],))
            if notes:
                st.caption("labeler notes")
                for n in notes:
                    st.write(f"**{n['labeler_id']}** (pass {n['pass_no']}): {n['note']}")

# ── runs and progress ─────────────────────────────────────────────────────────
with tab_runs:
    scores = metrics("score")
    if not scores.empty:
        st.subheader("Mean score per criterion — what CI gates on")
        st.line_chart(scores[scores["run_id"].isin(run_ids)]
                      .pivot_table(index="run_id", columns="criterion", values="value"), height=300)

    st.subheader("Labeling progress")
    total = db.query("SELECT count(*) AS n FROM items WHERE split IN ('dev','test')")[0]["n"]
    progress = pd.DataFrame(db.query(
        """SELECT labeler_id, pass_no, count(*) AS labels, max(created_at) AS last_label
           FROM human_labels GROUP BY labeler_id, pass_no ORDER BY labeler_id, pass_no"""))
    labelled = int(progress[progress["pass_no"] == 1]["labels"].sum()) if not progress.empty else 0
    a, b, c = st.columns(3)
    a.metric("labelable items", total)
    b.metric("first-pass labels", labelled)
    c.metric("re-labels (the ceiling)",
             int(progress[progress["pass_no"] == 2]["labels"].sum()) if not progress.empty else 0,
             help="20 is enough to estimate your own consistency")
    st.progress(min(labelled / total, 1.0) if total else 0.0)
    if not progress.empty:
        st.dataframe(progress, use_container_width=True, hide_index=True)

    st.subheader("What it costs")
    spend = metrics("cost")
    if spend.empty:
        st.caption("No cost rows yet — token counts are recorded from the next `make judge`.")
    else:
        recent = spend[spend["run_id"].isin(run_ids)]
        tokens = recent[recent["name"].isin(["input_tokens", "output_tokens"])]
        usd = recent[recent["name"] == "usd"]
        a, b, c = st.columns(3)
        a.metric("tokens in", f"{tokens[tokens['name'] == 'input_tokens']['value'].sum():,.0f}")
        b.metric("tokens out", f"{tokens[tokens['name'] == 'output_tokens']['value'].sum():,.0f}")
        c.metric("spend", f"${usd['value'].sum():.2f}" if not usd.empty else "—",
                 help="set JUDGE_PRICE_IN_PER_MTOK and JUDGE_PRICE_OUT_PER_MTOK to price runs")
        st.caption("A gate that runs on every pull request has a bill. This is it.")

    st.subheader("Runs")
    st.dataframe(visible.sort_values("id", ascending=False), use_container_width=True,
                 hide_index=True)
