"""
Personal What-If: Causal Lifestyle Explorer
============================================
A Streamlit app for exploring causal what-if scenarios about lifestyle habits.
Uses DoWhy + EconML to estimate causal effects and distinguish correlation
from causation.
"""

import streamlit as st
import pandas as pd
import networkx as nx
from pathlib import Path

from causal_graph import build_dag, plot_dag, TREATMENTS, OUTCOMES, NODE_META
from causal_model import estimate_ate, estimate_whatif_all_outcomes, run_refutations
from utils import (
    label, SLIDER_RANGES, DELTA_RANGES,
    whatif_bar_chart, refutation_table_html,
)

ROOT = Path(__file__).parent

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Causal Lifestyle Explorer",
    page_icon="🔬",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .stMetric { background: #F8F9FA; border-radius: 10px; padding: 12px; }
    h1 { color: #2D3436; }
    h2, h3 { color: #636E72; }
    .highlight-box {
        background: linear-gradient(135deg, #DFE6E9 0%, #F8F9FA 100%);
        border-radius: 12px; padding: 20px; margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🔬 Personal What-If: Causal Lifestyle Explorer")
st.markdown(
    "Explore **causal** what-if scenarios about your lifestyle habits. "
    "Unlike simple correlations, this tool uses a **causal DAG** and **DoWhy/EconML** "
    "to estimate the *true effect* of changing a habit — backed by refutation tests."
)
st.divider()

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv(ROOT / "data" / "lifestyle_data.csv")

data = load_data()

# ---------------------------------------------------------------------------
# Sidebar — Lifestyle Input Form
# ---------------------------------------------------------------------------

st.sidebar.header("📋 Your Lifestyle Profile")
st.sidebar.markdown("Adjust sliders to match your current habits.")

user_profile: dict[str, float] = {}
for habit in TREATMENTS:
    mn, mx, default, step = SLIDER_RANGES[habit]
    user_profile[habit] = st.sidebar.slider(
        label(habit), min_value=mn, max_value=mx, value=default, step=step,
    )

# Extra confounder inputs
st.sidebar.markdown("---")
st.sidebar.subheader("Context (confounders)")
user_profile["age"] = st.sidebar.slider("Age", 18, 65, 30, 1)
user_profile["work_hours"] = st.sidebar.slider("Work Hours / day", 4.0, 14.0, 8.0, 0.5)

# ---------------------------------------------------------------------------
# Main: What-If Intervention
# ---------------------------------------------------------------------------

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.header("🧪 What-If Intervention")
    treatment = st.selectbox(
        "Which habit would you change?",
        options=TREATMENTS,
        format_func=label,
    )

    mn, mx, default, step = DELTA_RANGES[treatment]
    delta = st.slider(
        f"Change in {label(treatment)}",
        min_value=mn, max_value=mx, value=default, step=step,
    )

    if delta > 0:
        st.info(f"**Scenario:** What if you *increased* {label(treatment)} by **{abs(delta):.1f}** units?")
    elif delta < 0:
        st.info(f"**Scenario:** What if you *decreased* {label(treatment)} by **{abs(delta):.1f}** units?")
    else:
        st.warning("Set a non-zero change to see effects.")

with col_right:
    st.header("📊 Estimated Causal Effects")

    if delta != 0:
        with st.spinner("Running causal inference…"):
            results = estimate_whatif_all_outcomes(
                treatment=treatment,
                user_profile=user_profile,
                delta=delta,
                data=data,
            )

        # Metric cards
        metric_cols = st.columns(len(results))
        for col, r in zip(metric_cols, results):
            change = r["predicted_change"]
            col.metric(
                label=label(r["outcome"]),
                value=f"{change:+.3f}",
                delta=f"{'↑' if change > 0 else '↓'} per {abs(delta):.1f} unit change",
                delta_color="normal" if change > 0 else "inverse",
            )

        # Bar chart
        fig = whatif_bar_chart(results)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("👈 Set a non-zero intervention to see predicted effects.")

# ---------------------------------------------------------------------------
# Causal DAG
# ---------------------------------------------------------------------------

st.divider()
st.header("🕸️ Causal DAG")
st.markdown(
    "This directed acyclic graph encodes our domain-knowledge causal assumptions. "
    "Arrows show the direction of causal influence. The highlighted path (if any) "
    "shows the treatment → outcome chain you are currently exploring."
)

G = build_dag()

# Build highlight edges from treatment through mediators to all outcomes
highlight_nodes = {treatment}
highlight_edges = set()
for outcome in OUTCOMES:
    try:
        for path in nx.all_simple_paths(G, treatment, outcome):
            highlight_nodes.update(path)
            for i in range(len(path) - 1):
                highlight_edges.add((path[i], path[i + 1]))
    except nx.NetworkXNoPath:
        pass

fig_dag = plot_dag(G, highlight_nodes=highlight_nodes, highlight_edges=highlight_edges)
st.plotly_chart(fig_dag, use_container_width=True)

# Legend
leg_cols = st.columns(4)
leg_cols[0].markdown("🟣 **Habits** (treatments)")
leg_cols[1].markdown("🔵 **Outcomes**")
leg_cols[2].markdown("⬜ **Confounders**")
leg_cols[3].markdown("⚫ **Mediators**")

# ---------------------------------------------------------------------------
# Trustworthiness — Refutation Tests
# ---------------------------------------------------------------------------

st.divider()
st.header("🛡️ Trustworthiness — Refutation Tests")
st.markdown(
    "Refutation tests check whether our causal estimate is robust. "
    "If a placebo treatment still shows an effect, or adding a random "
    "confounder changes the estimate drastically, we should be cautious."
)

ref_outcome = st.selectbox(
    "Check refutations for which outcome?",
    options=OUTCOMES,
    format_func=label,
    key="ref_outcome",
)

if st.button("🔍 Run Refutation Tests", type="primary"):
    with st.spinner("Running refutation tests (this may take a moment)…"):
        refutations = run_refutations(treatment, ref_outcome, data)
    st.markdown(refutation_table_html(refutations), unsafe_allow_html=True)

    # Summary
    passed = sum(1 for r in refutations if r["passed"] is True)
    total = sum(1 for r in refutations if r["passed"] is not None)
    if total > 0:
        if passed == total:
            st.success(f"All {total} refutation tests passed — estimate appears robust! ✅")
        elif passed > 0:
            st.warning(f"{passed}/{total} tests passed — estimate may need scrutiny.")
        else:
            st.error("No refutation tests passed — estimate may not be reliable.")

# ---------------------------------------------------------------------------
# ATE Reference Table
# ---------------------------------------------------------------------------

st.divider()
st.header("📈 Average Treatment Effects (Reference)")
st.markdown("Population-level causal effect of each habit on each outcome (per 1-unit increase).")

@st.cache_data
def compute_ate_table(_data: pd.DataFrame) -> pd.DataFrame:
    ate_data = []
    for t in TREATMENTS:
        row = {"Habit": label(t)}
        for o in OUTCOMES:
            ate = estimate_ate(t, o, _data)
            row[label(o)] = f'{ate["estimate"]:+.3f}'
        ate_data.append(row)
    return pd.DataFrame(ate_data)

with st.spinner("Computing ATEs…"):
    ate_df = compute_ate_table(data)
st.dataframe(ate_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.markdown(
    "<div style='text-align:center; color:#B2BEC3; font-size:0.85em;'>"
    "Built with Streamlit · DoWhy · EconML | "
    "Causal inference for everyday decisions"
    "</div>",
    unsafe_allow_html=True,
)
