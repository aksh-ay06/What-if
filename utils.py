"""
Utility helpers for UI formatting, metric cards, and charts.
"""

from __future__ import annotations
import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# Human-readable labels
# ---------------------------------------------------------------------------

LABELS = {
    "sleep": "Sleep (hrs/night)",
    "exercise": "Exercise (hrs/week)",
    "diet_quality": "Diet Quality (1–10)",
    "screen_time": "Screen Time (hrs/day)",
    "caffeine": "Caffeine (cups/day)",
    "stress": "Stress Level (1–10)",
    "energy": "Energy (1–10)",
    "mood": "Mood (1–10)",
    "productivity": "Productivity (1–10)",
    "age": "Age",
    "work_hours": "Work Hours/day",
}

SLIDER_RANGES = {
    "sleep":        (3.0, 12.0, 7.0, 0.5),
    "exercise":     (0.0, 14.0, 3.0, 0.5),
    "diet_quality": (1.0, 10.0, 5.0, 0.5),
    "screen_time":  (0.0, 16.0, 4.0, 0.5),
    "caffeine":     (0.0, 8.0, 2.0, 0.5),
    "stress":       (1.0, 10.0, 5.0, 0.5),
}

DELTA_RANGES = {
    "sleep":        (-3.0, 3.0, 1.0, 0.5),
    "exercise":     (-5.0, 5.0, 1.0, 0.5),
    "diet_quality": (-4.0, 4.0, 1.0, 0.5),
    "screen_time":  (-5.0, 5.0, -1.0, 0.5),
    "caffeine":     (-4.0, 4.0, -1.0, 0.5),
}


def label(name: str) -> str:
    return LABELS.get(name, name.replace("_", " ").title())


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def whatif_bar_chart(results: list[dict]) -> go.Figure:
    """
    Create a bar chart showing predicted change for each outcome.
    `results` is a list of dicts from estimate_whatif_all_outcomes.
    """
    outcomes = [label(r["outcome"]) for r in results]
    values = [r["predicted_change"] for r in results]
    ci_low = [r.get("ci_low") for r in results]
    ci_high = [r.get("ci_high") for r in results]

    has_ci = all(lo is not None and hi is not None for lo, hi in zip(ci_low, ci_high))
    errors_minus = [max(0, v - lo) for v, lo in zip(values, ci_low)] if has_ci else None
    errors_plus = [max(0, hi - v) for v, hi in zip(values, ci_high)] if has_ci else None

    colors = ["#E74C3C" if v < 0 else "#27AE60" for v in values]

    error_y_config = None
    if has_ci and errors_plus and errors_minus:
        error_y_config = dict(
            type="data",
            symmetric=False,
            array=errors_plus,
            arrayminus=errors_minus,
            color="#636E72",
            thickness=1.5,
            width=6,
        )

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=outcomes,
        y=values,
        marker_color=colors,
        error_y=error_y_config,
        text=[f"{v:+.3f}" for v in values],
        textposition="outside",
    ))

    fig.update_layout(
        title=dict(text="Estimated Causal Effect on Outcomes", x=0.5, font=dict(size=15)),
        yaxis_title="Predicted Change (units)",
        xaxis_title="",
        plot_bgcolor="white",
        height=380,
        margin=dict(l=40, r=40, t=60, b=40),
        yaxis=dict(gridcolor="#ECF0F1"),
    )
    return fig


def refutation_table_html(refutations: list[dict]) -> str:
    """Convert refutation results to a styled HTML table."""
    rows = []
    for r in refutations:
        status = "—"
        if r["passed"] is True:
            status = "✅ Passed"
        elif r["passed"] is False:
            status = "⚠️ Warning"

        new_eff = f'{r["new_effect"]:.4f}' if r["new_effect"] is not None else "N/A"
        rows.append(f"""
        <tr>
            <td style='padding:8px; font-weight:600;'>{r['test_name']}</td>
            <td style='padding:8px; color:#636E72; font-size:0.9em;'>{r['description']}</td>
            <td style='padding:8px; text-align:center;'>{r['estimated_effect']:.4f}</td>
            <td style='padding:8px; text-align:center;'>{new_eff}</td>
            <td style='padding:8px; text-align:center;'>{status}</td>
        </tr>""")

    return f"""
    <table style='width:100%; border-collapse:collapse; font-size:0.95em;'>
        <thead>
            <tr style='background:#F8F9FA; border-bottom:2px solid #DEE2E6;'>
                <th style='padding:8px; text-align:left;'>Test</th>
                <th style='padding:8px; text-align:left;'>What it checks</th>
                <th style='padding:8px; text-align:center;'>Original</th>
                <th style='padding:8px; text-align:center;'>After Test</th>
                <th style='padding:8px; text-align:center;'>Result</th>
            </tr>
        </thead>
        <tbody>
            {"".join(rows)}
        </tbody>
    </table>
    """
