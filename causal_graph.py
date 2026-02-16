"""
Causal DAG definition and visualization for the Lifestyle What-If Explorer.

The graph encodes domain-knowledge causal relationships between lifestyle
habits and outcomes (energy, mood, productivity).
"""

import networkx as nx
import plotly.graph_objects as go
import numpy as np

# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

# Edges: (cause, effect)
CAUSAL_EDGES = [
    # Lifestyle → Outcomes
    ("sleep", "energy"),
    ("sleep", "mood"),
    ("sleep", "productivity"),
    ("exercise", "energy"),
    ("exercise", "mood"),
    ("exercise", "stress"),
    ("diet_quality", "energy"),
    ("diet_quality", "mood"),
    ("screen_time", "sleep"),
    ("screen_time", "stress"),
    ("screen_time", "mood"),
    ("caffeine", "energy"),
    ("caffeine", "sleep"),
    ("stress", "mood"),
    ("stress", "productivity"),
    ("stress", "sleep"),
    # Confounders
    ("work_hours", "screen_time"),
    ("work_hours", "caffeine"),
    ("work_hours", "exercise"),
    ("work_hours", "stress"),
    ("age", "exercise"),
]

# Node metadata
NODE_META = {
    "sleep":        {"type": "habit",    "label": "Sleep (hrs)",      "color": "#6C5CE7"},
    "exercise":     {"type": "habit",    "label": "Exercise (hrs/wk)","color": "#00B894"},
    "diet_quality": {"type": "habit",    "label": "Diet Quality",     "color": "#FDCB6E"},
    "screen_time":  {"type": "habit",    "label": "Screen Time (hrs)","color": "#E17055"},
    "caffeine":     {"type": "habit",    "label": "Caffeine (cups)",  "color": "#D63031"},
    "stress":       {"type": "mediator", "label": "Stress",           "color": "#636E72"},
    "energy":       {"type": "outcome",  "label": "Energy",           "color": "#0984E3"},
    "mood":         {"type": "outcome",  "label": "Mood",             "color": "#E84393"},
    "productivity": {"type": "outcome",  "label": "Productivity",     "color": "#00CEC9"},
    "work_hours":   {"type": "confounder","label": "Work Hours",      "color": "#B2BEC3"},
    "age":          {"type": "confounder","label": "Age",             "color": "#B2BEC3"},
}

TREATMENTS = ["sleep", "exercise", "diet_quality", "screen_time", "caffeine"]
OUTCOMES   = ["energy", "mood", "productivity"]


def build_dag() -> nx.DiGraph:
    """Build and return the causal DAG as a NetworkX DiGraph."""
    G = nx.DiGraph()
    for src, dst in CAUSAL_EDGES:
        G.add_edge(src, dst)
    for node, meta in NODE_META.items():
        if node in G.nodes:
            G.nodes[node].update(meta)
    return G


def get_gml_string(G: nx.DiGraph) -> str:
    """Return a GML-style string of the DAG for DoWhy."""
    lines = ["graph [directed 1"]
    for n in G.nodes:
        lines.append(f'  node [id "{n}" label "{n}"]')
    for u, v in G.edges:
        lines.append(f'  edge [source "{u}" target "{v}"]')
    lines.append("]")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plotly visualisation
# ---------------------------------------------------------------------------

def _layered_layout(G: nx.DiGraph) -> dict:
    """Create a layered layout for the DAG."""
    layers = {
        "confounder": 0,
        "habit": 1,
        "mediator": 2,
        "outcome": 3,
    }
    # Group nodes by layer
    groups: dict[int, list[str]] = {}
    for n in G.nodes:
        layer = layers.get(G.nodes[n].get("type", "habit"), 1)
        groups.setdefault(layer, []).append(n)

    pos = {}
    for layer, nodes in groups.items():
        nodes_sorted = sorted(nodes)
        n_nodes = len(nodes_sorted)
        for i, node in enumerate(nodes_sorted):
            x = layer
            y = (i - (n_nodes - 1) / 2) * 1.5
            pos[node] = (x, y)
    return pos


def plot_dag(
    G: nx.DiGraph,
    highlight_nodes: set[str] | None = None,
    highlight_edges: set[tuple[str, str]] | None = None,
) -> go.Figure:
    """
    Return a Plotly figure of the causal DAG.

    Parameters
    ----------
    G : nx.DiGraph
        The causal graph.
    highlight_nodes : set[str] | None
        Nodes to highlight (e.g. all nodes on treatment → outcome paths).
    highlight_edges : set[tuple[str, str]] | None
        Edges to highlight as (source, target) tuples.
    """
    pos = _layered_layout(G)
    highlight_set = highlight_nodes or set()
    highlight_edges = highlight_edges or set()

    # --- Edge traces ---
    edge_traces = []
    for u, v in G.edges:
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        is_hl = (u, v) in highlight_edges
        # Arrow via annotation later; draw line here
        edge_traces.append(
            go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                mode="lines",
                line=dict(
                    width=3 if is_hl else 1.2,
                    color="#0984E3" if is_hl else "#B2BEC3",
                ),
                hoverinfo="none",
                showlegend=False,
            )
        )

    # --- Node trace ---
    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    for n in G.nodes:
        x, y = pos[n]
        meta = NODE_META.get(n, {})
        node_x.append(x)
        node_y.append(y)
        node_text.append(meta.get("label", n))
        base_color = meta.get("color", "#636E72")
        node_color.append(base_color)
        node_size.append(38 if n in highlight_set else 28)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        textfont=dict(size=11, color="#2D3436"),
        marker=dict(size=node_size, color=node_color, line=dict(width=2, color="white")),
        hoverinfo="text",
        showlegend=False,
    )

    fig = go.Figure(data=edge_traces + [node_trace])

    # Annotations for arrows
    for u, v in G.edges:
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        is_hl = (u, v) in highlight_edges
        # Shorten arrow so head doesn't overlap node
        dx, dy = x1 - x0, y1 - y0
        length = np.sqrt(dx ** 2 + dy ** 2)
        if length > 0:
            shorten = 0.15
            x1_s = x1 - shorten * dx / length
            y1_s = y1 - shorten * dy / length
        else:
            x1_s, y1_s = x1, y1
        fig.add_annotation(
            x=x1_s, y=y1_s, ax=x0, ay=y0,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.5,
            arrowwidth=2 if is_hl else 1,
            arrowcolor="#0984E3" if is_hl else "#B2BEC3",
        )

    fig.update_layout(
        title=dict(text="Causal DAG — Lifestyle → Outcomes", x=0.5, font=dict(size=16)),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        plot_bgcolor="white",
        margin=dict(l=20, r=20, t=50, b=20),
        height=480,
    )
    return fig
