"""
Improved Knowledge Graph Visualization with Community Coloring
--------------------------------------------------------------
- Colors nodes by community ID
- Node size scaled by frequency
- Hover shows full details
- Cleaner layout, no overlapping labels
- Works with updated GraphBuilder (type=freq fields)
"""

import sys
from pathlib import Path
import networkx as nx
from src.pipeline.ambedkargpt import AmbedkarGPT


def main():
    """Create interactive knowledge graph visualization."""

    # Check pre-built index
    metadata_path = Path("data/processed/metadata.json")
    if not metadata_path.exists():
        print("=" * 60)
        print("ERROR: Pre-built index not found!")
        print("Run: python build_index.py")
        print("=" * 60)
        sys.exit(1)

    # Load graph
    print("Loading index...")
    rag = AmbedkarGPT("config.yaml", mode="query")
    rag.load_index()

    graph = rag.graph
    communities = rag.communities  # {community_id: [entity_names]}

    if graph is None or graph.number_of_nodes() == 0:
        print("ERROR: Graph is empty!")
        sys.exit(1)

    print(f"✓ Loaded graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    # Try importing plotly
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        import plotly.offline as pyo
    except ImportError:
        print("\nERROR: Plotly not installed!")
        print("Install with: pip install plotly")
        sys.exit(1)

    # Assign community ID for each node
    node_to_comm = {}
    for comm_id, nodes in communities.items():
        for node in nodes:
            node_to_comm[node] = comm_id

    # Generate layout
    print("Computing layout (spring_layout)...")
    pos = nx.spring_layout(graph, k=0.7, iterations=40, seed=42)

    # Prepare edge drawing
    edge_x, edge_y = [], []
    for u, v in graph.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=0.4, color="#AAAAAA"),
        hoverinfo="none"
    )

    # Prepare nodes
    node_x, node_y, hover_texts = [], [], []
    node_sizes, node_colors, node_labels = [], [], []

    # Collect frequencies first to normalize sizes
    frequencies = [graph.nodes[node].get("freq") or graph.nodes[node].get("frequency") or 1 for node in graph.nodes()]
    max_freq = max(frequencies) if frequencies else 1
    min_freq = min(frequencies) if frequencies else 1

    for node in graph.nodes():

        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        data = graph.nodes[node]
        entity_type = data.get("type") or data.get("label") or "UNKNOWN"
        freq = data.get("freq") or data.get("frequency") or 1
        degree = graph.degree(node)
        comm_id = node_to_comm.get(node, -1)

        # Normalize size: scale between 8 and 25
        if max_freq > min_freq:
            normalized_size = 8 + ((freq - min_freq) / (max_freq - min_freq)) * 17
        else:
            normalized_size = 15
        
        node_sizes.append(normalized_size)
        node_colors.append(comm_id)
        
        # Shorten label for display
        label = node[:20] + "..." if len(node) > 20 else node
        node_labels.append(label)

        hover_texts.append(
            f"<b>{node}</b><br>"
            f"Type: {entity_type}<br>"
            f"Frequency: {freq}<br>"
            f"Degree: {degree}<br>"
            f"Community: {comm_id}"
        )

    # Generate a color palette
    unique_comms = sorted(set(node_colors))
    color_map = px.colors.qualitative.Dark24 * 20  # Support many communities

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_labels,
        textposition="bottom center",
        textfont=dict(size=8, color="black"),
        hoverinfo="text",
        hovertext=hover_texts,
        marker=dict(
            size=node_sizes,
            color=[color_map[c % len(color_map)] for c in node_colors],
            opacity=0.9,
            line=dict(width=1, color="black")
        )
    )

    # Create figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=f"Knowledge Graph Visualization<br>({graph.number_of_nodes()} Entities, {graph.number_of_edges()} Relationships)",
            title_x=0.5,
            showlegend=False,
            hovermode="closest",
            margin=dict(l=20, r=20, t=60, b=20),
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            plot_bgcolor="white"
        )
    )

    # Save file
    output = "knowledge_graph.html"
    pyo.plot(fig, filename=output, auto_open=False)

    print("\n" + "=" * 60)
    print("✓ Knowledge Graph Visualization Generated!")
    print("=" * 60)
    print(f"File saved at: {output}")
    print("Open in browser to explore.\n")


if __name__ == "__main__":
    main()
