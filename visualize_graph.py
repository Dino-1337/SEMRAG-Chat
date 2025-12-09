"""Generate interactive HTML visualization of the knowledge graph."""

import sys
from pathlib import Path
import networkx as nx
from src.pipeline.ambedkargpt import AmbedkarGPT


def main():
    """Create interactive knowledge graph visualization."""
    
    # Check if index exists
    metadata_path = Path("data/processed/metadata.json")
    if not metadata_path.exists():
        print("=" * 60)
        print("ERROR: Pre-built index not found!")
        print("=" * 60)
        print("Please build the index first:")
        print("  python build_index.py")
        print("=" * 60)
        sys.exit(1)
    
    # Load index
    print("\nLoading knowledge graph...")
    try:
        rag = AmbedkarGPT("config.yaml", mode="query")
        rag.load_index()
    except Exception as e:
        print(f"Error loading index: {e}")
        sys.exit(1)
    
    # Verify graph exists
    if rag.graph is None or rag.graph.number_of_nodes() == 0:
        print("ERROR: Knowledge graph is empty!")
        sys.exit(1)
    
    print(f"✓ Loaded graph: {rag.graph.number_of_nodes()} nodes, {rag.graph.number_of_edges()} edges")
    
    # Check for plotly
    try:
        import plotly.graph_objects as go
        import plotly.offline as pyo
    except ImportError:
        print("\nERROR: Plotly not installed!")
        print("Install with: pip install plotly")
        sys.exit(1)
    
    # Create visualization
    print("\nGenerating interactive visualization...")
    
    # Calculate layout
    pos = nx.spring_layout(rag.graph, k=1, iterations=50, seed=42)
    
    # Create edges
    edge_x = []
    edge_y = []
    for edge in rag.graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Create nodes
    node_x = []
    node_y = []
    node_text = []
    node_hover = []
    node_colors = []
    
    for node in rag.graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Node label (shortened for display)
        label = node[:30] + "..." if len(node) > 30 else node
        node_text.append(label)
        
        # Hover info
        entity_type = rag.graph.nodes[node].get('label', 'UNKNOWN')
        connections = rag.graph.degree(node)
        frequency = rag.graph.nodes[node].get('frequency', 0)
        
        hover_info = f"<b>{node}</b><br>"
        hover_info += f"Type: {entity_type}<br>"
        hover_info += f"Connections: {connections}<br>"
        hover_info += f"Frequency: {frequency}"
        node_hover.append(hover_info)
        
        # Color by connection count
        node_colors.append(connections)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="top center",
        textfont=dict(size=8),
        hovertext=node_hover,
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            reversescale=False,
            color=node_colors,
            size=10,
            colorbar=dict(
                thickness=15,
                title='Connections',
                xanchor='left'
            ),
            line=dict(width=2, color='white')
        )
    )
    
    # Create figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(
                text=f'Knowledge Graph - {rag.graph.number_of_nodes()} Entities, {rag.graph.number_of_edges()} Relationships',
                x=0.5,
                xanchor='center'
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[dict(
                text="Hover over nodes for details. Zoom and pan to explore.",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
    )
    
    # Save to HTML
    output_file = "knowledge_graph.html"
    pyo.plot(fig, filename=output_file, auto_open=False)
    
    print(f"\n{'='*60}")
    print("✓ Visualization created successfully!")
    print(f"{'='*60}")
    print(f"\nFile: {output_file}")
    print("\nOpen this file in your browser to explore the knowledge graph!")
    print("- Zoom: Scroll wheel or pinch")
    print("- Pan: Click and drag")
    print("- Details: Hover over nodes")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
