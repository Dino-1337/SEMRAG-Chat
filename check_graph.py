"""Quick script to check if knowledge graph was created and show basic stats."""

import sys
from pathlib import Path
from src.pipeline.ambedkargpt import AmbedkarGPT


def main():
    """Quick check of graph creation."""
    
    # Check if index exists
    metadata_path = Path("data/processed/metadata.json")
    if not metadata_path.exists():
        print("❌ ERROR: Pre-built index not found!")
        print("\nPlease build the index first by running:")
        print("  python build_index.py")
        sys.exit(1)
    
    print("Loading pre-built index...")
    rag = AmbedkarGPT("config.yaml", mode="query")
    rag.load_index()
    
    # Check graph
    if rag.graph is None:
        print("\n❌ Knowledge graph is None!")
        sys.exit(1)
    
    if rag.graph.number_of_nodes() == 0:
        print("\n❌ Knowledge graph is empty (0 nodes)!")
        sys.exit(1)
    
    print("\n✅ Knowledge graph loaded successfully!")
    print(f"\nQuick Stats:")
    print(f"  • Nodes: {rag.graph.number_of_nodes()}")
    print(f"  • Edges: {rag.graph.number_of_edges()}")
    print(f"  • Communities: {len(rag.communities)}")
    print(f"  • Entities: {len(rag.entities)}")
    print(f"  • Chunks: {len(rag.chunks)}")
    
    # Show sample entities
    print(f"\nSample Entities (first 10):")
    for i, node in enumerate(list(rag.graph.nodes())[:10], 1):
        entity_type = rag.graph.nodes[node].get('label', 'UNKNOWN')
        degree = rag.graph.degree(node)
        print(f"  {i:2d}. {node[:50]:50s} ({entity_type:10s}) - {degree} connections")
    
    print("\n💡 Tip: Run 'python visualize_graph.py' to create visualizations!")


if __name__ == "__main__":
    main()

