"""Knowledge graph visualization utilities."""

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from typing import Dict, List, Optional
import numpy as np


class GraphVisualizer:
    """Visualize knowledge graphs with communities and entities."""
    
    def __init__(self, figsize=(20, 15)):
        """Initialize visualizer."""
        self.figsize = figsize
    
    def visualize_graph(self,
                       graph: nx.Graph,
                       communities: Optional[Dict[int, List[str]]] = None,
                       output_path: str = "knowledge_graph.png",
                       layout: str = "spring",
                       node_size: int = 1000,
                       font_size: int = 8):
        """Visualize knowledge graph with optional community coloring."""
        if graph.number_of_nodes() == 0:
            print("Graph is empty, nothing to visualize.")
            return
        
        plt.figure(figsize=self.figsize)
        
        # Choose layout
        if layout == "spring":
            pos = nx.spring_layout(graph, k=1, iterations=50, seed=42)
        elif layout == "circular":
            pos = nx.circular_layout(graph)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(graph)
        elif layout == "spectral":
            pos = nx.spectral_layout(graph)
        else:
            pos = nx.spring_layout(graph, seed=42)
        
        # Color nodes by community if provided
        if communities:
            node_colors = self._get_community_colors(graph, communities)
            node_labels = {node: node[:20] + "..." if len(node) > 20 else node 
                          for node in graph.nodes()}
        else:
            node_colors = [graph.nodes[node].get('label', 'UNKNOWN') 
                         for node in graph.nodes()]
            unique_labels = list(set(node_colors))
            color_map = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
            label_to_color = {label: color_map[i] for i, label in enumerate(unique_labels)}
            node_colors = [label_to_color.get(color, 'gray') for color in node_colors]
            node_labels = {node: f"{node[:15]}...\n({graph.nodes[node].get('label', '?')})" 
                          if len(node) > 15 else f"{node}\n({graph.nodes[node].get('label', '?')})"
                          for node in graph.nodes()}
        
        # Draw nodes
        nx.draw_networkx_nodes(
            graph, pos,
            node_color=node_colors if isinstance(node_colors, list) and isinstance(node_colors[0], (str, tuple, list)) else 'lightblue',
            node_size=node_size,
            alpha=0.9
        )
        
        # Draw edges
        nx.draw_networkx_edges(
            graph, pos,
            alpha=0.5,
            width=1,
            edge_color='gray'
        )
        
        # Draw labels
        nx.draw_networkx_labels(
            graph, pos,
            labels=node_labels,
            font_size=font_size,
            font_weight='bold'
        )
        
        # Add title
        title = f"Knowledge Graph\n{graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
        if communities:
            title += f", {len(communities)} communities"
        plt.title(title, fontsize=16, fontweight='bold')
        
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Graph visualization saved to: {output_path}")
        plt.close()
    
    def _get_community_colors(self, graph: nx.Graph, communities: Dict[int, List[str]]) -> List:
        """Get colors for nodes based on communities."""
        num_communities = len(communities)
        colors = plt.cm.tab20(np.linspace(0, 1, min(num_communities, 20)))
        if num_communities > 20:
            colors = plt.cm.Set3(np.linspace(0, 1, num_communities))
        
        node_to_community = {}
        for comm_id, nodes in communities.items():
            for node in nodes:
                node_to_community[node] = comm_id
        
        node_colors = []
        for node in graph.nodes():
            comm_id = node_to_community.get(node, -1)
            if comm_id >= 0:
                node_colors.append(colors[comm_id % len(colors)])
            else:
                node_colors.append('gray')
        
        return node_colors
    
    def print_graph_statistics(self, graph: nx.Graph, communities: Optional[Dict[int, List[str]]] = None):
        """Print detailed statistics about the knowledge graph."""
        print("\n" + "=" * 60)
        print("KNOWLEDGE GRAPH STATISTICS")
        print("=" * 60)
        
        print(f"\n📊 Basic Statistics:")
        print(f"  • Total Nodes (Entities): {graph.number_of_nodes()}")
        print(f"  • Total Edges (Relationships): {graph.number_of_edges()}")
        print(f"  • Graph Density: {nx.density(graph):.4f}")
        print(f"  • Number of Connected Components: {nx.number_connected_components(graph)}")
        
        if graph.number_of_nodes() > 0:
            degrees = dict(graph.degree())
            print(f"\n📈 Node Degree Statistics:")
            print(f"  • Average Degree: {sum(degrees.values()) / len(degrees):.2f}")
            print(f"  • Max Degree: {max(degrees.values())}")
            print(f"  • Min Degree: {min(degrees.values())}")
            
            top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]
            print(f"\n🔗 Top 10 Most Connected Entities:")
            for i, (node, degree) in enumerate(top_nodes, 1):
                entity_type = graph.nodes[node].get('label', 'UNKNOWN')
                print(f"  {i:2d}. {node[:40]:40s} ({entity_type:10s}) - {degree} connections")
        
        # Entity type distribution
        entity_types = {}
        for node in graph.nodes():
            etype = graph.nodes[node].get('label', 'UNKNOWN')
            entity_types[etype] = entity_types.get(etype, 0) + 1
        
        if entity_types:
            print(f"\n🏷️  Entity Type Distribution:")
            for etype, count in sorted(entity_types.items(), key=lambda x: x[1], reverse=True):
                print(f"  • {etype:15s}: {count:4d} entities")
        
        # Community statistics
        if communities:
            print(f"\n👥 Community Statistics:")
            print(f"  • Total Communities: {len(communities)}")
            comm_sizes = [len(nodes) for nodes in communities.values()]
            print(f"  • Average Community Size: {sum(comm_sizes) / len(comm_sizes):.2f}")
            print(f"  • Largest Community: {max(comm_sizes)} entities")
            print(f"  • Smallest Community: {min(comm_sizes)} entities")
            
            print(f"\n📋 Top 5 Largest Communities:")
            sorted_comms = sorted(communities.items(), key=lambda x: len(x[1]), reverse=True)[:5]
            for i, (comm_id, nodes) in enumerate(sorted_comms, 1):
                print(f"  {i}. Community {comm_id}: {len(nodes)} entities")
                print(f"     Sample entities: {', '.join(nodes[:5])}")
                if len(nodes) > 5:
                    print(f"     ... and {len(nodes) - 5} more")
        
        print("=" * 60 + "\n")
    
    def export_graph_data(self, graph: nx.Graph, output_path: str = "knowledge_graph.json"):
        """Export graph data to JSON format."""
        import json
        
        graph_data = {
            "nodes": [],
            "edges": []
        }
        
        for node in graph.nodes():
            node_data = {
                "id": node,
                "label": graph.nodes[node].get('label', 'UNKNOWN'),
                "frequency": graph.nodes[node].get('frequency', 1),
                "chunk_ids": graph.nodes[node].get('chunk_ids', [])
            }
            graph_data["nodes"].append(node_data)
        
        for u, v, data in graph.edges(data=True):
            edge_data = {
                "source": u,
                "target": v,
                "relation": data.get('relation', 'unknown'),
                "weight": data.get('weight', 1)
            }
            graph_data["edges"].append(edge_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)
        
        print(f"Graph data exported to: {output_path}")

