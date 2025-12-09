"""Knowledge graph construction from entities and relationships."""

from typing import List, Dict
from collections import defaultdict
import networkx as nx
from src.graph.entity_extractor import Entity
from src.chunking.semantic_chunker import Chunk


class GraphBuilder:
    """Builds NetworkX graph from entities and relationships."""
    
    def __init__(self):
        """Initialize graph builder."""
        self.graph = nx.Graph()
    
    def build_graph(self, entities: List[Entity], relationships: List[Dict]) -> nx.Graph:
        """
        Build NetworkX graph from entities and relationships.
        
        Args:
            entities: List of entities
            relationships: List of relationship dicts with 'source', 'target', 'relation', 'weight'
            
        Returns:
            NetworkX graph
        """
        print("Building knowledge graph...")
        self.graph = nx.Graph()
        
        # Add nodes (entities)
        for entity in entities:
            self.graph.add_node(
                entity.text,
                label=entity.label,
                frequency=entity.frequency,
                chunk_ids=entity.chunk_ids
            )
        
        # Add edges (relationships)
        edge_counter = defaultdict(int)
        for rel in relationships:
            source = rel.get('source')
            target = rel.get('target')
            relation = rel.get('relation', 'unknown')
            
            if source in self.graph and target in self.graph:
                edge_key = (source, target, relation)
                edge_counter[edge_key] += 1
        
        for (source, target, relation), weight in edge_counter.items():
            self.graph.add_edge(
                source,
                target,
                relation=relation,
                weight=weight
            )
        
        print(f"Graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        return self.graph

