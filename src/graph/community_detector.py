"""Community detection in knowledge graphs."""

from typing import Dict, List
import networkx as nx
from collections import defaultdict

try:
    import leidenalg
    from igraph import Graph as IGraph
    LEIDEN_AVAILABLE = True
except ImportError:
    LEIDEN_AVAILABLE = False
    print("Warning: leidenalg/igraph not available. Using networkx community detection.")


class CommunityDetector:
    """Detects communities in knowledge graphs."""
    
    def __init__(self, community_resolution: float = 1.0):
        """
        Initialize community detector.
        
        Args:
            community_resolution: Resolution parameter for Leiden algorithm
        """
        self.community_resolution = community_resolution
    
    def detect_communities(self, graph: nx.Graph) -> Dict[int, List[str]]:
        """
        Detect communities in the knowledge graph.
        
        Args:
            graph: NetworkX graph
            
        Returns:
            Dictionary mapping community ID to list of entity names
        """
        if graph.number_of_nodes() == 0:
            return {}
        
        print("Detecting communities...")
        
        if LEIDEN_AVAILABLE:
            communities = self._detect_communities_leiden(graph)
        else:
            communities = self._detect_communities_louvain(graph)
        
        print(f"Found {len(communities)} communities")
        return communities
    
    def _detect_communities_leiden(self, graph: nx.Graph) -> Dict[int, List[str]]:
        """Detect communities using Leiden algorithm."""
        node_list = list(graph.nodes())
        node_to_idx = {node: i for i, node in enumerate(node_list)}
        
        edges = [(node_to_idx[u], node_to_idx[v]) 
                 for u, v in graph.edges()]
        
        ig = IGraph(edges, directed=False)
        
        partition = leidenalg.find_partition(
            ig,
            leidenalg.ModularityVertexPartition,
            resolution_parameter=self.community_resolution
        )
        
        communities = defaultdict(list)
        for i, community_id in enumerate(partition.membership):
            node_name = node_list[i]
            communities[community_id].append(node_name)
        
        return dict(communities)
    
    def _detect_communities_louvain(self, graph: nx.Graph) -> Dict[int, List[str]]:
        """Detect communities using NetworkX (fallback)."""
        try:
            import networkx.algorithms.community as nx_comm
            communities_dict = {}
            communities = nx_comm.greedy_modularity_communities(graph)
            for i, comm in enumerate(communities):
                communities_dict[i] = list(comm)
            return communities_dict
        except Exception:
            print("Warning: Using connected components as communities")
            communities = {}
            for i, component in enumerate(nx.connected_components(graph)):
                communities[i] = list(component)
            return communities
    
    def get_community_chunks(self, 
                            communities: Dict[int, List[str]], 
                            entities: Dict[str, any],
                            chunks: List) -> Dict[int, List[int]]:
        """
        Get chunk IDs associated with each community.
        
        Args:
            communities: Community dictionary
            entities: Dictionary of entities
            chunks: List of chunks
            
        Returns:
            Dictionary mapping community ID to list of chunk IDs
        """
        community_chunks = defaultdict(set)
        
        for comm_id, entity_names in communities.items():
            for entity_name in entity_names:
                if entity_name in entities:
                    entity = entities[entity_name]
                    if hasattr(entity, 'chunk_ids'):
                        community_chunks[comm_id].update(entity.chunk_ids)
        
        return {comm_id: list(chunk_ids) for comm_id, chunk_ids in community_chunks.items()}

