"""
Improved Community Detection for SEMRAG
- Works with canonical names
- Removes tiny/noisy communities
- Computes chunk membership cleanly
- Produces stable, sorted communities
"""

from typing import Dict, List, Set
from collections import defaultdict
import networkx as nx

try:
    import leidenalg
    from igraph import Graph as IGraph
    LEIDEN_AVAILABLE = True
except ImportError:
    LEIDEN_AVAILABLE = False
    print("Warning: Leiden unavailable — using Louvain.")


MIN_COMMUNITY_SIZE = 4     # prune tiny useless clusters
MAX_COMMUNITY_KEEP = 50    # soft cap for very large clusters


class CommunityDetector:
    def __init__(self, community_resolution: float = 1.0):
        self.community_resolution = community_resolution

    def detect_communities(self, graph: nx.Graph) -> Dict[int, List[str]]:
        if graph.number_of_nodes() == 0:
            return {}

        print("Detecting communities...")

        if LEIDEN_AVAILABLE:
            raw = self._detect_leiden(graph)
        else:
            raw = self._detect_louvain(graph)

        filtered = [sorted(list(nodes)) for nodes in raw if len(nodes) >= MIN_COMMUNITY_SIZE]

        filtered.sort(key=len, reverse=True)

        final = {i: comm for i, comm in enumerate(filtered)}

        print(f"Kept {len(final)} communities (min size = {MIN_COMMUNITY_SIZE})")
        return final

    def _detect_leiden(self, graph: nx.Graph) -> List[Set[str]]:
        """Using Leiden algorithm for community detection."""
        node_list = list(graph.nodes())
        idx_map = {n: i for i, n in enumerate(node_list)}

        edges = [(idx_map[u], idx_map[v]) for u, v in graph.edges()]
        ig = IGraph(edges=edges, directed=False)

        partition = leidenalg.find_partition(
            ig,
            leidenalg.ModularityVertexPartition,
            resolution_parameter=self.community_resolution
        )

        communities = defaultdict(list)
        for node_idx, comm_id in enumerate(partition.membership):
            communities[comm_id].append(node_list[node_idx])

        return list(communities.values())

    def _detect_louvain(self, graph: nx.Graph) -> List[Set[str]]:
        """Using Louvain algorithm as fallback when Leiden is not available."""
        import networkx.algorithms.community as nx_comm
        comms = nx_comm.greedy_modularity_communities(graph)
        return [set(c) for c in comms]

    def get_community_chunks(self,
                             communities: Dict[int, List[str]],
                             entities: Dict[str, any],
                             chunks: List) -> Dict[int, List[int]]:
        """Mapping communities to their associated chunk IDs."""

        comm_chunks = defaultdict(set)

        for cid, node_list in communities.items():
            for cname in node_list:
                if cname not in entities:
                    continue
                ent = entities[cname]
                for chunk_id in getattr(ent, "chunk_ids", []):
                    if 0 <= chunk_id < len(chunks):
                        comm_chunks[cid].add(chunk_id)

        return {cid: sorted(list(chset)) for cid, chset in comm_chunks.items()}
