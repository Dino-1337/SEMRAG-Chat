"""
GraphBuilder: canonical merging, edge weighting, and noise reduction.
Returns a clean NetworkX graph and alias->canonical map.
"""

from typing import List, Dict, Tuple
from collections import defaultdict
import re
import networkx as nx
from src.graph.entity_extractor import Entity


def canonicalize(name: str) -> str:
    if not name:
        return ""
    n = name.lower().strip()
    prefixes = ["dr.", "dr ", "mr.", "mr ", "prof.", "prof ", "mahatma ", "shri ", "babasaheb "]
    for p in prefixes:
        if n.startswith(p):
            n = n[len(p):].strip()
    n = re.sub(r"\b([a-z])\.\s*", r"\1 ", n)
    n = re.sub(r"[^\w\s]", "", n)
    n = " ".join(n.split())
    parts = n.split()
    while len(parts) > 1 and len(parts[0]) == 1:
        parts = parts[1:]
    return " ".join(parts)


class GraphBuilder:
    def __init__(self, min_edge_weight: int = 2, min_node_frequency: int = 1, prune_isolated: bool = True):
        self.graph = nx.Graph()
        self.min_edge_weight = max(1, int(min_edge_weight))
        self.min_node_frequency = max(1, int(min_node_frequency))
        self.prune_isolated = bool(prune_isolated)

    def build_graph(self, entities: List[Entity], relationships: List[Dict]) -> Tuple[nx.Graph, Dict[str, str]]:
        merged = self._merge_entities(entities)
        alias_map = self._add_nodes(merged)
        self._add_edges(relationships, alias_map)
        if self.prune_isolated:
            self._prune_isolated()
        self._prune_low_degree()
        self._prune_low_frequency_nodes()
        return self.graph, alias_map

    def _merge_entities(self, entities: List[Entity]) -> Dict[str, Dict]:
        merged = {}
        for ent in entities:
            c = canonicalize(ent.text)
            if not c:
                continue
            if c not in merged:
                merged[c] = {"aliases": set(), "label": ent.label, "chunk_ids": set(), "frequency": 0}
            merged[c]["aliases"].add(ent.text)
            merged[c]["chunk_ids"].update(getattr(ent, "chunk_ids", []))
            merged[c]["frequency"] += max(1, getattr(ent, "frequency", 1))
        return merged

    def _add_nodes(self, merged_entities: Dict[str, Dict]) -> Dict[str, str]:
        alias_to_canonical: Dict[str, str] = {}
        for cname, data in merged_entities.items():
            if data["frequency"] < self.min_node_frequency:
                continue
            self.graph.add_node(
                cname,
                label=data.get("label"),
                aliases=list(data.get("aliases", [])),
                chunk_ids=list(sorted(data.get("chunk_ids", []))),
                frequency=int(data.get("frequency", 0))
            )
            for a in data.get("aliases", []):
                alias_to_canonical[a] = cname
        return alias_to_canonical

    def _add_edges(self, relationships: List[Dict], alias_map: Dict[str, str]):
        edge_counter = defaultdict(int)
        for rel in relationships:
            src_raw = rel.get("source", "")
            tgt_raw = rel.get("target", "")
            relation = rel.get("relation", "related_to")
            weight = int(rel.get("weight", 1))

            src = canonicalize(src_raw)
            tgt = canonicalize(tgt_raw)
            if not src or not tgt or src == tgt:
                continue

            if src not in self.graph.nodes or tgt not in self.graph.nodes:
                continue

            a, b = (src, tgt) if src <= tgt else (tgt, src)
            key = (a, b, relation)
            edge_counter[key] += weight

        added = 0
        for (a, b, relation), w in edge_counter.items():
            if w < self.min_edge_weight:
                continue
            self.graph.add_edge(a, b, relation=relation, weight=int(w))
            added += 1

    def _prune_isolated(self):
        isolated = [n for n in self.graph.nodes() if self.graph.degree(n) == 0]
        if isolated:
            self.graph.remove_nodes_from(isolated)

    def _prune_low_degree(self):
        """Removing weakly connected nodes with very few connections."""
        low_degree = [n for n in self.graph.nodes() if self.graph.degree(n) <= 1]
        if low_degree:
            self.graph.remove_nodes_from(low_degree)
            print(f"Removed {len(low_degree)} low-degree nodes (degree <= 1)")

    def _prune_low_frequency_nodes(self):
        """Removing rare entities that appear infrequently in the corpus."""
        to_remove = [n for n, d in self.graph.nodes(data=True) if int(d.get("frequency", 0)) < self.min_node_frequency]
        if to_remove:
            self.graph.remove_nodes_from(to_remove)
