# local_search.py
from typing import List, Tuple, Dict, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from src.chunking.semantic_chunker import Chunk
from src.utils.vector_store import ChromaVectorStore
import networkx as nx
import re


def canonicalize(name: str) -> str:
    if not name:
        return ""
    n = name.lower().strip()
    for p in ("dr.", "dr ", "mr.", "mr ", "prof.", "prof ", "mahatma ", "shri ", "babasaheb "):
        if n.startswith(p):
            n = n[len(p):].strip()
    n = re.sub(r"[^\w\s]", "", n)
    return " ".join(n.split())


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def _minmax_scale(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mn = np.min(x)
    mx = np.max(x)
    denom = (mx - mn) + eps
    return (x - mn) / denom


class LocalRAGSearch:
    def __init__(self,
                 embedding_model_name: str = "all-MiniLM-L6-v2",
                 top_k: int = 5,
                 threshold: float = 0.12,
                 direct_top_factor: int = 5,
                 entity_top_chunks: int = 3,
                 neighbor_hops: int = 1,
                 direct_weight: float = 0.6,
                 entity_weight: float = 0.35,
                 neighbor_boost: float = 0.15,
                 vector_store: Optional[ChromaVectorStore] = None):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.top_k = top_k
        self.threshold = threshold
        self.direct_top_factor = direct_top_factor
        self.entity_top_chunks = entity_top_chunks
        self.neighbor_hops = neighbor_hops
        self.direct_weight = direct_weight
        self.entity_weight = entity_weight
        self.neighbor_boost = neighbor_boost
        self.vector_store = vector_store

    def compute_query_embedding(self, query: str) -> np.ndarray:
        return self.embedding_model.encode([query], convert_to_numpy=True)[0]

    def _entity_centroid(self, entity_name: str, entities: Dict[str, any], chunks: List[Chunk]) -> np.ndarray:
        ent = entities.get(entity_name)
        if not ent or not getattr(ent, "chunk_ids", None):
            return None
        chunk_ids = [cid for cid in ent.chunk_ids if cid < len(chunks) and chunks[cid].embedding is not None]
        if not chunk_ids:
            return None
        emb_list = [chunks[cid].embedding for cid in chunk_ids[: self.entity_top_chunks]]
        return np.mean(np.array(emb_list), axis=0)

    def _graph_neighbor_chunks(self, graph: nx.Graph, entity_name: str, depth: int = 1) -> List[int]:
        if entity_name not in graph:
            return []
        visited = {entity_name}
        frontier = {entity_name}
        for _ in range(depth):
            next_front = set()
            for node in frontier:
                for nbr in graph.neighbors(node):
                    if nbr not in visited:
                        visited.add(nbr)
                        next_front.add(nbr)
            frontier = next_front
        chunk_ids = set()
        for node in visited:
            data = graph.nodes.get(node, {})
            for cid in data.get("chunk_ids", []):
                chunk_ids.add(cid)
        return list(chunk_ids)

    def search(self,
               query: str,
               graph: nx.Graph,
               chunks: List[Chunk],
               entities: Dict[str, any]) -> List[Tuple[Chunk, float]]:

        q_emb = self.compute_query_embedding(query)
        results = {}  # chunk_id -> score (float)

        # DIRECT chunk similarity using ChromaDB
        if self.vector_store:
            # Query ChromaDB for top candidates
            top_n = max(self.top_k * self.direct_top_factor, self.top_k * 2)
            chunk_ids_str, similarities, metadatas = self.vector_store.query_chunks(
                q_emb, top_k=top_n
            )
            
            if chunk_ids_str:
                # Convert string IDs to integers
                chunk_ids = [int(cid) for cid in chunk_ids_str]
                
                # Normalize similarities to [0, 1] and apply weight
                sims_array = np.array(similarities)
                direct_scores = _minmax_scale(sims_array) * self.direct_weight
                
                # Store direct results
                for cid, score in zip(chunk_ids, direct_scores):
                    results[cid] = max(results.get(cid, 0.0), float(score))
        else:
            # Fallback to in-memory computation (for backward compatibility)
            chunk_embs = []
            chunk_id_map = []
            for cid, c in enumerate(chunks):
                if c.embedding is not None:
                    chunk_embs.append(c.embedding)
                    chunk_id_map.append(cid)
            if len(chunk_embs) == 0:
                return []
            emb_array = np.array(chunk_embs)

            # DIRECT chunk similarity
            sims = cosine_similarity(q_emb.reshape(1, -1), emb_array)[0]

            # pick top candidates to consider
            top_n = max(self.top_k * self.direct_top_factor, self.top_k * 2)
            direct_idxs = np.argsort(sims)[::-1][:top_n]
            direct_raw = sims[direct_idxs]
            direct_scores = _minmax_scale(direct_raw) * self.direct_weight

            for pos, idx in enumerate(direct_idxs):
                cid = chunk_id_map[idx]
                score = float(direct_scores[pos]) if pos < len(direct_scores) else 0.0
                results[cid] = max(results.get(cid, 0.0), score)

        # ENTITY-based: compute centroids for entities that have chunk_ids (cheap filter)
        entity_centroids = {}
        # limit to entities that actually point to chunks and have frequency > 0
        candidate_entity_names = [n for n, e in entities.items() if getattr(e, "chunk_ids", None)]
        # optional: reduce number of candidate entities to top-N by frequency to speed up
        candidate_entity_names = sorted(candidate_entity_names,
                                        key=lambda n: getattr(entities[n], "frequency", 0),
                                        reverse=True)[:200]

        for name in candidate_entity_names:
            centroid = self._entity_centroid(name, entities, chunks)
            if centroid is not None:
                entity_centroids[name] = centroid

        if entity_centroids:
            names = list(entity_centroids.keys())
            ent_embs = np.array([entity_centroids[n] for n in names])
            ent_sims = cosine_similarity(q_emb.reshape(1, -1), ent_embs)[0]
            # consider top candidate entities only
            ent_order = np.argsort(ent_sims)[::-1][: max(10, self.top_k * 3)]
            for i in ent_order:
                if ent_sims[i] < 0.01:
                    continue
                en = names[i]
                ent_score_norm = float(ent_sims[i])
                ent = entities.get(en)
                if not ent:
                    continue
                # for each chunk of this entity
                for cid in getattr(ent, "chunk_ids", []):
                    if cid >= len(chunks) or chunks[cid].embedding is None:
                        continue
                    chunk_sim = float(cosine_similarity(q_emb.reshape(1, -1),
                                                       chunks[cid].embedding.reshape(1, -1))[0][0])
                    combined = (ent_score_norm * self.entity_weight) + (chunk_sim * (1 - self.entity_weight) * 0.25)
                    neighbor_ids = self._graph_neighbor_chunks(graph, en, depth=self.neighbor_hops)
                    if cid in neighbor_ids:
                        combined += self.neighbor_boost
                    results[cid] = max(results.get(cid, 0.0), combined)

        # If results empty, fall back to direct top candidates
        if not results:
            for pos, idx in enumerate(direct_idxs[: self.top_k * 2]):
                cid = chunk_id_map[idx]
                score = float(direct_scores[pos]) if pos < len(direct_scores) else 0.0
                results[cid] = max(results.get(cid, 0.0), score)

        # Prepare final ranked list
        ranked = sorted(results.items(), key=lambda x: x[1], reverse=True)
        if not ranked:
            return []

        # Use relative filtering but ensure some results remain: compute a dynamic cutoff
        max_score = ranked[0][1]
        cutoff = max(self.threshold, 0.05 * max_score)
        filtered = [(cid, s) for cid, s in ranked if s >= cutoff]

        # If filtering removed everything, keep the top-k from ranked
        if not filtered:
            filtered = ranked[: self.top_k]

        out = []
        for cid, score in filtered[: self.top_k]:
            out.append((chunks[cid], float(score)))

        return out
