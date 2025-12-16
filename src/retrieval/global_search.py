# global_search.py
from typing import List, Tuple, Dict, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from src.chunking.semantic_chunker import Chunk
from src.utils.vector_store import ChromaVectorStore


def _minmax_scale(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mn, mx = np.min(x), np.max(x)
    return (x - mn) / (mx - mn + eps)


class GlobalRAGSearch:
    def __init__(self,
                 embedding_model_name: str = "all-MiniLM-L6-v2",
                 top_k: int = 3,
                 threshold: float = 0.2,
                 size_penalty_factor: float = 0.45,
                 small_comm_boost: float = 0.1,
                 vector_store: Optional[ChromaVectorStore] = None):

        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.top_k = top_k
        self.threshold = threshold
        self.size_penalty_factor = size_penalty_factor
        self.small_comm_boost = small_comm_boost
        self.vector_store = vector_store

    def compute_query_embedding(self, query: str) -> np.ndarray:
        return self.embedding_model.encode([query], convert_to_numpy=True)[0]

    def _size_penalty(self, comm_size: int, max_size: int) -> float:
        if max_size <= 0:
            return 0.0
        return (comm_size / max_size) * self.size_penalty_factor

    def search(self,
               query: str,
               communities: Dict[int, List[str]],
               community_summaries: Dict[int, str],
               community_summary_embeddings: Dict[int, np.ndarray],
               chunks: List[Chunk],
               community_chunks: Dict[int, List[int]]) -> List[Tuple[Chunk, float]]:

        if not community_summary_embeddings and not self.vector_store:
            return []

        q_emb = self.compute_query_embedding(query)
        scored_chunks = {}

        # Use ChromaDB if available
        if self.vector_store:
            # Query ChromaDB for top communities
            top_n = max(10, self.top_k * 3)
            comm_ids, similarities, summaries = self.vector_store.query_communities(
                q_emb, top_k=top_n
            )
            
            if not comm_ids:
                return []
            
            # Convert to arrays for processing
            raw_sims = np.array(similarities)
            summary_norm = _minmax_scale(raw_sims)
            
        else:
            # Fallback to in-memory computation
            comm_ids = list(community_summary_embeddings.keys())
            comm_embs = np.array([community_summary_embeddings[cid] for cid in comm_ids])

            # raw similarities
            raw_sims = cosine_similarity(q_emb.reshape(1, -1), comm_embs)[0]

            # normalize to preserve variation
            summary_norm = _minmax_scale(raw_sims)

        max_comm_size = max((len(community_chunks.get(cid, [])) for cid in comm_ids), default=1)

        # consider top scoring communities
        top_idxs = np.argsort(raw_sims)[::-1][: max(10, self.top_k * 3)]

        for idx in top_idxs:
            cid = comm_ids[idx]
            comm_raw = raw_sims[idx]

            # relative thresholding
            if comm_raw < np.max(raw_sims) * 0.35:
                continue

            penalty = self._size_penalty(len(community_chunks.get(cid, [])), max_comm_size)
            boost = self.small_comm_boost if len(community_chunks.get(cid, [])) < 0.3 * max_comm_size else 0.0

            summary_score = summary_norm[idx]
            summary_score = summary_score * (1 - penalty) + boost

            # score each chunk in the community
            for cid_chunk in community_chunks.get(cid, []):
                if cid_chunk >= len(chunks):
                    continue
                c = chunks[cid_chunk]
                if c.embedding is None:
                    continue

                chunk_sim = float(cosine_similarity(q_emb.reshape(1, -1),
                                                    c.embedding.reshape(1, -1))[0][0])

                combined = 0.6 * summary_score + 0.4 * chunk_sim

                prev = scored_chunks.get(cid_chunk)
                if prev is None or combined > prev[1]:
                    scored_chunks[cid_chunk] = (c, combined)

        results = sorted(scored_chunks.values(), key=lambda x: x[1], reverse=True)

        # safer filtering
        if results:
            top_score = results[0][1]
            filtered = [r for r in results if r[1] >= top_score * 0.30]
        else:
            filtered = []

        # guarantee return
        if not filtered:
            filtered = results[: self.top_k * 2]

        return filtered[: self.top_k * 2]
