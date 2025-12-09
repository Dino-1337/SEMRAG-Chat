"""Global RAG Search (Equation 5): Community-based chunk retrieval."""

from typing import List, Tuple, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from src.chunking.semantic_chunker import Chunk


class GlobalRAGSearch:
    """Implements Global RAG Search (Equation 5)."""
    
    def __init__(self,
                 embedding_model_name: str = "all-MiniLM-L6-v2",
                 top_k: int = 3,
                 threshold: float = 0.5):
        """
        Initialize global RAG search.
        
        Args:
            embedding_model_name: Name of embedding model
            top_k: Top K communities to retrieve
            threshold: Similarity threshold
        """
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.top_k = top_k
        self.threshold = threshold
    
    def compute_query_embedding(self, query: str) -> np.ndarray:
        """Compute embedding for a query."""
        return self.embedding_model.encode([query], convert_to_numpy=True)[0]
    
    def search(self,
              query: str,
              communities: Dict[int, List[str]],
              community_summaries: Dict[int, str],
              community_summary_embeddings: Dict[int, np.ndarray],
              chunks: List[Chunk],
              community_chunks: Dict[int, List[int]]) -> List[Tuple[Chunk, float]]:
        """
        Global RAG Search (Equation 5): Community-based chunk retrieval.
        
        Args:
            query: Query string
            communities: Dictionary mapping community ID to entity names
            community_summaries: Dictionary mapping community ID to summary text
            community_summary_embeddings: Dictionary mapping community ID to summary embedding
            chunks: List of chunks
            community_chunks: Dictionary mapping community ID to chunk IDs
            
        Returns:
            List of (chunk, similarity_score) tuples, sorted by score
        """
        query_embedding = self.compute_query_embedding(query)
        
        if not community_summary_embeddings:
            return []
        
        # Compute similarities with community summaries
        comm_ids = list(community_summary_embeddings.keys())
        comm_embeddings = np.array([community_summary_embeddings[cid] for cid in comm_ids])
        
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1),
            comm_embeddings
        )[0]
        
        # Get top-K communities
        top_comm_indices = np.argsort(similarities)[::-1][:self.top_k]
        relevant_comm_ids = [
            comm_ids[i] for i in top_comm_indices 
            if similarities[i] >= self.threshold
        ]
        
        # Collect chunks from relevant communities
        chunk_scores = {}
        for comm_id in relevant_comm_ids:
            if comm_id in community_chunks:
                comm_similarity = similarities[comm_ids.index(comm_id)]
                
                for chunk_id in community_chunks[comm_id]:
                    if chunk_id < len(chunks):
                        chunk = chunks[chunk_id]
                        if chunk.embedding is not None:
                            chunk_sim = cosine_similarity(
                                query_embedding.reshape(1, -1),
                                chunk.embedding.reshape(1, -1)
                            )[0][0]
                            
                            combined_score = comm_similarity * 0.5 + chunk_sim * 0.5
                            
                            if chunk_id not in chunk_scores or combined_score > chunk_scores[chunk_id][1]:
                                chunk_scores[chunk_id] = (chunk, combined_score)
        
        # Sort and return top results
        results = sorted(chunk_scores.values(), key=lambda x: x[1], reverse=True)
        results = [r for r in results if r[1] >= self.threshold]
        
        return results[:self.top_k * 2]  # Return more results for ranking

