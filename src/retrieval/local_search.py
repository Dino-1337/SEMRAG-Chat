"""Local RAG Search (Equation 4): Entity-based chunk retrieval."""

from typing import List, Tuple, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from src.chunking.semantic_chunker import Chunk
import networkx as nx


class LocalRAGSearch:
    """Implements Local RAG Search (Equation 4)."""
    
    def __init__(self,
                 embedding_model_name: str = "all-MiniLM-L6-v2",
                 top_k: int = 5,
                 threshold: float = 0.6):
        """
        Initialize local RAG search.
        
        Args:
            embedding_model_name: Name of embedding model
            top_k: Top K chunks to retrieve
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
              graph: nx.Graph,
              chunks: List[Chunk],
              entities: Dict[str, any]) -> List[Tuple[Chunk, float]]:
        """
        Local RAG Search (Equation 4): Entity-based chunk retrieval.
        
        Args:
            query: Query string
            graph: Knowledge graph
            chunks: List of chunks
            entities: Dictionary of entities
            
        Returns:
            List of (chunk, similarity_score) tuples, sorted by score
        """
        query_embedding = self.compute_query_embedding(query)
        
        # Get entity embeddings
        entity_embeddings = {}
        entity_chunk_ids = {}
        
        for entity_name, entity_data in entities.items():
            if entity_name in graph:
                chunk_ids = entity_data.chunk_ids if hasattr(entity_data, 'chunk_ids') else []
                if chunk_ids:
                    chunk_embeddings = [
                        chunks[chunk_id].embedding 
                        for chunk_id in chunk_ids 
                        if chunk_id < len(chunks) and chunks[chunk_id].embedding is not None
                    ]
                    if chunk_embeddings:
                        entity_embeddings[entity_name] = np.mean(chunk_embeddings, axis=0)
                        entity_chunk_ids[entity_name] = chunk_ids
        
        if not entity_embeddings:
            return []
        
        # Find relevant entities
        entity_names = list(entity_embeddings.keys())
        entity_emb_array = np.array([entity_embeddings[name] for name in entity_names])
        
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1),
            entity_emb_array
        )[0]
        
        top_entity_indices = np.argsort(similarities)[::-1][:self.top_k * 2]
        relevant_entity_names = [entity_names[i] for i in top_entity_indices 
                                if similarities[i] >= self.threshold]
        
        # Collect chunks from relevant entities
        chunk_scores = {}
        for entity_name in relevant_entity_names:
            if entity_name in entity_chunk_ids:
                for chunk_id in entity_chunk_ids[entity_name]:
                    if chunk_id < len(chunks):
                        chunk = chunks[chunk_id]
                        if chunk.embedding is not None:
                            chunk_sim = cosine_similarity(
                                query_embedding.reshape(1, -1),
                                chunk.embedding.reshape(1, -1)
                            )[0][0]
                            
                            if chunk_id not in chunk_scores or chunk_sim > chunk_scores[chunk_id][1]:
                                chunk_scores[chunk_id] = (chunk, chunk_sim)
        
        # Sort and return top K
        results = sorted(chunk_scores.values(), key=lambda x: x[1], reverse=True)
        results = [r for r in results if r[1] >= self.threshold]
        
        return results[:self.top_k]

