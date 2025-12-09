"""Result ranking and combination."""

from typing import List, Tuple
from src.chunking.semantic_chunker import Chunk


class ResultRanker:
    """Ranks and combines retrieval results."""
    
    def combine_results(self,
                       local_results: List[Tuple[Chunk, float]],
                       global_results: List[Tuple[Chunk, float]],
                       strategy: str = "weighted") -> List[Tuple[Chunk, float]]:
        """
        Combine local and global search results.
        
        Args:
            local_results: Results from local search
            global_results: Results from global search
            strategy: Combination strategy ("weighted", "union", "intersection")
            
        Returns:
            Combined and ranked results
        """
        if strategy == "union":
            combined = {}
            for chunk, score in local_results:
                chunk_id = chunk.chunk_id
                if chunk_id not in combined or score > combined[chunk_id][1]:
                    combined[chunk_id] = (chunk, score)
            
            for chunk, score in global_results:
                chunk_id = chunk.chunk_id
                if chunk_id not in combined or score > combined[chunk_id][1]:
                    combined[chunk_id] = (chunk, score)
            
            results = list(combined.values())
            results.sort(key=lambda x: x[1], reverse=True)
            return results
        
        elif strategy == "weighted":
            combined = {}
            
            # Local results get weight 0.6
            for chunk, score in local_results:
                chunk_id = chunk.chunk_id
                weighted_score = score * 0.6
                if chunk_id not in combined:
                    combined[chunk_id] = (chunk, weighted_score)
                else:
                    combined[chunk_id] = (chunk, max(combined[chunk_id][1], weighted_score))
            
            # Global results get weight 0.4
            for chunk, score in global_results:
                chunk_id = chunk.chunk_id
                weighted_score = score * 0.4
                if chunk_id not in combined:
                    combined[chunk_id] = (chunk, weighted_score)
                else:
                    combined[chunk_id] = (chunk, combined[chunk_id][1] + weighted_score)
            
            results = list(combined.values())
            results.sort(key=lambda x: x[1], reverse=True)
            return results
        
        else:
            return local_results

