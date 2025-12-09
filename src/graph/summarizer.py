"""Community summarization using LLM."""

from typing import List, Dict
from src.chunking.semantic_chunker import Chunk
from src.llm.llm_client import LLMClient


class CommunitySummarizer:
    """Generates summaries for communities of chunks."""
    
    def __init__(self, llm_client: LLMClient):
        """
        Initialize community summarizer.
        
        Args:
            llm_client: LLM client for generating summaries
        """
        self.llm_client = llm_client
    
    def generate_summaries(self, 
                          communities: Dict[int, List[str]],
                          community_chunks: Dict[int, List[int]],
                          chunks: List[Chunk]) -> Dict[int, str]:
        """
        Generate summaries for all communities.
        
        Args:
            communities: Dictionary mapping community ID to entity names
            community_chunks: Dictionary mapping community ID to chunk IDs
            chunks: List of chunks
            
        Returns:
            Dictionary mapping community ID to summary text
        """
        summaries = {}
        
        for comm_id, entity_names in communities.items():
            if comm_id in community_chunks:
                chunk_ids = community_chunks[comm_id]
                community_chunks_list = [chunks[cid] for cid in chunk_ids if cid < len(chunks)]
                
                if community_chunks_list:
                    summary = self.generate_summary(community_chunks_list, comm_id)
                    summaries[comm_id] = summary
        
        return summaries
    
    def generate_summary(self, community_chunks: List[Chunk], community_id: int) -> str:
        """
        Generate a summary for a community of chunks.
        
        Args:
            community_chunks: List of chunks in the community
            community_id: ID of the community
            
        Returns:
            Summary text
        """
        if not community_chunks:
            return ""
        
        combined_text = "\n\n".join([chunk.text for chunk in community_chunks[:10]])
        
        prompt = f"""Summarize the following text passages that are related to each other. 
Focus on the main themes, key concepts, and important information.

Text:
{combined_text[:3000]}

Summary:"""
        
        try:
            summary = self.llm_client.generate(prompt)
            return summary.strip()
        except Exception as e:
            print(f"Error generating community summary: {e}")
            first_chunk = community_chunks[0]
            return first_chunk.text[:200] + "..."

