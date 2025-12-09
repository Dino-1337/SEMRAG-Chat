"""Answer generation using LLM."""

from typing import List, Dict
from sentence_transformers import SentenceTransformer
from src.chunking.semantic_chunker import Chunk
from src.llm.llm_client import LLMClient
from src.llm.prompt_templates import PromptTemplates


class AnswerGenerator:
    """Generates answers to queries using retrieved context."""
    
    def __init__(self, llm_client: LLMClient):
        """
        Initialize answer generator.
        
        Args:
            llm_client: LLM client for generation
        """
        self.llm_client = llm_client
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.prompt_templates = PromptTemplates()
    
    def generate_answer(self,
                       query: str,
                       retrieved_chunks: List[Chunk],
                       entities: List[str] = None,
                       community_summaries: Dict[int, str] = None) -> str:
        """
        Generate answer to query using retrieved context.
        
        Args:
            query: User query
            retrieved_chunks: List of retrieved chunks
            entities: List of relevant entities (optional)
            community_summaries: Dictionary of community summaries (optional)
            
        Returns:
            Generated answer
        """
        # Prepare context
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks[:5], 1):
            context_parts.append(f"[Context {i}]\n{chunk.text}")
        
        context = "\n\n".join(context_parts)
        
        # Add entity information
        entity_info = ""
        if entities:
            entity_info = f"\n\nRelevant entities mentioned: {', '.join(entities[:10])}"
        
        # Add community summaries
        summary_info = ""
        if community_summaries:
            summaries_text = "\n".join([
                f"Community {cid}: {summary[:200]}"
                for cid, summary in list(community_summaries.items())[:3]
            ])
            summary_info = f"\n\nRelated themes:\n{summaries_text}"
        
        # Generate answer
        template = self.prompt_templates.get_answer_template()
        prompt = template.format(
            query=query,
            context=context[:4000],
            entity_info=entity_info,
            summary_info=summary_info
        )
        
        return self.llm_client.generate(prompt)
    
    def get_community_summary_embedding(self, summary: str) -> List[float]:
        """Get embedding for a community summary."""
        embedding = self.embedding_model.encode([summary], convert_to_numpy=True)[0]
        return embedding.tolist()

