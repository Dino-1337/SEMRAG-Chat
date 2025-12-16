"""
Improved Community Summarizer for SEMRAG
- Focuses on key entities
- Uses weighted chunk selection
- Produces structured summaries
- Cleaner, more accurate global search (Eq. 5)
"""

from typing import List, Dict
from src.chunking.semantic_chunker import Chunk
from src.llm.llm_client import LLMClient


class CommunitySummarizer:

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def generate_summaries(self,
                           communities: Dict[int, List[str]],
                           community_chunks: Dict[int, List[int]],
                           chunks: List[Chunk],
                           entities: Dict[str, any],
                           max_chunks: int = 6) -> Dict[int, str]:
        """Generating summaries for each community using LLM."""

        summaries = {}

        for comm_id, entity_names in communities.items():
            if comm_id not in community_chunks:
                continue

            chunk_ids = community_chunks[comm_id]
            selected_chunks = self._select_top_chunks(chunk_ids, chunks, entities, entity_names, max_chunks)

            if selected_chunks:
                summaries[comm_id] = self._summarize_community(selected_chunks, entity_names)

        return summaries

    def _select_top_chunks(self, chunk_ids, chunks, entities, entity_names, max_chunks):
        """Selecting most relevant chunks based on entity overlap."""
        # Weight chunks by how many entities appear in them
        scored = []

        entity_set = {e.lower() for e in entity_names}

        for cid in chunk_ids:
            if cid >= len(chunks):
                continue

            chunk = chunks[cid]

            score = 0
            for entity_name, entity_obj in entities.items():
                if entity_name.lower() in entity_set:
                    if cid in entity_obj.chunk_ids:
                        score += 1

            scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [c for _, c in scored[:max_chunks] if _ > 0]

    def _summarize_community(self, chunks, entity_names):
        """Creating concise summary of community using LLM."""
        text_blocks = [chunk.text for chunk in chunks]
        combined_text = "\n\n".join(text_blocks)

        top_entities = ", ".join(entity_names[:8])

        prompt = f"""
You are summarizing a community of semantically related text passages.

### IMPORTANT:
- Identify the main theme of the community
- Mention the key entities: {top_entities}
- Extract the core concepts and relationships
- Produce a concise but rich high-level summary
- Avoid hallucinations, stick to provided text

### TEXT:
{combined_text[:3500]}

### SUMMARY:
"""

        try:
            summary = self.llm_client.generate(prompt)
            return summary.strip()
        except Exception:
            return chunks[0].text[:200] + "..."
