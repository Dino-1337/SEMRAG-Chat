from typing import List, Dict
from sentence_transformers import SentenceTransformer
from src.chunking.semantic_chunker import Chunk
from src.llm.llm_client import LLMClient
from src.llm.prompt_templates import PromptTemplates


class AnswerGenerator:
    """Produces scholarly, grounded answers using retrieved chunks."""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.prompts = PromptTemplates()

    def _is_conceptual(self, q: str) -> bool:
        """Check if the question requires a long explanatory answer."""
        conceptual_keywords = [
            "why", "how", "explain", "reason", "origin", "purpose",
            "what does", "what is ambedkar's view", "in what way",
            "according to ambedkar", "analysis", "interpret"
        ]
        q_lower = q.lower()
        return any(k in q_lower for k in conceptual_keywords)

    def generate_answer(self,
                        query: str,
                        retrieved_chunks: List[Chunk],
                        entities: List[str] = None,
                        community_summaries: Dict[int, str] = None) -> str:

        # Choose depth based on question type
        if self._is_conceptual(query):
            max_chunks = 8   # give more evidence for explanation
        else:
            max_chunks = 4   # factual answers don’t need much noise

        # Build context
        sections = []
        for i, c in enumerate(retrieved_chunks[:max_chunks], 1):
            text = c.text.strip().replace("\n", " ")
            sections.append(f"[Chunk {i}] {text}")

        context = "\n\n".join(sections)

        ent_text = ", ".join(entities[:10]) if entities else "None"

        if community_summaries:
            summ_lines = [
                f"- {s.strip()[:250]}"
                for _, s in list(community_summaries.items())[:3]
            ]
            summary_text = "\n".join(summ_lines)
        else:
            summary_text = "None"

        # Build final prompt
        template = self.prompts.get_answer_template()
        prompt = template.format(
            query=query,
            context=context[:7000],
            entities=ent_text,
            summaries=summary_text
        )

        # Generate answer
        return self.llm_client.generate(prompt)

    def get_community_summary_embedding(self, summary: str):
        return self.embedder.encode([summary], convert_to_numpy=True)[0]
