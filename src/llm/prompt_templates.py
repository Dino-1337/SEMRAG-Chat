"""Prompt templates for LLM interactions."""

try:
    from langchain_core.prompts import PromptTemplate
except ImportError:
    from langchain.prompts import PromptTemplate


class PromptTemplates:
    """Manages prompt templates for different tasks."""
    
    @staticmethod
    def get_community_summary_template() -> PromptTemplate:
        """Get template for community summarization."""
        return PromptTemplate(
            input_variables=["text"],
            template="""Summarize the following text passages that are related to each other. 
Focus on the main themes, key concepts, and important information.

Text:
{text}

Summary:"""
        )
    
    @staticmethod
    def get_answer_template() -> PromptTemplate:
        """Get template for answer generation with strict grounding."""
        return PromptTemplate(
            input_variables=["query", "context", "entity_info", "summary_info"],
            template="""You are an AI assistant answering questions about Dr. B.R. Ambedkar's works and philosophy.

CRITICAL INSTRUCTIONS:
1. ONLY use information explicitly stated in the provided context
2. DO NOT infer, assume, or extrapolate beyond what is directly stated
3. If the context mentions a topic but doesn't answer the specific question, say so clearly
4. If the context is insufficient, respond: "The provided context does not contain enough information to answer this question."

Question: {query}

Context from the document:
{context}{entity_info}{summary_info}

Based STRICTLY on the provided context above, answer the question. If the context only mentions a topic in passing without providing the requested information, acknowledge this limitation.

Answer:"""
        )

