try:
    from langchain_core.prompts import PromptTemplate
except ImportError:
    from langchain.prompts import PromptTemplate


class PromptTemplates:

    @staticmethod
    def get_answer_template():
        return PromptTemplate(
            input_variables=["query", "context"],
            template="""
You are a scholarly assistant analyzing Dr. B.R. Ambedkar's writings.

Your task: Answer the question using ONLY the evidence provided below.

INSTRUCTIONS:
1. Read all the evidence chunks carefully
2. If the evidence contains relevant information, synthesize it into a clear, structured answer
3. Combine information from multiple chunks when they relate to the same topic
4. Use direct quotes when appropriate to support your explanation
5. If the evidence is incomplete or unclear, acknowledge this but still provide what information IS available
6. ONLY say "The provided context does not contain enough information" if the evidence is completely unrelated to the question

STYLE:
- Write in clear, academic prose
- Be objective and analytical
- Structure longer answers with logical flow
- Cite specific details from the evidence

----------------------------------------
QUESTION:
{query}
----------------------------------------

EVIDENCE (from Ambedkar's writings):
{context}
----------------------------------------

ANSWER (synthesize the evidence above):
"""
        )
