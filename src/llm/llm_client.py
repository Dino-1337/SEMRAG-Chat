# llm_client.py
from langchain_community.llms import Ollama

class LLMClient:
    def __init__(self,
                 model: str = "mistral:7b",
                 base_url: str = "http://localhost:11434",
                 temperature: float = 0.4,
                 max_tokens: int = 900):
        self.llm = Ollama(
            model=model,
            base_url=base_url,
            temperature=temperature,
            num_predict=max_tokens
        )

    def generate(self, prompt: str) -> str:
        try:
            return self.llm.invoke(prompt).strip()
        except Exception as e:
            print(f"Generation error: {e}")
            return ""
