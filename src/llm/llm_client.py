"""LLM client using Ollama via LangChain."""

from langchain_community.llms import Ollama


class LLMClient:
    """LLM client for generating text."""
    
    def __init__(self,
                 model: str = "mistral:7b",
                 base_url: str = "http://localhost:11434",
                 temperature: float = 0.7,
                 max_tokens: int = 1000):
        """
        Initialize LLM client.
        
        Args:
            model: Ollama model name
            base_url: Ollama API base URL
            temperature: Generation temperature
            max_tokens: Maximum tokens in response
        """
        self.llm = Ollama(
            model=model,
            base_url=base_url,
            temperature=temperature,
            num_predict=max_tokens
        )
    
    def generate(self, prompt: str) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text
        """
        try:
            response = self.llm.invoke(prompt)
            return response.strip()
        except Exception as e:
            print(f"Error generating text: {e}")
            return ""

