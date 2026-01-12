"""
LLM Handler Module
Handles Ollama integration for text generation.
"""

import requests
from typing import List, Dict, Generator


class OllamaLLM:
    """Ollama LLM client for local inference."""
    
    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434"
    ):
        """
        Initialize Ollama client.
        
        Args:
            model: Ollama model name
            base_url: Ollama server URL
        """
        self.model = model
        self.base_url = base_url
        self.generate_url = f"{base_url}/api/generate"
        self.chat_url = f"{base_url}/api/chat"
    
    def _build_prompt(self, query: str, context: List[str]) -> str:
        """
        Build RAG prompt with context.
        
        Args:
            query: User question
            context: Retrieved document chunks
            
        Returns:
            Formatted prompt
        """
        context_text = "\n\n".join([f"[{i+1}] {ctx}" for i, ctx in enumerate(context)])
        
        prompt = f"""You are a helpful AI assistant that answers questions based on provided documents.

PROVIDED DOCUMENTS:
{context_text}

USER QUESTION: {query}

INSTRUCTIONS:
1. Answer ONLY using information from the provided documents above
2. If the answer is not in the documents, clearly state "This information is not available in the provided documents"
3. Be clear, concise, and accurate
4. Cite which document section you used (e.g., "According to [1]...")
5. If multiple document sections support your answer, combine them coherently

ANSWER:"""
        
        return prompt
    
    def generate(
        self,
        query: str,
        context: List[str] = None,
        stream: bool = False
    ) -> str:
        """
        Generate response using Ollama.
        
        Args:
            query: User question
            context: Retrieved context chunks
            stream: Whether to stream response
            
        Returns:
            Generated response
        """
        # Build prompt
        if context:
            prompt = self._build_prompt(query, context)
        else:
            prompt = query
        
        # Prepare request
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_ctx": 2048
            }
        }
        
        try:
            response = requests.post(
                self.generate_url,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            if stream:
                return response  # Return raw response for streaming
            else:
                result = response.json()
                return result.get("response", "")
        
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Ollama bağlantı hatası: {str(e)}")
    
    def generate_stream(
        self,
        query: str,
        context: List[str] = None
    ) -> Generator[str, None, None]:
        """
        Generate streaming response.
        
        Args:
            query: User question
            context: Retrieved context chunks
            
        Yields:
            Response chunks
        """
        # Build prompt
        if context:
            prompt = self._build_prompt(query, context)
        else:
            prompt = query
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_ctx": 2048
            }
        }
        
        try:
            response = requests.post(
                self.generate_url,
                json=payload,
                stream=True,
                timeout=60
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    import json
                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"]
        
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Ollama bağlantı hatası: {str(e)}")
    
    def check_health(self) -> bool:
        """
        Check if Ollama server is running.
        
        Returns:
            True if server is accessible
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False


if __name__ == "__main__":
    # Test Ollama connection
    llm = OllamaLLM()
    print(f"Ollama Health Check: {llm.check_health()}")
    
    # Test generation
    if llm.check_health():
        response = llm.generate("Merhaba, nasılsın?")
        print(f"Response: {response}")
