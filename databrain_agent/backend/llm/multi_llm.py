"""Multi-LLM support with OpenAI and DeepSeek."""
from typing import Optional, Dict, Any
from langchain_openai import ChatOpenAI
import os
from databrain_agent.backend.llm.cost_tracker import CostTracker


class MultiLLM:
    """Router for multiple LLM providers."""
    
    def __init__(self, default_provider: str = "openai"):
        """Initialize LLM router."""
        self.default_provider = default_provider
        self.cost_tracker = CostTracker()
        self._llms = {}
        self._initialize_llms()
    
    def _initialize_llms(self):
        """Initialize available LLMs."""
        # OpenAI
        if os.getenv("OPENAI_API_KEY"):
            self._llms["openai"] = ChatOpenAI(
                model="gpt-4",
                temperature=0,
                api_key=os.getenv("OPENAI_API_KEY")
            )
            self._llms["openai-turbo"] = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0,
                api_key=os.getenv("OPENAI_API_KEY")
            )
        
        # DeepSeek (using OpenAI-compatible API)
        if os.getenv("DEEPSEEK_API_KEY"):
            self._llms["deepseek"] = ChatOpenAI(
                model="deepseek-chat",
                base_url="https://api.deepseek.com/v1",
                temperature=0,
                api_key=os.getenv("DEEPSEEK_API_KEY")
            )
    
    def get_llm(self, provider: Optional[str] = None) -> Any:
        """Get LLM instance for specified provider."""
        # Handle "Auto" or empty provider
        if not provider or provider.lower() == "auto":
            provider = None
        
        provider = provider or self.default_provider
        
        if provider not in self._llms:
            # Fallback logic for "Auto" mode
            # Try OpenAI first, then DeepSeek, then any available
            if "openai" in self._llms:
                provider = "openai"
            elif "openai-turbo" in self._llms:
                provider = "openai-turbo"
            elif "deepseek" in self._llms:
                provider = "deepseek"
            elif self._llms:
                provider = list(self._llms.keys())[0]
            else:
                raise ValueError("No LLM providers configured. Set OPENAI_API_KEY or DEEPSEEK_API_KEY")
        
        return self._llms[provider]
    
    def get_available_providers(self) -> list:
        """Get list of available LLM providers."""
        return list(self._llms.keys())
    
    def track_usage(self, provider: str, model: str, input_tokens: int, output_tokens: int):
        """Track token usage for cost calculation."""
        return self.cost_tracker.track_usage(provider, model, input_tokens, output_tokens)
