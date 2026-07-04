"""LLM Provider base classes and factory."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
from datetime import datetime


@dataclass
class LLMResponse:
    """Response from LLM provider."""

    content: str
    model: str
    provider: str
    tokens_used: int
    latency_ms: float
    timestamp: datetime = None
    metadata: dict = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}


class BaseLLMProvider(ABC):
    """Base class for LLM providers."""

    def __init__(self, api_key: str, model: str, temperature: float = 0.7):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.total_tokens = 0

    @abstractmethod
    def generate(
        self,
        prompt: str,
        context: Optional[str] = None,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Generate response from LLM."""
        pass

    @abstractmethod
    async def generate_async(
        self,
        prompt: str,
        context: Optional[str] = None,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Asynchronously generate response from LLM."""
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Get provider name."""
        pass

    def get_token_count(self) -> int:
        """Get total tokens used."""
        return self.total_tokens

    def reset_token_count(self) -> None:
        """Reset token counter."""
        self.total_tokens = 0
