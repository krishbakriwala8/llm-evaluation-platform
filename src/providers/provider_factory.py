"""Provider factory for creating LLM provider instances."""
from typing import Optional
from src.providers.base_provider import BaseLLMProvider
from src.providers.openai_provider import OpenAIProvider
from src.providers.claude_provider import ClaudeProvider
from src.utils.logger import logger


class ProviderFactory:
    """Factory for creating LLM provider instances."""

    _providers = {
        "openai": OpenAIProvider,
        "gpt-4": OpenAIProvider,
        "gpt-3.5": OpenAIProvider,
        "claude": ClaudeProvider,
        "claude-3": ClaudeProvider,
    }

    @classmethod
    def create(
        cls,
        provider_name: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
    ) -> BaseLLMProvider:
        """Create a provider instance."""
        provider_name = provider_name.lower()

        if provider_name not in cls._providers:
            raise ValueError(
                f"Unknown provider: {provider_name}. "
                f"Available: {list(cls._providers.keys())}"
            )

        provider_class = cls._providers[provider_name]
        logger.info(f"Creating {provider_name} provider")

        return provider_class(
            api_key=api_key, model=model, temperature=temperature
        )

    @classmethod
    def register(cls, name: str, provider_class: type) -> None:
        """Register a custom provider."""
        cls._providers[name.lower()] = provider_class
        logger.info(f"Registered custom provider: {name}")

    @classmethod
    def get_available_providers(cls) -> list[str]:
        """Get list of available providers."""
        return list(cls._providers.keys())
