"""Anthropic Claude LLM Provider implementation."""
import time
from typing import Optional
from anthropic import Anthropic
from src.providers.base_provider import BaseLLMProvider, LLMResponse
from src.utils.config import settings
from src.utils.logger import logger


class ClaudeProvider(BaseLLMProvider):
    """Anthropic Claude LLM provider."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
    ):
        api_key = api_key or settings.anthropic_api_key
        model = model or settings.anthropic_model
        temperature = temperature or settings.anthropic_temperature

        if not api_key:
            raise ValueError("Anthropic API key not found in configuration")

        super().__init__(api_key, model, temperature)
        self.client = Anthropic(api_key=api_key)

    def generate(
        self,
        prompt: str,
        context: Optional[str] = None,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Generate response from Claude."""
        try:
            # Build the full prompt with context
            full_prompt = f"{context}\n\nQuestion: {prompt}" if context else prompt

            start_time = time.time()
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system="You are a helpful and accurate assistant. Provide concise and factual responses.",
                messages=[{"role": "user", "content": full_prompt}],
                temperature=self.temperature,
            )
            latency = (time.time() - start_time) * 1000

            content = response.content[0].text
            tokens_used = response.usage.input_tokens + response.usage.output_tokens
            self.total_tokens += tokens_used

            logger.debug(
                f"Claude response generated - Tokens: {tokens_used}, Latency: {latency:.2f}ms"
            )

            return LLMResponse(
                content=content,
                model=self.model,
                provider="claude",
                tokens_used=tokens_used,
                latency_ms=latency,
            )
        except Exception as e:
            logger.error(f"Claude generation error: {e}")
            raise

    async def generate_async(
        self,
        prompt: str,
        context: Optional[str] = None,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Asynchronously generate response from Claude."""
        # For now, run sync version in thread pool
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.generate, prompt, context, max_tokens
        )

    def get_provider_name(self) -> str:
        """Get provider name."""
        return "claude"
