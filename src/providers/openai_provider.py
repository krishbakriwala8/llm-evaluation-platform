"""OpenAI LLM Provider implementation."""
import time
from typing import Optional
import openai
from src.providers.base_provider import BaseLLMProvider, LLMResponse
from src.utils.config import settings
from src.utils.logger import logger


class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT-based LLM provider."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
    ):
        api_key = api_key or settings.openai_api_key
        model = model or settings.openai_model
        temperature = temperature or settings.openai_temperature

        if not api_key:
            raise ValueError("OpenAI API key not found in configuration")

        super().__init__(api_key, model, temperature)
        openai.api_key = api_key
        self.client = openai.OpenAI(api_key=api_key)

    def generate(
        self,
        prompt: str,
        context: Optional[str] = None,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Generate response from OpenAI."""
        try:
            # Build the full prompt with context
            full_prompt = f"{context}\n\nQuestion: {prompt}" if context else prompt

            start_time = time.time()
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful and accurate assistant. Provide concise and factual responses.",
                    },
                    {"role": "user", "content": full_prompt},
                ],
                temperature=self.temperature,
                max_tokens=max_tokens,
            )
            latency = (time.time() - start_time) * 1000

            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            self.total_tokens += tokens_used

            logger.debug(
                f"OpenAI response generated - Tokens: {tokens_used}, Latency: {latency:.2f}ms"
            )

            return LLMResponse(
                content=content,
                model=self.model,
                provider="openai",
                tokens_used=tokens_used,
                latency_ms=latency,
            )
        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            raise

    async def generate_async(
        self,
        prompt: str,
        context: Optional[str] = None,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Asynchronously generate response from OpenAI."""
        # For now, run sync version in thread pool
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.generate, prompt, context, max_tokens
        )

    def get_provider_name(self) -> str:
        """Get provider name."""
        return "openai"
