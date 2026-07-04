"""Tests for LLM providers."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.providers.provider_factory import ProviderFactory
from src.providers.openai_provider import OpenAIProvider
from src.providers.claude_provider import ClaudeProvider


class TestProviderFactory:
    """Tests for provider factory."""

    def test_get_available_providers(self):
        """Test getting available providers."""
        providers = ProviderFactory.get_available_providers()
        assert "openai" in providers
        assert "claude" in providers

    def test_invalid_provider(self):
        """Test creating invalid provider."""
        with pytest.raises(ValueError):
            ProviderFactory.create("invalid_provider")

    def test_provider_aliases(self):
        """Test provider name aliases."""
        # These should all work
        assert ProviderFactory.get_available_providers()
        assert "gpt-4" in ProviderFactory.get_available_providers()
        assert "gpt-3.5" in ProviderFactory.get_available_providers()


class TestOpenAIProvider:
    """Tests for OpenAI provider."""

    def test_provider_initialization_no_key(self):
        """Test provider initialization without API key."""
        with pytest.raises(ValueError):
            OpenAIProvider(api_key="")

    @patch('openai.OpenAI')
    def test_provider_name(self, mock_openai):
        """Test provider name."""
        provider = OpenAIProvider(api_key="test-key")
        assert provider.get_provider_name() == "openai"

    @patch('openai.OpenAI')
    def test_token_tracking(self, mock_openai):
        """Test token tracking."""
        provider = OpenAIProvider(api_key="test-key")
        assert provider.get_token_count() == 0
        provider.total_tokens = 100
        assert provider.get_token_count() == 100


class TestClaudeProvider:
    """Tests for Claude provider."""

    def test_provider_initialization_no_key(self):
        """Test provider initialization without API key."""
        with pytest.raises(ValueError):
            ClaudeProvider(api_key="")

    @patch('anthropic.Anthropic')
    def test_provider_name(self, mock_anthropic):
        """Test provider name."""
        provider = ClaudeProvider(api_key="test-key")
        assert provider.get_provider_name() == "claude"

    @patch('anthropic.Anthropic')
    def test_token_tracking(self, mock_anthropic):
        """Test token tracking."""
        provider = ClaudeProvider(api_key="test-key")
        assert provider.get_token_count() == 0
        provider.total_tokens = 50
        assert provider.get_token_count() == 50
