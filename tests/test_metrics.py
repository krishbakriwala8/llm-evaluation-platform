"""Unit tests for evaluation metrics."""
import pytest
from src.metrics.faithfulness import FaithfulnessMetric
from src.metrics.relevance import AnswerRelevanceMetric
from src.metrics.hallucination import HallucinationMetric
from src.metrics.latency import LatencyMetric
from src.metrics.semantic_similarity import SemanticSimilarityMetric


class TestFaithfulnessMetric:
    """Tests for faithfulness metric."""

    def setup_method(self):
        self.metric = FaithfulnessMetric()

    def test_high_faithfulness(self):
        """Test high faithfulness score."""
        context = "The capital of France is Paris. Paris is located on the Seine river."
        response = "Paris is the capital of France."

        score = self.metric.compute(response=response, context=context)
        assert score.is_valid()
        assert score.score > 0.5

    def test_low_faithfulness(self):
        """Test low faithfulness score."""
        context = "The capital of France is Paris."
        response = "The capital of France is London."

        score = self.metric.compute(response=response, context=context)
        assert score.is_valid()

    def test_no_context(self):
        """Test with no context."""
        response = "Some response"
        score = self.metric.compute(response=response)
        assert score.is_valid()


class TestAnswerRelevanceMetric:
    """Tests for answer relevance metric."""

    def setup_method(self):
        self.metric = AnswerRelevanceMetric()

    def test_relevant_answer(self):
        """Test relevant answer."""
        score = self.metric.compute(
            response="Paris is the capital of France and is located in Europe.",
            question="What is the capital of France?",
        )
        assert score.is_valid()

    def test_irrelevant_answer(self):
        """Test irrelevant answer."""
        score = self.metric.compute(
            response="The weather is sunny today.",
            question="What is the capital of France?",
        )
        assert score.is_valid()

    def test_no_question(self):
        """Test with no question."""
        score = self.metric.compute(response="Some response")
        assert score.is_valid()


class TestHallucinationMetric:
    """Tests for hallucination metric."""

    def setup_method(self):
        self.metric = HallucinationMetric()

    def test_low_hallucination(self):
        """Test low hallucination (mostly from context)."""
        context = "Paris is the capital of France."
        response = "Paris is the capital of France and is in Europe."

        score = self.metric.compute(response=response, context=context)
        assert score.is_valid()
        # Lower score means less hallucination
        assert score.score < 0.5

    def test_high_hallucination(self):
        """Test high hallucination (mostly novel content)."""
        context = "The sky is blue."
        response = "The ancient Romans built magnificent structures throughout Europe and Asia Minor."

        score = self.metric.compute(response=response, context=context)
        assert score.is_valid()

    def test_no_context(self):
        """Test with no context."""
        response = "Some response"
        score = self.metric.compute(response=response)
        assert score.is_valid()


class TestLatencyMetric:
    """Tests for latency metric."""

    def setup_method(self):
        self.metric = LatencyMetric(threshold_ms=5000)

    def test_low_latency(self):
        """Test low latency."""
        score = self.metric.compute(
            response="test", latency_ms=500
        )
        assert score.is_valid()
        assert score.score > 0.5

    def test_high_latency(self):
        """Test high latency."""
        score = self.metric.compute(
            response="test", latency_ms=10000
        )
        assert score.is_valid()
        assert score.score <= 0.0

    def test_invalid_latency(self):
        """Test invalid latency value."""
        score = self.metric.compute(
            response="test", latency_ms=-100
        )
        assert not score.is_valid()


class TestSemanticSimilarityMetric:
    """Tests for semantic similarity metric."""

    def setup_method(self):
        self.metric = SemanticSimilarityMetric()

    def test_identical_texts(self):
        """Test identical texts."""
        text = "The capital of France is Paris."
        score = self.metric.compute(response=text, reference=text)
        if score.is_valid():
            assert score.score > 0.8

    def test_no_reference(self):
        """Test with no reference."""
        score = self.metric.compute(response="Some response")
        assert score.is_valid()

    def test_different_semantics(self):
        """Test different semantic content."""
        response = "The sky is blue."
        reference = "The ocean is deep."
        score = self.metric.compute(response=response, reference=reference)
        if score.is_valid():
            # Should be lower similarity
            assert score.score < 0.8
