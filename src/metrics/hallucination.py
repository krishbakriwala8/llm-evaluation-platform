"""Hallucination metric - detects hallucinated information in responses."""
from typing import Optional
from src.metrics import BaseMetric, MetricScore
from src.utils.logger import logger


class HallucinationMetric(BaseMetric):
    """Detects hallucinations in LLM responses."""

    def __init__(self):
        super().__init__("hallucination", threshold=0.1)  # Lower is better

    def compute(
        self,
        response: str,
        reference: Optional[str] = None,
        context: Optional[str] = None,
        **kwargs,
    ) -> MetricScore:
        """Compute hallucination score (0 = no hallucination, 1 = high hallucination)."""
        try:
            if not context:
                # Without context, we can't detect hallucinations
                return MetricScore(
                    name=self.name,
                    score=0.5,
                    details={"reason": "No context provided"},
                )

            # Compute hallucination score based on context coverage
            score = self._compute_hallucination_score(response, context)

            return MetricScore(
                name=self.name,
                score=score,
                details={"method": "context_coverage"},
            )

        except Exception as e:
            logger.error(f"Hallucination computation error: {e}")
            return MetricScore(name=self.name, score=0.0, error=str(e))

    def _compute_hallucination_score(self, response: str, context: str) -> float:
        """Compute hallucination score based on context overlap."""
        response_words = set(response.lower().split())
        context_words = set(context.lower().split())

        if not response_words:
            return 0.0

        # Words in response that are not in context
        novel_words = response_words - context_words

        # Stop words (common words that can appear anywhere)
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
            "to",
            "of",
            "in",
            "for",
            "on",
            "at",
            "by",
            "from",
        }

        novel_words = novel_words - stop_words

        # Hallucination score: ratio of novel words to total words
        score = len(novel_words) / len(response_words) if response_words else 0.0
        score = min(score, 1.0)  # Clamp to [0, 1]

        return score
