"""Faithfulness metric - measures if response is grounded in context."""
from typing import Optional
from sentence_transformers import CrossEncoder
from src.metrics import BaseMetric, MetricScore
from src.utils.logger import logger


class FaithfulnessMetric(BaseMetric):
    """Evaluates if response is faithful to the provided context."""

    def __init__(self):
        super().__init__("faithfulness", threshold=0.7)
        try:
            # Use cross-encoder for entailment detection
            self.model = CrossEncoder(
                "cross-encoder/qnli-distilroberta-base", max_length=512
            )
        except Exception as e:
            logger.warning(f"Failed to load faithfulness model: {e}")
            self.model = None

    def compute(
        self,
        response: str,
        reference: Optional[str] = None,
        context: Optional[str] = None,
        **kwargs,
    ) -> MetricScore:
        """Compute faithfulness score."""
        try:
            if not context:
                return MetricScore(
                    name=self.name,
                    score=0.5,
                    details={"reason": "No context provided"},
                )

            if not self.model:
                # Fallback: basic string overlap
                return self._compute_fallback(response, context)

            # Use cross-encoder for entailment
            try:
                scores = self.model.predict([[context, response]])
                # Convert to 0-1 range
                score = float((scores[0] + 1) / 2)  # Normalize from [-1, 1] to [0, 1]
                score = max(0, min(1, score))  # Clamp to [0, 1]

                return MetricScore(
                    name=self.name,
                    score=score,
                    details={"model": "cross-encoder/qnli-distilroberta-base"},
                )
            except Exception as e:
                logger.warning(f"Cross-encoder prediction error: {e}")
                return self._compute_fallback(response, context)

        except Exception as e:
            logger.error(f"Faithfulness computation error: {e}")
            return MetricScore(name=self.name, score=0.0, error=str(e))

    def _compute_fallback(self, response: str, context: str) -> MetricScore:
        """Fallback method using simple overlap."""
        # Split into words and compute overlap
        response_words = set(response.lower().split())
        context_words = set(context.lower().split())

        if not context_words:
            return MetricScore(name=self.name, score=0.5)

        overlap = len(response_words & context_words)
        score = min(overlap / len(context_words), 1.0)

        return MetricScore(
            name=self.name,
            score=score,
            details={"method": "word_overlap", "overlap_ratio": score},
        )
