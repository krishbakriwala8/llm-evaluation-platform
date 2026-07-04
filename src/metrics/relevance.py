"""Answer Relevance metric - measures if response answers the question."""
from typing import Optional
from sentence_transformers import CrossEncoder
from src.metrics import BaseMetric, MetricScore
from src.utils.logger import logger


class AnswerRelevanceMetric(BaseMetric):
    """Evaluates if response is relevant to the question."""

    def __init__(self):
        super().__init__("relevance", threshold=0.7)
        try:
            # Use cross-encoder for relevance detection
            self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception as e:
            logger.warning(f"Failed to load relevance model: {e}")
            self.model = None

    def compute(
        self,
        response: str,
        reference: Optional[str] = None,
        context: Optional[str] = None,
        **kwargs,
    ) -> MetricScore:
        """Compute answer relevance score."""
        try:
            # Use question from kwargs or context
            question = kwargs.get("question", "")
            if not question and context:
                question = context.split("Question:")[-1].strip()

            if not question or not response:
                return MetricScore(
                    name=self.name,
                    score=0.5,
                    details={"reason": "Missing question or response"},
                )

            if not self.model:
                return self._compute_fallback(question, response)

            try:
                # Use cross-encoder for relevance scoring
                scores = self.model.predict([[question, response]])
                # Normalize sigmoid output to 0-1
                score = 1 / (1 + (1.0 / scores[0] - 1))
                score = max(0, min(1, score))  # Clamp

                return MetricScore(
                    name=self.name,
                    score=score,
                    details={"model": "cross-encoder/ms-marco-MiniLM-L-6-v2"},
                )
            except Exception as e:
                logger.warning(f"Cross-encoder prediction error: {e}")
                return self._compute_fallback(question, response)

        except Exception as e:
            logger.error(f"Relevance computation error: {e}")
            return MetricScore(name=self.name, score=0.0, error=str(e))

    def _compute_fallback(self, question: str, response: str) -> MetricScore:
        """Fallback method using keyword matching."""
        question_words = set(question.lower().split())
        response_words = set(response.lower().split())

        if not question_words:
            return MetricScore(name=self.name, score=0.5)

        # Remove common stop words
        stop_words = {"the", "a", "an", "and", "or", "is", "are", "was"}
        question_words = question_words - stop_words
        response_words = response_words - stop_words

        if not question_words:
            return MetricScore(name=self.name, score=0.5)

        overlap = len(question_words & response_words)
        score = min(overlap / len(question_words), 1.0)

        return MetricScore(
            name=self.name,
            score=score,
            details={"method": "keyword_matching", "overlap_ratio": score},
        )
