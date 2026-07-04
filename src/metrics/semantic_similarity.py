"""Semantic similarity metric - measures semantic overlap between responses."""
from typing import Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from src.metrics import BaseMetric, MetricScore
from src.utils.logger import logger


class SemanticSimilarityMetric(BaseMetric):
    """Measures semantic similarity between response and reference."""

    def __init__(self):
        super().__init__("semantic_similarity", threshold=0.7)
        try:
            self.model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2"
            )
        except Exception as e:
            logger.warning(f"Failed to load semantic similarity model: {e}")
            self.model = None

    def compute(
        self,
        response: str,
        reference: Optional[str] = None,
        context: Optional[str] = None,
        **kwargs,
    ) -> MetricScore:
        """Compute semantic similarity score."""
        try:
            if not reference:
                return MetricScore(
                    name=self.name,
                    score=0.5,
                    details={"reason": "No reference provided"},
                )

            if not self.model:
                return MetricScore(
                    name=self.name,
                    score=0.5,
                    details={"reason": "Model not available"},
                )

            # Compute embeddings
            response_embedding = self.model.encode(response)
            reference_embedding = self.model.encode(reference)

            # Compute cosine similarity
            similarity = cosine_similarity(
                [response_embedding], [reference_embedding]
            )[0][0]

            # Clamp to [0, 1]
            score = max(0, min(1, float(similarity)))

            return MetricScore(
                name=self.name,
                score=score,
                details={"model": "sentence-transformers/all-MiniLM-L6-v2"},
            )

        except Exception as e:
            logger.error(f"Semantic similarity computation error: {e}")
            return MetricScore(name=self.name, score=0.0, error=str(e))
