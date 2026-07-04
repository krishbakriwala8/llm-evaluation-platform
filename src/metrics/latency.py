"""Latency metric - measures response generation time."""
from typing import Optional
from src.metrics import BaseMetric, MetricScore
from src.utils.logger import logger


class LatencyMetric(BaseMetric):
    """Measures response generation latency."""

    def __init__(self, threshold_ms: float = 5000):
        super().__init__("latency", threshold=0.0)  # Not used for latency
        self.threshold_ms = threshold_ms

    def compute(
        self,
        response: str,
        reference: Optional[str] = None,
        context: Optional[str] = None,
        **kwargs,
    ) -> MetricScore:
        """Compute latency score."""
        try:
            latency_ms = kwargs.get("latency_ms", 0)

            if not isinstance(latency_ms, (int, float)) or latency_ms < 0:
                return MetricScore(
                    name=self.name,
                    score=0.0,
                    error="Invalid latency value",
                )

            # Normalize latency to 0-1 scale (lower is better)
            # 0-1000ms = 1.0 (excellent)
            # 5000ms+ = 0.0 (poor)
            normalized_score = max(0, 1.0 - (latency_ms / self.threshold_ms))

            return MetricScore(
                name=self.name,
                score=normalized_score,
                details={
                    "latency_ms": latency_ms,
                    "threshold_ms": self.threshold_ms,
                },
            )

        except Exception as e:
            logger.error(f"Latency computation error: {e}")
            return MetricScore(name=self.name, score=0.0, error=str(e))
