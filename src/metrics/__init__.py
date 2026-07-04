"""Evaluation metrics for LLM responses."""
from abc import ABC, abstractmethod
from typing import Optional
from dataclasses import dataclass


@dataclass
class MetricScore:
    """Metric score result."""

    name: str
    score: float  # 0-1
    details: dict = None
    error: Optional[str] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}

    def is_valid(self) -> bool:
        """Check if score is valid."""
        return self.error is None and 0 <= self.score <= 1


class BaseMetric(ABC):
    """Base class for evaluation metrics."""

    def __init__(self, name: str, threshold: float = 0.7):
        self.name = name
        self.threshold = threshold

    @abstractmethod
    def compute(
        self,
        response: str,
        reference: Optional[str] = None,
        context: Optional[str] = None,
        **kwargs,
    ) -> MetricScore:
        """Compute metric score."""
        pass

    def passes_threshold(self, score: float) -> bool:
        """Check if score passes threshold."""
        return score >= self.threshold
