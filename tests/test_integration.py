"""Integration tests for evaluation pipeline."""
import pytest
from unittest.mock import Mock, patch
from src.evaluators import EvaluationPipeline, EvaluationConfig
from src.datasets import QAPair


class TestEvaluationPipeline:
    """Tests for evaluation pipeline."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return EvaluationConfig(
            providers=[],  # No providers for unit test
            metrics=["latency"],
        )

    @pytest.fixture
    def pipeline(self, config):
        """Create test pipeline."""
        return EvaluationPipeline(config)

    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset."""
        return [
            QAPair(
                question="What is 2+2?",
                answer="4",
                context="Math problem.",
            ),
            QAPair(
                question="What is the capital of France?",
                answer="Paris",
                context="France is in Europe. Its capital is Paris.",
            ),
        ]

    def test_pipeline_initialization(self, pipeline):
        """Test pipeline initialization."""
        assert pipeline.config is not None
        assert isinstance(pipeline.providers, dict)
        assert isinstance(pipeline.metrics, dict)

    def test_metric_initialization(self, pipeline):
        """Test metric initialization."""
        assert "latency" in pipeline.metrics

    def test_summary_creation(self, pipeline, sample_dataset):
        """Test evaluation summary creation."""
        summary = pipeline._create_summary(
            results=[],
            total_samples=len(sample_dataset),
            failed_count=0,
        )
        assert summary.total_samples == len(sample_dataset)
        assert summary.failed_samples == 0

    def test_pipeline_with_no_providers(self, pipeline, sample_dataset):
        """Test pipeline evaluation with no providers."""
        # Should complete without error
        summary = pipeline.evaluate(sample_dataset)
        assert summary is not None

    def test_print_summary(self, pipeline, sample_dataset, capsys):
        """Test summary printing."""
        summary = pipeline._create_summary(
            results=[],
            total_samples=len(sample_dataset),
            failed_count=0,
        )
        pipeline.print_summary(summary)
        captured = capsys.readouterr()
        assert "EVALUATION SUMMARY" in captured.out
