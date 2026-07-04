"""Main evaluation pipeline for benchmarking LLMs."""
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import Optional, List
from datetime import datetime

from src.providers.provider_factory import ProviderFactory
from src.datasets import DatasetLoader, QAPair
from src.metrics import MetricScore
from src.metrics.faithfulness import FaithfulnessMetric
from src.metrics.relevance import AnswerRelevanceMetric
from src.metrics.hallucination import HallucinationMetric
from src.metrics.latency import LatencyMetric
from src.metrics.semantic_similarity import SemanticSimilarityMetric
from src.utils.config import EvaluationConfig, settings, DATA_DIR
from src.utils.logger import logger
from src.utils.cache import cache


@dataclass
class EvaluationResult:
    """Result of evaluating a single QA pair."""

    question: str
    provider: str
    model: str
    response: str
    reference_answer: str
    latency_ms: float
    tokens_used: int
    metrics: dict  # name -> MetricScore
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def get_metric_score(self, metric_name: str) -> Optional[float]:
        """Get score for a specific metric."""
        metric = self.metrics.get(metric_name)
        return metric.score if metric else None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "question": self.question,
            "provider": self.provider,
            "model": self.model,
            "response": self.response,
            "reference_answer": self.reference_answer,
            "latency_ms": self.latency_ms,
            "tokens_used": self.tokens_used,
            "metrics": {
                name: {"score": metric.score, "details": metric.details}
                for name, metric in self.metrics.items()
            },
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class EvaluationSummary:
    """Summary of evaluation results."""

    total_samples: int
    providers: List[str]
    metrics_results: dict  # provider -> metric -> list of scores
    failed_samples: int = 0
    total_latency_ms: float = 0.0
    total_tokens: int = 0
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def get_metric_average(self, provider: str, metric: str) -> float:
        """Get average score for a metric."""
        scores = self.metrics_results.get(provider, {}).get(metric, [])
        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    def get_metric_stats(self, provider: str, metric: str) -> dict:
        """Get statistics for a metric."""
        scores = self.metrics_results.get(provider, {}).get(metric, [])
        if not scores:
            return {"avg": 0, "min": 0, "max": 0, "count": 0}

        return {
            "avg": sum(scores) / len(scores),
            "min": min(scores),
            "max": max(scores),
            "count": len(scores),
        }

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        summary = {
            "timestamp": self.timestamp.isoformat(),
            "total_samples": self.total_samples,
            "failed_samples": self.failed_samples,
            "providers": self.providers,
            "total_latency_ms": self.total_latency_ms,
            "total_tokens": self.total_tokens,
            "metrics": {},
        }

        for provider in self.providers:
            summary["metrics"][provider] = {}
            for metric in self.metrics_results.get(provider, {}).keys():
                summary["metrics"][provider][metric] = self.get_metric_stats(
                    provider, metric
                )

        return summary


class EvaluationPipeline:
    """Main evaluation pipeline for benchmarking LLMs."""

    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config or EvaluationConfig()
        self.dataset_loader = DatasetLoader()
        self.providers = {}
        self.metrics = {}
        self._initialize_providers()
        self._initialize_metrics()

    def _initialize_providers(self):
        """Initialize LLM providers."""
        for provider_name in self.config.providers:
            try:
                provider = ProviderFactory.create(provider_name)
                self.providers[provider_name] = provider
                logger.info(f"Initialized provider: {provider_name}")
            except Exception as e:
                logger.error(f"Failed to initialize provider {provider_name}: {e}")

    def _initialize_metrics(self):
        """Initialize evaluation metrics."""
        metric_classes = {
            "faithfulness": FaithfulnessMetric,
            "relevance": AnswerRelevanceMetric,
            "hallucination": HallucinationMetric,
            "latency": LatencyMetric,
            "semantic_similarity": SemanticSimilarityMetric,
        }

        for metric_name in self.config.metrics:
            if metric_name in metric_classes:
                try:
                    metric = metric_classes[metric_name]()
                    self.metrics[metric_name] = metric
                    logger.info(f"Initialized metric: {metric_name}")
                except Exception as e:
                    logger.error(f"Failed to initialize metric {metric_name}: {e}")

    def evaluate(
        self, dataset: Optional[List[QAPair]] = None
    ) -> EvaluationSummary:
        """Run complete evaluation pipeline."""
        if dataset is None:
            dataset = self.dataset_loader.load(
                self.config.dataset_name, self.config.max_samples
            )

        if not dataset:
            logger.error("No dataset loaded")
            return None

        logger.info(
            f"Starting evaluation: {len(dataset)} samples, "
            f"{len(self.providers)} providers, {len(self.metrics)} metrics"
        )

        results = []
        failed_count = 0

        with ThreadPoolExecutor(max_workers=settings.max_workers) as executor:
            futures = {}
            for idx, qa_pair in enumerate(dataset):
                for provider_name, provider in self.providers.items():
                    future = executor.submit(
                        self._evaluate_single, qa_pair, provider_name, provider, idx
                    )
                    futures[future] = (idx, provider_name)

            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Evaluation error: {e}")
                    failed_count += 1

        # Aggregate results
        summary = self._create_summary(results, len(dataset), failed_count)
        self._save_results(results, summary)

        return summary

    def _evaluate_single(
        self, qa_pair: QAPair, provider_name: str, provider, idx: int
    ) -> Optional[EvaluationResult]:
        """Evaluate a single QA pair with a provider."""
        try:
            # Generate response
            llm_response = provider.generate(
                prompt=qa_pair.question,
                context=qa_pair.context,
            )

            # Compute metrics
            metrics_results = {}
            for metric_name, metric in self.metrics.items():
                try:
                    if metric_name == "latency":
                        score = metric.compute(
                            response=llm_response.content,
                            reference=qa_pair.answer,
                            context=qa_pair.context,
                            latency_ms=llm_response.latency_ms,
                        )
                    else:
                        score = metric.compute(
                            response=llm_response.content,
                            reference=qa_pair.answer,
                            context=qa_pair.context,
                            question=qa_pair.question,
                        )
                    metrics_results[metric_name] = score
                except Exception as e:
                    logger.error(f"Metric computation error ({metric_name}): {e}")
                    metrics_results[metric_name] = MetricScore(
                        name=metric_name, score=0.0, error=str(e)
                    )

            result = EvaluationResult(
                question=qa_pair.question,
                provider=provider_name,
                model=provider.model,
                response=llm_response.content,
                reference_answer=qa_pair.answer,
                latency_ms=llm_response.latency_ms,
                tokens_used=llm_response.tokens_used,
                metrics=metrics_results,
            )

            logger.debug(
                f"Evaluated Q{idx + 1} with {provider_name}: "
                f"Latency={llm_response.latency_ms:.2f}ms, "
                f"Tokens={llm_response.tokens_used}"
            )

            return result

        except Exception as e:
            logger.error(f"Error evaluating Q{idx + 1} with {provider_name}: {e}")
            return None

    def _create_summary(
        self, results: List[EvaluationResult], total_samples: int, failed_count: int
    ) -> EvaluationSummary:
        """Create evaluation summary from results."""
        metrics_results = {}

        for provider_name in self.providers.keys():
            metrics_results[provider_name] = {}
            for metric_name in self.metrics.keys():
                metrics_results[provider_name][metric_name] = []

        total_latency = 0.0
        total_tokens = 0

        for result in results:
            for metric_name, metric_score in result.metrics.items():
                if metric_score.is_valid():
                    metrics_results[result.provider][metric_name].append(
                        metric_score.score
                    )

            total_latency += result.latency_ms
            total_tokens += result.tokens_used

        summary = EvaluationSummary(
            total_samples=total_samples,
            providers=list(self.providers.keys()),
            metrics_results=metrics_results,
            failed_samples=failed_count,
            total_latency_ms=total_latency,
            total_tokens=total_tokens,
        )

        return summary

    def _save_results(
        self, results: List[EvaluationResult], summary: EvaluationSummary
    ):
        """Save evaluation results to files."""
        results_dir = DATA_DIR / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed results
        results_file = results_dir / f"results_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
        logger.info(f"Saved detailed results: {results_file}")

        # Save summary
        summary_file = results_dir / f"summary_{timestamp}.json"
        with open(summary_file, "w") as f:
            json.dump(summary.to_dict(), f, indent=2)
        logger.info(f"Saved summary: {summary_file}")

        return results_file, summary_file

    def print_summary(self, summary: EvaluationSummary):
        """Print evaluation summary."""
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        print(f"Timestamp: {summary.timestamp.isoformat()}")
        print(f"Total Samples: {summary.total_samples}")
        print(f"Failed Samples: {summary.failed_samples}")
        print(f"Total Latency: {summary.total_latency_ms:.2f}ms")
        print(f"Total Tokens: {summary.total_tokens}")
        print("\nMetrics by Provider:")
        print("-" * 80)

        for provider in summary.providers:
            print(f"\n{provider.upper()}:")
            for metric in self.config.metrics:
                stats = summary.get_metric_stats(provider, metric)
                if stats["count"] > 0:
                    print(
                        f"  {metric:20s}: avg={stats['avg']:.4f}, "
                        f"min={stats['min']:.4f}, max={stats['max']:.4f}"
                    )

        print("\n" + "=" * 80)
