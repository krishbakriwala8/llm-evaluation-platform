"""Script to compare different LLM providers."""
import sys
import json
from pathlib import Path
from typing import Optional, List
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluators import EvaluationPipeline, EvaluationConfig
from src.utils.logger import logger
from src.utils.config import DATA_DIR


class ProviderComparator:
    """Compare performance of different LLM providers."""

    def __init__(self):
        self.results = {}

    def compare(
        self,
        providers: List[str],
        dataset_name: str = "finance_qa",
        max_samples: Optional[int] = None,
    ) -> dict:
        """Compare providers on a dataset."""
        logger.info(f"Comparing providers: {providers}")

        config = EvaluationConfig(
            providers=providers,
            dataset_name=dataset_name,
            max_samples=max_samples,
            metrics=[
                "faithfulness",
                "relevance",
                "hallucination",
                "latency",
            ],
        )

        pipeline = EvaluationPipeline(config)
        summary = pipeline.evaluate()

        return self._format_comparison(summary, providers)

    def _format_comparison(self, summary, providers: List[str]) -> dict:
        """Format comparison results."""
        comparison = {
            "timestamp": summary.timestamp.isoformat(),
            "providers": {},
            "winner": {},  # Best provider for each metric
        }

        for provider in providers:
            comparison["providers"][provider] = {}
            for metric in ["faithfulness", "relevance", "hallucination", "latency"]:
                stats = summary.get_metric_stats(provider, metric)
                comparison["providers"][provider][metric] = {
                    "avg": round(stats["avg"], 4),
                    "min": round(stats["min"], 4),
                    "max": round(stats["max"], 4),
                }

        # Determine winners
        for metric in ["faithfulness", "relevance", "halluci nation", "latency"]:
            best_provider = None
            best_value = None

            for provider in providers:
                value = comparison["providers"][provider].get(metric, {})
                if not value:
                    continue

                avg = value["avg"]

                # For hallucination and latency, lower is better
                if metric in ["hallucination", "latency"]:
                    if best_value is None or avg < best_value:
                        best_provider = provider
                        best_value = avg
                else:
                    # For faithfulness and relevance, higher is better
                    if best_value is None or avg > best_value:
                        best_provider = provider
                        best_value = avg

            if best_provider:
                comparison["winner"][metric] = {
                    "provider": best_provider,
                    "score": best_value,
                }

        return comparison

    def print_comparison(self, comparison: dict):
        """Print formatted comparison."""
        print("\n" + "=" * 100)
        print("PROVIDER COMPARISON REPORT")
        print("=" * 100)
        print(f"Timestamp: {comparison['timestamp']}\n")

        # Print detailed metrics
        for provider, metrics in comparison["providers"].items():
            print(f"\n{provider.upper()}:")
            print("-" * 50)
            for metric, stats in metrics.items():
                print(
                    f"  {metric:20s}: avg={stats['avg']:7.4f}, "
                    f"min={stats['min']:7.4f}, max={stats['max']:7.4f}"
                )

        # Print winners
        print("\n" + "=" * 100)
        print("WINNERS:")
        print("-" * 50)
        for metric, winner in comparison["winner"].items():
            print(f"  {metric:20s}: {winner['provider']:15s} (score: {winner['score']:.4f})")
        print("=" * 100 + "\n")

    def save_comparison(self, comparison: dict, output_file: Optional[Path] = None):
        """Save comparison to file."""
        if output_file is None:
            output_file = (
                DATA_DIR / "results" / "comparison_latest.json"
            )

        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(comparison, f, indent=2)

        logger.info(f"Comparison saved: {output_file}")


def main():
    """Run provider comparison."""
    import argparse

    parser = argparse.ArgumentParser(description="Compare LLM providers")
    parser.add_argument(
        "--providers",
        nargs="+",
        default=["openai", "claude"],
        help="Providers to compare",
    )
    parser.add_argument(
        "--dataset", default="finance_qa", help="Dataset to use"
    )
    parser.add_argument(
        "--max-samples", type=int, help="Maximum samples to evaluate"
    )
    parser.add_argument(
        "--output", help="Output file for comparison"
    )

    args = parser.parse_args()

    comparator = ProviderComparator()
    comparison = comparator.compare(
        providers=args.providers,
        dataset_name=args.dataset,
        max_samples=args.max_samples,
    )

    comparator.print_comparison(comparison)
    comparator.save_comparison(comparison, Path(args.output) if args.output else None)


if __name__ == "__main__":
    main()
