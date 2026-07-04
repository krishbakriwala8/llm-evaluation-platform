"""Main entry point for running evaluations."""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from src.evaluators import EvaluationPipeline, EvaluationConfig
from src.datasets.datasets import create_benchmark_datasets
from src.utils.logger import logger


def main():
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(
        description="LLM Evaluation & Benchmarking Platform"
    )
    parser.add_argument(
        "command",
        choices=["evaluate", "benchmark", "compare"],
        help="Command to run",
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        help="LLM providers to evaluate",
    )
    parser.add_argument(
        "--dataset",
        default="finance_qa",
        help="Dataset to use",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum samples to evaluate",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        help="Metrics to compute",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate report",
    )

    args = parser.parse_args()

    # Create datasets
    try:
        create_benchmark_datasets()
    except Exception as e:
        logger.warning(f"Could not create datasets: {e}")

    if args.command == "evaluate":
        providers = args.providers or ["openai"]
        metrics = args.metrics or [
            "faithfulness",
            "relevance",
            "hallucination",
            "latency",
        ]

        config = EvaluationConfig(
            providers=providers,
            dataset_name=args.dataset,
            max_samples=args.max_samples,
            metrics=metrics,
        )

        pipeline = EvaluationPipeline(config)
        summary = pipeline.evaluate()
        pipeline.print_summary(summary)

    elif args.command == "benchmark":
        from scripts.benchmark_suite import main as benchmark_main
        benchmark_main(
            providers=args.providers or ["openai"],
            dataset=args.dataset,
            max_samples=args.max_samples,
            output_report=args.report or True,
        )

    elif args.command == "compare":
        from scripts.compare_providers import ProviderComparator
        comparator = ProviderComparator()
        comparison = comparator.compare(
            providers=args.providers or ["openai", "claude"],
            dataset_name=args.dataset,
            max_samples=args.max_samples,
        )
        comparator.print_comparison(comparison)
        comparator.save_comparison(comparison)


if __name__ == "__main__":
    main()
