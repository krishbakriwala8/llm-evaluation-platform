"""Script to run benchmark evaluations and generate reports."""
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluators import EvaluationPipeline, EvaluationConfig
from src.datasets.datasets import create_benchmark_datasets
from src.utils.logger import logger
from src.utils.config import DATA_DIR


def generate_html_report(summary_file: Path, output_file: Optional[Path] = None) -> Path:
    """Generate HTML report from evaluation summary."""
    if output_file is None:
        output_file = summary_file.parent / "report.html"

    with open(summary_file, "r") as f:
        summary_data = json.load(f)

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>LLM Evaluation Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1, h2 {{
            color: #333;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background-color: #f9f9f9;
            border: 1px solid #ddd;
            padding: 15px;
            border-radius: 5px;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #0066cc;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #0066cc;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .timestamp {{
            color: #666;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>LLM Evaluation Report</h1>
        <p class="timestamp">Generated: {summary_data['timestamp']}</p>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <h3>Total Samples</h3>
                <div class="metric-value">{summary_data['total_samples']}</div>
            </div>
            <div class="metric-card">
                <h3>Failed Samples</h3>
                <div class="metric-value" style="color: #cc0000;">{summary_data['failed_samples']}</div>
            </div>
            <div class="metric-card">
                <h3>Total Latency</h3>
                <div class="metric-value">{summary_data['total_latency_ms']:.2f}ms</div>
            </div>
            <div class="metric-card">
                <h3>Total Tokens</h3>
                <div class="metric-value">{summary_data['total_tokens']}</div>
            </div>
        </div>

        <h2>Metrics by Provider</h2>
        <table>
            <tr>
                <th>Provider</th>
                <th>Metric</th>
                <th>Average</th>
                <th>Min</th>
                <th>Max</th>
                <th>Samples</th>
            </tr>
"""

    for provider in summary_data["providers"]:
        metrics = summary_data["metrics"].get(provider, {})
        for metric, stats in metrics.items():
            html += f"""
            <tr>
                <td>{provider}</td>
                <td>{metric}</td>
                <td>{stats['avg']:.4f}</td>
                <td>{stats['min']:.4f}</td>
                <td>{stats['max']:.4f}</td>
                <td>{stats['count']}</td>
            </tr>
"""

    html += """
        </table>
    </div>
</body>
</html>
"""

    with open(output_file, "w") as f:
        f.write(html)

    logger.info(f"HTML report generated: {output_file}")
    return output_file


def main(
    providers: list[str] = None,
    dataset: str = "finance_qa",
    max_samples: int = None,
    output_report: bool = True,
):
    """Run benchmark evaluation."""
    providers = providers or ["openai"]

    logger.info(f"Starting benchmark evaluation")
    logger.info(f"Providers: {providers}")
    logger.info(f"Dataset: {dataset}")
    logger.info(f"Max samples: {max_samples}")

    # Create datasets if they don't exist
    try:
        create_benchmark_datasets()
    except Exception as e:
        logger.warning(f"Could not create datasets: {e}")

    # Create evaluation config
    config = EvaluationConfig(
        providers=providers,
        dataset_name=dataset,
        max_samples=max_samples,
        metrics=[
            "faithfulness",
            "relevance",
            "hallucination",
            "latency",
            "semantic_similarity",
        ],
    )

    # Run evaluation
    pipeline = EvaluationPipeline(config)
    summary = pipeline.evaluate()

    if summary:
        pipeline.print_summary(summary)

        if output_report:
            results_dir = DATA_DIR / "results"
            latest_summary = max(
                results_dir.glob("summary_*.json"), key=lambda p: p.stat().st_mtime
            )
            generate_html_report(latest_summary)
    else:
        logger.error("Evaluation failed")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run LLM evaluation benchmark")
    parser.add_argument(
        "--providers",
        nargs="+",
        default=["openai"],
        help="LLM providers to evaluate",
    )
    parser.add_argument(
        "--dataset", default="finance_qa", help="Dataset to use for evaluation"
    )
    parser.add_argument(
        "--max-samples", type=int, help="Maximum number of samples to evaluate"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        default=True,
        help="Generate HTML report",
    )

    args = parser.parse_args()
    main(
        providers=args.providers,
        dataset=args.dataset,
        max_samples=args.max_samples,
        output_report=args.report,
    )
