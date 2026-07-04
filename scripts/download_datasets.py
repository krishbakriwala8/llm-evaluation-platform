"""Script to download and create benchmark datasets."""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datasets.datasets import create_benchmark_datasets
from src.utils.logger import logger


def main():
    """Download and create benchmark datasets."""
    logger.info("Creating benchmark datasets...")
    try:
        create_benchmark_datasets()
        logger.info("Benchmark datasets created successfully!")
    except Exception as e:
        logger.error(f"Error creating datasets: {e}")
        raise


if __name__ == "__main__":
    main()
