"""Dataset loading and management."""
import json
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass
from src.utils.logger import logger
from src.utils.config import DATA_DIR


@dataclass
class QAPair:
    """Question-Answer pair with context."""

    question: str
    answer: str
    context: str
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DatasetLoader:
    """Load and manage evaluation datasets."""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or DATA_DIR / "datasets"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load(self, dataset_name: str, max_samples: Optional[int] = None) -> List[QAPair]:
        """Load a dataset by name."""
        dataset_path = self.data_dir / f"{dataset_name}.json"

        if not dataset_path.exists():
            logger.warning(f"Dataset not found: {dataset_path}")
            return []

        try:
            with open(dataset_path, "r") as f:
                data = json.load(f)

            pairs = []
            for item in data:
                pair = QAPair(
                    question=item.get("question", ""),
                    answer=item.get("answer", ""),
                    context=item.get("context", ""),
                    metadata=item.get("metadata", {}),
                )
                pairs.append(pair)

            if max_samples:
                pairs = pairs[:max_samples]

            logger.info(f"Loaded {len(pairs)} samples from {dataset_name}")
            return pairs

        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name}: {e}")
            return []

    def save(self, dataset_name: str, pairs: List[QAPair]) -> bool:
        """Save a dataset."""
        dataset_path = self.data_dir / f"{dataset_name}.json"

        try:
            data = [
                {
                    "question": pair.question,
                    "answer": pair.answer,
                    "context": pair.context,
                    "metadata": pair.metadata,
                }
                for pair in pairs
            ]

            with open(dataset_path, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved {len(pairs)} samples to {dataset_name}")
            return True

        except Exception as e:
            logger.error(f"Error saving dataset {dataset_name}: {e}")
            return False

    def list_datasets(self) -> List[str]:
        """List available datasets."""
        datasets = [f.stem for f in self.data_dir.glob("*.json")]
        logger.info(f"Found {len(datasets)} datasets")
        return datasets
