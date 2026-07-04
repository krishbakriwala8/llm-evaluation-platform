"""Tests for dataset loading."""
import pytest
import json
from pathlib import Path
from src.datasets import DatasetLoader, QAPair
from src.utils.config import DATA_DIR


class TestDatasetLoader:
    """Tests for dataset loader."""

    @pytest.fixture
    def loader(self):
        """Create dataset loader."""
        return DatasetLoader()

    @pytest.fixture
    def sample_dataset(self, tmp_path):
        """Create sample dataset."""
        data = [
            {
                "question": "What is AI?",
                "answer": "Artificial Intelligence is...",
                "context": "AI context...",
                "metadata": {"category": "tech"},
            }
        ]
        dataset_file = tmp_path / "test_qa.json"
        with open(dataset_file, "w") as f:
            json.dump(data, f)
        return dataset_file

    def test_qa_pair_creation(self):
        """Test QA pair creation."""
        pair = QAPair(
            question="What is AI?",
            answer="AI is...",
            context="Context...",
        )
        assert pair.question == "What is AI?"
        assert pair.metadata == {}

    def test_list_available_datasets(self, loader):
        """Test listing available datasets."""
        datasets = loader.list_datasets()
        assert isinstance(datasets, list)

    def test_save_and_load(self, loader, tmp_path):
        """Test saving and loading datasets."""
        loader.data_dir = tmp_path
        pairs = [
            QAPair(
                question="Q1",
                answer="A1",
                context="C1",
            )
        ]

        # Save
        loader.save("test", pairs)
        assert (tmp_path / "test.json").exists()

        # Load
        loaded = loader.load("test")
        assert len(loaded) == 1
        assert loaded[0].question == "Q1"

    def test_max_samples(self, loader):
        """Test loading with max samples."""
        datasets = loader.list_datasets()
        if datasets:
            loaded = loader.load(datasets[0], max_samples=1)
            assert len(loaded) <= 1
