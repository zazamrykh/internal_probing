"""
Base dataset classes and interfaces for internal probing experiments.

Provides unified interface for working with QA datasets and enriching them
with model generations, activations, and uncertainty scores.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Union, Optional
from typing_extensions import TypedDict, NotRequired
import pickle
import json

import pandas as pd
import torch


class DatasetSample(TypedDict):
    """
    Standard sample format for internal probing datasets.

    All datasets internally use dict format with these fields.
    Additional fields can be added through enrichment.
    """
    # Required base fields
    prompt: str
    gt_answers: list[str]

    # Optional enriched fields (added by enrichers). Here is examples from semantic entropy probes method.
    greedy_answer: NotRequired[str]
    is_correct: NotRequired[float]
    activations: NotRequired[dict]
    sampling_answers: NotRequired[list[str]]
    se_raw: NotRequired[float]
    se_binary: NotRequired[int]
    se_gamma: NotRequired[float]
    se_weight: NotRequired[float]
    semantic_ids: NotRequired[list[int]]


class BaseDataset(ABC):
    """
    Base class for all datasets in internal probing experiments.

    Provides unified interface for:
    - Loading data from various sources (list[dict], DataFrame, files)
    - Accessing samples in standard format
    - Evaluating correctness of generated answers
    - Saving/loading enriched datasets

    All datasets internally store data as list[dict] for consistency.
    """

    def __init__(
        self,
        data: Union[list[dict], pd.DataFrame, Path, str],
        name: Optional[str] = None
    ):
        """
        Initialize dataset from various sources.

        Args:
            data: Dataset source - can be:
                  - list[dict]: Direct data
                  - pd.DataFrame: Pandas dataframe
                  - Path/str: Path to saved dataset (pickle/json)
            name: Optional dataset name for identification
        """
        self.name = name or self.__class__.__name__
        self._data: list[dict] = self._normalize_data(data)

    def _normalize_data(self, data: Union[list[dict], pd.DataFrame, Path, str]) -> list[dict]:
        """
        Convert any input format to standard list[dict] format.

        Args:
            data: Input data in various formats

        Returns:
            Normalized list of dicts
        """
        if isinstance(data, (Path, str)):
            path = Path(data)
            if path.suffix == '.pkl' or path.suffix == '.pickle' or path.suffix == '.pt':
                with open(path, 'rb') as f:
                    loaded = pickle.load(f)
                    if isinstance(loaded, list):
                        return loaded
                    elif isinstance(loaded, pd.DataFrame):
                        return loaded.to_dict('records')
                    else:
                        raise ValueError(f"Unsupported pickle content type: {type(loaded)}")
            elif path.suffix == '.json':
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")

        elif isinstance(data, pd.DataFrame):
            return data.to_dict('records')

        elif isinstance(data, list):
            return data

        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self._data)

    def __getitem__(self, idx: int) -> dict:
        """
        Get sample by index.

        Args:
            idx: Sample index

        Returns:
            Sample dict with all available fields
        """
        return self._data[idx]

    def __iter__(self):
        """Iterate over dataset samples."""
        return iter(self._data)

    @abstractmethod
    def evaluate_correctness(
        self,
        prompt: str,
        generated_answer: str,
        gt_answers: list[str]
    ) -> float:
        """
        Evaluate correctness of generated answer.

        This method must be implemented by subclasses to define
        dataset-specific correctness evaluation logic.

        Args:
            prompt: Original prompt/question
            generated_answer: Model's generated answer
            gt_answers: List of ground truth answers

        Returns:
            Correctness score in [0, 1]:
            - 1.0 = definitely correct
            - 0.0 = definitely incorrect
            - 0.5 = uncertain
        """
        pass

    def to_dataframe(self) -> pd.DataFrame:
        """
        Export dataset to pandas DataFrame.

        Returns:
            DataFrame with all samples and fields
        """
        return pd.DataFrame(self._data)

    def save(self, path: Union[Path, str], format: str = "pickle"):
        """
        Save dataset to file.

        Saves both data and class information for proper reconstruction.

        Args:
            path: Output file path
            format: Save format - "pickle", "json", or "torch"
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format == "pickle" or format == "pkl":
            # Save both data and class info
            save_dict = {
                'data': self._data,
                'class': self.__class__,
                'name': self.name
            }
            with open(path, 'wb') as f:
                pickle.dump(save_dict, f)

        elif format == "torch" or format == "pt":
            save_dict = {
                'data': self._data,
                'class': self.__class__,
                'name': self.name
            }
            torch.save(save_dict, path)

        elif format == "json":
            # Convert any torch tensors to lists for JSON serialization
            json_data = []
            for sample in self._data:
                json_sample = {}
                for key, value in sample.items():
                    if isinstance(value, torch.Tensor):
                        json_sample[key] = value.tolist()
                    else:
                        json_sample[key] = value
                json_data.append(json_sample)

            with open(path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)

        else:
            raise ValueError(f"Unsupported format: {format}")

    @classmethod
    def load(cls, path: Union[Path, str], **kwargs) -> "BaseDataset":
        """
        Load dataset from file.

        Automatically reconstructs the correct dataset class.

        Args:
            path: Path to saved dataset
            **kwargs: Additional arguments for dataset constructor

        Returns:
            Loaded dataset instance
        """
        path = Path(path)

        # Load the file
        if path.suffix in ['.pkl', '.pickle']:
            with open(path, 'rb') as f:
                loaded = pickle.load(f)
        elif path.suffix in ['.pt']:
            loaded = torch.load(path)
        else:
            # Fallback to old behavior for JSON
            return cls(data=path, **kwargs)

        # Check if it's new format (dict with class info) or old format (just list)
        if isinstance(loaded, dict) and 'class' in loaded and 'data' in loaded:
            # New format: use saved class
            dataset_class = loaded['class']
            data = loaded['data']
            name = loaded.get('name')
            return dataset_class(data=data, name=name, **kwargs)
        else:
            # Old format: use cls (may fail for BaseDataset)
            return cls(data=loaded if isinstance(loaded, list) else path, **kwargs)

    def get_subset(self, indices: list[int]) -> "BaseDataset":
        """
        Create subset of dataset with specified indices.

        Args:
            indices: List of sample indices to include

        Returns:
            New dataset instance with subset of samples
        """
        subset_data = [self._data[i] for i in indices]
        return self.__class__(data=subset_data, name=f"{self.name}_subset")

    def split(
        self,
        train_size: int,
        val_size: int,
        test_size: int,
        shuffle: bool = True,
        seed: int = 42
    ) -> tuple["BaseDataset", "BaseDataset", "BaseDataset"]:
        """
        Split dataset into train/val/test sets.

        Args:
            train_size: Number of training samples
            val_size: Number of validation samples
            test_size: Number of test samples
            shuffle: Whether to shuffle before splitting
            seed: Random seed for shuffling

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        import random

        total_size = train_size + val_size + test_size
        if total_size > len(self):
            raise ValueError(
                f"Requested {total_size} samples but dataset has only {len(self)}"
            )

        indices = list(range(len(self)))
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(indices)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:train_size + val_size + test_size]

        train_data = [self._data[i] for i in train_indices]
        val_data = [self._data[i] for i in val_indices]
        test_data = [self._data[i] for i in test_indices]

        return (
            self.__class__(data=train_data, name=f"{self.name}_train"),
            self.__class__(data=val_data, name=f"{self.name}_val"),
            self.__class__(data=test_data, name=f"{self.name}_test")
        )

    def cycle_to_size(self, target_size: int) -> "BaseDataset":
        """
        Create cyclic dataset by repeating samples to reach target size.

        Useful for batch size tuning experiments where you need a specific
        number of samples regardless of original dataset size.

        Args:
            target_size: Desired number of samples in output dataset

        Returns:
            New dataset instance with cycled samples

        Example:
            >>> dataset = TriviaQADataset(data=[...])  # 100 samples
            >>> cycled = dataset.cycle_to_size(250)  # 250 samples (100 + 100 + 50)
        """
        if target_size <= 0:
            raise ValueError(f"target_size must be positive, got {target_size}")

        if len(self) == 0:
            raise ValueError("Cannot cycle empty dataset")

        # Calculate how many full cycles + remainder we need
        n_full_cycles = target_size // len(self)
        remainder = target_size % len(self)

        # Build cycled data
        cycled_data = []
        for _ in range(n_full_cycles):
            cycled_data.extend(self._data)

        # Add remainder
        if remainder > 0:
            cycled_data.extend(self._data[:remainder])

        return self.__class__(
            data=cycled_data,
            name=f"{self.name}_cycled_{target_size}"
        )
