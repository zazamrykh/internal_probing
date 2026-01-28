"""
Dataset module for internal probing experiments.

Provides unified interface for working with QA datasets and enriching them
with model generations, activations, and uncertainty scores.
"""

from src.dataset.base import BaseDataset, DatasetSample
from src.dataset.evaluators import (
    CorrectnessEvaluator,
    SubstringMatchEvaluator,
    ExactMatchEvaluator,
    CustomEvaluator,
    create_evaluator,
)

__all__ = [
    "BaseDataset",
    "DatasetSample",
    "CorrectnessEvaluator",
    "SubstringMatchEvaluator",
    "ExactMatchEvaluator",
    "CustomEvaluator",
    "create_evaluator",
]
