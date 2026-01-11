"""
Dataset enrichers for adding computed fields to samples.

Enrichers add fields like generations, activations, and uncertainty scores
to dataset samples without modifying the original data structure.
"""

from src.dataset.enrichers.base import BaseEnricher
from src.dataset.enrichers.generation import GreedyGenerationEnricher
from src.dataset.enrichers.activation import ActivationEnricher
from src.dataset.enrichers.semantic_entropy import SemanticEntropyEnricher

__all__ = [
    "BaseEnricher",
    "GreedyGenerationEnricher",
    "ActivationEnricher",
    "SemanticEntropyEnricher",
]
