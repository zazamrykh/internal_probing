"""
It is needed for compatibility with src/experiments pipeline
Here we describe how to train scorer (if needed) and how to get final model for test eval.
Method implementations for uncertainty estimation.

This module provides concrete implementations of MethodInterface:
- LinearProbeMethod: sklearn-based linear probes on activations
- PEPMethod: Prompt Embedding Probes with learnable embeddings
- SemanticEntropyMethod: Semantic entropy-based uncertainty estimation
"""

from src.experiments.methods.base import MethodInterface, MethodFactory
from src.experiments.methods.semantic_entropy import SemanticEntropyMethod

# LinearProbeMethod and PEPMethod will be imported when implemented
# from src.experiments.methods.linear_probe import LinearProbeMethod
# from src.experiments.methods.pep import PEPMethod

__all__ = [
    "MethodInterface",
    "MethodFactory",
    "SemanticEntropyMethod",
    # "LinearProbeMethod",
    # "PEPMethod",
]
