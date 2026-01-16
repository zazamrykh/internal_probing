"""
Sampling submodule of scoring module. Using for methods of scoring based on sampling for unconfidence estimation.

Here can be implemented such methods as naive entropy, semantic entopy, kernel language entropy, method of INSIDE paper
"""

from src.scoring.sampling.inputs import SamplerInput
from src.scoring.sampling.base import SamplerInterface
from src.scoring.sampling.semantic_entropy import SemanticEntropyScorer

__all__ = [
    "SamplerInput",
    "SamplerInterface",
    "SemanticEntropyScorer"
]
