"""
Scoring module for uncertainty and quality estimation.

This module provides interfaces and implementations for various scoring methods:
- Sampling-based methods (Semantic Entropy, Naive Entropy, etc.)
- Probe-based methods (Linear Probes for uncertainty/accuracy prediction)
- Reward model-based methods (future)
"""

from src.scoring.base import ScorerInterface
from src.scoring.inputs import ScorerInput, ProbeInput, RewardModelInput
from src.scoring.linear_probe import LinearProbeScorer

__all__ = [
    "ScorerInterface",
    "ScorerInput",
    "SamplerInput",
    "ProbeInput",
    "RewardModelInput",
    "LinearProbeScorer",
]
