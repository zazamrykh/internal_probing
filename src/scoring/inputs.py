"""
Input dataclasses for different scorer types.

These dataclasses provide type-safe, structured inputs for uncertainty estimation methods,
replacing the previous approach with multiple Optional parameters.
"""

from dataclasses import dataclass
from typing import List, Optional
import torch

from src.model.base import ModelWrapper


@dataclass
class ScorerInput:
    """
    Base class for scorer inputs.

    All specific input types inherit from this base class.
    """
    pass


@dataclass
class ProbeInput(ScorerInput):
    """
    Input for probe-based uncertainty estimation methods.

    Used by methods that predict uncertainty from cached activations:
    - Linear Probes (Semantic Entropy Probes, Accuracy Probes)

    Attributes:
        activations: Tensor of hidden state activations, shape (batch_size, hidden_dim)
    """
    activations: torch.Tensor


@dataclass
class RewardModelInput(ScorerInput):
    """
    (Maybe for future)
    Input for reward model-based uncertainty estimation.

    Used by methods that score prompt-answer pairs:
    - Reward Models
    - Verifier Models

    Attributes:
        prompts: List of text prompts
        answers: List of generated answers corresponding to prompts
        model_wrapper: Optional ModelWrapper if reward model needs it
    """
    prompts: List[str]
    answers: List[str]
    model_wrapper: Optional[ModelWrapper] = None
