"""
Base interface for uncertainty estimation scorers.

This module provides the abstract base class for all uncertainty estimation methods,
using dataclass-based inputs for type safety and clarity.
"""

from abc import ABC, abstractmethod
from typing import Union, Dict, Any, Optional
import numpy as np

from src.scoring.inputs import ScorerInput


class ScorerInterface(ABC):
    """
    Abstract base class for uncertainty estimation scorers.

    All uncertainty estimation scorers (sampling-based, probe-based, reward-model-based)
    inherit from this class and implement the estimate() method.

    """

    @abstractmethod
    def estimate(self, input: ScorerInput) -> Union[float, np.ndarray]:
        """
        Estimate uncertainty score(s) for given input.

        Args:
            input: ScorerInput subclass (SamplerInput, ProbeInput, or RewardModelInput)
                   containing all necessary data for this scorer type

        Returns:
            Uncertainty (or other) score(s):
            - float: Single uncertainty score
            - np.ndarray: Array of scores (e.g., for batch of samples)

            Higher scores typically indicate higher uncertainty/lower confidence.
        """
        pass


    @abstractmethod
    def get_input(self, sample: dict) -> ScorerInput:
        """ Get input for estimate method based on dataset sample """
        pass
