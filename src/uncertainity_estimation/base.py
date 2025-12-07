from abc import ABC, abstractmethod
from typing import List, Optional

from activation_extractor.base import Activation

class ScorerInterface(ABC):
    @abstractmethod
    def estimate(self, model, prompts, answers: Optional[str] = None, activation : Optional[List[Activation]] = None, **kwargs):
        """
        Estimate uncertainty for a prompt / prompt-answer / activations.

        Args:
            model: LLM model
            prompts: input prompt for sampling-based methods
            answers: answers for each prompts for reward-model-based methods
            activation: cached activations for linear probes based methods
            **kwargs: method-specific parameters

        Returns:
            float: uncertainty score (higher = more uncertain)
        """
        pass
