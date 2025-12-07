import torch
from transformers import GenerationConfig

from abc import abstractmethod
from typing import List, Union
from uncertainity_estimation.base import ScorerInterface


class SamplerInterface(ScorerInterface):
    """Methods that generate multiple answers and measure consistency"""

    def __init__(self, n_samples=10, sampling_batch_size=1, generation_config: GenerationConfig = None):
        self.n_samples = n_samples
        self.sampling_batch_size = sampling_batch_size
        self.generation_config = generation_config


    def estimate(self, model, prompts, answers=None, activation=None, **kwargs):
        if answers is not None:
            raise ValueError(
                f"{self.__class__.__name__} doesn't use provided answer. "
                "It generates multiple samples internally."
            )

        if activation is not None:
            raise ValueError(
                f"{self.__class__.__name__} doesn't use provided activation. "
                "It generates multiple samples internally."
            )

        if prompts is None:
            raise ValueError(
                f"{self.__class__.__name__} must be given prompts. "
                "It generates multiple answers internally based by given prompts."
            )


        # Generate multiple answers
        generated = self.generate_samples(model, prompts)
        return self.compute_consistency(generated)

    @abstractmethod
    def generate_samples(self, model, prompts: Union[List[List[torch.long]], List[str]]):
        pass


    @abstractmethod
    def compute_consistency(self, generated: Union[List[str], List[List[torch.long]], List[torch.Tensor]]):
        """
        Compute consistency of generations of model which can be interpreted as confidence of model

        Args:
            generated: Could be generated texts represented by str of token-like format.
                Also can be torch tensors – embeddings which can be used for INSIDE method

        Returns:
            float: score – consisntency or confidence of answer
        """
        pass
