"""
Sampling-based scorers for uncertainty estimation.

These scorers generate multiple responses and measure their consistency
to estimate uncertainty (e.g., Semantic Entropy, Naive Entropy).
"""

from abc import abstractmethod
from typing import List, Union, Optional
import torch
from transformers import GenerationConfig

from src.scoring.base import ScorerInterface
from src.scoring.inputs import SamplerInput


class SamplerInterface(ScorerInterface):
    """
    Base class for sampling-based uncertainty estimation.

    These methods generate multiple responses for each prompt and measure
    their consistency/diversity to estimate uncertainty.

    Each implementation is free to define its own generation and scoring logic.
    Some may need log probabilities, others embeddings, others just text.

    Attributes:
        n_samples: Number of responses to generate per prompt
        sampling_batch_size: Batch size for generation (memory vs speed tradeoff)
        generation_config: Configuration for sampling (temperature, top_p, etc.)
    """

    def __init__(
        self,
        n_samples: int = 10,
        sampling_batch_size: int = 1,
        generation_config: Optional[GenerationConfig] = None
    ):
        """
        Initialize sampler.

        Args:
            n_samples: Number of samples to generate per prompt
            sampling_batch_size: Batch size for generation
            generation_config: Generation configuration (if None, uses default sampling config)
        """
        self.n_samples = n_samples
        self.sampling_batch_size = sampling_batch_size
        self.generation_config = generation_config or self._default_generation_config()

    def _default_generation_config(self) -> GenerationConfig:
        """Default sampling configuration."""
        return GenerationConfig(
            do_sample=True,
            temperature=1.0,
            top_p=0.95,
            num_beams=1,
            return_dict_in_generate=True,
            output_logits=True,
        )

    @abstractmethod
    def estimate(self, input: SamplerInput) -> Union[float, List[float]]:
        """
        Estimate uncertainty by sampling multiple responses.

        Each implementation defines its own logic for:
        - How to generate samples (with/without log probs, embeddings, etc.)
        - How to compute uncertainty from samples

        Args:
            input: SamplerInput containing prompts and model_wrapper

        Returns:
            Uncertainty score(s) - float for single prompt, list for multiple
        """
        if not isinstance(input, SamplerInput):
            raise TypeError(
                f"{self.__class__.__name__} requires SamplerInput, "
                f"got {type(input).__name__}"
            )
        pass
