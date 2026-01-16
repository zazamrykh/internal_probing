from dataclasses import dataclass
from typing import List

from src.model.base import ModelWrapper
from src.scoring import ScorerInput

@dataclass
class SamplerInput(ScorerInput):
    """
    Input for sampling-based uncertainty estimation methods.

    Used by methods that generate multiple responses and measure consistency:
    - Semantic Entropy
    - Naive Entropy
    - Kernel Language Entropy (KLE)

    Attributes:
        prompts: List of text prompts to generate responses for
        model_wrapper: ModelWrapper instance for generation
    """
    prompts: List[str]
    model_wrapper: ModelWrapper
