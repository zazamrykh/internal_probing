"""
Precomputed scorer for extracting uncertainty values from dataset samples.

This scorer is used when uncertainty scores (e.g., semantic entropy) have been
precomputed during the enrichment phase and stored in the dataset samples.
"""

import numpy as np
from typing import Union

from src.scoring.base import ScorerInterface
from src.scoring.inputs import SampleInput


class PrecomputedScorer(ScorerInterface):
    """
    Scorer that extracts precomputed uncertainty values from dataset samples.

    Used when uncertainty scores have been computed during enrichment and
    stored in the dataset. Simply extracts the specified field from the sample.

    SCORING CONVENTION: Returns uncertainty/hallucination score
    - Higher score = higher risk of hallucination/error
    - Lower score = higher confidence/correctness

    Example:
        >>> scorer = PrecomputedScorer(field_name="se_raw")
        >>> input_data = SampleInput(sample={"se_raw": 0.75, "is_correct": True})
        >>> uncertainty = scorer.estimate(input_data)
        >>> print(f"Uncertainty: {uncertainty}")  # 0.75
    """

    def __init__(self, field_name: str = "se_raw"):
        """
        Initialize precomputed scorer.

        Args:
            field_name: Name of the field in the sample dict containing
                       the precomputed uncertainty score.
                       Common values: "se_raw", "se_binary"
        """
        self.field_name = field_name

    def estimate(self, input: SampleInput) -> Union[float, np.ndarray]:
        """
        Extract precomputed uncertainty score from sample.

        Args:
            input: SampleInput containing the dataset sample dict

        Returns:
            Precomputed uncertainty score from the specified field

        Raises:
            TypeError: If input is not SampleInput
            KeyError: If field_name is not found in sample
        """
        if not isinstance(input, SampleInput):
            raise TypeError(
                f"{self.__class__.__name__} requires SampleInput, "
                f"got {type(input).__name__}"
            )

        sample = input.sample

        if self.field_name not in sample:
            raise KeyError(
                f"Field '{self.field_name}' not found in sample. "
                f"Available fields: {list(sample.keys())}"
            )

        return float(sample[self.field_name])


    def get_input(self, sample: dict):
        """ Retruns input for estimate method from dataset sample """
        return SampleInput(sample=sample)
