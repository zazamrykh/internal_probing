"""
Activation enricher for extracting hidden states from model.

Based on semantic_entropy_probes.ipynb implementation.
"""

import torch
from typing import List

from src.dataset.enrichers.base import BaseEnricher
from src.model.base import ModelWrapper


class ActivationEnricher(BaseEnricher):
    """
    Enricher that extracts hidden state activations from model.

    Adds fields:
    - activations: dict with structure:
        {
            "positions": list[int],
            "layers": list[int],
            "acts": {pos: {layer: tensor(hidden_dim)}},
            "gen_len": int
        }

    Requires that sample already has 'greedy_answer' field from GreedyGenerationEnricher.

    Based on extract_activations_for_probe() from semantic_entropy_probes.ipynb (lines 473-571).
    """

    def __init__(
        self,
        model_wrapper: ModelWrapper,
        layers: List[int],
        positions: List[int],
        inplace: bool = False,
        verbose: bool = True
    ):
        """
        Initialize activation enricher.

        Args:
            model_wrapper: ModelWrapper instance for activation extraction
            layers: List of layer indices to extract (0-indexed)
            positions: List of positions in generated part to extract
                      (0 = first generated token, -2 = second-to-last, etc.)
            inplace: Whether to modify samples in-place
            verbose: Whether to print progress
        """
        super().__init__(inplace=inplace, verbose=verbose)
        self.model_wrapper = model_wrapper
        self.layers = layers
        self.positions = positions

    def enrich_sample(self, sample: dict, **kwargs) -> dict:
        """
        Extract activations for sample.

        Requires sample to have 'prompt' and 'greedy_answer' fields.
        The greedy_answer must have been generated previously.

        Args:
            sample: Sample dict with 'prompt' and 'greedy_answer'
            **kwargs: Override parameters (layers, positions)

        Returns:
            Sample with added 'activations' field
        """
        if 'greedy_answer' not in sample:
            raise ValueError(
                "Sample must have 'greedy_answer' field. "
                "Run GreedyGenerationEnricher first."
            )

        prompt = sample['prompt']
        greedy_answer = sample['greedy_answer']

        # Override parameters if provided
        layers = kwargs.get('layers', self.layers)
        positions = kwargs.get('positions', self.positions)

        # Get token IDs for prompt
        inputs = self.model_wrapper.prepare_inputs([prompt])
        prompt_ids = inputs['input_ids'][0]  # (prompt_len,)

        # Tokenize the greedy answer to get its token IDs
        # This follows the pattern from notebook where we tokenize the generated text
        gen_tokens = self.model_wrapper.tokenizer.encode(
            greedy_answer,
            add_special_tokens=False,
            return_tensors='pt'
        )[0]  # (gen_len,)

        # Extract activations using ModelWrapper method
        # This corresponds to extract_activations_for_probe() in notebook
        activations = self.model_wrapper.extract_activations_reforward(
            prompt_ids=prompt_ids,
            generated_ids=gen_tokens,
            layers=layers,
            positions=positions
        )

        # Add to sample
        sample['activations'] = activations

        return sample
