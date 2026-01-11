"""
Generation enrichers for adding model outputs to dataset samples.

Based on semantic_entropy_probes.ipynb implementation.
"""

import torch
from typing import Optional

from src.dataset.enrichers.base import BaseEnricher
from src.model.base import ModelWrapper


class GreedyGenerationEnricher(BaseEnricher):
    """
    Enricher that adds greedy generation outputs to samples.

    Adds fields:
    - greedy_answer: str - The greedy decoded answer
    - is_correct: float - Correctness score (requires evaluator)

    Based on greedy_generate_text() and process_one_example_greedy_with_acts()
    from semantic_entropy_probes.ipynb (lines 425-622).
    """

    def __init__(
        self,
        model_wrapper: ModelWrapper,
        evaluator=None,
        max_new_tokens: int = 64,
        inplace: bool = False,
        verbose: bool = True
    ):
        """
        Initialize greedy generation enricher.

        Args:
            model_wrapper: ModelWrapper instance for generation (contains system_prompt)
            evaluator: CorrectnessEvaluator for computing is_correct field
            max_new_tokens: Maximum tokens to generate
            inplace: Whether to modify samples in-place
            verbose: Whether to print progress
        """
        super().__init__(inplace=inplace, verbose=verbose)
        self.model_wrapper = model_wrapper
        self.evaluator = evaluator
        self.max_new_tokens = max_new_tokens

    def enrich_sample(self, sample: dict, **kwargs) -> dict:
        """
        Add greedy generation to sample.

        Args:
            sample: Sample dict with 'prompt' and 'gt_answers' fields
            **kwargs: Override parameters (max_new_tokens)

        Returns:
            Sample with added 'greedy_answer' and optionally 'is_correct'
        """
        prompt = sample['prompt']
        gt_answers = sample.get('gt_answers', [])

        # Override parameters if provided
        max_new_tokens = kwargs.get('max_new_tokens', self.max_new_tokens)

        # Generate greedy answer using ModelWrapper.generate_greedy()
        # This follows the pattern from greedy_generate_text() in notebook (lines 425-459)
        generated_texts, log_probs = self.model_wrapper.generate_greedy(
            prompts=[prompt],
            max_new_tokens=max_new_tokens
        )

        # Extract generated text
        greedy_answer = generated_texts[0]

        # Add to sample
        sample['greedy_answer'] = greedy_answer

        # Compute correctness if evaluator provided
        if self.evaluator is not None:
            is_correct = self.evaluator.evaluate(
                prompt=prompt,
                generated_answer=greedy_answer,
                gt_answers=gt_answers
            )
            sample['is_correct'] = float(is_correct)

        return sample
