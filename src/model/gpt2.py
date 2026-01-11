"""
GPT-2 model wrapper implementation.

GPT-2 doesn't have a chat template, so we use plain text prompts.
"""

from typing import Dict, List
import logging

import torch

from src.model.base import ModelWrapper

logger = logging.getLogger(__name__)


class GPT2Model(ModelWrapper):
    """
    Wrapper for GPT-2 family models.

    GPT-2 models don't support chat templates, so this wrapper
    uses plain text prompts with optional system prompt prepended.

    Example:
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> wrapper = GPT2Model(model, tokenizer)
        >>> texts, log_probs = wrapper.generate_greedy(["What is 2+2?"])
    """

    def prepare_inputs(
        self,
        prompts: List[str],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for GPT-2 using plain tokenization.

        Args:
            prompts: List of text prompts
            **kwargs: Additional arguments (ignored for GPT-2)

        Returns:
            Dictionary with input_ids and attention_mask
        """
        # Prepend system prompt if provided
        if self.system_prompt:
            prompts = [f"{self.system_prompt}\n\n{p}" for p in prompts]

        # Plain tokenization with left padding (for batch generation)
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            padding_side="left",
        )

        logger.debug(f"Prepared inputs for {len(prompts)} prompts, shape: {inputs['input_ids'].shape}")

        return inputs.to(self.device)
