"""
Mistral model wrapper implementation.

Supports chat templates and optional 8-bit quantization for memory efficiency.
"""

from typing import Dict, List, Optional
import logging

import torch
from transformers import BitsAndBytesConfig

from src.model.base import ModelWrapper

logger = logging.getLogger(__name__)


class MistralModel(ModelWrapper):
    """
    Wrapper for Mistral family models (Mistral-7B-Instruct, etc.).

    Mistral models support chat templates and can be quantized to 8-bit
    for reduced memory usage (important for GPUs with limited VRAM).

    Example:
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> from src.model import MistralModel
        >>>
        >>> # With 8-bit quantization (default)
        >>> model = AutoModelForCausalLM.from_pretrained(
        ...     "mistralai/Mistral-7B-Instruct-v0.1",
        ...     quantization_config=MistralModel.get_quantization_config(),
        ...     device_map="auto"
        ... )
        >>> tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
        >>> wrapper = MistralModel(model, tokenizer, system_prompt="Answer briefly.")
        >>>
        >>> texts, log_probs = wrapper.generate_greedy(["What is 2+2?"])
    """

    @staticmethod
    def get_quantization_config(load_in_8bit: bool = True) -> Optional[BitsAndBytesConfig]:
        """
        Get quantization configuration for memory-efficient loading.

        Args:
            load_in_8bit: If True, use 8-bit quantization (default)

        Returns:
            BitsAndBytesConfig for 8-bit quantization, or None if disabled
        """
        if not load_in_8bit:
            return None

        return BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16,
        )

    def prepare_inputs(
        self,
        prompts: List[str],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for Mistral using chat template.

        Mistral uses a specific chat template format. If system_prompt is provided,
        it's added as a system message.

        Args:
            prompts: List of text prompts
            **kwargs: Additional arguments (e.g., enable_thinking for Qwen models)

        Returns:
            Dictionary with input_ids and attention_mask
        """
        # Build messages for each prompt
        messages_list = []
        for prompt in prompts:
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": prompt})
            messages_list.append(messages)

        # Apply chat template with left padding for batch generation
        inputs = self.tokenizer.apply_chat_template(
            messages_list,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            padding_side="left",
        )

        return inputs.to(self.device)
