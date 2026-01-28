"""
Base model wrapper for unified interface across different LLM architectures.

This module provides abstract base class for model wrappers that handle:
- Input preparation (chat templates, tokenization)
- Text generation with log probabilities
- Activation extraction from hidden states
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
import logging

import torch
import torch.nn.functional as F
from transformers import (
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

logger = logging.getLogger(__name__)


class ModelWrapper(ABC):
    """
    Abstract base class for LLM model wrappers.

    Provides unified interface for different model architectures (Mistral, GPT-2, etc.)
    with support for generation and activation extraction.

    Attributes:
        model: HuggingFace transformers model
        tokenizer: HuggingFace tokenizer
        system_prompt: Optional system prompt to prepend to all queries
        device: Device where model is located
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize model wrapper.

        Args:
            model: Pre-loaded HuggingFace model
            tokenizer: Corresponding tokenizer
            system_prompt: Optional system prompt for all generations
        """
        self.model = model
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.device = model.device

        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        logger.info(
            f"Initialized {self.__class__.__name__} on device {self.device}"
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        system_prompt: str = "",
        quantization: Optional[str] = None,
        device_map: str = "auto",
        **kwargs
    ) -> 'ModelWrapper':
        """
        Factory method to create appropriate ModelWrapper from model name.

        Automatically detects model type from name and creates the right wrapper.

        Args:
            model_name: HuggingFace model name or path
            system_prompt: Optional system prompt for all generations
            quantization: Quantization type: "8bit", "4bit", or None
            device_map: Device map for model loading
            **kwargs: Additional arguments for model loading

        Returns:
            Appropriate ModelWrapper subclass instance

        Example:
            >>> wrapper = ModelWrapper.from_pretrained(
            ...     "mistralai/Mistral-7B-Instruct-v0.1",
            ...     quantization="8bit"
            ... )
        """
        # Import here to avoid circular imports
        from src.model.mistral import MistralModel
        from src.model.gpt2 import GPT2Model

        logger.info(f"Loading model: {model_name}")

        # Setup quantization if requested
        quant_config = None
        if quantization == '8bit':
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
        elif quantization == '4bit':
            quant_config = BitsAndBytesConfig(load_in_4bit=True)

        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            quantization_config=quant_config,
            **kwargs
        )

        # Setup padding
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
        if hasattr(model, 'generation_config'):
            model.generation_config.pad_token_id = tokenizer.pad_token_id

        # Detect model type from name and create appropriate wrapper
        model_name_lower = model_name.lower()

        if 'mistral' in model_name_lower:
            wrapper = MistralModel(
                model=model,
                tokenizer=tokenizer,
                system_prompt=system_prompt
            )
            logger.info(f"Created MistralModel wrapper")
        elif 'gpt2' in model_name_lower or 'gpt-2' in model_name_lower:
            wrapper = GPT2Model(
                model=model,
                tokenizer=tokenizer,
                system_prompt=system_prompt
            )
            logger.info(f"Created GPT2Model wrapper")
        else:
            raise ValueError(
                f"Unknown model type for '{model_name}'. "
                f"Supported: mistral, gpt2"
            )

        return wrapper

    def prepare_inputs(
        self,
        prompts: List[str],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare model inputs from text prompts.

        This method handles model-specific input formatting including:
        - Chat template application (if applicable)
        - Tokenization
        - Padding and attention masks

        Default implementation uses chat template if available, otherwise plain tokenization.
        Override this method for model-specific behavior.

        Args:
            prompts: List of text prompts
            **kwargs: Additional model-specific arguments

        Returns:
            Dictionary with input_ids, attention_mask, and other tensors
        """
        # Try to use chat template if available
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template is not None:
            messages_list = []
            for prompt in prompts:
                messages = []
                if self.system_prompt:
                    messages.append({"role": "system", "content": self.system_prompt})
                messages.append({"role": "user", "content": prompt})
                messages_list.append(messages)

            inputs = self.tokenizer.apply_chat_template(
                messages_list,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
                padding_side="left",
            )
        else:
            # Fallback: plain tokenization
            logger.debug("Chat template not available, using plain tokenization")
            if self.system_prompt:
                prompts = [f"{self.system_prompt}\n\n{p}" for p in prompts]

            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                padding_side="left",
            )

        return inputs.to(self.device)

    def generate(
        self,
        prompts: List[str],
        generation_config: Optional[GenerationConfig] = None,
        return_log_probs: bool = True,
        return_ids: bool = False,
        only_generated: bool = True,
        **kwargs
    ) -> Tuple[Union[List[str], List[torch.Tensor]], Optional[List[torch.Tensor]]]:
        """
        Generate text from prompts with optional log probabilities.

        Args:
            prompts: List of input prompts
            generation_config: Generation configuration (temperature, top_p, etc.)
            return_log_probs: If True, return log probabilities for each generated token
            return_ids: If True, return token IDs instead of decoded text
            only_generated: If True, return only generated part (exclude prompt)
            **kwargs: Additional generation arguments

        Returns:
            Tuple of (generated_texts_or_ids, log_probs_list)
            - generated_texts_or_ids: List of generated strings or token ID tensors
            - log_probs_list: List of log probability tensors (one per sequence) if return_log_probs=True
        """
        if generation_config is None:
            generation_config = GenerationConfig(
                num_beams=1,
                do_sample=False,
                return_dict_in_generate=True,
                output_logits=True,
            )

        # Ensure we get logits for log prob calculation
        if return_log_probs:
            generation_config.output_logits = True
            generation_config.return_dict_in_generate = True

        # Prepare inputs
        inputs = self.prepare_inputs(prompts, **kwargs)
        input_ids = inputs["input_ids"]

        logger.debug(
            f"Generating for {len(prompts)} prompts with config: {generation_config}"
        )

        # Generate
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, generation_config=generation_config)

        sequences = outputs.sequences  # (batch, prompt_len + gen_len)
        batch_size = sequences.size(0)

        # Extract generated part if requested
        if only_generated:
            trunc_seq_list, log_probs_list = self._truncate_output(
                outputs, input_ids, return_log_probs
            )

            if not return_ids:
                texts = [
                    self.tokenizer.decode(seq, skip_special_tokens=True)
                    for seq in trunc_seq_list
                ]
                return texts, log_probs_list if return_log_probs else None
            else:
                return trunc_seq_list, log_probs_list if return_log_probs else None
        else:
            # Return full sequences
            if not return_ids:
                texts = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)
                logits_list = outputs.get("logits") if return_log_probs else None
                return texts, logits_list
            else:
                seq_list = [sequences[i] for i in range(batch_size)]
                logits_list = outputs.get("logits") if return_log_probs else None
                return seq_list, logits_list

    def generate_greedy(
        self,
        prompts: List[str],
        max_new_tokens: int = 64,
        **kwargs
    ) -> Tuple[List[str], List[torch.Tensor]]:
        """
        Generate text using greedy decoding (deterministic).

        Convenience method for greedy generation with log probabilities.

        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum number of tokens to generate
            **kwargs: Additional arguments passed to prepare_inputs

        Returns:
            Tuple of (generated_texts, log_probs_list)
        """
        gen_config = GenerationConfig(
            do_sample=False,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            return_dict_in_generate=True,
            output_logits=True,
        )

        return self.generate(
            prompts,
            generation_config=gen_config,
            return_log_probs=True,
            only_generated=True,
            **kwargs
        )

    def extract_activations_reforward(
        self,
        prompt_ids: torch.Tensor,
        generated_ids: torch.Tensor,
        layers: List[int],
        positions: List[int],
    ) -> Dict:
        """
        Extract hidden state activations by re-forwarding full sequence through model.

        This is a simple but memory-intensive approach: concatenate prompt + generation
        and forward through model with output_hidden_states=True.

        Args:
            prompt_ids: Prompt token IDs, shape (1, prompt_len) or (prompt_len,)
            generated_ids: Generated token IDs, shape (gen_len,)
            layers: List of layer indices to extract (0-indexed)
            positions: List of positions in generated part to extract
                      (0 = first generated token, -1 = last, -2 = second-to-last)

        Returns:
            Dictionary with:
                - positions: List of positions extracted
                - layers: List of layers extracted
                - acts: Dict[position][layer] -> torch.Tensor (hidden_dim,) on CPU
                - gen_len: Length of generated sequence
        """
        # Ensure prompt_ids is 2D
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)

        # Move tensors to same device before concatenation
        prompt_ids = prompt_ids.to(self.device)
        generated_ids = generated_ids.to(self.device)

        # Concatenate prompt and generation
        full_ids = torch.cat([prompt_ids[0], generated_ids], dim=0).unsqueeze(0)
        attention_mask = torch.ones_like(full_ids)

        logger.debug(
            f"Extracting activations: prompt_len={prompt_ids.shape[1]}, "
            f"gen_len={len(generated_ids)}, layers={layers}, positions={positions}"
        )

        # Forward pass with hidden states
        with torch.inference_mode():
            outputs = self.model(
                input_ids=full_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )

        hidden_states = outputs.hidden_states  # Tuple of (num_layers+1) tensors

        prompt_len = prompt_ids.shape[1]
        gen_len = generated_ids.shape[0]

        # Extract activations at specified positions and layers
        acts = {}
        for pos in positions:
            # Convert position to absolute index in generated part
            idx = pos if pos >= 0 else gen_len + pos
            idx = max(0, min(idx, gen_len - 1))  # Clamping to available diapasone

            # Absolute position in full sequence
            full_pos = prompt_len + idx

            acts[pos] = {}
            for layer in layers:
                # hidden_states[0] is embedding layer, so layer i is at index i+1
                hidden = hidden_states[layer + 1][0, full_pos, :]
                acts[pos][layer] = hidden.detach().cpu()

        return {
            "positions": positions,
            "layers": layers,
            "acts": acts,
            "gen_len": gen_len,
        }

    def _truncate_output(
        self,
        outputs,
        input_ids: torch.Tensor,
        return_log_probs: bool = True,
    ) -> Tuple[List[torch.Tensor], Optional[List[torch.Tensor]]]:
        """
        Truncate model outputs to only generated part and compute log probabilities.

        Args:
            outputs: Model generation outputs with sequences and logits
            input_ids: Original input token IDs
            return_log_probs: Whether to compute log probabilities

        Returns:
            Tuple of (truncated_sequences, log_probs_list)
        """
        sequences = outputs.sequences
        trunc_seq_list = []

        # Truncate each sequence to generated part only
        for i, seq in enumerate(sequences):
            first_idx = len(input_ids[i])

            # Find end of generation (EOS or PAD token)
            if seq[-1] == self.tokenizer.eos_token_id or seq[-1] == self.tokenizer.pad_token_id:
                eos_positions = torch.nonzero(seq == self.tokenizer.eos_token_id)
                if len(eos_positions) > 0:
                    last_idx = eos_positions[-1].item()
                else:
                    last_idx = len(seq) - 1
            else:
                last_idx = len(seq) - 1

            trunc_seq = seq[first_idx : last_idx + 1]
            trunc_seq_list.append(trunc_seq)

        if not return_log_probs:
            return trunc_seq_list, None

        # Compute log probabilities for each generated token
        result_list = [[] for _ in range(len(trunc_seq_list))]

        for step, logits in enumerate(outputs.logits):
            for i, seq in enumerate(trunc_seq_list):
                if step >= len(seq):
                    continue

                # Log probability of the actual token generated
                log_prob = F.log_softmax(logits[i], dim=0)[seq[step]]
                result_list[i].append(log_prob.item())

        # Convert to tensors
        result_tensors_list = []
        for trunc_seq, log_probs in zip(trunc_seq_list, result_list):
            assert len(trunc_seq) == len(log_probs), (
                f"Sequence length {len(trunc_seq)} != log_probs length {len(log_probs)}"
            )
            result_tensors_list.append(torch.tensor(log_probs))

        return trunc_seq_list, result_tensors_list

    # TODO: Future enhancement - hook-based activation extraction during generation
    # def extract_activations_hooks(self, prompts, generation_config, layers, positions):
    #     """Extract activations during generation using hooks (more efficient)."""
    #     pass
