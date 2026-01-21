"""
Prompt Embedding Probe Model (PEPModel) for uncertainty estimation.

This module implements end-to-end learnable probing where both soft prompt embeddings
and a linear probe are trained jointly to predict uncertainty/correctness from internal
model activations.

Key features:
- Learnable soft prompt embeddings injected into the sequence
- Linear probe applied to activations at specified layer/position
- Support for both TBG (To Be Generated) and SLT (Second Last Token) setups
- Compatible with ScorerInterface for unified API
"""

import logging
from typing import Optional, Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import GenerationConfig

from src.scoring.base import ScorerInterface
from src.scoring.inputs import PromptEmbeddingInput, ScorerInput
from src.model.base import ModelWrapper

logger = logging.getLogger(__name__)


class PEPModel(nn.Module, ScorerInterface):
    """
    Prompt Embedding Probe Model for uncertainty estimation.

    Combines learnable soft prompt embeddings with a linear probe to predict
    uncertainty/correctness from internal model activations. The base LLM
    remains frozen - only embeddings and probe are trainable.

    Architecture:
        1. Learnable prompt embeddings (n_embeddings x hidden_dim)
        2. Frozen base LLM
        3. Linear probe (hidden_dim -> 1) applied to activations of last token (embedding)

    The forward pass always:
        - Takes token IDs as input
        - Converts to embeddings
        - Appends learnable prompt embeddings at the end
        - Runs through frozen model
        - Extracts activation of the LAST token (which corresponds to the last prompt embedding)
        - Applies linear probe to this activation

    Attributes:
        model_wrapper: ModelWrapper containing the frozen base LLM
        prompt_embeddings: Learnable soft token embeddings
        probe: Linear layer for prediction
        probe_layer: Which layer to extract activations from (0-indexed)
        probe_position: Token position for probing (used for sequence truncation before forward)
        n_embeddings: Number of soft prompt tokens
        hidden_dim: Dimension of model hidden states
        probe_type: Type of probe ('accuracy' or 'sep')

    Example:
        >>> from src.model import MistralModel
        >>> wrapper = MistralModel(model, tokenizer)
        >>>
        >>> # Create PEPModel
        >>> pep_model = PEPModel(
        ...     model_wrapper=wrapper,
        ...     n_embeddings=1,
        ...     probe_layer=16,
        ...     probe_position=-2,
        ...     probe_type='accuracy'
        ... )
        >>>
        >>> # Score a prompt
        >>> score = pep_model.score_prompt("What is the capital of France?")
        >>> print(f"Uncertainty score: {score:.3f}")
    """

    def __init__(
        self,
        model_wrapper: ModelWrapper,
        n_embeddings: int = 1,
        probe_layer: int = 16,
        probe_position: int = -2,
        probe_type: str = "accuracy",
        embedding_init_std: float = 0.02,
        param_dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize PEPModel.

        Args:
            model_wrapper: ModelWrapper with frozen base LLM
            n_embeddings: Number of learnable soft prompt tokens
            probe_layer: Layer index to extract activations from (0-indexed)
            probe_position: Token position for probing - used to truncate sequence before forward
                           (0=first token, -2=second-to-last, etc.)
            probe_type: Type of probe - 'accuracy' (predicts P(correct)) or 'sep' (predicts P(high_SE))
            embedding_init_std: Standard deviation for embedding initialization
            param_dtype: Dtype for trainable parameters (torch.float32 or torch.float16)
        """
        super().__init__()

        self.model_wrapper = model_wrapper
        self.n_embeddings = n_embeddings
        self.probe_layer = probe_layer
        self.probe_position = probe_position
        self.probe_type = probe_type
        self.param_dtype = param_dtype

        # Get hidden dimension from model
        self.hidden_dim = model_wrapper.model.config.hidden_size

        # Initialize learnable prompt embeddings in specified dtype
        self.prompt_embeddings = nn.Parameter(
            torch.randn(n_embeddings, self.hidden_dim, dtype=param_dtype) * embedding_init_std
        )

        # Initialize linear probe in specified dtype
        self.probe = nn.Linear(self.hidden_dim, 1, dtype=param_dtype)

        # Freeze base model parameters
        for param in self.model_wrapper.model.parameters():
            param.requires_grad = False

        logger.info(
            f"Initialized PEPModel: n_embeddings={n_embeddings}, "
            f"layer={probe_layer}, position={probe_position}, "
            f"hidden_dim={self.hidden_dim}, probe_type={probe_type}, "
            f"param_dtype={param_dtype}"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_activations: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with embedding injection and probing.

        This method ALWAYS:
        1. Converts input_ids to embeddings
        2. Appends prompt_embeddings at the end
        3. Runs through frozen model
        4. Extracts activation of the LAST token (corresponds to last prompt embedding)
        5. Applies probe to this activation

        The caller is responsible for truncating input_ids to the desired position
        before calling this method.

        Args:
            input_ids: Token IDs, shape (batch_size, seq_len)
            attention_mask: Attention mask, shape (batch_size, seq_len)
            return_activations: If True, return (probe_logits, activations)

        Returns:
            If return_activations=True: (probe_logits, activations)
            If return_activations=False: probe_logits only
            probe_logits shape: (batch_size, 1)
            activations shape: (batch_size, hidden_dim)
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Get token embeddings from model's embedding layer
        embedding_layer = self.model_wrapper.model.get_input_embeddings()
        token_embeds = embedding_layer(input_ids)  # (batch, seq_len, hidden_dim)

        # Append prompt embeddings at the end
        # Shape: (batch, n_embeddings, hidden_dim)
        # Match dtype of token embeddings for model forward pass
        prompt_embeds = self.prompt_embeddings.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        if prompt_embeds.dtype != token_embeds.dtype:
            prompt_embeds = prompt_embeds.to(token_embeds.dtype)

        # Concatenate: [token_embeds, prompt_embeds]
        inputs_embeds = torch.cat([token_embeds, prompt_embeds], dim=1)

        # Update attention mask to include prompt embeddings
        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size, input_ids.shape[1], device=device
            )

        # Extend attention mask for prompt embeddings
        prompt_attention = torch.ones(
            batch_size, self.n_embeddings, device=device
        )
        extended_attention_mask = torch.cat(
            [attention_mask, prompt_attention], dim=1
        )

        # Forward pass through frozen model
        with torch.no_grad():
            outputs = self.model_wrapper.model(
                inputs_embeds=inputs_embeds,
                attention_mask=extended_attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )

        # Extract hidden states from specified layer
        # hidden_states[0] is embedding layer, so layer i is at index i+1
        hidden_states = outputs.hidden_states[self.probe_layer + 1]
        # Shape: (batch, total_seq_len, hidden_dim)
        # where total_seq_len = seq_len + n_embeddings

        # Extract activation of the LAST token (corresponds to last prompt embedding)
        activations = hidden_states[:, -1, :]  # (batch, hidden_dim)

        # Apply probe (convert activations to match probe dtype)
        if activations.dtype != self.param_dtype:
            activations = activations.to(self.param_dtype)
        probe_logits = self.probe(activations)  # (batch, 1)

        if return_activations:
            return probe_logits, activations
        else:
            return probe_logits

    def _truncate_sequence_to_position(
        self,
        prompt_ids: torch.Tensor,
        generated_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Truncate sequence to probe_position before adding embeddings.

        This method handles the logic of where to truncate the sequence based on
        probe_position, similar to extract_activations_reforward in ModelWrapper.

        For TBG (probe_position >= 0): truncates to position in generated part
        For SLT (probe_position < 0): uses full generation, position is relative to end

        Args:
            prompt_ids: Prompt token IDs, shape (prompt_len,)
            generated_ids: Generated token IDs, shape (gen_len,)

        Returns:
            Truncated sequence, shape (truncated_len,)
        """
        prompt_len = prompt_ids.shape[0]
        gen_len = generated_ids.shape[0]

        # Ensure both tensors are on the same device
        if prompt_ids.device != generated_ids.device:
            prompt_ids = prompt_ids.to(generated_ids.device)

        # Convert position to absolute index in generated part
        if self.probe_position >= 0:
            # TBG: Positive position means index from start of generation
            # position=0 means BEFORE any generation (just prompt)
            if self.probe_position == 0:
                # No generation, just prompt
                return prompt_ids
            else:
                # Include tokens up to (but not including) probe_position
                idx = min(self.probe_position - 1, gen_len - 1)
                truncated_gen = generated_ids[:idx + 1]
        else:
            # SLT: Negative position means index from end of generation
            idx = max(0, gen_len + self.probe_position)
            truncated_gen = generated_ids[:idx + 1]

        # Concatenate prompt + truncated generation
        full_sequence = torch.cat([prompt_ids, truncated_gen], dim=0)

        return full_sequence

    def score_with_answer(
        self,
        prompt: str,
        answer: str,
    ) -> float:
        """
        Score a prompt-answer pair (post-generation scoring).

        Tokenizes prompt + answer, truncates to probe_position, injects learnable
        embeddings, extracts activations, and applies probe to predict uncertainty.

        Args:
            prompt: Text prompt
            answer: Generated answer text

        Returns:
            Uncertainty score (0-1), higher = more uncertain/incorrect
        """
        # Tokenize prompt
        inputs = self.model_wrapper.prepare_inputs([prompt])
        prompt_ids = inputs['input_ids'][0]  # (prompt_len,)

        # Tokenize answer
        answer_ids = self.model_wrapper.tokenizer.encode(
            answer,
            add_special_tokens=False,
            return_tensors='pt'
        )[0]  # (answer_len,)

        # Ensure tensors are 1D
        if prompt_ids.dim() == 2:
            prompt_ids = prompt_ids[0]

        device = self.model_wrapper.device
        prompt_ids = prompt_ids.to(device)
        answer_ids = answer_ids.to(device)

        # Truncate sequence to probe_position
        truncated_seq = self._truncate_sequence_to_position(
            prompt_ids, answer_ids
        )

        # Add batch dimension and run forward
        input_ids = truncated_seq.unsqueeze(0)  # (1, seq_len)

        # Forward pass (will append embeddings and extract last token activation)
        with torch.no_grad():
            logits, _ = self.forward(
                input_ids=input_ids,
                return_activations=True
            )
            prob = torch.sigmoid(logits).item()

        # Convert to uncertainty score based on probe type
        if self.probe_type == "accuracy":
            # Probe predicts P(correct), return P(incorrect)
            return 1.0 - prob
        else:  # probe_type == "sep"
            # Probe predicts P(high_SE), return as-is
            return prob

    def score_prompt(
        self,
        prompt: str,
        max_new_tokens: int = 64,
    ) -> float:
        """
        Score a prompt by generating answer first, then scoring.

        Generates complete answer using greedy decoding, then scores it.

        Args:
            prompt: Text prompt
            max_new_tokens: Maximum tokens to generate

        Returns:
            Uncertainty score (0-1), higher = more uncertain/incorrect
        """
        # Generate greedy answer
        answers, _ = self.model_wrapper.generate_greedy(
            prompts=[prompt],
            max_new_tokens=max_new_tokens
        )
        answer = answers[0]

        # Score the generated answer
        return self.score_with_answer(prompt, answer)

    def generate_with_scoring(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        callback: Optional[Callable[[float, str], None]] = None,
        generation_config: Optional[GenerationConfig] = None,
    ) -> Tuple[str, float]:
        """
        Generate answer and compute uncertainty score.

        For SLT setup (probe_position < 0):
            - Generates complete answer
            - Scores using full generation
            - Calls callback with final score

        For TBG setup (probe_position >= 0):
            - Generates until probe_position
            - Computes score at that point
            - Calls callback with early score
            - Continues generation to completion
            - Returns both full answer and early score

        Args:
            prompt: Text prompt
            max_new_tokens: Maximum tokens to generate
            callback: Optional callback function(score, partial_answer) called when score is computed
            generation_config: Optional generation config (max_new_tokens will be overridden)

        Returns:
            Tuple of (generated_text, uncertainty_score)
        """
        # Handle generation config
        if generation_config is not None:
            if generation_config.max_new_tokens != max_new_tokens:
                import warnings
                warnings.warn(
                    f"generation_config.max_new_tokens ({generation_config.max_new_tokens}) "
                    f"will be overridden with max_new_tokens ({max_new_tokens})",
                    UserWarning
                )
            gen_config = generation_config
            # Ensure we always get dict output for consistent handling
            gen_config.return_dict_in_generate = True
        else:
            gen_config = GenerationConfig(
                do_sample=False,
                num_beams=1,
                return_dict_in_generate=True,
                output_logits=False,
            )
        if self.probe_position < 0:
            # SLT setup: generate fully, then score
            gen_config.max_new_tokens = max_new_tokens

            inputs = self.model_wrapper.prepare_inputs([prompt])
            with torch.inference_mode():
                outputs = self.model_wrapper.model.generate(
                    **inputs,
                    generation_config=gen_config
                )

            # Decode answer
            prompt_len = inputs['input_ids'].shape[1]
            answer_ids = outputs.sequences[0, prompt_len:]
            answer = self.model_wrapper.tokenizer.decode(
                answer_ids,
                skip_special_tokens=True
            )

            score = self.score_with_answer(prompt, answer)

            if callback is not None:
                callback(score, answer)

            return answer, score
        else:
            # TBG setup: generate -> score -> continue generation
            inputs = self.model_wrapper.prepare_inputs([prompt])
            prompt_len = inputs['input_ids'].shape[1]

            # Step 1: Generate until probe_position
            if self.probe_position == 0:
                # No generation needed, score immediately on prompt
                partial_seq = inputs['input_ids']
                partial_answer = ""
            else:
                # Generate up to probe_position tokens
                gen_config.max_new_tokens = self.probe_position

                with torch.inference_mode():
                    outputs = self.model_wrapper.model.generate(
                        **inputs,
                        generation_config=gen_config
                    )

                partial_seq = outputs.sequences  # (1, prompt_len + probe_position)
                partial_answer = self.model_wrapper.tokenizer.decode(
                    partial_seq[0, prompt_len:],
                    skip_special_tokens=True
                )

            # Step 2: Compute score using forward pass
            with torch.no_grad():
                logits, _ = self.forward(
                    input_ids=partial_seq,
                    return_activations=True
                )
                prob = torch.sigmoid(logits).item()

            # Convert to uncertainty score
            if self.probe_type == "accuracy":
                score = 1.0 - prob
            else:  # sep
                score = prob

            # Step 3: Call callback with early score
            if callback is not None:
                callback(score, partial_answer)

            # Step 4: Continue generation to completion
            remaining_tokens = max_new_tokens - self.probe_position

            if remaining_tokens > 0:
                # Continue from where we stopped
                gen_config.max_new_tokens = remaining_tokens

                with torch.inference_mode():
                    outputs = self.model_wrapper.model.generate(
                        input_ids=partial_seq,
                        generation_config=gen_config
                    )

                # Decode full answer
                full_answer = self.model_wrapper.tokenizer.decode(
                    outputs.sequences[0, prompt_len:],
                    skip_special_tokens=True
                )
            else:
                full_answer = partial_answer

            return full_answer, score

    def estimate(self, input: ScorerInput) -> float:
        """
        Estimate uncertainty score (ScorerInterface implementation).

        Dispatches to score_prompt() or score_with_answer() based on input type.

        Args:
            input: PromptEmbeddingInput with prompt and optional answer

        Returns:
            Uncertainty score (0-1), higher = more uncertain/incorrect
        """
        if not isinstance(input, PromptEmbeddingInput):
            raise TypeError(
                f"{self.__class__.__name__} requires PromptEmbeddingInput, "
                f"got {type(input).__name__}"
            )

        if input.answer is None:
            # No answer provided, generate and score
            return self.score_prompt(input.prompt)
        else:
            # Answer provided, score directly
            return self.score_with_answer(input.prompt, input.answer)

    def save(self, path: str):
        """
        Save model state to file.

        Args:
            path: Path to save file (.pt or .pth)
        """
        state = {
            'prompt_embeddings': self.prompt_embeddings.data,
            'probe_weight': self.probe.weight.data,
            'probe_bias': self.probe.bias.data,
            'n_embeddings': self.n_embeddings,
            'probe_layer': self.probe_layer,
            'probe_position': self.probe_position,
            'probe_type': self.probe_type,
            'hidden_dim': self.hidden_dim,
        }
        torch.save(state, path)
        logger.info(f"Saved PEPModel to {path}")

    @classmethod
    def load(
        cls,
        path: str,
        model_wrapper: ModelWrapper,
    ) -> 'PEPModel':
        """
        Load model state from file.

        Args:
            path: Path to saved file
            model_wrapper: ModelWrapper to use (must match original)

        Returns:
            Loaded PEPModel instance
        """
        state = torch.load(path, map_location=model_wrapper.device)

        # Create model with saved configuration
        model = cls(
            model_wrapper=model_wrapper,
            n_embeddings=state['n_embeddings'],
            probe_layer=state['probe_layer'],
            probe_position=state['probe_position'],
            probe_type=state['probe_type'],
        )

        # Load parameters
        model.prompt_embeddings.data = state['prompt_embeddings'].to(model_wrapper.device)
        model.probe.weight.data = state['probe_weight'].to(model_wrapper.device)
        model.probe.bias.data = state['probe_bias'].to(model_wrapper.device)

        logger.info(f"Loaded PEPModel from {path}")
        return model
