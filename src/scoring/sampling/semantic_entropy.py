"""
Semantic Entropy scorer for uncertainty estimation.

Based on the paper "Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation
in Natural Language Generation" (Kuhn et al., 2023).

Original implementation: https://github.com/jlko/semantic_uncertainty
Functions adapted from external/semantic_uncertainty/semantic_uncertainty/uncertainty/uncertainty_measures/semantic_entropy.py
"""

import logging
from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, GenerationConfig

from src.scoring.sampling.base import SamplerInterface
from src.scoring.sampling.inputs import SamplerInput

logger = logging.getLogger(__name__)


class EntailmentRoBERTa:
    """
    RoBERTa-based entailment model for semantic clustering.

    Uses RoBERTa fine-tuned on MNLI to check if two texts entail each other.
    Adapted from original semantic_uncertainty repository.
    """

    def __init__(self, model_name: str = "roberta-large-mnli"):
        """
        Initialize entailment model.

        Tries to load from local path first, falls back to HuggingFace if not found.

        Args:
            model_name: HuggingFace model name or local path
        """
        # Check if it's a local path
        local_path = Path(model_name)
        if not local_path.is_absolute():
            # Try relative path from project root
            local_path = Path("../models") / model_name

        if local_path.exists():
            model_path = str(local_path)
            logger.info(f"Loading entailment model from local path: {model_path}")
        else:
            model_path = model_name
            logger.info(f"Loading entailment model from HuggingFace: {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.device = self.model.device
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        # MNLI labels: 0=contradiction, 1=neutral, 2=entailment
        self.entail_idx = 2
        self.contra_idx = 0

        logger.info(f"Entailment model loaded on device: {self.device}")

    def check_implication(
        self,
        premise: str,
        hypothesis: str,
        example: Optional[dict] = None
    ) -> int:
        """
        Check if premise entails hypothesis.

        Args:
            premise: First text
            hypothesis: Second text
            example: Optional dict with 'question' key (for compatibility)

        Returns:
            Label: 0=contradiction, 1=neutral, 2=entailment
        """
        inputs = self.tokenizer(
            premise, hypothesis,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = F.softmax(logits, dim=-1)[0]
        label_id = int(torch.argmax(probs).item())

        return label_id


def get_semantic_ids(strings_list, model, strict_entailment=False, example=None):
    """
    Group list of predictions into semantic meaning.

    Copied from original semantic_uncertainty repository with minimal changes.
    Source: external/semantic_uncertainty/semantic_uncertainty/uncertainty/uncertainty_measures/semantic_entropy.py
    """

    def are_equivalent(text1, text2):

        implication_1 = model.check_implication(text1, text2, example=example)
        implication_2 = model.check_implication(text2, text1, example=example)  # pylint: disable=arguments-out-of-order
        assert (implication_1 in [0, 1, 2]) and (implication_2 in [0, 1, 2])

        if strict_entailment:
            semantically_equivalent = (implication_1 == 2) and (implication_2 == 2)

        else:
            implications = [implication_1, implication_2]
            # Check if none of the implications are 0 (contradiction) and not both of them are neutral.
            semantically_equivalent = (0 not in implications) and ([1, 1] != implications)

        return semantically_equivalent

    # Initialise all ids with -1.
    semantic_set_ids = [-1] * len(strings_list)
    # Keep track of current id.
    next_id = 0
    for i, string1 in enumerate(strings_list):
        # Check if string1 already has an id assigned.
        if semantic_set_ids[i] == -1:
            # If string1 has not been assigned an id, assign it next_id.
            semantic_set_ids[i] = next_id
            for j in range(i+1, len(strings_list)):
                # Search through all remaining strings. If they are equivalent to string1, assign them the same id.
                if are_equivalent(string1, strings_list[j]):
                    semantic_set_ids[j] = next_id
            next_id += 1

    assert -1 not in semantic_set_ids

    return semantic_set_ids


def logsumexp_by_id(semantic_ids, log_likelihoods, agg='sum_normalized'):
    """
    Sum probabilities with the same semantic id.

    Log-Sum-Exp because input and output probabilities in log space.

    Copied from original semantic_uncertainty repository with minimal changes.
    Source: external/semantic_uncertainty/semantic_uncertainty/uncertainty/uncertainty_measures/semantic_entropy.py
    """
    unique_ids = sorted(list(set(semantic_ids)))
    assert unique_ids == list(range(len(unique_ids)))
    log_likelihood_per_semantic_id = []

    for uid in unique_ids:
        # Find positions in `semantic_ids` which belong to the active `uid`.
        id_indices = [pos for pos, x in enumerate(semantic_ids) if x == uid]
        # Gather log likelihoods at these indices.
        id_log_likelihoods = [log_likelihoods[i] for i in id_indices]
        if agg == 'sum_normalized':
            # log_lik_norm = id_log_likelihoods - np.prod(log_likelihoods)
            log_lik_norm = id_log_likelihoods - np.log(np.sum(np.exp(log_likelihoods)))  # p(cluster) / p(total) = sum p(sen from cluster) / p(total)
            logsumexp_value = np.log(np.sum(np.exp(log_lik_norm)))
        else:
            raise ValueError
        log_likelihood_per_semantic_id.append(logsumexp_value)

    return log_likelihood_per_semantic_id


def predictive_entropy_rao(log_probs):
    """
    Compute predictive entropy from log probabilities.

    Copied from original semantic_uncertainty repository.
    Source: external/semantic_uncertainty/semantic_uncertainty/uncertainty/uncertainty_measures/semantic_entropy.py
    """
    entropy = -np.sum(np.exp(log_probs) * log_probs)
    return entropy


class SemanticEntropyScorer(SamplerInterface):
    """
    Semantic Entropy scorer for uncertainty estimation.

    Generates multiple responses, clusters them semantically using an entailment model,
    and computes entropy over the semantic clusters.

    High entropy indicates high uncertainty (many different semantic meanings).
    Low entropy indicates low uncertainty (responses agree semantically).

    Example:
        >>> from src.model import GPT2Model
        >>> from src.scoring.sampling.inputs import SamplerInput
        >>>
        >>> # Initialize
        >>> scorer = SemanticEntropyScorer(
        ...     entailment_model_name="roberta-large-mnli",
        ...     n_samples=10
        ... )
        >>>
        >>> # Score
        >>> input = SamplerInput(prompts=["What is 2+2?"], model_wrapper=model_wrapper)
        >>> entropy = scorer.estimate(input)
    """

    def __init__(
        self,
        entailment_model_name: str = "roberta-large-mnli",
        n_samples: int = 10,
        sampling_batch_size: int = 1,
        generation_config: Optional[GenerationConfig] = None,
        strict_entailment: bool = False,
    ):
        """
        Initialize Semantic Entropy scorer.

        Args:
            entailment_model_name: HuggingFace model name or local path for entailment model
            n_samples: Number of responses to generate per prompt
            sampling_batch_size: Batch size for generation
            generation_config: Generation configuration (if None, uses default)
            strict_entailment: If True, require strict bidirectional entailment for clustering
        """
        super().__init__(n_samples, sampling_batch_size, generation_config)

        self.entailment_model = EntailmentRoBERTa(entailment_model_name)
        self.strict_entailment = strict_entailment

        logger.info(
            f"Initialized SemanticEntropyScorer: n_samples={n_samples}, "
            f"strict_entailment={strict_entailment}"
        )

    def estimate(
        self,
        input: SamplerInput,
        return_details: bool = False
    ) -> Union[float, List[float], dict, List[dict]]:
        """
        Estimate semantic entropy for given prompts.

        Args:
            input: SamplerInput containing prompts and model_wrapper
            return_details: If True, return dict with entropy, samples, and cluster IDs

        Returns:
            If return_details=False:
                Semantic entropy value(s) - float for single prompt, list for multiple
            If return_details=True:
                Dict (or list of dicts) with keys:
                - 'entropy': float - semantic entropy value
                - 'sampling_answers': list[str] - generated responses
                - 'semantic_ids': list[int] - cluster ID for each response
        """
        if not isinstance(input, SamplerInput):
            raise TypeError(
                f"{self.__class__.__name__} requires SamplerInput, "
                f"got {type(input).__name__}"
            )

        results = []

        # Process each prompt
        for prompt in input.prompts:
            # Generate samples with log probabilities
            prompt_samples = []

            for batch_start in range(0, self.n_samples, self.sampling_batch_size):
                batch_end = min(batch_start + self.sampling_batch_size, self.n_samples)
                batch_size = batch_end - batch_start
                batch_prompts = [prompt] * batch_size

                # Generate with log probs
                texts, log_probs_list = input.model_wrapper.generate(
                    prompts=batch_prompts,
                    generation_config=self.generation_config,
                    return_log_probs=True,
                    only_generated=True
                )

                # Compute average log prob for each generation
                for text, log_probs in zip(texts, log_probs_list):
                    avg_log_prob = log_probs.mean().item() if len(log_probs) > 0 else float('-inf')
                    prompt_samples.append((text, avg_log_prob))

            # Separate texts and log probs
            texts = [text for text, _ in prompt_samples]
            avg_log_probs = np.array([lp for _, lp in prompt_samples])

            # Get semantic clusters
            example = {"question": prompt}
            semantic_ids = get_semantic_ids(
                texts,
                self.entailment_model,
                strict_entailment=self.strict_entailment,
                example=example
            )

            logger.debug(
                f"Clustered {len(texts)} responses into {len(set(semantic_ids))} semantic clusters"
            )

            # Compute log probability per cluster
            logp_per_cluster = logsumexp_by_id(semantic_ids, avg_log_probs, agg='sum_normalized')

            # Compute entropy
            entropy = predictive_entropy_rao(np.array(logp_per_cluster))

            # Store result
            if return_details:
                results.append({
                    'entropy': float(entropy),
                    'sampling_answers': texts,
                    'semantic_ids': semantic_ids
                })
            else:
                results.append(entropy)

        # Return single value/dict if only one prompt, otherwise list
        return results[0] if len(results) == 1 else results
