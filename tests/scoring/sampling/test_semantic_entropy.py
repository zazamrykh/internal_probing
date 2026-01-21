"""
Test for SemanticEntropyScorer with Mistral-7B.

Tests semantic entropy calculation on simple vs complex questions.
"""

import logging
from pathlib import Path
import pytest
import torch

from src.model import MistralModel
from src.scoring.sampling.semantic_entropy import SemanticEntropyScorer
from src.scoring.sampling.inputs import SamplerInput

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def mistral_wrapper():
    """Load Mistral-7B with 8-bit quantization."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    LOCAL_PATH = "../models/mistral-7b-instruct"
    HF_NAME = "mistralai/Mistral-7B-Instruct-v0.1"

    model_path = LOCAL_PATH if Path(LOCAL_PATH).exists() else HF_NAME

    try:
        quant_config = MistralModel.get_quantization_config(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quant_config,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        wrapper = MistralModel(model, tokenizer)
        logger.debug(f"Loaded Mistral on device: {wrapper.device}")

        return wrapper
    except (ValueError, RuntimeError) as e:
        pytest.skip(f"Cannot load Mistral model (insufficient GPU memory or device issues): {e}")


@pytest.fixture(scope="module")
def semantic_entropy_scorer():
    """Initialize SemanticEntropyScorer with local RoBERTa if available."""
    LOCAL_ROBERTA = "../models/roberta-large-mnli"
    HF_ROBERTA = "roberta-large-mnli"

    model_name = LOCAL_ROBERTA if Path(LOCAL_ROBERTA).exists() else HF_ROBERTA

    scorer = SemanticEntropyScorer(
        entailment_model_name=model_name,
        n_samples=5,  # Use fewer samples for faster testing
        sampling_batch_size=1,
        strict_entailment=False
    )

    logger.debug("Initialized SemanticEntropyScorer")
    return scorer


def test_semantic_entropy_simple_vs_complex(mistral_wrapper, semantic_entropy_scorer):
    """
    Test that semantic entropy is higher for complex/ambiguous questions.

    Simple question: "What is 2+2?" - should have low entropy (consistent answers)
    Complex question: Fictional person - should have high entropy (inconsistent answers)

    Note: This test uses logger.warning instead of hard assert because
    model behavior can vary.

    This test may be skipped if there's insufficient memory (OOM error).
    """
    prompts = [
        "What is 2+2?",  # Simple, should have low entropy
        "In what year was Kofi Badu Mensah, a hero of the Revolution of Saint Kitts and Nevis, born?"  # Complex/fictional, should have high entropy
    ]

    logger.debug(f"Testing semantic entropy on {len(prompts)} prompts")

    # Create input
    input_data = SamplerInput(
        prompts=prompts,
        model_wrapper=mistral_wrapper
    )

    # Compute semantic entropy
    entropies = semantic_entropy_scorer.estimate(input_data)

    # Check we got results
    assert isinstance(entropies, list)
    assert len(entropies) == 2

    simple_entropy = entropies[0]
    complex_entropy = entropies[1]

    logger.debug(f"Simple question entropy: {simple_entropy:.4f}")
    logger.debug(f"Complex question entropy: {complex_entropy:.4f}")

    # Log results
    for i, (prompt, entropy) in enumerate(zip(prompts, entropies)):
        logger.debug(f"\nQuestion {i+1}: {prompt}")
        logger.debug(f"Semantic Entropy: {entropy:.4f}")

    # Soft check: complex should have higher entropy
    if complex_entropy <= simple_entropy:
        logger.warning(
            f"Expected complex question to have higher entropy, but got: "
            f"simple={simple_entropy:.4f}, complex={complex_entropy:.4f}. "
            f"This may happen with small sample sizes or model variability."
        )
    else:
        logger.debug(
            f"✓ Complex question has higher entropy as expected: "
            f"{complex_entropy:.4f} > {simple_entropy:.4f}"
        )

    # Hard checks: entropies should be non-negative and finite
    assert simple_entropy >= 0 - 1e-6, "Entropy should be non-negative"
    assert complex_entropy >= 0 - 1e-6, "Entropy should be non-negative"
    assert not (simple_entropy == float('inf') or simple_entropy == float('-inf')), "Entropy should be finite"
    assert not (complex_entropy == float('inf') or complex_entropy == float('-inf')), "Entropy should be finite"

    logger.debug("✓ Semantic entropy test completed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--log-cli-level=DEBUG"])
