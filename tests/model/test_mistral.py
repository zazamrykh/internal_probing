"""
Test for MistralModel wrapper.

Simple test to verify generation and activation extraction work correctly.
"""

import logging
import os
from pathlib import Path
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.model import MistralModel

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def mistral_wrapper():
    """
    Load Mistral-7B-Instruct model with 8-bit quantization.

    Tries to load from local path first, falls back to HuggingFace if not found.
    Note: This requires ~7GB VRAM with quantization.
    """
    LOCAL_MODEL_PATH = "../models/mistral-7b-instruct"
    HF_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"

    # Check if local model exists
    if Path(LOCAL_MODEL_PATH).exists():
        model_path = LOCAL_MODEL_PATH
        logger.debug(f"Using local model from: {LOCAL_MODEL_PATH}")
    else:
        model_path = HF_MODEL_NAME
        logger.debug(f"Local model not found, downloading from HuggingFace: {HF_MODEL_NAME}")

    # Load with 8-bit quantization
    quant_config = MistralModel.get_quantization_config(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quant_config,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    wrapper = MistralModel(
        model=model,
        tokenizer=tokenizer,
        system_prompt="Answer as short as possible"
    )

    logger.debug(f"Loaded Mistral model on device: {wrapper.device}")

    return wrapper


def test_mistral_generation_and_activation_extraction(mistral_wrapper):
    """
    Test Mistral generation and activation extraction.

    Tests:
    1. Generate answers for 2 questions with log probs
    2. Extract activations for first question
    """
    # Two questions: one simple, one complex/impossible
    prompts = [
        "What is capital of France?",
        "In what year was Kofi Badu Mensah, a hero of the Revolution of Saint Kitts and Nevis, born?"
    ]

    logger.debug(f"Generating answers for {len(prompts)} questions")

    # Generate with log probabilities
    texts, log_probs = mistral_wrapper.generate(
        prompts,
        generation_config=None,  # Use default greedy
        return_log_probs=True,
        only_generated=True
    )

    # Basic checks
    assert len(texts) == 2
    assert len(log_probs) == 2

    for i, (text, lp) in enumerate(zip(texts, log_probs)):
        assert isinstance(text, str)
        assert len(text) > 0
        assert isinstance(lp, torch.Tensor)
        assert (lp <= 0).all()  # Log probs should be negative

        logger.debug(f"Question {i+1}: {prompts[i]}")
        logger.debug(f"Answer: {text}")
        logger.debug(f"Log probs: shape={lp.shape}, mean={lp.mean().item():.3f}")

    # Extract activations for first prompt
    logger.debug("\nExtracting activations for first question")

    # Get prompt IDs
    prompt_inputs = mistral_wrapper.prepare_inputs([prompts[0]])
    prompt_ids = prompt_inputs["input_ids"][0]

    # Get generated IDs
    generated_ids = mistral_wrapper.tokenizer.encode(
        texts[0],
        add_special_tokens=False,
        return_tensors="pt"
    )[0]

    logger.debug(f"Prompt length: {len(prompt_ids)}, Generated length: {len(generated_ids)}")

    # Extract activations from a few layers
    # Mistral-7B has 32 layers (0-31)
    layers = [0, 15, 31]
    positions = [0, -2]  # First and second-to-last token

    activations = mistral_wrapper.extract_activations_reforward(
        prompt_ids=prompt_ids,
        generated_ids=generated_ids,
        layers=layers,
        positions=positions
    )

    # Check structure
    assert activations["positions"] == positions
    assert activations["layers"] == layers
    assert activations["gen_len"] == len(generated_ids)

    # Check activations exist and have correct shape
    for pos in positions:
        for layer in layers:
            act = activations["acts"][pos][layer]
            assert isinstance(act, torch.Tensor)
            assert act.device.type == "cpu"
            assert act.dim() == 1
            # Mistral-7B has hidden size 4096
            assert act.shape[0] == 4096

            logger.debug(
                f"Activation pos={pos}, layer={layer}: "
                f"shape={act.shape}, mean={act.mean().item():.3f}"
            )

    logger.debug("Mistral generation and activation extraction successful")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--log-cli-level=DEBUG"])
