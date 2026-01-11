"""
Tests for ModelWrapper base class and GPT2Model implementation.

These tests verify:
1. Methods work without errors
2. Generated outputs are reasonable
3. Activation extraction works correctly
"""

import logging
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.model import GPT2Model

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def gpt2_wrapper():
    """
    Load GPT-2 model once for all tests.

    Using smallest GPT-2 model for fast testing.
    """
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Set padding token
    tokenizer.pad_token = tokenizer.eos_token

    wrapper = GPT2Model(
        model=model,
        tokenizer=tokenizer,
        system_prompt="You are a helpful assistant. Answer concisely."
    )

    return wrapper


def test_gpt2_initialization(gpt2_wrapper):
    """Test that GPT2Model initializes correctly."""
    assert gpt2_wrapper.model is not None
    assert gpt2_wrapper.tokenizer is not None
    assert gpt2_wrapper.system_prompt is not None
    assert gpt2_wrapper.device is not None
    logger.debug(f"GPT2Model initialized on device: {gpt2_wrapper.device}")


def test_prepare_inputs(gpt2_wrapper):
    """Test input preparation with system prompt."""
    prompts = ["What is 2+2?", "What is the capital of France?"]

    inputs = gpt2_wrapper.prepare_inputs(prompts)

    # Check output structure
    assert "input_ids" in inputs
    assert "attention_mask" in inputs
    assert inputs["input_ids"].shape[0] == 2  # batch size
    assert inputs["input_ids"].device == gpt2_wrapper.device

    # Check that system prompt was prepended
    decoded = gpt2_wrapper.tokenizer.decode(inputs["input_ids"][0])
    assert "helpful assistant" in decoded.lower()

    logger.debug(f"Prepared inputs shape: {inputs['input_ids'].shape}")
    logger.debug(f"First prompt with system prompt:\n{decoded[:100]}...")


def test_generate_greedy_two_questions(gpt2_wrapper):
    """
    Test greedy generation on two questions with log probabilities.

    This is the main test - verifies generation works and outputs are reasonable.
    """
    prompts = [
        "What is 2+2?",
        "What is the capital of France?"
    ]

    texts, log_probs = gpt2_wrapper.generate_greedy(
        prompts,
        max_new_tokens=20
    )

    # Check outputs
    assert len(texts) == 2
    assert len(log_probs) == 2

    # Check each generation
    for i, (text, lp) in enumerate(zip(texts, log_probs)):
        assert isinstance(text, str)
        assert len(text) > 0
        assert isinstance(lp, torch.Tensor)
        assert lp.shape[0] > 0  # Has some tokens

        logger.debug(f"{'='*60}")
        logger.debug(f"Question {i+1}: {prompts[i]}")
        logger.debug(f"Answer: {text}")
        logger.debug(f"Log probs shape: {lp.shape}")
        logger.debug(f"Mean log prob: {lp.mean().item():.3f}")
        logger.debug(f"First 5 log probs: {lp[:5].tolist()}")

    # Sanity check: log probs should be negative
    for lp in log_probs:
        assert (lp <= 0).all(), "Log probabilities should be <= 0"

    logger.debug(f"Generated {len(texts)} responses successfully")


def test_extract_activations_reforward(gpt2_wrapper):
    """
    Test activation extraction on generated text.

    Uses output from generation to extract activations at specific layers/positions.
    """
    prompt = "What is 2+2?"

    # First generate
    texts, log_probs = gpt2_wrapper.generate_greedy(
        [prompt],
        max_new_tokens=15
    )

    text = texts[0]
    logger.debug(f"Generated text: {text}")

    # Get prompt and generated IDs for activation extraction
    prompt_inputs = gpt2_wrapper.prepare_inputs([prompt])
    prompt_ids = prompt_inputs["input_ids"][0]  # (prompt_len,)

    # Tokenize generated text to get IDs
    generated_ids = gpt2_wrapper.tokenizer.encode(
        text,
        add_special_tokens=False,
        return_tensors="pt"
    )[0]

    logger.debug(f"Prompt length: {len(prompt_ids)}, Generated length: {len(generated_ids)}")

    # Extract activations from layers 0, 6, 11 (GPT-2 has 12 layers)
    # At positions 0 (first token) and -2 (second-to-last token)
    layers = [0, 6, 11]
    positions = [0, -2]

    activations = gpt2_wrapper.extract_activations_reforward(
        prompt_ids=prompt_ids,
        generated_ids=generated_ids,
        layers=layers,
        positions=positions
    )

    # Check structure
    assert "positions" in activations
    assert "layers" in activations
    assert "acts" in activations
    assert "gen_len" in activations

    assert activations["positions"] == positions
    assert activations["layers"] == layers
    assert activations["gen_len"] == len(generated_ids)

    # Check activations for each position and layer
    for pos in positions:
        assert pos in activations["acts"]
        for layer in layers:
            assert layer in activations["acts"][pos]
            act = activations["acts"][pos][layer]

            # Check it's a tensor on CPU with correct shape
            assert isinstance(act, torch.Tensor)
            assert act.device.type == "cpu"
            assert act.dim() == 1  # (hidden_dim,)
            assert act.shape[0] == 768  # GPT-2 hidden size

            logger.debug(
                f"Activation at pos={pos}, layer={layer}: shape={act.shape}, "
                f"mean={act.mean().item():.3f}, std={act.std().item():.3f}"
            )

    logger.debug(f"Extracted activations for {len(positions)} positions × {len(layers)} layers")


def test_generate_with_return_ids(gpt2_wrapper):
    """Test generation returning token IDs instead of text."""
    prompts = ["Hello"]

    # Use return_ids mode
    from transformers import GenerationConfig
    gen_config = GenerationConfig(
        do_sample=False,
        max_new_tokens=5,
        return_dict_in_generate=True,
        output_logits=True,
    )

    ids, log_probs = gpt2_wrapper.generate(
        prompts,
        generation_config=gen_config,
        return_ids=True,
        only_generated=True
    )

    assert isinstance(ids[0], torch.Tensor)
    assert ids[0].dim() == 1
    logger.debug(f"Generated token IDs: {ids[0].tolist()}")


def test_activation_extraction_edge_cases(gpt2_wrapper):
    """Test activation extraction with edge cases."""
    prompt = "Hi"
    texts, _ = gpt2_wrapper.generate_greedy([prompt], max_new_tokens=3)

    prompt_inputs = gpt2_wrapper.prepare_inputs([prompt])
    prompt_ids = prompt_inputs["input_ids"][0]
    generated_ids = gpt2_wrapper.tokenizer.encode(
        texts[0], add_special_tokens=False, return_tensors="pt"
    )[0]

    # Test with negative position
    acts = gpt2_wrapper.extract_activations_reforward(
        prompt_ids, generated_ids, layers=[0], positions=[-1]
    )
    assert -1 in acts["acts"]

    # Test with invalid position should raise error
    with pytest.raises(ValueError, match="out of range"):
        gpt2_wrapper.extract_activations_reforward(
            prompt_ids, generated_ids, layers=[0], positions=[100]
        )

    logger.debug("Edge cases handled correctly")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s", "--log-cli-level=DEBUG"])
