"""
Tests for PEPModel (Prompt Embedding Probe Model).
"""

import pytest
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.scoring.prompt_embedding import PEPModel
from src.scoring.inputs import PromptEmbeddingInput
from src.model import GPT2Model


@pytest.fixture
def gpt2_wrapper():
    """Load GPT-2 model wrapper for testing."""
    model_name = "gpt2"

    # Use CUDA if available for faster tests
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    wrapper = GPT2Model(model, tokenizer)
    return wrapper


@pytest.fixture
def pep_model(gpt2_wrapper):
    """Create PEPModel instance for testing."""
    model = PEPModel(
        model_wrapper=gpt2_wrapper,
        n_embeddings=1,
        probe_layer=6,  # Middle layer for GPT-2
        probe_position=-2,  # SLT setup
        probe_type="accuracy",
    )
    return model


def test_pep_model_initialization(gpt2_wrapper):
    """Test PEPModel initialization."""
    model = PEPModel(
        model_wrapper=gpt2_wrapper,
        n_embeddings=2,
        probe_layer=8,
        probe_position=0,
        probe_type="sep",
    )

    assert model.n_embeddings == 2
    assert model.probe_layer == 8
    assert model.probe_position == 0
    assert model.probe_type == "sep"
    assert model.hidden_dim == 768  # GPT-2 hidden size

    # Check that embeddings are initialized
    assert model.prompt_embeddings.shape == (2, 768)

    # Check that probe is initialized
    assert model.probe.in_features == 768
    assert model.probe.out_features == 1

    # Check that base model is frozen
    for param in model.model_wrapper.model.parameters():
        assert not param.requires_grad

    # Check that embeddings and probe are trainable
    assert model.prompt_embeddings.requires_grad
    assert model.probe.weight.requires_grad
    assert model.probe.bias.requires_grad


def test_forward_pass(pep_model):
    """Test forward pass with embedding injection."""
    # Create dummy input
    input_ids = torch.randint(0, 1000, (2, 10))  # batch_size=2, seq_len=10

    # Forward pass
    logits, activations = pep_model.forward(input_ids, return_activations=True)

    # Check shapes
    assert logits.shape == (2, 1)
    assert activations.shape == (2, 768)

    # Check that logits are finite
    assert torch.isfinite(logits).all()

    # Test without returning activations
    logits_only = pep_model.forward(input_ids, return_activations=False)
    assert logits_only.shape == (2, 1)
    assert torch.allclose(logits, logits_only)


def test_truncate_sequence_tbg(pep_model):
    """Test sequence truncation for TBG setup."""
    prompt_ids = torch.tensor([1, 2, 3, 4, 5])
    generated_ids = torch.tensor([10, 11, 12, 13, 14])

    # Test position=0 (TBG: no generation, just prompt)
    pep_model.probe_position = 0
    result = pep_model._truncate_sequence_to_position(prompt_ids, generated_ids)
    expected = prompt_ids
    assert torch.equal(result, expected), f"position=0: expected {expected}, got {result}"

    # Test position=1 (first generated token included)
    pep_model.probe_position = 1
    result = pep_model._truncate_sequence_to_position(prompt_ids, generated_ids)
    expected = torch.cat([prompt_ids, generated_ids[:1]], dim=0)  # Prompt + first token
    assert torch.equal(result, expected), f"position=1: expected {expected}, got {result}"

    # Test position=2 (first 2 tokens included)
    pep_model.probe_position = 2
    result = pep_model._truncate_sequence_to_position(prompt_ids, generated_ids)
    expected = torch.cat([prompt_ids, generated_ids[:2]], dim=0)
    assert torch.equal(result, expected), f"position=2: expected {expected}, got {result}"

    # Test position=3 (first 3 tokens included)
    pep_model.probe_position = 3
    result = pep_model._truncate_sequence_to_position(prompt_ids, generated_ids)
    expected = torch.cat([prompt_ids, generated_ids[:3]], dim=0)
    assert torch.equal(result, expected), f"position=3: expected {expected}, got {result}"


def test_truncate_sequence_slt(pep_model):
    """Test sequence truncation for SLT setup."""
    prompt_ids = torch.tensor([1, 2, 3, 4, 5])
    generated_ids = torch.tensor([10, 11, 12, 13, 14])

    # Test position=-1 (last token)
    pep_model.probe_position = -1
    result = pep_model._truncate_sequence_to_position(prompt_ids, generated_ids)
    expected = torch.cat([prompt_ids, generated_ids], dim=0)
    assert torch.equal(result, expected)

    # Test position=-2 (second-to-last)
    pep_model.probe_position = -2
    result = pep_model._truncate_sequence_to_position(prompt_ids, generated_ids)
    expected = torch.cat([prompt_ids, generated_ids[:-1]], dim=0)
    assert torch.equal(result, expected)

    # Test position=-3
    pep_model.probe_position = -3
    result = pep_model._truncate_sequence_to_position(prompt_ids, generated_ids)
    expected = torch.cat([prompt_ids, generated_ids[:-2]], dim=0)
    assert torch.equal(result, expected)


def test_score_with_answer(pep_model):
    """Test scoring a prompt-answer pair."""
    prompt = "What is 2+2?"
    answer = "4"

    score = pep_model.score_with_answer(prompt, answer)

    # Check that score is valid probability
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_score_prompt(pep_model):
    """Test scoring a prompt (with generation)."""
    prompt = "The capital of France is"

    score = pep_model.score_prompt(prompt, max_new_tokens=10)

    # Check that score is valid probability
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_generate_with_scoring_slt(gpt2_wrapper):
    """Test generation with scoring in SLT setup."""
    model = PEPModel(
        model_wrapper=gpt2_wrapper,
        n_embeddings=1,
        probe_layer=6,
        probe_position=-2,  # SLT
        probe_type="accuracy",
    )

    prompt = "The capital of France is"
    answer, score = model.generate_with_scoring(prompt, max_new_tokens=10)

    # Check that answer is non-empty string
    assert isinstance(answer, str)
    assert len(answer) > 0

    # Check that score is valid probability
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_generate_with_scoring_tbg(gpt2_wrapper):
    """Test generation with scoring in TBG setup."""
    model = PEPModel(
        model_wrapper=gpt2_wrapper,
        n_embeddings=1,
        probe_layer=6,
        probe_position=3,  # TBG - score after 3 tokens
        probe_type="accuracy",
    )

    prompt = "The capital of France is"
    answer, score = model.generate_with_scoring(prompt, max_new_tokens=10)

    # Check that answer is non-empty string
    assert isinstance(answer, str)
    assert len(answer) > 0

    # Check that score is valid probability
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_generate_with_scoring_tbg_position_zero(gpt2_wrapper):
    """Test generation with scoring in TBG setup at position 0."""
    model = PEPModel(
        model_wrapper=gpt2_wrapper,
        n_embeddings=1,
        probe_layer=6,
        probe_position=0,  # TBG - score before any generation
        probe_type="accuracy",
    )

    prompt = "The capital of France is"
    answer, score = model.generate_with_scoring(prompt, max_new_tokens=10)

    # Check that answer is non-empty string
    assert isinstance(answer, str)
    assert len(answer) > 0

    # Check that score is valid probability
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_generate_with_scoring_callback(gpt2_wrapper):
    """Test that callback is called during generation."""
    model = PEPModel(
        model_wrapper=gpt2_wrapper,
        n_embeddings=1,
        probe_layer=6,
        probe_position=2,  # TBG
        probe_type="accuracy",
    )

    # Track callback invocations
    callback_data = []

    def test_callback(score, partial_answer):
        callback_data.append({
            'score': score,
            'partial_answer': partial_answer
        })

    prompt = "The capital of France is"
    answer, score = model.generate_with_scoring(
        prompt,
        max_new_tokens=10,
        callback=test_callback
    )

    # Check that callback was called exactly once
    assert len(callback_data) == 1

    # Check callback received valid data
    assert isinstance(callback_data[0]['score'], float)
    assert 0.0 <= callback_data[0]['score'] <= 1.0
    assert isinstance(callback_data[0]['partial_answer'], str)

    # Check that final score matches callback score
    assert score == callback_data[0]['score']


def test_generate_with_scoring_custom_gen_config(gpt2_wrapper):
    """Test generation with custom generation config."""
    from transformers import GenerationConfig

    model = PEPModel(
        model_wrapper=gpt2_wrapper,
        n_embeddings=1,
        probe_layer=6,
        probe_position=-2,
        probe_type="accuracy",
    )

    # Create custom config with different max_new_tokens (should trigger warning)
    custom_config = GenerationConfig(
        do_sample=True,
        temperature=0.7,
        max_new_tokens=5,  # Different from what we'll pass
    )

    prompt = "The capital of France is"

    # Should trigger warning about max_new_tokens override
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        answer, score = model.generate_with_scoring(
            prompt,
            max_new_tokens=10,
            generation_config=custom_config
        )

        # Check that warning was raised
        assert len(w) == 1
        assert "max_new_tokens" in str(w[0].message)

    # Check results are still valid
    assert isinstance(answer, str)
    assert len(answer) > 0
    assert 0.0 <= score <= 1.0


def test_estimate_with_prompt_only(pep_model):
    """Test estimate() with prompt only (generates answer)."""
    input_data = PromptEmbeddingInput(prompt="What is 2+2?")

    score = pep_model.estimate(input_data)

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_estimate_with_answer(pep_model):
    """Test estimate() with prompt and answer."""
    input_data = PromptEmbeddingInput(
        prompt="What is 2+2?",
        answer="4"
    )

    score = pep_model.estimate(input_data)

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_save_and_load(pep_model, gpt2_wrapper, tmp_path):
    """Test saving and loading model."""
    # Modify embeddings and probe
    with torch.no_grad():
        pep_model.prompt_embeddings.fill_(1.0)
        pep_model.probe.weight.fill_(2.0)
        pep_model.probe.bias.fill_(3.0)

    # Save
    save_path = tmp_path / "test_pep_model.pt"
    pep_model.save(str(save_path))

    # Load
    loaded_model = PEPModel.load(str(save_path), gpt2_wrapper)

    # Check that parameters match
    assert torch.allclose(
        loaded_model.prompt_embeddings,
        pep_model.prompt_embeddings
    )
    assert torch.allclose(
        loaded_model.probe.weight,
        pep_model.probe.weight
    )
    assert torch.allclose(
        loaded_model.probe.bias,
        pep_model.probe.bias
    )

    # Check that configuration matches
    assert loaded_model.n_embeddings == pep_model.n_embeddings
    assert loaded_model.probe_layer == pep_model.probe_layer
    assert loaded_model.probe_position == pep_model.probe_position
    assert loaded_model.probe_type == pep_model.probe_type


def test_multiple_embeddings(gpt2_wrapper):
    """Test with multiple prompt embeddings."""
    model = PEPModel(
        model_wrapper=gpt2_wrapper,
        n_embeddings=5,
        probe_layer=6,
        probe_position=-2,
        probe_type="accuracy",
    )

    assert model.prompt_embeddings.shape == (5, 768)

    # Test forward pass
    input_ids = torch.randint(0, 1000, (1, 10))
    logits, activations = model.forward(input_ids)

    assert logits.shape == (1, 1)
    assert activations.shape == (1, 768)


def test_different_layers(gpt2_wrapper):
    """Test probing at different layers."""
    for layer in [0, 6, 11]:  # GPT-2 has 12 layers (0-11)
        model = PEPModel(
            model_wrapper=gpt2_wrapper,
            n_embeddings=1,
            probe_layer=layer,
            probe_position=-2,
            probe_type="accuracy",
        )

        score = model.score_with_answer("test", "answer")
        assert 0.0 <= score <= 1.0
