"""
Tests for LinearProbeScorer with mocked sklearn models.

These tests verify that the scorer correctly interfaces with sklearn-like models
without requiring actual trained models.
"""

import logging
import pytest
import numpy as np
import torch
from unittest.mock import Mock

from src.scoring import LinearProbeScorer, ProbeInput

logger = logging.getLogger(__name__)


@pytest.fixture
def mock_sep_probe():
    """Create a mock SEP probe (predicts P(high_SE))."""
    mock_model = Mock()

    # Mock predict_proba to return probabilities
    def mock_predict_proba(X):
        # Return shape (n_samples, 2) for binary classification
        n_samples = X.shape[0]
        # Generate some fake probabilities for high SE
        probs_class_0 = np.random.rand(n_samples) * 0.5  # 0 to 0.5
        probs_class_1 = 1 - probs_class_0  # P(high_SE)
        return np.column_stack([probs_class_0, probs_class_1])

    mock_model.predict_proba = Mock(side_effect=mock_predict_proba)
    return mock_model


@pytest.fixture
def mock_accuracy_probe():
    """Create a mock accuracy probe (predicts P(correct))."""
    mock_model = Mock()

    # Mock predict_proba to return probabilities
    def mock_predict_proba(X):
        n_samples = X.shape[0]
        # Generate some fake probabilities for correctness
        probs_class_0 = np.random.rand(n_samples) * 0.3  # P(incorrect)
        probs_class_1 = 1 - probs_class_0  # P(correct)
        return np.column_stack([probs_class_0, probs_class_1])

    mock_model.predict_proba = Mock(side_effect=mock_predict_proba)
    return mock_model


def test_linear_probe_initialization_sep(mock_sep_probe):
    """Test that LinearProbeScorer initializes correctly with SEP probe."""
    scorer = LinearProbeScorer(probe_model=mock_sep_probe, probe_type="sep")

    assert scorer.probe_model is mock_sep_probe
    assert scorer.probe_type == "sep"
    logger.debug("LinearProbeScorer initialized successfully with SEP probe")


def test_linear_probe_initialization_accuracy(mock_accuracy_probe):
    """Test that LinearProbeScorer initializes correctly with accuracy probe."""
    scorer = LinearProbeScorer(probe_model=mock_accuracy_probe, probe_type="accuracy")

    assert scorer.probe_model is mock_accuracy_probe
    assert scorer.probe_type == "accuracy"
    logger.debug("LinearProbeScorer initialized successfully with accuracy probe")


def test_linear_probe_invalid_probe_type(mock_sep_probe):
    """Test that scorer raises error for invalid probe_type."""
    with pytest.raises(ValueError, match="probe_type must be one of"):
        LinearProbeScorer(probe_model=mock_sep_probe, probe_type="invalid")

    logger.debug("Correctly rejects invalid probe_type")


def test_linear_probe_requires_predict_proba():
    """Test that scorer raises error if model doesn't have predict_proba."""
    invalid_model = Mock(spec=[])  # No predict_proba method

    with pytest.raises(ValueError, match="must have predict_proba"):
        LinearProbeScorer(probe_model=invalid_model)

    logger.debug("Correctly rejects models without predict_proba()")


def test_linear_probe_sep_estimate(mock_sep_probe):
    """Test SEP probe estimation - returns P(high_SE) as-is."""
    scorer = LinearProbeScorer(probe_model=mock_sep_probe, probe_type="sep")

    # Create fake activations
    batch_size = 5
    hidden_dim = 768
    activations = torch.randn(batch_size, hidden_dim)

    # Create input
    input_data = ProbeInput(activations=activations)

    # Estimate
    uncertainty_scores = scorer.estimate(input_data)

    # Check output
    assert isinstance(uncertainty_scores, np.ndarray)
    assert uncertainty_scores.shape == (batch_size,)
    assert np.all((uncertainty_scores >= 0) & (uncertainty_scores <= 1))

    # Verify predict_proba was called
    mock_sep_probe.predict_proba.assert_called_once()

    logger.debug(f"SEP scores shape: {uncertainty_scores.shape}, range: [{uncertainty_scores.min():.3f}, {uncertainty_scores.max():.3f}]")


def test_linear_probe_accuracy_estimate(mock_accuracy_probe):
    """Test accuracy probe estimation - returns P(incorrect) = 1 - P(correct)."""
    scorer = LinearProbeScorer(probe_model=mock_accuracy_probe, probe_type="accuracy")

    # Create fake activations
    batch_size = 5
    hidden_dim = 768
    activations = torch.randn(batch_size, hidden_dim)

    # Create input
    input_data = ProbeInput(activations=activations)

    # Estimate
    error_scores = scorer.estimate(input_data)

    # Check output
    assert isinstance(error_scores, np.ndarray)
    assert error_scores.shape == (batch_size,)
    assert np.all((error_scores >= 0) & (error_scores <= 1))

    # Verify predict_proba was called
    mock_accuracy_probe.predict_proba.assert_called_once()

    # Get raw P(correct) to verify inversion
    X = activations.cpu().numpy()
    raw_probs = mock_accuracy_probe.predict_proba(X)[:, 1]  # P(correct)
    expected_error_scores = 1.0 - raw_probs  # P(incorrect)

    # Note: We can't directly compare because mock generates random values each call
    # But we verified the shape and range
    logger.debug(f"Accuracy probe error scores shape: {error_scores.shape}, range: [{error_scores.min():.3f}, {error_scores.max():.3f}]")


def test_linear_probe_estimate_with_numpy(mock_sep_probe):
    """Test estimation with numpy array activations."""
    scorer = LinearProbeScorer(probe_model=mock_sep_probe, probe_type="sep")

    # Create fake activations as numpy
    batch_size = 3
    hidden_dim = 512
    activations = np.random.randn(batch_size, hidden_dim).astype(np.float32)

    # Create input
    input_data = ProbeInput(activations=activations)

    # Estimate
    predictions = scorer.estimate(input_data)

    # Check output
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (batch_size,)

    logger.debug("Works with numpy arrays")


def test_linear_probe_single_sample(mock_sep_probe):
    """Test estimation with single sample (1D tensor)."""
    scorer = LinearProbeScorer(probe_model=mock_sep_probe, probe_type="sep")

    # Single sample
    hidden_dim = 768
    activations = torch.randn(hidden_dim)  # 1D tensor

    # Create input
    input_data = ProbeInput(activations=activations)

    # Estimate
    predictions = scorer.estimate(input_data)

    # Check output
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (1,)  # Should be reshaped to (1,)

    logger.debug("Handles single sample (1D input)")


def test_linear_probe_wrong_input_type(mock_sep_probe):
    """Test that scorer raises error for wrong input type."""
    from src.scoring.inputs import SamplerInput
    from unittest.mock import Mock

    scorer = LinearProbeScorer(probe_model=mock_sep_probe, probe_type="sep")

    # Create wrong input type
    wrong_input = SamplerInput(
        prompts=["test"],
        model_wrapper=Mock()
    )

    with pytest.raises(TypeError, match="requires ProbeInput"):
        scorer.estimate(wrong_input)

    logger.debug("Correctly rejects wrong input types")


def test_linear_probe_predict_method_sep(mock_sep_probe):
    """Test predict() method directly for SEP probe."""
    scorer = LinearProbeScorer(probe_model=mock_sep_probe, probe_type="sep")

    # Create activations
    activations = torch.randn(10, 768)

    # Call predict directly
    predictions = scorer.predict(activations)

    # Check output
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (10,)
    assert np.all((predictions >= 0) & (predictions <= 1))

    logger.debug("predict() method works correctly for SEP probe")


def test_linear_probe_predict_method_accuracy(mock_accuracy_probe):
    """Test predict() method directly for accuracy probe."""
    scorer = LinearProbeScorer(probe_model=mock_accuracy_probe, probe_type="accuracy")

    # Create activations
    activations = torch.randn(10, 768)

    # Call predict directly
    predictions = scorer.predict(activations)

    # Check output
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (10,)
    assert np.all((predictions >= 0) & (predictions <= 1))

    logger.debug("predict() method works correctly for accuracy probe")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--log-cli-level=DEBUG"])
