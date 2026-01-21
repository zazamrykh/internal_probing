"""
Tests for PEPTrainer - Prompt Embedding Probe training.

Verifies that PEPTrainer can:
1. Train PEPModel on toy dataset
2. Make predictions with trained model
3. Handle sample weights correctly
4. Save and load trained models
"""

import pytest
import numpy as np
import torch
import tempfile
import logging
from pathlib import Path

from src.dataset.base import BaseDataset
from src.model import GPT2Model
from src.training import PEPTrainer
from src.scoring.prompt_embedding import PEPModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class ToyDataset(BaseDataset):
    """Toy dataset for testing."""

    def evaluate_correctness(self, prompt, generated_answer, gt_answers):
        """Dummy evaluation - check if answer contains any ground truth."""
        for gt in gt_answers:
            if gt.lower() in generated_answer.lower():
                return 1.0
        return 0.0


def create_toy_pep_dataset(n_samples: int = 10, seed: int = 42) -> ToyDataset:
    """
    Create toy dataset for PEP training.

    Each sample has:
    - prompt: Question
    - gt_answers: Ground truth answers
    - greedy_answer: Generated answer
    - is_correct: Binary correctness label
    - se_binary: Binary semantic entropy label (optional)
    - se_weight: Sample weight for SE (optional)
    """
    rng = np.random.RandomState(seed)

    data = []
    for i in range(n_samples):
        # Create simple math questions
        a, b = rng.randint(1, 10), rng.randint(1, 10)
        correct_answer = str(a + b)

        # Sometimes generate correct, sometimes incorrect answer
        is_correct = rng.rand() > 0.5
        if is_correct:
            greedy_answer = correct_answer
        else:
            greedy_answer = str(a + b + rng.randint(1, 5))

        # SE labels (correlated with correctness but not perfectly)
        se_binary = 1 if (rng.rand() > 0.6 and not is_correct) else 0
        se_weight = rng.uniform(0.5, 1.5)

        data.append({
            'prompt': f'What is {a} + {b}?',
            'gt_answers': [correct_answer],
            'greedy_answer': greedy_answer,
            'is_correct': float(is_correct),
            'se_binary': float(se_binary),
            'se_weight': se_weight,
        })

    return ToyDataset(data=data, name='toy_pep_dataset')


@pytest.fixture(scope="module")
def gpt2_wrapper():
    """Load GPT-2 model for testing."""
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    wrapper = GPT2Model(
        model=model,
        tokenizer=tokenizer,
        system_prompt=""
    )

    return wrapper


@pytest.fixture
def toy_datasets():
    """Create train/val/test toy datasets."""
    train_data = create_toy_pep_dataset(n_samples=20, seed=42)
    val_data = create_toy_pep_dataset(n_samples=5, seed=142)
    test_data = create_toy_pep_dataset(n_samples=5, seed=242)

    return train_data, val_data, test_data


@pytest.fixture
def pep_trainer(gpt2_wrapper):
    """Create PEPTrainer instance."""
    return PEPTrainer(
        model_wrapper=gpt2_wrapper,
        n_embeddings=1,
        learning_rate=1e-2,
        n_epochs=2,  # Few epochs for fast testing
        batch_size=4,
        seed=42
    )


def test_pep_trainer_initialization(pep_trainer, gpt2_wrapper):
    """Test PEPTrainer initializes correctly."""
    assert pep_trainer.model_wrapper is gpt2_wrapper
    assert pep_trainer.n_embeddings == 1
    assert pep_trainer.learning_rate == 1e-2
    assert pep_trainer.n_epochs == 2
    assert pep_trainer.batch_size == 4
    logger.info("PEPTrainer initialized successfully")


def test_fit_basic(pep_trainer, toy_datasets):
    """Test basic training on toy dataset."""
    train_data, _, _ = toy_datasets

    # Train model
    model, training_info = pep_trainer.fit(
        train_data,
        position=-2,  # SLT setup
        layer=6,
        target_field='is_correct',
        verbose=False
    )

    # Check model is trained
    assert isinstance(model, PEPModel)
    assert model.prompt_embeddings.requires_grad
    assert model.probe_layer == 6
    assert model.probe_position == -2

    # Check training info
    assert isinstance(training_info, dict)
    assert 'total_iterations' in training_info
    assert 'training_time' in training_info

    logger.info(f"Trained PEPModel: embeddings shape={model.prompt_embeddings.shape}")


def test_fit_with_weights(pep_trainer, toy_datasets):
    """Test training with sample weights."""
    train_data, _, _ = toy_datasets

    # Train with weights
    model, training_info = pep_trainer.fit(
        train_data,
        position=-2,
        layer=6,
        target_field='se_binary',
        weight_field='se_weight',
        verbose=False
    )

    assert isinstance(model, PEPModel)
    assert model.probe_type == "sep"  # Should detect from target_field
    assert isinstance(training_info, dict)
    logger.info("Training with sample weights successful")


def test_predict(pep_trainer, toy_datasets):
    """Test making predictions with trained model."""
    train_data, _, test_data = toy_datasets

    # Train model
    model, _ = pep_trainer.fit(
        train_data,
        position=-2,
        layer=6,
        target_field='is_correct',
        verbose=False
    )

    # Get predictions as array
    predictions = pep_trainer.predict(
        model, test_data,
        position=-2,
        layer=6,
        add_to_dataset=False
    )

    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (len(test_data),)
    assert np.all((predictions >= 0) & (predictions <= 1))

    logger.info(f"Predictions: {predictions}")
    logger.info(f"Mean prediction: {predictions.mean():.3f}")


def test_predict_add_to_dataset(pep_trainer, toy_datasets):
    """Test adding predictions to dataset."""
    train_data, _, test_data = toy_datasets

    # Train model
    model, _ = pep_trainer.fit(
        train_data,
        position=-2,
        layer=6,
        target_field='is_correct',
        verbose=False
    )

    # Add predictions to dataset
    test_with_preds = pep_trainer.predict(
        model, test_data,
        position=-2,
        layer=6,
        add_to_dataset=True,
        prediction_field='pep_score'
    )

    # Check predictions were added
    assert 'pep_score' in test_with_preds[0]
    assert len(test_with_preds) == len(test_data)

    logger.info("Predictions added to dataset successfully")


def test_train_cv(pep_trainer, toy_datasets):
    """Test training with cross-validation."""
    train_data, val_data, test_data = toy_datasets

    # Train with CV
    model, metrics = pep_trainer.train_cv(
        train_data, val_data, test_data,
        position=-2,
        layer=6,
        target_field='is_correct',
        k_folds=2,  # Small for fast testing
        compute_metrics=True
    )

    # Check model
    assert isinstance(model, PEPModel)

    # Check metrics
    assert 'position' in metrics
    assert 'layer' in metrics
    assert 'target' in metrics
    assert 'test_auc' in metrics
    assert 'test_logloss' in metrics

    assert metrics['position'] == -2
    assert metrics['layer'] == 6
    assert metrics['target'] == 'is_correct'

    logger.info(f"CV training complete. Test AUC: {metrics.get('test_auc', 'N/A')}")


def test_save_and_load_model(pep_trainer, toy_datasets):
    """Test saving and loading trained PEPModel."""
    train_data, _, test_data = toy_datasets

    # Train model
    model, _ = pep_trainer.fit(
        train_data,
        position=-2,
        layer=6,
        target_field='is_correct',
        verbose=False
    )

    # Get predictions before saving
    preds_before = pep_trainer.predict(
        model, test_data,
        position=-2, layer=6,
        add_to_dataset=False
    )

    # Save model
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / 'test_pep_model'
        model.save(model_path)

        # Load model
        loaded_model = PEPModel.load(model_path, pep_trainer.model_wrapper)

        # Get predictions after loading
        preds_after = pep_trainer.predict(
            loaded_model, test_data,
            position=-2, layer=6,
            add_to_dataset=False
        )

        # Check predictions are the same
        np.testing.assert_array_almost_equal(preds_before, preds_after, decimal=5)

        logger.info("Model save/load successful, predictions match")


def test_different_positions(pep_trainer, toy_datasets):
    """Test training with different probe positions."""
    train_data, _, _ = toy_datasets

    # Test TBG setup (position 0)
    model_tbg, _ = pep_trainer.fit(
        train_data,
        position=0,
        layer=6,
        target_field='is_correct',
        verbose=False
    )
    assert model_tbg.probe_position == 0

    # Test SLT setup (position -2)
    model_slt, _ = pep_trainer.fit(
        train_data,
        position=-2,
        layer=6,
        target_field='is_correct',
        verbose=False
    )
    assert model_slt.probe_position == -2

    logger.info("Training with different positions successful")


def test_different_targets(pep_trainer, toy_datasets):
    """Test training with different target fields."""
    train_data, _, _ = toy_datasets

    # Train for accuracy
    model_acc, _ = pep_trainer.fit(
        train_data,
        position=-2,
        layer=6,
        target_field='is_correct',
        verbose=False
    )
    assert model_acc.probe_type == "accuracy"

    # Train for SE
    model_sep, _ = pep_trainer.fit(
        train_data,
        position=-2,
        layer=6,
        target_field='se_binary',
        verbose=False
    )
    assert model_sep.probe_type == "sep"

    logger.info("Training with different targets successful")


if __name__ == '__main__':
    """Run tests with verbose output."""
    pytest.main([__file__, "-v", "-s", "--log-cli-level=INFO"])
