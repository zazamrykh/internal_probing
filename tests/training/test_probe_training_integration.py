"""
Integration test for probe training module.

This test demonstrates the complete workflow of training linear probes
on internal activations, similar to semantic_entropy_probes.ipynb.

The test uses a toy dataset with random activations to demonstrate:
1. Creating a dataset with activations
2. Training probes with cross-validation
3. Computing and evaluating metrics
4. Saving and loading trained probes
5. Making predictions on test set
6. Generating visualizations
"""

import pytest
import numpy as np
import torch
import tempfile
import logging
from pathlib import Path
from sklearn.linear_model import LogisticRegression

from src.dataset.base import BaseDataset
from src.training import (
    ProbeManager,
    build_Xy_from_dataset,
    save_probe,
    load_probe,
    find_best_probe,
    plot_auc_by_layer,
    plot_sep_vs_accuracy_auc
)

logger = logging.getLogger(__name__)


class ToyDataset(BaseDataset):
    """Toy dataset for testing."""

    def evaluate_correctness(self, prompt, generated_answer, gt_answers):
        """Dummy evaluation."""
        return 1.0


def create_toy_sample(
    idx: int,
    layers: list[int],
    positions: list[int],
    hidden_dim: int = 10,
    seed: int = 42
):
    """
    Create a single toy sample with random activations.

    Args:
        idx: Sample index
        layers: List of layer indices
        positions: List of position indices
        hidden_dim: Dimension of activation vectors
        seed: Random seed

    Returns:
        Dict with sample data including activations
    """
    rng = np.random.RandomState(seed + idx)

    # Create activations for all layer/position combinations
    activations = {
        'positions': positions,
        'layers': layers,
        'acts': {}
    }

    for pos in positions:
        activations['acts'][pos] = {}
        for layer in layers:
            # Random activation vector
            act = torch.tensor(rng.randn(hidden_dim), dtype=torch.float32)
            activations['acts'][pos][layer] = act

    # Random labels
    is_correct = rng.randint(0, 2)
    se_binary = rng.randint(0, 2)
    se_raw = rng.uniform(0, 2)
    se_weight = rng.uniform(0.5, 1.5)

    return {
        'prompt': f'Question {idx}?',
        'gt_answers': [f'Answer {idx}'],
        'greedy_answer': f'Generated answer {idx}',
        'is_correct': is_correct,
        'activations': activations,
        'se_binary': se_binary,
        'se_raw': se_raw,
        'se_weight': se_weight,
        'se_gamma': 1.0,
    }


def create_toy_dataset(
    n_samples: int,
    layers: list[int],
    positions: list[int],
    hidden_dim: int = 10,
    seed: int = 42
) -> ToyDataset:
    """
    Create toy dataset with random activations.

    Args:
        n_samples: Number of samples
        layers: List of layer indices (e.g., [2, 4, 6, 8, 10])
        positions: List of positions (e.g., [0, -2])
        hidden_dim: Dimension of activation vectors
        seed: Random seed

    Returns:
        ToyDataset with activations
    """
    data = [
        create_toy_sample(i, layers, positions, hidden_dim, seed)
        for i in range(n_samples)
    ]
    return ToyDataset(data=data, name='toy_dataset')


@pytest.fixture
def toy_datasets():
    """Create train/val/test toy datasets."""
    layers = [2, 4, 6, 8, 10]
    positions = [0, -2]
    hidden_dim = 10

    train_data = create_toy_dataset(100, layers, positions, hidden_dim, seed=42)
    val_data = create_toy_dataset(20, layers, positions, hidden_dim, seed=142)
    test_data = create_toy_dataset(20, layers, positions, hidden_dim, seed=242)

    return train_data, val_data, test_data, layers, positions


@pytest.fixture
def probe_manager():
    """Create ProbeManager instance."""
    return ProbeManager(
        probe_class=LogisticRegression,
        probe_params={'C': 1.0, 'max_iter': 500, 'solver': 'lbfgs'},
        seed=42
    )


def test_build_Xy_from_dataset(toy_datasets):
    """Test extracting X, y from dataset."""
    train_data, _, _, layers, positions = toy_datasets

    # Extract features for first position and layer
    X, y, w = build_Xy_from_dataset(
        train_data,
        position=positions[0],
        layer=layers[0],
        target_field='se_binary',
        weight_field='se_weight'
    )

    assert X.shape == (100, 10), f"Expected shape (100, 10), got {X.shape}"
    assert y.shape == (100,), f"Expected shape (100,), got {y.shape}"
    assert w.shape == (100,), f"Expected shape (100,), got {w.shape}"
    assert X.dtype == np.float32
    assert y.dtype == np.int64


def test_train_single_probe(probe_manager, toy_datasets):
    """Test training a single probe."""
    train_data, _, _, layers, positions = toy_datasets

    # Train probe
    probe = probe_manager.fit(
        train_data,
        position=positions[0],
        layer=layers[2],  # Middle layer
        target_field='se_binary'
    )

    # Check probe is trained
    assert hasattr(probe, 'coef_'), "Probe should be fitted"
    assert probe.coef_.shape[1] == 10, "Probe should have 10 features"


def test_train_cv(probe_manager, toy_datasets):
    """Test training with cross-validation."""
    train_data, val_data, test_data, layers, positions = toy_datasets

    # Train with CV
    probe, metrics = probe_manager.train_cv(
        train_data, val_data, test_data,
        position=positions[0],
        layer=layers[2],
        target_field='se_binary',
        k_folds=3
    )

    # Check probe is trained
    assert hasattr(probe, 'coef_'), "Probe should be fitted"

    # Check metrics
    assert 'cv_auc_mean' in metrics
    assert 'cv_auc_std' in metrics
    assert 'test_auc' in metrics
    assert 'test_logloss' in metrics
    assert 'n_trainval' in metrics
    assert 'n_test' in metrics

    assert metrics['n_trainval'] == 120  # train + val
    assert metrics['n_test'] == 20

    logger.debug(f"\nCV AUC: {metrics['cv_auc_mean']:.3f} ± {metrics['cv_auc_std']:.3f}")
    logger.debug(f"Test AUC: {metrics['test_auc']:.3f}")


def test_train_every_combination(probe_manager, toy_datasets):
    """
    Test training probes for all combinations.

    This is the main workflow similar to semantic_entropy_probes.ipynb.
    """
    train_data, val_data, test_data, layers, positions = toy_datasets

    # Train all combinations
    results = probe_manager.train_every_combination(
        train_data, val_data, test_data,
        positions=positions,
        layers=layers,
        targets=['se_binary', 'is_correct'],
        k_folds=3,
        weight_field='se_weight',
        use_weights_for_targets=['se_binary'],
        eval=True,
        verbose=True
    )

    # Check results
    expected_count = len(positions) * len(layers) * 2  # 2 targets
    assert len(results) == expected_count, f"Expected {expected_count} results, got {len(results)}"

    # Check each result has required fields
    for result in results:
        assert 'position' in result
        assert 'layer' in result
        assert 'target' in result
        assert 'probe' in result
        assert 'cv_auc_mean' in result
        assert 'test_auc' in result

    # Find best probe for each target
    best_sep = find_best_probe(results, target='se_binary', metric='cv_auc_mean')
    best_acc = find_best_probe(results, target='is_correct', metric='cv_auc_mean')

    logger.debug(f"\nBest SEP probe: layer={best_sep['layer']}, pos={best_sep['position']}, "
          f"CV AUC={best_sep['cv_auc_mean']:.3f}")
    logger.debug(f"Best Accuracy probe: layer={best_acc['layer']}, pos={best_acc['position']}, "
          f"CV AUC={best_acc['cv_auc_mean']:.3f}")


def test_predict(probe_manager, toy_datasets):
    """Test making predictions with trained probe."""
    train_data, _, test_data, layers, positions = toy_datasets

    # Train probe
    probe = probe_manager.fit(
        train_data,
        position=positions[0],
        layer=layers[2],
        target_field='se_binary'
    )

    # Get predictions as array
    probs = probe_manager.predict(
        probe, test_data,
        position=positions[0],
        layer=layers[2],
        add_to_dataset=False,
        return_proba=True
    )

    assert probs.shape == (20,), f"Expected shape (20,), got {probs.shape}"
    assert np.all((probs >= 0) & (probs <= 1)), "Probabilities should be in [0, 1]"

    # Add predictions to dataset
    test_with_preds = probe_manager.predict(
        probe, test_data,
        position=positions[0],
        layer=layers[2],
        add_to_dataset=True,
        prediction_field='sep_score'
    )

    # Check predictions were added
    assert 'sep_score' in test_with_preds[0]
    assert len(test_with_preds) == 20


def test_save_and_load_probe(probe_manager, toy_datasets):
    """Test saving and loading trained probes."""
    train_data, val_data, test_data, layers, positions = toy_datasets

    # Train probe
    probe, metrics = probe_manager.train_cv(
        train_data, val_data, test_data,
        position=positions[0],
        layer=layers[2],
        target_field='se_binary',
        k_folds=3
    )

    # Save probe
    with tempfile.TemporaryDirectory() as tmpdir:
        probe_path = Path(tmpdir) / 'test_probe'

        save_probe(
            probe, metrics, probe_path,
            metadata={'position': positions[0], 'layer': layers[2]}
        )

        # Load probe
        loaded_probe, loaded_metrics, loaded_metadata = load_probe(
            probe_path, load_metrics=True, load_metadata=True
        )

        # Check loaded probe works
        X_test, _, _ = build_Xy_from_dataset(
            test_data, positions[0], layers[2], 'se_binary'
        )

        orig_preds = probe.predict_proba(X_test)[:, 1]
        loaded_preds = loaded_probe.predict_proba(X_test)[:, 1]

        np.testing.assert_array_almost_equal(orig_preds, loaded_preds)

        # Check metrics and metadata
        assert loaded_metrics['test_auc'] == metrics['test_auc']
        assert loaded_metadata['layer'] == layers[2]

        logger.debug(f"\nSuccessfully saved and loaded probe")
        logger.debug(f"Test AUC: {loaded_metrics['test_auc']:.3f}")


def test_get_probe_predictions_on_test(probe_manager, toy_datasets):
    """Test getting predictions from multiple probes."""
    train_data, val_data, test_data, layers, positions = toy_datasets

    # Train a few probes
    probes = {}

    for target in ['se_binary', 'is_correct']:
        probe, _ = probe_manager.train_cv(
            train_data, val_data, test_data,
            position=positions[0],
            layer=layers[2],
            target_field=target,
            k_folds=3
        )
        probes[(target, positions[0], layers[2])] = probe

    # Get predictions
    predictions = probe_manager.get_probe_predictions_on_test(
        probes, test_data, target_field='is_correct'
    )

    # Check predictions
    assert len(predictions) == 2
    for key, preds in predictions.items():
        assert 'p_correct' in preds
        assert 'y_test' in preds
        assert preds['p_correct'].shape == (20,)
        assert preds['y_test'].shape == (20,)

    # Compute AUC for correctness
    auc_results = probe_manager.compute_auc_for_correctness(probes, test_data)

    assert len(auc_results) == 2
    for result in auc_results:
        assert 'target' in result
        assert 'test_auc_is_correct' in result
        logger.debug(f"\n{result['target']} probe AUC for correctness: {result['test_auc_is_correct']:.3f}")


def test_visualization(probe_manager, toy_datasets):
    """Test visualization functions (without displaying plots)."""
    train_data, val_data, test_data, layers, positions = toy_datasets

    # Train probes
    results = probe_manager.train_every_combination(
        train_data, val_data, test_data,
        positions=positions,
        layers=layers,
        targets=['se_binary', 'is_correct'],
        k_folds=3,
        eval=True,
        verbose=False
    )

    # Test plot_auc_by_layer (just check it doesn't crash)
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for testing

    try:
        plot_auc_by_layer(
            results,
            target_field='se_binary',
            title='Test: AUC vs Layer (se_binary)'
        )
        logger.debug("\nplot_auc_by_layer executed successfully")
    except Exception as e:
        pytest.fail(f"plot_auc_by_layer failed: {e}")

    # Test plot_sep_vs_accuracy_auc
    # First compute AUC for correctness
    probes = {(r['target'], r['position'], r['layer']): r['probe'] for r in results}
    auc_results = probe_manager.compute_auc_for_correctness(probes, test_data)

    try:
        plot_sep_vs_accuracy_auc(
            auc_results,
            position=positions[0],
            title='Test: SEP vs Accuracy (pos=0)'
        )
        logger.debug("plot_sep_vs_accuracy_auc executed successfully")
    except Exception as e:
        pytest.fail(f"plot_sep_vs_accuracy_auc failed: {e}")


if __name__ == '__main__':
    """
    Run integration test as a demonstration.

    This shows the complete workflow:
    1. Create toy dataset with activations
    2. Train probes for all combinations
    3. Evaluate and compare probes
    4. Save/load best probes
    5. Make predictions
    6. Generate visualizations
    """
    logger.debug("=" * 70)
    logger.debug("PROBE TRAINING INTEGRATION TEST - DEMONSTRATION")
    logger.debug("=" * 70)

    # Create datasets
    logger.debug("\n1. Creating toy datasets...")
    layers = [2, 4, 6, 8, 10]
    positions = [0, -2]
    train_data = create_toy_dataset(100, layers, positions, seed=42)
    val_data = create_toy_dataset(20, layers, positions, seed=142)
    test_data = create_toy_dataset(20, layers, positions, seed=242)
    logger.debug(f"   Train: {len(train_data)} samples")
    logger.debug(f"   Val: {len(val_data)} samples")
    logger.debug(f"   Test: {len(test_data)} samples")

    # Create manager
    logger.debug("\n2. Creating ProbeManager...")
    manager = ProbeManager(
        probe_class=LogisticRegression,
        probe_params={'C': 1.0, 'max_iter': 500, 'solver': 'lbfgs'},
        seed=42
    )

    # Train all combinations
    logger.debug("\n3. Training probes for all combinations...")
    results = manager.train_every_combination(
        train_data, val_data, test_data,
        positions=positions,
        layers=layers,
        targets=['se_binary', 'is_correct'],
        k_folds=3,
        weight_field='se_weight',
        use_weights_for_targets=['se_binary'],
        eval=True,
        verbose=True
    )

    # Find best probes
    logger.debug("\n4. Finding best probes...")
    best_sep = find_best_probe(results, target='se_binary')
    best_acc = find_best_probe(results, target='is_correct')
    logger.debug(f"   Best SEP: layer={best_sep['layer']}, pos={best_sep['position']}, "
          f"CV AUC={best_sep['cv_auc_mean']:.3f}")
    logger.debug(f"   Best Accuracy: layer={best_acc['layer']}, pos={best_acc['position']}, "
          f"CV AUC={best_acc['cv_auc_mean']:.3f}")

    # Save probes
    logger.debug("\n5. Saving best probes...")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        save_probe(best_sep['probe'], best_sep, tmpdir / 'best_sep')
        save_probe(best_acc['probe'], best_acc, tmpdir / 'best_acc')
        logger.debug(f"   Saved to {tmpdir}")

        # Load and verify
        loaded_sep, _ = load_probe(tmpdir / 'best_sep')
        logger.debug(f"   Loaded SEP probe successfully")

    # Make predictions
    logger.debug("\n6. Making predictions on test set...")
    probes = {(r['target'], r['position'], r['layer']): r['probe'] for r in results}
    predictions = manager.get_probe_predictions_on_test(probes, test_data)
    logger.debug(f"   Generated predictions for {len(predictions)} probes")

    # Compute AUC for correctness
    logger.debug("\n7. Computing AUC for predicting correctness...")
    auc_results = manager.compute_auc_for_correctness(probes, test_data)
    for result in auc_results[:4]:  # Show first 4
        logger.debug(f"   {result['target']} (layer={result['layer']}, pos={result['position']}): "
              f"AUC={result['test_auc_is_correct']:.3f}")

    logger.debug("\n" + "=" * 70)
    logger.debug("DEMONSTRATION COMPLETE")
    logger.debug("=" * 70)
