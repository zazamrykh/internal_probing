# Training Module for Linear Probes

This module provides tools for training and evaluating linear probes on internal model activations for uncertainty estimation and correctness prediction.

## Quick Start

```python
from sklearn.linear_model import LogisticRegression
from src.training import ProbeManager
from src.dataset import TriviaQADataset

# 1. Load dataset with activations (already enriched)
train_data = TriviaQADataset.load('data/train_with_activations.pt')
val_data = TriviaQADataset.load('data/val_with_activations.pt')
test_data = TriviaQADataset.load('data/test_with_activations.pt')

# 2. Create ProbeManager
manager = ProbeManager(
    probe_class=LogisticRegression,
    probe_params={'C': 1.0, 'max_iter': 500, 'solver': 'lbfgs'},
    seed=42
)

# 3. Train probes for all combinations
results = manager.train_every_combination(
    train_data, val_data, test_data,
    positions=[0, -2],  # TBG and SLT
    layers=[0, 8, 16, 24, 31],
    targets=['se_binary', 'is_correct'],
    k_folds=5,
    weight_field='se_weight',
    use_weights_for_targets=['se_binary'],
    eval=True,
    verbose=True
)

# 4. Find and save best probes
from src.training import find_best_probe, save_probe

best_sep = find_best_probe(results, target='se_binary')
save_probe(best_sep['probe'], best_sep, 'models/probes/best_sep')

# 5. Visualize results
from src.training import plot_auc_by_layer

plot_auc_by_layer(
    results,
    target_field='se_binary',
    title='Test AUROC vs Layer (SEP)'
)
```

## Main Components

### ProbeManager

Main class for training and evaluating probes:

- **`fit()`** - Train single probe on dataset
- **`predict()`** - Make predictions with trained probe
- **`train_cv()`** - Train with cross-validation
- **`train_every_combination()`** - Train for all layer/position/target combinations
- **`get_probe_predictions_on_test()`** - Get predictions from multiple probes
- **`compute_auc_for_correctness()`** - Compute AUC for predicting correctness

### Utilities (`src/training/utils.py`)

- **`build_Xy_from_dataset()`** - Extract features and labels from dataset
- **`compute_cv_metrics()`** - Compute cross-validation metrics
- **`compute_test_metrics()`** - Compute test set metrics
- **`merge_datasets()`** - Merge train and validation datasets

### Visualization (`src/training/visualization.py`)

- **`plot_auc_by_layer()`** - Plot AUC vs layer for different positions
- **`plot_sep_vs_accuracy_auc()`** - Compare SEP vs Accuracy probes

### I/O (`src/training/io.py`)

- **`save_probe()`** - Save probe with metrics and metadata
- **`load_probe()`** - Load saved probe
- **`find_best_probe()`** - Find best probe from results
- **`save_all_probes()`** - Save multiple probes

## Testing

Run integration tests to see complete workflow:

```bash
# Run all tests
pytest tests/training/test_probe_training_integration.py -v

# Run as demonstration
python tests/training/test_probe_training_integration.py
```

## References

Based on implementation from `notebooks/semantic_entropy_probes.ipynb` and the semantic entropy probes methodology.
