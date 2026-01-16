"""
Utility functions for probe training and evaluation.

Provides helper functions for:
- Data preparation (extracting X, y from datasets)
- Metrics computation (AUC, log loss, etc.)
- Cross-validation utilities
"""

import numpy as np
import torch
from typing import Optional, Union
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import StratifiedKFold

from src.dataset.base import BaseDataset


def build_Xy_from_dataset(
    dataset: Union[BaseDataset, list[dict]],
    position: int,
    layer: int,
    target_field: str = "se_binary",
    weight_field: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Extract features (X), labels (y), and optional weights from dataset.

    Based on build_Xy_from_dataset() from semantic_entropy_probes.ipynb (lines 1408-1450).

    Args:
        dataset: Dataset with activations or list of sample dicts
        position: Token position in generated sequence (0, -2, etc.)
        layer: Layer index to extract activations from
        target_field: Name of target field in samples
        weight_field: Optional name of sample weight field

    Returns:
        Tuple of (X, y, weights):
        - X: Array of shape (n_samples, hidden_dim)
        - y: Array of shape (n_samples,) with target labels
        - weights: Array of shape (n_samples,) or None

    Example:
        >>> X, y, w = build_Xy_from_dataset(
        ...     train_data, position=0, layer=16,
        ...     target_field='se_binary', weight_field='se_weight'
        ... )
        >>> print(X.shape, y.shape)
        (2500, 4096) (2500,)
    """
    X = []
    y = []
    w = []

    # Iterate directly over dataset (works for both BaseDataset and list)
    for ex in dataset:
        # Extract activation for specified position and layer
        h = ex["activations"]["acts"][position][layer]  # torch tensor on CPU
        X.append(h.numpy() if hasattr(h, "numpy") else np.array(h))

        # Extract target label
        y.append(int(ex[target_field]))

        # Extract weight if specified
        if weight_field is not None:
            w.append(float(ex.get(weight_field, 1.0)))

    X = np.stack(X, axis=0).astype(np.float32)
    y = np.array(y, dtype=np.int64)
    weights = np.array(w, dtype=np.float32) if weight_field is not None else None

    return X, y, weights


def compute_cv_metrics(
    clf,
    X: np.ndarray,
    y: np.ndarray,
    cv: StratifiedKFold,
    sample_weight: Optional[np.ndarray] = None
) -> dict:
    """
    Compute cross-validation metrics for a classifier.

    Args:
        clf: Sklearn-compatible classifier (not fitted)
        X: Feature matrix of shape (n_samples, n_features)
        y: Target labels of shape (n_samples,)
        cv: Cross-validation splitter
        sample_weight: Optional sample weights

    Returns:
        Dict with CV metrics:
        - fold_auc: List of AUC scores per fold
        - fold_logloss: List of log loss scores per fold
        - cv_auc_mean: Mean AUC across folds
        - cv_auc_std: Std of AUC across folds
        - cv_logloss_mean: Mean log loss across folds
        - cv_logloss_std: Std of log loss across folds
    """
    fold_auc = []
    fold_logloss = []

    for tr_idx, va_idx in cv.split(X, y):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[va_idx], y[va_idx]
        w_tr = sample_weight[tr_idx] if sample_weight is not None else None

        # Check if both classes are present in train and validation
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2:
            # Skip this fold if only one class present
            continue

        # Clone classifier and fit
        from sklearn.base import clone
        clf_fold = clone(clf)
        clf_fold.fit(X_tr, y_tr, sample_weight=w_tr)

        # Predict on validation fold
        p_va = clf_fold.predict_proba(X_va)[:, 1]

        # Compute metrics only if both classes present
        try:
            fold_auc.append(roc_auc_score(y_va, p_va))
            fold_logloss.append(log_loss(y_va, p_va))
        except ValueError:
            # Skip fold if metric computation fails
            continue

    # Return NaN if no valid folds
    if len(fold_auc) == 0:
        return {
            'fold_auc': [],
            'fold_logloss': [],
            'cv_auc_mean': float('nan'),
            'cv_auc_std': float('nan'),
            'cv_logloss_mean': float('nan'),
            'cv_logloss_std': float('nan'),
        }

    return {
        'fold_auc': fold_auc,
        'fold_logloss': fold_logloss,
        'cv_auc_mean': float(np.mean(fold_auc)),
        'cv_auc_std': float(np.std(fold_auc)),
        'cv_logloss_mean': float(np.mean(fold_logloss)),
        'cv_logloss_std': float(np.std(fold_logloss)),
    }


def compute_test_metrics(
    clf,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> dict:
    """
    Compute test set metrics for a trained classifier.

    Args:
        clf: Trained sklearn-compatible classifier
        X_test: Test feature matrix
        y_test: Test target labels

    Returns:
        Dict with test metrics:
        - test_auc: AUC score on test set
        - test_logloss: Log loss on test set
    """
    p_test = clf.predict_proba(X_test)[:, 1]

    # Check if both classes are present
    if len(np.unique(y_test)) < 2:
        return {
            'test_auc': float('nan'),
            'test_logloss': float('nan'),
        }

    try:
        return {
            'test_auc': float(roc_auc_score(y_test, p_test)),
            'test_logloss': float(log_loss(y_test, p_test)),
        }
    except ValueError:
        # Return NaN if metric computation fails
        return {
            'test_auc': float('nan'),
            'test_logloss': float('nan'),
        }


def merge_datasets(
    train_data: Union[BaseDataset, list[dict]],
    val_data: Union[BaseDataset, list[dict]]
) -> list[dict]:
    """
    Merge train and validation datasets into single trainval dataset.

    Args:
        train_data: Training dataset
        val_data: Validation dataset

    Returns:
        List of all samples from both datasets
    """
    # Extract data lists
    if isinstance(train_data, BaseDataset):
        train_list = train_data._data
    else:
        train_list = train_data

    if isinstance(val_data, BaseDataset):
        val_list = val_data._data
    else:
        val_list = val_data

    return list(train_list) + list(val_list)
