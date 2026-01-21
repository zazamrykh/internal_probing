"""
ProbeManager class for training and evaluating linear probes.

Provides unified interface for:
- Training single probes
- Cross-validation training
- Batch training for multiple configurations
- Prediction and evaluation
"""

import numpy as np
from typing import Any, Union, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from copy import deepcopy

from src.dataset.base import BaseDataset
from src.training.base import BaseProbeTrainer
from src.training.utils import (
    build_Xy_from_dataset,
    compute_cv_metrics,
    compute_test_metrics,
    merge_datasets
)


class ProbeManager(BaseProbeTrainer):
    """
    Manager for training and evaluating linear probes on internal activations.

    Handles:
    - Training probes with cross-validation
    - Batch training for multiple layer/position combinations
    - Prediction and evaluation
    - Model persistence

    Example:
        >>> from sklearn.linear_model import LogisticRegression
        >>> manager = ProbeManager(
        ...     probe_class=LogisticRegression,
        ...     probe_params={'C': 1.0, 'max_iter': 500, 'solver': 'lbfgs'}
        ... )
        >>>
        >>> # Train single probe with CV
        >>> probe, metrics = manager.train_cv(
        ...     train_data=train_dataset,
        ...     val_data=val_dataset,
        ...     test_data=test_dataset,
        ...     position=0,
        ...     layer=16,
        ...     target_field='se_binary',
        ...     k_folds=5
        ... )
        >>>
        >>> # Train all combinations
        >>> results = manager.train_every_combination(
        ...     train_data=train_dataset,
        ...     val_data=val_dataset,
        ...     test_data=test_dataset,
        ...     positions=[0, -2],
        ...     layers=[0, 8, 16, 24, 31],
        ...     targets=['se_binary', 'is_correct']
        ... )
    """

    def __init__(
        self,
        probe_class=LogisticRegression,
        probe_params: Optional[dict] = None,
        seed: int = 42
    ):
        """
        Initialize probe manager.

        Args:
            probe_class: Sklearn-compatible classifier class
            probe_params: Parameters for probe initialization
            seed: Random seed for reproducibility
        """
        super().__init__(seed=seed)
        self.probe_class = probe_class
        self.probe_params = probe_params or {}

    def _create_probe(self) -> Any:
        """Create new probe instance with configured parameters."""
        return self.probe_class(**self.probe_params)

    def fit(
        self,
        dataset: Union[BaseDataset, list[dict]],
        position: int,
        layer: int,
        target_field: str,
        weight_field: Optional[str] = None,
        **fit_kwargs
    ) -> Any:
        """
        Train single probe on dataset.

        Args:
            dataset: Dataset with activations
            position: Token position to use (0, -2, etc.)
            layer: Layer index to use
            target_field: Target field name ('se_binary', 'is_correct')
            weight_field: Optional sample weights field
            **fit_kwargs: Additional arguments for probe.fit()

        Returns:
            Trained probe model

        Example:
            >>> probe = manager.fit(
            ...     train_data, position=0, layer=16,
            ...     target_field='se_binary'
            ... )
        """
        # Extract data
        X, y, weights = build_Xy_from_dataset(
            dataset, position, layer, target_field, weight_field
        )

        # Create and train probe
        probe = self._create_probe()
        probe.fit(X, y, sample_weight=weights, **fit_kwargs)

        return probe

    def predict(
        self,
        probe: Any,
        dataset: Union[BaseDataset, list[dict]],
        position: int,
        layer: int,
        add_to_dataset: bool = True,
        prediction_field: str = "probe_prediction",
        return_proba: bool = True
    ) -> Union[np.ndarray, BaseDataset, list[dict]]:
        """
        Make predictions with trained probe.

        Args:
            probe: Trained probe model
            dataset: Test dataset
            position: Token position used in training
            layer: Layer index used in training
            add_to_dataset: If True, add predictions to dataset
            prediction_field: Name for prediction field
            return_proba: If True, return probabilities; else class labels

        Returns:
            If add_to_dataset=True: Modified dataset
            If add_to_dataset=False: Array of predictions

        Example:
            >>> # Get predictions as array
            >>> probs = manager.predict(
            ...     probe, test_data, position=0, layer=16,
            ...     add_to_dataset=False
            ... )
            >>>
            >>> # Add predictions to dataset
            >>> test_data = manager.predict(
            ...     probe, test_data, position=0, layer=16,
            ...     add_to_dataset=True, prediction_field='sep_score'
            ... )
        """
        # Extract features by iterating over dataset
        X = []
        for ex in dataset:
            h = ex["activations"]["acts"][position][layer]
            X.append(h.numpy() if hasattr(h, "numpy") else np.array(h))
        X = np.stack(X, axis=0).astype(np.float32)

        # Make predictions
        if return_proba:
            predictions = probe.predict_proba(X)[:, 1]
        else:
            predictions = probe.predict(X)

        if add_to_dataset:
            # Create new dataset with predictions added
            data_copy = []
            for i, ex in enumerate(dataset):
                ex_copy = deepcopy(ex)
                ex_copy[prediction_field] = float(predictions[i])
                data_copy.append(ex_copy)

            # Return in same format as input
            if isinstance(dataset, BaseDataset):
                return dataset.__class__(data=data_copy, name=dataset.name)
            else:
                return data_copy
        else:
            return predictions

    def train_cv(
        self,
        train_data: Union[BaseDataset, list[dict]],
        val_data: Union[BaseDataset, list[dict]],
        test_data: Union[BaseDataset, list[dict]],
        position: int,
        layer: int,
        target_field: str,
        k_folds: int = 5,
        weight_field: Optional[str] = None,
        compute_metrics: bool = True
    ) -> tuple[Any, dict]:
        """
        Train probe with cross-validation.

        Based on train_and_eval_probe_cv() from semantic_entropy_probes.ipynb (lines 1452-1580).

        Args:
            train_data: Training dataset
            val_data: Validation dataset
            test_data: Test dataset
            position: Token position
            layer: Layer index
            target_field: Target field name
            k_folds: Number of CV folds
            weight_field: Optional sample weights
            compute_metrics: Whether to compute evaluation metrics

        Returns:
            Tuple of (trained_probe, metrics_dict)
            metrics_dict contains:
                - cv_auc_mean, cv_auc_std
                - cv_logloss_mean, cv_logloss_std
                - test_auc, test_logloss
                - n_trainval, n_test, n_iter_used

        Example:
            >>> probe, metrics = manager.train_cv(
            ...     train_data, val_data, test_data,
            ...     position=0, layer=16, target_field='se_binary',
            ...     k_folds=5
            ... )
            >>> print(f"CV AUC: {metrics['cv_auc_mean']:.3f} ± {metrics['cv_auc_std']:.3f}")
            >>> print(f"Test AUC: {metrics['test_auc']:.3f}")
        """
        # Merge train and val for cross-validation
        trainval_data = merge_datasets(train_data, val_data)

        # Extract data
        w_field = weight_field if weight_field else None
        X_tv, y_tv, w_tv = build_Xy_from_dataset(
            trainval_data, position, layer, target_field, w_field
        )
        X_test, y_test, _ = build_Xy_from_dataset(
            test_data, position, layer, target_field, None
        )

        metrics = {
            "position": position,
            "layer": layer,
            "target": target_field,
            "k_folds": k_folds,
            "n_trainval": int(len(y_tv)),
            "n_test": int(len(y_test)),
        }

        # Check class balance
        unique_classes, class_counts = np.unique(y_tv, return_counts=True)
        min_class_count = int(class_counts.min())

        if len(unique_classes) < 2:
            import warnings
            warnings.warn(
                f"Only one class present in trainval data for target '{target_field}'. "
                f"Metrics may be unreliable.",
                UserWarning
            )
        elif min_class_count < k_folds:
            import warnings
            warnings.warn(
                f"Least populated class has only {min_class_count} samples, "
                f"which is less than k_folds={k_folds}. Some CV folds will be skipped.",
                UserWarning
            )

        # Compute CV metrics if requested
        if compute_metrics:
            cv = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=self.seed)
            probe_for_cv = self._create_probe()
            cv_metrics = compute_cv_metrics(probe_for_cv, X_tv, y_tv, cv, w_tv)
            metrics.update(cv_metrics)

        # Train final model on full trainval
        final_probe = self._create_probe()
        final_probe.fit(X_tv, y_tv, sample_weight=w_tv)

        # Compute test metrics
        if compute_metrics:
            test_metrics = compute_test_metrics(final_probe, X_test, y_test)
            metrics.update(test_metrics)

        # Add iteration count if available
        if hasattr(final_probe, 'n_iter_'):
            n_iter = final_probe.n_iter_
            metrics['n_iter_used'] = int(n_iter[0] if isinstance(n_iter, np.ndarray) else n_iter)

        return final_probe, metrics

    # train_every_combination is inherited from BaseProbeTrainer with default implementation

    def get_probe_predictions_on_test(
        self,
        probes: dict,
        test_data: Union[BaseDataset, list[dict]],
        target_field: str = "is_correct"
    ) -> dict:
        """
        Get predictions from multiple probes on test set.

        Based on get_probe_predictions_on_test() from semantic_entropy_probes.ipynb (lines 1709-1763).

        Args:
            probes: Dict mapping (target, position, layer) -> trained probe
            test_data: Test dataset with activations
            target_field: Field to evaluate against (default: 'is_correct')

        Returns:
            Dict mapping (target, pos, layer) -> {
                'p_correct': np.ndarray of P(correct) predictions,
                'y_test': np.ndarray of true labels
            }

        Example:
            >>> probes = {
            ...     ('se_binary', 0, 16): sep_probe,
            ...     ('is_correct', 0, 16): acc_probe
            ... }
            >>> predictions = manager.get_probe_predictions_on_test(probes, test_data)
            >>>
            >>> # Compute AUC for each probe
            >>> from sklearn.metrics import roc_auc_score
            >>> for key, preds in predictions.items():
            ...     auc = roc_auc_score(preds['y_test'], preds['p_correct'])
            ...     print(f"{key}: AUC = {auc:.3f}")
        """
        results = {}

        for (target, pos, layer), clf in probes.items():
            # Extract features and labels
            X_test, y_test, _ = build_Xy_from_dataset(
                test_data, position=pos, layer=layer, target_field=target_field
            )

            # Get predictions
            p_test = clf.predict_proba(X_test)[:, 1]  # P(class=1)

            # Convert to P(correct) based on probe type
            if target == "se_binary":
                # SEP probe: P(high_SE) -> invert to P(correct)
                p_correct = 1.0 - p_test
            else:  # is_correct probe
                # Accuracy probe: P(correct) directly
                p_correct = p_test

            results[(target, pos, layer)] = {
                "p_correct": p_correct,
                "y_test": y_test,
            }

        return results

    def compute_auc_for_correctness(
        self,
        probes: dict,
        test_data: Union[BaseDataset, list[dict]]
    ) -> list[dict]:
        """
        Compute AUC for predicting correctness from all probes.

        Wrapper around get_probe_predictions_on_test() that returns a list of dicts.

        Args:
            probes: Dict mapping (target, position, layer) -> trained probe
            test_data: Test dataset

        Returns:
            List of dicts with columns: ['target', 'position', 'layer', 'test_auc_is_correct']

        Example:
            >>> auc_results = manager.compute_auc_for_correctness(probes, test_data)
            >>> for result in auc_results:
            ...     print(f"{result['target']} at layer {result['layer']}: AUC = {result['test_auc_is_correct']:.3f}")
        """
        probe_preds = self.get_probe_predictions_on_test(probes, test_data, target_field="is_correct")

        auc_results = []
        for (target, pos, layer), preds in probe_preds.items():
            p_correct = preds["p_correct"]
            y_test = preds["y_test"]

            auc = roc_auc_score(y_test, p_correct)

            auc_results.append({
                "target": target,
                "position": pos,
                "layer": layer,
                "test_auc_is_correct": float(auc),
            })

        return auc_results
