"""
Base interface for probe training methods.

Provides abstract base class that defines common interface for different
probe training approaches (sklearn-based, end-to-end learnable, etc.).
"""

from abc import ABC, abstractmethod
from typing import Any, Union, Optional
import numpy as np

from src.dataset.base import BaseDataset


class BaseProbeTrainer(ABC):
    """
    Abstract base class for probe training methods.

    Defines common interface for training probes on internal model activations.
    Different implementations can use sklearn models, PyTorch models, or other approaches.

    All trainers should support:
    - Training single probes with fit()
    - Making predictions with predict()
    - Cross-validation training with train_cv()
    - Batch training for multiple configurations with train_every_combination()
    """

    def __init__(self, seed: int = 42):
        """
        Initialize trainer.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed

    @abstractmethod
    def fit(
        self,
        dataset: Union[BaseDataset, list[dict]],
        position: int,
        layer: int,
        target_field: str,
        weight_field: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Train single probe on dataset.

        Args:
            dataset: Dataset with activations and targets
            position: Token position to use (0, -2, etc.)
            layer: Layer index to use
            target_field: Target field name ('se_binary', 'is_correct', etc.)
            weight_field: Optional sample weights field
            **kwargs: Additional training arguments

        Returns:
            Trained probe model
        """
        pass

    @abstractmethod
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
        """
        pass

    @abstractmethod
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
        compute_metrics: bool = True,
        **kwargs
    ) -> tuple[Any, dict]:
        """
        Train probe with cross-validation.

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
            **kwargs: Additional training arguments

        Returns:
            Tuple of (trained_probe, metrics_dict)
        """
        pass

    def train_every_combination(
        self,
        train_data: Union[BaseDataset, list[dict]],
        val_data: Union[BaseDataset, list[dict]],
        test_data: Union[BaseDataset, list[dict]],
        positions: list[int],
        layers: list[int],
        targets: Optional[list[str]] = None,
        k_folds: int = 5,
        weight_field: Optional[str] = None,
        use_weights_for_targets: Optional[list[str]] = None,
        eval: bool = True,
        verbose: bool = True,
        **kwargs
    ) -> list[dict]:
        """
        Train probes for all combinations of positions, layers, and targets.

        Default implementation iterates over all combinations and calls train_cv().
        Subclasses can override for custom behavior.

        Args:
            train_data: Training dataset
            val_data: Validation dataset
            test_data: Test dataset
            positions: List of token positions to try
            layers: List of layer indices to try
            targets: List of target fields (default: ['se_binary', 'is_correct'])
            k_folds: Number of CV folds
            weight_field: Sample weights field name
            use_weights_for_targets: List of targets to use weights for
            eval: Whether to compute evaluation metrics
            verbose: Whether to print progress
            **kwargs: Additional training arguments

        Returns:
            List of dicts, each containing:
                - 'position': int
                - 'layer': int
                - 'target': str
                - 'probe': trained model
                - 'metrics': dict (if eval=True)
        """
        if targets is None:
            targets = ['se_binary', 'is_correct']

        if use_weights_for_targets is None:
            use_weights_for_targets = []

        all_results = []
        total_combinations = len(targets) * len(positions) * len(layers)
        current = 0

        for target in targets:
            for pos in positions:
                for layer in layers:
                    current += 1

                    # Determine if we should use weights for this target
                    use_w = target in use_weights_for_targets

                    # Train probe
                    probe, metrics = self.train_cv(
                        train_data, val_data, test_data,
                        position=pos,
                        layer=layer,
                        target_field=target,
                        k_folds=k_folds,
                        weight_field=weight_field if use_w else None,
                        compute_metrics=eval,
                        **kwargs
                    )

                    result = {
                        'position': pos,
                        'layer': layer,
                        'target': target,
                        'probe': probe,
                    }

                    if eval:
                        result['metrics'] = metrics
                        # Also add metrics at top level for easier access
                        result.update(metrics)

                    all_results.append(result)

                    if verbose:
                        if eval and 'cv_auc_mean' in metrics:
                            print(
                                f"[{current}/{total_combinations}] "
                                f"target={target}, pos={pos}, layer={layer}: "
                                f"cv_auc={metrics['cv_auc_mean']:.3f}±{metrics['cv_auc_std']:.3f}, "
                                f"test_auc={metrics['test_auc']:.3f}"
                            )
                        else:
                            print(
                                f"[{current}/{total_combinations}] "
                                f"Trained: target={target}, pos={pos}, layer={layer}"
                            )

        return all_results
