"""
Linear Probe Method for uncertainty estimation.

Uses sklearn-based linear classifiers trained on model activations
to predict uncertainty/correctness.
"""

import logging
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List

from sklearn.linear_model import LogisticRegression

from src.experiments.methods.base import MethodInterface
from src.scoring.base import ScorerInterface
from src.scoring.linear_probe import LinearProbeScorer
from src.training.probe_manager import ProbeManager
from src.training.io import save_probe, load_probe, find_best_probe

logger = logging.getLogger(__name__)


class LinearProbeMethod(MethodInterface):
    """
    Linear probe method using sklearn classifiers.

    Trains linear probes (e.g., LogisticRegression) on model activations
    to predict uncertainty or correctness. Supports training multiple probes
    for different layer/position/target combinations and selecting the best one.

    Example config:
        [method]
        type = "linear_probe"
        layers = [0, 8, 16, 24, 31]
        positions = [-2]
        targets = ["is_correct"]
        k_folds = 5

        # Probe parameters
        probe_class = "LogisticRegression"
        C = 1.0
        max_iter = 500
        solver = "lbfgs"

        # Selection
        selection_metric = "val_auc"  # Metric to select best probe (renamed from test_auc)

        # Saving
        output_dir = "exp_results/linear_probe"
        save_all_probes = false
        clear_checkpoints = false
    """

    def __init__(
        self,
        method_config: Dict[str, Any],
        **kwargs
    ):
        """
        Initialize Linear Probe method.

        Args:
            method_config: Configuration dictionary with keys:
                - layers: List[int] - layer indices to probe
                - positions: List[int] - token positions to probe
                - targets: List[str] - target fields (e.g., ["is_correct", "se_binary"])
                - k_folds: int (default: 5) - number of CV folds
                - probe_class: str (default: "LogisticRegression") - sklearn classifier
                - C: float (default: 1.0) - regularization parameter
                - max_iter: int (default: 500) - max iterations
                - solver: str (default: "lbfgs") - solver type
                - weight_field: str (optional) - sample weights field
                - use_weights_for_targets: List[str] (optional) - targets to use weights for
                - selection_metric: str (default: "test_auc") - metric for best probe selection
                - output_dir: str (optional) - directory for saving probes
                - save_all_probes: bool (default: False) - save all or just best
                - clear_checkpoints: bool (default: False) - clear output dir before training
            **kwargs: Additional arguments (ignored)
        """
        self.config = method_config

        # Required parameters
        self.layers = method_config['layers']
        self.positions = method_config['positions']
        self.targets = method_config['targets']

        # Training parameters
        self.k_folds = method_config.get('k_folds', 5)
        self.weight_field = method_config.get('weight_field')
        self.use_weights_for_targets = method_config.get('use_weights_for_targets', [])

        # Probe parameters
        probe_class_name = method_config.get('probe_class', 'LogisticRegression')
        self.probe_params = {
            'C': method_config.get('C', 1.0),
            'max_iter': method_config.get('max_iter', 500),
            'solver': method_config.get('solver', 'lbfgs'),
        }

        # Selection parameters
        # Note: ProbeManager uses "test_auc" internally, but we call it "val_auc"
        # because in the Method context, we're evaluating on validation data
        self.selection_metric = method_config.get('selection_metric', 'cv_auc_mean')
        # Map to internal metric name
        self._internal_metric = 'test_auc' if self.selection_metric == 'val_auc' else self.selection_metric

        # Output parameters
        self.output_dir = method_config.get('output_dir')
        self.save_all_probes = method_config.get('save_all_probes', False)
        self.clear_checkpoints = method_config.get('clear_checkpoints', False)

        # Create probe manager
        probe_class = self._get_probe_class(probe_class_name)
        self.probe_manager = ProbeManager(
            probe_class=probe_class,
            probe_params=self.probe_params,
            seed=method_config.get('seed', 42)
        )

        # Training results storage
        self._training_results: Optional[List[Dict]] = None
        self._best_result: Optional[Dict] = None

        logger.info(
            f"Initialized LinearProbeMethod: "
            f"layers={self.layers}, positions={self.positions}, "
            f"targets={self.targets}, k_folds={self.k_folds}"
        )

    def _get_probe_class(self, class_name: str):
        """Get sklearn classifier class by name."""
        if class_name == 'LogisticRegression':
            return LogisticRegression
        else:
            raise ValueError(f"Unknown probe class: {class_name}")

    def train(self, train_data, val_data, **kwargs) -> Dict[str, Any]:
        """
        Train linear probes for all layer/position/target combinations.

        Args:
            train_data: Training dataset with activations
            val_data: Validation dataset with activations
            **kwargs: Additional arguments (test_data can be passed here)

        Returns:
            Dict with training results including:
            - all_results: List of results for all combinations
            - best_result: Best probe configuration
            - n_probes_trained: Number of probes trained
        """
        # Get test_data from kwargs or use val_data
        test_data = kwargs.get('test_data', val_data)

        # Clear output directory if requested
        if self.clear_checkpoints and self.output_dir:
            output_path = Path(self.output_dir)
            if output_path.exists():
                logger.info(f"Clearing output directory: {output_path}")
                shutil.rmtree(output_path)

        logger.info(
            f"Training {len(self.layers) * len(self.positions) * len(self.targets)} "
            f"linear probes..."
        )

        # Train all combinations
        self._training_results = self.probe_manager.train_every_combination(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            positions=self.positions,
            layers=self.layers,
            targets=self.targets,
            k_folds=self.k_folds,
            weight_field=self.weight_field,
            use_weights_for_targets=self.use_weights_for_targets,
            eval=True,
            verbose=True,
        )

        # Find best probe (use internal metric name for lookup)
        try:
            self._best_result = find_best_probe(
                self._training_results,
                metric=self._internal_metric
            )
            metric_value = self._best_result.get(self._internal_metric, 'N/A')
            logger.info(
                f"Best probe: layer={self._best_result['layer']}, "
                f"position={self._best_result['position']}, "
                f"target={self._best_result['target']}, "
                f"{self.selection_metric}={metric_value}"
            )
        except Exception as e:
            logger.warning(f"Could not find best probe: {e}")
            if self._training_results:
                self._best_result = self._training_results[0]

        # Save probes if output_dir specified
        if self.output_dir:
            self._save_probes()

        # Prepare results summary (without probe objects for JSON serialization)
        # Also rename test_* metrics to val_* for clarity
        results_summary = []
        for r in self._training_results:
            summary = {}
            for k, v in r.items():
                if k == 'probe':
                    continue
                # Rename test_* to val_* (since in Method context it's validation data)
                if k.startswith('test_'):
                    new_key = 'val_' + k[5:]
                    summary[new_key] = v
                else:
                    summary[k] = v
            results_summary.append(summary)

        return {
            "method": "linear_probe",
            "n_probes_trained": len(self._training_results),
            "best_config": {
                "layer": self._best_result['layer'],
                "position": self._best_result['position'],
                "target": self._best_result['target'],
                self.selection_metric: self._best_result.get(self._internal_metric),
            },
            "all_results": results_summary,
        }

    def _save_probes(self):
        """Save trained probes to output directory."""
        output_path = Path(self.output_dir) / 'probes'
        output_path.mkdir(parents=True, exist_ok=True)

        if self.save_all_probes:
            # Save all probes
            for result in self._training_results:
                if 'probe' not in result:
                    continue

                target = result.get('target', 'unknown')
                pos = result.get('position', 0)
                layer = result.get('layer', 0)

                filename = f"probe_pos{pos}_layer{layer}_{target}"
                probe_path = output_path / filename

                # Extract metrics (without probe)
                metrics = {k: v for k, v in result.items() if k != 'probe'}

                save_probe(result['probe'], metrics, probe_path)

            logger.info(f"Saved {len(self._training_results)} probes to {output_path}")
        else:
            # Save only best probe
            if self._best_result and 'probe' in self._best_result:
                target = self._best_result.get('target', 'unknown')
                pos = self._best_result.get('position', 0)
                layer = self._best_result.get('layer', 0)

                filename = f"best_probe_pos{pos}_layer{layer}_{target}"
                probe_path = output_path / filename

                metrics = {k: v for k, v in self._best_result.items() if k != 'probe'}

                save_probe(self._best_result['probe'], metrics, probe_path)
                logger.info(f"Saved best probe to {probe_path}")

    def get_scorer(self) -> ScorerInterface:
        """
        Get scorer for inference using the best trained probe.

        Returns:
            LinearProbeScorer configured with the best probe
        """
        if self._best_result is None or 'probe' not in self._best_result:
            raise RuntimeError(
                "No trained probe available. Call train() first."
            )

        # Determine probe type based on target
        target = self._best_result.get('target', 'is_correct')
        if 'se' in target.lower():
            probe_type = 'sep'
        else:
            probe_type = 'accuracy'

        logger.info(
            f"Creating LinearProbeScorer: "
            f"layer={self._best_result['layer']}, "
            f"position={self._best_result['position']}, "
            f"probe_type={probe_type}"
        )

        # Store layer/position info for use in StandardExperiment
        scorer = LinearProbeScorer(
            probe_model=self._best_result['probe'],
            probe_type=probe_type
        )

        # Attach metadata for StandardExperiment to use
        scorer.layer = self._best_result['layer']
        scorer.position = self._best_result['position']
        scorer.target = target

        return scorer
