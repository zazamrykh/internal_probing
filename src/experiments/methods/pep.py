"""
PEP (Prompt Embedding Probes) Method for uncertainty estimation.

Uses learnable soft prompt embeddings combined with linear probes
trained on model activations to predict uncertainty/correctness.
"""

from typing import Dict, Any, Optional, List
import logging
import shutil
import json
from pathlib import Path
import torch

from src.experiments.methods.base import MethodInterface
from src.scoring.base import ScorerInterface
from src.scoring.prompt_embedding import PEPModel
from src.training.pep_trainer import PEPTrainer

logger = logging.getLogger(__name__)


class PEPMethod(MethodInterface):
    """
    PEP (Prompt Embedding Probes) method using learnable embeddings.

    Trains PEP models (learnable soft prompt embeddings + linear probe) on model
    activations to predict uncertainty or correctness. Supports training multiple
    models for different layer/position/target combinations and selecting the best one.

    Example config:
        [method]
        type = "pep"
        layers = [0, 8, 16, 24, 31]
        positions = [-2]
        targets = ["is_correct"]

        # PEP parameters
        n_embeddings = 1
        learning_rate = 0.001
        n_epochs = 10
        batch_size = 8
        weight_decay = 0.0
        optimizer_type = "adamw"

        # Early stopping
        early_stopping_patience = 10
        early_stopping_metric = "auc"
        val_check_interval = 100

        # Selection
        selection_metric = "best_metric"  # Metric to select best model

        # Saving
        output_dir = "exp_results/pep"
        save_best_model = true
        clear_checkpoints = false
        return_history = false
    """

    def __init__(
        self,
        method_config: Dict[str, Any],
        model_wrapper=None,
        **kwargs
    ):
        """
        Initialize PEP method.

        Args:
            method_config: Configuration dictionary with keys:
                - layers: List[int] - layer indices to probe
                - positions: List[int] - token positions to probe
                - targets: List[str] - target fields (e.g., ["is_correct"])
                - n_embeddings: int (default: 1) - number of soft prompt tokens
                - learning_rate: float (default: 0.001) - learning rate
                - n_epochs: int (default: 10) - number of epochs
                - batch_size: int (default: 8) - batch size
                - weight_decay: float (default: 0.0) - L2 regularization
                - optimizer_type: str (default: "adamw") - optimizer type
                - early_stopping_patience: int (optional) - early stopping patience
                - early_stopping_metric: str (default: "auc") - metric for early stopping
                - val_check_interval: int (default: 100) - validation check interval
                - val_samples: int (optional) - number of validation samples to use
                - max_samples: int (optional) - max training samples
                - max_training_time: float (optional) - max training time in seconds
                - selection_metric: str (default: "best_metric") - metric for best model selection
                - output_dir: str (optional) - directory for saving models
                - save_best_model: bool (default: True) - save best model during training
                - clear_checkpoints: bool (default: False) - clear output dir before training
                - return_history: bool (default: False) - return training history
            model_wrapper: ModelWrapper instance (required)
            **kwargs: Additional arguments (ignored)
        """
        assert model_wrapper is not None, 'Must provide model_wrapper for PEPMethod'
        self.model_wrapper = model_wrapper
        self.config = method_config

        # Required parameters
        self.layers = method_config['layers']
        self.positions = method_config['positions']
        self.targets = method_config['targets']

        # Training parameters
        self.weight_field = method_config.get('weight_field')
        self.use_weights_for_targets = method_config.get('use_weights_for_targets', [])

        # Selection parameters
        self.selection_metric = method_config.get('selection_metric', 'best_metric')

        # Output parameters
        self.output_dir = method_config.get('output_dir')
        self.save_all_models = method_config.get('save_all_models', False)
        self.clear_checkpoints = method_config.get('clear_checkpoints', False)
        self.return_history = method_config.get('return_history', False)

        # Create PEP trainer
        self.pep_trainer = PEPTrainer(
            model_wrapper=model_wrapper,
            n_embeddings=method_config.get('n_embeddings', 1),
            learning_rate=method_config.get('learning_rate', 0.001),
            n_epochs=method_config.get('n_epochs', 10),
            batch_size=method_config.get('batch_size', 8),
            weight_decay=method_config.get('weight_decay', 0.0),
            optimizer_type=method_config.get('optimizer_type', 'adamw'),
            embedding_init_std=method_config.get('embedding_init_std', 0.02),
            seed=method_config.get('seed', 42),
            device=method_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'),
            early_stopping_patience=method_config.get('early_stopping_patience'),
            early_stopping_metric=method_config.get('early_stopping_metric', 'auc'),
            save_best_model=method_config.get('save_best_model', True),
            checkpoint_dir=None,  # Will be set per model during training
            max_samples=method_config.get('max_samples'),
            max_training_time=method_config.get('max_training_time'),
            val_check_interval=method_config.get('val_check_interval', 100),
            val_samples=method_config.get('val_samples'),
        )

        # Training results storage
        self._training_results: Optional[List[Dict]] = None
        self._best_model_path: Optional[str] = None

        logger.info(
            f"Initialized PEPMethod: "
            f"layers={self.layers}, positions={self.positions}, "
            f"targets={self.targets}"
        )

        logger.debug(f"Initialized PEPMethod with config {method_config}")

    def train(self, train_data, val_data, **kwargs) -> Dict[str, Any]:
        """
        Train PEP models for all layer/position/target combinations.

        Args:
            train_data: Training dataset with greedy answers
            val_data: Validation dataset with greedy answers
            **kwargs: Additional arguments (test_data can be passed here)

        Returns:
            Dict with training results including:
            - all_results: List of results for all combinations
            - best_result: Best model configuration
            - n_models_trained: Number of models trained
        """
        # Clear output directory if requested
        if self.clear_checkpoints and self.output_dir:
            output_path = Path(self.output_dir)
            if output_path.exists():
                logger.info(f"Clearing output directory: {output_path}")
                shutil.rmtree(output_path)

        # Get test_data from kwargs if provided
        test_data = kwargs.get('test_data', val_data)

        # Set directories
        checkpoint_dir_base = None
        best_model_dir = None
        if self.output_dir:
            checkpoint_dir_base = str(Path(self.output_dir) / 'checkpoints')
            best_model_dir = str(Path(self.output_dir) / 'best_model')

        # Train all combinations using PEPTrainer.train_every_combination
        results = self.pep_trainer.train_every_combination(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            positions=self.positions,
            layers=self.layers,
            targets=self.targets,
            weight_field=self.weight_field,
            use_weights_for_targets=self.use_weights_for_targets,
            return_history=self.return_history,
            checkpoint_dir_base=checkpoint_dir_base,
            save_all_models=self.save_all_models,
            best_model_dir=best_model_dir,
            selection_metric=self.selection_metric,
            verbose=True,
        )

        self._training_results = results

        # Store best model path
        if best_model_dir:
            self._best_model_path = str(Path(best_model_dir) / 'best_model.pt')

            # Load and log best model info
            metadata_path = Path(best_model_dir) / 'best_model_info.json'
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    best_info = json.load(f)
                logger.info(
                    f"Best PEP model: layer={best_info['layer']}, "
                    f"position={best_info['position']}, "
                    f"target={best_info['target']}, "
                    f"{self.selection_metric}={best_info.get(self.selection_metric, 'N/A')}"
                )

        # Prepare results summary (already without model objects)
        results_summary = []
        for r in results:
            summary = dict(r)
            # Remove history if not requested
            if 'history' in summary and not self.return_history:
                summary.pop('history')
            results_summary.append(summary)

        # Get best config from metadata
        best_config = None
        if best_model_dir:
            metadata_path = Path(best_model_dir) / 'best_model_info.json'
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    best_config = json.load(f)

        return {
            "method": "pep",
            "n_models_trained": len(results),
            "best_config": best_config,
            "all_results": results_summary,
        }

    def get_scorer(self) -> ScorerInterface:
        """
        Get scorer for inference by loading the best trained PEP model from disk.

        Returns:
            PEPModel configured as scorer (already implements ScorerInterface)
        """
        if self._best_model_path is None:
            raise RuntimeError(
                "No trained PEP model available. Call train() first."
            )

        best_model_path = Path(self._best_model_path)
        if not best_model_path.exists():
            raise FileNotFoundError(
                f"Best model file not found: {best_model_path}"
            )

        # Load metadata
        metadata_path = best_model_path.parent / 'best_model_info.json'
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Best model metadata not found: {metadata_path}"
            )

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        logger.info(
            f"Loading best PEP model from {best_model_path}: "
            f"layer={metadata['layer']}, "
            f"position={metadata['position']}, "
            f"target={metadata['target']}"
        )

        # Load model from disk
        model = PEPModel.load(
            path=str(best_model_path),
            model_wrapper=self.model_wrapper
        )

        logger.info("Successfully loaded PEP model as scorer")

        # PEPModel already implements ScorerInterface
        return model
