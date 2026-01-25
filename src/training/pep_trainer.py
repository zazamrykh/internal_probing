"""
PEPTrainer - Trainer for Prompt Embedding Probe Model.

Implements end-to-end training of learnable prompt embeddings and linear probe
while keeping the base LLM frozen.
"""

import logging
import time
from pathlib import Path
from typing import Any, Union, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss
from tqdm import tqdm
import json

from src.dataset.base import BaseDataset
from src.training.base import BaseProbeTrainer
from src.training.utils import merge_datasets
from src.scoring.prompt_embedding import PEPModel
from src.model.base import ModelWrapper

logger = logging.getLogger(__name__)


class PEPTrainer(BaseProbeTrainer):
    """
    Trainer for Prompt Embedding Probe Model (PEPModel).

    Handles end-to-end training of learnable prompt embeddings and linear probe
    using PyTorch optimization. The base LLM remains frozen.

    Example:
        >>> trainer = PEPTrainer(
        ...     model_wrapper=wrapper,
        ...     n_embeddings=1,
        ...     learning_rate=1e-3,
        ...     n_epochs=10,
        ...     batch_size=8
        ... )
        >>>
        >>> # Train model
        >>> pep_model, metrics = trainer.train_cv(
        ...     train_data, val_data, test_data,
        ...     position=-2, layer=16, target_field='is_correct'
        ... )
    """

    def __init__(
        self,
        model_wrapper: ModelWrapper,
        n_embeddings: int = 1,
        learning_rate: float = 1e-3,
        n_epochs: int = 10,
        batch_size: int = 8,
        weight_decay: float = 0.0,
        optimizer_type: str = "adam",
        embedding_init_std: float = 0.02,
        seed: int = 42,
        device: Optional[str] = None,
        # Early stopping parameters
        early_stopping_patience: Optional[int] = None,
        early_stopping_metric: str = "auc",
        # Best model tracking
        save_best_model: bool = False,
        checkpoint_dir: Optional[str] = None,
        # Training limits
        max_samples: Optional[int] = None,
        max_training_time: Optional[float] = None,
        # Validation
        val_check_interval: int = 1,
        val_samples = None
    ):
        """
        Initialize PEP trainer.

        Args:
            model_wrapper: ModelWrapper with base LLM
            n_embeddings: Number of learnable soft prompt tokens
            learning_rate: Learning rate for optimization
            n_epochs: Number of training epochs (ignored if max_samples is set)
            batch_size: Batch size for training
            weight_decay: L2 regularization weight
            optimizer_type: Optimizer type ("adam", "sgd", "adamw")
            embedding_init_std: Std for embedding initialization
            seed: Random seed
            device: Device for training (None = use model's device)
            early_stopping_patience: Stop if metric doesn't improve for N checks (None = disabled)
            early_stopping_metric: Metric for early stopping ("auc" or "loss")
            save_best_model: If True, return best model instead of final
            checkpoint_dir: Directory for saving checkpoints (None = no checkpointing)
            max_samples: Maximum number of samples to train on (overrides n_epochs)
            max_training_time: Maximum training time in seconds (None = no limit)
            val_check_interval: Validate every N iterations/batches (for early stopping)
        """
        super().__init__(seed=seed)

        self.model_wrapper = model_wrapper
        self.n_embeddings = n_embeddings
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.optimizer_type = optimizer_type.lower()
        self.embedding_init_std = embedding_init_std
        self.device = device or model_wrapper.device

        # Early stopping
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_metric = early_stopping_metric

        # Best model tracking
        self.save_best_model = save_best_model
        self.checkpoint_dir = checkpoint_dir

        # Training limits
        self.max_samples = max_samples
        self.max_training_time = max_training_time
        self.val_check_interval = val_check_interval
        self.val_samples = val_samples

        if logger.isEnabledFor(logging.DEBUG):
            args = dict(locals())
            args.pop("self", None)

            args["model_wrapper"] = type(args["model_wrapper"]).__name__

            logger.debug("PEPTrainer.__init__ args=%s", args)


        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)

    def _evaluate_on_validation(
        self,
        model: PEPModel,
        val_dataloader: DataLoader,
        target_field: str,
        weight_field: Optional[str] = None,
        val_samples: int = None,
    ) -> dict[str, float]:
        """
        Evaluate model on validation set.

        Args:
            model: PEPModel to evaluate
            val_dataloader: Validation DataLoader
            target_field: Target field name
            weight_field: Optional sample weights field
            val_samples: If not None, evaluate on at most this many *examples* (not batches)

        Returns:
            Dictionary with 'loss' and 'auc' metrics
        """
        model.eval()
        all_preds = []
        all_targets = []
        total_loss = 0.0

        n_batches = 0
        n_examples_seen = 0

        with torch.no_grad():
            for batch in val_dataloader:
                # Stop early if we already evaluated enough examples
                if val_samples is not None and n_examples_seen >= val_samples:
                    break

                if weight_field:
                    input_ids, attention_mask, targets, weights = batch
                    weights = weights.to(self.device)
                else:
                    input_ids, attention_mask, targets = batch
                    weights = None

                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                targets = targets.to(self.device)

                # If we want only val_samples examples, cut the last batch
                if val_samples is not None:
                    remaining = val_samples - n_examples_seen
                    if remaining <= 0:
                        break
                    if targets.shape[0] > remaining:
                        input_ids = input_ids[:remaining]
                        attention_mask = attention_mask[:remaining]
                        targets = targets[:remaining]
                        if weights is not None:
                            weights = weights[:remaining]

                # Forward pass
                logits, _ = model.forward(
                    input_ids,
                    attention_mask=attention_mask,
                    return_activations=True,
                )
                logits = logits.squeeze(-1)

                # Compute loss
                if weights is not None:
                    criterion = nn.BCEWithLogitsLoss(weight=weights, reduction="mean")
                else:
                    criterion = nn.BCEWithLogitsLoss(reduction="mean")

                loss = criterion(logits, targets)
                total_loss += loss.item()
                n_batches += 1

                # Store predictions and targets for AUC
                probs = torch.sigmoid(logits)
                all_preds.extend(probs.detach().cpu().numpy().tolist())
                all_targets.extend(targets.detach().cpu().numpy().tolist())

                n_examples_seen += targets.shape[0]

        model.train()

        avg_loss = total_loss / n_batches if n_batches > 0 else float("inf")

        try:
            auc = roc_auc_score(all_targets, all_preds)
        except ValueError:
            # e.g. only one class present in all_targets in this truncated subset
            logger.warning("Cannot calculate roc auc during eval (not enough class variety)")
            auc = np.nan

        return {"loss": avg_loss, "auc": auc}


    def _save_checkpoint(
        self,
        model: PEPModel,
        checkpoint_path: Path,
        iteration: int,
        metrics: dict[str, float],
        config: Optional[dict] = None,
    ):
        """
        Save model checkpoint to disk.

        Args:
            model: PEPModel to save
            checkpoint_path: Path to save checkpoint
            iteration: Current iteration number
            metrics: Validation metrics
            config: Optional training config to save
        """
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'model_state_dict': model.state_dict(),
            'iteration': iteration,
            'metrics': metrics,
            'config': config or {},
        }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def _load_checkpoint(self, model: PEPModel, checkpoint_path: Path) -> dict:
        """
        Load model checkpoint from disk.

        Args:
            model: PEPModel to load weights into
            checkpoint_path: Path to checkpoint file

        Returns:
            Dictionary with checkpoint metadata
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded checkpoint from {checkpoint_path}")

        return {
            'iteration': checkpoint.get('iteration', 0),
            'metrics': checkpoint.get('metrics', {}),
            'config': checkpoint.get('config', {}),
        }

    def _create_optimizer(self, model: PEPModel) -> torch.optim.Optimizer:
        """Create optimizer for model parameters."""
        params = [
            {'params': model.prompt_embeddings, 'lr': self.learning_rate},
            {'params': model.probe.parameters(), 'lr': self.learning_rate},
        ]

        if self.optimizer_type == "adam":
            return optim.Adam(params, weight_decay=self.weight_decay)
        elif self.optimizer_type == "adamw":
            return optim.AdamW(params, weight_decay=self.weight_decay)
        elif self.optimizer_type == "sgd":
            return optim.SGD(params, weight_decay=self.weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_type}")

    def _prepare_dataloader(
        self,
        dataset: Union[BaseDataset, list[dict]],
        model: PEPModel,
        target_field: str,
        answer_field: str = 'greedy_answer',
        weight_field: Optional[str] = None,
        shuffle: bool = True,
    ) -> DataLoader:
        """
        Prepare DataLoader for training.

        Processes dataset: prompt + answer -> truncate to position -> create batches

        Returns:
            DataLoader with batches of (input_ids, targets, weights)
        """
        processed_sequences = []
        targets = []
        weights = [] if weight_field else None

        for sample in dataset:
            # Get prompt and answer
            prompt = sample['prompt']
            answer = sample[answer_field]

            # Tokenize
            inputs = self.model_wrapper.prepare_inputs([prompt])
            prompt_ids = inputs['input_ids'][0]  # (prompt_len,)

            answer_ids = self.model_wrapper.tokenizer.encode(
                answer,
                add_special_tokens=False,
                return_tensors='pt'
            )[0]  # (answer_len,)

            # Truncate to probe position
            truncated_seq = model._truncate_sequence_to_position(
                prompt_ids, answer_ids
            )

            processed_sequences.append(truncated_seq)
            targets.append(float(sample[target_field]))

            if weight_field and weight_field in sample:
                weights.append(float(sample[weight_field]))

        # Pad sequences to same length for batching
        max_len = max(len(seq) for seq in processed_sequences)
        padded_sequences = []
        attention_masks = []

        pad_token_id = self.model_wrapper.tokenizer.pad_token_id

        for seq in processed_sequences:
            pad_len = max_len - len(seq)
            # Pad on the left (for causal LM)
            padded = torch.cat([
                torch.full((pad_len,), pad_token_id, dtype=seq.dtype),
                seq
            ])
            mask = torch.cat([
                torch.zeros(pad_len, dtype=torch.long),
                torch.ones(len(seq), dtype=torch.long)
            ])
            padded_sequences.append(padded)
            attention_masks.append(mask)

        # Create tensors
        input_ids = torch.stack(padded_sequences)
        attention_mask = torch.stack(attention_masks)
        targets_tensor = torch.tensor(targets, dtype=torch.float32)

        if weights:
            weights_tensor = torch.tensor(weights, dtype=torch.float32)
            dataset_torch = TensorDataset(input_ids, attention_mask, targets_tensor, weights_tensor)
        else:
            dataset_torch = TensorDataset(input_ids, attention_mask, targets_tensor)

        return DataLoader(
            dataset_torch,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=False
        )

    def fit(
        self,
        dataset: Union[BaseDataset, list[dict]],
        target_field: str,
        position: int = None,
        layer: int = None,
        weight_field: Optional[str] = None,
        answer_field: str = 'greedy_answer',
        val_dataset: Optional[Union[BaseDataset, list[dict]]] = None,
        return_history: bool = False,
        verbose: bool = True,
        create_new_model = True,
        model = None,
        **kwargs
    ) -> tuple[PEPModel, dict]:
        """
        Train PEPModel on dataset with optional early stopping and best model tracking.

        Args:
            dataset: Training dataset with greedy answers
            position: Token position for probing
            layer: Layer index for probing
            target_field: Target field name (also determines probe_type)
            weight_field: Optional sample weights field
            answer_field: Field name containing generated answers
            val_dataset: Optional validation dataset for early stopping and history tracking
            return_history: If True, include training history in training_info['history']
            verbose: Whether to print progress
            create_new_model: Whether to create new model or fit given
            model: in case of create_new_model = False here you can give model to train
            **kwargs: Additional arguments

        Returns:
            Tuple of (trained_model, training_info_dict)

            training_info_dict contains: best_iteration, best_metrics, stopped_early, etc.
            If return_history=True, also includes 'history' key with list of validation checkpoints:
                [{'iteration': int, 'n_samples_seen': int, 'elapsed_time': float,
                  'val_auc': float, 'val_loss': float}, ...]
        """
        # Determine probe type from target field
        probe_type = "sep" if "se" in target_field else "accuracy"

        if create_new_model:
            assert layer is not None and position is not None, 'You have to specify layer and pos for create_new_model = True'
            # Create model
            model = PEPModel(
                model_wrapper=self.model_wrapper,
                n_embeddings=self.n_embeddings,
                probe_layer=layer,
                probe_position=position,
                probe_type=probe_type,
                embedding_init_std=self.embedding_init_std,
            )
        else:
            assert model is not None, 'You should specify model in case of "create_new_model = False"'
            
        model = model.to(self.device)
        model.train()

        # Prepare DataLoaders
        train_dataloader = self._prepare_dataloader(
            dataset, model, target_field, answer_field, weight_field, shuffle=True
        )

        val_dataloader = None
        if val_dataset is not None:
            val_dataloader = self._prepare_dataloader(
                val_dataset, model, target_field, answer_field, weight_field, shuffle=False
            )

        # Create optimizer
        optimizer = self._create_optimizer(model)

        # Setup checkpoint directory
        checkpoint_path = None
        if self.checkpoint_dir:
            checkpoint_path = Path(self.checkpoint_dir) / "best_model.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        # Training state
        best_metric = float('inf') if self.early_stopping_metric == 'loss' else -float('inf')
        best_iteration = 0
        patience_counter = 0
        stopped_early = False
        total_samples_seen = 0
        start_time = time.time()

        # History tracking
        history = [] if return_history else None

        # Calculate max iterations
        samples_per_epoch = len(dataset)
        if self.max_samples:
            logger.info(f'Train using max_samples for {self.max_samples} dataset points')
            max_iterations = (self.max_samples + self.batch_size - 1) // self.batch_size
            n_epochs = (self.max_samples + samples_per_epoch - 1) // samples_per_epoch
        else:
            max_iterations = float('inf')
            logger.info(f'Train using n_epochs for {self.n_epochs}')
            n_epochs = self.n_epochs

        # Training config for checkpoint
        training_config = {
            'n_embeddings': self.n_embeddings,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'position': position,
            'layer': layer,
            'target_field': target_field,
            'probe_type': probe_type,
        }

        # Training loop
        global_iteration = 0

        for epoch in range(n_epochs):
            epoch_loss = 0.0
            n_batches = 0

            iterator = train_dataloader
            if verbose:
                iterator = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{n_epochs}")

            for batch in iterator:
                # Check time limit
                if self.max_training_time:
                    elapsed = time.time() - start_time
                    if elapsed > self.max_training_time:
                        logger.info(f"Reached time limit ({self.max_training_time}s)")
                        stopped_early = True
                        break

                # Check sample limit
                if global_iteration >= max_iterations:
                    logger.info(f"Reached sample limit ({self.max_samples} samples)")
                    stopped_early = True
                    break

                if weight_field:
                    input_ids, attention_mask, targets, weights = batch
                    weights = weights.to(self.device)
                else:
                    input_ids, attention_mask, targets = batch
                    weights = None

                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                logits, _ = model.forward(
                    input_ids,
                    attention_mask=attention_mask,
                    return_activations=True
                )
                logits = logits.squeeze(-1)

                # Compute loss
                if weights is not None:
                    criterion = nn.BCEWithLogitsLoss(weight=weights, reduction='mean')
                else:
                    criterion = nn.BCEWithLogitsLoss(reduction='mean')

                loss = criterion(logits, targets)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1
                global_iteration += 1
                total_samples_seen += len(targets)

                # Validation check
                if val_dataloader and (global_iteration % self.val_check_interval == 0):
                    val_metrics = self._evaluate_on_validation(
                        model, val_dataloader, target_field, weight_field, val_samples = self.val_samples
                    )

                    current_metric = val_metrics[self.early_stopping_metric]

                    # Record history if requested
                    if return_history:
                        history_entry = {
                            'iteration': global_iteration,
                            'n_samples_seen': total_samples_seen,
                            'elapsed_time': time.time() - start_time,
                            'val_auc': val_metrics['auc'],
                            'val_loss': val_metrics['loss'],
                        }
                        history.append(history_entry)

                    if verbose:
                        logger.info(
                            f"Iteration {global_iteration}: "
                            f"val_loss={val_metrics['loss']:.4f}, "
                            f"val_auc={val_metrics['auc']:.4f}"
                        )

                    # Check if this is the best model
                    is_better = False
                    if self.early_stopping_metric == 'loss':
                        is_better = current_metric < best_metric
                    else:  # auc
                        is_better = current_metric > best_metric

                    if is_better:
                        best_metric = current_metric
                        best_iteration = global_iteration
                        patience_counter = 0

                        # Save best model
                        if self.save_best_model or checkpoint_path:
                            if checkpoint_path:
                                self._save_checkpoint(
                                    model, checkpoint_path, global_iteration,
                                    val_metrics, training_config
                                )
                    else:
                        patience_counter += 1

                    # Early stopping check
                    if self.early_stopping_patience and patience_counter >= self.early_stopping_patience:
                        logger.info(
                            f"Early stopping triggered at iteration {global_iteration} "
                            f"(patience={self.early_stopping_patience})"
                        )
                        stopped_early = True
                        break

            if stopped_early:
                break

            avg_loss = epoch_loss / n_batches if n_batches > 0 else 0.0
            if verbose:
                logger.info(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {avg_loss:.4f}")

        # Load best model if requested
        if self.save_best_model and checkpoint_path and checkpoint_path.exists():
            checkpoint_info = self._load_checkpoint(model, checkpoint_path)
            logger.info(
                f"Loaded best model from iteration {checkpoint_info['iteration']} "
                f"with metrics: {checkpoint_info['metrics']}"
            )

        model.eval()

        # Training info
        training_info = {
            'total_iterations': global_iteration,
            'total_samples_seen': total_samples_seen,
            'stopped_early': stopped_early,
            'best_iteration': best_iteration,
            'best_metric': best_metric,
            'training_time': time.time() - start_time,
        }

        # Add history if requested
        if return_history and history:
            training_info['history'] = history

        return model, training_info

    def predict(
        self,
        probe: PEPModel,
        dataset: Union[BaseDataset, list[dict]],
        position: int = None,
        layer: int = None,
        add_to_dataset: bool = True,
        prediction_field: str = "probe_prediction",
        return_proba: bool = True
    ) -> Union[np.ndarray, BaseDataset, list[dict]]:
        """
        Make predictions with trained PEPModel.

        Args:
            probe: Trained PEPModel
            dataset: Test dataset
            position: Token position (must match training)
            layer: Layer index (must match training)
            add_to_dataset: If True, add predictions to dataset
            prediction_field: Name for prediction field
            return_proba: If True, return probabilities

        Returns:
            If add_to_dataset=True: Modified dataset
            If add_to_dataset=False: Array of predictions
        """
        probe.eval()
        predictions = []

        with torch.no_grad():
            for sample in dataset:
                prompt = sample['prompt']
                answer = sample['greedy_answer']

                # Get score (already returns probability)
                score = probe.score_with_answer(prompt, answer)
                predictions.append(score)

        predictions = np.array(predictions)

        if add_to_dataset:
            from copy import deepcopy
            data_copy = []
            for i, ex in enumerate(dataset):
                ex_copy = deepcopy(ex)
                ex_copy[prediction_field] = float(predictions[i])
                data_copy.append(ex_copy)

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
        compute_metrics: bool = True,
        answer_field: str = 'greedy_answer',
        use_val_for_early_stopping: bool = False,
        **kwargs
    ) -> tuple[PEPModel, dict]:
        """
        Train PEPModel with K-fold cross-validation or simple train/val split.

        Two modes:
        1. use_val_for_early_stopping=False (default): True K-fold CV
           - Merges train+val into trainval
           - Performs K-fold CV on trainval to estimate generalization
           - Trains final model on full trainval
           - Evaluates on test

        2. use_val_for_early_stopping=True: Train/val split with early stopping
           - Trains on train_data
           - Validates on val_data for early stopping
           - No K-fold CV performed
           - Evaluates on test

        Args:
            train_data: Training dataset
            val_data: Validation dataset
            test_data: Test dataset
            position: Token position
            layer: Layer index
            target_field: Target field name
            k_folds: Number of CV folds (only used if use_val_for_early_stopping=False)
            weight_field: Optional sample weights
            compute_metrics: Whether to compute metrics
            answer_field: Field containing generated answers
            use_val_for_early_stopping: If True, use val_data for early stopping instead of CV
            **kwargs: Additional arguments (e.g., early_stopping_patience)

        Returns:
            Tuple of (trained_model, metrics_dict)
            metrics_dict contains:
                - cv_auc_mean, cv_auc_std (if CV performed)
                - test_auc, test_logloss
                - training_info from fit()
        """
        if use_val_for_early_stopping:
            # Mode 2: Train/val split with early stopping
            logger.info(f"Training PEPModel on {len(train_data)} samples with validation on {len(val_data)}")
            final_model, training_info = self.fit(
                train_data,
                position=position,
                layer=layer,
                target_field=target_field,
                weight_field=weight_field,
                answer_field=answer_field,
                val_dataset=val_data,
                verbose=True,
                **kwargs
            )

            metrics = {
                "position": position,
                "layer": layer,
                "target": target_field,
                "k_folds": 0,  # No CV performed
                "n_train": len(train_data),
                "n_val": len(val_data),
                "n_test": len(test_data),
                **training_info,
            }
        else:
            # Mode 1: True K-fold CV
            trainval_data = merge_datasets(train_data, val_data)
            logger.info(f"Performing {k_folds}-fold CV on {len(trainval_data)} samples")

            # Perform K-fold CV to estimate generalization
            cv_aucs = []
            cv_losses = []

            # Create stratified folds
            targets = np.array([s[target_field] for s in trainval_data])
            skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=self.seed)

            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets)):
                logger.info(f"Training fold {fold_idx + 1}/{k_folds}")

                # Split data
                fold_train = [trainval_data[i] for i in train_idx]
                fold_val = [trainval_data[i] for i in val_idx]

                # Train model on fold
                fold_model, _ = self.fit(
                    fold_train,
                    position=position,
                    layer=layer,
                    target_field=target_field,
                    weight_field=weight_field,
                    answer_field=answer_field,
                    val_dataset=None,  # No early stopping during CV
                    verbose=False,
                    **kwargs
                )

                # Evaluate on fold validation set
                fold_preds = self.predict(
                    fold_model, fold_val, position, layer,
                    add_to_dataset=False
                )
                fold_targets = np.array([s[target_field] for s in fold_val])

                try:
                    fold_auc = roc_auc_score(fold_targets, fold_preds)
                    cv_aucs.append(fold_auc)
                except ValueError:
                    pass

                try:
                    fold_loss = log_loss(fold_targets, fold_preds)
                    cv_losses.append(fold_loss)
                except ValueError:
                    pass

            # Train final model on full trainval
            logger.info(f"Training final model on {len(trainval_data)} samples")
            final_model, training_info = self.fit(
                trainval_data,
                position=position,
                layer=layer,
                target_field=target_field,
                weight_field=weight_field,
                answer_field=answer_field,
                val_dataset=None,
                verbose=True,
                **kwargs
            )

            metrics = {
                "position": position,
                "layer": layer,
                "target": target_field,
                "k_folds": k_folds,
                "n_trainval": len(trainval_data),
                "n_test": len(test_data),
                "cv_auc_mean": float(np.mean(cv_aucs)) if cv_aucs else np.nan,
                "cv_auc_std": float(np.std(cv_aucs)) if cv_aucs else np.nan,
                "cv_logloss_mean": float(np.mean(cv_losses)) if cv_losses else np.nan,
                "cv_logloss_std": float(np.std(cv_losses)) if cv_losses else np.nan,
                **training_info,
            }

        # Evaluate on test set
        if compute_metrics:
            test_preds = self.predict(
                final_model, test_data, position, layer,
                add_to_dataset=False
            )
            test_targets = np.array([s[target_field] for s in test_data])

            try:
                test_auc = roc_auc_score(test_targets, 1 - test_preds)  # Invert for correctness
                metrics['test_auc'] = float(test_auc)
            except ValueError:
                metrics['test_auc'] = np.nan

            try:
                test_logloss = log_loss(test_targets, 1 - test_preds)
                metrics['test_logloss'] = float(test_logloss)
            except ValueError:
                metrics['test_logloss'] = np.nan

        return final_model, metrics

    def train_every_combination(
        self,
        train_data: Union[BaseDataset, list[dict]],
        val_data: Union[BaseDataset, list[dict]],
        test_data: Union[BaseDataset, list[dict]],
        positions: list[int],
        layers: list[int],
        targets: list[str],
        k_folds: int = 5,
        weight_field: Optional[str] = None,
        use_weights_for_targets: Optional[str] = None,
        answer_field: str = 'greedy_answer',
        use_cv: bool = True,
        use_val_for_early_stopping: bool = True,
        verbose: bool = True,
        do_test_eval = True,
        **kwargs
    ) -> list[dict]:
        """
        Train PEP models for all combinations of positions, layers, and targets.

        Similar to ProbeManager.train_every_combination but for PEP models.

        Args:
            train_data: Training dataset
            val_data: Validation dataset
            test_data: Test dataset
            positions: List of token positions to try
            layers: List of layer indices to try
            targets: List of target fields to try
            k_folds: Number of CV folds (if use_cv=True)
            weight_field: Optional sample weights field
            use_weights_for_targets: Targets that should use weights
            answer_field: Field containing generated answers
            use_cv: Whether to use cross-validation
            verbose: Whether to print progress
            **kwargs: Additional arguments

        Returns:
            List of result dicts with keys: position, layer, target, probe, metrics
        """
        use_weights_for_targets = use_weights_for_targets or []
        results = []

        total = len(positions) * len(layers) * len(targets)
        current = 0

        for position in positions:
            for layer in layers:
                for target in targets:
                    current += 1
                    if verbose:
                        logger.info(
                            f"[{current}/{total}] Training PEP: "
                            f"pos={position}, layer={layer}, target={target}"
                        )

                    # Determine if we should use weights for this target
                    use_weights = target in use_weights_for_targets
                    current_weight_field = weight_field if use_weights else None

                    logger.debug(f"DEBUG: use_cv={use_cv}, use_val_for_early_stopping={use_val_for_early_stopping}")
                    logger.debug(f"DEBUG: train_data size={len(train_data)}, val_data size={len(val_data)}")

                    if use_cv:
                        # Train with CV
                        model, metrics = self.train_cv(
                            train_data, val_data, test_data,
                            position=position,
                            layer=layer,
                            target_field=target,
                            k_folds=k_folds,
                            weight_field=current_weight_field,
                            answer_field=answer_field,
                            compute_metrics=True,
                            use_val_for_early_stopping=use_val_for_early_stopping,
                            **kwargs
                        )
                    else:
                        # Train without CV - use train only, val for early stopping if requested
                        if use_val_for_early_stopping:
                            logger.debug(f"DEBUG: Training on train_data ({len(train_data)} samples) with val_data ({len(val_data)} samples) for early stopping")
                            # Train on train_data, validate on val_data
                            model, training_info = self.fit(
                                train_data,
                                position=position,
                                layer=layer,
                                target_field=target,
                                weight_field=current_weight_field,
                                answer_field=answer_field,
                                val_dataset=val_data,
                                verbose=verbose,
                                **kwargs
                            )
                        else:
                            # Train on merged train+val
                            from src.training.utils import merge_datasets
                            trainval = merge_datasets(train_data, val_data)
                            logger.debug(f"DEBUG: Training on merged trainval ({len(trainval)} samples)")

                            model, training_info = self.fit(
                                trainval,
                                position=position,
                                layer=layer,
                                target_field=target,
                                weight_field=current_weight_field,
                                answer_field=answer_field,
                                val_dataset=None,
                                verbose=verbose,
                                **kwargs
                            )

                        metrics = {
                            'position': position,
                            'layer': layer,
                            'target': target,
                            **training_info,  # Add training info
                        }

                        if not do_test_eval:
                            continue

                        # Compute test metrics
                        test_preds = self.predict(
                            model, test_data,
                            position=position, layer=layer,
                            add_to_dataset=False
                        )

                        test_targets = np.array([s[target] for s in test_data])

                        try:
                            test_auc = roc_auc_score(test_targets, 1 - test_preds)
                        except ValueError:
                            test_auc = np.nan

                        try:
                            test_logloss = log_loss(test_targets, 1 - test_preds)
                        except ValueError:
                            test_logloss = np.nan

                        metrics = metrics | {'test_auc': test_auc, 'test_logloss': test_logloss}

                    results.append({
                        'probe': model,
                        **metrics
                    })

        if verbose:
            logger.info(f"Trained {len(results)} PEP models")

        return results
