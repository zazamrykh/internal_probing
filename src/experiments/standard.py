"""
Standard experiment runner with full pipeline.

Executes all 5 phases:
1. Model loading
2. Dataset loading
3. Dataset enrichment
4. Method application (with training)
5. Metrics computation
"""

import logging
import json
import time
from pathlib import Path
from typing import Dict, Any

from src.experiments.base import BaseExperimentRunner, ExperimentPhase
from src.config import MethodType
from src.dataset.enrichers import EnricherFactory
from src.experiments.methods.base import MethodFactory
from src.scoring.base import ScorerInterface
from src.scoring.inputs import SampleInput, ProbeInput, PromptEmbeddingInput
from src.scoring.precomputed import PrecomputedScorer
from src.scoring.linear_probe import LinearProbeScorer
from sklearn.metrics import roc_auc_score, accuracy_score
import torch
import numpy as np

logger = logging.getLogger(__name__)


class StandardExperiment(BaseExperimentRunner):
    """
    Standard experiment runner for complete pipeline.

    Supports:
    - Linear probes (sklearn-based)
    - PEP (Prompt Embedding Probes)
    - Sampling-based methods (Semantic Entropy, etc.)

    The method is determined by config.method.type and automatically:
    1. Trains if needed (probes)
    2. Selects best configuration (if multiple layers/positions)
    3. Computes metrics on test set

    Example config:
        [experiment]
        name = "pep_triviaqa"
        output_dir = "exp_results/pep_triviaqa"

        [model]
        name = "mistralai/Mistral-7B-Instruct-v0.1"

        [dataset]
        type = "triviaqa"
        n_samples = 100

        [enricher]
        [[enricher.pipeline]]
        type = "greedy_generation"

        [[enricher.pipeline]]
        type = "activation"
        layers = [0, 8, 16, 24, 31]
        positions = [0, -2]

        [method]
        type = "pep"
        layers = [0, 8, 16, 24, 31]
        positions = [0, -2]
        targets = ["is_correct"]
        n_embeddings = 1
        learning_rate = 0.001
        batch_size = 8
        use_best = true

        [metrics]
        compute_auc = true
        compute_accuracy = true
    """

    def __init__(self, config):
        super().__init__(config)
        self.scorer: ScorerInterface = None
        self.training_results: Dict[str, Any] = None
        self.phase_times: Dict[str, float] = {}

        # Validate config early
        self._validate_config()

    def _validate_config(self):
        """Validate configuration before running experiment."""
        errors = []

        # Validate method.type
        method_config = self.config.method
        method_type = method_config.get('type')
        if not method_type:
            errors.append("method.type is required")
        elif method_type not in [
            MethodType.LINEAR_PROBE, MethodType.PEP, MethodType.SEMANTIC_ENTROPY,
            'linear_probe', 'pep', 'semantic_entropy', 'se'
        ]:
            errors.append(
                f"Unsupported method.type: {method_type}. "
                f"Supported: linear_probe, pep, semantic_entropy"
            )

        # Validate method-specific parameters
        if method_type in [MethodType.LINEAR_PROBE, 'linear_probe']:
            if 'layers' not in method_config:
                errors.append("method.layers is required for linear_probe")
            if 'positions' not in method_config:
                errors.append("method.positions is required for linear_probe")
            if 'targets' not in method_config:
                errors.append("method.targets is required for linear_probe")

        elif method_type in [MethodType.PEP, 'pep']:
            if 'layers' not in method_config:
                errors.append("method.layers is required for pep")
            if 'positions' not in method_config:
                errors.append("method.positions is required for pep")
            if 'targets' not in method_config:
                errors.append("method.targets is required for pep")

        # semantic_entropy has no required parameters (all have defaults)

        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            raise ValueError(error_msg)

    def should_run_phase(self, phase: ExperimentPhase) -> bool:
        """Run all phases by default, but allow skipping via config."""
        # Check if enrichment should be skipped

        if phase == ExperimentPhase.MODEL_LOADING:
            if self.config.dataset.get('enriched_path', False) and self.config.method.get('type', False) == 'linear_probe':
                return False
            elif self.config.model.get('skip', False):
                return False

        if phase == ExperimentPhase.DATASET_ENRICHMENT:
            enricher_config = self.config.enricher if 'pipeline' in self.config.enricher else self.config.enrichment
            if enricher_config.get('skip', False):
                return False
            if 'pipeline' not in enricher_config or len(enricher_config['pipeline']) == 0:
                return False

        # Check if metrics should be skipped
        if phase == ExperimentPhase.METRICS:
            metrics_config = self.config.metrics
            if metrics_config.get('skip', False):
                return False

        return True

    def run(self):
        """Execute experiment pipeline with timing."""
        logger.info("="*70)
        logger.info(f"Starting experiment: {self.config.experiment.get('name')}")
        logger.info(f"Runner: {self.__class__.__name__}")
        logger.info("="*70)

        phases = [
            (ExperimentPhase.MODEL_LOADING, self.phase_load_model),
            (ExperimentPhase.DATASET_LOADING, self.phase_load_dataset),
            (ExperimentPhase.DATASET_ENRICHMENT, self.phase_enrich_dataset),
            (ExperimentPhase.METHOD_APPLY, self.phase_apply_method),
            (ExperimentPhase.METRICS, self.phase_compute_metrics),
        ]

        for phase, phase_func in phases:
            if self.should_run_phase(phase):
                logger.info(f"\n{'='*70}")
                logger.info(f"PHASE: {phase.value.upper()}")
                logger.info(f"{'='*70}")

                start_time = time.time()
                phase_func()
                elapsed = time.time() - start_time

                self.phase_times[phase.value] = elapsed
                logger.info(f"Phase completed in {elapsed:.2f}s")
            else:
                logger.info(f"\nSkipping phase: {phase.value}")

        logger.info(f"\n{'='*70}")
        logger.info("Experiment completed successfully!")
        logger.info(f"{'='*70}")
        logger.info("\nPhase timings:")
        for phase, elapsed in self.phase_times.items():
            logger.info(f"  {phase}: {elapsed:.2f}s")
        logger.info(f"  Total: {sum(self.phase_times.values()):.2f}s")

        # Save full config after experiment completion
        self._save_experiment_config()

    def _save_experiment_config(self):
        """Save the full experiment configuration to output directory."""
        config_path = self.output_dir / 'experiment_config.toml'
        try:
            self.config.save_to_toml(str(config_path))
            logger.info(f"Saved experiment config to {config_path}")
        except Exception as e:
            logger.warning(f"Failed to save experiment config: {e}")

    def phase_enrich_dataset(self):
        """Apply enrichers to datasets."""
        enricher_config = self.config.enricher if 'pipeline' in self.config.enricher else self.config.enrichment

        if 'pipeline' not in enricher_config:
            logger.warning("No enrichment pipeline specified")
            return

        pipeline = enricher_config['pipeline']
        logger.info(f"Applying {len(pipeline)} enrichers...")

        for i, enricher_spec in enumerate(pipeline):
            enricher_type = enricher_spec['type']
            logger.info(f"[{i+1}/{len(pipeline)}] Applying {enricher_type} enricher...")

            enricher = EnricherFactory.create(
                enricher_spec,
                self.model_wrapper,
                self.evaluator
            )

            self.train_data = enricher(self.train_data)
            self.val_data = enricher(self.val_data)
            self.test_data = enricher(self.test_data)

            logger.info(f"{enricher_type} enricher applied")

        # Save enriched datasets if requested
        save_enriched = enricher_config.get('save_enriched', False)
        if save_enriched:
            save_path = self.output_dir / 'enriched_datasets'
            save_path.mkdir(exist_ok=True)

            self.train_data.save(save_path / 'train.pkl')
            self.val_data.save(save_path / 'val.pkl')
            self.test_data.save(save_path / 'test.pkl')

            config_save_path = save_path / 'config.toml'
            self.config.save_to_toml(str(config_save_path))

            logger.info(f"Saved enriched datasets to {save_path}")

    def phase_apply_method(self):
        """Apply uncertainty estimation method (with training if needed)."""
        method_config = self.config.method
        method_type = method_config['type']  # Already validated

        logger.info(f"Applying method: {method_type}")

        # Create method using factory
        method = MethodFactory.create(
            method_type=method_type,
            method_config=method_config,
            model_wrapper=self.model_wrapper
        )

        # Train the method
        logger.info("Training method...")
        self.training_results = method.train(
            self.train_data,
            self.val_data
        )

        # Get scorer for inference
        self.scorer = method.get_scorer()

        logger.info("Method applied successfully")


    def phase_compute_metrics(self):
        """Compute and save metrics on test set."""
        logger.info("Computing metrics on test set...")

        metrics_config = self.config.metrics
        method_config = self.config.method
        method_type = method_config.get('type')

        # Determine target field for ground truth
        # For SE method, we compare against is_correct (uncertainty should predict errors)
        if method_type in [MethodType.SEMANTIC_ENTROPY, 'semantic_entropy', 'se']:
            target = method_config.get('target', 'is_correct')
        else:
            targets = method_config.get('targets', ['is_correct'])
            target = targets[0]  # Use first target for metrics

        logger.info(f"Using target field: '{target}' for metrics")

        # Collect predictions and ground truth
        predictions = []
        ground_truth = []

        for sample in self.test_data:
            # Get ground truth
            if target not in sample:
                logger.warning(f"Target '{target}' not found in sample, skipping")
                continue

            gt = sample[target]
            ground_truth.append(gt)

            # Get prediction from scorer
            scorer_input = self.scorer.get_input(sample)

            pred = self.scorer.estimate(scorer_input)
            predictions.append(pred)

        logger.info(f"Collected {len(predictions)} predictions")

        # Compute metrics
        metrics = {}

        if metrics_config.get('compute_auc', True) and len(set(ground_truth)) > 1:
            # For uncertainty scores: higher score = higher uncertainty = more likely incorrect
            # So we need to check if we should invert for AUC calculation
            # If target is is_correct (1=correct, 0=incorrect), and scorer returns
            # uncertainty (high=uncertain=likely incorrect), then:
            # - For predicting correctness: use 1 - predictions
            # - For predicting errors: use predictions directly

            # By convention, we compute AUC for predicting CORRECTNESS
            # So if scorer returns uncertainty, we invert
            auc_predictions = [1.0 - p for p in predictions]
            auc = roc_auc_score(ground_truth, auc_predictions)
            metrics['auc'] = float(auc)
            logger.info(f"AUC (predicting correctness): {auc:.4f}")

        if metrics_config.get('compute_accuracy', True):
            # Convert predictions to binary (threshold at 0.5)
            # Since predictions are uncertainty scores, high = uncertain = predict incorrect
            binary_preds = [0 if p > 0.5 else 1 for p in predictions]
            acc = accuracy_score(ground_truth, binary_preds)
            metrics['accuracy'] = float(acc)
            logger.info(f"Accuracy: {acc:.4f}")

        # Add phase timings to metrics
        metrics['phase_times'] = self.phase_times
        metrics['total_time'] = sum(self.phase_times.values())

        # Add method info
        metrics['method_type'] = method_type
        metrics['target'] = target
        metrics['n_test_samples'] = len(predictions)

        # Save metrics
        metrics_path = self.output_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Saved metrics to {metrics_path}")

        # Save training results if available
        if self.training_results:
            training_results_path = self.output_dir / 'training_results.json'
            with open(training_results_path, 'w') as f:
                json.dump(self.training_results, f, indent=2)
            logger.info(f"Saved training results to {training_results_path}")


if __name__ == '__main__':
    """
    Command-line interface for running standard experiments.

    Usage:
        python src/experiments/standard.py configs/examples/pep_triviaqa_new.toml
        python -m src.experiments.standard configs/examples/linear_probe_new.toml --log-level DEBUG
    """
    import sys
    import argparse
    from src.config import ExperimentConfig

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(
        description='Run standard experiment from TOML config'
    )
    parser.add_argument(
        'config',
        type=str,
        help='Path to TOML configuration file'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )

    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    try:
        config = ExperimentConfig.from_toml(args.config)
    except FileNotFoundError:
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    try:
        runner = StandardExperiment(config)
        runner.run()
        logger.info("Experiment completed successfully!")
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise
