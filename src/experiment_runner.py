"""
Experiment runner for internal probing experiments.

Provides unified interface for running complete experimental pipelines
from dataset loading through enrichment, training, and evaluation.
"""

import logging
import json
from pathlib import Path
from typing import Optional, Union, Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.config import ExperimentConfig, EnricherType, EvaluatorType
from src.dataset import BaseDataset, SubstringMatchEvaluator
from src.dataset.implementations import TriviaQADataset
from src.dataset.enrichers import EnricherFactory
from src.model import MistralModel, GPT2Model
from src.training import ProbeManager, PEPTrainer, save_probe, plot_auc_by_layer

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """
    Main class for running internal probing experiments.

    Handles complete pipeline:
    1. Load dataset
    2. Load model
    3. Apply enrichers sequentially
    4. Train probes (linear or PEP)
    5. Evaluate and save results

    Example:
        >>> config = ExperimentConfig.from_toml("configs/experiment.toml")
        >>> runner = ExperimentRunner(config)
        >>> runner.run()
    """

    def __init__(self, config: ExperimentConfig):
        """
        Initialize experiment runner.

        Args:
            config: Experiment configuration
        """
        self.config = config
        # Validate critical config parameters
        self._validate_config()

        output_dir = config.get('experiment.output_dir', 'exp_results', warn_on_fallback=True)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Will be set during run()
        self.model_wrapper = None
        self.evaluator = None
        self.train_data = None
        self.val_data = None
        self.test_data = None

        logger.info(f"Initialized ExperimentRunner: {config.experiment.get('name')}")
        logger.info(f"Output directory: {self.output_dir}")

    def _validate_config(self):
        """Validate that critical configuration parameters are present."""
        errors = []

        # Check experiment name
        if 'name' not in self.config.experiment:
            errors.append("experiment.name is required")

        # Check if only enrichment mode
        only_enrichment = self.config.get('experiment.only_enrichment', False)

        # Check dataset config (if not using pre-enriched)
        if 'enriched_path' not in self.config.dataset:
            if 'dataset_type' not in self.config.dataset:
                errors.append("dataset.dataset_type is required (or provide dataset.enriched_path)")

            # Check n_samples consistency only if all split sizes are provided
            if all(k in self.config.dataset for k in ['n_samples', 'n_train', 'n_val', 'n_test']):
                samples_sum = self.config.dataset['n_train'] + self.config.dataset['n_val'] + self.config.dataset['n_test']
                if self.config.dataset['n_samples'] != samples_sum:
                    errors.append("n_samples must be equal to sum of n_train, n_val and n_test")

        # Check model config
        if 'model_type' not in self.config.model:
            errors.append("model.model_type is required")
        if 'model_name_or_path' not in self.config.model:
            errors.append("model.model_name_or_path is required")

        # Only validate probe and training config if not in enrichment-only mode
        if not only_enrichment:
            # Check probe config
            if 'probe_type' not in self.config.probe:
                errors.append("probe.probe_type is required")

            # Check training config
            if 'positions' not in self.config.training:
                errors.append("training.positions is required")
            if 'layers' not in self.config.training:
                errors.append("training.layers is required")
            if 'targets' not in self.config.training:
                errors.append("training.targets is required")
        else:
            # In enrichment-only mode, ensure save_enriched is enabled
            if not self.config.get('enrichment.save_enriched', False):
                logger.warning("only_enrichment=True but enrichment.save_enriched=False. "
                             "Enriched datasets will not be saved!")

        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            raise ValueError(error_msg)

    def run(self):
        """
        Run complete experiment pipeline.

        Steps:
        1. Load or create datasets
        2. Load model
        3. Apply enrichment pipeline
        4. Train probes (if not only_enrichment)
        5. Evaluate and save results (if not only_enrichment)
        """
        logger.info("="*70)
        logger.info(f"Starting experiment: {self.config.experiment.get('name')}")
        logger.info("="*70)

        only_enrichment = self.config.get('experiment.only_enrichment', False)
        if only_enrichment:
            logger.info("Running in ENRICHMENT-ONLY mode")

        # Step 1: Load datasets
        self._load_or_create_datasets()

        # Step 2: Load model
        self._load_model()

        # Step 3: Apply enrichers
        if self.config.get('enrichment.skip', False):
            logger.info("Skipping enrichment (enrichment.skip=true)")
        else:
            self._apply_enrichers()

        # If only enrichment, stop here
        if only_enrichment:
            logger.info("="*70)
            logger.info("Enrichment completed successfully!")
            logger.info("="*70)
            return

        # Step 4: Train probes
        results = self._train_probes()

        # Step 5: Save and visualize
        self._save_and_visualize(results)

        logger.info("="*70)
        logger.info("Experiment completed successfully!")
        logger.info("="*70)

    def _load_or_create_datasets(self):
        """Load datasets from file or create from source."""
        dataset_config = self.config.dataset

        # Check if pre-enriched datasets are provided
        if 'enriched_path' in dataset_config:
            logger.info("Loading pre-enriched datasets...")
            self._load_enriched_datasets()
            return

        # Otherwise create from source
        logger.info("Creating datasets from source...")
        dataset_type = dataset_config['dataset_type']  # No fallback, validated

        if dataset_type == 'triviaqa':
            dataset = TriviaQADataset.from_huggingface(
                split=dataset_config.get('split', 'validation'),
                n_samples=dataset_config.get('n_samples', 100),
                seed=dataset_config.get('seed', 42),
                shuffle_buffer=dataset_config.get('shuffle_buffer', 10000)
            )
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

        # Split dataset
        n_train = dataset_config.get('n_train', 60, warn_on_fallback=True)
        n_val = dataset_config.get('n_val', 20, warn_on_fallback=True)
        n_test = dataset_config.get('n_test', 20, warn_on_fallback=True)

        self.train_data, self.val_data, self.test_data = dataset.split(
            n_train, n_val, n_test,
            seed=dataset_config.get('seed', 42)
        )

        logger.info(f"Created datasets: train={len(self.train_data)}, "
                   f"val={len(self.val_data)}, test={len(self.test_data)}")

        # Apply dataset cycling if requested (useful for batch size tuning)
        if 'cycle_train_to_size' in dataset_config:
            target_size = dataset_config['cycle_train_to_size']
            logger.info(f"Cycling train dataset to {target_size} samples")
            self.train_data = self.train_data.cycle_to_size(target_size)
            logger.info(f"Train dataset cycled: {len(self.train_data)} samples")

        if 'cycle_val_to_size' in dataset_config:
            target_size = dataset_config['cycle_val_to_size']
            logger.info(f"Cycling val dataset to {target_size} samples")
            self.val_data = self.val_data.cycle_to_size(target_size)
            logger.info(f"Val dataset cycled: {len(self.val_data)} samples")

        if 'cycle_test_to_size' in dataset_config:
            target_size = dataset_config['cycle_test_to_size']
            logger.info(f"Cycling test dataset to {target_size} samples")
            self.test_data = self.test_data.cycle_to_size(target_size)
            logger.info(f"Test dataset cycled: {len(self.test_data)} samples")

    def _load_enriched_datasets(self):
        """Load pre-enriched datasets from files."""
        enriched_path = Path(self.config.dataset['enriched_path'])

        self.train_data = BaseDataset.load(enriched_path / 'train.pkl')
        self.val_data = BaseDataset.load(enriched_path / 'val.pkl')
        self.test_data = BaseDataset.load(enriched_path / 'test.pkl')

        logger.info(f"Loaded enriched datasets from {enriched_path}")
        logger.info(f"Loaded datasets: train={len(self.train_data)}, "
                   f"val={len(self.val_data)}, test={len(self.test_data)}")

        # Apply dataset cycling if requested (useful for batch size tuning)
        dataset_config = self.config.dataset

        if 'cycle_train_to_size' in dataset_config:
            target_size = dataset_config['cycle_train_to_size']
            logger.info(f"Cycling train dataset to {target_size} samples")
            self.train_data = self.train_data.cycle_to_size(target_size)
            logger.info(f"Train dataset cycled: {len(self.train_data)} samples")

        if 'cycle_val_to_size' in dataset_config:
            target_size = dataset_config['cycle_val_to_size']
            logger.info(f"Cycling val dataset to {target_size} samples")
            self.val_data = self.val_data.cycle_to_size(target_size)
            logger.info(f"Val dataset cycled: {len(self.val_data)} samples")

        if 'cycle_test_to_size' in dataset_config:
            target_size = dataset_config['cycle_test_to_size']
            logger.info(f"Cycling test dataset to {target_size} samples")
            self.test_data = self.test_data.cycle_to_size(target_size)
            logger.info(f"Test dataset cycled: {len(self.test_data)} samples")

    def _load_model(self):
        """Load model and create wrapper."""
        model_config = self.config.model

        logger.info(f"Loading model: {model_config['model_type']}")

        # Setup quantization if requested
        quant_config = None
        if 'quantization' in model_config:
            quant_type = model_config['quantization']
            if quant_type == '8bit':
                quant_config = BitsAndBytesConfig(load_in_8bit=True)
            elif quant_type == '4bit':
                quant_config = BitsAndBytesConfig(load_in_4bit=True)

        # Load model and tokenizer
        model_name = model_config['model_name_or_path']  # No fallback, validated

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=model_config.get('device_map', 'auto'),
            quantization_config=quant_config
        )

        # Setup padding
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
        if hasattr(model, 'generation_config'):
            model.generation_config.pad_token_id = tokenizer.pad_token_id

        # Create wrapper
        model_type = model_config['model_type']  # No fallback, validated
        system_prompt = model_config.get('system_prompt', '', warn_on_fallback=True)

        if model_type == 'mistral' or 'mistral' in model_name.lower():
            self.model_wrapper = MistralModel(
                model=model,
                tokenizer=tokenizer,
                system_prompt=system_prompt
            )
        elif model_type == 'gpt2' or 'gpt2' in model_name.lower():
            self.model_wrapper = GPT2Model(
                model=model,
                tokenizer=tokenizer,
                system_prompt=system_prompt
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        logger.info(f"Model loaded: {model_type}")

        # Create evaluator
        evaluator_type = model_config.get('evaluator', 'substring_match')
        if evaluator_type == 'substring_match':
            self.evaluator = SubstringMatchEvaluator()
        else:
            raise ValueError(f"Unknown evaluator type: {evaluator_type}")

    def _apply_enrichers(self):
        """Apply enrichment pipeline to datasets."""
        enrichment_config = self.config.enrichment

        if 'pipeline' not in enrichment_config:
            logger.warning("No enrichment pipeline specified")
            return

        pipeline = enrichment_config['pipeline']
        logger.info(f"Applying {len(pipeline)} enrichers...")

        for i, enricher_config in enumerate(pipeline):
            enricher_type = enricher_config['type']
            logger.info(f"[{i+1}/{len(pipeline)}] Applying {enricher_type} enricher...")

            # Create enricher
            enricher = EnricherFactory.create(
                enricher_config,
                self.model_wrapper,
                self.evaluator
            )

            # Apply to all datasets
            # Enrichers now return dataset of same type as input
            self.train_data = enricher(self.train_data)
            self.val_data = enricher(self.val_data)
            self.test_data = enricher(self.test_data)

            logger.info(f"{enricher_type} enricher applied")

        # Save enriched datasets if requested
        # enrichment_config is ConfigDict, not the pipeline list
        if self.config.enrichment.get('save_enriched', False):
            save_path = self.output_dir / 'enriched_datasets'
            save_path.mkdir(exist_ok=True)

            self.train_data.save(save_path / 'train.pkl')
            self.val_data.save(save_path / 'val.pkl')
            self.test_data.save(save_path / 'test.pkl')

            # Save config alongside enriched datasets
            config_save_path = save_path / 'config.toml'
            self.config.save_to_toml(str(config_save_path))

            logger.info(f"Saved enriched datasets to {save_path}")
            logger.info(f"Saved config to {config_save_path}")

    def _train_probes(self):
        """Train probes based on configuration."""
        probe_config = self.config.probe
        training_config = self.config.training

        probe_type = probe_config['probe_type']  # No fallback, validated

        if probe_type == 'linear':
            return self._train_linear_probes(probe_config, training_config)
        elif probe_type == 'pep' or probe_type == 'prompt_embedding':
            return self._train_pep_probes(probe_config, training_config)
        else:
            raise ValueError(f"Unknown probe type: {probe_type}")

    def _train_linear_probes(self, probe_config, training_config):
        """Train linear probes using ProbeManager."""
        from sklearn.linear_model import LogisticRegression

        logger.info("Training linear probes...")

        # Create ProbeManager
        probe_params = training_config.get('probe_params', {})
        probe_params.setdefault('C', 1.0)
        probe_params.setdefault('max_iter', 500)
        probe_params.setdefault('solver', 'lbfgs')

        manager = ProbeManager(
            probe_class=LogisticRegression,
            probe_params=probe_params,
            seed=self.config.get('experiment.seed', 42)
        )

        # Train for all combinations
        k_folds = training_config.get('k_folds', 5, warn_on_fallback=True)

        results = manager.train_every_combination(
            self.train_data,
            self.val_data,
            self.test_data,
            positions=training_config['positions'],  # No fallback, validated
            layers=training_config['layers'],  # No fallback, validated
            targets=training_config['targets'],  # No fallback, validated
            k_folds=k_folds,
            weight_field=training_config.get('weight_field'),
            use_weights_for_targets=training_config.get('use_weights_for_targets', []),
            eval=True,
            verbose=True
        )

        logger.info(f"Trained {len(results)} linear probes")
        return results

    def _train_pep_probes(self, probe_config, training_config):
        """Train PEP probes using PEPTrainer."""
        logger.info("Training PEP probes...")

        # Create PEPTrainer
        pep_params = training_config.get('pep_params', {})

        # Auto-generate checkpoint_dir if not specified but save_best_model is enabled
        checkpoint_dir = pep_params.get('checkpoint_dir')
        if pep_params.get('save_best_model', False) and not checkpoint_dir:
            checkpoint_dir = str(self.output_dir / 'checkpoints')
            logger.info(f"Auto-generated checkpoint_dir: {checkpoint_dir}")

        trainer = PEPTrainer(
            model_wrapper=self.model_wrapper,
            n_embeddings=pep_params.get('n_embeddings', 1),
            learning_rate=pep_params.get('learning_rate', 1e-3),
            n_epochs=pep_params.get('n_epochs', 10),
            batch_size=pep_params.get('batch_size', 8),
            weight_decay=pep_params.get('weight_decay', 0.0),
            optimizer_type=pep_params.get('optimizer', 'adam'),
            seed=self.config.get('experiment.seed', 42),
            # Early stopping parameters
            early_stopping_patience=pep_params.get('early_stopping_patience'),
            early_stopping_metric=pep_params.get('early_stopping_metric', 'auc'),
            # Best model tracking
            save_best_model=pep_params.get('save_best_model', False),
            checkpoint_dir=checkpoint_dir,
            # Training limits
            max_samples=pep_params.get('max_samples'),
            max_training_time=pep_params.get('max_training_time'),
            val_check_interval=pep_params.get('val_check_interval', 1),
        )

        # Use train_every_combination from PEPTrainer
        k_folds = training_config.get('k_folds', 5, warn_on_fallback=True)
        use_cv = training_config.get('use_cv', False, warn_on_fallback=True)

        # Check if we should use validation for early stopping
        # Read from training config, or auto-detect if early stopping/best model is enabled
        use_val_for_early_stopping = training_config.get('use_val_for_early_stopping')
        if use_val_for_early_stopping is None:
            # Auto-detect: use val if early stopping or save_best_model is enabled
            use_val_for_early_stopping = (
                pep_params.get('early_stopping_patience') is not None
                or pep_params.get('save_best_model', False)
            )

        # Check if return_history is requested
        return_history = pep_params.get('return_history', False)

        results = trainer.train_every_combination(
            self.train_data,
            self.val_data,
            self.test_data,
            positions=training_config['positions'],  # No fallback, validated
            layers=training_config['layers'],  # No fallback, validated
            targets=training_config['targets'],  # No fallback, validated
            k_folds=k_folds,
            weight_field=training_config.get('weight_field'),
            use_weights_for_targets=training_config.get('use_weights_for_targets', []),
            use_cv=use_cv,
            use_val_for_early_stopping=use_val_for_early_stopping,
            return_history=return_history,
            verbose=True,
            do_test_eval=self.config.training.get('do_test_eval', True)
        )

        logger.info(f"Trained {len(results)} PEP probes")
        return results

    def _save_and_visualize(self, results):
        """Save results and create visualizations."""
        logger.info("Saving results and creating visualizations...")

        # Save probes
        probes_dir = self.output_dir / 'probes'
        probes_dir.mkdir(exist_ok=True)

        for i, result in enumerate(results):
            probe_name = f"probe_pos{result['position']}_layer{result['layer']}_{result['target']}"
            probe = result['probe']

            # Extract metrics without probe object
            metrics_only = {k: v for k, v in result.items() if k != 'probe'}

            # Check if it's PEPModel or sklearn probe
            from src.scoring.prompt_embedding import PEPModel
            if isinstance(probe, PEPModel):
                # Save PEPModel
                probe.save(probes_dir / probe_name)
                # Also save metrics separately
                with open(probes_dir / f"{probe_name}_metrics.json", 'w') as f:
                    json.dump(metrics_only, f, indent=2)
            else:
                # Save sklearn probe using save_probe
                save_probe(
                    probe,
                    metrics_only,
                    probes_dir / probe_name
                )

        # Save metrics as JSON
        metrics = [{k: v for k, v in r.items() if k != 'probe'} for r in results]
        with open(self.output_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Saved {len(results)} probes to {probes_dir}")

        # Create visualizations if multiple layers
        layers = list(set(r['layer'] for r in results))
        targets = list(set(r['target'] for r in results))

        if len(layers) > 1:
            for target in targets:
                try:
                    plot_auc_by_layer(
                        results,
                        target_field=target,
                        title=f'Test AUC vs Layer ({target})'
                    )
                    import matplotlib.pyplot as plt
                    plt.savefig(self.output_dir / f'auc_by_layer_{target}.png')
                    plt.close()
                    logger.info(f"Saved visualization: auc_by_layer_{target}.png")
                except Exception as e:
                    logger.warning(f"Failed to create visualization for {target}: {e}")



if __name__ == '__main__':
    """
    Command-line interface for running experiments.

    Usage:
        python src/experiment_runner.py configs/experiments/sep_linear_triviaqa.toml
        python -m src.experiment_runner configs/experiments/pep_triviaqa_example.toml
    """
    import sys
    import argparse

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Run internal probing experiments from TOML config'
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

    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Load config
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

    # Run experiment
    try:
        runner = ExperimentRunner(config)
        runner.run()
        logger.info("Experiment completed successfully!")
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        # Re-raise to show full traceback
        raise
