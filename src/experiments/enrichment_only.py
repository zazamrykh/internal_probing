"""
Enrichment-only experiment runner.

Runs only model loading, dataset loading, and enrichment phases.
Useful for pre-processing datasets that can be reused in multiple experiments.
"""

import logging
from pathlib import Path

from src.experiments.base import BaseExperimentRunner, ExperimentPhase
from src.dataset.enrichers import EnricherFactory

logger = logging.getLogger(__name__)


class EnrichmentOnlyExperiment(BaseExperimentRunner):
    """
    Experiment runner for dataset enrichment only.

    Executes phases:
    1. Model loading
    2. Dataset loading
    3. Dataset enrichment (with saving)

    Skips:
    4. Method application
    5. Metrics computation

    Example config:
        [experiment]
        name = "enrich_triviaqa"
        output_dir = "exp_results/enriched"

        [model]
        name = "mistralai/Mistral-7B-Instruct-v0.1"
        quantization = "8bit"
        system_prompt = "Answer briefly."
        evaluator = "substring_match"

        [dataset]
        type = "triviaqa"
        n_samples = 100
        n_train = 60
        n_val = 20
        n_test = 20

        [enricher]
        save_enriched = true

        [[enricher.pipeline]]
        type = "greedy_generation"

        [[enricher.pipeline]]
        type = "activation"
        layers = [0, 8, 16, 24, 31]
        positions = [0, -2]
    """

    def should_run_phase(self, phase: ExperimentPhase) -> bool:
        """Only run model loading, dataset loading, and enrichment phases."""
        return phase in [
            ExperimentPhase.MODEL_LOADING,
            ExperimentPhase.DATASET_LOADING,
            ExperimentPhase.DATASET_ENRICHMENT,
        ]

    # Inherit phase_load_model and phase_load_dataset from BaseExperimentRunner

    def phase_enrich_dataset(self):
        """Apply enrichers to datasets and save."""
        # Support both new 'enricher' and legacy 'enrichment' sections
        enricher_config = self.config.enricher if 'pipeline' in self.config.enricher else self.config.enrichment

        if 'pipeline' not in enricher_config:
            logger.warning("No enrichment pipeline specified")
            return

        pipeline = enricher_config['pipeline']
        logger.info(f"Applying {len(pipeline)} enrichers...")

        for i, enricher_spec in enumerate(pipeline):
            enricher_type = enricher_spec['type']
            logger.info(f"[{i+1}/{len(pipeline)}] Applying {enricher_type} enricher...")

            # Create enricher
            enricher = EnricherFactory.create(
                enricher_spec,
                self.model_wrapper,
                self.evaluator
            )

            # Apply to all datasets
            self.train_data = enricher(self.train_data)
            self.val_data = enricher(self.val_data)
            self.test_data = enricher(self.test_data)

            logger.info(f"{enricher_type} enricher applied")

        # Save enriched datasets
        save_enriched = enricher_config.get('save_enriched', True)
        if save_enriched:
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
        else:
            logger.warning("save_enriched=False, enriched datasets will not be saved!")

    def phase_apply_method(self):
        """Not used in enrichment-only mode."""
        pass

    def phase_compute_metrics(self):
        """Not used in enrichment-only mode."""
        pass


if __name__ == '__main__':
    """
    Command-line interface for running enrichment-only experiments.

    Usage:
        python src/experiments/enrichment_only.py configs/examples/enrichment_only_new.toml
        python -m src.experiments.enrichment_only configs/examples/enrichment_only_new.toml --log-level DEBUG
    """
    import sys
    import argparse
    from src.config import ExperimentConfig

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Run enrichment-only experiment from TOML config'
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
        runner = EnrichmentOnlyExperiment(config)
        runner.run()
        logger.info("Enrichment completed successfully!")
    except Exception as e:
        logger.error(f"Enrichment failed: {e}")
        # Re-raise to show full traceback
        raise
