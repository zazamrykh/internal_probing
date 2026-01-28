"""
Base experiment runner with 5-phase architecture.

Provides modular experiment pipeline:
1. Model loading
2. Dataset loading
3. Dataset enrichment
4. Method application
5. Metrics computation
"""

import logging
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any

from src.config import ExperimentConfig
from src.dataset import BaseDataset, CorrectnessEvaluator
from src.model.base import ModelWrapper

logger = logging.getLogger(__name__)


class ExperimentPhase(str, Enum):
    """Experiment execution phases."""
    MODEL_LOADING = "model_loading"
    DATASET_LOADING = "dataset_loading"
    DATASET_ENRICHMENT = "dataset_enrichment"
    METHOD_APPLY = "method_apply"
    METRICS = "metrics"


class BaseExperimentRunner(ABC):
    """
    Base class for experiment runners with 5-phase architecture.

    Phases:
    1. Model loading - load and wrap LLM
    2. Dataset loading - create or load base datasets
    3. Dataset enrichment - apply enrichers (generation, activations, SE, etc.)
    4. Method application - apply uncertainty estimation method (with optional training)
    5. Metrics computation - evaluate and save results

    Subclasses can:
    - Override phase_* methods to customize behavior
    - Override should_run_phase() to skip phases
    - Add custom logic before/after phases

    Example:
        >>> config = ExperimentConfig.from_toml("config.toml")
        >>> runner = StandardExperiment(config)
        >>> runner.run()
    """

    def __init__(self, config: ExperimentConfig):
        """
        Initialize experiment runner.

        Args:
            config: Experiment configuration
        """
        self.config = config

        # Setup output directory
        output_dir = config.experiment['output_dir']
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # State variables - set during execution
        self.model_wrapper: Optional[ModelWrapper] = None
        self.evaluator = None
        self.train_data: Optional[BaseDataset] = None
        self.val_data: Optional[BaseDataset] = None
        self.test_data: Optional[BaseDataset] = None
        self.results: Optional[Dict[str, Any]] = None

        logger.info(f"Initialized {self.__class__.__name__}: {config.experiment.get('name')}")
        logger.info(f"Output directory: {self.output_dir}")

    def run(self):
        """
        Execute experiment pipeline.

        Runs phases in order, skipping those where should_run_phase() returns False.
        """
        logger.info("="*70)
        logger.info(f"Starting experiment: {self.config.experiment.get('name')}")
        logger.info(f"Runner: {self.__class__.__name__}")
        logger.info("="*70)

        # Phase 1: Model loading
        if self.should_run_phase(ExperimentPhase.MODEL_LOADING):
            logger.info(f"\n{'='*70}")
            logger.info("PHASE 1: Model Loading")
            logger.info(f"{'='*70}")
            self.phase_load_model()
        else:
            logger.info(f"\nSkipping phase: {ExperimentPhase.MODEL_LOADING}")

        # Phase 2: Dataset loading
        if self.should_run_phase(ExperimentPhase.DATASET_LOADING):
            logger.info(f"\n{'='*70}")
            logger.info("PHASE 2: Dataset Loading")
            logger.info(f"{'='*70}")
            self.phase_load_dataset()
        else:
            logger.info(f"\nSkipping phase: {ExperimentPhase.DATASET_LOADING}")

        # Phase 3: Dataset enrichment
        if self.should_run_phase(ExperimentPhase.DATASET_ENRICHMENT):
            logger.info(f"\n{'='*70}")
            logger.info("PHASE 3: Dataset Enrichment")
            logger.info(f"{'='*70}")
            self.phase_enrich_dataset()
        else:
            logger.info(f"\nSkipping phase: {ExperimentPhase.DATASET_ENRICHMENT}")

        # Phase 4: Method application
        if self.should_run_phase(ExperimentPhase.METHOD_APPLY):
            logger.info(f"\n{'='*70}")
            logger.info("PHASE 4: Method Application")
            logger.info(f"{'='*70}")
            self.phase_apply_method()
        else:
            logger.info(f"\nSkipping phase: {ExperimentPhase.METHOD_APPLY}")

        # Phase 5: Metrics computation
        if self.should_run_phase(ExperimentPhase.METRICS):
            logger.info(f"\n{'='*70}")
            logger.info("PHASE 5: Metrics Computation")
            logger.info(f"{'='*70}")
            self.phase_compute_metrics()
        else:
            logger.info(f"\nSkipping phase: {ExperimentPhase.METRICS}")

        logger.info(f"\n{'='*70}")
        logger.info("Experiment completed successfully!")
        logger.info(f"{'='*70}")

    def should_run_phase(self, phase: ExperimentPhase) -> bool:
        """
        Determine if a phase should be executed.

        Override in subclasses to skip phases.

        Args:
            phase: Phase to check

        Returns:
            True if phase should run, False to skip
        """
        return True

    def phase_load_model(self):
        """
        Phase 1: Load and wrap LLM model.

        Uses ModelWrapper.from_pretrained() factory method.
        Supports both new (model.name) and legacy (model.model_name_or_path) config formats.

        Sets:
        - self.model_wrapper
        - self.evaluator (for correctness evaluation)
        """
        model_config = self.config.model

        # Get model name - support both new and legacy formats
        model_name = model_config.get('name', model_config.get('model_name_or_path'))
        if not model_name:
            raise ValueError("model.name or model.model_name_or_path is required")

        # Create model wrapper using factory
        self.model_wrapper = ModelWrapper.from_pretrained(
            model_name=model_name,
            system_prompt=model_config.get('system_prompt', ''),
            quantization=model_config.get('quantization'),
            device_map=model_config.get('device_map', 'auto'),
        )

        # Create evaluator
        evaluator_type = model_config.get('evaluator', 'substring_match')
        self.evaluator = CorrectnessEvaluator.create(evaluator_type)

        logger.info(f"Model and evaluator loaded successfully")

    def phase_load_dataset(self):
        """
        Phase 2: Load or create base datasets.

        Supports:
        - Loading from pre-enriched files (dataset.enriched_path)
        - Creating from source (dataset.type)
        - Legacy format (dataset.dataset_type)

        Sets:
        - self.train_data
        - self.val_data
        - self.test_data
        """
        dataset_config = self.config.dataset

        # Check if pre-enriched datasets are provided
        if 'enriched_path' in dataset_config:
            logger.info("Loading pre-enriched datasets...")
            enriched_path = Path(dataset_config['enriched_path'])

            self.train_data = BaseDataset.load(enriched_path / 'train.pkl')
            self.val_data = BaseDataset.load(enriched_path / 'val.pkl')
            self.test_data = BaseDataset.load(enriched_path / 'test.pkl')

            logger.info(f"Loaded enriched datasets from {enriched_path}")
            logger.info(f"Datasets: train={len(self.train_data)}, "
                       f"val={len(self.val_data)}, test={len(self.test_data)}")
            return

        # Otherwise create from source
        logger.info("Creating datasets from source...")
        dataset_type = dataset_config.get('type', dataset_config.get('dataset_type'))
        if not dataset_type:
            raise ValueError("dataset.type or dataset.dataset_type is required")

        # Create dataset using factory
        dataset = BaseDataset.from_source(
            dataset_type=dataset_type,
            split=dataset_config.get('split', 'validation'),
            n_samples=dataset_config['n_samples'],
            seed=dataset_config.get('seed', 42),
            shuffle_buffer=dataset_config.get('shuffle_buffer', 10000)
        )

        # Split dataset
        n_train = dataset_config['n_train']
        n_val = dataset_config['n_val']
        n_test = dataset_config['n_test']

        self.train_data, self.val_data, self.test_data = dataset.split(
            n_train, n_val, n_test,
            seed=dataset_config.get('seed', 42)
        )

        logger.info(f"Created datasets: train={len(self.train_data)}, "
                   f"val={len(self.val_data)}, test={len(self.test_data)}")

    @abstractmethod
    def phase_enrich_dataset(self):
        """
        Phase 3: Apply enrichers to datasets.

        Enrichers add fields like:
        - greedy_answer, is_correct (generation enricher)
        - activations (activation enricher)
        - se_raw, se_binary, se_weight (semantic entropy enricher)

        Should update:
        - self.train_data
        - self.val_data
        - self.test_data
        """
        pass

    @abstractmethod
    def phase_apply_method(self):
        """
        Phase 4: Apply uncertainty estimation method.

        May include training (for probe-based methods).
        Should set:
        - self.results (method-specific results)
        """
        pass

    @abstractmethod
    def phase_compute_metrics(self):
        """
        Phase 5: Compute and save metrics.

        Should:
        - Evaluate method performance
        - Save results to output_dir
        - Create visualizations
        """
        pass
