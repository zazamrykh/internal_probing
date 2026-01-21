"""
Factory for creating dataset enrichers from configuration.

Provides unified interface for instantiating enrichers based on type and config.
"""

from typing import Dict, Any, Optional

from src.config.enums import EnricherType
from src.dataset.enrichers.generation import GreedyGenerationEnricher
from src.dataset.enrichers.activation import ActivationEnricher
from src.dataset.enrichers.semantic_entropy import SemanticEntropyEnricher
from src.scoring.sampling.semantic_entropy import SemanticEntropyScorer
from src.model.base import ModelWrapper


class EnricherFactory:
    """
    Factory for creating dataset enrichers.

    Creates enrichers based on EnricherType and configuration dict.
    Handles different initialization requirements for each enricher type.

    Example:
        >>> config = {
        ...     'type': 'greedy_generation',
        ...     'max_new_tokens': 64
        ... }
        >>> enricher = EnricherFactory.create(
        ...     config, model_wrapper, evaluator
        ... )
    """

    @staticmethod
    def create(
        enricher_config: Dict[str, Any],
        model_wrapper: ModelWrapper,
        evaluator=None
    ):
        """
        Create enricher from configuration.

        Args:
            enricher_config: Configuration dict with 'type' and enricher-specific params
            model_wrapper: ModelWrapper instance (required for all enrichers)
            evaluator: Correctness evaluator (required for greedy_generation)

        Returns:
            Initialized enricher instance

        Raises:
            ValueError: If enricher type is unknown or required params missing
        """
        enricher_type = EnricherType(enricher_config['type'])

        if enricher_type == EnricherType.GREEDY_GENERATION:
            return EnricherFactory._create_greedy_generation(
                enricher_config, model_wrapper, evaluator
            )
        elif enricher_type == EnricherType.ACTIVATION:
            return EnricherFactory._create_activation(
                enricher_config, model_wrapper
            )
        elif enricher_type == EnricherType.SEMANTIC_ENTROPY:
            return EnricherFactory._create_semantic_entropy(
                enricher_config, model_wrapper
            )
        else:
            raise ValueError(f"Unknown enricher type: {enricher_type}")

    @staticmethod
    def _create_greedy_generation(
        config: Dict[str, Any],
        model_wrapper: ModelWrapper,
        evaluator
    ) -> GreedyGenerationEnricher:
        """Create GreedyGenerationEnricher from config."""
        if evaluator is None:
            raise ValueError(
                "evaluator is required for greedy_generation enricher"
            )

        return GreedyGenerationEnricher(
            model_wrapper=model_wrapper,
            evaluator=evaluator,
            max_new_tokens=config.get('max_new_tokens', 64),
            inplace=config.get('inplace', False),
            verbose=config.get('verbose', True)
        )

    @staticmethod
    def _create_activation(
        config: Dict[str, Any],
        model_wrapper: ModelWrapper
    ) -> ActivationEnricher:
        """Create ActivationEnricher from config."""
        if 'layers' not in config:
            raise ValueError("'layers' is required for activation enricher")
        if 'positions' not in config:
            raise ValueError("'positions' is required for activation enricher")

        return ActivationEnricher(
            model_wrapper=model_wrapper,
            layers=config['layers'],
            positions=config['positions'],
            inplace=config.get('inplace', False),
            verbose=config.get('verbose', True)
        )

    @staticmethod
    def _create_semantic_entropy(
        config: Dict[str, Any],
        model_wrapper: ModelWrapper
    ) -> SemanticEntropyEnricher:
        """Create SemanticEntropyEnricher from config."""
        # Create SemanticEntropyScorer
        scorer = SemanticEntropyScorer(
            entailment_model_name=config.get(
                'entailment_model',
                'roberta-large-mnli'
            ),
            n_samples=config.get('n_samples', 10),
            sampling_batch_size=config.get('sampling_batch_size', 1)
        )

        return SemanticEntropyEnricher(
            scorer=scorer,
            model_wrapper=model_wrapper,
            binarize=config.get('binarize', True),
            gamma=config.get('gamma', None),
            fit_gamma=config.get('fit_gamma', True),
            add_weights=config.get('add_weights', False),
            inplace=config.get('inplace', False),
            verbose=config.get('verbose', True)
        )
