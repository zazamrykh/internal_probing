"""
Semantic Entropy Method for uncertainty estimation.

This method supports two modes:
1. Precomputed mode (use_precomputed=True): Uses SE values computed during enrichment
2. On-the-fly mode (use_precomputed=False): Computes SE during inference
"""

import logging
from typing import Dict, Any, Optional

from src.experiments.methods.base import MethodInterface
from src.scoring.base import ScorerInterface
from src.scoring.precomputed import PrecomputedScorer
from src.scoring.sampling.semantic_entropy import SemanticEntropyScorer
from src.model.base import ModelWrapper

logger = logging.getLogger(__name__)


class SemanticEntropyMethod(MethodInterface):
    """
    Semantic Entropy method for uncertainty estimation.

    Supports two modes controlled by `use_precomputed` config parameter:

    1. **Precomputed mode** (use_precomputed=True, default):
       - Uses SE values computed during the enrichment phase
       - Returns PrecomputedScorer that extracts `se_raw` from samples
       - Fast inference, requires prior enrichment with SemanticEntropyEnricher

    2. **On-the-fly mode** (use_precomputed=False):
       - Computes SE during inference by generating samples and clustering
       - Returns SemanticEntropyScorer that generates and clusters responses
       - Slower but doesn't require prior enrichment

    Example config:
        [method]
        type = "semantic_entropy"
        use_precomputed = true  # Use values from enrichment
        field_name = "se_raw"   # Field to extract (precomputed mode)

        # On-the-fly mode parameters (when use_precomputed = false):
        n_samples = 10
        entailment_model = "roberta-large-mnli"
        strict_entailment = false
    """

    def __init__(
        self,
        method_config: Dict[str, Any],
        model_wrapper: Optional[ModelWrapper] = None,
        **kwargs
    ):
        """
        Initialize Semantic Entropy method.

        Args:
            method_config: Configuration dictionary with keys:
                - use_precomputed: bool (default: True) - whether to use precomputed values
                - field_name: str (default: "se_raw") - field name for precomputed mode
                - n_samples: int (default: 10) - number of samples for on-the-fly mode
                - entailment_model: str (default: "roberta-large-mnli") - entailment model
                - strict_entailment: bool (default: False) - strict entailment clustering
                - sampling_batch_size: int (default: 1) - batch size for sampling
            model_wrapper: ModelWrapper instance (required for on-the-fly mode)
            **kwargs: Additional arguments (ignored)
        """
        self.config = method_config
        self.model_wrapper = model_wrapper

        # Mode selection
        self.use_precomputed = method_config.get('use_precomputed', True)

        # Precomputed mode parameters
        self.field_name = method_config.get('field_name', 'se_raw')

        # On-the-fly mode parameters
        self.n_samples = method_config.get('n_samples', 10)
        self.entailment_model = method_config.get('entailment_model', 'roberta-large-mnli')
        self.strict_entailment = method_config.get('strict_entailment', False)
        self.sampling_batch_size = method_config.get('sampling_batch_size', 1)

        # Scorer will be created in get_scorer()
        self._scorer: Optional[ScorerInterface] = None

        logger.info(
            f"Initialized SemanticEntropyMethod: "
            f"use_precomputed={self.use_precomputed}, "
            f"field_name={self.field_name if self.use_precomputed else 'N/A'}"
        )

    def train(self, train_data, val_data, **kwargs) -> Dict[str, Any]:
        """
        No training required for Semantic Entropy method.

        This method doesn't require training - it either uses precomputed values
        or computes SE on-the-fly during inference.

        Args:
            train_data: Training dataset (unused)
            val_data: Validation dataset (unused)
            **kwargs: Additional arguments (unused)

        Returns:
            Empty dict (no training results)
        """
        logger.info("SemanticEntropyMethod: No training required")

        # Validate configuration
        if not self.use_precomputed and self.model_wrapper is None:
            raise ValueError(
                "model_wrapper is required for on-the-fly SE computation "
                "(use_precomputed=False)"
            )

        return {
            "method": "semantic_entropy",
            "use_precomputed": self.use_precomputed,
            "training_required": False,
        }

    def get_scorer(self) -> ScorerInterface:
        """
        Get scorer for inference.

        Returns:
            - PrecomputedScorer if use_precomputed=True
            - SemanticEntropyScorer if use_precomputed=False
        """
        if self._scorer is not None:
            return self._scorer

        if self.use_precomputed:
            logger.info(f"Creating PrecomputedScorer with field_name='{self.field_name}'")
            self._scorer = PrecomputedScorer(field_name=self.field_name)
        else:
            logger.info(
                f"Creating SemanticEntropyScorer: "
                f"n_samples={self.n_samples}, "
                f"entailment_model={self.entailment_model}"
            )
            self._scorer = SemanticEntropyScorer(
                entailment_model_name=self.entailment_model,
                n_samples=self.n_samples,
                sampling_batch_size=self.sampling_batch_size,
                strict_entailment=self.strict_entailment,
            )

        return self._scorer
