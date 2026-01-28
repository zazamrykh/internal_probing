"""
Base interface and factory for uncertainty estimation methods.

Methods encapsulate the logic for:
1. Training (if needed) - e.g., training probes
2. Getting a scorer for inference - returns best scorer after training

This separates method-specific logic from experiment runners.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import logging

from src.scoring.base import ScorerInterface

logger = logging.getLogger(__name__)


class MethodInterface(ABC):
    """
    Base interface for uncertainty estimation methods.

    Each method implements:
    - train(): Train the method (if applicable)
    - get_scorer(): Get scorer for inference/metrics

    Examples:
    - LinearProbeMethod: trains sklearn probes, returns best probe scorer
    - PEPMethod: trains PEP models, returns best PEP scorer
    - SemanticEntropyMethod: no training, returns SE scorer directly
    """

    @abstractmethod
    def train(self, train_data, val_data, **kwargs) -> Dict[str, Any]:
        """
        Train the method (if applicable).

        Args:
            train_data: Training dataset
            val_data: Validation dataset
            **kwargs: Method-specific training parameters

        Returns:
            Dictionary with training results/metadata
        """
        pass

    @abstractmethod
    def get_scorer(self) -> ScorerInterface:
        """
        Get scorer for inference.

        For methods with multiple configurations (e.g., different layers/positions),
        returns the best one based on validation metrics.

        Returns:
            ScorerInterface instance ready for inference
        """
        pass


class MethodFactory:
    """
    Factory for creating uncertainty estimation methods.

    Creates appropriate method instance based on method_type.
    """

    @staticmethod
    def create(
        method_type: str,
        method_config: Dict[str, Any],
        model_wrapper=None,
        **kwargs
    ) -> MethodInterface:
        """
        Create method instance from configuration.

        Args:
            method_type: Type of method ("linear_probe", "pep", etc.)
            method_config: Method configuration dictionary
            model_wrapper: Model wrapper (required for some methods)
            **kwargs: Additional arguments

        Returns:
            MethodInterface instance

        Example:
            >>> method = MethodFactory.create(
            ...     method_type="pep",
            ...     method_config=config.method,
            ...     model_wrapper=model_wrapper
            ... )
            >>> results = method.train(train_data, val_data)
            >>> scorer = method.get_scorer()
        """
        # Import here to avoid circular imports
        from src.experiments.methods.linear_probe import LinearProbeMethod
        from src.experiments.methods.pep import PEPMethod
        from src.experiments.methods.semantic_entropy import SemanticEntropyMethod

        if method_type in ['linear_probe', 'linear']:
            return LinearProbeMethod(
                method_config=method_config,
                **kwargs
            )
        elif method_type == 'pep':
            if model_wrapper is None:
                raise ValueError("PEPMethod requires model_wrapper")
            return PEPMethod(
                method_config=method_config,
                model_wrapper=model_wrapper,
                **kwargs
            )
        elif method_type in ['semantic_entropy', 'se']:
            return SemanticEntropyMethod(
                method_config=method_config,
                model_wrapper=model_wrapper,  # Optional for precomputed mode
                **kwargs
            )
        else:
            raise ValueError(
                f"Unknown method type: '{method_type}'. "
                f"Supported: linear_probe, pep, semantic_entropy"
            )
