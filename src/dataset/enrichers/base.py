"""
Base enricher class for dataset enrichment operations.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
from copy import deepcopy


class BaseEnricher(ABC):
    """
    Base class for dataset enrichers.

    Enrichers add computed fields to dataset samples, such as:
    - Model generations (greedy, sampling)
    - Hidden state activations
    - Uncertainty scores (semantic entropy, etc.)

    Enrichers operate on individual samples or batches and return
    modified copies without changing the original data.
    """

    def __init__(self, inplace: bool = False, verbose: bool = True):
        """
        Initialize enricher.

        Args:
            inplace: If True, modify samples in-place. If False, create copies.
            verbose: If True, print progress information
        """
        self.inplace = inplace
        self.verbose = verbose

    @abstractmethod
    def enrich_sample(self, sample: dict, **kwargs) -> dict:
        """
        Enrich a single sample with computed fields.

        Args:
            sample: Input sample dict
            **kwargs: Additional parameters for enrichment

        Returns:
            Enriched sample dict (copy or modified in-place)
        """
        pass

    def enrich_dataset(
        self,
        dataset,
        verbose_every: Optional[int] = 100,
        **kwargs
    ):
        """
        Enrich entire dataset.

        Args:
            dataset: Dataset object or list of samples
            verbose_every: Print progress every N samples (None to disable)
            **kwargs: Additional parameters passed to enrich_sample

        Returns:
            Dataset of same type as input (or list if input was list)
        """
        # Handle both dataset objects and raw lists
        is_dataset_object = hasattr(dataset, '_data')
        if is_dataset_object:
            samples = dataset._data
            dataset_class = type(dataset)
        elif isinstance(dataset, list):
            samples = dataset
            dataset_class = None
        else:
            raise ValueError(f"Unsupported dataset type: {type(dataset)}")

        # Determine if we should modify in-place
        if self.inplace:
            output = samples
        else:
            output = deepcopy(samples)

        # Enrich each sample
        for i, sample in enumerate(output):
            enriched = self.enrich_sample(sample, **kwargs)

            # Update sample with enriched fields
            if not self.inplace:
                output[i] = enriched

            # Progress logging
            if self.verbose and verbose_every and (i + 1) % verbose_every == 0:
                print(f"[{self.__class__.__name__}] Processed {i+1}/{len(output)} samples")

        # Return same type as input
        if is_dataset_object:
            return dataset_class(output)
        else:
            return output

    def __call__(self, dataset, **kwargs):
        """Allow enricher to be called as a function."""
        return self.enrich_dataset(dataset, **kwargs)
