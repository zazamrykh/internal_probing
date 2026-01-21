"""
Semantic entropy enricher for uncertainty estimation.

Based on semantic_entropy_probes.ipynb implementation.
Uses SemanticEntropyScorer for computation.
"""

import numpy as np
from typing import Optional

from src.dataset.enrichers.base import BaseEnricher
from src.scoring.sampling.semantic_entropy import SemanticEntropyScorer
from src.scoring.sampling.inputs import SamplerInput


class SemanticEntropyEnricher(BaseEnricher):
    """
    Enricher that computes semantic entropy through sampling and clustering.

    Adds fields:
    - sampling_answers: list[str] - Sampled answers
    - se_raw: float - Raw semantic entropy value
    - semantic_ids: list[int] - Cluster IDs for each sample
    - se_binary: int - Binarized SE label (0=low, 1=high) [if binarize=True]
    - se_gamma: float - Threshold used for binarization [if binarize=True]
    - se_weight: float - Sample weight based on distance from threshold [if add_weights=True]

    Based on enrich_dataset_with_sampling_and_se() from semantic_entropy_probes.ipynb (lines 1028-1208).
    Uses SemanticEntropyScorer for SE computation.
    """

    def __init__(
        self,
        scorer: SemanticEntropyScorer,
        model_wrapper=None,
        binarize: bool = True,
        gamma: Optional[float] = None,
        fit_gamma: bool = True,
        add_weights: bool = False,
        inplace: bool = False,
        verbose: bool = True
    ):
        """
        Initialize semantic entropy enricher.

        Args:
            scorer: SemanticEntropyScorer instance for SE computation
            model_wrapper: ModelWrapper for generation (can be set later)
            binarize: Whether to binarize SE into high/low
            gamma: Threshold for binarization (None = auto-compute)
            fit_gamma: Whether to fit gamma from data
            add_weights: Whether to compute sample weights
            inplace: Whether to modify samples in-place
            verbose: Whether to print progress
        """
        super().__init__(inplace=inplace, verbose=verbose)
        self.scorer = scorer
        self.model_wrapper = model_wrapper
        self.binarize = binarize
        self.gamma = gamma
        self.fit_gamma = fit_gamma
        self.add_weights = add_weights

        # Store SE values for gamma fitting
        self._se_values_for_gamma = []

    def enrich_sample(self, sample: dict, model_wrapper=None, **kwargs) -> dict:
        """
        Add semantic entropy to sample.

        Args:
            sample: Sample dict with 'prompt' field
            model_wrapper: ModelWrapper for generation (optional if set in __init__)
            **kwargs: Override parameters

        Returns:
            Sample with SE fields added
        """
        # Use provided model_wrapper or fall back to stored one
        wrapper = model_wrapper or self.model_wrapper
        if wrapper is None:
            raise ValueError(
                "model_wrapper must be provided either in __init__ or enrich_sample"
            )

        prompt = sample['prompt']

        # Create input for scorer
        scorer_input = SamplerInput(
            prompts=[prompt],
            model_wrapper=wrapper
        )

        # Compute SE with details
        result = self.scorer.estimate(scorer_input, return_details=True)

        # Add to sample
        sample['sampling_answers'] = result['sampling_answers']
        sample['se_raw'] = result['entropy']
        sample['semantic_ids'] = result['semantic_ids']

        # Store for gamma fitting if needed
        if self.binarize and self.gamma is None and self.fit_gamma:
            self._se_values_for_gamma.append(sample['se_raw'])

        return sample

    def enrich_dataset(
        self,
        dataset,
        model_wrapper=None,
        verbose_every: Optional[int] = 100,
        **kwargs
    ):
        """
        Enrich entire dataset with SE, including binarization if requested.

        Based on enrich_dataset_with_sampling_and_se() from notebook (lines 1028-1208).

        Args:
            dataset: Dataset to enrich
            model_wrapper: ModelWrapper for generation (optional if set in __init__)
            verbose_every: Print progress every N samples
            **kwargs: Additional parameters

        Returns:
            Dataset of same type as input
        """
        # Use provided model_wrapper or fall back to stored one
        wrapper = model_wrapper or self.model_wrapper
        if wrapper is None:
            raise ValueError(
                "model_wrapper must be provided either in __init__ or enrich_dataset"
            )

        # Reset SE values for gamma fitting
        self._se_values_for_gamma = []

        # First pass: compute SE for all samples
        # Pass model_wrapper to enrich_sample via kwargs
        output = super().enrich_dataset(
            dataset,
            verbose_every=verbose_every,
            model_wrapper=wrapper,
            **kwargs
        )

        # Extract samples from output (could be dataset or list)
        if hasattr(output, '_data'):
            samples = output._data
            dataset_class = type(output)
        else:
            samples = output
            dataset_class = None

        # Second pass: binarization if requested
        if self.binarize:
            se_arr = np.array([s['se_raw'] for s in samples], dtype=float)

            # Determine gamma
            if self.gamma is None:
                if not self.fit_gamma:
                    raise ValueError(
                        "gamma is None but fit_gamma=False. "
                        "Pass gamma explicitly or set fit_gamma=True."
                    )
                gamma = self._find_optimal_threshold(se_arr)
            else:
                gamma = self.gamma

            # Binarize
            se_binary = (se_arr > gamma).astype(int)

            # Compute weights if requested
            if self.add_weights:
                se_weight = self._compute_se_weights(se_arr, se_binary, gamma)
            else:
                se_weight = np.ones_like(se_arr, dtype=float)

            # Add to samples
            for sample, y, w in zip(samples, se_binary.tolist(), se_weight.tolist()):
                sample['se_gamma'] = float(gamma)
                sample['se_binary'] = int(y)
                sample['se_weight'] = float(w)

        # Return same type as output from parent
        if dataset_class is not None:
            return dataset_class(samples)
        else:
            return samples

    @staticmethod
    def _find_optimal_threshold(se_values: np.ndarray) -> float:
        """
        Find optimal threshold using variance minimization.

        Based on find_optimal_threshold_gamma() from notebook (lines 885-951).
        """
        x = np.asarray(se_values, dtype=float)
        x = x[~np.isnan(x)]

        if len(x) < 2:
            return float(x[0]) if len(x) == 1 else 0.0

        xs = np.sort(x)
        candidates = (xs[:-1] + xs[1:]) / 2.0

        best_gamma = candidates[0]
        best_loss = float("inf")

        for gamma in candidates:
            low = x[x < gamma]
            high = x[x >= gamma]

            if len(low) == 0 or len(high) == 0:
                continue

            low_mean = low.mean()
            high_mean = high.mean()
            loss = ((low - low_mean) ** 2).sum() + ((high - high_mean) ** 2).sum()

            if loss < best_loss:
                best_loss = loss
                best_gamma = gamma

        return float(best_gamma)

    @staticmethod
    def _compute_se_weights(
        se_values: np.ndarray,
        se_binary: np.ndarray,
        gamma: float
    ) -> np.ndarray:
        """
        Compute sample weights based on distance from threshold.

        Based on compute_se_weights() from notebook (lines 952-1016).
        """
        x = np.asarray(se_values, dtype=float)
        y = np.asarray(se_binary, dtype=int)

        w = np.ones_like(x, dtype=float)

        low_vals = x[y == 0]
        high_vals = x[y == 1]

        if len(low_vals) == 0 or len(high_vals) == 0:
            return np.ones_like(x, dtype=float)

        low_extreme = float(low_vals.min())
        high_extreme = float(high_vals.max())

        low_denom = max(gamma - low_extreme, 1e-8)
        high_denom = max(high_extreme - gamma, 1e-8)

        for i in range(len(x)):
            if y[i] == 0:
                t = (gamma - x[i]) / low_denom
            else:
                t = (x[i] - gamma) / high_denom
            t = float(np.clip(t, 0.0, 1.0))
            w[i] = 2.0 * t

        # Normalize to mean=1.0
        mean_w = float(w.mean()) if len(w) > 0 else 1.0
        if mean_w > 0:
            w = w / mean_w

        return w
