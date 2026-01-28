"""
Probe-based scorers for uncertainty/quality estimation.

These scorers predict scores from model activations without requiring generation,
making them very fast for inference.

SCORING CONVENTION:
All scorers follow the convention: **higher score = higher hallucination risk**
This means:
- High score = high uncertainty/error probability (bad)
- Low score = low uncertainty/high confidence (good)

Current implementations:
- LinearProbeScorer: Uses trained sklearn models (LogisticRegression, etc.)

"""

import numpy as np
import torch

from src.scoring.base import ScorerInterface
from src.scoring.inputs import ProbeInput


class LinearProbeScorer(ScorerInterface):
    """
    Linear probe scorer using trained sklearn-like models.

    Uses a pre-trained model (e.g., LogisticRegression) to predict scores
    from cached activations. The probe is trained externally and passed
    to the scorer for inference.

    SCORING CONVENTION: Returns uncertainty/hallucination probability
    - Higher score = higher risk of hallucination/error
    - Lower score = higher confidence/correctness

    Supported probe types:
    1. SEP (Semantic Entropy Probes): trained to predict P(high_SE)
       - Returns P(high_SE) directly (high = uncertain = bad)

    2. Accuracy Probes: trained to predict P(correct)
       - Returns P(incorrect) = 1 - P(correct) (high = incorrect = bad)

    Example (SEP probe):
        >>> from sklearn.linear_model import LogisticRegression
        >>> from src.scoring.inputs import ProbeInput
        >>>
        >>> # Train SEP probe to predict high semantic entropy
        >>> sep_probe = LogisticRegression(C=1.0, max_iter=500)
        >>> sep_probe.fit(X_train, y_train_se_binary)  # y=1 means high SE
        >>>
        >>> # Create scorer (no inversion needed)
        >>> scorer = LinearProbeScorer(probe_model=sep_probe, probe_type="sep")
        >>>
        >>> # Returns P(high_SE) - higher = more uncertain
        >>> input_data = ProbeInput(activations=X_test)
        >>> uncertainty_scores = scorer.estimate(input_data)

    Example (Accuracy probe):
        >>> # Train accuracy probe to predict correctness
        >>> acc_probe = LogisticRegression(C=1.0, max_iter=500)
        >>> acc_probe.fit(X_train, y_train_is_correct)  # y=1 means correct
        >>>
        >>> # Create scorer (will invert to return P(incorrect))
        >>> scorer = LinearProbeScorer(probe_model=acc_probe, probe_type="accuracy")
        >>>
        >>> # Returns P(incorrect) = 1 - P(correct) - higher = more errors
        >>> input_data = ProbeInput(activations=X_test)
        >>> error_scores = scorer.estimate(input_data)
    """

    def __init__(self, probe_model, probe_type: str = "sep", layer=None, pos=None):
        """
        Initialize linear probe scorer.

        Args:
            probe_model: Trained sklearn-like model with predict_proba() method
                        (e.g., LogisticRegression, MLPClassifier, etc.)
            probe_type: Type of probe, determines output semantics:
                       - "sep": Probe predicts P(high_SE), returns as-is
                       - "accuracy": Probe predicts P(correct), inverts to P(incorrect)
                       Default: "sep"

        Raises:
            ValueError: If probe_model lacks predict_proba() or probe_type is invalid
        """
        self.probe_model = probe_model
        self.layer = layer
        self.pos = pos
        # Validate probe_type
        valid_types = {"sep", "accuracy"}
        if probe_type not in valid_types:
            raise ValueError(
                f"probe_type must be one of {valid_types}, got '{probe_type}'"
            )
        self.probe_type = probe_type

        # Verify model has required methods
        if not hasattr(probe_model, 'predict_proba'):
            raise ValueError(
                f"probe_model must have predict_proba() method, "
                f"got {type(probe_model).__name__}"
            )

    def estimate(self, input: ProbeInput) -> np.ndarray:
        """
        Estimate uncertainty/hallucination scores from activations.

        Args:
            input: ProbeInput containing activations tensor

        Returns:
            Array of uncertainty scores, shape (batch_size,)
            Higher values indicate higher hallucination risk
        """
        if not isinstance(input, ProbeInput):
            raise TypeError(
                f"{self.__class__.__name__} requires ProbeInput, "
                f"got {type(input).__name__}"
            )

        return self.predict(input.activations)

    def predict(self, activations: torch.Tensor) -> np.ndarray:
        """
        Predict uncertainty/hallucination scores from activations.

        Args:
            activations: Tensor of shape (batch_size, hidden_dim)

        Returns:
            Array of uncertainty scores, shape (batch_size,)
            - For SEP probes: P(high_SE) - higher = more uncertain
            - For accuracy probes: P(incorrect) = 1 - P(correct) - higher = more errors
        """
        # Convert to numpy if needed
        if isinstance(activations, torch.Tensor):
            X = activations.cpu().numpy()
        else:
            X = np.asarray(activations)

        # Ensure 2D shape
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Predict probabilities for positive class (class=1)
        probs = self.probe_model.predict_proba(X)[:, 1]

        # Apply semantics based on probe type
        if self.probe_type == "sep":
            # SEP probe: class=1 means high SE (uncertain)
            # Return as-is: higher = more uncertain
            return probs
        else:  # probe_type == "accuracy"
            # Accuracy probe: class=1 means correct
            # Invert to get P(incorrect): higher = more errors
            return 1.0 - probs


    def get_input(self, sample):
        layer = getattr(self, 'layer', None)
        position = getattr(self, 'position', None)

        if layer is None or position is None:
            raise ValueError(
                "LinearProbeScorer must have 'layer' and 'position' attributes set"
            )

        # Extract activations from sample
        if 'activations' not in sample:
            raise KeyError(
                f"Sample does not contain 'activations'. "
                f"Make sure ActivationEnricher was applied."
            )

        acts = sample['activations']['acts']
        if position not in acts:
            raise KeyError(
                f"Position {position} not found in activations. "
                f"Available positions: {list(acts.keys())}"
            )
        if layer not in acts[position]:
            raise KeyError(
                f"Layer {layer} not found in activations for position {position}. "
                f"Available layers: {list(acts[position].keys())}"
            )

        activation = acts[position][layer]

        # Convert to tensor if needed
        if isinstance(activation, np.ndarray):
            activation = torch.from_numpy(activation)
        elif not isinstance(activation, torch.Tensor):
            activation = torch.tensor(activation)

        return ProbeInput(activations=activation)
