"""
Flexible configuration loader for experiments.

Loads TOML configs as dictionaries with minimal validation.
New parameters can be added to TOML files without code changes.
"""

import toml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from src.config.enums import ProbeType, TargetType, DatasetType, ModelType

logger = logging.getLogger(__name__)


class ConfigDict(dict):
    """
    Dictionary wrapper that supports warn_on_fallback parameter in get().

    Allows logging warnings when using default values.
    """

    def get(self, key, default=None, warn_on_fallback=False):
        """
        Get value with optional warning on fallback.

        Args:
            key: Dictionary key
            default: Default value if key not found
            warn_on_fallback: If True, log warning when using default

        Returns:
            Value or default
        """
        value = super().get(key, default)

        if value is None and default is not None and warn_on_fallback:
            logger.warning(
                f"Config key '{key}' not found, using default: {default}"
            )
            return default

        return value if value is not None else default


class ExperimentConfig:
    """
    Flexible experiment configuration loaded from TOML.

    Stores configuration as nested dictionaries, allowing easy addition
    of new parameters without modifying code.

    Example:
        >>> config = ExperimentConfig.from_toml("configs/experiment.toml")
        >>> print(config.model['model_type'])
        >>> print(config.training.get('early_stopping', False))  # New param with default
    """

    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize from configuration dictionary.

        Args:
            config_dict: Nested dictionary from TOML file
        """
        self._config = config_dict

        # Store main sections as ConfigDict for warn_on_fallback support
        self.experiment = ConfigDict(config_dict.get('experiment', {}))
        self.model = ConfigDict(config_dict.get('model', {}))
        self.dataset = ConfigDict(config_dict.get('dataset', {}))
        self.probe = ConfigDict(config_dict.get('probe', {}))
        self.training = ConfigDict(config_dict.get('training', {}))
        self.enrichment = ConfigDict(config_dict.get('enrichment', {}))

        # Validate required fields
        self._validate()

    def _validate(self):
        """Validate that required fields are present and have valid values."""
        # Check experiment name
        if 'name' not in self.experiment:
            raise ValueError("experiment.name is required")

        # Validate enum fields if present
        if 'probe_type' in self.probe:
            try:
                ProbeType(self.probe['probe_type'])
            except ValueError:
                valid = [e.value for e in ProbeType]
                raise ValueError(
                    f"Invalid probe_type: {self.probe['probe_type']}. "
                    f"Must be one of {valid}"
                )

        if 'target' in self.training:
            try:
                TargetType(self.training['target'])
            except ValueError:
                valid = [e.value for e in TargetType]
                raise ValueError(
                    f"Invalid target: {self.training['target']}. "
                    f"Must be one of {valid}"
                )

        if 'dataset_type' in self.dataset:
            try:
                DatasetType(self.dataset['dataset_type'])
            except ValueError:
                valid = [e.value for e in DatasetType]
                raise ValueError(
                    f"Invalid dataset_type: {self.dataset['dataset_type']}. "
                    f"Must be one of {valid}"
                )

        if 'model_type' in self.model:
            try:
                ModelType(self.model['model_type'])
            except ValueError:
                valid = [e.value for e in ModelType]
                raise ValueError(
                    f"Invalid model_type: {self.model['model_type']}. "
                    f"Must be one of {valid}"
                )

    @classmethod
    def from_toml(cls, config_path: str) -> 'ExperimentConfig':
        """
        Load configuration from TOML file.

        Args:
            config_path: Path to TOML configuration file

        Returns:
            ExperimentConfig instance

        Example:
            >>> config = ExperimentConfig.from_toml("configs/my_experiment.toml")
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        logger.info(f"Loading config from {config_path}")
        config_dict = toml.load(path)

        return cls(config_dict)

    def get(self, key: str, default: Any = None, warn_on_fallback: bool = False) -> Any:
        """
        Get configuration value with dot notation.

        Args:
            key: Key in dot notation (e.g., 'training.learning_rate')
            default: Default value if key not found
            warn_on_fallback: If True, log warning when using default value

        Returns:
            Configuration value or default

        Example:
            >>> lr = config.get('training.learning_rate', 0.001)
            >>> new_param = config.get('training.early_stopping', False, warn_on_fallback=True)
        """
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    if warn_on_fallback and default is not None:
                        logger.warning(
                            f"Config key '{key}' not found, using default: {default}"
                        )
                    return default
            else:
                if warn_on_fallback and default is not None:
                    logger.warning(
                        f"Config key '{key}' not found, using default: {default}"
                    )
                return default

        return value

    def to_dict(self) -> Dict[str, Any]:
        """Return full configuration as dictionary."""
        return self._config.copy()

    def save_to_toml(self, path: str):
        """
        Save configuration to TOML file.

        Args:
            path: Path to save TOML file

        Example:
            >>> config.save_to_toml("saved_config.toml")
        """
        import toml
        from pathlib import Path

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'w') as f:
            toml.dump(self._config, f)

        logger.info(f"Saved config to {save_path}")

    def __repr__(self) -> str:
        return f"ExperimentConfig(name='{self.experiment.get('name')}')"
