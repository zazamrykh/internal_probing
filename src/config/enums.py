"""
Enums for configuration system.

Defines valid values for various configuration options.
"""

from enum import Enum


class ProbeType(str, Enum):
    """Types of probing methods."""
    LINEAR = "linear"
    PROMPT_EMBEDDING = "pep"


class TargetType(str, Enum):
    """Target fields for prediction."""
    ACCURACY = "is_correct"
    SEMANTIC_ENTROPY = "se_binary"
    SE_RAW = "se_raw"


class DatasetType(str, Enum):
    """Supported datasets."""
    TRIVIAQA = "triviaqa"


class ModelType(str, Enum):
    """Supported LLM models."""
    MISTRAL_7B = "mistral-7b-instruct"
    GPT2 = "gpt2"



class EnricherType(str, Enum):
    """Types of dataset enrichers."""
    GREEDY_GENERATION = "greedy_generation"
    ACTIVATION = "activation"
    SEMANTIC_ENTROPY = "semantic_entropy"


class EvaluatorType(str, Enum):
    """Types of correctness evaluators."""
    SUBSTRING_MATCH = "substring_match"
