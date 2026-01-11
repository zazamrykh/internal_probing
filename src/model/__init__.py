"""
Model wrappers for unified interface across different LLM architectures.
"""

from src.model.base import ModelWrapper
from src.model.gpt2 import GPT2Model
from src.model.mistral import MistralModel

__all__ = ["ModelWrapper", "GPT2Model", "MistralModel"]
