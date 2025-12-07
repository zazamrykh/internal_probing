from abc import ABC, abstractmethod
from typing import Optional

import torch

from uncertainity_estimation.base import ScorerInterface

class ProbeInterface(ScorerInterface):
    def __init__(self, linear_probe : torch.Module):
        self.linear_probe = linear_probe

    def estimate(self, model, prompts=None, answers=None, activation=None, **kwargs):
        if answers is not None:
            raise ValueError(
                f"{self.__class__.__name__} doesn't use provided answer. "
                "It predict score using internal activation."
            )
        if prompts is not None:
            raise ValueError(
                f"{self.__class__.__name__} doesn't use provided prompt. "
                "It predict score using internal activation."
            )
        if activation is None:
            raise ValueError(
                f"{self.__class__.__name__} must got activation but it is None"
            )

        score = self.predict(activation)
        return score


    @abstractmethod
    def predict(self, activation):
        pass
