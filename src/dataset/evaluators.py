"""
Correctness evaluators for different datasets.

Provides various methods to evaluate whether a generated answer
is correct given the ground truth answers.
"""

from abc import ABC, abstractmethod
from typing import Optional


class CorrectnessEvaluator(ABC):
    """
    Base class for correctness evaluation strategies.

    Different datasets may require different evaluation methods
    (substring match, exact match, NLI-based, etc.)
    """

    @abstractmethod
    def evaluate(
        self,
        prompt: str,
        generated_answer: str,
        gt_answers: list[str]
    ) -> float:
        """
        Evaluate correctness of generated answer.

        Args:
            prompt: Original prompt/question
            generated_answer: Model's generated answer
            gt_answers: List of ground truth answers

        Returns:
            Correctness score in [0, 1]:
            - 1.0 = definitely correct
            - 0.0 = definitely incorrect
            - Values in between indicate uncertainty
        """
        pass

    @classmethod
    def create(cls, evaluator_type: str = "substring_match", **kwargs) -> 'CorrectnessEvaluator':
        """
        Factory method to create evaluator by type.

        Args:
            evaluator_type: Type of evaluator ("substring_match", "exact_match", "custom")
            **kwargs: Additional arguments for evaluator constructor

        Returns:
            CorrectnessEvaluator instance

        Example:
            >>> evaluator = CorrectnessEvaluator.create("substring_match")
            >>> evaluator = CorrectnessEvaluator.create("exact_match", case_sensitive=True)
        """
        if evaluator_type == "substring_match":
            return SubstringMatchEvaluator(
                case_sensitive=kwargs.get('case_sensitive', False)
            )
        elif evaluator_type == "exact_match":
            return ExactMatchEvaluator(
                case_sensitive=kwargs.get('case_sensitive', False),
                normalize=kwargs.get('normalize', True)
            )
        elif evaluator_type == "custom":
            if 'eval_fn' not in kwargs:
                raise ValueError("custom evaluator requires 'eval_fn' argument")
            return CustomEvaluator(kwargs['eval_fn'])
        else:
            raise ValueError(
                f"Unknown evaluator type: '{evaluator_type}'. "
                f"Supported: substring_match, exact_match, custom"
            )


class SubstringMatchEvaluator(CorrectnessEvaluator):
    """
    Evaluator using case-insensitive substring matching.

    Used in TriviaQA and similar QA datasets where the answer
    might appear as part of a longer generated text.

    Example:
        >>> evaluator = SubstringMatchEvaluator()
        >>> evaluator.evaluate(
        ...     prompt="What is the capital of France?",
        ...     generated_answer="The capital of France is Paris.",
        ...     gt_answers=["Paris", "paris"]
        ... )
        1.0
    """

    def __init__(self, case_sensitive: bool = False):
        """
        Initialize substring match evaluator.

        Args:
            case_sensitive: Whether to perform case-sensitive matching
        """
        self.case_sensitive = case_sensitive

    def evaluate(
        self,
        prompt: str,
        generated_answer: str,
        gt_answers: list[str]
    ) -> float:
        """
        Check if any ground truth answer appears in generated answer.

        Args:
            prompt: Original prompt (not used in this evaluator)
            generated_answer: Model's generated answer
            gt_answers: List of ground truth answers

        Returns:
            1.0 if any GT answer found in generated answer, 0.0 otherwise
        """
        if not gt_answers:
            return 0.0

        if self.case_sensitive:
            pred = generated_answer
            aliases = gt_answers
        else:
            pred = generated_answer.casefold()
            aliases = [a.casefold() for a in gt_answers]

        # Check if any alias appears as substring
        return 1.0 if any(a in pred for a in aliases if a) else 0.0


class ExactMatchEvaluator(CorrectnessEvaluator):
    """
    Evaluator using exact string matching.

    Requires the generated answer to exactly match one of the
    ground truth answers (after optional normalization).

    Example:
        >>> evaluator = ExactMatchEvaluator(normalize=True)
        >>> evaluator.evaluate(
        ...     prompt="What is 2+2?",
        ...     generated_answer="  4  ",
        ...     gt_answers=["4", "four"]
        ... )
        1.0
    """

    def __init__(
        self,
        case_sensitive: bool = False,
        normalize: bool = True
    ):
        """
        Initialize exact match evaluator.

        Args:
            case_sensitive: Whether to perform case-sensitive matching
            normalize: Whether to strip whitespace and normalize
        """
        self.case_sensitive = case_sensitive
        self.normalize = normalize

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        if self.normalize:
            text = text.strip()
        if not self.case_sensitive:
            text = text.casefold()
        return text

    def evaluate(
        self,
        prompt: str,
        generated_answer: str,
        gt_answers: list[str]
    ) -> float:
        """
        Check if generated answer exactly matches any GT answer.

        Args:
            prompt: Original prompt (not used in this evaluator)
            generated_answer: Model's generated answer
            gt_answers: List of ground truth answers

        Returns:
            1.0 if exact match found, 0.0 otherwise
        """
        if not gt_answers:
            return 0.0

        pred = self._normalize_text(generated_answer)
        aliases = [self._normalize_text(a) for a in gt_answers]

        return 1.0 if pred in aliases else 0.0


class CustomEvaluator(CorrectnessEvaluator):
    """
    Evaluator using custom user-provided function.

    Allows for arbitrary evaluation logic specific to a dataset.

    Example:
        >>> def my_eval(prompt, answer, gt_answers):
        ...     # Custom logic
        ...     return 1.0 if answer.startswith("The answer is") else 0.0
        >>>
        >>> evaluator = CustomEvaluator(my_eval)
    """

    def __init__(self, eval_fn):
        """
        Initialize custom evaluator.

        Args:
            eval_fn: Function with signature:
                    (prompt: str, generated_answer: str, gt_answers: list[str]) -> float
        """
        self.eval_fn = eval_fn

    def evaluate(
        self,
        prompt: str,
        generated_answer: str,
        gt_answers: list[str]
    ) -> float:
        """
        Evaluate using custom function.

        Args:
            prompt: Original prompt
            generated_answer: Model's generated answer
            gt_answers: List of ground truth answers

        Returns:
            Correctness score from custom function
        """
        return self.eval_fn(prompt, generated_answer, gt_answers)


# Backward compatibility: keep function wrapper
def create_evaluator(evaluator_type: str = "substring_match", **kwargs) -> CorrectnessEvaluator:
    """
    Factory function to create evaluator by type (backward compatibility wrapper).

    Prefer using CorrectnessEvaluator.create() instead.
    """
    return CorrectnessEvaluator.create(evaluator_type, **kwargs)
