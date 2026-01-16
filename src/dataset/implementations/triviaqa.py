"""
TriviaQA dataset implementation.

Based on semantic_entropy_probes.ipynb implementation.
"""

from typing import Optional, Union
from pathlib import Path
from itertools import islice

import pandas as pd
from datasets import load_dataset

from src.dataset.base import BaseDataset
from src.dataset.evaluators import SubstringMatchEvaluator


class TriviaQADataset(BaseDataset):
    """
    TriviaQA dataset for question answering.

    Implements correctness evaluation using substring matching,
    as done in semantic_entropy_probes.ipynb.

    Example:
        >>> # Load from HuggingFace
        >>> dataset = TriviaQADataset.from_huggingface(
        ...     split="validation",
        ...     n_samples=1000,
        ...     seed=42
        ... )
        >>>
        >>> # Access samples
        >>> sample = dataset[0]
        >>> print(sample['prompt'], sample['gt_answers'])
    """

    def __init__(
        self,
        data: Union[list[dict], pd.DataFrame, Path, str],
        name: Optional[str] = None,
        evaluator: Optional[SubstringMatchEvaluator] = None
    ):
        """
        Initialize TriviaQA dataset.

        Args:
            data: Dataset source
            name: Optional dataset name
            evaluator: Correctness evaluator (default: SubstringMatchEvaluator)
        """
        super().__init__(data, name or "TriviaQA")

        # Use substring match evaluator by default (as in notebook)
        self.evaluator = evaluator or SubstringMatchEvaluator(case_sensitive=False)

    def evaluate_correctness(
        self,
        prompt: str,
        generated_answer: str,
        gt_answers: list[str]
    ) -> float:
        """
        Evaluate correctness using substring matching.

        Based on is_correct_triviaqa() from notebook (lines 463-471).

        Args:
            prompt: Original prompt (not used for TriviaQA)
            generated_answer: Model's generated answer
            gt_answers: List of ground truth answers

        Returns:
            1.0 if any GT answer found in generated answer, 0.0 otherwise
        """
        return self.evaluator.evaluate(prompt, generated_answer, gt_answers)

    @classmethod
    def from_huggingface(
        cls,
        split: str = "validation",
        n_samples: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 42,
        shuffle_buffer: int = 10000,
        streaming: bool = True,
        **kwargs
    ) -> "TriviaQADataset":
        """
        Load TriviaQA from HuggingFace datasets.

        Based on loading logic from notebook (lines 251-312).

        Args:
            split: Dataset split ("train", "validation", "test")
            n_samples: Number of samples to load (None = all)
            shuffle: Whether to shuffle the dataset
            seed: Random seed for shuffling
            shuffle_buffer: Buffer size for streaming shuffle
            **kwargs: Additional arguments for dataset constructor

        Returns:
            TriviaQADataset instance
        """
        # Load dataset in streaming mode
        ds_stream = load_dataset(
            "mandarjoshi/trivia_qa",
            "rc.nocontext",
            split=split,
            streaming=streaming,
        )

        # Shuffle if requested
        if shuffle:
            ds_stream = ds_stream.shuffle(seed=seed, buffer_size=shuffle_buffer)

        # Take n_samples if specified
        if n_samples is not None:
            samples = list(islice(ds_stream, n_samples))
        else:
            samples = list(ds_stream)

        # Simplify format (based on simplify_triviaqa_example from notebook)
        simplified_samples = []
        for ex in samples:
            q = ex["question"]

            # Extract GT answers
            ans = ex.get("answer", {})
            gt = ans.get("normalized_aliases") or []
            if not gt:
                nv = ans.get("normalized_value")
                if nv:
                    gt = [nv]

            simplified_samples.append({
                "prompt": q,
                "gt_answers": gt,
            })

        return cls(data=simplified_samples, **kwargs)

    @classmethod
    def from_saved(
        cls,
        path: Union[Path, str],
        **kwargs
    ) -> "TriviaQADataset":
        """
        Load TriviaQA from saved file.

        Args:
            path: Path to saved dataset
            **kwargs: Additional arguments for dataset constructor

        Returns:
            TriviaQADataset instance
        """
        return cls(data=path, **kwargs)
