"""
Test TriviaQA dataset with enrichment pipeline.

Tests the workflow from semantic_entropy_probes.ipynb:
1. Load TriviaQA from HuggingFace (streaming, 3 samples)
2. Enrich with GreedyGenerationEnricher
3. Enrich with ActivationEnricher
4. Enrich with SemanticEntropyEnricher
"""

import logging
import pytest

from src.dataset.implementations.triviaqa import TriviaQADataset
from src.dataset.enrichers import (
    GreedyGenerationEnricher,
    ActivationEnricher,
    SemanticEntropyEnricher
)
from src.model.gpt2 import GPT2Model
from src.scoring.sampling.semantic_entropy import SemanticEntropyScorer
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def triviaqa_dataset():
    """Load small TriviaQA dataset from HuggingFace (streaming mode)."""
    logger.debug("Loading TriviaQA dataset (3 samples)...")

    dataset = TriviaQADataset.from_huggingface(
        split="validation",
        n_samples=3,
        shuffle=True,
        seed=42
    )

    logger.debug(f"Loaded {len(dataset)} samples")
    return dataset


@pytest.fixture(scope="module")
def gpt2_model():
    """Load GPT-2 model for testing."""
    logger.debug("Loading GPT-2 model...")

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    model_wrapper = GPT2Model(
        model=model,
        tokenizer=tokenizer,
        system_prompt=None
    )

    logger.debug("GPT-2 model loaded")
    return model_wrapper


def test_triviaqa_enrichment_pipeline(triviaqa_dataset, gpt2_model):
    """
    Test full enrichment pipeline on real TriviaQA data.

    This test mimics the workflow from semantic_entropy_probes.ipynb:
    - Step 4: Greedy generation + activations
    - Step 5: Semantic entropy calculation
    """
    dataset = triviaqa_dataset

    logger.debug("\n" + "="*60)
    logger.debug("Starting enrichment pipeline test")
    logger.debug("="*60)

    # Show original samples
    logger.debug("\nOriginal samples:")
    for i, sample in enumerate(dataset):
        logger.debug(f"  [{i}] Q: {sample['prompt'][:80]}...")
        logger.debug(f"      GT: {sample['gt_answers'][:3]}")

    # Step 1: Greedy Generation Enrichment
    logger.debug("\n" + "-"*60)
    logger.debug("Step 1: Greedy Generation Enrichment")
    logger.debug("-"*60)

    gen_enricher = GreedyGenerationEnricher(
        model_wrapper=gpt2_model,
        evaluator=dataset.evaluator,
        max_new_tokens=20,
        inplace=True,
        verbose=True
    )

    enriched_data = gen_enricher.enrich_dataset(dataset, verbose_every=1)

    # Verify greedy generation fields
    for i, sample in enumerate(enriched_data):
        assert 'greedy_answer' in sample, f"Sample {i} missing 'greedy_answer'"
        assert 'is_correct' in sample, f"Sample {i} missing 'is_correct'"
        logger.debug(f"  [{i}] Answer: '{sample['greedy_answer']}'")
        logger.debug(f"      Correct: {sample['is_correct']}")

    # Step 2: Activation Enrichment
    logger.debug("\n" + "-"*60)
    logger.debug("Step 2: Activation Enrichment")
    logger.debug("-"*60)

    act_enricher = ActivationEnricher(
        model_wrapper=gpt2_model,
        layers=[0, 6, 11],  # GPT-2 has 12 layers (0-11)
        positions=[0, -2],  # TBG and SLT positions
        inplace=True,
        verbose=True
    )

    enriched_data = act_enricher.enrich_dataset(enriched_data, verbose_every=1)

    # Verify activation fields
    for i, sample in enumerate(enriched_data):
        assert 'activations' in sample, f"Sample {i} missing 'activations'"
        acts = sample['activations']
        assert 'positions' in acts
        assert 'layers' in acts
        assert 'acts' in acts
        assert len(acts['acts']) == 2  # 2 positions
        assert len(acts['acts'][0]) == 3  # 3 layers
        logger.debug(f"  [{i}] Extracted activations: {len(acts['acts'])} positions, {len(acts['acts'][0])} layers")

    # Step 3: Semantic Entropy Enrichment
    logger.debug("\n" + "-"*60)
    logger.debug("Step 3: Semantic Entropy Enrichment")
    logger.debug("-"*60)

    # Create scorer
    se_scorer = SemanticEntropyScorer(
        entailment_model_name="roberta-large-mnli",
        n_samples=5,  # Small number for testing
        sampling_batch_size=1
    )

    se_enricher = SemanticEntropyEnricher(
        scorer=se_scorer,
        binarize=True,
        fit_gamma=True,
        add_weights=False,
        inplace=True,
        verbose=True
    )

    enriched_data = se_enricher.enrich_dataset(
        enriched_data,
        model_wrapper=gpt2_model,
        verbose_every=1
    )

    # Verify SE fields
    for i, sample in enumerate(enriched_data):
        assert 'sampling_answers' in sample, f"Sample {i} missing 'sampling_answers'"
        assert 'se_raw' in sample, f"Sample {i} missing 'se_raw'"
        assert 'semantic_ids' in sample, f"Sample {i} missing 'semantic_ids'"
        assert 'se_binary' in sample, f"Sample {i} missing 'se_binary'"
        assert 'se_gamma' in sample, f"Sample {i} missing 'se_gamma'"

        logger.debug(f"  [{i}] SE: {sample['se_raw']:.3f}, Binary: {sample['se_binary']}, Clusters: {len(set(sample['semantic_ids']))}")
        logger.debug(f"      Samples: {sample['sampling_answers'][:2]}...")

    # Final verification: all expected fields present
    logger.debug("\n" + "="*60)
    logger.debug("Final verification")
    logger.debug("="*60)

    expected_fields = [
        'prompt', 'gt_answers',  # Original
        'greedy_answer', 'is_correct',  # Generation
        'activations',  # Activations
        'sampling_answers', 'se_raw', 'semantic_ids', 'se_binary', 'se_gamma'  # SE
    ]

    for i, sample in enumerate(enriched_data):
        for field in expected_fields:
            assert field in sample, f"Sample {i} missing field '{field}'"

    sample = enriched_data[0]
    logger.debug(f'One sample keys: {sample.keys()}\n')

    logger.debug(f'Prompt: {sample['prompt']}, GT answers (first 3): {sample['gt_answers'][:3]}')
    logger.debug(f'Answer: {sample['greedy_answer']}, Correctness: {sample['is_correct']}\n')

    logger.debug(f'Captured activations keys: {sample['activations'].keys()}')
    logger.debug(f'Activations positions: {sample['activations']['positions']} and layes: {sample['activations']['layers']}')
    logger.debug(f'Activation hidden dim: {len(sample['activations']['acts'][0][0])} and first values: {sample['activations']['acts'][0][0][:5]}\n')

    logger.debug(f'Sampled answers: {sample['sampling_answers']}, Cluster ids: {sample['semantic_ids']}')
    logger.debug(f'Raw semantic entropy: {sample['se_raw']}, binary se labels: {sample['se_binary']}')
    logger.debug(f'Semantic entropy gamma (th): {sample['se_gamma']}, se weigths: {sample['se_weight']}')

    logger.debug(f"All {len(enriched_data)} samples have all {len(expected_fields)} expected fields")
    logger.debug("Enrichment pipeline test PASSED!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--log-cli-level=INFO"])
