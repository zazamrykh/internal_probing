"""
Tests for ExperimentRunner.

Tests configuration validation, enricher factory, and basic pipeline.
"""

import pytest
import tempfile
from pathlib import Path

from src.config import ExperimentConfig, EnricherType
from src.experiment_runner import ExperimentRunner
from src.dataset.enrichers import EnricherFactory
from src.model import GPT2Model
from src.dataset import SubstringMatchEvaluator
from transformers import AutoModelForCausalLM, AutoTokenizer


def test_config_validation_missing_experiment_name():
    """Test that missing experiment name raises error."""
    config_dict = {
        'experiment': {},  # Missing 'name'
        'dataset': {'dataset_type': 'triviaqa'},
        'model': {'model_type': 'gpt2', 'model_name_or_path': 'gpt2'},
        'probe': {'probe_type': 'linear'},
        'training': {'positions': [0], 'layers': [0], 'targets': ['is_correct']}
    }

    # Validation happens in ExperimentConfig.__init__
    with pytest.raises(ValueError, match="experiment.name is required"):
        config = ExperimentConfig(config_dict)


def test_config_validation_missing_dataset_type():
    """Test that missing dataset type raises error."""
    config_dict = {
        'experiment': {'name': 'test'},
        'dataset': {},  # Missing 'dataset_type'
        'model': {'model_type': 'gpt2', 'model_name_or_path': 'gpt2'},
        'probe': {'probe_type': 'linear'},
        'training': {'positions': [0], 'layers': [0], 'targets': ['is_correct']}
    }

    config = ExperimentConfig(config_dict)

    with pytest.raises(ValueError, match="dataset.dataset_type is required"):
        runner = ExperimentRunner(config)


def test_config_validation_missing_model_params():
    """Test that missing model parameters raise error."""
    config_dict = {
        'experiment': {'name': 'test'},
        'dataset': {'dataset_type': 'triviaqa'},
        'model': {},  # Missing model_type and model_name_or_path
        'probe': {'probe_type': 'linear'},
        'training': {'positions': [0], 'layers': [0], 'targets': ['is_correct']}
    }

    config = ExperimentConfig(config_dict)

    with pytest.raises(ValueError, match="model.model_type is required"):
        runner = ExperimentRunner(config)


def test_config_validation_missing_training_params():
    """Test that missing training parameters raise error."""
    config_dict = {
        'experiment': {'name': 'test'},
        'dataset': {'dataset_type': 'triviaqa'},
        'model': {'model_type': 'gpt2', 'model_name_or_path': 'gpt2'},
        'probe': {'probe_type': 'linear'},
        'training': {}  # Missing positions, layers, targets
    }

    config = ExperimentConfig(config_dict)

    with pytest.raises(ValueError, match="training.positions is required"):
        runner = ExperimentRunner(config)


def test_config_validation_with_enriched_path():
    """Test that enriched_path bypasses dataset_type requirement."""
    config_dict = {
        'experiment': {'name': 'test'},
        'dataset': {'enriched_path': '/path/to/data'},  # No dataset_type needed
        'model': {'model_type': 'gpt2', 'model_name_or_path': 'gpt2'},
        'probe': {'probe_type': 'linear'},
        'training': {'positions': [0], 'layers': [0], 'targets': ['is_correct']}
    }

    config = ExperimentConfig(config_dict)

    # Should not raise error
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dict['experiment']['output_dir'] = tmpdir
        config = ExperimentConfig(config_dict)
        runner = ExperimentRunner(config)
        assert runner.config.experiment['name'] == 'test'


def test_config_validation_success():
    """Test that valid config passes validation."""
    config_dict = {
        'experiment': {'name': 'test_experiment'},
        'dataset': {'dataset_type': 'triviaqa'},
        'model': {'model_type': 'gpt2', 'model_name_or_path': 'gpt2'},
        'probe': {'probe_type': 'linear'},
        'training': {
            'positions': [0, -2],
            'layers': [0, 6, 11],
            'targets': ['is_correct']
        }
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        config_dict['experiment']['output_dir'] = tmpdir
        config = ExperimentConfig(config_dict)
        runner = ExperimentRunner(config)

        assert runner.config.experiment['name'] == 'test_experiment'
        assert runner.output_dir == Path(tmpdir)


@pytest.fixture(scope="module")
def gpt2_wrapper():
    """Create GPT-2 wrapper for testing."""
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    return GPT2Model(model=model, tokenizer=tokenizer, system_prompt="")


def test_enricher_factory_greedy_generation(gpt2_wrapper):
    """Test EnricherFactory creates GreedyGenerationEnricher correctly."""
    config = {
        'type': 'greedy_generation',
        'max_new_tokens': 20,
        'verbose': False
    }

    evaluator = SubstringMatchEvaluator()
    enricher = EnricherFactory.create(config, gpt2_wrapper, evaluator)

    assert enricher is not None
    assert enricher.max_new_tokens == 20
    assert enricher.model_wrapper is gpt2_wrapper
    assert enricher.evaluator is evaluator


def test_enricher_factory_activation(gpt2_wrapper):
    """Test EnricherFactory creates ActivationEnricher correctly."""
    config = {
        'type': 'activation',
        'layers': [0, 6, 11],
        'positions': [0, -2],
        'verbose': False
    }

    enricher = EnricherFactory.create(config, gpt2_wrapper, None)

    assert enricher is not None
    assert enricher.layers == [0, 6, 11]
    assert enricher.positions == [0, -2]
    assert enricher.model_wrapper is gpt2_wrapper


def test_enricher_factory_semantic_entropy(gpt2_wrapper):
    """Test EnricherFactory creates SemanticEntropyEnricher correctly."""
    config = {
        'type': 'semantic_entropy',
        'n_samples': 5,
        'binarize': True,
        'add_weights': False,
        'verbose': False
    }

    enricher = EnricherFactory.create(config, gpt2_wrapper, None)

    assert enricher is not None
    assert enricher.scorer.n_samples == 5
    assert enricher.binarize is True
    assert enricher.add_weights is False
    assert enricher.model_wrapper is gpt2_wrapper


def test_enricher_factory_missing_required_params():
    """Test that missing required parameters raise error."""
    config = {
        'type': 'activation',
        # Missing 'layers' and 'positions'
    }

    with pytest.raises(ValueError, match="'layers' is required"):
        EnricherFactory.create(config, None, None)


def test_enricher_factory_unknown_type():
    """Test that unknown enricher type raises error."""
    config = {
        'type': 'unknown_enricher'
    }

    with pytest.raises(ValueError):
        EnricherFactory.create(config, None, None)


def test_config_warn_on_fallback(caplog):
    """Test that warn_on_fallback logs warnings."""
    import logging
    caplog.set_level(logging.WARNING)

    config_dict = {
        'experiment': {'name': 'test'},
        'dataset': {'dataset_type': 'triviaqa'},
        'model': {'model_type': 'gpt2', 'model_name_or_path': 'gpt2'},
        'probe': {'probe_type': 'linear'},
        'training': {'positions': [0], 'layers': [0], 'targets': ['is_correct']}
    }

    config = ExperimentConfig(config_dict)

    # Get with fallback and warning
    value = config.get('nonexistent.key', 'default_value', warn_on_fallback=True)

    assert value == 'default_value'
    assert "Config key 'nonexistent.key' not found" in caplog.text


if __name__ == '__main__':
    pytest.main([__file__, "-v", "-s"])
