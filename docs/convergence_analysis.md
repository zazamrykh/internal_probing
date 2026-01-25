# Convergence Analysis for PEP Models

## Overview

Convergence analysis helps determine optimal training duration for PEP models by tracking validation metrics during training. This prevents both undertraining and overfitting.

## Quick Start

```bash
# Run convergence analysis with default config
python scripts.analyze_convergence

# Or specify custom config
python -m scripts.analyze_convergence --config configs/tuning/convergence_analysis_triviaqa_mistral.toml
```

## Configuration

Key parameters in `configs/tuning/convergence_analysis.toml`:

```toml
[dataset]
enriched_path = "exp_results/pep_example/enriched_datasets"  # Use pre-enriched data

[probe.pep_params]
# Enable history tracking (REQUIRED)
return_history = true
val_check_interval = 50  # Validate every N iterations

# Training limits
n_samples_for_train = 10000  # Total samples to train on
# max_time = 600  # Optional time limit in seconds
```

## How It Works

1. **Training**: Runs PEP training with validation checks at regular intervals
2. **History Tracking**: Records validation AUC and loss at each checkpoint
3. **Plateau Detection**: Identifies when improvement stops (threshold: 0.01 over 5 checkpoints)
4. **Recommendations**: Suggests optimal `n_samples_for_train` with 30% safety margin

## Output

The script generates:

- `convergence_analysis.png` - Two plots showing AUC vs samples and AUC vs time
- `convergence_analysis.json` - Detailed analysis with recommendations
- `metrics.json` - Full training history

## Example Output

```
Best val AUC: 0.8542 at 6000 samples
Convergence detected: True
Convergence at: 5500 samples (245.3s)
Recommended: 7150 samples (319.0s)
```

## Interpreting Results

- **Best val AUC**: Highest validation performance achieved
- **Convergence point**: Where improvement plateaus
- **Recommended samples**: Safe training duration (convergence + 30% margin)

Use recommended values in your production configs to ensure consistent convergence without wasting compute.

## Advanced Usage

Adjust plateau detection sensitivity:

```python
# In analyze_convergence.py
analysis = analyze_convergence(
    history,
    plateau_threshold=0.005,  # Stricter threshold
    plateau_window=10         # Longer window
)
