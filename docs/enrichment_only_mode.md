# Enrichment-Only Mode

## Overview

The `only_enrichment` flag allows you to run the experiment pipeline in a mode that only performs dataset enrichment without training probes. This is useful when you want to:

1. Pre-process and enrich datasets for later use
2. Cache expensive enrichment operations (e.g., greedy generation, activation extraction)
3. Separate data preparation from model training

## Configuration

Add the following to your TOML config:

```toml
[experiment]
only_enrichment = true  # Enable enrichment-only mode

[enrichment]
save_enriched = true  # Must be true to save enriched datasets
```

## Validation Rules

When `only_enrichment = true`:

- **Required sections**: `experiment`, `dataset`, `model`, `enrichment`
- **Optional sections**: `probe`, `training` (will be ignored)
- **Warning**: If `enrichment.save_enriched = false`, a warning will be logged since enriched datasets won't be saved

When `only_enrichment = false` (default):

- All sections are required as usual

## Example Config

See [`configs/experiments/enrichment_only_example.toml`](../configs/experiments/enrichment_only_example.toml) for a complete example.

## Usage

```bash
# Run enrichment-only pipeline
python -m src.experiment_runner configs/experiments/enrichment_only_example.toml

# Later, use the enriched datasets for training
python -m src.experiment_runner configs/experiments/train_with_enriched.toml
```

## Pipeline Steps

In enrichment-only mode, the pipeline executes:

1. ✅ Load or create datasets
2. ✅ Load model
3. ✅ Apply enrichment pipeline
4. ❌ Train probes (skipped)
5. ❌ Save and visualize results (skipped)

## Output

Enriched datasets will be saved to:
```
{output_dir}/enriched_datasets/
├── train.pkl
├── val.pkl
└── test.pkl
```

These can be loaded in subsequent experiments using:

```toml
[dataset]
enriched_path = "exp_results/enrichment_only_demo/enriched_datasets"
