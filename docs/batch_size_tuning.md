# Batch Size Tuning

Automatic batch size selection for PEP model training to maximize GPU utilization and training speed.

## Quick Start

```bash
python scripts/tune_batch_size.py \
    --base-config configs/tuning/batch_size_triviaqa_mistral.toml \
    --batch-sizes 2 4 8 16 32 64 \
    --n-samples 32 \
    --output-dir exp_results/batch_size_tuning/triviaqa_mistral7b
```

## How It Works

1. **Linear Search**: Tests each batch size sequentially
2. **Clean Experiments**: Each test runs in isolated subprocess with fresh GPU memory
3. **OOM Detection**: Stops when CUDA out of memory error occurs
4. **Metrics Collection**: Tracks training time, samples/sec, GPU memory usage

## Configuration

Base config must specify:
- `dataset.enriched_path`: Path to pre-enriched datasets (train.pkl, val.pkl, test.pkl)
- `model`: Model configuration
- `training.pep_params`: PEP training parameters (batch_size will be overridden)

The script automatically:
- Sets `dataset.cycle_train_to_size` to match `--n-samples`
- Disables CV and early stopping for speed
- Runs each experiment with specified batch size

## Parameters

- `--base-config`: Base TOML configuration file
- `--batch-sizes`: List of batch sizes to test (default: 2 4 8 16 32 64)
- `--n-samples`: Training samples per experiment (default: 500)
- `--n-epochs`: Number of epochs (default: 1)
- `--output-dir`: Results directory (default: results/batch_size_tuning)
- `--no-cleanup`: Keep temporary files for debugging

## Output

**results.json**: Detailed metrics for each batch size
```json
{
  "batch_size": 16,
  "success": true,
  "training_time": 45.2,
  "samples_per_second": 22.1,
  "gpu_memory": {"allocated_mb": 2048, "reserved_mb": 2560, ...}
}
```

**batch_size_tuning.png**: Visualization with 4 plots:
- Training time vs batch size
- Samples/sec vs batch size
- GPU memory vs batch size
- Memory efficiency vs batch size

## Recommendation

Script recommends batch size with highest samples/sec throughput.
