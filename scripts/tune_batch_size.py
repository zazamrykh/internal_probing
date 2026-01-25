#!/usr/bin/env python3
"""
Batch size tuning script for PEP model training.

Performs linear search over batch sizes to find optimal value that:
1. Fits in GPU memory (no OOM errors)
2. Provides good training speed (samples/second)

Usage:
    python scripts/tune_batch_size.py \
        --base-config configs/experiments/pep_base.toml \
        --batch-sizes 2 4 8 16 32 64 \
        --n-samples 500 \
        --output-dir results/batch_size_tuning
"""

import argparse
import json
import logging
import subprocess
import sys
import time
import tempfile
import shutil
from pathlib import Path
from typing import Optional

import toml
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.gpu_monitor import get_gpu_memory_info, clear_gpu_cache, log_gpu_memory_info

logger = logging.getLogger(__name__)


class BatchSizeTuner:
    """
    Tunes batch size for PEP model training.

    Runs experiments with different batch sizes and tracks:
    - OOM errors
    - Training time
    - GPU memory usage
    - Samples per second
    """

    def __init__(
        self,
        base_config_path: str,
        batch_sizes: list[int],
        n_samples: int,
        n_epochs: int = 1,
        output_dir: str = "results/batch_size_tuning",
        cleanup: bool = True,
    ):
        """
        Initialize batch size tuner.

        Args:
            base_config_path: Path to base TOML config
            batch_sizes: List of batch sizes to test
            n_samples: Number of samples for training (dataset will be cycled if needed)
            n_epochs: Number of epochs to train
            output_dir: Directory to save results
            cleanup: Whether to delete temporary files after experiments
        """
        self.base_config_path = Path(base_config_path)
        self.batch_sizes = sorted(batch_sizes)
        self.n_samples = n_samples
        self.n_epochs = n_epochs
        self.output_dir = Path(output_dir)
        self.cleanup = cleanup

        # Load base config
        if not self.base_config_path.exists():
            raise FileNotFoundError(f"Base config not found: {base_config_path}")

        self.base_config = toml.load(self.base_config_path)

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Results storage
        self.results = []

        logger.info(f"Initialized BatchSizeTuner")
        logger.info(f"Base config: {self.base_config_path}")
        logger.info(f"Batch sizes to test: {self.batch_sizes}")
        logger.info(f"Training samples: {n_samples} (x{n_epochs} epochs)")
        logger.info(f"Output directory: {self.output_dir}")

    def _create_temp_config(self, batch_size: int, temp_dir: Path) -> Path:
        """
        Create temporary config with specified batch size.

        Args:
            batch_size: Batch size to set
            temp_dir: Temporary directory for config

        Returns:
            Path to temporary config file
        """
        config = self.base_config.copy()

        # Set batch size
        if 'training' not in config:
            config['training'] = {}
        if 'pep_params' not in config['training']:
            config['training']['pep_params'] = {}

        config['training']['pep_params']['batch_size'] = batch_size

        # Set max_samples instead of n_epochs for precise control
        # This ensures we train on exactly n_samples regardless of dataset size
        config['training']['pep_params']['max_samples'] = self.n_samples * self.n_epochs

        # Remove n_epochs to avoid confusion
        if 'n_epochs' in config['training']['pep_params']:
            del config['training']['pep_params']['n_epochs']

        # Disable CV and early stopping for speed
        config['training']['use_cv'] = False
        config['training']['pep_params']['early_stopping_patience'] = None
        config['training']['pep_params']['save_best_model'] = False

        # Set output to temp directory
        config['experiment']['output_dir'] = str(temp_dir / 'exp_output')

        # Save temp config
        temp_config_path = temp_dir / f'config_bs{batch_size}.toml'
        with open(temp_config_path, 'w') as f:
            toml.dump(config, f)

        return temp_config_path

    def _run_experiment(self, batch_size: int) -> dict:
        """
        Run single experiment with given batch size.

        Args:
            batch_size: Batch size to test

        Returns:
            Dictionary with experiment results
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"Testing batch_size={batch_size}")
        logger.info(f"{'='*70}")

        # Clear GPU cache before experiment
        clear_gpu_cache()
        log_gpu_memory_info(prefix="Before experiment: ")

        # Create temporary directory
        temp_dir = Path(tempfile.mkdtemp(prefix=f'batch_tune_bs{batch_size}_'))

        try:
            # Create temp config
            temp_config = self._create_temp_config(batch_size, temp_dir)
            logger.info(f"Created temp config: {temp_config}")

            # Run experiment via subprocess
            cmd = [
                sys.executable,
                '-m',
                'src.experiment_runner',
                str(temp_config)
            ]

            logger.info(f"Running command: {' '.join(cmd)}")

            start_time = time.time()

            result = subprocess.run(
                cmd,
                cwd=project_root,
                capture_output=True,
                text=True,
            )

            elapsed_time = time.time() - start_time

            # Check for OOM error
            oom_error = self._check_oom_error(result.stderr)

            if oom_error:
                logger.warning(f"OOM error detected for batch_size={batch_size}")
                return {
                    'batch_size': batch_size,
                    'success': False,
                    'oom_error': True,
                    'training_time': None,
                    'samples_per_second': None,
                    'gpu_memory': None,
                    'error_message': 'CUDA out of memory',
                }

            # Check if experiment succeeded
            if result.returncode != 0:
                logger.error(f"Experiment failed with return code {result.returncode}")
                logger.error(f"stderr: {result.stderr[-500:]}")  # Last 500 chars
                return {
                    'batch_size': batch_size,
                    'success': False,
                    'oom_error': False,
                    'training_time': None,
                    'samples_per_second': None,
                    'gpu_memory': None,
                    'error_message': result.stderr[-200:] if result.stderr else 'Unknown error',
                }

            # Extract training time from metrics
            training_time = self._extract_training_time(temp_dir)

            if training_time is None:
                logger.warning("Could not extract training time, using elapsed time")
                training_time = elapsed_time

            # Calculate samples per second
            total_samples = self.n_samples * self.n_epochs
            samples_per_second = total_samples / training_time if training_time > 0 else 0

            # Get GPU memory info
            gpu_memory = get_gpu_memory_info()

            log_gpu_memory_info(prefix="After experiment: ")

            logger.info(f"✓ Success! Training time: {training_time:.2f}s, "
                       f"Speed: {samples_per_second:.1f} samples/s")

            return {
                'batch_size': batch_size,
                'success': True,
                'oom_error': False,
                'training_time': training_time,
                'samples_per_second': samples_per_second,
                'gpu_memory': gpu_memory,
                'error_message': None,
            }

        finally:
            # Cleanup temp directory if requested
            if self.cleanup and temp_dir.exists():
                shutil.rmtree(temp_dir)
                logger.debug(f"Cleaned up temp directory: {temp_dir}")

    def _check_oom_error(self, stderr: str) -> bool:
        """
        Check if stderr contains OOM error.

        Args:
            stderr: Standard error output

        Returns:
            True if OOM error detected
        """
        oom_patterns = [
            'CUDA out of memory',
            'OutOfMemoryError',
            'out of memory',
        ]

        return any(pattern in stderr for pattern in oom_patterns)

    def _extract_training_time(self, temp_dir: Path) -> Optional[float]:
        """
        Extract training time from experiment output.

        Args:
            temp_dir: Temporary experiment directory

        Returns:
            Training time in seconds, or None if not found
        """
        metrics_path = temp_dir / 'exp_output' / 'metrics.json'

        if not metrics_path.exists():
            return None

        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)

            # Extract training_time from first result
            if metrics and len(metrics) > 0:
                return metrics[0].get('training_time')
        except Exception as e:
            logger.warning(f"Failed to extract training time: {e}")

        return None

    def run(self):
        """
        Run batch size tuning experiments.

        Tests all batch sizes and saves results.
        """
        logger.info(f"\n{'='*70}")
        logger.info("Starting batch size tuning")
        logger.info(f"{'='*70}\n")

        for batch_size in self.batch_sizes:
            result = self._run_experiment(batch_size)
            self.results.append(result)

            # Save intermediate results
            self._save_results()

            # If OOM, stop testing larger batch sizes
            if result['oom_error']:
                logger.info(f"\nStopping: OOM error at batch_size={batch_size}")
                logger.info(f"Maximum working batch size: {self._get_max_working_batch_size()}")
                break

        # Create visualizations
        self._create_visualizations()

        # Print summary
        self._print_summary()

        logger.info(f"\n{'='*70}")
        logger.info("Batch size tuning completed!")
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info(f"{'='*70}\n")

    def _save_results(self):
        """Save results to JSON file."""
        results_path = self.output_dir / 'results.json'

        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        logger.debug(f"Saved results to {results_path}")

    def _get_max_working_batch_size(self) -> Optional[int]:
        """Get maximum batch size that worked without OOM."""
        working = [r for r in self.results if r['success']]
        return max([r['batch_size'] for r in working]) if working else None

    def _create_visualizations(self):
        """Create visualization plots."""
        # Filter successful results
        successful = [r for r in self.results if r['success']]

        if not successful:
            logger.warning("No successful experiments, skipping visualization")
            return

        batch_sizes = [r['batch_size'] for r in successful]
        training_times = [r['training_time'] for r in successful]
        samples_per_sec = [r['samples_per_second'] for r in successful]
        memory_used = [r['gpu_memory']['reserved_mb'] for r in successful]

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Batch Size Tuning Results', fontsize=16)

        # Plot 1: Training time vs batch size
        axes[0, 0].plot(batch_sizes, training_times, 'o-', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('Batch Size')
        axes[0, 0].set_ylabel('Training Time (seconds)')
        axes[0, 0].set_title('Training Time vs Batch Size')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xscale('log', base=2)

        # Plot 2: Samples per second vs batch size
        axes[0, 1].plot(batch_sizes, samples_per_sec, 'o-', linewidth=2, markersize=8, color='green')
        axes[0, 1].set_xlabel('Batch Size')
        axes[0, 1].set_ylabel('Samples per Second')
        axes[0, 1].set_title('Training Speed vs Batch Size')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_xscale('log', base=2)

        # Plot 3: GPU memory vs batch size
        axes[1, 0].plot(batch_sizes, memory_used, 'o-', linewidth=2, markersize=8, color='red')
        axes[1, 0].set_xlabel('Batch Size')
        axes[1, 0].set_ylabel('GPU Memory Reserved (MB)')
        axes[1, 0].set_title('GPU Memory Usage vs Batch Size')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xscale('log', base=2)

        # Plot 4: Efficiency (samples/sec per MB)
        efficiency = np.array(samples_per_sec) / np.array(memory_used)
        axes[1, 1].plot(batch_sizes, efficiency, 'o-', linewidth=2, markersize=8, color='purple')
        axes[1, 1].set_xlabel('Batch Size')
        axes[1, 1].set_ylabel('Efficiency (samples/s per MB)')
        axes[1, 1].set_title('Memory Efficiency vs Batch Size')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_xscale('log', base=2)

        plt.tight_layout()

        # Save figure
        plot_path = self.output_dir / 'batch_size_tuning.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved visualization to {plot_path}")

        plt.close()

    def _print_summary(self):
        """Print summary of results."""
        logger.info(f"\n{'='*70}")
        logger.info("SUMMARY")
        logger.info(f"{'='*70}")

        successful = [r for r in self.results if r['success']]
        failed = [r for r in self.results if not r['success']]

        logger.info(f"Total experiments: {len(self.results)}")
        logger.info(f"Successful: {len(successful)}")
        logger.info(f"Failed: {len(failed)}")

        if successful:
            logger.info(f"\nSuccessful batch sizes:")
            for r in successful:
                logger.info(
                    f"  batch_size={r['batch_size']:3d}: "
                    f"{r['training_time']:6.2f}s, "
                    f"{r['samples_per_second']:6.1f} samples/s, "
                    f"{r['gpu_memory']['reserved_mb']:6.0f} MB"
                )

            # Find optimal batch size (best samples/sec)
            best = max(successful, key=lambda x: x['samples_per_second'])
            logger.info(f"\n✓ RECOMMENDED batch_size: {best['batch_size']}")
            logger.info(f"  Training speed: {best['samples_per_second']:.1f} samples/s")
            logger.info(f"  GPU memory: {best['gpu_memory']['reserved_mb']:.0f} MB")

        if failed:
            logger.info(f"\nFailed batch sizes:")
            for r in failed:
                reason = "OOM" if r['oom_error'] else "Error"
                logger.info(f"  batch_size={r['batch_size']:3d}: {reason}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Tune batch size for PEP model training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--base-config',
        type=str,
        required=True,
        help='Path to base TOML configuration file'
    )

    parser.add_argument(
        '--batch-sizes',
        type=int,
        nargs='+',
        default=[2, 4, 8, 16, 32, 64],
        help='List of batch sizes to test'
    )

    parser.add_argument(
        '--n-samples',
        type=int,
        default=500,
        help='Number of samples for training (dataset will be cycled if needed)'
    )

    parser.add_argument(
        '--n-epochs',
        type=int,
        default=1,
        help='Number of epochs to train'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/batch_size_tuning',
        help='Directory to save results'
    )

    parser.add_argument(
        '--no-cleanup',
        action='store_true',
        help='Do not delete temporary files (useful for debugging)'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run tuner
    tuner = BatchSizeTuner(
        base_config_path=args.base_config,
        batch_sizes=args.batch_sizes,
        n_samples=args.n_samples,
        n_epochs=args.n_epochs,
        output_dir=args.output_dir,
        cleanup=not args.no_cleanup,
    )

    tuner.run()


if __name__ == '__main__':
    main()
