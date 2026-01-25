#!/usr/bin/env python3
"""
Convergence analysis script for PEP model training.

Runs experiment with return_history=True, then analyzes convergence
from saved metrics to determine optimal training duration.

Usage:
    python scripts/analyze_convergence.py \
        --config configs/convergence/base.toml \
        --output-dir results/convergence_analysis
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import ExperimentConfig
from src.experiment_runner import ExperimentRunner

logger = logging.getLogger(__name__)


def analyze_convergence(history: list[dict], plateau_threshold: float = 0.01, plateau_window: int = 5) -> dict:
    """
    Analyze convergence from training history.

    Args:
        history: List of validation checkpoints
        plateau_threshold: Improvement threshold for plateau detection
        plateau_window: Number of checkpoints to check for plateau

    Returns:
        Dictionary with convergence analysis
    """
    if not history:
        logger.warning("No history to analyze")
        return {}

    logger.info("Analyzing convergence...")

    # Extract metrics
    iterations = [h['iteration'] for h in history]
    n_samples = [h['n_samples_seen'] for h in history]
    times = [h['elapsed_time'] for h in history]
    val_aucs = [h['val_auc'] for h in history]

    # Find best validation AUC
    best_idx = np.argmax(val_aucs)
    best_val_auc = val_aucs[best_idx]
    best_iteration = iterations[best_idx]
    best_n_samples = n_samples[best_idx]
    best_time = times[best_idx]

    # Detect plateau
    convergence_idx = None
    if len(val_aucs) >= plateau_window + 1:
        for i in range(plateau_window, len(val_aucs)):
            recent_aucs = val_aucs[i-plateau_window:i+1]
            improvements = [recent_aucs[j+1] - recent_aucs[j] for j in range(plateau_window)]

            if all(imp < plateau_threshold for imp in improvements):
                convergence_idx = i
                break

    if convergence_idx is not None:
        convergence_iteration = iterations[convergence_idx]
        convergence_samples = n_samples[convergence_idx]
        convergence_time = times[convergence_idx]
        convergence_auc = val_aucs[convergence_idx]
    else:
        # No plateau detected, use best
        convergence_iteration = best_iteration
        convergence_samples = best_n_samples
        convergence_time = best_time
        convergence_auc = best_val_auc

    # Recommended values with safety margin
    safety_margin = 1.3  # 30% extra
    recommended_samples = int(convergence_samples * safety_margin)
    recommended_time = convergence_time * safety_margin

    analysis = {
        'best_val_auc': float(best_val_auc),
        'best_iteration': int(best_iteration),
        'best_n_samples': int(best_n_samples),
        'best_time': float(best_time),
        'convergence_detected': convergence_idx is not None,
        'convergence_iteration': int(convergence_iteration),
        'convergence_samples': int(convergence_samples),
        'convergence_time': float(convergence_time),
        'convergence_auc': float(convergence_auc),
        'recommended_samples': recommended_samples,
        'recommended_time': float(recommended_time),
        'plateau_threshold': plateau_threshold,
        'plateau_window': plateau_window,
    }

    logger.info(f"Best val AUC: {best_val_auc:.4f} at {best_n_samples} samples")
    logger.info(f"Convergence detected: {analysis['convergence_detected']}")
    if convergence_idx is not None:
        logger.info(f"Convergence at: {convergence_samples} samples ({convergence_time:.1f}s)")
    logger.info(f"Recommended: {recommended_samples} samples ({recommended_time:.1f}s)")

    return analysis


def create_visualizations(history: list[dict], analysis: dict, output_dir: Path):
    """Create convergence visualization plots."""
    if not history:
        logger.warning("No history to visualize")
        return

    logger.info("Creating visualizations...")

    # Extract data
    n_samples = [h['n_samples_seen'] for h in history]
    times = [h['elapsed_time'] for h in history]
    val_aucs = [h['val_auc'] for h in history]

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('PEP Model Convergence Analysis', fontsize=16)

    # Plot 1: AUC vs n_samples
    axes[0].plot(n_samples, val_aucs, 'o-', linewidth=2, markersize=6, label='Validation AUC')
    axes[0].axhline(y=analysis['best_val_auc'], color='g', linestyle='--',
                   label=f"Best: {analysis['best_val_auc']:.4f}")
    if analysis['convergence_detected']:
        axes[0].axvline(x=analysis['convergence_samples'], color='r', linestyle='--',
                       label=f"Convergence: {analysis['convergence_samples']} samples")
    axes[0].axvline(x=analysis['recommended_samples'], color='orange', linestyle=':',
                   label=f"Recommended: {analysis['recommended_samples']} samples")
    axes[0].set_xlabel('Number of Training Samples')
    axes[0].set_ylabel('Validation AUC')
    axes[0].set_title('Convergence: AUC vs Training Samples')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Plot 2: AUC vs time
    axes[1].plot(times, val_aucs, 'o-', linewidth=2, markersize=6, label='Validation AUC', color='purple')
    axes[1].axhline(y=analysis['best_val_auc'], color='g', linestyle='--',
                   label=f"Best: {analysis['best_val_auc']:.4f}")
    if analysis['convergence_detected']:
        axes[1].axvline(x=analysis['convergence_time'], color='r', linestyle='--',
                       label=f"Convergence: {analysis['convergence_time']:.1f}s")
    axes[1].axvline(x=analysis['recommended_time'], color='orange', linestyle=':',
                   label=f"Recommended: {analysis['recommended_time']:.1f}s")
    axes[1].set_xlabel('Training Time (seconds)')
    axes[1].set_ylabel('Validation AUC')
    axes[1].set_title('Convergence: AUC vs Training Time')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()

    # Save figure
    plot_path = output_dir / 'convergence_analysis.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved visualization to {plot_path}")

    plt.close()


def print_summary(analysis: dict):
    """Print summary of convergence analysis."""
    logger.info(f"\n{'='*70}")
    logger.info("CONVERGENCE ANALYSIS SUMMARY")
    logger.info(f"{'='*70}")

    logger.info(f"\nBest Performance:")
    logger.info(f"  Validation AUC: {analysis['best_val_auc']:.4f}")
    logger.info(f"  At samples: {analysis['best_n_samples']}")
    logger.info(f"  At time: {analysis['best_time']:.1f}s")

    if analysis['convergence_detected']:
        logger.info(f"\nConvergence Detected:")
        logger.info(f"  At samples: {analysis['convergence_samples']}")
        logger.info(f"  At time: {analysis['convergence_time']:.1f}s")
        logger.info(f"  Validation AUC: {analysis['convergence_auc']:.4f}")
    else:
        logger.info(f"\nNo clear convergence plateau detected")

    logger.info(f"\n✓ RECOMMENDATIONS:")
    logger.info(f"  Training samples: {analysis['recommended_samples']}")
    logger.info(f"  Training time limit: {analysis['recommended_time']:.1f}s")
    logger.info(f"  (includes 30% safety margin)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Analyze PEP model convergence',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--config',
        type=str,
        default='configs/tuning/convergence_analysis.toml',
        help='Path to experiment TOML config (must have return_history=true in pep_params)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Override output directory from config'
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

    logger.info(f"\n{'='*70}")
    logger.info("Starting convergence analysis")
    logger.info(f"{'='*70}\n")

    # Load config
    config = ExperimentConfig.from_toml(args.config)
    logger.info(f"Loaded config {config}")

    # Override output_dir if specified
    if args.output_dir:
        config.experiment['output_dir'] = args.output_dir

    output_dir = Path(config.experiment['output_dir'])

    # Run experiment
    logger.info("Running experiment with history tracking...")
    runner = ExperimentRunner(config)
    runner.run()

    # Load results
    metrics_path = output_dir / 'metrics.json'
    if not metrics_path.exists():
        logger.error(f"Metrics file not found: {metrics_path}")
        sys.exit(1)

    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    # Extract history from first result
    if not metrics or 'history' not in metrics[0]:
        logger.error("No history found in metrics. Make sure return_history=true in config.")
        sys.exit(1)

    history = metrics[0]['history']
    logger.info(f"Loaded history with {len(history)} checkpoints")

    # Analyze convergence
    analysis = analyze_convergence(history)

    # Create visualizations
    create_visualizations(history, analysis, output_dir)

    # Save analysis
    analysis_path = output_dir / 'convergence_analysis.json'
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    logger.info(f"Saved analysis to {analysis_path}")

    # Print summary
    print_summary(analysis)

    logger.info(f"\n{'='*70}")
    logger.info("Convergence analysis completed!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"{'='*70}\n")


if __name__ == '__main__':
    main()
