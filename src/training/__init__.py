"""
Training module for linear probes on internal activations.

Provides:
- ProbeManager: Main class for training and evaluating probes
- Utility functions for data preparation and metrics
- Visualization functions for results analysis
- I/O functions for model persistence
"""

from src.training.base import BaseProbeTrainer
from src.training.probe_manager import ProbeManager
from src.training.pep_trainer import PEPTrainer
from src.training.utils import (
    build_Xy_from_dataset,
    compute_cv_metrics,
    compute_test_metrics,
    merge_datasets
)
from src.training.visualization import (
    plot_auc_by_layer,
    plot_sep_vs_accuracy_auc,
    plot_history_grid
)
from src.training.io import (
    save_probe,
    load_probe,
    find_best_probe,
    save_all_probes
)

__all__ = [
    # Base class
    'BaseProbeTrainer',
    # Main classes
    'ProbeManager',
    'PEPTrainer',
    # Utilities
    'build_Xy_from_dataset',
    'compute_cv_metrics',
    'compute_test_metrics',
    'merge_datasets',
    # Visualization
    'plot_auc_by_layer',
    'plot_sep_vs_accuracy_auc',
    # I/O
    'save_probe',
    'load_probe',
    'find_best_probe',
    'save_all_probes',
    'plot_history_grid'
]
