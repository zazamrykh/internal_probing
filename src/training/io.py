"""
Model I/O utilities for saving and loading trained probes.

Provides functions for:
- Saving probes with metrics and metadata
- Loading probes
- Finding best probes from results
"""

import pickle
import json
from pathlib import Path
from typing import Any, Union, Optional
import pandas as pd


def save_probe(
    probe: Any,
    metrics: dict,
    path: Union[str, Path],
    metadata: Optional[dict] = None
) -> None:
    """
    Save trained probe with metrics and optional metadata.

    Saves three files:
    - {path}.pkl: Pickled probe model
    - {path}_metrics.json: Metrics dictionary
    - {path}_metadata.json: Optional metadata

    Args:
        probe: Trained sklearn-compatible probe
        metrics: Dict with training/evaluation metrics
        path: Base path for saving (without extension)
        metadata: Optional metadata dict (e.g., training config)

    Example:
        >>> save_probe(
        ...     probe=trained_clf,
        ...     metrics={'cv_auc_mean': 0.85, 'test_auc': 0.83},
        ...     path='models/probes/sep_probe',
        ...     metadata={'position': 0, 'layer': 16, 'target': 'se_binary'}
        ... )
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save probe model
    probe_path = path.with_suffix('.pkl')
    with open(probe_path, 'wb') as f:
        pickle.dump(probe, f)

    # Save metrics
    metrics_path = path.parent / f"{path.stem}_metrics.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Save metadata if provided
    if metadata is not None:
        metadata_path = path.parent / f"{path.stem}_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)


def load_probe(
    path: Union[str, Path],
    load_metrics: bool = True,
    load_metadata: bool = True
) -> Union[Any, tuple[Any, dict], tuple[Any, dict, dict]]:
    """
    Load trained probe with optional metrics and metadata.

    Args:
        path: Base path to probe (without extension)
        load_metrics: Whether to load metrics file
        load_metadata: Whether to load metadata file

    Returns:
        If load_metrics=False and load_metadata=False: probe
        If load_metrics=True and load_metadata=False: (probe, metrics)
        If load_metrics=True and load_metadata=True: (probe, metrics, metadata)

    Example:
        >>> probe, metrics = load_probe('models/probes/sep_probe')
        >>> print(f"Loaded probe with test AUC: {metrics['test_auc']:.3f}")
    """
    path = Path(path)

    # Load probe
    probe_path = path.with_suffix('.pkl')
    with open(probe_path, 'rb') as f:
        probe = pickle.load(f)

    if not load_metrics and not load_metadata:
        return probe

    result = [probe]

    # Load metrics if requested
    if load_metrics:
        metrics_path = path.parent / f"{path.stem}_metrics.json"
        if metrics_path.exists():
            with open(metrics_path, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
            result.append(metrics)
        else:
            result.append({})

    # Load metadata if requested
    if load_metadata:
        metadata_path = path.parent / f"{path.stem}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            result.append(metadata)
        else:
            result.append({})

    return tuple(result) if len(result) > 1 else result[0]


def find_best_probe(
    results: Union[list[dict], pd.DataFrame],
    metric: str = "cv_auc_mean",
    target: Optional[str] = None,
    position: Optional[int] = None,
    layer: Optional[int] = None
) -> dict:
    """
    Find best probe from results based on metric.

    Args:
        results: List of result dicts or DataFrame
        metric: Metric to optimize (default: 'cv_auc_mean')
        target: Optional filter by target field
        position: Optional filter by position
        layer: Optional filter by layer

    Returns:
        Dict with best probe's results

    Example:
        >>> results = manager.train_every_combination(...)
        >>> best_sep = find_best_probe(results, target='se_binary')
        >>> print(f"Best SEP: layer={best_sep['layer']}, AUC={best_sep['cv_auc_mean']:.3f}")
    """
    # Convert to DataFrame if needed
    if isinstance(results, list):
        df = pd.DataFrame(results)
    else:
        df = results.copy()

    # Apply filters
    if target is not None:
        df = df[df['target'] == target]
    if position is not None:
        df = df[df['position'] == position]
    if layer is not None:
        df = df[df['layer'] == layer]

    if len(df) == 0:
        raise ValueError("No results match the specified filters")

    # Find best by metric
    best_idx = df[metric].idxmax()
    return df.loc[best_idx].to_dict()


def save_all_probes(
    results: list[dict],
    base_dir: Union[str, Path],
    save_all: bool = False
) -> None:
    """
    Save probes from training results.

    Args:
        results: List of result dicts from train_every_combination()
        base_dir: Base directory for saving probes
        save_all: If True, save all probes; if False, save only best per target

    Example:
        >>> results = manager.train_every_combination(...)
        >>> save_all_probes(results, 'models/probes/experiment1', save_all=False)
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    if save_all:
        # Save all probes
        for i, result in enumerate(results):
            if 'probe' not in result:
                continue

            target = result.get('target', 'unknown')
            pos = result.get('position', 0)
            layer = result.get('layer', 0)

            filename = f"{target}_pos{pos}_layer{layer}"
            path = base_dir / filename

            probe = result.pop('probe')  # Remove probe from metrics
            save_probe(probe, result, path)
            result['probe'] = probe  # Restore probe
    else:
        # Save only best probe per target
        df = pd.DataFrame(results)
        for target in df['target'].unique():
            best = find_best_probe(results, target=target)

            if 'probe' not in best:
                continue

            pos = best.get('position', 0)
            layer = best.get('layer', 0)

            filename = f"best_{target}_pos{pos}_layer{layer}"
            path = base_dir / filename

            probe = best.pop('probe')
            save_probe(probe, best, path)
