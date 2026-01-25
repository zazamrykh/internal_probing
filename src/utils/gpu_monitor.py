"""
GPU memory monitoring utilities for batch size tuning experiments.

Provides functions to track GPU memory usage during training.
"""

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def get_gpu_memory_info(device: Optional[int] = None) -> dict[str, float]:
    """
    Get current GPU memory usage information.

    Args:
        device: GPU device index (None = current device)

    Returns:
        Dictionary with memory info in MB:
        - allocated_mb: Memory allocated by tensors
        - reserved_mb: Memory reserved by caching allocator
        - free_mb: Free memory available
        - total_mb: Total GPU memory
        - utilization: Memory utilization percentage (0-100)

    Example:
        >>> info = get_gpu_memory_info()
        >>> print(f"Using {info['allocated_mb']:.0f} MB / {info['total_mb']:.0f} MB")
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, returning zero memory info")
        return {
            'allocated_mb': 0.0,
            'reserved_mb': 0.0,
            'free_mb': 0.0,
            'total_mb': 0.0,
            'utilization': 0.0,
        }

    if device is None:
        device = torch.cuda.current_device()

    # Get memory stats in bytes
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)

    # Get total memory
    total = torch.cuda.get_device_properties(device).total_memory

    # Calculate free memory (total - reserved)
    free = total - reserved

    # Convert to MB
    mb = 1024 * 1024
    allocated_mb = allocated / mb
    reserved_mb = reserved / mb
    free_mb = free / mb
    total_mb = total / mb

    # Calculate utilization
    utilization = (reserved / total * 100) if total > 0 else 0.0

    return {
        'allocated_mb': allocated_mb,
        'reserved_mb': reserved_mb,
        'free_mb': free_mb,
        'total_mb': total_mb,
        'utilization': utilization,
    }


def log_gpu_memory_info(device: Optional[int] = None, prefix: str = ""):
    """
    Log current GPU memory usage.

    Args:
        device: GPU device index (None = current device)
        prefix: Optional prefix for log message

    Example:
        >>> log_gpu_memory_info(prefix="After training:")
    """
    info = get_gpu_memory_info(device)

    if info['total_mb'] == 0:
        logger.info(f"{prefix}GPU not available")
        return

    logger.info(
        f"{prefix}GPU Memory: "
        f"{info['allocated_mb']:.0f} MB allocated, "
        f"{info['reserved_mb']:.0f} MB reserved, "
        f"{info['free_mb']:.0f} MB free, "
        f"{info['total_mb']:.0f} MB total "
        f"({info['utilization']:.1f}% utilization)"
    )


def clear_gpu_cache():
    """
    Clear GPU cache to free up memory.

    Useful between experiments to ensure clean memory state.

    Example:
        >>> clear_gpu_cache()
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.debug("Cleared GPU cache")


def check_oom_risk(threshold_mb: float = 500, device: Optional[int] = None) -> bool:
    """
    Check if there's risk of OOM error based on free memory.

    Args:
        threshold_mb: Minimum free memory in MB to consider safe
        device: GPU device index (None = current device)

    Returns:
        True if free memory is below threshold (OOM risk), False otherwise

    Example:
        >>> if check_oom_risk(threshold_mb=1000):
        ...     print("Warning: Low GPU memory!")
    """
    info = get_gpu_memory_info(device)

    if info['total_mb'] == 0:
        return False  # No GPU, no OOM risk

    return info['free_mb'] < threshold_mb
