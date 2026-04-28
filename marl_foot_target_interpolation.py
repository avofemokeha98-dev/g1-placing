"""Utilities for smoothing MARL foot-target offsets."""

from __future__ import annotations

import torch


def smooth_foot_target_offset(
    current: torch.Tensor,
    previous: torch.Tensor,
    *,
    alpha: float = 0.45,
    max_delta_m: float = 0.035,
) -> torch.Tensor:
    """Exponential smoothing with per-step displacement clamp.

    Args:
        current: Current raw target offsets, shape (N, D).
        previous: Previous smoothed offsets, shape (N, D).
        alpha: Smoothing factor in [0, 1]. Larger = faster tracking.
        max_delta_m: Maximum change per step for each element.
    """
    alpha = float(max(0.0, min(1.0, alpha)))
    max_delta_m = float(max(0.0, max_delta_m))

    blended = previous + alpha * (current - previous)
    if max_delta_m <= 0.0:
        return blended

    delta = blended - previous
    delta = torch.clamp(delta, -max_delta_m, max_delta_m)
    return previous + delta
