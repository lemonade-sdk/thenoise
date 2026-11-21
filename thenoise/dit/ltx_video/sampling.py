"""LTX's released two-stage sigma recipe, resampled to the requested step count.

The distilled release publishes both grids rather than a formula: the main ladder is run
whole at the low resolution, then the video half is spatially upscaled, re-noised to the
published refine level and polished on the refine grid. The request's ``steps`` therefore
sizes the main stage, while the refine pass keeps its published span and the same share of
the recipe it has in the release — three of the eight intervals.
"""
from __future__ import annotations

from thenoise.samplers import sigma_ladder

#: The released distilled ladder: eight intervals, one per denoise step at the default.
MAIN_GRID = (1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0)

#: The refine pass, from the re-noise level the release hands the upscaler to clean.
REFINE_GRID = (0.909375, 0.725, 0.421875, 0.0)


def stage_grids(steps: int) -> tuple[list[float], list[float]]:
    """``(main, refine)`` sigma grids for a request of ``steps`` main steps.

    At the released count this returns both published grids unchanged.
    """
    refine_steps = max(1, round(steps * (len(REFINE_GRID) - 1) / (len(MAIN_GRID) - 1)))
    return sigma_ladder(MAIN_GRID, steps), sigma_ladder(REFINE_GRID, refine_steps)


__all__ = ["MAIN_GRID", "REFINE_GRID", "stage_grids"]
