"""Existing-station catchments (CLAUDE.md §6, 09)."""

from __future__ import annotations

import numpy as np

from khatt.engine.catchment import WalkGraph, assign_nearest, catchments


def station_catchment_pop(walk: WalkGraph, r9_pop: np.ndarray, lat, lng, cutoff_min: float):
    """Residents within `cutoff_min` walk, each r9 cell counted once (nearest station).

    Returns (catchment_pop, snap_m) arrays aligned with the input stations.
    """
    nodes, snap_m = walk.snap(lat, lng)
    catches = catchments(walk, nodes, cutoff_min)
    owner, _ = assign_nearest(catches, len(r9_pop), cutoff_min)
    has = owner >= 0
    pop = np.bincount(owner[has], weights=np.asarray(r9_pop)[has], minlength=len(nodes))
    return pop.astype(np.float32), snap_m
