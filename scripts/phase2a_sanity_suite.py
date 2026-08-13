"""Phase 2a throwaway evaluation script -- NOT a permanent module (per the
task: "Fixed-TWAP baseline (throwaway script, not a permanent module)").
Combines steps 4 and 5 of the Phase 2a task list since step 4's TWAP
baseline exists solely to be the yardstick step 5's sanity suite compares
against; a separate one-function file would be ceremonial.

Run: PYTHONPATH=. .venv/bin/python scripts/phase2a_sanity_suite.py
"""
from __future__ import annotations

import numpy as np

from src.envs.lob_execution_env import (
    ORDER_TYPE_HOLD,
    ORDER_TYPE_LIMIT,
    ORDER_TYPE_MARKET,
    SIZE_FRACTIONS,
    LOBExecutionEnv,
)

# ---------------------------------------------------------------------------
# Step 4: Fixed-TWAP baseline -- trivial non-learning policy. Passive limit
# order (posted at the touch, offset=0) for 1/N of the ORIGINAL parent order
# per equal time slice, cancel-and-market-order if that slice's target isn't
# fully filled by slice end. ("1/N of remaining inventory" in the task's
# phrasing is read as "the slice's share of the still-outstanding parent
# order" -- literally recomputing 1/N of whatever remains at each slice start
# would asymptotically approach but never reach completion, which can't be
# what's intended for a baseline meant to fully execute by construction.)
# ---------------------------------------------------------------------------


def _closest_size_frac_idx(frac: float) -> int:
    fracs = np.array(SIZE_FRACTIONS)
    return int(np.argmin(np.abs(fracs - frac)))


class TWAPPolicy:
    def __init__(self, n_slices: int = 10) -> None:
        self.n_slices = n_slices
        self._current_slice = -1
        self._qty_remaining_at_slice_start = 0.0

    def reset(self) -> None:
        self._current_slice = -1
        self._qty_remaining_at_slice_start = 0.0

    def act(self, env: LOBExecutionEnv, info: dict) -> np.ndarray:
        slice_ticks = env.horizon_ticks / self.n_slices
        ticks_elapsed = info["ticks_elapsed"]
        slice_idx = min(self.n_slices - 1, int(ticks_elapsed // slice_ticks))
        slice_end_tick = (slice_idx + 1) * slice_ticks

        if slice_idx != self._current_slice:
            self._current_slice = slice_idx
            self._qty_remaining_at_slice_start = env.qty_remaining

        slice_target = env.qty_total / self.n_slices
        filled_this_slice = self._qty_remaining_at_slice_start - env.qty_remaining
        slice_unfilled = max(0.0, slice_target - filled_this_slice)

        if slice_unfilled <= 1e-9 or env.qty_remaining <= 1e-9:
            return np.array([ORDER_TYPE_HOLD, 5, 0])

        is_last_tick_of_slice = (ticks_elapsed + 1) >= slice_end_tick
        frac_of_remaining = min(1.0, slice_unfilled / env.qty_remaining)
        size_idx = _closest_size_frac_idx(frac_of_remaining)

        if is_last_tick_of_slice:
            return np.array([ORDER_TYPE_MARKET, 5, size_idx])  # force this slice's completion
        if env._resting is not None:
            return np.array([ORDER_TYPE_HOLD, 5, 0])  # already resting, let it work
        return np.array([ORDER_TYPE_LIMIT, 5, size_idx])  # offset idx 5 -> offset 0, post at touch
