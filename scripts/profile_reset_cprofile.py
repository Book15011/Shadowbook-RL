"""cProfile-level breakdown of reset(), to see exactly which sub-computation inside
_precompute_feature_series (and _build_ticks) dominates -- finer than the coarse
per-method timing in profile_reset.py.

Run: PYTHONPATH=. .venv/bin/python scripts/profile_reset_cprofile.py
"""
from __future__ import annotations

import cProfile
import pstats

from src.data.split import load_split
from src.envs.lob_execution_env import LOBExecutionEnv


def main() -> None:
    pool = load_split("train")[:10]
    date_range = (pool[0].isoformat(), pool[-1].isoformat())
    env = LOBExecutionEnv(horizon_ticks=3000, lookback_ticks=10, date_range=date_range)
    env.reset(seed=42)

    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(15):
        env.reset()
    profiler.disable()

    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")
    stats.print_stats(25)
    print("\n=== by tottime (self time, excludes sub-calls) ===")
    stats.sort_stats("tottime")
    stats.print_stats(25)


if __name__ == "__main__":
    main()
