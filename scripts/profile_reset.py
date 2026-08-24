"""Profiles env.reset() internals directly (monkeypatch instrumentation, no source
files modified) -- breaks down the 51% reset() share already measured
(docs/reports/phase4_l2_reconciliation_and_plan.md) into _load_day (tagged hit/miss),
_build_ticks, _precompute_feature_series, and everything else.

Two scenarios, both real data, real code path:
  A) the same fixed 10-day pool scripts/benchmark_controlled.py uses (cache holds 5 of 10)
  B) the real, full 405-day train split (cache holds 5 of 405 -- near-always-miss)

Each scenario: reset(seed=42) once, then ~40 more reset() calls with NO seed argument --
matching how SB3/SubprocVecEnv actually drives an env after its one seeded initial reset
(the env's own internal RNG advances and picks new files on each subsequent call).

Run: PYTHONPATH=. .venv/bin/python scripts/profile_reset.py
"""
from __future__ import annotations

import statistics
import time

from src.data.split import load_split
from src.envs.lob_execution_env import LOBExecutionEnv

N_RESETS = 40


def _instrument(env_cls):
    orig_load_day = env_cls._load_day
    orig_build_ticks = env_cls._build_ticks
    orig_precompute = env_cls._precompute_feature_series

    timings = {"load_day_hit": [], "load_day_miss": [], "build_ticks": [], "precompute": []}

    def timed_load_day(self, path):
        was_cached = path in self._day_cache
        t0 = time.perf_counter()
        result = orig_load_day(self, path)
        dt = time.perf_counter() - t0
        timings["load_day_hit" if was_cached else "load_day_miss"].append(dt)
        return result

    def timed_build_ticks(self, day_df, start, end):
        t0 = time.perf_counter()
        result = orig_build_ticks(self, day_df, start, end)
        timings["build_ticks"].append(time.perf_counter() - t0)
        return result

    def timed_precompute(self):
        t0 = time.perf_counter()
        result = orig_precompute(self)
        timings["precompute"].append(time.perf_counter() - t0)
        return result

    env_cls._load_day = timed_load_day
    env_cls._build_ticks = timed_build_ticks
    env_cls._precompute_feature_series = timed_precompute
    return timings, (orig_load_day, orig_build_ticks, orig_precompute)


def _restore(env_cls, originals):
    env_cls._load_day, env_cls._build_ticks, env_cls._precompute_feature_series = originals


def run_scenario(name: str, date_range: tuple[str, str], pool_size: int) -> None:
    print(f"\n=== Scenario {name}: pool_size={pool_size}, date_range={date_range} ===")
    timings, originals = _instrument(LOBExecutionEnv)
    env = LOBExecutionEnv(horizon_ticks=3000, lookback_ticks=10, date_range=date_range)

    total_times = []
    obs, info = env.reset(seed=42)
    for i in range(N_RESETS):
        t0 = time.perf_counter()
        obs, info = env.reset()
        total_times.append(time.perf_counter() - t0)

    _restore(LOBExecutionEnv, originals)

    n_hit = len(timings["load_day_hit"])
    n_miss = len(timings["load_day_miss"])
    n_load = n_hit + n_miss
    hit_rate = 100 * n_hit / n_load if n_load else float("nan")

    mean_total = statistics.mean(total_times)
    mean_hit = statistics.mean(timings["load_day_hit"]) if timings["load_day_hit"] else float("nan")
    mean_miss = statistics.mean(timings["load_day_miss"]) if timings["load_day_miss"] else float("nan")
    mean_build = statistics.mean(timings["build_ticks"])
    mean_precompute = statistics.mean(timings["precompute"])
    mean_load_day_blended = statistics.mean(timings["load_day_hit"] + timings["load_day_miss"])
    mean_other = mean_total - mean_load_day_blended - mean_build - mean_precompute

    print(f"resets measured: {N_RESETS + 1} (1 seeded + {N_RESETS} unseeded, matching real SB3 usage)")
    print(f"_load_day: {n_load} calls, {n_hit} hit / {n_miss} miss -> hit rate {hit_rate:.1f}%")
    print(f"  mean hit:  {mean_hit*1000:.2f}ms" if n_hit else "  mean hit:  n/a (0 hits)")
    print(f"  mean miss: {mean_miss*1000:.2f}ms" if n_miss else "  mean miss: n/a (0 misses)")
    print(f"  blended mean (what a real run actually pays): {mean_load_day_blended*1000:.2f}ms")
    print(f"_build_ticks:  mean {mean_build*1000:.2f}ms ({100*mean_build/mean_total:.1f}% of reset())")
    print(f"_precompute_feature_series: mean {mean_precompute*1000:.2f}ms ({100*mean_precompute/mean_total:.1f}% of reset())")
    print(f"other (legacy_ticks slicing, norm stats, funding z, top_depths, misc draws): "
          f"mean {mean_other*1000:.2f}ms ({100*mean_other/mean_total:.1f}% of reset())")
    print(f"TOTAL reset(): mean {mean_total*1000:.2f}ms, stdev {statistics.stdev(total_times)*1000:.2f}ms "
          f"(n={N_RESETS})")


def main() -> None:
    controlled_pool = load_split("train")[:10]
    controlled_range = (controlled_pool[0].isoformat(), controlled_pool[-1].isoformat())
    run_scenario("A (benchmark's controlled 10-day pool)", controlled_range, 10)

    full_train = load_split("train")
    full_range = (full_train[0].isoformat(), full_train[-1].isoformat())
    run_scenario("B (real full 405-day train split)", full_range, 405)


if __name__ == "__main__":
    main()
