"""Captures a byte-exact snapshot of env.reset()/env.step() outputs for a set of
fixed seeds, real data, real code path -- used to prove seed-equivalence
before/after the reset()-optimization edit (Task 3's hard constraint). Run once
against the unedited code (saves the "before" snapshot), once again after the
edit (saves "after"), then compare_reset_snapshots.py diffs them.

For each fixed seed: env.reset(seed=seed), then step() a fixed, observation-
independent action ([0, 0, 0], the same every tick) for up to MAX_STEPS ticks or
until terminated/truncated -- observation-independent so any divergence in the
captured trajectory can only come from the env's own reset()/step() computation,
never from an action policy reacting differently to slightly different obs.
Also does one extra reset() (no seed, matching how SB3 drives an env after its
first seeded reset) and repeats the same step sequence, to additionally catch any
divergence specific to the day-cache/second-reset path.

Run: PYTHONPATH=. .venv/bin/python scripts/capture_reset_snapshot.py <before|after>
"""
from __future__ import annotations

import pickle
import sys

import numpy as np

from src.data.split import load_split
from src.envs.lob_execution_env import LOBExecutionEnv

SEEDS = list(range(10))
MAX_STEPS = 200
FIXED_ACTION = np.array([0, 0, 0])


def run_one_seed(seed: int) -> dict:
    pool = load_split("train")[:10]
    date_range = (pool[0].isoformat(), pool[-1].isoformat())
    env = LOBExecutionEnv(horizon_ticks=3000, lookback_ticks=10, date_range=date_range)

    result = {}

    obs0, info0 = env.reset(seed=seed)
    result["reset1_obs"] = np.array(obs0, copy=True)
    result["reset1_qty_total"] = env.qty_total
    result["reset1_side"] = env.side
    result["reset1_arrival_price"] = env.arrival_price

    obs_trace, reward_trace = [], []
    terminal_is = None
    for _ in range(MAX_STEPS):
        obs, r, terminated, truncated, info = env.step(FIXED_ACTION)
        obs_trace.append(np.array(obs, copy=True))
        reward_trace.append(float(r))
        if terminated or truncated:
            terminal_is = info["implementation_shortfall"]
            break
    result["reset1_obs_trace"] = np.stack(obs_trace)
    result["reset1_reward_trace"] = np.array(reward_trace, dtype=float)
    result["reset1_terminal_is_total_bps"] = None if terminal_is is None else terminal_is.is_total_bps
    result["reset1_terminal_is_fill_ratio"] = None if terminal_is is None else terminal_is.fill_ratio

    # Second reset, NO seed argument -- matches SB3's own post-first-reset usage,
    # and specifically exercises the day-cache-hit path (this env instance's cache
    # already holds whatever day reset1 loaded).
    obs1, info1 = env.reset()
    result["reset2_obs"] = np.array(obs1, copy=True)
    result["reset2_qty_total"] = env.qty_total
    result["reset2_side"] = env.side
    result["reset2_arrival_price"] = env.arrival_price

    obs_trace2, reward_trace2 = [], []
    terminal_is2 = None
    for _ in range(MAX_STEPS):
        obs, r, terminated, truncated, info = env.step(FIXED_ACTION)
        obs_trace2.append(np.array(obs, copy=True))
        reward_trace2.append(float(r))
        if terminated or truncated:
            terminal_is2 = info["implementation_shortfall"]
            break
    result["reset2_obs_trace"] = np.stack(obs_trace2)
    result["reset2_reward_trace"] = np.array(reward_trace2, dtype=float)
    result["reset2_terminal_is_total_bps"] = None if terminal_is2 is None else terminal_is2.is_total_bps
    result["reset2_terminal_is_fill_ratio"] = None if terminal_is2 is None else terminal_is2.fill_ratio

    return result


def main() -> None:
    tag = sys.argv[1] if len(sys.argv) > 1 else "before"
    snapshot = {seed: run_one_seed(seed) for seed in SEEDS}
    out_path = f"/tmp/reset_snapshot_{tag}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(snapshot, f)
    print(f"wrote {out_path} ({len(SEEDS)} seeds)")


if __name__ == "__main__":
    main()
