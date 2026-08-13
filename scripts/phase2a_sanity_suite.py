"""Phase 2a throwaway evaluation script -- NOT a permanent module (per the
task: "Fixed-TWAP baseline (throwaway script, not a permanent module)").
Combines steps 4 and 5 of the Phase 2a task list since step 4's TWAP
baseline exists solely to be the yardstick step 5's sanity suite compares
against; a separate one-function file would be ceremonial.

Run: PYTHONPATH=. .venv/bin/python scripts/phase2a_sanity_suite.py
"""
from __future__ import annotations

import numpy as np
import pytest

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


# ---------------------------------------------------------------------------
# Step 5: sanity suite. Per the task, this IS "Phase 2 done" -- the actual
# correctness bar for the env/matching-engine/reward stack, not a nice-to-have.
# ---------------------------------------------------------------------------


class NoOpPolicy:
    """Never trades -- exists purely to prove the IS decomposition's no-fill
    edge case (see reward.py module docstring) is correct end-to-end."""

    def reset(self, env: LOBExecutionEnv) -> None:
        pass

    def act(self, env: LOBExecutionEnv, info: dict) -> np.ndarray:
        return np.array([ORDER_TYPE_HOLD, 5, 0])


class OraclePolicy:
    """Cheats: at reset(), pre-scans the ENTIRE episode window (including
    ticks the agent hasn't reached yet) to find the single best mid-price
    tick for this order's side, waits (HOLD) until that tick, then executes
    the full remaining quantity there in one MARKET order. A real agent
    cannot see the future; this exists only to sanity-check that near-
    optimal execution registers as near-zero IS and beats TWAP."""

    def __init__(self) -> None:
        self._target_tick_idx: int | None = None

    def reset(self, env: LOBExecutionEnv) -> None:
        start = env._episode_start
        end = min(start + env.horizon_ticks, len(env._ticks))
        mids = np.array([env._ticks[i].mid_price for i in range(start, end)])
        best_offset = int(np.argmin(mids)) if env.side == 1 else int(np.argmax(mids))
        self._target_tick_idx = start + best_offset

    def act(self, env: LOBExecutionEnv, info: dict) -> np.ndarray:
        if env.qty_remaining <= 1e-9:
            return np.array([ORDER_TYPE_HOLD, 5, 0])
        if env._tick_idx >= self._target_tick_idx:
            return np.array([ORDER_TYPE_MARKET, 5, 4])
        return np.array([ORDER_TYPE_HOLD, 5, 0])


def run_episode(env: LOBExecutionEnv, policy, seed: int, horizon_ticks: int):
    obs, info = env.reset(seed=seed)
    if hasattr(policy, "reset"):
        sig_reset = policy.reset
        try:
            sig_reset(env)
        except TypeError:
            sig_reset()
    total_reward = 0.0
    for _ in range(horizon_ticks + 1):
        action = policy.act(env, info)
        obs, r, term, trunc, info = env.step(action)
        total_reward += r
        if term or trunc:
            break
    is_result = info["implementation_shortfall"]
    maker_qty = sum(f["qty"] for f in env._episode_fills if f.get("is_maker"))
    taker_qty = sum(f["qty"] for f in env._episode_fills if not f.get("is_maker"))
    filled_qty = maker_qty + taker_qty
    maker_fill_frac = maker_qty / filled_qty if filled_qty > 0 else None
    return {
        "total_reward": total_reward,
        "is_result": is_result,
        "qty_total": env.qty_total,
        "qty_remaining": env.qty_remaining,
        "side": env.side,
        "scenario_depth_ratio": info["scenario_depth_ratio"],
        "ticks_elapsed": info["ticks_elapsed"],
        "maker_qty": maker_qty,
        "taker_qty": taker_qty,
        "maker_fill_frac": maker_fill_frac,
        "n_fills": len(env._episode_fills),
    }


def _fmt_is(is_result) -> str:
    exec_str = f"{is_result.is_exec_bps:.4f}" if is_result.is_exec_bps is not None else "None (undefined, 0 fills)"
    return (
        f"fill_ratio={is_result.fill_ratio:.4f} is_exec_bps={exec_str} "
        f"is_opp_bps={is_result.is_opp_bps:.4f} fees_bps={is_result.fees_bps:.4f} "
        f"is_total_bps={is_result.is_total_bps:.4f}"
    )


def main() -> None:
    HORIZON_TICKS = 2000
    N_EPISODES = 50

    print("=" * 78)
    print("PHASE 2a SANITY SUITE")
    print("=" * 78)

    # -----------------------------------------------------------------
    # Check A: no-op loses EXACTLY the opportunity-cost IS component.
    # -----------------------------------------------------------------
    print("\n--- Check A: no-op policy vs Section 5.1 IS decomposition ---")
    env = LOBExecutionEnv(horizon_ticks=HORIZON_TICKS, lookback_ticks=10)
    noop = NoOpPolicy()
    noop_results = []
    for seed in range(10):
        result = run_episode(env, noop, seed, HORIZON_TICKS)
        is_r = result["is_result"]
        assert is_r.fill_ratio == 0.0, f"seed={seed}: no-op should never fill, got fill_ratio={is_r.fill_ratio}"
        assert is_r.is_exec_bps is None, f"seed={seed}: is_exec_bps should be undefined (None) with 0 fills"
        assert is_r.fees_bps == 0.0, f"seed={seed}: fees should be exactly 0 with 0 fills"
        assert is_r.is_total_bps == pytest.approx(is_r.is_opp_bps, abs=1e-9), (
            f"seed={seed}: is_total_bps ({is_r.is_total_bps}) must equal is_opp_bps "
            f"({is_r.is_opp_bps}) EXACTLY for a no-op episode"
        )
        noop_results.append(result)
        print(f"  seed={seed:2d} side={result['side']:+d}  {_fmt_is(is_r)}")
    print(f"  ALL {len(noop_results)} no-op episodes: PASS (fill_ratio=0, is_exec undefined, "
          f"fees=0, is_total==is_opp exactly)")

    # -----------------------------------------------------------------
    # Checks B & C: oracle vs TWAP vs no-op on 50 MATCHED episodes (same
    # seed -> same sampled window/side/size across all three policies).
    # -----------------------------------------------------------------
    print(f"\n--- Checks B & C: oracle vs TWAP vs no-op, {N_EPISODES} matched episodes ---")
    oracle = OraclePolicy()
    twap = TWAPPolicy(n_slices=10)

    oracle_results, twap_results, noop50_results = [], [], []
    for seed in range(N_EPISODES):
        oracle_results.append(run_episode(env, oracle, seed, HORIZON_TICKS))
        twap_results.append(run_episode(env, twap, seed, HORIZON_TICKS))
        noop50_results.append(run_episode(env, noop, seed, HORIZON_TICKS))

    def is_total_list(results):
        return np.array([r["is_result"].is_total_bps for r in results])

    def fill_ratio_list(results):
        return np.array([r["is_result"].fill_ratio for r in results])

    oracle_is = is_total_list(oracle_results)
    twap_is = is_total_list(twap_results)
    noop_is = is_total_list(noop50_results)
    twap_fill = fill_ratio_list(twap_results)
    oracle_fill = fill_ratio_list(oracle_results)

    print(f"\n  oracle IS_total_bps: mean={oracle_is.mean():.4f} std={oracle_is.std():.4f} "
          f"min={oracle_is.min():.4f} max={oracle_is.max():.4f}")
    print(f"  TWAP   IS_total_bps: mean={twap_is.mean():.4f} std={twap_is.std():.4f} "
          f"min={twap_is.min():.4f} max={twap_is.max():.4f}")
    print(f"  no-op  IS_total_bps: mean={noop_is.mean():.4f} std={noop_is.std():.4f} "
          f"min={noop_is.min():.4f} max={noop_is.max():.4f}")

    assert oracle_is.mean() < twap_is.mean(), (
        f"oracle mean IS ({oracle_is.mean():.4f}) should be meaningfully better (lower) "
        f"than TWAP mean IS ({twap_is.mean():.4f})"
    )
    improvement = twap_is.mean() - oracle_is.mean()
    print(f"  oracle beats TWAP by {improvement:.4f} bps on mean IS_total: PASS")

    assert np.all(twap_fill > 0.999), f"TWAP should fully complete every episode by construction, got min fill_ratio={twap_fill.min()}"
    print(f"  TWAP fill_ratio: mean={twap_fill.mean():.6f} min={twap_fill.min():.6f} (all episodes fully completed): PASS")
    print(f"  oracle fill_ratio: mean={oracle_fill.mean():.6f} min={oracle_fill.min():.6f}")
    print(f"  {N_EPISODES}/{N_EPISODES} TWAP episodes ran without crashing: PASS")

    scenario_ratios = np.array([r["scenario_depth_ratio"] for r in twap_results])
    print(f"\n  scenario difficulty (order size / median top-of-book depth): "
          f"mean={scenario_ratios.mean():.3f} min={scenario_ratios.min():.3f} max={scenario_ratios.max():.3f}")

    twap_maker_fracs = np.array([r["maker_fill_frac"] for r in twap_results if r["maker_fill_frac"] is not None])
    twap_n_fills = np.array([r["n_fills"] for r in twap_results])
    print(f"\n  TWAP fill-rate breakdown (maker=passive limit fill, taker=forced market-order fill):")
    print(f"    maker (passive) share of filled qty: mean={twap_maker_fracs.mean():.4f} "
          f"min={twap_maker_fracs.min():.4f} max={twap_maker_fracs.max():.4f}")
    print(f"    taker (forced completion) share of filled qty: mean={(1-twap_maker_fracs).mean():.4f}")
    print(f"    fills per episode: mean={twap_n_fills.mean():.2f} min={twap_n_fills.min()} max={twap_n_fills.max()}")

    print("\n--- Full per-policy summary (50 episodes) ---")
    for name, results in [("no-op", noop50_results), ("oracle", oracle_results), ("TWAP", twap_results)]:
        is_vals = is_total_list(results)
        fill_vals = fill_ratio_list(results)
        rewards = np.array([r["total_reward"] for r in results])
        print(f"  {name:6s}: IS_total_bps mean={is_vals.mean():8.4f} std={is_vals.std():7.4f} | "
              f"fill_ratio mean={fill_vals.mean():.4f} | total_reward mean={rewards.mean():9.4f}")

    print("\n" + "=" * 78)
    print("PHASE 2a SANITY SUITE: ALL CHECKS PASSED")
    print("=" * 78)


if __name__ == "__main__":
    main()
