"""Does CANCEL_AND_REPLACE actually improve implementation shortfall in this
environment, at all? Four rounds of reward engineering treated near-0% REPLACE
usage as a problem to fix, without ever testing whether higher usage would
help. This script settles the question directly with hand-written heuristic
policies -- no RL, no training, no GPU, no model loading.

Reuses the same eval population (LOBExecutionEnv, load_split("val")), the same
50 paired seeds (5,000,000..5,000,049), and the same TWAPPolicy/run_episode
helpers as scripts/phase2a_sanity_suite.py and every reproduction script this
project has used. IS_total_bps/fill_ratio come from
compute_implementation_shortfall(), which takes no reward_weights argument at
all and is computed from fills/qty_total/arrival_price/terminal_mid_price only
-- confirmed independent of RewardWeights/reward.py's current state (verified
by reading step()'s source, not assumed; see the probe report for the exact
grep evidence).
"""
from __future__ import annotations

import json
import time

import numpy as np
from scipy import stats

from scripts.phase2a_sanity_suite import TWAPPolicy, run_episode
from src.data.split import load_split
from src.envs.lob_execution_env import (
    LOBExecutionEnv,
    ORDER_TYPE_HOLD,
    ORDER_TYPE_LIMIT,
    ORDER_TYPE_CANCEL_REPLACE,
)

HORIZON_TICKS = 3000
LOOKBACK_TICKS = 10
EVAL_SEED_BASE = 5_000_000
N_EVAL_EPISODES = 50
FULL_SIZE_FRAC_IDX = 4  # SIZE_FRACTIONS[4] == 1.0 -- see lob_execution_env.py


class PassivePolicy:
    """A. PASSIVE(offset): place a single LIMIT at a fixed offset, full
    remaining size (no repricing ever, so no reason to hold size back), then
    HOLD every subsequent tick until filled or horizon. offset follows
    LOBExecutionEnv's own convention (_place_limit): higher offset is more
    aggressive for BOTH sides (price = best_bid + offset*TICK for buy,
    best_ask - offset*TICK for sell), price_offset_idx = offset + 5."""

    def __init__(self, offset: int):
        self.offset = offset
        self._placed = False

    def reset(self, env) -> None:
        self._placed = False

    def act(self, env, info: dict) -> np.ndarray:
        if env.qty_remaining <= 0 or self._placed:
            return np.array([ORDER_TYPE_HOLD, 5, 0])
        self._placed = True
        return np.array([ORDER_TYPE_LIMIT, self.offset + 5, FULL_SIZE_FRAC_IDX])


class ReplaceActivePolicy:
    """B. REPLACE-ACTIVE(initial_offset, staleness_n, step): place a LIMIT at
    initial_offset, full size. Every tick, if an order is resting and unfilled
    for >= staleness_n ticks, CANCEL_AND_REPLACE at a more aggressive offset
    (current offset += step, capped at +5 = guaranteed-crossing/marketable).
    If nothing is resting and qty_remaining > 0 (first placement, or a prior
    crossing replace only partially filled and consumed the resting slot),
    place fresh at the current (possibly already-escalated) offset -- the
    escalation itself only advances on a staleness-triggered replace of an
    order that is actually still resting, not on every re-placement."""

    def __init__(self, initial_offset: int, staleness_n: int, step: int):
        self.initial_offset = initial_offset
        self.staleness_n = staleness_n
        self.step = step
        self._current_offset = initial_offset

    def reset(self, env) -> None:
        self._current_offset = self.initial_offset

    def act(self, env, info: dict) -> np.ndarray:
        if env.qty_remaining <= 0:
            return np.array([ORDER_TYPE_HOLD, 5, 0])
        offset_idx = min(10, max(0, self._current_offset + 5))
        if env._resting is None:
            return np.array([ORDER_TYPE_LIMIT, offset_idx, FULL_SIZE_FRAC_IDX])
        ticks_resting = env._tick_idx - env._resting_placed_tick_idx
        if ticks_resting >= self.staleness_n:
            self._current_offset = min(5, self._current_offset + self.step)
            offset_idx = min(10, max(0, self._current_offset + 5))
            return np.array([ORDER_TYPE_CANCEL_REPLACE, offset_idx, FULL_SIZE_FRAC_IDX])
        return np.array([ORDER_TYPE_HOLD, 5, 0])


def run_config(env, policy, seeds, label: str) -> dict:
    t0 = time.time()
    is_bps, fill_ratios, n_replace, n_market, n_total = [], [], 0, 0, 0
    for seed in seeds:
        result = run_episode(env, policy, seed=seed, horizon_ticks=HORIZON_TICKS)
        is_bps.append(result["is_result"].is_total_bps)
        fill_ratios.append(result["is_result"].fill_ratio)
    is_bps = np.array(is_bps)
    fill_ratios = np.array(fill_ratios)
    dt = time.time() - t0
    print(
        f"  {label:40s} IS_total_bps mean={is_bps.mean():8.4f} std={is_bps.std():7.4f} "
        f"fill_ratio={fill_ratios.mean():.4f}  ({dt:.1f}s)"
    )
    return {"label": label, "is_bps": is_bps, "fill_ratio": fill_ratios, "seconds": dt}


def paired_report(name_a: str, r_a: dict, name_b: str, r_b: dict) -> dict:
    diff = r_b["is_bps"] - r_a["is_bps"]
    t_stat, t_p = stats.ttest_rel(r_b["is_bps"], r_a["is_bps"])
    w_stat, w_p = stats.wilcoxon(r_b["is_bps"], r_a["is_bps"])
    print(
        f"\n{name_b} vs {name_a}: mean diff ({name_b}-{name_a})={diff.mean():.4f}bps "
        f"std={diff.std():.4f}"
    )
    print(f"  paired t-test:        t={t_stat:.4f}  p={t_p:.4f}")
    print(f"  Wilcoxon signed-rank: W={w_stat:.4f}  p={w_p:.4f}")
    return {
        "a": name_a, "b": name_b, "mean_diff": float(diff.mean()), "std_diff": float(diff.std()),
        "t_stat": float(t_stat), "t_p": float(t_p), "w_stat": float(w_stat), "w_p": float(w_p),
    }


def main() -> None:
    val_dates = load_split("val")
    val_date_range = (val_dates[0].isoformat(), val_dates[-1].isoformat())
    print(f"val date_range: {val_date_range} ({len(val_dates)} days)")
    seeds = [EVAL_SEED_BASE + i for i in range(N_EVAL_EPISODES)]

    env = LOBExecutionEnv(
        date_range=val_date_range, horizon_ticks=HORIZON_TICKS, lookback_ticks=LOOKBACK_TICKS
    )

    print("\n=== A: PASSIVE(offset) sweep ===")
    a_offsets = [-5, -4, -3, -2, -1, 0, 1]
    a_results = {}
    for offset in a_offsets:
        label = f"A(offset={offset:+d})"
        a_results[label] = run_config(env, PassivePolicy(offset), seeds, label)

    print("\n=== B: REPLACE-ACTIVE(initial_offset, staleness_n, step) sweep ===")
    b_initial_offsets = [-5, -3, -1]
    b_staleness_ns = [20, 100, 300]
    b_steps = [1, 2]
    b_results = {}
    for io in b_initial_offsets:
        for sn in b_staleness_ns:
            for st in b_steps:
                label = f"B(init={io:+d},N={sn},step={st})"
                b_results[label] = run_config(
                    env, ReplaceActivePolicy(io, sn, st), seeds, label
                )

    print("\n=== C: TWAP baseline ===")
    twap_result = run_config(env, TWAPPolicy(n_slices=10), seeds, "TWAP")

    best_a_label = min(a_results, key=lambda k: a_results[k]["is_bps"].mean())
    best_b_label = min(b_results, key=lambda k: b_results[k]["is_bps"].mean())
    best_a = a_results[best_a_label]
    best_b = b_results[best_b_label]

    print(f"\n=== Best A: {best_a_label} (IS_total_bps mean={best_a['is_bps'].mean():.4f}) ===")
    print(f"=== Best B: {best_b_label} (IS_total_bps mean={best_b['is_bps'].mean():.4f}) ===")

    print("\n" + "=" * 70)
    print("PAIRED COMPARISONS (n=50, same seeds every arm)")
    print("=" * 70)
    comparisons = {
        "best_b_vs_best_a": paired_report(best_a_label, best_a, best_b_label, best_b),
        "best_a_vs_twap": paired_report("TWAP", twap_result, best_a_label, best_a),
        "best_b_vs_twap": paired_report("TWAP", twap_result, best_b_label, best_b),
    }

    n_b_configs = len(b_results)
    bonferroni_alpha = 0.05 / n_b_configs
    print(
        f"\nMULTIPLE-COMPARISONS NOTE: {n_b_configs} B configurations were swept and the "
        f"best one selected post hoc -- the naive p-value on best_b_vs_best_a above is "
        f"optimistic (exploratory, not a pre-registered single test). Bonferroni-corrected "
        f"significance threshold for the B-sweep dimension: alpha={bonferroni_alpha:.5f} "
        f"(0.05/{n_b_configs}). best_b_vs_best_a p={comparisons['best_b_vs_best_a']['t_p']:.4f} "
        f"{'CLEARS' if comparisons['best_b_vs_best_a']['t_p'] < bonferroni_alpha else 'DOES NOT CLEAR'} "
        f"this corrected bar."
    )

    out = {
        "val_date_range": val_date_range,
        "seeds": seeds,
        "a_results": {
            k: {"is_bps": v["is_bps"].tolist(), "fill_ratio": v["fill_ratio"].tolist(), "seconds": v["seconds"]}
            for k, v in a_results.items()
        },
        "b_results": {
            k: {"is_bps": v["is_bps"].tolist(), "fill_ratio": v["fill_ratio"].tolist(), "seconds": v["seconds"]}
            for k, v in b_results.items()
        },
        "twap_result": {
            "is_bps": twap_result["is_bps"].tolist(), "fill_ratio": twap_result["fill_ratio"].tolist(),
        },
        "best_a_label": best_a_label,
        "best_b_label": best_b_label,
        "comparisons": comparisons,
        "bonferroni_alpha": bonferroni_alpha,
        "n_b_configs": n_b_configs,
        "n_a_configs": len(a_results),
    }
    with open("/tmp/replace_value_probe_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved full results to /tmp/replace_value_probe_results.json")


if __name__ == "__main__":
    main()
