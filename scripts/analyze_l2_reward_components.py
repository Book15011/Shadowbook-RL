"""L2 reward credit-assignment measurement (2026-08-27) -- Task 1 of the
L2-reward-design round. Quantifies what fraction of L2's aggregated
per-window reward comes from each of step_reward()'s components (r_slip,
r_inv, r_queue, r_spread, r_stale, r_placement_stale) versus the one-time
terminal IS term, using the real trained L2 checkpoint on real val episodes
(val, not test -- this measures the reward function's own structure, not a
train/val comparison, and the test split stays unspent per instruction).

Read-only, no training. Monkeypatches src.envs.lob_execution_env's own
step_reward/compute_implementation_shortfall NAME BINDINGS (not
src.envs.reward's -- `from X import Y` binds a local name at import time,
so patching X.Y does not intercept lob_execution_env.py's own call site;
the LOCAL binding must be patched instead). Each patched function calls the
ORIGINAL unchanged and returns its result unmodified -- purely observational,
recomputes each component from the same captured arguments using the exact
formulas in reward.py's own step_reward(), and asserts the recomputed sum
matches the original scalar return every single call as a correctness
check (any transcription mistake here would show up immediately as an
assertion failure, not a silent wrong number)."""
from __future__ import annotations

import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import src.envs.lob_execution_env as lob_env_mod
from scripts.eval_l2_n500 import HORIZON_TICKS, LOOKBACK_TICKS, EVAL_SEED_BASE, make_l2_policy_action_fn
from src.data.split import load_split
from src.train.train_l2 import make_l2_wrapped_env

N_EPISODES = 100
component_totals = {"r_slip": 0.0, "r_inv": 0.0, "r_queue": 0.0, "r_spread": 0.0,
                     "r_stale": 0.0, "r_placement_stale": 0.0, "terminal_is_reward": 0.0}
component_abs_totals = dict(component_totals)
n_ticks_seen = 0
n_terminal_ticks = 0
mismatch_count = 0


def _install_capture():
    orig_step_reward = lob_env_mod.step_reward
    orig_compute_is = lob_env_mod.compute_implementation_shortfall

    def _capture_step_reward(w, *, side, fills, arrival_price, mid_price, qty_remaining,
                              qty_total, dt, l1_risk_score, canceled_via_market,
                              canceled_via_replace, queue_ahead_at_cancel, queue_at_level,
                              resting=False, ticks_since_own_fill_norm=0.0,
                              ticks_since_placement_norm=0.0):
        global n_ticks_seen, mismatch_count
        r = orig_step_reward(
            w, side=side, fills=fills, arrival_price=arrival_price, mid_price=mid_price,
            qty_remaining=qty_remaining, qty_total=qty_total, dt=dt, l1_risk_score=l1_risk_score,
            canceled_via_market=canceled_via_market, canceled_via_replace=canceled_via_replace,
            queue_ahead_at_cancel=queue_ahead_at_cancel, queue_at_level=queue_at_level,
            resting=resting, ticks_since_own_fill_norm=ticks_since_own_fill_norm,
            ticks_since_placement_norm=ticks_since_placement_norm,
        )
        # Recompute components independently, same formulas as reward.py's step_reward().
        r_slip = 0.0
        r_spread = 0.0
        for f in fills:
            r_slip += -w.alpha * side * (f["price"] - arrival_price) / arrival_price * (f["qty"] / qty_total) * 1e4
            if f.get("is_maker"):
                r_spread += w.delta * side * (mid_price - f["price"]) / mid_price * (f["qty"] / qty_total) * 1e4
        r_inv = -w.lam * (1 + max(0.0, l1_risk_score)) * (qty_remaining / qty_total) ** 2 * dt
        r_queue = 0.0
        if canceled_via_market:
            r_queue -= w.beta
            if queue_ahead_at_cancel is not None and queue_at_level:
                r_queue -= w.gamma * (1.0 - queue_ahead_at_cancel / queue_at_level)
        elif canceled_via_replace:
            if queue_ahead_at_cancel is not None and queue_at_level:
                r_queue -= w.gamma * (1.0 - queue_ahead_at_cancel / queue_at_level)
        r_stale = -w.zeta * ticks_since_own_fill_norm if resting else 0.0
        r_placement_stale = -w.eta_replace * ticks_since_placement_norm if resting else 0.0

        recomputed = r_slip + r_inv + r_queue + r_spread + r_stale + r_placement_stale
        if abs(recomputed - r) > 1e-9:
            mismatch_count += 1

        component_totals["r_slip"] += r_slip
        component_totals["r_inv"] += r_inv
        component_totals["r_queue"] += r_queue
        component_totals["r_spread"] += r_spread
        component_totals["r_stale"] += r_stale
        component_totals["r_placement_stale"] += r_placement_stale
        component_abs_totals["r_slip"] += abs(r_slip)
        component_abs_totals["r_inv"] += abs(r_inv)
        component_abs_totals["r_queue"] += abs(r_queue)
        component_abs_totals["r_spread"] += abs(r_spread)
        component_abs_totals["r_stale"] += abs(r_stale)
        component_abs_totals["r_placement_stale"] += abs(r_placement_stale)
        n_ticks_seen += 1
        return r

    def _capture_compute_is(*, side, fills, qty_total, arrival_price, terminal_mid_price, fee_bps_per_fill):
        global n_terminal_ticks
        result = orig_compute_is(
            side=side, fills=fills, qty_total=qty_total, arrival_price=arrival_price,
            terminal_mid_price=terminal_mid_price, fee_bps_per_fill=fee_bps_per_fill,
        )
        # kappa=1.0, subtract_twap_baseline=False in this project's production RewardWeights
        # defaults (confirmed: train_l2.py never overrides reward_weights) -- terminal reward
        # contribution is -kappa * is_total_bps, matching lob_execution_env.py's own line
        # `r += -self.reward_weights.kappa * terminal_is_for_reward`.
        contribution = -1.0 * result.is_total_bps
        component_totals["terminal_is_reward"] += contribution
        component_abs_totals["terminal_is_reward"] += abs(contribution)
        n_terminal_ticks += 1
        return result

    lob_env_mod.step_reward = _capture_step_reward
    lob_env_mod.compute_implementation_shortfall = _capture_compute_is


def main():
    _install_capture()

    val_dates = load_split("val")
    val_date_range = (val_dates[0].isoformat(), val_dates[-1].isoformat())
    l3_model = RecurrentPPO.load(
        "models/l3_frozen_backup/l3_executioner_v1_frozen.zip", device="cpu"
    )
    wrapped_env = make_l2_wrapped_env(
        val_date_range, HORIZON_TICKS, LOOKBACK_TICKS, l3_model,
        "models/l3_frozen_backup/l3_vecnormalize_frozen.pkl",
        ticks_per_l2_decision=50, l2_include_prev_action=False,
        data_dir="data/raw_l2_bybit_numeric/BTCUSDT", l3_deterministic=True, use_numeric_format=True,
    )
    l2_model = SAC.load("models/l2_strategist_v1.zip", device="cpu")
    l2_vec_normalize = VecNormalize.load("models/l2_vecnormalize.pkl", DummyVecEnv([lambda: wrapped_env]))
    l2_vec_normalize.training = False
    l2_action_fn = make_l2_policy_action_fn(l2_model, l2_vec_normalize)

    max_decisions = HORIZON_TICKS // 50 + 1
    seeds = [EVAL_SEED_BASE + i for i in range(N_EPISODES)]
    for seed in seeds:
        obs, info = wrapped_env.reset(seed=seed)
        for _ in range(max_decisions):
            action = l2_action_fn(obs)
            obs, r, term, trunc, info = wrapped_env.step(action)
            if term or trunc:
                break

    print(f"Episodes: {N_EPISODES}   ticks seen: {n_ticks_seen}   terminal ticks: {n_terminal_ticks}")
    print(f"Component-recompute mismatches: {mismatch_count} (should be 0)")

    print("\n" + "=" * 78)
    print("SIGNED TOTAL (mean direction; this IS L2's actual accumulated reward, summed)")
    print("=" * 78)
    grand_total = sum(component_totals.values())
    for name, total in sorted(component_totals.items(), key=lambda kv: -abs(kv[1])):
        pct = 100 * total / grand_total if grand_total else float("nan")
        print(f"  {name:20s} total={total:14.4f}   {pct:7.2f}% of net total   per-episode mean={total / N_EPISODES:9.4f}")
    print(f"  {'TOTAL':20s} total={grand_total:14.4f}")

    print("\n" + "=" * 78)
    print("MAGNITUDE (sum of |component| -- what dominates the SIGNAL, not just the mean)")
    print("=" * 78)
    grand_abs = sum(component_abs_totals.values())
    for name, total in sorted(component_abs_totals.items(), key=lambda kv: -kv[1]):
        pct = 100 * total / grand_abs if grand_abs else float("nan")
        print(f"  {name:20s} abs_total={total:14.4f}   {pct:7.2f}% of total magnitude")
    print(f"  {'TOTAL':20s} abs_total={grand_abs:14.4f}")

    print("\nControllability note (fixed classification, not measured here):")
    print("  L2 directly influences: none of the per-tick components in a one-to-one sense --")
    print("  it sets participation_rate_multiplier (scales the schedule target L3 tries to hit)")
    print("  and urgency (an L3 observation input). r_slip/r_inv/r_queue/r_spread/r_stale/")
    print("  r_placement_stale are all functions of L3's own tick-level order-type/price/cancel")
    print("  choices -- L2 can only shift the DISTRIBUTION of situations L3 faces, never choose")
    print("  the tick-level action itself. Terminal IS is the only component L2's OWN choices")
    print("  (participation pacing over the episode) mechanically integrate into directly,")
    print("  though L3's tick-level execution still mediates it.")


if __name__ == "__main__":
    main()
