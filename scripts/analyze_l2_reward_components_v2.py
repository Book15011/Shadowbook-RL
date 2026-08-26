"""Task 2 re-measurement: component breakdown under l2_reward_mode="potential_is_shaping"
(docs/reports/l2_reward_redesign_proposal.md's approved primary design, implemented in
src/envs/l2_reward.py). Companion to scripts/analyze_l2_reward_components.py (the OLD
scheme's measurement) -- same 100 real val episodes, same trained... no model needed here
either (uses TWAP-passthrough neutral action, same as the telescoping test, so this
measures the REWARD FUNCTION's own structure, not a policy-specific artifact).

Under the new scheme, L2's entire reward is terminal-IS-derived by construction (the old
r_slip/r_inv/r_queue/r_spread/r_stale/r_placement_stale aggregation is gone, replaced
entirely -- see l2_reward.py). The meaningful re-measurement is the INTERNAL composition of
that terminal-IS-derived signal itself: compute_implementation_shortfall()'s own three
components (exec_contribution, is_opp_bps, fees_bps), decomposed the same way Task 1
decomposed the old six -- as PER-WINDOW DELTAS (since l2_window_reward is itself a delta,
Phi(t)-Phi(t-1), each of these three sub-terms has its own delta contributing to that
total), summed signed and by magnitude across many decisions.

Read-only, no training. Monkeypatches src.envs.l2_reward's own compute_implementation_shortfall
NAME BINDING (same `from X import Y` local-binding subtlety as the original script)."""
from __future__ import annotations

import numpy as np
from sb3_contrib import RecurrentPPO

import src.envs.l2_reward as l2_reward_mod
from src.data.split import load_split
from src.train.train_l2 import make_l2_wrapped_env

N_EPISODES = 100
_TWAP_PASSTHROUGH_ACTION = np.array([1.0, 0.5], dtype=np.float32)

component_totals = {"exec": 0.0, "opp": 0.0, "fees": 0.0}
component_abs_totals = dict(component_totals)
n_windows = 0


def _install_capture():
    orig_compute_is = l2_reward_mod.compute_implementation_shortfall
    prev = {"exec": 0.0, "opp": 0.0, "fees": 0.0}

    def _capture(*, side, fills, qty_total, arrival_price, terminal_mid_price, fee_bps_per_fill):
        global n_windows
        result = orig_compute_is(
            side=side, fills=fills, qty_total=qty_total, arrival_price=arrival_price,
            terminal_mid_price=terminal_mid_price, fee_bps_per_fill=fee_bps_per_fill,
        )
        exec_contribution = result.fill_ratio * result.is_exec_bps if result.is_exec_bps is not None else 0.0
        cur = {"exec": exec_contribution, "opp": result.is_opp_bps, "fees": result.fees_bps}
        for k in component_totals:
            delta = -1.0 * (cur[k] - prev[k])  # kappa=1.0 production default, same sign convention as l2_potential
            component_totals[k] += delta
            component_abs_totals[k] += abs(delta)
        prev.update(cur)
        n_windows += 1
        return result

    l2_reward_mod.compute_implementation_shortfall = _capture
    return prev


def main():
    prev_state = _install_capture()

    val_dates = load_split("val")
    val_date_range = (val_dates[0].isoformat(), val_dates[-1].isoformat())
    l3_model = RecurrentPPO.load("models/l3_frozen_backup/l3_executioner_v1_frozen.zip", device="cpu")
    wrapper = make_l2_wrapped_env(
        val_date_range, horizon_ticks=3000, lookback_ticks=10, l3_model=l3_model,
        l3_vecnormalize_path="models/l3_frozen_backup/l3_vecnormalize_frozen.pkl",
        ticks_per_l2_decision=50, l2_include_prev_action=False,
        data_dir="data/raw_l2_bybit_numeric/BTCUSDT", l3_deterministic=True,
        use_numeric_format=True, l2_reward_mode="potential_is_shaping",
    )

    max_decisions = 3000 // 50 + 1
    seeds = [5_000_000 + i for i in range(N_EPISODES)]
    for seed in seeds:
        wrapper.reset(seed=seed)
        prev_state["exec"] = prev_state["opp"] = prev_state["fees"] = 0.0  # Phi(0)=0 per episode
        for _ in range(max_decisions):
            _, r, term, trunc, _ = wrapper.step(_TWAP_PASSTHROUGH_ACTION)
            if term or trunc:
                break

    print(f"Episodes: {N_EPISODES}   windows seen: {n_windows}")
    print("\nHEADLINE: under potential_is_shaping, L2's ENTIRE reward is terminal-IS-derived")
    print("by construction (100%) -- the old r_slip/r_inv/r_queue/r_spread/r_stale/")
    print("r_placement_stale aggregation no longer exists in L2's reward at all. Compare:")
    print("  OLD (l3_passthrough): terminal-IS-derived = 6.9% of net reward, 11.6% of magnitude")
    print("  NEW (potential_is_shaping): terminal-IS-derived = 100% of net reward, 100% of magnitude")

    print("\n" + "=" * 78)
    print("INTERNAL COMPOSITION of the new signal (exec / opportunity / fees, per-window deltas)")
    print("=" * 78)
    grand_total = sum(component_totals.values())
    for name, total in sorted(component_totals.items(), key=lambda kv: -abs(kv[1])):
        pct = 100 * total / grand_total if grand_total else float("nan")
        print(f"  {name:8s} total={total:12.4f}   {pct:7.2f}% of net total   per-episode mean={total / N_EPISODES:9.4f}")
    print(f"  {'TOTAL':8s} total={grand_total:12.4f}")

    grand_abs = sum(component_abs_totals.values())
    print("\nBy magnitude:")
    for name, total in sorted(component_abs_totals.items(), key=lambda kv: -kv[1]):
        pct = 100 * total / grand_abs if grand_abs else float("nan")
        print(f"  {name:8s} abs_total={total:12.4f}   {pct:7.2f}% of total magnitude")


if __name__ == "__main__":
    main()
