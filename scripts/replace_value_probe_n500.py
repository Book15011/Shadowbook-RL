"""Adequate-power follow-up to scripts/replace_value_probe.py. At n=50,
best-B vs TWAP had only ~14.7% power to detect the observed -0.482bps effect
(std_diff=3.71) -- see docs/reports/l3_replace_value_probe.md's power-analysis
section for the full derivation. This re-runs ONLY the single pre-registered
config the original sweep identified as best-B (init=-5, N=100, step=1)
against TWAP, at n=500 (~83% power for the observed effect size), on the same
held-out val population. Deliberately does NOT re-sweep the 18-config B grid,
and does NOT include a PASSIVE arm -- confirmed separately that no PASSIVE
configuration in that policy family reaches comparable (TWAP/B-like) fill;
40.4% at offset=0 is a structural ceiling, not a sweep gap, so a fill-fair
PASSIVE comparison cannot be constructed at all, let alone pre-registered.
Testing only the one pre-selected config keeps this a clean, non-exploratory
test rather than another multiple-comparisons search.
"""
from __future__ import annotations

import json

from scripts.phase2a_sanity_suite import TWAPPolicy
from scripts.replace_value_probe import (
    HORIZON_TICKS,
    LOOKBACK_TICKS,
    EVAL_SEED_BASE,
    ReplaceActivePolicy,
    run_config,
    paired_report,
)
from src.data.split import load_split
from src.envs.lob_execution_env import LOBExecutionEnv

N_EVAL_EPISODES = 500
BEST_B_LABEL = "B(init=-5,N=100,step=1)"


def main() -> None:
    val_dates = load_split("val")
    val_date_range = (val_dates[0].isoformat(), val_dates[-1].isoformat())
    print(f"val date_range: {val_date_range} ({len(val_dates)} days)")
    seeds = [EVAL_SEED_BASE + i for i in range(N_EVAL_EPISODES)]
    print(f"n={N_EVAL_EPISODES} paired seeds {seeds[0]}..{seeds[-1]}")

    env = LOBExecutionEnv(
        date_range=val_date_range, horizon_ticks=HORIZON_TICKS, lookback_ticks=LOOKBACK_TICKS
    )

    best_b = run_config(env, ReplaceActivePolicy(-5, 100, 1), seeds, BEST_B_LABEL)
    twap = run_config(env, TWAPPolicy(n_slices=10), seeds, "TWAP")

    comparison = paired_report("TWAP", twap, BEST_B_LABEL, best_b)

    out = {
        "n": N_EVAL_EPISODES,
        "seeds": seeds,
        "best_b_label": BEST_B_LABEL,
        "best_b": {"is_bps": best_b["is_bps"].tolist(), "fill_ratio": best_b["fill_ratio"].tolist()},
        "twap": {"is_bps": twap["is_bps"].tolist(), "fill_ratio": twap["fill_ratio"].tolist()},
        "comparison": comparison,
    }
    with open("/tmp/replace_value_probe_n500_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved to /tmp/replace_value_probe_n500_results.json")


if __name__ == "__main__":
    main()
