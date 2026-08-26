"""L2 diagnostics: action distribution, train-vs-val gap, per-day breakdown
(2026-08-26, run after the real n=500 evaluation came back a clean negative --
see docs/TRACK_STATUS.md's L2 section for that result).

Three diagnostics, one script, reusing scripts/eval_l2_n500.py's exact
methodology (same seeds, same run_arm/run_wrapped_episode/paired_report
machinery, IMPORTED not reimplemented) so results are directly
comparable/poolable with that already-recorded n=500 result:

1. Action distribution (arm 1 only): logs L2's actual
   participation_rate_multiplier/urgency at every decision, to distinguish
   policy collapse (near-constant [1.0, 0.5], the neutral/TWAP-passthrough
   action) from active-but-harmful steering.
2. Train-vs-val gap: --split {val,train} selects which real, already-frozen
   date pool (data/splits/l2_bybit_btcusdt_split.json, via src/data/split.py's
   load_split -- the SAME function and SAME range-construction pattern
   train_l2.py's own main() used to pick the real run's train_date_range)
   to evaluate against. Confirmed before writing this: the real training run
   used train_date_range=('2024-04-18','2025-07-15') (405 days) and
   val_date_range=('2025-07-16','2025-08-02') (18 days) -- chronologically
   disjoint, no leakage (see logs/l2_train_real_l2v1_20260825.log's own
   startup print). Running both and comparing IS_total_bps distinguishes
   overfitting (large gap) from no-learnable-signal (both mediocre).
3. Per-day breakdown: every episode's picked calendar day is captured via a
   safe, inert monkeypatch on LOBExecutionEnv._load_day/_load_day_numeric --
   same "capture hook only exists in this process, zero effect on the
   RNG-driven file_idx pick or the returned data" pattern
   scripts/replay_episode.py already validated, just applied at a different,
   naturally-safe capture point (the file_idx pick already happened in
   reset() by the time _load_day(_numeric) is called with the resolved
   path -- wrapping it is pure passthrough, cannot perturb draw order).
   Lets the aggregate n=500 result be split by day to check whether the
   negative is broad-based or driven by a handful of regime-specific days.

Does NOT touch the test split (data/splits/l2_bybit_btcusdt_split.json's
test_dates) -- diagnostics only, per instruction. Does NOT retrain or modify
any checkpoint -- --l2-checkpoint/--l2-vecnormalize/--l3-checkpoint/
--l3-vecnormalize are read-only inputs, same discipline as eval_l2_n500.py.

Correctness check built into this round's own usage (not a separate test
file -- same "throwaway investigation tooling" status as
scripts/benchmark_controlled.py, scripts/profile_reset.py etc. this project
has used before): running this script at --split val with the same
checkpoints/seeds/n=500 MUST reproduce eval_l2_n500.py's already-recorded
arm means (L2=1.2330, TWAP-passthrough=1.0237, Pure TWAP=0.8893) exactly,
since every component is deterministic given a seed (established and
verified earlier this project). Any mismatch means this script introduced a
bug, not a real behavior change -- checked before trusting anything else it
reports.

Run:
  PYTHONPATH=. .venv/bin/python -m scripts.eval_l2_diagnostics \\
    --l2-checkpoint models/l2_strategist_v1.zip \\
    --l2-vecnormalize models/l2_vecnormalize.pkl \\
    --l3-checkpoint models/l3_frozen_backup/l3_executioner_v1_frozen.zip \\
    --l3-vecnormalize models/l3_frozen_backup/l3_vecnormalize_frozen.pkl \\
    --n 500 --use-numeric-format --data-dir data/raw_l2_bybit_numeric/BTCUSDT \\
    --split val --output-json models/l2_diagnostics_val.json \\
    --output-episodes-csv models/l2_diagnostics_val_episodes.csv \\
    --output-actions-csv models/l2_diagnostics_val_actions.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from typing import Any

import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import src.envs.lob_execution_env as lob_env_mod
from scripts.eval_l2_n500 import (
    EVAL_SEED_BASE,
    HORIZON_TICKS,
    LOOKBACK_TICKS,
    _TWAP_PASSTHROUGH_ACTION,
    make_l2_policy_action_fn,
    paired_report,
    run_arm,
    run_wrapped_episode,
)
from scripts.phase2a_sanity_suite import TWAPPolicy, run_episode
from src.data.split import load_split
from src.envs.lob_execution_env import LOBExecutionEnv
from src.train.train_l2 import make_l2_wrapped_env


def install_day_capture() -> dict[str, Any]:
    """See module docstring, Diagnostic 3. Returns a dict whose "path" key is
    updated on every _load_day/_load_day_numeric call -- read it immediately
    after each env.reset() (and before the NEXT env.reset() on any
    LOBExecutionEnv instance) to get that episode's picked file."""
    state: dict[str, Any] = {"path": None}
    orig_load_day = lob_env_mod.LOBExecutionEnv._load_day
    orig_load_day_numeric = lob_env_mod.LOBExecutionEnv._load_day_numeric

    def _capture(self, path):
        state["path"] = path
        return orig_load_day(self, path)

    def _capture_numeric(self, path):
        state["path"] = path
        return orig_load_day_numeric(self, path)

    lob_env_mod.LOBExecutionEnv._load_day = _capture
    lob_env_mod.LOBExecutionEnv._load_day_numeric = _capture_numeric
    return state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="L2 diagnostics: action distribution, train/val gap, per-day breakdown."
    )
    parser.add_argument("--l2-checkpoint", type=str, required=True)
    parser.add_argument("--l2-vecnormalize", type=str, required=True)
    parser.add_argument("--l3-checkpoint", type=str, required=True)
    parser.add_argument("--l3-vecnormalize", type=str, required=True)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--ticks-per-l2-decision", type=int, default=50)
    parser.add_argument("--l2-include-prev-action", action="store_true")
    parser.add_argument("--data-dir", type=str, default="data/raw_l2_bybit/BTCUSDT")
    parser.add_argument("--use-numeric-format", action="store_true")
    parser.add_argument("--device", default="cpu", choices=["cuda", "cpu"])
    parser.add_argument(
        "--split", choices=["val", "train"], default="val",
        help="Which real, frozen date pool to evaluate against (Diagnostic 2). Never 'test'.",
    )
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument(
        "--output-episodes-csv", type=str, default=None,
        help="Per-episode rows (arm,seed,day,is_total_bps,fill_ratio) for Diagnostic 3.",
    )
    parser.add_argument(
        "--output-actions-csv", type=str, default=None,
        help="Per-decision L2 action log (seed,tick_idx,participation_mult,urgency) for Diagnostic 1.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    assert args.split != "test", "DO NOT TOUCH THE TEST SPLIT -- diagnostics only, per instruction."
    seeds = [EVAL_SEED_BASE + i for i in range(args.n)]
    max_decisions = HORIZON_TICKS // args.ticks_per_l2_decision + 1

    dates = load_split(args.split)
    date_range = (dates[0].isoformat(), dates[-1].isoformat())
    print(f"split={args.split} date_range={date_range} ({len(dates)} real days)")
    print(f"n={args.n} paired seeds {seeds[0]}..{seeds[-1]}")
    print(f"l2-checkpoint: {args.l2_checkpoint}")
    print(f"l2-vecnormalize: {args.l2_vecnormalize}")
    print(f"l3-checkpoint: {args.l3_checkpoint}")
    print(f"l3-vecnormalize: {args.l3_vecnormalize}\n")

    day_capture = install_day_capture()

    l3_model = RecurrentPPO.load(args.l3_checkpoint, device="cpu")
    wrapped_env = make_l2_wrapped_env(
        date_range, HORIZON_TICKS, LOOKBACK_TICKS, l3_model, args.l3_vecnormalize,
        args.ticks_per_l2_decision, args.l2_include_prev_action,
        data_dir=args.data_dir, l3_deterministic=True, use_numeric_format=args.use_numeric_format,
    )
    l2_model = SAC.load(args.l2_checkpoint, device=args.device)
    l2_vec_normalize = VecNormalize.load(args.l2_vecnormalize, DummyVecEnv([lambda: wrapped_env]))
    l2_vec_normalize.training = False

    episode_rows: list[dict] = []  # Diagnostic 3
    action_rows: list[dict] = []  # Diagnostic 1

    def _episode_day() -> str:
        p = day_capture["path"]
        return p.stem if p is not None else "unknown"

    l2_action_fn_raw = make_l2_policy_action_fn(l2_model, l2_vec_normalize)

    def l2_action_fn_logged(seed_for_log: int):
        tick_counter = {"i": 0}

        def _fn(obs: np.ndarray) -> np.ndarray:
            action = l2_action_fn_raw(obs)
            action_rows.append(
                {
                    "seed": seed_for_log,
                    "tick_idx": tick_counter["i"],
                    "participation_mult": float(action[0]),
                    "urgency": float(action[1]),
                }
            )
            tick_counter["i"] += 1
            return action

        return _fn

    print("=== Arm 1: trained L2 policy ===")

    def _l2_episode(seed: int) -> dict:
        result = run_wrapped_episode(wrapped_env, seed, l2_action_fn_logged(seed), max_decisions)
        episode_rows.append(
            {
                "arm": "l2", "seed": seed, "day": _episode_day(),
                "is_total_bps": result["is_result"].is_total_bps,
                "fill_ratio": result["is_result"].fill_ratio,
            }
        )
        return result

    arm1 = run_arm("L2 (trained)", seeds, _l2_episode)

    print("\n=== Arm 2: TWAP-passthrough (frozen L3, unsteered) ===")

    def _passthrough_episode(seed: int) -> dict:
        result = run_wrapped_episode(wrapped_env, seed, lambda obs: _TWAP_PASSTHROUGH_ACTION, max_decisions)
        episode_rows.append(
            {
                "arm": "twap_passthrough", "seed": seed, "day": _episode_day(),
                "is_total_bps": result["is_result"].is_total_bps,
                "fill_ratio": result["is_result"].fill_ratio,
            }
        )
        return result

    arm2 = run_arm("TWAP-passthrough", seeds, _passthrough_episode)

    print("\n=== Arm 3: pure TWAP (base env) ===")
    base_env = LOBExecutionEnv(
        data_dir=args.data_dir, date_range=date_range, horizon_ticks=HORIZON_TICKS,
        lookback_ticks=LOOKBACK_TICKS, use_numeric_format=args.use_numeric_format,
    )
    twap_policy = TWAPPolicy(n_slices=10)

    def _twap_episode(seed: int) -> dict:
        result = run_episode(base_env, twap_policy, seed=seed, horizon_ticks=HORIZON_TICKS)
        episode_rows.append(
            {
                "arm": "pure_twap", "seed": seed, "day": _episode_day(),
                "is_total_bps": result["is_result"].is_total_bps,
                "fill_ratio": result["is_result"].fill_ratio,
            }
        )
        return result

    arm3 = run_arm("Pure TWAP", seeds, _twap_episode)

    print("\n" + "=" * 70)
    print("PAIRED COMPARISONS")
    print("=" * 70)
    cmp_1v2 = paired_report(arm2["label"], arm2, arm1["label"], arm1)
    cmp_1v3 = paired_report(arm3["label"], arm3, arm1["label"], arm1)

    # Diagnostic 1: action distribution (arm 1 only)
    p_mult = np.array([r["participation_mult"] for r in action_rows])
    urg = np.array([r["urgency"] for r in action_rows])
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 1: L2 action distribution (every decision, every episode)")
    print("=" * 70)
    print(f"n_decisions={len(action_rows)}")
    for name, arr, lo, hi, neutral in [
        ("participation_mult", p_mult, 0.0, 2.0, 1.0),
        ("urgency", urg, 0.0, 1.0, 0.5),
    ]:
        pct = np.percentile(arr, [1, 10, 25, 50, 75, 90, 99])
        at_lo = float(np.mean(np.isclose(arr, lo, atol=1e-3)))
        at_hi = float(np.mean(np.isclose(arr, hi, atol=1e-3)))
        print(
            f"{name}: mean={arr.mean():.4f} std={arr.std():.4f} "
            f"(neutral={neutral}, bounds=[{lo},{hi}])"
        )
        print(
            f"  percentiles [1,10,25,50,75,90,99] = "
            f"[{pct[0]:.4f}, {pct[1]:.4f}, {pct[2]:.4f}, {pct[3]:.4f}, {pct[4]:.4f}, {pct[5]:.4f}, {pct[6]:.4f}]"
        )
        print(f"  fraction at lower bound={at_lo:.4f}, fraction at upper bound={at_hi:.4f}")

    by_seed: dict[int, list[dict]] = {}
    for r in action_rows:
        by_seed.setdefault(r["seed"], []).append(r)
    within_ep_p_std = float(np.mean([np.std([x["participation_mult"] for x in v]) for v in by_seed.values()]))
    within_ep_u_std = float(np.mean([np.std([x["urgency"] for x in v]) for v in by_seed.values()]))
    between_ep_p_std = float(np.std([np.mean([x["participation_mult"] for x in v]) for v in by_seed.values()]))
    between_ep_u_std = float(np.std([np.mean([x["urgency"] for x in v]) for v in by_seed.values()]))
    print(f"\nWithin-episode std (responds to state mid-episode?):  participation_mult={within_ep_p_std:.4f}, urgency={within_ep_u_std:.4f}")
    print(f"Between-episode std of per-episode means (differs by episode/day at all?): participation_mult={between_ep_p_std:.4f}, urgency={between_ep_u_std:.4f}")
    print("(all four near 0 -> policy has collapsed to a near-constant action; non-trivial within-episode std -> actively steering)")

    if args.output_episodes_csv:
        with open(args.output_episodes_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["arm", "seed", "day", "is_total_bps", "fill_ratio"])
            w.writeheader()
            w.writerows(episode_rows)
        print(f"\nWrote {args.output_episodes_csv} ({len(episode_rows)} rows)")

    if args.output_actions_csv:
        with open(args.output_actions_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["seed", "tick_idx", "participation_mult", "urgency"])
            w.writeheader()
            w.writerows(action_rows)
        print(f"Wrote {args.output_actions_csv} ({len(action_rows)} rows)")

    result = {
        "split": args.split, "date_range": date_range, "n": args.n,
        "arms": {
            "l2": {"is_bps_mean": float(arm1["is_bps"].mean()), "is_bps_std": float(arm1["is_bps"].std()),
                   "fill_ratio_mean": float(arm1["fill_ratio"].mean())},
            "twap_passthrough": {"is_bps_mean": float(arm2["is_bps"].mean()), "is_bps_std": float(arm2["is_bps"].std()),
                                  "fill_ratio_mean": float(arm2["fill_ratio"].mean())},
            "pure_twap": {"is_bps_mean": float(arm3["is_bps"].mean()), "is_bps_std": float(arm3["is_bps"].std()),
                          "fill_ratio_mean": float(arm3["fill_ratio"].mean())},
        },
        "l2_vs_twap_passthrough": cmp_1v2,
        "l2_vs_pure_twap": cmp_1v3,
        "action_distribution": {
            "participation_mult_mean": float(p_mult.mean()), "participation_mult_std": float(p_mult.std()),
            "urgency_mean": float(urg.mean()), "urgency_std": float(urg.std()),
            "within_episode_participation_std_mean": within_ep_p_std,
            "within_episode_urgency_std_mean": within_ep_u_std,
            "between_episode_participation_std": between_ep_p_std,
            "between_episode_urgency_std": between_ep_u_std,
        },
    }
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
