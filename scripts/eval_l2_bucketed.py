"""Volatility-stratified evaluation on TRAIN days (2026-08-27, follow-up to
the regime-matching finding: train's aggregate L2-vs-passthrough advantage
collapsed from -0.253 to -0.013bps when restricted to val's own volatility
range). This script asks the natural next question: does the advantage
correspondingly STRENGTHEN above val's range, where val has no equivalent
at all?

CONFOUNDED WITH MEMORIZATION -- READ BEFORE TRUSTING ANY RESULT HERE. These
are TRAINING days the policy was optimized against for 2,000,000 steps. Any
edge found in a high-volatility bucket cannot be distinguished from L2
having memorized these specific volatile days, as opposed to having learned
something that genuinely transfers to volatile conditions in general. This
script cannot separate those two explanations -- it can only tell you
whether there IS an edge on these particular seen days, not whether it
would hold on unseen volatile days. Do not treat a favorable result here as
held-out evidence.

Reuses eval_l2_n500.py's exact functions (imported, not reimplemented).
Restricts the env's file pool to an explicit, non-contiguous list of dates
(a volatility bucket) via a safe post-construction override of
LOBExecutionEnv._files -- the SAME instance attribute the class's own
reset() reads len()/indexes from every call, set here BEFORE any reset() is
called, so it cannot perturb the RNG draw order (file_idx is drawn fresh
against whatever _files currently is; overriding it before first use is
equivalent to having constructed the env with that exact file set from the
start, no different than filtering by date_range would have done for a
contiguous range).

Run: PYTHONPATH=. .venv/bin/python -m scripts.eval_l2_bucketed \\
  --l2-checkpoint models/l2_strategist_v1.zip \\
  --l2-vecnormalize models/l2_vecnormalize.pkl \\
  --l3-checkpoint models/l3_frozen_backup/l3_executioner_v1_frozen.zip \\
  --l3-vecnormalize models/l3_frozen_backup/l3_vecnormalize_frozen.pkl \\
  --n 500 --use-numeric-format --bucket moderate \\
  --output-json models/l2_bucketed_moderate.json
"""
from __future__ import annotations

import os

# Thread-capping is MANDATORY, not an optimization -- see eval_l2_n500.py's own comment
# at this same location for the measured 1,353%-CPU/33-minute/zero-output incident this
# fixes. Must be set before torch is imported anywhere (transitively, below), so this
# sits above every other import, same placement as train_l2.py's own fix.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import json

import pandas as pd
import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

torch.set_num_threads(1)  # defense-in-depth -- see comment above the env vars

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

VAL_MAX_VOL_BPS = 0.1882  # val's own max realized_vol_bps -- see models/l2_day_conditions_val.csv

BUCKET_DEFS = {
    "calm": lambda v: v <= VAL_MAX_VOL_BPS,
    "moderate": lambda v: (v > VAL_MAX_VOL_BPS) & (v <= 0.30),
    "high": lambda v: v > 0.30,
}


def bucket_dates(bucket: str) -> list[str]:
    cond = pd.read_csv("models/l2_day_conditions_train.csv")
    mask = BUCKET_DEFS[bucket](cond["realized_vol_bps"])
    return sorted(cond.loc[mask, "day"].tolist())


def restrict_to_dates(env: LOBExecutionEnv, dates: set[str]) -> None:
    env._files = [p for p in env._files if p.stem.replace("l2-BTCUSDT-", "") in dates]
    if not env._files:
        raise ValueError("No files matched the requested date bucket.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Volatility-stratified L2 evaluation on TRAIN days.")
    parser.add_argument("--l2-checkpoint", type=str, required=True)
    parser.add_argument("--l2-vecnormalize", type=str, required=True)
    parser.add_argument("--l3-checkpoint", type=str, required=True)
    parser.add_argument("--l3-vecnormalize", type=str, required=True)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--ticks-per-l2-decision", type=int, default=50)
    parser.add_argument("--l2-include-prev-action", action="store_true")
    parser.add_argument("--data-dir", type=str, default="data/raw_l2_bybit_numeric/BTCUSDT")
    parser.add_argument("--use-numeric-format", action="store_true")
    parser.add_argument("--device", default="cpu", choices=["cuda", "cpu"])
    parser.add_argument("--bucket", choices=list(BUCKET_DEFS), required=True)
    parser.add_argument("--output-json", type=str, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    seeds = [EVAL_SEED_BASE + i for i in range(args.n)]
    max_decisions = HORIZON_TICKS // args.ticks_per_l2_decision + 1

    train_dates = load_split("train")
    full_train_range = (train_dates[0].isoformat(), train_dates[-1].isoformat())
    dates = set(bucket_dates(args.bucket))
    print(f"bucket={args.bucket}  n_days={len(dates)}  dates={sorted(dates)[:3]}...{sorted(dates)[-3:]}")
    print(f"n={args.n} paired seeds {seeds[0]}..{seeds[-1]}")

    l3_model = RecurrentPPO.load(args.l3_checkpoint, device="cpu")
    wrapped_env = make_l2_wrapped_env(
        full_train_range, HORIZON_TICKS, LOOKBACK_TICKS, l3_model, args.l3_vecnormalize,
        args.ticks_per_l2_decision, args.l2_include_prev_action,
        data_dir=args.data_dir, l3_deterministic=True, use_numeric_format=args.use_numeric_format,
    )
    restrict_to_dates(wrapped_env.unwrapped, dates)
    print(f"wrapped_env restricted to {len(wrapped_env.unwrapped._files)} files")

    l2_model = SAC.load(args.l2_checkpoint, device=args.device)
    l2_vec_normalize = VecNormalize.load(args.l2_vecnormalize, DummyVecEnv([lambda: wrapped_env]))
    l2_vec_normalize.training = False

    print(f"=== Arm 1: trained L2 policy (bucket={args.bucket}) ===")
    l2_action_fn = make_l2_policy_action_fn(l2_model, l2_vec_normalize)
    arm1 = run_arm("L2 (trained)", seeds, lambda s: run_wrapped_episode(wrapped_env, s, l2_action_fn, max_decisions))

    print(f"\n=== Arm 2: TWAP-passthrough (bucket={args.bucket}) ===")
    arm2 = run_arm(
        "TWAP-passthrough", seeds,
        lambda s: run_wrapped_episode(wrapped_env, s, lambda obs: _TWAP_PASSTHROUGH_ACTION, max_decisions),
    )

    print(f"\n=== Arm 3: pure TWAP (bucket={args.bucket}) ===")
    base_env = LOBExecutionEnv(
        data_dir=args.data_dir, date_range=full_train_range, horizon_ticks=HORIZON_TICKS,
        lookback_ticks=LOOKBACK_TICKS, use_numeric_format=args.use_numeric_format,
    )
    restrict_to_dates(base_env, dates)
    twap_policy = TWAPPolicy(n_slices=10)
    arm3 = run_arm(
        "Pure TWAP", seeds,
        lambda s: run_episode(base_env, twap_policy, seed=s, horizon_ticks=HORIZON_TICKS),
    )

    print("\n" + "=" * 70)
    print(f"PAIRED COMPARISONS (bucket={args.bucket}, n_days={len(dates)})")
    print("=" * 70)
    cmp_1v2 = paired_report(arm2["label"], arm2, arm1["label"], arm1)
    cmp_1v3 = paired_report(arm3["label"], arm3, arm1["label"], arm1)

    result = {
        "bucket": args.bucket, "n_days": len(dates), "dates": sorted(dates), "n": args.n,
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
    }
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nWrote {args.output_json}")


if __name__ == "__main__":
    main()
