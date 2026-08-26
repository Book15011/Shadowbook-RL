"""L2 test-split confirmation (2026-08-27) -- THE ONLY EVALUATION THE TEST
SPLIT EVER GETS FROM THIS PROJECT. One run, no re-runs, no parameter
adjustments after seeing the number. See docs/TRACK_STATUS.md's L2 section,
commit 2de9fab, for the pre-registered claim and interpretation of each
outcome -- written and committed BEFORE this script was even built, let
alone run, so its timestamp predates any test-split result.

Reuses scripts/eval_l2_n500.py's exact functions (imported, not
reimplemented) -- same methodology, same EVAL_SEED_BASE=5,000,000
paired-seed convention, same three arms. The only differences: points at
load_split("test") instead of "val", reports the day count alongside the
episode count (500 episodes from 18 days are not 500 independent samples --
same caveat as val), and adds an action-type distribution (L3's own
discrete order-type choice: HOLD/LIMIT/MARKET/CANCEL_AND_REPLACE, the first
component of the env's MultiDiscrete([4,11,5]) action space) via a safe,
inert monkeypatch on LOBExecutionEnv.step() -- captures action[0] before
delegating to the original, unchanged step(); cannot perturb anything since
step() receives the action as an argument, it does not draw it from RNG.

Mechanically smoke-tested on VAL (n=5, never test) before this file's own
real invocation against test -- testing the script is not the same as
spending the holdout; only the latter is restricted to once. See the smoke
test's own record in TRACK_STATUS.md/session log, not repeated here.

Run (the one real invocation):
  PYTHONPATH=. .venv/bin/python -m scripts.eval_l2_test_confirmation \\
    --l2-checkpoint models/l2_strategist_v1.zip \\
    --l2-vecnormalize models/l2_vecnormalize.pkl \\
    --l3-checkpoint models/l3_frozen_backup/l3_executioner_v1_frozen.zip \\
    --l3-vecnormalize models/l3_frozen_backup/l3_vecnormalize_frozen.pkl \\
    --n 500 --use-numeric-format --output-json models/l2_test_confirmation.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import src.envs.lob_execution_env as lob_env_mod
from sb3_contrib import RecurrentPPO
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

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

ORDER_TYPE_NAMES = {0: "HOLD", 1: "LIMIT", 2: "MARKET", 3: "CANCEL_AND_REPLACE"}


def install_action_type_capture() -> dict:
    """See module docstring. Returns a dict; call reset_bucket(state, label)
    before each arm to start a fresh tally, read state['counts'][label] after."""
    state: dict = {"active_label": None, "counts": {}}
    orig_step = lob_env_mod.LOBExecutionEnv.step

    def _capture_step(self, action):
        label = state["active_label"]
        if label is not None:
            order_type = int(np.asarray(action).reshape(-1)[0])
            state["counts"].setdefault(label, {}).setdefault(order_type, 0)
            state["counts"][label][order_type] += 1
        return orig_step(self, action)

    lob_env_mod.LOBExecutionEnv.step = _capture_step
    return state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="L2 TEST-SPLIT confirmation -- one-shot, see module docstring.")
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
    parser.add_argument(
        "--split", choices=["val", "test"], default="test",
        help="Default test (the real invocation). val exists ONLY for mechanics smoke-testing "
        "this script itself -- never run this script against test more than once.",
    )
    parser.add_argument("--output-json", type=str, default=None)
    return parser


def print_action_dist(label: str, counts: dict) -> dict:
    total = sum(counts.values())
    dist = {ORDER_TYPE_NAMES.get(k, k): v / total for k, v in sorted(counts.items())}
    print(f"  {label} action-type distribution (n_ticks={total}):")
    for name, frac in dist.items():
        print(f"    {name:20s} {frac:.4f}")
    return dist


def main() -> None:
    args = build_parser().parse_args()
    seeds = [EVAL_SEED_BASE + i for i in range(args.n)]
    max_decisions = HORIZON_TICKS // args.ticks_per_l2_decision + 1

    dates = load_split(args.split)
    date_range = (dates[0].isoformat(), dates[-1].isoformat())
    print(f"{'*' * 70}")
    print(f"SPLIT = {args.split.upper()}" + ("  <<< REAL TEST-SPLIT RUN >>>" if args.split == "test" else "  (smoke test only)"))
    print(f"{'*' * 70}")
    print(f"date_range: {date_range} ({len(dates)} real days)")
    print(f"n={args.n} paired seeds {seeds[0]}..{seeds[-1]}  ({args.n} episodes / {len(dates)} days = "
          f"{args.n / len(dates):.1f} episodes/day on average -- NOT {args.n} independent samples)")

    capture = install_action_type_capture()

    l3_model = RecurrentPPO.load(args.l3_checkpoint, device="cpu")
    wrapped_env = make_l2_wrapped_env(
        date_range, HORIZON_TICKS, LOOKBACK_TICKS, l3_model, args.l3_vecnormalize,
        args.ticks_per_l2_decision, args.l2_include_prev_action,
        data_dir=args.data_dir, l3_deterministic=True, use_numeric_format=args.use_numeric_format,
    )
    l2_model = SAC.load(args.l2_checkpoint, device=args.device)
    l2_vec_normalize = VecNormalize.load(args.l2_vecnormalize, DummyVecEnv([lambda: wrapped_env]))
    l2_vec_normalize.training = False

    print("\n=== Arm 1: trained L2 policy ===")
    capture["active_label"] = "l2"
    l2_action_fn = make_l2_policy_action_fn(l2_model, l2_vec_normalize)
    arm1 = run_arm("L2 (trained)", seeds, lambda s: run_wrapped_episode(wrapped_env, s, l2_action_fn, max_decisions))
    dist1 = print_action_dist("L2", capture["counts"]["l2"])

    print("\n=== Arm 2: TWAP-passthrough (frozen L3, unsteered) ===")
    capture["active_label"] = "twap_passthrough"
    arm2 = run_arm(
        "TWAP-passthrough", seeds,
        lambda s: run_wrapped_episode(wrapped_env, s, lambda obs: _TWAP_PASSTHROUGH_ACTION, max_decisions),
    )
    dist2 = print_action_dist("TWAP-passthrough", capture["counts"]["twap_passthrough"])

    print("\n=== Arm 3: pure TWAP (base env) ===")
    base_env = LOBExecutionEnv(
        data_dir=args.data_dir, date_range=date_range, horizon_ticks=HORIZON_TICKS,
        lookback_ticks=LOOKBACK_TICKS, use_numeric_format=args.use_numeric_format,
    )
    capture["active_label"] = "pure_twap"
    twap_policy = TWAPPolicy(n_slices=10)
    arm3 = run_arm(
        "Pure TWAP", seeds,
        lambda s: run_episode(base_env, twap_policy, seed=s, horizon_ticks=HORIZON_TICKS),
    )
    dist3 = print_action_dist("Pure TWAP", capture["counts"]["pure_twap"])
    capture["active_label"] = None

    print("\n" + "=" * 70)
    print("PAIRED COMPARISONS")
    print("=" * 70)
    cmp_1v2 = paired_report(arm2["label"], arm2, arm1["label"], arm1)
    cmp_1v3 = paired_report(arm3["label"], arm3, arm1["label"], arm1)

    print("\n" + "=" * 70)
    print("PRE-REGISTERED CLAIM (docs/TRACK_STATUS.md, commit 2de9fab, written before this ran):")
    print("  L2 does NOT achieve lower IS_total_bps than TWAP-passthrough,")
    print("  with BOTH t-test AND Wilcoxon agreeing at p<0.05.")
    print("=" * 70)
    l2_beats_passthrough = cmp_1v2["mean_diff"] < 0 and cmp_1v2["t_p"] < 0.05 and cmp_1v2["w_p"] < 0.05
    print(f"  L2 beats TWAP-passthrough (both tests, p<0.05): {'YES -- CLAIM REJECTED' if l2_beats_passthrough else 'NO -- CLAIM HOLDS'}")
    if l2_beats_passthrough:
        print("  Per the pre-registered interpretation: this is ONE anomalous result against a")
        print("  large, consistent body of contrary evidence (val n=500, unrestricted train n=500,")
        print("  calm/moderate/high volatility strata n=292/500/500 -- all null-to-negative).")
        print("  Read as an anomaly, NOT as 'L2 works after all'.")

    result = {
        "split": args.split, "date_range": date_range, "n_days": len(dates), "n": args.n,
        "episodes_per_day": args.n / len(dates),
        "arms": {
            "l2": {"is_bps_mean": float(arm1["is_bps"].mean()), "is_bps_std": float(arm1["is_bps"].std()),
                   "fill_ratio_mean": float(arm1["fill_ratio"].mean()), "action_type_dist": dist1},
            "twap_passthrough": {"is_bps_mean": float(arm2["is_bps"].mean()), "is_bps_std": float(arm2["is_bps"].std()),
                                  "fill_ratio_mean": float(arm2["fill_ratio"].mean()), "action_type_dist": dist2},
            "pure_twap": {"is_bps_mean": float(arm3["is_bps"].mean()), "is_bps_std": float(arm3["is_bps"].std()),
                          "fill_ratio_mean": float(arm3["fill_ratio"].mean()), "action_type_dist": dist3},
        },
        "l2_vs_twap_passthrough": cmp_1v2,
        "l2_vs_pure_twap": cmp_1v3,
        "l2_beats_passthrough_preregistered_bar": l2_beats_passthrough,
    }
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nWrote {args.output_json}")


if __name__ == "__main__":
    main()
