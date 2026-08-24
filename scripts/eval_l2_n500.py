"""L2 n=500 evaluation harness (2026-08-25, built while the real L2 training run is
live -- do not point this at real data until that run completes).

Reuses L3's own established n=500 methodology rather than designing fresh, so results
pool directly with the existing checkpoint table (docs/reports/l3_frozen_handoff.md):
same EVAL_SEED_BASE=5,000,000 paired-seed convention (scripts/replace_value_probe.py),
same load_split("val") population, same paired t-test + Wilcoxon signed-rank reporting
(scipy.stats.ttest_rel/wilcoxon). Arm 2's TWAP-passthrough baseline and its
_run_episode()-style loop are ported directly from train_l2.py's own
ValISEvalCallback -- the same baseline the in-training eval callback already tracks,
not a fresh design.

Three arms, paired across the SAME seed list:
  1. L2 -- the trained SAC policy (+ its own paired VecNormalize) steering the frozen
     L3 policy through FrozenL3Wrapper.
  2. TWAP-passthrough -- L2 always outputs [1.0, 0.5] (the env's own on-schedule
     defaults): frozen L3 unsteered. Answers "does learned steering beat no steering."
  3. Pure TWAP -- scripts/phase2a_sanity_suite.py's TWAPPolicy, unmodified, on the BASE
     LOBExecutionEnv (no L3/L2 involved at all) -- this is the exact policy/methodology
     that produced the existing table's TWAP row (0.889), so arm 3's own number here is
     directly poolable with it, not just comparable in spirit.

Pre-registered success bar (stated here, before any real result exists, so it cannot be
rationalized after the fact): arm 1 must beat arm 2 (the frozen-L3-alone analog of the
existing table's 0.994) with BOTH the paired t-test AND Wilcoxon agreeing (both
p<0.05, same direction) to count as a real win, not just a nominal one -- this project
has been burned before by a single significant test with a tiny, practically-irrelevant
effect (the budget-extension result, Cohen's d_z=0.076) and by underpowered n=50 reads
overturned at proper n (three separate instances, per l3_frozen_handoff.md). Effect
size (Cohen's d_z = mean_diff / std_diff, the standardized paired-differences size) is
reported alongside every p-value for exactly that reason -- ideally arm 1 also beats
TWAP itself (0.889), a higher bar than beating arm 2.

CLI takes explicit --l2-checkpoint/--l2-vecnormalize/--l3-checkpoint/--l3-vecnormalize
paths -- no defaults, same discipline as train_l2.py, so a real run can't launch by
omission.

Run (mechanics only, synthetic data -- see tests/test_eval_l2_n500.py for the actual
mechanics test): PYTHONPATH=. .venv/bin/python scripts/eval_l2_n500.py --help
Run (real, only after training completes):
  PYTHONPATH=. .venv/bin/python scripts/eval_l2_n500.py \\
    --l2-checkpoint models/l2_strategist_v1.zip \\
    --l2-vecnormalize models/l2_vecnormalize.pkl \\
    --l3-checkpoint models/l3_frozen_backup/l3_executioner_v1_frozen.zip \\
    --l3-vecnormalize models/l3_frozen_backup/l3_vecnormalize_frozen.pkl \\
    --n 500 --use-numeric-format
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Any, Callable

import numpy as np
from scipy import stats
from sb3_contrib import RecurrentPPO
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from scripts.phase2a_sanity_suite import TWAPPolicy, run_episode
from src.data.split import load_split
from src.envs.lob_execution_env import LOBExecutionEnv
from src.envs.wrappers import FrozenL3Wrapper
from src.train.train_l2 import make_l2_wrapped_env

EVAL_SEED_BASE = 5_000_000  # same base as scripts/replace_value_probe.py / train_l3.py's
                            # own ValISEvalCallback -- required for pooling with the
                            # existing table, not an arbitrary choice.
_TWAP_PASSTHROUGH_ACTION = np.array([1.0, 0.5], dtype=np.float32)
HORIZON_TICKS = 3000
LOOKBACK_TICKS = 10


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="L2 n=500 evaluation: trained policy vs. TWAP-passthrough vs. pure TWAP.",
    )
    parser.add_argument("--l2-checkpoint", type=str, required=True, help="Trained L2 SAC checkpoint .zip.")
    parser.add_argument(
        "--l2-vecnormalize", type=str, required=True,
        help="Paired VecNormalize .pkl for --l2-checkpoint (L2's own obs/reward normalization, "
        "e.g. models/l2_vecnormalize.pkl). Must be from the SAME training run -- nothing here "
        "verifies that pairing beyond the observation_space shape check VecNormalize.load() "
        "itself does, same discipline as --l3-vecnormalize below.",
    )
    parser.add_argument("--l3-checkpoint", type=str, required=True, help="Frozen L3 RecurrentPPO checkpoint .zip.")
    parser.add_argument(
        "--l3-vecnormalize", type=str, required=True,
        help="Paired VecNormalize .pkl for --l3-checkpoint.",
    )
    parser.add_argument("--n", type=int, default=500, help="Number of paired eval episodes per arm (default 500).")
    parser.add_argument(
        "--ticks-per-l2-decision", type=int, default=50,
        help="Must match the value --l2-checkpoint was actually trained with.",
    )
    parser.add_argument(
        "--l2-include-prev-action", action="store_true",
        help="Must match --l2-checkpoint's own training config (L2_FULL_OBS_DIM vs "
        "L2_BASE_OBS_DIM) -- default False matches train_l2.py's own default.",
    )
    parser.add_argument("--data-dir", type=str, default="data/raw_l2_bybit/BTCUSDT")
    parser.add_argument(
        "--use-numeric-format", action="store_true",
        help="Read the converted numeric (*.npzst) archive instead of the original "
        "(*.parquet) one -- see src/data/l2_numeric_format.py. Default False (original "
        "format), matching LOBExecutionEnv's own default.",
    )
    parser.add_argument("--device", default="cpu", choices=["cuda", "cpu"])
    parser.add_argument(
        "--output-json", type=str, default=None,
        help="Optional path to write the full result dict as JSON, in addition to stdout.",
    )
    return parser


def run_wrapped_episode(
    env: FrozenL3Wrapper, seed: int, action_fn: Callable[[np.ndarray], np.ndarray], max_decisions: int,
) -> dict[str, Any]:
    """Ported from train_l2.py's ValISEvalCallback._run_episode -- same loop, same
    info["implementation_shortfall"] extraction (passes through FrozenL3Wrapper.step()
    unchanged, see that wrapper's own module docstring)."""
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    for _ in range(max_decisions):
        action = action_fn(obs)
        obs, r, term, trunc, info = env.step(action)
        total_reward += r
        if term or trunc:
            break
    return {"total_reward": total_reward, "is_result": info["implementation_shortfall"]}


def make_l2_policy_action_fn(model: SAC, vec_normalize: VecNormalize | None) -> Callable[[np.ndarray], np.ndarray]:
    def _action_fn(obs: np.ndarray) -> np.ndarray:
        obs_for_policy = obs[None, :]
        if vec_normalize is not None:
            obs_for_policy = vec_normalize.normalize_obs(obs_for_policy)
        action, _ = model.predict(obs_for_policy, deterministic=True)
        return action[0]

    return _action_fn


def run_arm(
    label: str, seeds: list[int], episode_fn: Callable[[int], dict[str, Any]],
) -> dict[str, np.ndarray]:
    t0 = time.time()
    is_bps, fill_ratios = [], []
    for seed in seeds:
        result = episode_fn(seed)
        is_bps.append(result["is_result"].is_total_bps)
        fill_ratios.append(result["is_result"].fill_ratio)
    is_bps = np.array(is_bps)
    fill_ratios = np.array(fill_ratios)
    dt = time.time() - t0
    print(
        f"  {label:24s} IS_total_bps mean={is_bps.mean():8.4f} std={is_bps.std():7.4f} "
        f"fill_ratio={fill_ratios.mean():.4f}  ({dt:.1f}s, n={len(seeds)})"
    )
    return {"label": label, "is_bps": is_bps, "fill_ratio": fill_ratios, "seconds": dt}


def paired_report(name_a: str, r_a: dict, name_b: str, r_b: dict) -> dict:
    """Ported from scripts/replace_value_probe.py's paired_report, extended with
    Cohen's d_z (paired-samples effect size, mean_diff/std_diff) -- this project has
    repeatedly been misled by significance without magnitude (the budget-extension
    result: nominally significant vs. TWAP alone, p=0.034/0.044, but d_z=0.076 against
    the checkpoint it was extending, i.e. practically zero) and by underpowered n=50
    reads overturned at proper n, so p-values are never reported alone here."""
    diff = r_b["is_bps"] - r_a["is_bps"]
    t_stat, t_p = stats.ttest_rel(r_b["is_bps"], r_a["is_bps"])
    w_stat, w_p = stats.wilcoxon(r_b["is_bps"], r_a["is_bps"])
    d_z = float(diff.mean() / diff.std()) if diff.std() > 0 else float("nan")
    print(
        f"\n{name_b} vs {name_a}: mean diff ({name_b}-{name_a})={diff.mean():.4f}bps "
        f"std={diff.std():.4f}  Cohen's d_z={d_z:.4f}"
    )
    print(f"  paired t-test:        t={t_stat:.4f}  p={t_p:.4f}")
    print(f"  Wilcoxon signed-rank: W={w_stat:.4f}  p={w_p:.4f}")
    return {
        "a": name_a, "b": name_b, "mean_diff": float(diff.mean()), "std_diff": float(diff.std()),
        "d_z": d_z, "t_stat": float(t_stat), "t_p": float(t_p), "w_stat": float(w_stat), "w_p": float(w_p),
    }


def main() -> None:
    args = build_parser().parse_args()
    seeds = [EVAL_SEED_BASE + i for i in range(args.n)]
    max_decisions = HORIZON_TICKS // args.ticks_per_l2_decision + 1

    val_dates = load_split("val")
    val_date_range = (val_dates[0].isoformat(), val_dates[-1].isoformat())
    print(f"val date_range: {val_date_range} ({len(val_dates)} real days)")
    print(f"n={args.n} paired seeds {seeds[0]}..{seeds[-1]}")
    print(f"l2-checkpoint: {args.l2_checkpoint}")
    print(f"l2-vecnormalize: {args.l2_vecnormalize}")
    print(f"l3-checkpoint: {args.l3_checkpoint}")
    print(f"l3-vecnormalize: {args.l3_vecnormalize}\n")

    l3_model = RecurrentPPO.load(args.l3_checkpoint, device="cpu")
    wrapped_env = make_l2_wrapped_env(
        val_date_range, HORIZON_TICKS, LOOKBACK_TICKS, l3_model, args.l3_vecnormalize,
        args.ticks_per_l2_decision, args.l2_include_prev_action,
        data_dir=args.data_dir, l3_deterministic=True, use_numeric_format=args.use_numeric_format,
    )

    l2_model = SAC.load(args.l2_checkpoint, device=args.device)
    l2_vec_normalize = VecNormalize.load(args.l2_vecnormalize, DummyVecEnv([lambda: wrapped_env]))
    l2_vec_normalize.training = False

    print("=== Arm 1: trained L2 policy ===")
    l2_action_fn = make_l2_policy_action_fn(l2_model, l2_vec_normalize)
    arm1 = run_arm("L2 (trained)", seeds, lambda s: run_wrapped_episode(wrapped_env, s, l2_action_fn, max_decisions))

    print("\n=== Arm 2: TWAP-passthrough (frozen L3, unsteered) ===")
    arm2 = run_arm(
        "TWAP-passthrough", seeds,
        lambda s: run_wrapped_episode(wrapped_env, s, lambda obs: _TWAP_PASSTHROUGH_ACTION, max_decisions),
    )

    print("\n=== Arm 3: pure TWAP (base env, poolable with the existing table) ===")
    base_env = LOBExecutionEnv(
        data_dir=args.data_dir, date_range=val_date_range, horizon_ticks=HORIZON_TICKS,
        lookback_ticks=LOOKBACK_TICKS, use_numeric_format=args.use_numeric_format,
    )
    twap_policy = TWAPPolicy(n_slices=10)
    arm3 = run_arm(
        "Pure TWAP", seeds,
        lambda s: run_episode(base_env, twap_policy, seed=s, horizon_ticks=HORIZON_TICKS),
    )

    print("\n" + "=" * 70)
    print("PAIRED COMPARISONS")
    print("=" * 70)
    cmp_1v2 = paired_report(arm2["label"], arm2, arm1["label"], arm1)
    cmp_1v3 = paired_report(arm3["label"], arm3, arm1["label"], arm1)

    print("\n" + "=" * 70)
    print("PRE-REGISTERED SUCCESS BAR (stated before results, not after):")
    print("  L2 must beat TWAP-passthrough (arm 2) with BOTH t-test AND Wilcoxon")
    print("  agreeing (both p<0.05, same direction) -- ideally also beat pure TWAP.")
    print("=" * 70)
    beats_arm2 = cmp_1v2["mean_diff"] < 0 and cmp_1v2["t_p"] < 0.05 and cmp_1v2["w_p"] < 0.05
    beats_twap = cmp_1v3["mean_diff"] < 0 and cmp_1v3["t_p"] < 0.05 and cmp_1v3["w_p"] < 0.05
    print(f"  Beats TWAP-passthrough (required): {'YES' if beats_arm2 else 'NO'}")
    print(f"  Beats pure TWAP (stretch goal):     {'YES' if beats_twap else 'NO'}")

    result = {
        "n": args.n, "seeds": [seeds[0], seeds[-1]],
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
        "beats_twap_passthrough": beats_arm2,
        "beats_pure_twap": beats_twap,
    }
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nWrote {args.output_json}")


if __name__ == "__main__":
    main()
