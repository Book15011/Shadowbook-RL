"""L2 (Strategist) SAC training -- architecture_spec.md Section 4.1, wired against the
real FrozenL3Wrapper (src/envs/wrappers.py) and LOBExecutionEnv, not the Section 4.1
reference train_l2.py snippet directly -- see docs/reports/phase4_l2_reconciliation_and_plan.md
Part A for why (tier=/l2_action_space/apply_l2_action/etc. don't exist on the real,
single-tier env; the wrapper is the entire L2/L3 integration layer, not a thin adapter).

Which SAC hyperparameters below are independently derived/re-confirmed for L2's real
cadence vs. carried over un-re-derived from Section 4.1's reference snippet -- see each
kwarg's own comment below. Only buffer_size and gamma were ever independently checked
(docs/reports/phase4_l2_reconciliation_and_plan.md FINAL SPEC Step 3, against the real
60-decisions/episode cadence and the real 405-day train split). batch_size/tau/
learning_rate/train_freq/gradient_steps are Section 4.1's literal reference values, used
as-is because they were never flagged as needing independent derivation the way
buffer_size/gamma were -- this is a deliberate distinction, not an oversight, so this
script's comments say explicitly which is which rather than reading as if everything here
was checked.

L2 has no reward function of its own. FrozenL3Wrapper.step() aggregates L3's existing
per-tick reward.step_reward() output as-is (see wrappers.py's agg_reward accumulation) --
no new L2-specific term is added anywhere. There is therefore no L2-analogous surface to
train_l3.py's --reward-zeta/--reward-eta-replace override flags: those tune the SAME
underlying reward function L2's aggregated signal already inherits, and this script
deliberately does not override LOBExecutionEnv's reward_weights (it is left at
RewardWeights()'s own default). That default is also what the current frozen L3 checkpoint
was actually trained under (docs/TRACK_STATUS.md's L3 entry: "RewardWeights() real
defaults") -- so L2 trains against exactly the reward shape its frozen L3 policy already
learned to act under, by construction, not by needing an override flag to match it.

Also NOT added this round, matching "wiring only" scope: no VecNormalize around the L2-level
SAC env itself (Section 4.1's reference train_l2.py snippet doesn't show one either, unlike
its L3 counterpart, which explicitly does) and no held-out validation/eval callback
analogous to train_l3.py's ValISEvalCallback (computing L2-level IS_bps vs. a TWAP baseline
is real, separate work, not part of this round's task). A plain CheckpointCallback is
included since periodic saving is basic training hygiene, not a design decision.

Run (wiring/smoke-test only this round -- no real training launch):
PYTHONPATH=. .venv/bin/python -m src.train.train_l2 \\
    --l3-checkpoint models/l3_executioner_v1.zip \\
    --l3-vecnormalize models/l3_vecnormalize.pkl \\
    --total-timesteps 200 \\
    --smoke-test
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from src.data.split import load_split
from src.envs.lob_execution_env import LOBExecutionEnv
from src.envs.wrappers import FrozenL3Wrapper

_SMOKE_TEST_MAX_TIMESTEPS = 10_000  # guards against --smoke-test + a real-sized budget by accident


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_l2_env(
    train_date_range: tuple[str, str],
    horizon_ticks: int,
    lookback_ticks: int,
    l3_model: RecurrentPPO,
    l3_vecnormalize_path: str,
    ticks_per_l2_decision: int,
    l2_include_prev_action: bool,
):
    env = LOBExecutionEnv(
        date_range=train_date_range, horizon_ticks=horizon_ticks, lookback_ticks=lookback_ticks,
    )
    wrapped = FrozenL3Wrapper(
        env, l3_model, l3_vecnormalize_path,
        ticks_per_l2_decision=ticks_per_l2_decision,
        l2_include_prev_action=l2_include_prev_action,
    )
    return Monitor(wrapped)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--l3-checkpoint", type=str, required=True,
        help="Frozen L3 RecurrentPPO policy .zip, e.g. models/l3_executioner_v1.zip. No "
        "default: the L3 track's checkpoint-quality question is still open (see "
        "docs/TRACK_STATUS.md), so every invocation must name it explicitly rather than "
        "silently inheriting whatever happens to be sitting at a default path. This "
        "script does not decide which checkpoint is 'the' frozen one -- that call "
        "belongs to the L3 track, not here.",
    )
    parser.add_argument(
        "--l3-vecnormalize", type=str, required=True,
        help="Paired VecNormalize .pkl for --l3-checkpoint, e.g. models/l3_vecnormalize.pkl. "
        "Must be from the SAME training run as --l3-checkpoint -- nothing here verifies "
        "that pairing beyond the observation_space shape check VecNormalize.load() itself "
        "performs (see wrappers.py).",
    )
    parser.add_argument(
        "--total-timesteps", type=int, required=True,
        help="SAC training budget. No default -- Section 4.1's real target is 2,000,000; "
        "pass a small value (e.g. a couple hundred) explicitly together with --smoke-test "
        "for a mechanics-only run. Required so nobody launches the real budget by omission.",
    )
    parser.add_argument(
        "--ticks-per-l2-decision", type=int, default=50,
        help="L2 decision cadence in L3 ticks. architecture_spec.md Section 4.1 and "
        "Section 4.3 are both settled at 50 -- see docs/reports/"
        "phase4_l2_reconciliation_and_plan.md FINAL SPEC Step 1 (moderate, not high, "
        "confidence -- the Section 4.3 fix was an unexplained one-line edit).",
    )
    parser.add_argument(
        "--horizon-ticks", type=int, default=3000,
        help="Matches configs/ppo_l3.yaml's real production value -- the underlying "
        "LOBExecutionEnv config the frozen L3 checkpoint was actually trained against.",
    )
    parser.add_argument("--lookback-ticks", type=int, default=10)
    parser.add_argument(
        "--l2-include-prev-action", action="store_true", default=False,
        help="Adds L2's own raw previous action (2 dims) to its observation. Defaults "
        "OFF -- see docs/reports/phase4_l2_reconciliation_and_plan.md's correction: the "
        "recurrent-policy precedent originally used to justify defaulting this ON doesn't "
        "transfer to SAC's plain MlpPolicy. Kept as a toggle for a future ablation, not "
        "presented as precedent-backed.",
    )
    parser.add_argument(
        "--smoke-test", action="store_true", default=False,
        help="Labels this run as a mechanics-only smoke test everywhere in output/logs/ "
        "save paths. Does not change training mechanics -- only naming/logging and a "
        "sanity cap on --total-timesteps (see _SMOKE_TEST_MAX_TIMESTEPS). A smoke test "
        "verifies the wrapper/SAC integration runs end-to-end without shape/interface "
        "errors; its reward/loss numbers are not a performance signal and must not be "
        "extrapolated from.",
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--progress-bar", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.smoke_test and args.total_timesteps > _SMOKE_TEST_MAX_TIMESTEPS:
        raise ValueError(
            f"--smoke-test with --total-timesteps={args.total_timesteps} exceeds the "
            f"smoke-test sanity cap ({_SMOKE_TEST_MAX_TIMESTEPS}) -- either lower "
            "--total-timesteps or drop --smoke-test if a real run is genuinely intended "
            "(a real run also needs its own separate go-ahead, not just this flag)."
        )

    run_label = "SMOKE TEST" if args.smoke_test else "TRAINING RUN"
    print(f"=== L2 SAC {run_label} ===")
    if args.smoke_test:
        print(
            "[SMOKE TEST] mechanics-only run -- NOT a training run. Verifies the "
            "FrozenL3Wrapper/SAC integration runs end-to-end without shape/interface "
            "errors. Numbers below are not a performance signal; do not extrapolate."
        )

    cuda_available = torch.cuda.is_available()
    print(f"cuda available: {cuda_available}")
    if args.device == "cuda" and not cuda_available:
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False")

    for path, label in ((args.l3_checkpoint, "l3-checkpoint"), (args.l3_vecnormalize, "l3-vecnormalize")):
        if not Path(path).exists():
            raise FileNotFoundError(f"--{label} not found: {path}")
    print(f"l3-checkpoint: {args.l3_checkpoint} (sha256 {_sha256(args.l3_checkpoint)})")
    print(f"l3-vecnormalize: {args.l3_vecnormalize}")
    print(
        "NOTE: this script does not decide which L3 checkpoint is 'the' frozen one -- "
        "that is the L3 track's own open call (docs/TRACK_STATUS.md), not resolved here. "
        "Whatever is passed via --l3-checkpoint is used as-is, unverified beyond existing "
        "on disk and loading successfully."
    )

    train_dates = load_split("train")
    train_date_range = (train_dates[0].isoformat(), train_dates[-1].isoformat())
    print(f"train date_range: {train_date_range} ({len(train_dates)} real days)")

    l3_model = RecurrentPPO.load(args.l3_checkpoint, device=args.device)

    env = make_l2_env(
        train_date_range, args.horizon_ticks, args.lookback_ticks,
        l3_model, args.l3_vecnormalize, args.ticks_per_l2_decision,
        args.l2_include_prev_action,
    )

    model = SAC(
        "MlpPolicy", env,
        # -- Derived + re-confirmed for L2's real cadence (docs/reports/
        # phase4_l2_reconciliation_and_plan.md FINAL SPEC Step 3) --
        buffer_size=500_000,  # ~8,333 L2-episode-equivalents of coverage, ~25% of the full 2M-step run's transition volume.
        gamma=0.995,  # re-derived on L2's OWN cadence (not L3's tick-level reasoning): effective horizon ~3.3x the 60-decision episode length, defensible given the terminal-IS-dominated reward structure.
        # -- Section 4.1 reference values, carried over as-is -- NOT independently
        # derived for L2 the way the two above were; use as-is per instruction absent a
        # concrete reason not to. --
        batch_size=256,
        tau=0.005,
        learning_rate=3e-4,
        train_freq=1,
        gradient_steps=1,
        tensorboard_log=("logs/l2_sac_smoke/" if args.smoke_test else "logs/l2_sac/"),
        device=args.device,
        verbose=1,
    )
    print(f"model device actually in use: {model.device}")

    ckpt_dir = "models/l2_checkpoints_smoke/" if args.smoke_test else "models/l2_checkpoints/"
    ckpt_prefix = "l2_sac_smoke" if args.smoke_test else "l2_sac"
    checkpoint_cb = CheckpointCallback(
        save_freq=max(1, args.total_timesteps // 4) if args.smoke_test else 50_000,
        save_path=ckpt_dir,
        name_prefix=ckpt_prefix,
    )

    model.learn(total_timesteps=args.total_timesteps, callback=checkpoint_cb, progress_bar=args.progress_bar)

    Path("models").mkdir(exist_ok=True)
    save_name = "models/l2_strategist_smoke_test" if args.smoke_test else "models/l2_strategist_v1"
    model.save(save_name)
    print(f"Saved model to {save_name}.zip")
    if args.smoke_test:
        print("[SMOKE TEST] complete -- mechanics-only, not a performance signal. No real training was launched.")


if __name__ == "__main__":
    main()
