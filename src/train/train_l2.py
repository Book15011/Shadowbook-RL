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

Still NOT added, matching "wiring only" scope from the round this script was first built:
no VecNormalize around the L2-level SAC env itself (Section 4.1's reference train_l2.py
snippet doesn't show one either, unlike its L3 counterpart) -- decided (recommend adding
before a real run, does not block further work) but not implemented, see docs/reports/
phase4_l2_reconciliation_and_plan.md's CURRENT STATE section.

Held-out eval (ValISEvalCallback below) IS now built -- modeled directly on train_l3.py's
own ValISEvalCallback (same held-out-val-split, paired-seed, real-IS-metric conventions),
adapted for L2's own cost profile: eval_freq/n_eval_episodes default to values sized
against this project's own measured L2 throughput (docs/reports/
phase4_l2_reconciliation_and_plan.md's Task 1 throughput measurement -- ~4.15
decisions/sec, ~half of which is env.reset() overhead), not copied from L3's own n=50
(sized for statistical significance testing, a different goal than "catch an obviously
broken run early," which is what this needs to do cheaply). Building this callback's own
determinism test surfaced a real bug in wrappers.py (FrozenL3Wrapper's inner L3 predict()
calls were hardcoded non-deterministic regardless of caller intent -- see that module's
own docstring, correction 3, for the fix): the eval env below is constructed with
l3_deterministic=True, the training env is not.

Run (wiring/smoke-test only so far -- no real training launch):
PYTHONPATH=. .venv/bin/python -m src.train.train_l2 \\
    --l3-checkpoint models/l3_executioner_v1.zip \\
    --l3-vecnormalize models/l3_vecnormalize.pkl \\
    --total-timesteps 200 \\
    --smoke-test --no-eval
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
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


def make_l2_wrapped_env(
    date_range: tuple[str, str],
    horizon_ticks: int,
    lookback_ticks: int,
    l3_model: RecurrentPPO,
    l3_vecnormalize_path: str,
    ticks_per_l2_decision: int,
    l2_include_prev_action: bool,
    data_dir: str = "data/raw_l2_bybit/BTCUSDT",
    l3_deterministic: bool = False,
) -> FrozenL3Wrapper:
    # data_dir defaults to LOBExecutionEnv's own default -- exposed as a parameter (not
    # previously) so tests can point this at a small synthetic data_dir instead of the
    # real archive; main() below never passes it, so real invocations are unaffected.
    # l3_deterministic default (False) matches training-time exploration -- ValISEvalCallback
    # below overrides this to True for its own eval env (see wrappers.py's module
    # docstring, correction 3: eval needs the frozen L3 policy itself to be reproducible,
    # not just L2's own action selection).
    env = LOBExecutionEnv(
        data_dir=data_dir, date_range=date_range, horizon_ticks=horizon_ticks, lookback_ticks=lookback_ticks,
    )
    return FrozenL3Wrapper(
        env, l3_model, l3_vecnormalize_path,
        ticks_per_l2_decision=ticks_per_l2_decision,
        l2_include_prev_action=l2_include_prev_action,
        l3_deterministic=l3_deterministic,
    )


def make_l2_env(*args, **kwargs) -> Monitor:
    """Training env: same construction as make_l2_wrapped_env, Monitor-wrapped for SAC's
    own rollout/episode-stat tracking. The eval callback below builds its own separate
    instance directly via make_l2_wrapped_env (no Monitor -- it drives episodes manually,
    same convention as train_l3.py's ValISEvalCallback)."""
    return Monitor(make_l2_wrapped_env(*args, **kwargs))


class ValISEvalCallback(BaseCallback):
    """L2-analog of train_l3.py's ValISEvalCallback -- same held-out-val-split,
    paired-seed, real-IS-metric-via-compute_implementation_shortfall() conventions
    (through info["implementation_shortfall"], unchanged as it passes through
    FrozenL3Wrapper.step()). Not in Section 4.1's reference code, same as L3's version.

    Baseline: "TWAP passthrough" -- L2 always outputs participation_rate_multiplier=1.0,
    urgency=0.5, i.e. the env's own default linear-TWAP schedule with zero L2 steering.
    The L2-analog of L3's TWAPPolicy baseline: does the LEARNED L2 policy beat doing
    nothing (letting the frozen L3 policy execute on-schedule, unsteered)? Computed once
    at training start and cached, same rationale as L3's version (a fixed, non-learning
    baseline has no reason to be re-evaluated every firing), from the same paired seed
    list every firing reuses.

    Unlike for L3, eval cost is a first-order design constraint here, not an afterthought
    -- see docs/reports/phase4_l2_reconciliation_and_plan.md's Task 1 throughput
    measurement (this round): each L2 decision costs up to ticks_per_l2_decision real
    L3.predict()+env.step() calls, and roughly HALF of L2's measured training wall-clock
    is env.reset() overhead (L2 episodes end well short of the full 60-decision horizon,
    so reset() is paid disproportionately often per unit of training compared to L3's own
    tick-level training). eval_freq/n_eval_episodes defaults (see train_l2.py's CLI help
    for the arithmetic) are sized against the measured ~4.15 decisions/sec rate, not
    copied from L3's own n=50 (itself sized for statistical significance testing -- a
    different goal than "catch an obviously broken run early" cheaply, which is what
    this needs to do).
    """

    EVAL_SEED_BASE = 5_000_000
    # On-schedule, neutral urgency -- zero L2 steering, matches env.l2_urgency's own
    # neutral default (0.5).
    _TWAP_PASSTHROUGH_ACTION = np.array([1.0, 0.5], dtype=np.float32)

    def __init__(
        self,
        val_date_range: tuple[str, str],
        horizon_ticks: int,
        lookback_ticks: int,
        ticks_per_l2_decision: int,
        l3_model: RecurrentPPO,
        l3_vecnormalize_path: str,
        l2_include_prev_action: bool,
        eval_freq: int,
        n_eval_episodes: int,
        verbose: int = 0,
        data_dir: str = "data/raw_l2_bybit/BTCUSDT",
    ) -> None:
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self._max_decisions = horizon_ticks // ticks_per_l2_decision + 1
        self._eval_env = make_l2_wrapped_env(
            val_date_range, horizon_ticks, lookback_ticks,
            l3_model, l3_vecnormalize_path, ticks_per_l2_decision, l2_include_prev_action,
            data_dir=data_dir, l3_deterministic=True,
        )
        # Fixed once, reused for every arm and every firing -- same paired-design
        # convention as train_l3.py's ValISEvalCallback.
        self._eval_seeds = [self.EVAL_SEED_BASE + i for i in range(n_eval_episodes)]
        self._twap_passthrough_is_bps: np.ndarray | None = None
        self._twap_passthrough_fill: np.ndarray | None = None
        self._last_eval_step = 0

    def _run_episode(self, seed: int, action_fn: Callable[[np.ndarray], np.ndarray]) -> dict[str, Any]:
        obs, info = self._eval_env.reset(seed=seed)
        total_reward = 0.0
        for _ in range(self._max_decisions):
            action = action_fn(obs)
            obs, r, term, trunc, info = self._eval_env.step(action)
            total_reward += r
            if term or trunc:
                break
        return {"total_reward": total_reward, "is_result": info["implementation_shortfall"]}

    def _l2_policy_action(self, obs: np.ndarray) -> np.ndarray:
        vec_normalize = self.model.get_vec_normalize_env()
        obs_for_policy = obs[None, :]
        if vec_normalize is not None:
            obs_for_policy = vec_normalize.normalize_obs(obs_for_policy)
        action, _ = self.model.predict(obs_for_policy, deterministic=True)
        return action[0]

    def _on_training_start(self) -> None:
        results = [
            self._run_episode(seed=s, action_fn=lambda obs: self._TWAP_PASSTHROUGH_ACTION)
            for s in self._eval_seeds
        ]
        self._twap_passthrough_is_bps = np.array([r["is_result"].is_total_bps for r in results])
        self._twap_passthrough_fill = np.array([r["is_result"].fill_ratio for r in results])
        if self.verbose:
            print(
                f"[ValISEvalCallback] TWAP-passthrough baseline on val "
                f"({self.n_eval_episodes} episodes, paired seeds {self._eval_seeds[0]}.."
                f"{self._eval_seeds[-1]}): IS_total_bps mean="
                f"{self._twap_passthrough_is_bps.mean():.4f} fill_ratio mean="
                f"{self._twap_passthrough_fill.mean():.4f}"
            )

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_step < self.eval_freq:
            return True
        self._last_eval_step = self.num_timesteps

        results = [self._run_episode(seed=s, action_fn=self._l2_policy_action) for s in self._eval_seeds]
        l2_is_bps = np.array([r["is_result"].is_total_bps for r in results])
        l2_fill = np.array([r["is_result"].fill_ratio for r in results])
        # is_exec_bps is None (not NaN) when fill_ratio == 0 -- same reward.py convention
        # train_l3.py's ValISEvalCallback already handles this way; dtype=float turns
        # None into nan, then nanmean/nanstd average only over episodes where it was
        # actually defined.
        l2_is_exec_bps = np.array([r["is_result"].is_exec_bps for r in results], dtype=float)
        l2_is_opp_bps = np.array([r["is_result"].is_opp_bps for r in results])

        self.logger.record("eval/val_l2_is_total_bps_mean", float(l2_is_bps.mean()))
        self.logger.record("eval/val_l2_is_total_bps_std", float(l2_is_bps.std()))
        self.logger.record("eval/val_l2_is_exec_bps_mean", float(np.nanmean(l2_is_exec_bps)))
        self.logger.record("eval/val_l2_is_exec_bps_std", float(np.nanstd(l2_is_exec_bps)))
        self.logger.record("eval/val_l2_is_opp_bps_mean", float(l2_is_opp_bps.mean()))
        self.logger.record("eval/val_l2_is_opp_bps_std", float(l2_is_opp_bps.std()))
        self.logger.record("eval/val_l2_fill_ratio_mean", float(l2_fill.mean()))
        self.logger.record("eval/val_twap_passthrough_is_total_bps_mean", float(self._twap_passthrough_is_bps.mean()))
        self.logger.record("eval/val_twap_passthrough_fill_ratio_mean", float(self._twap_passthrough_fill.mean()))
        self.logger.record(
            "eval/val_l2_beats_twap_passthrough_bps",
            float(self._twap_passthrough_is_bps.mean() - l2_is_bps.mean()),
        )
        self.logger.dump(self.num_timesteps)

        if self.verbose:
            print(
                f"[ValISEvalCallback] step={self.num_timesteps} "
                f"(paired seeds {self._eval_seeds[0]}..{self._eval_seeds[-1]}, n={self.n_eval_episodes}) "
                f"L2 IS_total_bps mean={l2_is_bps.mean():.4f} "
                f"(TWAP-passthrough baseline {self._twap_passthrough_is_bps.mean():.4f})"
            )
        return True


def build_parser() -> argparse.ArgumentParser:
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
        "for a mechanics-only run. Required so nobody launches the real budget by omission. "
        "NOTE (docs/reports/phase4_l2_reconciliation_and_plan.md Task 1): at this "
        "project's own measured L2 throughput (~4.15 decisions/sec, single env, no "
        "parallelism), 2,000,000 steps is roughly ~5.5 days of wall-clock -- confirm "
        "that's genuinely intended before launching at that scale.",
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
        "--eval", action=argparse.BooleanOptionalAction, default=True,
        help="Periodic held-out-val evaluation (ValISEvalCallback) against a fixed "
        "TWAP-passthrough baseline. Defaults ON -- a real multi-day run (see "
        "--total-timesteps' note) should not fly blind by default; pass --no-eval to "
        "skip it (e.g. for a faster --smoke-test, since the baseline computation alone "
        "costs roughly n_eval_episodes x ~4.3s, paid once upfront at training start "
        "regardless of --eval-freq).",
    )
    parser.add_argument(
        "--eval-freq", type=int, default=10_000,
        help="Timesteps between eval firings. Default 10,000 -- at this project's "
        "measured ~4.15 decisions/sec (docs/reports/phase4_l2_reconciliation_and_plan.md "
        "Task 1), that's roughly every ~40 minutes, sized to catch an obviously broken "
        "run within about an hour without adding meaningful overhead (see "
        "--n-eval-episodes for the cost this buys).",
    )
    parser.add_argument(
        "--n-eval-episodes", type=int, default=10,
        help="Episodes per eval firing. Default 10 -- NOT sized for statistical "
        "significance testing the way train_l3.py's n=50 is (a different goal); sized "
        "to be cheap at L2's measured rate. Each episode costs roughly one env.reset() "
        "(~2.0s, cited from docs/TRACK_STATUS.md's L3 measurement) plus ~18 decisions x "
        "~0.13s (this project's own measured non-reset per-decision cost) ~= ~4.3s, so "
        "n=10 is ~43s per firing -- at --eval-freq's default (~40min between firings), "
        "that's under 2%% overhead. Raise this only if the n=10 read proves too noisy to "
        "act on in practice; it isn't chosen for statistical power.",
    )
    parser.add_argument(
        "--smoke-test", action="store_true", default=False,
        help="Labels this run as a mechanics-only smoke test everywhere in output/logs/ "
        "save paths. Does not change training mechanics -- only naming/logging and a "
        "sanity cap on --total-timesteps (see _SMOKE_TEST_MAX_TIMESTEPS). A smoke test "
        "verifies the wrapper/SAC integration runs end-to-end without shape/interface "
        "errors; its reward/loss numbers are not a performance signal and must not be "
        "extrapolated from. Combine with --no-eval for the fastest possible smoke run.",
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--progress-bar", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()

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
        if args.eval:
            print(
                "[SMOKE TEST] --eval is on (default) -- this adds the eval baseline's "
                "one-time upfront cost (~n_eval_episodes x ~4.3s) even to a short smoke "
                "run. Pass --no-eval for the fastest possible smoke run."
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
    callbacks = [checkpoint_cb]

    if args.eval:
        val_dates = load_split("val")
        val_date_range = (val_dates[0].isoformat(), val_dates[-1].isoformat())
        print(f"val   date_range: {val_date_range} ({len(val_dates)} real days)")
        eval_cb = ValISEvalCallback(
            val_date_range=val_date_range,
            horizon_ticks=args.horizon_ticks,
            lookback_ticks=args.lookback_ticks,
            ticks_per_l2_decision=args.ticks_per_l2_decision,
            l3_model=l3_model,
            l3_vecnormalize_path=args.l3_vecnormalize,
            l2_include_prev_action=args.l2_include_prev_action,
            eval_freq=args.eval_freq,
            n_eval_episodes=args.n_eval_episodes,
            verbose=1,
        )
        callbacks.append(eval_cb)
    else:
        print("--no-eval: held-out evaluation disabled for this run.")

    model.learn(
        total_timesteps=args.total_timesteps, callback=CallbackList(callbacks),
        progress_bar=args.progress_bar,
    )

    Path("models").mkdir(exist_ok=True)
    save_name = "models/l2_strategist_smoke_test" if args.smoke_test else "models/l2_strategist_v1"
    model.save(save_name)
    print(f"Saved model to {save_name}.zip")
    if args.smoke_test:
        print("[SMOKE TEST] complete -- mechanics-only, not a performance signal. No real training was launched.")


if __name__ == "__main__":
    main()
