"""L3 (Executioner) RecurrentPPO training -- architecture_spec.md Section 4.1,
reconciled against the real LOBExecutionEnv API. The Section 4.1 reference
train_l3.py illustrates LOBExecutionEnv(tier="l3", l2_override=..., seed=rank),
which does not match the real constructor.

Step 0 reconciliation (verified directly against src/envs/lob_execution_env.py,
not assumed from memory):
  - LOBExecutionEnv has no tier=, l2_override=, or seed= constructor params.
    Real params: data_dir, horizon_ticks, lookback_ticks, tick_interval_s,
    min_size_mult, max_size_mult, reward_weights, fee_bps_per_fill, date_range,
    funding_rate_dir, l1_risk_score, l1_confidence, l2_urgency,
    l2_target_slice_ratio_override.
  - L2 stub = fixed_twap needs NO extra wiring. l2_target_slice_ratio (obs
    idx 15) computed DEFAULT (_compute_l2_target_slice_ratio(), active
    whenever l2_target_slice_ratio_override is None -- which is itself the
    default) already IS a fixed-linear-TWAP-schedule fraction
    (ticks_elapsed/horizon_ticks). Built this way in Phase 2b specifically to
    serve as this stub. Simply not passing l2_target_slice_ratio_override (and
    leaving l2_urgency at its own neutral 0.5 default) gives exactly the
    L2-stubbed-as-fixed-TWAP environment Section 4.1 calls for.
  - seed=rank belongs on RecurrentPPO(..., seed=...), not on each env.
    Confirmed against the installed SB3 2.3.2 source:
    BaseAlgorithm.set_random_seed() calls env.seed(seed) on the VecEnv, and
    SubprocVecEnv.seed() distributes seed+idx to each worker automatically on
    its next reset() -- the real mechanism for the per-worker diversity that
    the Section 4.1 illustrative seed=rank was going for.

Run: PYTHONPATH=. .venv/bin/python -m src.train.train_l3 [--total-timesteps N]
[--n-envs N] [--eval-freq N] [--n-eval-episodes N] [--config configs/ppo_l3.yaml]
[--no-progress-bar] [--run-name NAME] [--overwrite-canonical]

Save-path safety (added after a bounded probe run's final save silently
overwrote a verified checkpoint -- see docs/reports/l3_replace_value_probe.md's
"separately flagged" note): the final save no longer unconditionally writes to
models/l3_executioner_v1.zip / l3_vecnormalize.pkl. If those files already
exist, the save is redirected to a run-tagged path instead
(models/l3_executioner_v1_<run-name>.zip, etc.) unless --overwrite-canonical is
passed explicitly. --run-name also namespaces this run's periodic
CheckpointCallback files, so two runs' intermediate checkpoints can no longer
silently collide either (confirmed this had already happened once, silently,
before this fix: the direction-inversion probe's own 250k/500k-step periodic
checkpoints overwrote v1's identically-named ones from its earlier, unrelated
run). See resolve_final_save_paths() below for the exact logic, and
tests/test_train_l3.py for its test coverage.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

from scripts.phase2a_sanity_suite import TWAPPolicy, run_episode
from src.data.split import load_split
from src.envs.lob_execution_env import LOBExecutionEnv
from src.envs.reward import RewardWeights


def make_env(
    date_range: tuple[str, str], horizon_ticks: int, lookback_ticks: int,
    reward_weights: RewardWeights | None = None,
):
    """L2 stub = fixed_twap is the default behavior the environment already
    has (see module docstring) -- no l2_override kwarg exists or is needed.
    reward_weights: None keeps LOBExecutionEnv's own default (RewardWeights());
    passed explicitly only when --reward-zeta overrides it (see main()), for
    the Part B/C coefficient sweep in docs/reports/phase3_l3_baseline_milestone.md."""
    def _init():
        return LOBExecutionEnv(
            date_range=date_range,
            horizon_ticks=horizon_ticks,
            lookback_ticks=lookback_ticks,
            reward_weights=reward_weights,
        )
    return _init


def resolve_final_save_paths(
    run_name: str, overwrite_canonical: bool, models_dir: Path = Path("models"),
) -> tuple[str, str]:
    """Decide where this run's final model/VecNormalize save should go.
    Refuses to silently overwrite an existing canonical checkpoint -- a
    bounded probe run's final save clobbered a verified one here once
    already (see docs/reports/l3_replace_value_probe.md) -- unless
    overwrite_canonical is explicitly True. The two canonical files are
    treated as a pair (checked with OR, not AND): if either already exists,
    both outputs redirect together, so a run can never leave a mismatched
    model/VecNormalize pair behind by only overwriting one of them.
    Pure path-decision logic, no I/O beyond the existence check -- kept
    separate from main() specifically so it's unit-testable without a GPU,
    training loop, or real config/data files.

    Returns (model_save_stem, vecnorm_save_path). model_save_stem has no
    .zip suffix, matching SB3 model.save()'s own convention (it appends
    .zip itself)."""
    canonical_model = models_dir / "l3_executioner_v1.zip"
    canonical_vecnorm = models_dir / "l3_vecnormalize.pkl"
    if (canonical_model.exists() or canonical_vecnorm.exists()) and not overwrite_canonical:
        return (
            str(models_dir / f"l3_executioner_v1_{run_name}"),
            str(models_dir / f"l3_vecnormalize_{run_name}.pkl"),
        )
    return str(models_dir / "l3_executioner_v1"), str(models_dir / "l3_vecnormalize.pkl")


class ValISEvalCallback(BaseCallback):
    """Periodic held-out evaluation against load_split("val") ONLY -- never
    test. Not in the Section 4.1 reference code (it has no eval loop at all);
    built here per the explicit requirement from the Phase 3 task. Uses the
    real business metric (compute_implementation_shortfall(), via the same
    run_episode() pattern already established in
    scripts/phase2a_sanity_suite.py for every prior sanity check this project
    has done), not SB3 stock reward-based EvalCallback.

    Baseline: TWAPPolicy from phase2a_sanity_suite.py, confirmed to fit the
    naive same-level limit-order baseline description from Section 6.2 -- it
    always posts passively at the touch (offset=0, never adaptive: the
    "same-level" part) under the identical fixed-TWAP pacing schedule L2
    stubs for L3 too (the slicing/forced-completion logic), forcing MARKET
    completion only when a slice would otherwise miss its target. This gives
    a fair, apples-to-apples comparison: does the LEARNED tick-level policy
    for L3 beat the simplest possible tick-level policy under the same macro
    pacing constraint?

    Paired design: both arms are evaluated against the exact SAME fixed set
    of eval seeds (self._eval_seeds, generated once at construction from
    EVAL_SEED_BASE and never regenerated). Since a seed deterministically
    fixes the day/window/side/qty draw (see Phase 2a), this pins the TWAP
    baseline and every single L3 eval firing across the whole run to the
    identical episode set -- matched, not independently sampled, exactly the
    pattern the Phase 2a sanity suite already used to compare policies. The
    TWAP arm is still computed once at training start and cached as a
    constant reference line (a fixed, non-learning policy has no reason to
    be re-evaluated every callback firing), but it is cached FROM the same
    seed list that every L3 firing reuses, so every reported comparison
    stays apples-to-apples throughout training, not just at one point in
    time.
    """

    EVAL_SEED_BASE = 5_000_000

    def __init__(
        self,
        val_date_range: tuple[str, str],
        horizon_ticks: int,
        lookback_ticks: int,
        eval_freq: int,
        n_eval_episodes: int,
        verbose: int = 0,
        reward_weights: RewardWeights | None = None,
    ) -> None:
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.horizon_ticks = horizon_ticks
        self._eval_env = LOBExecutionEnv(
            date_range=val_date_range, horizon_ticks=horizon_ticks, lookback_ticks=lookback_ticks,
            reward_weights=reward_weights,
        )
        # Fixed once, reused for every arm and every firing -- the paired-design guarantee.
        self._eval_seeds = [self.EVAL_SEED_BASE + i for i in range(n_eval_episodes)]
        self._twap_is_bps: np.ndarray | None = None
        self._twap_fill: np.ndarray | None = None
        self._last_eval_step = 0

    def _on_training_start(self) -> None:
        twap = TWAPPolicy(n_slices=10)
        results = [
            run_episode(self._eval_env, twap, seed=s, horizon_ticks=self.horizon_ticks)
            for s in self._eval_seeds
        ]
        self._twap_is_bps = np.array([r["is_result"].is_total_bps for r in results])
        self._twap_fill = np.array([r["is_result"].fill_ratio for r in results])
        if self.verbose:
            print(
                f"[ValISEvalCallback] TWAP baseline on val ({self.n_eval_episodes} episodes, "
                f"paired seeds {self._eval_seeds[0]}..{self._eval_seeds[-1]}): "
                f"IS_total_bps mean={self._twap_is_bps.mean():.4f} fill_ratio mean={self._twap_fill.mean():.4f}"
            )

    def _run_l3_episode(self, seed: int) -> dict:
        """Mirrors the pattern from run_episode() in phase2a_sanity_suite.py,
        but driving actions from the RecurrentPPO policy instead of a fixed
        policy object -- handles LSTM state carry-forward and VecNormalize
        observation normalization, neither of which a non-recurrent policy
        needs, which is why this cannot just call run_episode() directly."""
        vec_normalize = self.model.get_vec_normalize_env()
        env = self._eval_env
        obs, info = env.reset(seed=seed)
        lstm_states = None
        episode_start = np.ones((1,), dtype=bool)
        total_reward = 0.0
        for _ in range(self.horizon_ticks + 1):
            obs_for_policy = obs[None, :]
            if vec_normalize is not None:
                obs_for_policy = vec_normalize.normalize_obs(obs_for_policy)
            action, lstm_states = self.model.predict(
                obs_for_policy, state=lstm_states, episode_start=episode_start, deterministic=True,
            )
            episode_start = np.zeros((1,), dtype=bool)
            obs, r, term, trunc, info = env.step(action[0])
            total_reward += r
            if term or trunc:
                break
        is_result = info["implementation_shortfall"]
        return {"total_reward": total_reward, "is_result": is_result}

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_step < self.eval_freq:
            return True
        self._last_eval_step = self.num_timesteps

        results = [self._run_l3_episode(seed=s) for s in self._eval_seeds]
        l3_is_bps = np.array([r["is_result"].is_total_bps for r in results])
        l3_fill = np.array([r["is_result"].fill_ratio for r in results])
        # is_exec_bps is None (not NaN) when fill_ratio == 0 -- reward.py's documented
        # convention for "undefined, no fills to average", not "zero cost". dtype=float
        # converts None -> nan on construction; nanmean/nanstd then correctly average only
        # over episodes where it was actually defined. is_opp_bps has no such case (always
        # computed regardless of fill_ratio), so it uses plain mean/std like is_total_bps.
        l3_is_exec_bps = np.array([r["is_result"].is_exec_bps for r in results], dtype=float)
        l3_is_opp_bps = np.array([r["is_result"].is_opp_bps for r in results])

        self.logger.record("eval/val_l3_is_total_bps_mean", float(l3_is_bps.mean()))
        self.logger.record("eval/val_l3_is_total_bps_std", float(l3_is_bps.std()))
        self.logger.record("eval/val_l3_is_exec_bps_mean", float(np.nanmean(l3_is_exec_bps)))
        self.logger.record("eval/val_l3_is_exec_bps_std", float(np.nanstd(l3_is_exec_bps)))
        self.logger.record("eval/val_l3_is_opp_bps_mean", float(l3_is_opp_bps.mean()))
        self.logger.record("eval/val_l3_is_opp_bps_std", float(l3_is_opp_bps.std()))
        self.logger.record("eval/val_l3_fill_ratio_mean", float(l3_fill.mean()))
        self.logger.record("eval/val_twap_baseline_is_total_bps_mean", float(self._twap_is_bps.mean()))
        self.logger.record("eval/val_twap_baseline_fill_ratio_mean", float(self._twap_fill.mean()))
        self.logger.record("eval/val_l3_beats_twap_bps", float(self._twap_is_bps.mean() - l3_is_bps.mean()))
        self.logger.dump(self.num_timesteps)

        if self.verbose:
            print(
                f"[ValISEvalCallback] step={self.num_timesteps} "
                f"(paired seeds {self._eval_seeds[0]}..{self._eval_seeds[-1]}, n={self.n_eval_episodes}) "
                f"L3 IS_total_bps mean={l3_is_bps.mean():.4f} (TWAP baseline {self._twap_is_bps.mean():.4f})"
            )
        return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ppo_l3.yaml")
    parser.add_argument(
        "--total-timesteps", type=int, default=None,
        help="Override config total_timesteps (e.g. for a short smoke test)",
    )
    parser.add_argument("--n-envs", type=int, default=None, help="Override config n_envs")
    parser.add_argument(
        "--eval-freq", type=int, default=None,
        help="Override config eval.eval_freq_timesteps (e.g. for a short smoke test)",
    )
    parser.add_argument(
        "--n-eval-episodes", type=int, default=None,
        help="Override config eval.n_eval_episodes (e.g. for a short smoke test)",
    )
    parser.add_argument(
        "--progress-bar", action=argparse.BooleanOptionalAction, default=True,
        help="tqdm progress bar (default on). Use --no-progress-bar when stdout is "
        "redirected to a log file (e.g. a long unattended nohup run) -- tqdm's "
        "carriage-return redraws do not collapse in a plain file the way they do "
        "on a real terminal, and repeat every refresh for the run's full duration.",
    )
    parser.add_argument(
        "--resume-from", type=str, default=None,
        help="Path to a model checkpoint .zip to resume from (e.g. "
        "models/l3_checkpoints/l3_ppo_1750000_steps.zip). Requires "
        "--resume-vecnormalize. --total-timesteps is interpreted as REMAINING "
        "steps, not an absolute target -- SB3 adds the checkpoint's own "
        "num_timesteps back in automatically when reset_num_timesteps=False.",
    )
    parser.add_argument(
        "--resume-vecnormalize", type=str, default=None,
        help="Path to the matching VecNormalize .pkl for --resume-from (e.g. "
        "models/l3_checkpoints/l3_ppo_vecnormalize_1750000_steps.pkl).",
    )
    parser.add_argument(
        "--reward-zeta", type=float, default=None,
        help="Override RewardWeights.zeta (the experimental staleness-penalty "
        "coefficient -- see docs/reports/phase3_l3_baseline_milestone.md Part B) "
        "for this run only, without editing src/envs/reward.py's shared default. "
        "Needed for the coefficient sweep so multiple candidate values can run "
        "(including in parallel) off one checkout. Unset keeps RewardWeights()'s "
        "own default.",
    )
    parser.add_argument(
        "--reward-eta-replace", type=float, default=None,
        help="Override RewardWeights.eta_replace (the placement-anchored staleness "
        "coefficient -- see docs/reports/phase3_l3_baseline_milestone.md Part A) for "
        "this run only. Combine with --reward-zeta 0.0 to isolate this new term own "
        "effect, per this project established practice of changing one variable at "
        "a time. Unset keeps RewardWeights()'s own default (0.0, inert).",
    )
    parser.add_argument(
        "--subtract-twap-baseline", action="store_true",
        help="Enable RewardWeights.subtract_twap_baseline -- variance reduction, "
        "NOT an objective change (see reward.py module docstring EXPERIMENTAL 5 "
        "and docs/reports/l3_twap_baseline_reward.md). Off by default, matching "
        "RewardWeights()'s own default of False.",
    )
    parser.add_argument(
        "--warm-start-weights", type=str, default=None,
        help="Path to a model checkpoint .zip to initialize weights from, WITHOUT "
        "resuming the run it came from -- distinct from --resume-from/"
        "--resume-vecnormalize, which pair together and continue the original run's "
        "step counter and VecNormalize stats. This loads ONLY the policy weights; "
        "VecNormalize is built fresh (accumulating stats from this run's step zero, "
        "not reused from the source checkpoint's run), and the step counter resets "
        "to 0 (reset_num_timesteps=True, same as a from-scratch run) so "
        "--total-timesteps is an absolute count on a fresh 0-based axis, directly "
        "comparable to a from-scratch run at matched step counts. Mutually exclusive "
        "with --resume-from. See docs/reports/phase3_l3_baseline_milestone.md, "
        "'Physics fix + init-strategy probe'.",
    )
    parser.add_argument(
        "--run-name", type=str, default=None,
        help="Tag for this run's checkpoint filenames, used (a) as the fallback "
        "final-save name if models/l3_executioner_v1.zip already exists and "
        "--overwrite-canonical is not given, and (b) to namespace this run's "
        "periodic CheckpointCallback files so two runs' intermediate checkpoints "
        "can never silently collide. Defaults to a UTC timestamp "
        "(YYYYmmdd_HHMMSS) if not given.",
    )
    parser.add_argument(
        "--overwrite-canonical", action="store_true",
        help="Allow this run's final save to overwrite an existing "
        "models/l3_executioner_v1.zip / l3_vecnormalize.pkl. Off by default -- a "
        "bounded probe run silently overwrote a verified checkpoint here once "
        "already (see docs/reports/l3_replace_value_probe.md). Pass this only "
        "when this run is deliberately meant to supersede the current canonical "
        "checkpoint.",
    )
    args = parser.parse_args()
    if bool(args.resume_from) != bool(args.resume_vecnormalize):
        raise ValueError("--resume-from and --resume-vecnormalize must be given together")
    if args.warm_start_weights and args.resume_from:
        raise ValueError("--warm-start-weights and --resume-from are mutually exclusive")
    run_name = args.run_name or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    ppo_cfg = cfg["ppo_l3"]
    eval_cfg = cfg["eval"]
    ckpt_cfg = cfg["checkpoint"]
    env_cfg = cfg["env"]

    n_envs = args.n_envs or ppo_cfg["n_envs"]
    total_timesteps = args.total_timesteps or ppo_cfg["total_timesteps"]
    eval_freq = args.eval_freq or eval_cfg["eval_freq_timesteps"]
    n_eval_episodes = args.n_eval_episodes or eval_cfg["n_eval_episodes"]

    cuda_available = torch.cuda.is_available()
    print(f"cuda available: {cuda_available}")
    if ppo_cfg["device"] == "cuda" and not cuda_available:
        raise RuntimeError("config requests device=cuda but torch.cuda.is_available() is False")

    train_dates = load_split("train")
    val_dates = load_split("val")
    train_date_range = (train_dates[0].isoformat(), train_dates[-1].isoformat())
    val_date_range = (val_dates[0].isoformat(), val_dates[-1].isoformat())
    print(f"train date_range: {train_date_range} ({len(train_dates)} real days)")
    print(f"val   date_range: {val_date_range} ({len(val_dates)} real days)")

    reward_overrides = {}
    if args.reward_zeta is not None:
        reward_overrides["zeta"] = args.reward_zeta
    if args.reward_eta_replace is not None:
        reward_overrides["eta_replace"] = args.reward_eta_replace
    if args.subtract_twap_baseline:
        reward_overrides["subtract_twap_baseline"] = True
    reward_weights = RewardWeights(**reward_overrides) if reward_overrides else None
    if reward_overrides:
        print(f"reward_weights override: {reward_overrides} (all other weights at RewardWeights() defaults)")

    vec_env = SubprocVecEnv([
        make_env(train_date_range, env_cfg["horizon_ticks"], env_cfg["lookback_ticks"], reward_weights)
        for _ in range(n_envs)
    ])
    vec_env = VecMonitor(vec_env)
    if args.resume_from:
        vec_env = VecNormalize.load(args.resume_vecnormalize, vec_env)
    else:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=5.0, gamma=ppo_cfg["gamma"])

    if args.resume_from:
        model = RecurrentPPO.load(args.resume_from, device=ppo_cfg["device"])
        model.set_env(vec_env)
        # RecurrentPPO.load() reconstructs the model with env=None (no env= kwarg was
        # passed to .load() above), so set_random_seed()'s internal `if self.env is not
        # None: self.env.seed(seed)` was skipped during load -- and set_env() on its own
        # never seeds anything (confirmed directly against the installed SB3 source: it
        # only wraps/validates the env and assigns self.env, no seeding call at all). A
        # fresh run seeds correctly only because __init__ sets self.env BEFORE
        # _setup_model() calls set_random_seed(); here that ordering is inverted, so the
        # equivalent call has to happen explicitly, now that self.env is finally real.
        # Calling the same method a fresh run relies on (not just vec_env.seed()
        # directly) also reseeds the global python/numpy/torch RNGs and the action
        # space, matching a fresh run's _setup_model() effect exactly, not just the
        # env-specific piece of it.
        model.set_random_seed(model.seed)
        print(f"resumed from {args.resume_from}: loaded num_timesteps={model.num_timesteps}")
    elif args.warm_start_weights:
        model = RecurrentPPO.load(args.warm_start_weights, device=ppo_cfg["device"])
        model.set_env(vec_env)
        # Same reseeding rationale as the --resume-from branch above (.load()
        # constructs with env=None, so the seed-the-env step inside
        # set_random_seed() never ran until set_env() gave it a real env).
        model.set_random_seed(model.seed)
        # Unlike --resume-from, model.num_timesteps here still holds the SOURCE
        # checkpoint's original count (e.g. ~20M) -- reset_num_timesteps=True below
        # (via `not args.resume_from`, which is True here since this is a distinct
        # flag) resets it to 0 at the start of .learn(), so --total-timesteps counts
        # fresh steps on a 0-based axis, matched against a from-scratch run.
        print(
            f"warm-started WEIGHTS ONLY from {args.warm_start_weights} "
            f"(source checkpoint's num_timesteps={model.num_timesteps}, discarded -- "
            "VecNormalize is fresh, step counter resets to 0 at learn() start)"
        )
    else:
        model = RecurrentPPO(
            ppo_cfg["policy"], vec_env,
            n_steps=ppo_cfg["n_steps"],
            batch_size=ppo_cfg["batch_size"],
            n_epochs=ppo_cfg["n_epochs"],
            gamma=ppo_cfg["gamma"],
            gae_lambda=ppo_cfg["gae_lambda"],
            clip_range=ppo_cfg["clip_range"],
            ent_coef=ppo_cfg["ent_coef"],
            vf_coef=ppo_cfg["vf_coef"],
            max_grad_norm=ppo_cfg["max_grad_norm"],
            learning_rate=ppo_cfg["learning_rate"],
            policy_kwargs=dict(
                lstm_hidden_size=ppo_cfg["policy_kwargs"]["lstm_hidden_size"],
                n_lstm_layers=ppo_cfg["policy_kwargs"]["n_lstm_layers"],
                net_arch=ppo_cfg["policy_kwargs"]["net_arch"],
            ),
            tensorboard_log=ppo_cfg["tensorboard_log"],
            device=ppo_cfg["device"],
            verbose=ppo_cfg["verbose"],
            seed=ppo_cfg["seed"],
        )
    print(f"model device actually in use: {model.device}")

    checkpoint_cb = CheckpointCallback(
        save_freq=max(1, ckpt_cfg["save_freq_timesteps"] // n_envs),
        save_path="models/l3_checkpoints/",
        name_prefix=f"l3_ppo_{run_name}",
        save_vecnormalize=ckpt_cfg["save_vecnormalize"],
    )
    eval_cb = ValISEvalCallback(
        val_date_range=val_date_range,
        horizon_ticks=env_cfg["horizon_ticks"],
        lookback_ticks=env_cfg["lookback_ticks"],
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        verbose=1,
        reward_weights=reward_weights,
    )

    model.learn(
        total_timesteps=total_timesteps, callback=CallbackList([checkpoint_cb, eval_cb]),
        progress_bar=args.progress_bar, reset_num_timesteps=not args.resume_from,
    )

    Path("models").mkdir(exist_ok=True)
    model_save_stem, vecnorm_save_path = resolve_final_save_paths(
        run_name, args.overwrite_canonical, Path("models")
    )
    if model_save_stem != "models/l3_executioner_v1":
        print(
            "models/l3_executioner_v1.zip already exists and --overwrite-canonical "
            "was not given -- NOT overwriting it (this is exactly what silently "
            f"clobbered a verified checkpoint once before). Saving this run's final "
            f"checkpoint to {model_save_stem}.zip / {vecnorm_save_path} instead. Pass "
            "--overwrite-canonical if this run is deliberately meant to supersede the "
            "current canonical checkpoint."
        )
    model.save(model_save_stem)
    vec_env.save(vecnorm_save_path)
    print(f"Saved model to {model_save_stem}, VecNormalize to {vecnorm_save_path}")


if __name__ == "__main__":
    main()
