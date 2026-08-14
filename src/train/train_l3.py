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
"""
from __future__ import annotations

import argparse
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


def make_env(date_range: tuple[str, str], horizon_ticks: int, lookback_ticks: int):
    """L2 stub = fixed_twap is the default behavior the environment already
    has (see module docstring) -- no l2_override kwarg exists or is needed."""
    def _init():
        return LOBExecutionEnv(
            date_range=date_range,
            horizon_ticks=horizon_ticks,
            lookback_ticks=lookback_ticks,
        )
    return _init


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
    ) -> None:
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.horizon_ticks = horizon_ticks
        self._eval_env = LOBExecutionEnv(
            date_range=val_date_range, horizon_ticks=horizon_ticks, lookback_ticks=lookback_ticks,
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

        self.logger.record("eval/val_l3_is_total_bps_mean", float(l3_is_bps.mean()))
        self.logger.record("eval/val_l3_is_total_bps_std", float(l3_is_bps.std()))
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
    args = parser.parse_args()

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

    vec_env = SubprocVecEnv([
        make_env(train_date_range, env_cfg["horizon_ticks"], env_cfg["lookback_ticks"])
        for _ in range(n_envs)
    ])
    vec_env = VecMonitor(vec_env)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=5.0, gamma=ppo_cfg["gamma"])

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
        name_prefix="l3_ppo",
        save_vecnormalize=ckpt_cfg["save_vecnormalize"],
    )
    eval_cb = ValISEvalCallback(
        val_date_range=val_date_range,
        horizon_ticks=env_cfg["horizon_ticks"],
        lookback_ticks=env_cfg["lookback_ticks"],
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        verbose=1,
    )

    model.learn(
        total_timesteps=total_timesteps, callback=CallbackList([checkpoint_cb, eval_cb]), progress_bar=True,
    )

    Path("models").mkdir(exist_ok=True)
    model.save("models/l3_executioner_v1")
    vec_env.save("models/l3_vecnormalize.pkl")
    print("Saved model to models/l3_executioner_v1, VecNormalize to models/l3_vecnormalize.pkl")


if __name__ == "__main__":
    main()
