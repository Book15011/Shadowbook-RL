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
learning_rate/train_freq are Section 4.1's literal reference values, used as-is because
they were never flagged as needing independent derivation the way buffer_size/gamma
were -- this is a deliberate distinction, not an oversight, so this script's comments
say explicitly which is which rather than reading as if everything here was checked.
gradient_steps IS newly derived this round (see _resolve_gradient_steps below) --
train_freq=1/gradient_steps=1 was a single-env-only pairing, silently wrong once
n_envs>1 (see that function's own docstring).

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
phase4_l2_reconciliation_and_plan.md's CURRENT STATE section. Deliberately NOT added in
this round either (the vectorization round) -- out of that round's own stated scope, and
tests/test_train_l2.py's test_get_vec_normalize_env_is_none_when_l2_obs_not_normalized
pins this as a canary, not silently left stale.

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

VECTORIZATION ROUND (this round -- see docs/TRACK_STATUS.md's L2 entry for the full
report): training now runs SubprocVecEnv([n_envs worker envs]), each worker loading its
OWN frozen L3 RecurrentPPO checkpoint on CPU (device="cpu" is hardcoded per worker,
independent of --device, which only controls the SAC policy's own device) -- ported
directly from scripts/benchmark_controlled.py / benchmark_controlled_numeric.py's proven,
measured methodology, not a fresh design. Three things that pattern required getting
right, each verified this round (see docs/TRACK_STATUS.md for the actual verification
runs, not just the reasoning below):
  1. Thread-capping is MANDATORY, not an optimization: OMP_NUM_THREADS/MKL_NUM_THREADS
     env vars (set at module import time, below, before torch is ever imported by
     anything) plus torch.set_num_threads(1) inside each worker's own _init() (must be
     called post-fork, inside the worker process, not the parent -- env vars alone are
     read by torch at its OWN import time inside each worker, which is fine, but the
     explicit call is kept too for defense against any code path that imports torch
     before reading the env var). An earlier, un-thread-capped attempt at this pattern
     (this project's own prior round) measured 7-9x SLOWER than the thread-capped
     version -- N worker processes each spinning up a full-width BLAS/OMP thread pool on
     a shared, finite core count is textbook oversubscription, not a hypothetical risk.
  2. Per-worker LSTM state isolation is correct BY CONSTRUCTION, not by convention: each
     worker's _init() (make_l2_subproc_env below) constructs its own env + FrozenL3Wrapper
     + RecurrentPPO instance from scratch, inside that worker's own forked/spawned
     process -- FrozenL3Wrapper._l3_lstm_state (wrappers.py) is a plain instance
     attribute, and SubprocVecEnv workers are separate OS processes with no shared
     memory for it. The failure mode this guards against (one shared instance handed to
     every worker closure, e.g. `[the_same_env] * n_envs`) is not what this code does --
     verified empirically anyway this round (see docs/TRACK_STATUS.md), not just assumed
     from the architecture.
  3. Seed reproducibility needs an extra step SB3 does NOT do for you: SAC(seed=...) ->
     BaseAlgorithm.set_random_seed() seeds the calling (main) process's python/numpy/
     torch RNGs and calls env.seed(seed) on the VecEnv, which SubprocVecEnv resolves to
     per-worker env.seed(seed+idx) (gym's own np_random) on each worker's NEXT reset() --
     confirmed against the installed SB3 2.3.2 source, same mechanism train_l3.py's own
     module docstring already documented for RecurrentPPO. What this does NOT reach is a
     SubprocVecEnv worker's own separate process's torch RNG -- and the frozen L3 policy's
     predict(deterministic=False) call (the TRAINING-time default, see
     l2_include_prev_action/l3_deterministic below) samples from exactly that RNG. Left
     unseeded, two runs with an identical --seed would still diverge through the frozen
     L3's own stochastic action choices. Fixed by torch.manual_seed(seed + rank) inside
     each worker's _init(), using the same seed+idx offset convention SB3 itself uses for
     env-level seeding (for consistency, not because the offset itself matters).

Run (wiring/smoke-test only so far -- no real multi-day training launch):
PYTHONPATH=. .venv/bin/python -m src.train.train_l2 \\
    --l3-checkpoint models/l3_executioner_v1.zip \\
    --l3-vecnormalize models/l3_vecnormalize.pkl \\
    --total-timesteps 200 \\
    --smoke-test --no-eval
"""
from __future__ import annotations

import os

# Read by torch at ITS OWN import time (below) in every process this script touches,
# including each SubprocVecEnv worker (which imports torch fresh inside _init(), after
# fork/spawn) -- must be set before torch is imported anywhere, so this sits above every
# other import. See module docstring point 1: oversubscription previously measured this
# project's own vectorized-env pattern at 7-9x slower without it.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from src.data.split import load_split
from src.envs.lob_execution_env import LOBExecutionEnv
from src.envs.wrappers import FrozenL3Wrapper

_SMOKE_TEST_MAX_TIMESTEPS = 10_000  # guards against --smoke-test + a real-sized budget by accident

# Numeric-format archive (src/data/l2_numeric_format.py's *.npzst files), conversion +
# equivalence-verified as of this project's numeric-format round (770/770 seed
# comparisons, byte-identical vs. the original parquet/JSON archive across all 441 days --
# see docs/TRACK_STATUS.md). Now the default training input -- PARQUET_DATA_DIR below is
# LOBExecutionEnv's own original-format default, kept only for an explicit opt-out.
NUMERIC_DATA_DIR = "data/raw_l2_bybit_numeric/BTCUSDT"
PARQUET_DATA_DIR = "data/raw_l2_bybit/BTCUSDT"


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
    use_numeric_format: bool = False,
) -> FrozenL3Wrapper:
    # data_dir defaults to LOBExecutionEnv's own default -- exposed as a parameter (not
    # previously) so tests can point this at a small synthetic data_dir instead of the
    # real archive; main() below never passes it, so real invocations are unaffected.
    # l3_deterministic default (False) matches training-time exploration -- ValISEvalCallback
    # below overrides this to True for its own eval env (see wrappers.py's module
    # docstring, correction 3: eval needs the frozen L3 policy itself to be reproducible,
    # not just L2's own action selection).
    # use_numeric_format (this round): appended as a trailing, defaulted kwarg
    # specifically so every existing positional call site (tests/test_train_l2.py) keeps
    # working unchanged -- see LOBExecutionEnv's own constructor for what this actually
    # switches (the *.npzst vs *.parquet glob + read path; see this module's docstring
    # for the equivalence verification behind defaulting main()'s own real invocations to
    # True).
    env = LOBExecutionEnv(
        data_dir=data_dir, date_range=date_range, horizon_ticks=horizon_ticks, lookback_ticks=lookback_ticks,
        use_numeric_format=use_numeric_format,
    )
    return FrozenL3Wrapper(
        env, l3_model, l3_vecnormalize_path,
        ticks_per_l2_decision=ticks_per_l2_decision,
        l2_include_prev_action=l2_include_prev_action,
        l3_deterministic=l3_deterministic,
    )


def make_l2_env(*args, **kwargs) -> Monitor:
    """Single, non-vectorized training env: same construction as make_l2_wrapped_env,
    Monitor-wrapped for SAC's own rollout/episode-stat tracking. Kept unchanged from
    before this round -- tests/test_train_l2.py exercises this directly for fast,
    non-multiprocess mechanics testing. main()'s own real training path no longer uses
    this (see make_l2_subproc_env below): a real run always goes through SubprocVecEnv,
    even at --n-envs 1, so the production code path matches exactly what
    scripts/benchmark_controlled_numeric.py measured -- VecMonitor wraps the whole vec
    env there, not each sub-env individually, which is why this Monitor-per-env function
    is not reused for that path. The eval callback below builds its own separate instance
    directly via make_l2_wrapped_env (no Monitor -- it drives episodes manually, same
    convention as train_l3.py's ValISEvalCallback)."""
    return Monitor(make_l2_wrapped_env(*args, **kwargs))


def make_l2_subproc_env(
    rank: int,
    date_range: tuple[str, str],
    horizon_ticks: int,
    lookback_ticks: int,
    ticks_per_l2_decision: int,
    l2_include_prev_action: bool,
    data_dir: str,
    use_numeric_format: bool,
    l3_checkpoint_path: str,
    l3_vecnormalize_path: str,
    seed: int,
) -> Callable[[], FrozenL3Wrapper]:
    """SubprocVecEnv worker factory for real (vectorized) training -- ported from
    scripts/benchmark_controlled_numeric.py's make_env(), not a fresh design. Returns a
    thunk (SubprocVecEnv's own required interface): everything inside _init() runs AFTER
    fork/spawn, inside the worker's own process, which is what makes the thread-capping,
    per-worker L3 checkpoint load, and per-worker torch seeding below actually land where
    they need to -- see this module's docstring, VECTORIZATION ROUND, for why each of
    these three is required and not just defensive boilerplate."""

    def _init() -> FrozenL3Wrapper:
        torch.set_num_threads(1)
        torch.manual_seed(seed + rank)
        l3_model = RecurrentPPO.load(l3_checkpoint_path, device="cpu")
        return make_l2_wrapped_env(
            date_range, horizon_ticks, lookback_ticks,
            l3_model, l3_vecnormalize_path, ticks_per_l2_decision, l2_include_prev_action,
            data_dir=data_dir, use_numeric_format=use_numeric_format,
        )

    return _init


def _resolve_gradient_steps(n_envs: int, override: int | None) -> int:
    """How many SAC gradient updates to run per training() call, given n_envs parallel
    workers. Pure function, no env/model needed -- unit-testable in isolation (see
    tests/test_train_l2.py).

    Confirmed against the installed SB3 2.3.2 source (stable_baselines3/common/
    off_policy_algorithm.py): with train_freq=(1, "step") (this script's fixed value,
    Section 4.1's own reference), collect_rollouts()'s `num_collected_steps` counter --
    the thing train_freq is actually compared against -- increments once per
    env.step(actions) CALL, regardless of env.num_envs; self.num_timesteps, by contrast,
    increments by env.num_envs each such call. So train_freq=1 always triggers exactly
    ONE training() call per env.step() call, no matter how many parallel workers that one
    call advanced -- meaning gradient_steps (how many gradient updates that ONE call
    performs) sets the update-to-data ratio directly: n_envs new transitions arrive per
    training() call, and gradient_steps updates are drawn against them.

    The single-env script this was ported from used gradient_steps=1 (Section 4.1's
    literal reference value, 1 gradient step per 1 new transition -- UTD ratio 1.0).
    Carrying that same literal value over unchanged to n_envs>1 would silently cut the
    UTD ratio to 1/n_envs (e.g. 1/4 at this script's own default --n-envs), a real change
    to SAC's sample efficiency that was never a deliberate choice -- Section 4.1's
    reference was single-env only and was never re-derived for parallel workers the way
    buffer_size/gamma explicitly were (see this module's own docstring). Default (no
    --gradient-steps override) resolves to n_envs, preserving the same 1-gradient-step-
    per-transition ratio the reference value gave at n_envs=1. Override only for a
    deliberate ablation of the UTD ratio itself.
    """
    if override is not None:
        return override
    return max(1, n_envs)


def resolve_l2_final_save_paths(
    run_name: str, overwrite_canonical: bool, models_dir: Path = Path("models"),
) -> str:
    """L2 analog of train_l3.py's resolve_final_save_paths -- same rationale: a bounded
    L3 probe run's final save once silently overwrote a verified checkpoint this way (see
    train_l3.py's own module docstring and docs/reports/l3_replace_value_probe.md), and
    this project's own convention since then is that no training script's final save
    unconditionally overwrites its canonical checkpoint. L2 has no VecNormalize to pair
    (this script does not wrap the training env in VecNormalize -- see module docstring),
    so this resolves a single model path, not a pair the way train_l3.py's version does.

    Pure path-decision logic, no I/O beyond the existence check -- kept separate from
    main() specifically so it's unit-testable without a GPU, training loop, or real
    config/data files (see tests/test_train_l2.py, mirroring tests/test_train_l3.py's own
    coverage of the L3 version). Does not apply to --smoke-test saves, which already use
    a fixed, clearly-namespaced path (models/l2_strategist_smoke_test) with no collision
    risk against the canonical checkpoint.

    Returns model_save_stem (no .zip suffix, matching SB3 model.save()'s own convention).
    """
    canonical_model = models_dir / "l2_strategist_v1.zip"
    if canonical_model.exists() and not overwrite_canonical:
        return str(models_dir / f"l2_strategist_v1_{run_name}")
    return str(models_dir / "l2_strategist_v1")


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
    L3.predict()+env.step() calls, and roughly HALF of L2's measured single-env training
    wall-clock is env.reset() overhead (L2 episodes end well short of the full
    60-decision horizon, so reset() is paid disproportionately often per unit of training
    compared to L3's own tick-level training). eval_freq/n_eval_episodes defaults (see
    train_l2.py's CLI help for the arithmetic) are sized against the measured single-env
    ~4.15 decisions/sec rate (this callback's own eval env is always single-process,
    never vectorized -- see module docstring), not copied from L3's own n=50 (itself
    sized for statistical significance testing -- a different goal than "catch an
    obviously broken run early" cheaply, which is what this needs to do).
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
        use_numeric_format: bool = False,
    ) -> None:
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self._max_decisions = horizon_ticks // ticks_per_l2_decision + 1
        self._eval_env = make_l2_wrapped_env(
            val_date_range, horizon_ticks, lookback_ticks,
            l3_model, l3_vecnormalize_path, ticks_per_l2_decision, l2_include_prev_action,
            data_dir=data_dir, l3_deterministic=True, use_numeric_format=use_numeric_format,
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
        "When --resume-from is given, this is REMAINING steps, not an absolute target -- "
        "SB3 adds the checkpoint's own num_timesteps back in automatically (see "
        "reset_num_timesteps below). NOTE: wall-clock now depends on --n-envs -- see that "
        "flag's own help for the measured throughput this project's numeric-format "
        "controlled benchmark found at each --n-envs setting.",
    )
    parser.add_argument(
        "--n-envs", type=int, default=4,
        help="Parallel training environments (SubprocVecEnv), each running its own frozen "
        "L3 CPU inference, thread-capped to 1 (mandatory -- see module docstring, an "
        "earlier un-capped attempt at this same pattern measured 7-9x SLOWER from thread "
        "oversubscription). Default 4, not the higher-throughput 8: "
        "scripts/benchmark_controlled_numeric.py measured n_envs=8 at 26.2GB RSS (against "
        "a near-EMPTY replay buffer -- the buffer's own eventual full-500,000-transition "
        "footprint is a separate, much smaller ~174MB regardless of n_envs, since SAC's "
        "buffer_size is a TOTAL transition cap divided by n_envs internally, confirmed "
        "against the installed SB3 source) and n_envs=4 at 91% of n_envs=8's measured "
        "throughput (23.847 vs 31.808 dec/s) for half the process footprint. Given this "
        "box's OOM history (the numeric-format conversion's own N_WORKERS=8 run was "
        "OOM-killed -- see docs/TRACK_STATUS.md) 4 is the safer default for an unattended "
        "multi-day run. Pass --n-envs 8 explicitly to trade that margin for throughput.",
    )
    parser.add_argument(
        "--gradient-steps", type=int, default=None,
        help="SAC gradient updates per training() call. Default (unset) resolves to "
        "--n-envs, preserving a 1-gradient-step-per-collected-transition ratio regardless "
        "of --n-envs -- see _resolve_gradient_steps()'s own docstring for the exact SB3 "
        "mechanics this corrects for (train_freq=1 triggers one training() call per "
        "env.step() call regardless of n_envs, so leaving gradient_steps fixed at "
        "Section 4.1's single-env reference value of 1 would silently cut the "
        "update-to-data ratio to 1/n_envs). Override only for a deliberate UTD ablation.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed for SAC's own python/numpy/torch/action_space RNGs and, via SB3's own "
        "set_random_seed()->env.seed() chain, each SubprocVecEnv worker's env-level "
        "np_random (worker i gets seed+i, SB3's own convention). Also threaded explicitly "
        "to each worker's OWN process via torch.manual_seed(seed+i) inside "
        "make_l2_subproc_env -- required for the frozen L3 policy's stochastic "
        "predict(deterministic=False) sampling to be reproducible too, since SB3's own "
        "seeding never reaches into a SubprocVecEnv worker's separate process (confirmed "
        "against the installed SB3 source; see module docstring point 3). No seed existed "
        "on this script at all before this round. Ignored when --resume-from is given --"
        "the resumed model's OWN stored seed (from its original construction) is reused "
        "instead, matching train_l3.py's own --resume-from precedent.",
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
        "--use-numeric-format", action=argparse.BooleanOptionalAction, default=True,
        help="Read the converted numeric (.npzst) archive instead of the original "
        "parquet/JSON one. Defaults ON: the numeric-format conversion is complete and "
        "equivalence-verified (770/770 fixed-seed comparisons, byte-identical vs. the "
        "original format across all 441 converted days -- see docs/TRACK_STATUS.md), and "
        "it is now this project's production training input, not an experimental "
        "alternative -- this flag exists so that choice is always explicit in any given "
        "run's own printed output and argv, never just inherited silently. Pass "
        "--no-use-numeric-format (typically together with --data-dir "
        f"{PARQUET_DATA_DIR}) to deliberately train against the original archive instead "
        "-- which itself is never written to by this script or the conversion either way.",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Overrides the data directory implied by --use-numeric-format "
        f"({NUMERIC_DATA_DIR} when on, {PARQUET_DATA_DIR} when off). Set this explicitly "
        "only to point at something other than the two project-standard archives (e.g. a "
        "small synthetic dir for testing) -- for normal use, --use-numeric-format alone "
        "already selects the correct default path.",
    )
    parser.add_argument(
        "--eval", action=argparse.BooleanOptionalAction, default=True,
        help="Periodic held-out-val evaluation (ValISEvalCallback) against a fixed "
        "TWAP-passthrough baseline. Defaults ON -- a real multi-day run (see "
        "--total-timesteps' note) should not fly blind by default; pass --no-eval to "
        "skip it (e.g. for a faster --smoke-test, since the baseline computation alone "
        "costs roughly n_eval_episodes x ~4.3s, paid once upfront at training start "
        "regardless of --eval-freq). The eval env itself is always single-process, never "
        "vectorized (see module docstring) -- its cost does not scale with --n-envs.",
    )
    parser.add_argument(
        "--eval-freq", type=int, default=10_000,
        help="Timesteps between eval firings. Default 10,000 -- at this project's "
        "measured single-env ~4.15 decisions/sec (docs/reports/"
        "phase4_l2_reconciliation_and_plan.md Task 1), that's roughly every ~40 minutes "
        "of TRAINING wall-clock at --n-envs 1; a higher --n-envs reaches this many "
        "timesteps proportionally faster (throughput scales with --n-envs, see that "
        "flag's help), so eval fires proportionally more often in wall-clock terms too, "
        "sized to catch an obviously broken run early without adding meaningful overhead "
        "(see --n-eval-episodes for the fixed per-firing cost this buys, independent of "
        "--n-envs).",
    )
    parser.add_argument(
        "--n-eval-episodes", type=int, default=10,
        help="Episodes per eval firing. Default 10 -- NOT sized for statistical "
        "significance testing the way train_l3.py's n=50 is (a different goal); sized "
        "to be cheap at L2's measured single-env rate (the eval env is never "
        "vectorized). Each episode costs roughly one env.reset() (~2.0s, cited from "
        "docs/TRACK_STATUS.md's L3 measurement) plus ~18 decisions x ~0.13s (this "
        "project's own measured non-reset per-decision cost) ~= ~4.3s, so n=10 is ~43s "
        "per firing. Raise this only if the n=10 read proves too noisy to act on in "
        "practice; it isn't chosen for statistical power.",
    )
    parser.add_argument(
        "--checkpoint-freq-timesteps", type=int, default=50_000,
        help="Timesteps between periodic checkpoint saves for a REAL (non-smoke-test) "
        "run -- internally divided by --n-envs per SB3's own CheckpointCallback "
        "documentation (save_freq counts callback firings = env.step() calls on the "
        "VecEnv, each of which advances --n-envs timesteps at once). Also saves the "
        "replay buffer alongside the model at each firing (CheckpointCallback's "
        "save_replay_buffer=True, real runs only -- see --resume-replay-buffer). Smoke "
        "tests instead save at total_timesteps//4 regardless of this flag, without a "
        "replay buffer, and are not run-name-tagged (see --run-name). Sized against this "
        "box's OOM history: lower this if losing up to this many steps of progress to an "
        "unattended crash is not acceptable at the chosen --n-envs/throughput.",
    )
    parser.add_argument(
        "--run-name", type=str, default=None,
        help="Tag for this run's checkpoint filenames (models/l2_checkpoints/"
        "l2_sac_<run-name>_*) and, if models/l2_strategist_v1.zip already exists and "
        "--overwrite-canonical is not given, the fallback final-save name too -- same "
        "rationale and pattern as train_l3.py's --run-name (see "
        "resolve_l2_final_save_paths() above and docs/reports/l3_replace_value_probe.md "
        "for the incident that motivated it there). Defaults to a UTC timestamp "
        "(YYYYmmdd_HHMMSS) if not given. Does not affect --smoke-test saves, which "
        "already use a fixed, clearly-namespaced path with no collision risk.",
    )
    parser.add_argument(
        "--overwrite-canonical", action="store_true",
        help="Allow this run's final save to overwrite an existing "
        "models/l2_strategist_v1.zip. Off by default -- see --run-name and "
        "train_l3.py's own precedent for the incident this guards against "
        "(docs/reports/l3_replace_value_probe.md). Does not apply to --smoke-test.",
    )
    parser.add_argument(
        "--resume-from", type=str, default=None,
        help="Path to an SAC model checkpoint .zip to resume from (e.g. "
        "models/l2_checkpoints/l2_sac_<run-name>_250000_steps.zip). --total-timesteps is "
        "then interpreted as REMAINING steps, not an absolute target. See "
        "--resume-replay-buffer to also restore the collected replay buffer (recommended "
        "but not required -- SAC can resume with an empty buffer and refill it, at some "
        "cost to immediate post-resume sample quality, not a correctness problem).",
    )
    parser.add_argument(
        "--resume-replay-buffer", type=str, default=None,
        help="Path to a replay buffer .pkl saved alongside --resume-from (produced "
        "automatically by CheckpointCallback's save_replay_buffer=True on every real-run "
        "checkpoint -- see --checkpoint-freq-timesteps -- e.g. "
        "models/l2_checkpoints/l2_sac_<run-name>_replay_buffer_250000_steps.pkl). "
        "Requires --resume-from. Optional even then (e.g. resuming from a checkpoint "
        "saved before this existed) -- omitting it still resumes training mechanics "
        "correctly, just without the previously collected transitions.",
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
    if args.resume_replay_buffer and not args.resume_from:
        raise ValueError("--resume-replay-buffer requires --resume-from")

    run_name = args.run_name or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    args.data_dir = args.data_dir or (NUMERIC_DATA_DIR if args.use_numeric_format else PARQUET_DATA_DIR)

    run_label = "SMOKE TEST" if args.smoke_test else "TRAINING RUN"
    print(f"=== L2 SAC {run_label} (run_name={run_name}) ===")
    print(
        f"data format: {'numeric (.npzst)' if args.use_numeric_format else 'parquet/JSON'} "
        f"-- data_dir={args.data_dir}"
    )
    print(f"n_envs={args.n_envs}, seed={args.seed}")
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

    # Loaded once here (device=args.device, typically cuda) for the eval callback's own
    # single-process env below -- distinct from each SubprocVecEnv training worker's OWN
    # separately-loaded, CPU-pinned copy (make_l2_subproc_env). The eval env is never
    # vectorized and never competes with training workers for cores, so it has no reason
    # to share the CPU-only constraint those workers do.
    l3_model = RecurrentPPO.load(args.l3_checkpoint, device=args.device)

    vec_env = SubprocVecEnv([
        make_l2_subproc_env(
            i, train_date_range, args.horizon_ticks, args.lookback_ticks,
            args.ticks_per_l2_decision, args.l2_include_prev_action,
            args.data_dir, args.use_numeric_format,
            args.l3_checkpoint, args.l3_vecnormalize, args.seed,
        )
        for i in range(args.n_envs)
    ])
    vec_env = VecMonitor(vec_env)

    gradient_steps = _resolve_gradient_steps(args.n_envs, args.gradient_steps)
    print(f"gradient_steps={gradient_steps} (train_freq=1 step, i.e. once per {args.n_envs}-transition batch)")

    if args.resume_from:
        model = SAC.load(args.resume_from, device=args.device)
        model.set_env(vec_env)
        # SAC.load() reconstructs the model with env=None (no env= kwarg passed to
        # .load() above), so set_random_seed()'s internal `if self.env is not None:
        # self.env.seed(seed)` was skipped during load -- same reasoning and fix as
        # train_l3.py's own --resume-from branch (see that script's comment on this same
        # call). model.seed holds the ORIGINAL run's seed (persisted through save/load),
        # reused here rather than --seed so a resume never silently seeds differently
        # from the run it's continuing.
        model.set_random_seed(model.seed)
        if args.resume_replay_buffer:
            model.load_replay_buffer(args.resume_replay_buffer)
            print(
                f"resumed replay buffer from {args.resume_replay_buffer} "
                f"({model.replay_buffer.size()} transitions)"
            )
        # SAC.load() restores gradient_steps from the ORIGINAL run's pickled
        # hyperparams, read back as self.gradient_steps by learn()'s own training loop
        # (confirmed against the installed SB3 source) -- if this resume invocation uses
        # a DIFFERENT --n-envs than the original run (e.g. restarting at a smaller
        # --n-envs after a crash, or scaling up once more headroom is confirmed), the
        # loaded value would silently stay matched to the OLD n_envs, not this run's,
        # quietly reintroducing the same UTD-ratio mismatch _resolve_gradient_steps
        # exists to prevent. Recomputed and reapplied explicitly so a resume's
        # update-to-data ratio always matches ITS OWN --n-envs/--gradient-steps, not
        # whatever the original run happened to use.
        model.gradient_steps = gradient_steps
        print(f"resumed from {args.resume_from}: loaded num_timesteps={model.num_timesteps}")
    else:
        model = SAC(
            "MlpPolicy", vec_env,
            # -- Derived + re-confirmed for L2's real cadence (docs/reports/
            # phase4_l2_reconciliation_and_plan.md FINAL SPEC Step 3) --
            buffer_size=500_000,  # TOTAL transition cap (SB3 divides by n_envs internally, confirmed against source) -- ~8,333 L2-episode-equivalents of coverage, ~25% of the full 2M-step run's transition volume, independent of --n-envs.
            gamma=0.995,  # re-derived on L2's OWN cadence (not L3's tick-level reasoning): effective horizon ~3.3x the 60-decision episode length, defensible given the terminal-IS-dominated reward structure.
            # -- Section 4.1 reference values, carried over as-is -- NOT independently
            # derived for L2 the way the two above were; use as-is per instruction absent a
            # concrete reason not to. --
            batch_size=256,
            tau=0.005,
            learning_rate=3e-4,
            train_freq=1,
            gradient_steps=gradient_steps,  # see _resolve_gradient_steps -- NOT the Section 4.1 literal value once n_envs>1.
            tensorboard_log=("logs/l2_sac_smoke/" if args.smoke_test else "logs/l2_sac/"),
            device=args.device,
            seed=args.seed,
            verbose=1,
        )
    print(f"model device actually in use: {model.device}")

    ckpt_dir = "models/l2_checkpoints_smoke/" if args.smoke_test else "models/l2_checkpoints/"
    ckpt_prefix = "l2_sac_smoke" if args.smoke_test else f"l2_sac_{run_name}"
    if args.smoke_test:
        checkpoint_cb = CheckpointCallback(
            save_freq=max(1, args.total_timesteps // 4),
            save_path=ckpt_dir,
            name_prefix=ckpt_prefix,
        )
    else:
        checkpoint_cb = CheckpointCallback(
            save_freq=max(1, args.checkpoint_freq_timesteps // args.n_envs),
            save_path=ckpt_dir,
            name_prefix=ckpt_prefix,
            save_replay_buffer=True,
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
            data_dir=args.data_dir,
            use_numeric_format=args.use_numeric_format,
        )
        callbacks.append(eval_cb)
    else:
        print("--no-eval: held-out evaluation disabled for this run.")

    model.learn(
        total_timesteps=args.total_timesteps, callback=CallbackList(callbacks),
        progress_bar=args.progress_bar, reset_num_timesteps=not args.resume_from,
    )

    Path("models").mkdir(exist_ok=True)
    if args.smoke_test:
        save_name = "models/l2_strategist_smoke_test"
    else:
        save_name = resolve_l2_final_save_paths(run_name, args.overwrite_canonical, Path("models"))
        if save_name != "models/l2_strategist_v1":
            print(
                "models/l2_strategist_v1.zip already exists and --overwrite-canonical "
                "was not given -- NOT overwriting it (this is exactly what silently "
                f"clobbered a verified L3 checkpoint once before -- see "
                "docs/reports/l3_replace_value_probe.md). Saving this run's final "
                f"checkpoint to {save_name}.zip instead. Pass --overwrite-canonical if "
                "this run is deliberately meant to supersede the current canonical "
                "checkpoint."
            )
    model.save(save_name)
    print(f"Saved model to {save_name}.zip")
    if args.smoke_test:
        print("[SMOKE TEST] complete -- mechanics-only, not a performance signal. No real training was launched.")


if __name__ == "__main__":
    main()
