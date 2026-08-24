"""Diagnostic follow-up to benchmark_parallel_l2.py's first run, which showed parallel
envs performing WORSE than single-env (0.14x/0.11x "speedup" at n_envs=2/4) -- the
opposite of the design's prediction. This version tests the leading hypothesis (CPU
thread oversubscription: each worker's torch/BLAS backend defaulting to multi-threaded
ops, N processes x multiple threads each competing for only 16 physical cores -- OBSERVED
live during the first run via `ps aux` showing ~375% CPU per worker process) by capping
each worker to a single thread, AND separates one-time startup cost (subprocess spawn +
model/VecNormalize load + first cold-cache reset()) from steady-state per-step rate, since
a short benchmark's total_timesteps is small enough that startup cost could otherwise
dominate and be mistaken for a genuine steady-state regression.

Run: PYTHONPATH=. .venv/bin/python scripts/benchmark_parallel_l2_v2.py
"""
from __future__ import annotations

import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from sb3_contrib import RecurrentPPO
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from src.data.split import load_split
from src.train.train_l2 import make_l2_wrapped_env

L3_CHECKPOINT = "models/l3_frozen_backup/l3_executioner_v1_frozen.zip"
L3_VECNORM = "models/l3_frozen_backup/l3_vecnormalize_frozen.pkl"

TOTAL_TIMESTEPS = 120


def make_env(train_date_range: tuple[str, str]):
    def _init():
        import torch
        torch.set_num_threads(1)  # the hypothesis under test
        l3_model = RecurrentPPO.load(L3_CHECKPOINT, device="cpu")
        return make_l2_wrapped_env(
            train_date_range, 3000, 10, l3_model, L3_VECNORM, 50, False,
        )

    return _init


def run_benchmark(n_envs: int, train_date_range: tuple[str, str]) -> None:
    vec_env = SubprocVecEnv([make_env(train_date_range) for _ in range(n_envs)])
    vec_env = VecMonitor(vec_env)

    model = SAC(
        "MlpPolicy", vec_env,
        buffer_size=500_000, gamma=0.995, batch_size=256, tau=0.005,
        learning_rate=3e-4, train_freq=1, gradient_steps=1, learning_starts=0,
        device="cuda", verbose=0,
    )

    # Split timing: first vec_env.step() call (pays subprocess-spawn-adjacent startup +
    # each worker's first, cold-cache env.reset()) vs everything after, to separate
    # one-time startup cost from steady-state per-step rate.
    t0 = time.perf_counter()
    obs = vec_env.reset()
    t_reset = time.perf_counter() - t0

    t1 = time.perf_counter()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=False)
    t_learn = time.perf_counter() - t1
    actual_steps = model.num_timesteps

    vec_env.close()
    steady_rate = actual_steps / t_learn
    print(
        f"n_envs={n_envs}: vec_env.reset()={t_reset:.3f}s (one-time, {n_envs} workers' "
        f"first cold reset, parallel) | learn()={t_learn:.3f}s for {actual_steps} steps "
        f"-> {steady_rate:.3f} decisions/sec (excludes the initial reset above)"
    )


def main() -> None:
    train_dates = load_split("train")
    train_date_range = (train_dates[0].isoformat(), train_dates[-1].isoformat())
    print(f"train date_range: {train_date_range} ({len(train_dates)} real days)")
    print("OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, torch.set_num_threads(1) per worker\n")

    baseline_rate = 4.194  # single-env, GPU L3 inference, prior measurement
    print(f"single-env baseline: {baseline_rate:.3f} decisions/sec\n")

    run_benchmark(2, train_date_range)
    run_benchmark(4, train_date_range)


if __name__ == "__main__":
    main()
