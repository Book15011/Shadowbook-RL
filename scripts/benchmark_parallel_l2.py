"""Minimal benchmark (NOT a production implementation) of parallelizing L2 training
across multiple envs via SubprocVecEnv, each worker running its own frozen L3 model copy
on CPU (per this round's measured finding: CPU predict() is FASTER than GPU for this
specific tiny model -- 0.94ms vs 1.72ms/call -- and sidesteps all VRAM contention with
L1's concurrent Ollama usage). Measures real decisions/sec at n_envs=2 and n_envs=4
against the single-env ~4.19 decisions/sec baseline measured by
scripts/profile_l2_throughput.py. Short benchmark run only, per instruction.

Run: PYTHONPATH=. .venv/bin/python scripts/benchmark_parallel_l2.py
"""
from __future__ import annotations

import time

from sb3_contrib import RecurrentPPO
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from src.data.split import load_split
from src.train.train_l2 import make_l2_wrapped_env

L3_CHECKPOINT = "models/l3_frozen_backup/l3_executioner_v1_frozen.zip"
L3_VECNORM = "models/l3_frozen_backup/l3_vecnormalize_frozen.pkl"

TOTAL_TIMESTEPS = 120  # cumulative across all envs, matched to the single-env baseline's own scale


def make_env(train_date_range: tuple[str, str]):
    """Top-level factory (not a closure over a shared model object) -- each SubprocVecEnv
    worker is a SEPARATE PROCESS, so this must construct its OWN L3 model + VecNormalize
    instance inside the worker process, not share one across processes. device='cpu' per
    this round's measured recommendation (Option A+C combined)."""

    def _init():
        l3_model = RecurrentPPO.load(L3_CHECKPOINT, device="cpu")
        return make_l2_wrapped_env(
            train_date_range, 3000, 10, l3_model, L3_VECNORM, 50, False,
        )

    return _init


def run_benchmark(n_envs: int, train_date_range: tuple[str, str]) -> float:
    vec_env = SubprocVecEnv([make_env(train_date_range) for _ in range(n_envs)])
    vec_env = VecMonitor(vec_env)

    model = SAC(
        "MlpPolicy", vec_env,
        buffer_size=500_000, gamma=0.995, batch_size=256, tau=0.005,
        learning_rate=3e-4, train_freq=1, gradient_steps=1, learning_starts=0,
        device="cuda", verbose=0,  # SAC's own policy update stays centralized on GPU regardless of n_envs
    )

    t0 = time.perf_counter()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=False)
    elapsed = time.perf_counter() - t0
    actual_steps = model.num_timesteps  # SB3 rounds up to a multiple of n_envs

    vec_env.close()
    rate = actual_steps / elapsed
    print(f"n_envs={n_envs}: {actual_steps} steps in {elapsed:.3f}s -> {rate:.3f} decisions/sec")
    return rate


def main() -> None:
    train_dates = load_split("train")
    train_date_range = (train_dates[0].isoformat(), train_dates[-1].isoformat())
    print(f"train date_range: {train_date_range} ({len(train_dates)} real days)")

    baseline_rate = 4.194  # measured by scripts/profile_l2_throughput.py, single env, GPU L3 inference
    print(f"\nsingle-env baseline (GPU L3 inference, prior measurement): {baseline_rate:.3f} decisions/sec\n")

    for n_envs in (2, 4):
        rate = run_benchmark(n_envs, train_date_range)
        speedup = rate / baseline_rate
        print(f"  -> {speedup:.2f}x speedup vs single-env baseline\n")


if __name__ == "__main__":
    main()
