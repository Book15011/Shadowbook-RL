"""Cheap, bounded companion to scripts/benchmark_controlled.py -- same seed,
same TOTAL_TIMESTEPS, same thread-capping/measurement machinery, but drawing
from the REAL full 405-day train split instead of the narrow 10-day pool, to
get a real (not purely arithmetic) realistic-cache-rate throughput number.
Only n_envs in (1, 8) and 1 trial each (not 3) -- explicitly a bounded check to
show which direction and roughly how far the narrow pool's 48.8%-vs-2.4%
cache-hit gap shifts throughput, not a full replacement for the controlled
benchmark's own trustworthy 3-trial numbers over its fixed pool.

Run: PYTHONPATH=. .venv/bin/python scripts/benchmark_realistic_pool.py
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import time

from sb3_contrib import RecurrentPPO
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from src.data.split import load_split
from src.train.train_l2 import make_l2_wrapped_env

L3_CHECKPOINT = "models/l3_frozen_backup/l3_executioner_v1_frozen.zip"
L3_VECNORM = "models/l3_frozen_backup/l3_vecnormalize_frozen.pkl"

SEED = 42
TOTAL_TIMESTEPS = 480
N_ENVS_SWEEP = (1, 8)


def make_env(rank: int, date_range: tuple[str, str]):
    def _init():
        import torch
        torch.set_num_threads(1)
        l3_model = RecurrentPPO.load(L3_CHECKPOINT, device="cpu")
        return make_l2_wrapped_env(date_range, 3000, 10, l3_model, L3_VECNORM, 50, False)

    return _init


def run_trial(n_envs: int, date_range: tuple[str, str]) -> dict:
    vec_env = SubprocVecEnv([make_env(i, date_range) for i in range(n_envs)])
    vec_env = VecMonitor(vec_env)
    model = SAC(
        "MlpPolicy", vec_env,
        buffer_size=500_000, gamma=0.995, batch_size=256, tau=0.005,
        learning_rate=3e-4, train_freq=1, gradient_steps=1, learning_starts=0,
        device="cuda", seed=SEED, verbose=0,
    )
    t0 = time.perf_counter()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=False)
    elapsed = time.perf_counter() - t0
    actual_steps = model.num_timesteps
    vec_env.close()
    return {"n_envs": n_envs, "elapsed": elapsed, "steps": actual_steps, "rate": actual_steps / elapsed}


def main() -> None:
    full_train = load_split("train")
    date_range = (full_train[0].isoformat(), full_train[-1].isoformat())
    print(f"REALISTIC pool: {len(full_train)} real days, date_range={date_range}, seed={SEED}\n")
    for n_envs in N_ENVS_SWEEP:
        r = run_trial(n_envs, date_range)
        print(f"n_envs={n_envs}: elapsed={r['elapsed']:.1f}s steps={r['steps']} rate={r['rate']:.3f}/s")


if __name__ == "__main__":
    main()
