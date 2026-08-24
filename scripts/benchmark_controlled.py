"""Controlled parallel-env benchmark (NOT a production implementation) -- fixes the two
variance sources identified in the prior round's noisy results (unfixed seed, unfixed
date_range): every configuration and trial below samples from the SAME small, fixed pool
of real market days, and uses the SAME fixed seed (so SubprocVecEnv distributes seed+idx
identically across repeated trials of the same n_envs value, matching train_l3.py's own
established seeding convention). Thread-capping (torch.set_num_threads(1) + OMP/MKL env
vars) is applied from the start, since the prior round established it's mandatory, not
optional tuning.

Sweeps n_envs = 1, 2, 4, 8, 3 repeated trials each, reporting mean/stdev (not a single
number), real RSS/VRAM measured via /proc/<pid>/status and nvidia-smi (not estimated).

Run: PYTHONPATH=. .venv/bin/python scripts/benchmark_controlled.py
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import statistics
import subprocess
import time

from sb3_contrib import RecurrentPPO
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from src.data.split import load_split
from src.train.train_l2 import make_l2_wrapped_env

L3_CHECKPOINT = "models/l3_frozen_backup/l3_executioner_v1_frozen.zip"
L3_VECNORM = "models/l3_frozen_backup/l3_vecnormalize_frozen.pkl"

SEED = 42
# 480: large enough that even the slowest expected configuration (~4.2 dec/sec, the prior
# round's single-env baseline) takes ~114s (~27 episodes at ~18 decisions/episode),
# keeping one-time per-trial overhead (subprocess spawn + first cold-cache reset, a few
# seconds, confirmed last round to overlap well across workers) a small fraction of trial
# wall-clock -- without a separate reset/learn split this round (see module docstring: a
# manual pre-emptive vec_env.reset() risks double-counting against SB3's own internal
# first reset inside learn(), so this round measures the full learn() call as one
# unambiguous block instead). Divisible by 1/2/4/8 for clean per-worker step counts.
TOTAL_TIMESTEPS = 480
N_TRIALS = 3
N_ENVS_SWEEP = (1, 2, 4, 8)


def _fixed_date_range() -> tuple[str, str]:
    """First 10 REAL, gap-free dates from the actual persisted train split (not an
    arbitrary calendar range that might silently include known gap dates)."""
    train_dates = load_split("train")
    first_ten = train_dates[:10]
    return (first_ten[0].isoformat(), first_ten[-1].isoformat())


def make_env(rank: int, date_range: tuple[str, str]):
    def _init():
        import torch
        torch.set_num_threads(1)
        l3_model = RecurrentPPO.load(L3_CHECKPOINT, device="cpu")
        return make_l2_wrapped_env(date_range, 3000, 10, l3_model, L3_VECNORM, 50, False)

    return _init


def _read_rss_mb(pid: int) -> float:
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except (FileNotFoundError, ProcessLookupError):
        return 0.0
    return 0.0


def _read_vram_mb(pids: set[int]) -> float:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        total = 0.0
        for line in out.stdout.strip().splitlines():
            if not line.strip():
                continue
            pid_str, mem_str = (x.strip() for x in line.split(","))
            if int(pid_str) in pids:
                total += float(mem_str)
        return total
    except Exception:
        return float("nan")


def run_trial(n_envs: int, trial_idx: int, date_range: tuple[str, str]) -> dict:
    vec_env = SubprocVecEnv([make_env(i, date_range) for i in range(n_envs)])
    vec_env = VecMonitor(vec_env)

    model = SAC(
        "MlpPolicy", vec_env,
        buffer_size=500_000, gamma=0.995, batch_size=256, tau=0.005,
        learning_rate=3e-4, train_freq=1, gradient_steps=1, learning_starts=0,
        device="cuda", seed=SEED, verbose=0,
    )
    # SAC's own _setup_model() already called vec_env.seed(SEED) as a side effect of
    # passing seed= above (confirmed pattern, see train_l3.py's own reconciliation of
    # SB3 2.3.2's set_random_seed() -> env.seed() -> SubprocVecEnv distributing seed+idx
    # per worker) -- not calling it again here to avoid double-seeding ambiguity.

    worker_pids = {p.pid for p in vec_env.processes}
    all_pids = worker_pids | {os.getpid()}

    t0 = time.perf_counter()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=False)
    elapsed = time.perf_counter() - t0
    actual_steps = model.num_timesteps

    rss_mb = sum(_read_rss_mb(pid) for pid in all_pids)
    vram_mb = _read_vram_mb(all_pids)

    vec_env.close()
    rate = actual_steps / elapsed
    return {
        "n_envs": n_envs, "trial": trial_idx, "elapsed": elapsed,
        "actual_steps": actual_steps, "rate": rate, "rss_mb": rss_mb, "vram_mb": vram_mb,
    }


def main() -> None:
    date_range = _fixed_date_range()
    print(f"Fixed date_range for this ENTIRE benchmark (all configs, all trials): {date_range}")
    print(f"seed={SEED}, total_timesteps/trial={TOTAL_TIMESTEPS}, n_trials={N_TRIALS}")
    print("OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, torch.set_num_threads(1) per worker\n")

    results = []
    for n_envs in N_ENVS_SWEEP:
        print(f"=== n_envs={n_envs} ===")
        for trial in range(N_TRIALS):
            r = run_trial(n_envs, trial, date_range)
            results.append(r)
            print(
                f"  trial {trial}: elapsed={r['elapsed']:.3f}s steps={r['actual_steps']} "
                f"rate={r['rate']:.3f}/s RSS={r['rss_mb']:.0f}MB VRAM={r['vram_mb']:.0f}MB"
            )
        rates = [r["rate"] for r in results if r["n_envs"] == n_envs]
        mean_rate = statistics.mean(rates)
        stdev_rate = statistics.stdev(rates) if len(rates) > 1 else 0.0
        cov = 100 * stdev_rate / mean_rate if mean_rate else float("nan")
        print(f"  mean rate: {mean_rate:.3f}/s (stdev {stdev_rate:.3f}, CoV {cov:.1f}%)\n")

    print("=== Summary ===")
    header = f"{'n_envs':>7} {'mean_rate':>11} {'stdev':>8} {'CoV%':>6} {'speedup':>9} {'efficiency':>11} {'RSS_MB':>8} {'VRAM_MB':>9}"
    print(header)
    baseline_mean = None
    for n_envs in N_ENVS_SWEEP:
        subset = [r for r in results if r["n_envs"] == n_envs]
        rates = [r["rate"] for r in subset]
        mean_rate = statistics.mean(rates)
        stdev_rate = statistics.stdev(rates) if len(rates) > 1 else 0.0
        cov = 100 * stdev_rate / mean_rate if mean_rate else float("nan")
        if baseline_mean is None:
            baseline_mean = mean_rate
        speedup = mean_rate / baseline_mean
        efficiency = speedup / n_envs * 100
        mean_rss = statistics.mean(r["rss_mb"] for r in subset)
        mean_vram = statistics.mean(r["vram_mb"] for r in subset)
        print(
            f"{n_envs:>7} {mean_rate:>11.3f} {stdev_rate:>8.3f} {cov:>6.1f} "
            f"{speedup:>8.2f}x {efficiency:>10.1f}% {mean_rss:>8.0f} {mean_vram:>9.0f}"
        )


if __name__ == "__main__":
    main()
