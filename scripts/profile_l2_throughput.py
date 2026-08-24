"""Profiling script (NOT a production feature) -- decomposes L2 training wall-clock into
L3.predict() / env.step() / env.reset() / SAC gradient-update time, via monkeypatch-based
instrumentation of the REAL production code paths (make_l2_wrapped_env, the real frozen
checkpoint, real archive data) -- no source files modified, no reimplementation of the
wrapper's own logic. Short benchmark run only, per instruction -- not a training run.

Run: PYTHONPATH=. .venv/bin/python scripts/profile_l2_throughput.py
"""
from __future__ import annotations

import hashlib
import time

from sb3_contrib import RecurrentPPO
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from src.data.split import load_split
from src.train.train_l2 import make_l2_wrapped_env

L3_CHECKPOINT = "models/l3_frozen_backup/l3_executioner_v1_frozen.zip"
L3_VECNORM = "models/l3_frozen_backup/l3_vecnormalize_frozen.pkl"
EXPECTED_CHECKPOINT_SHA256 = "a5443e2a4c6c1d4427d4ce1cb83e65d622ea688d8953f5bf94b29e87fbcaa77d"
EXPECTED_VECNORM_SHA256 = "b459e17784c239be48069c47a7da6454610b4674a99e5d513d3ef0b616c182d8"

TOTAL_TIMESTEPS = 120  # short benchmark, per instruction -- not a training run


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    actual_ckpt_sha = sha256(L3_CHECKPOINT)
    actual_vecnorm_sha = sha256(L3_VECNORM)
    print(f"checkpoint sha256: {actual_ckpt_sha}")
    print(f"  matches handoff doc: {actual_ckpt_sha == EXPECTED_CHECKPOINT_SHA256}")
    print(f"vecnormalize sha256: {actual_vecnorm_sha}")
    print(f"  matches handoff doc: {actual_vecnorm_sha == EXPECTED_VECNORM_SHA256}")
    assert actual_ckpt_sha == EXPECTED_CHECKPOINT_SHA256, "checkpoint does not match l3_frozen_handoff.md"
    assert actual_vecnorm_sha == EXPECTED_VECNORM_SHA256, "vecnormalize does not match l3_frozen_handoff.md"

    timings = {"l3_predict": 0.0, "env_step": 0.0, "env_reset": 0.0, "sac_train": 0.0}
    counts = {"l3_predict": 0, "env_step": 0, "env_reset": 0, "sac_train": 0}

    l3_model = RecurrentPPO.load(L3_CHECKPOINT, device="cuda")
    orig_predict = l3_model.predict

    def timed_predict(*args, **kwargs):
        t0 = time.perf_counter()
        result = orig_predict(*args, **kwargs)
        timings["l3_predict"] += time.perf_counter() - t0
        counts["l3_predict"] += 1
        return result

    l3_model.predict = timed_predict

    train_dates = load_split("train")
    train_date_range = (train_dates[0].isoformat(), train_dates[-1].isoformat())
    print(f"train date_range: {train_date_range} ({len(train_dates)} real days)")

    wrapped_env = make_l2_wrapped_env(
        train_date_range, 3000, 10, l3_model, L3_VECNORM, 50, False,
    )

    inner_env = wrapped_env.env  # the base LOBExecutionEnv -- real matching-engine + real parquet I/O
    orig_env_step = inner_env.step
    orig_env_reset = inner_env.reset

    def timed_env_step(action):
        t0 = time.perf_counter()
        result = orig_env_step(action)
        timings["env_step"] += time.perf_counter() - t0
        counts["env_step"] += 1
        return result

    def timed_env_reset(**kwargs):
        t0 = time.perf_counter()
        result = orig_env_reset(**kwargs)
        timings["env_reset"] += time.perf_counter() - t0
        counts["env_reset"] += 1
        return result

    inner_env.step = timed_env_step
    inner_env.reset = timed_env_reset

    train_env = Monitor(wrapped_env)

    # learning_starts=0 (not the real default of 100): this profiling run is short by
    # design, and the point is per-gradient-step COST, not warmup behavior -- 0 maximizes
    # the sample of timed sac_train() calls within a short run. Every other hyperparameter
    # matches train_l2.py's real production values exactly.
    model = SAC(
        "MlpPolicy", train_env,
        buffer_size=500_000, gamma=0.995, batch_size=256, tau=0.005,
        learning_rate=3e-4, train_freq=1, gradient_steps=1, learning_starts=0,
        device="cuda", verbose=0,
    )
    orig_train = model.train

    def timed_train(*args, **kwargs):
        t0 = time.perf_counter()
        result = orig_train(*args, **kwargs)
        timings["sac_train"] += time.perf_counter() - t0
        counts["sac_train"] += 1
        return result

    model.train = timed_train

    t_wall_start = time.perf_counter()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=False)
    t_wall = time.perf_counter() - t_wall_start

    accounted = sum(timings.values())
    other = t_wall - accounted

    print(f"\n=== Profiling result: {TOTAL_TIMESTEPS} L2 decisions ===")
    print(f"total wall-clock: {t_wall:.3f}s ({TOTAL_TIMESTEPS / t_wall:.3f} decisions/sec)")
    print(f"{'component':<15} {'total_s':>10} {'pct':>8} {'n_calls':>10} {'per_call_ms':>14}")
    for key in ("env_reset", "l3_predict", "env_step", "sac_train"):
        pct = 100.0 * timings[key] / t_wall
        per_call_ms = 1000.0 * timings[key] / counts[key] if counts[key] else float("nan")
        print(f"{key:<15} {timings[key]:>10.3f} {pct:>7.1f}% {counts[key]:>10} {per_call_ms:>13.3f}ms")
    print(f"{'other/overhead':<15} {other:>10.3f} {100.0*other/t_wall:>7.1f}%")


if __name__ == "__main__":
    main()
