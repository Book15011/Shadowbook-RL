"""Mechanics test for scripts/eval_l2_n500.py -- synthetic data, tiny/untrained models,
CPU only, n=3. Proves the harness's own machinery (both arms' episode loops, the
paired-report statistics, the pre-registered-bar check) runs end-to-end without error;
does NOT assert anything about untrained-model IS/fill_ratio values themselves (there is
no reason a randomly-initialized SAC policy should beat or lose to anything in a
meaningful way -- this is a mechanics test, not a results test). The real n=500
evaluation against real checkpoints is explicitly NOT run here, per instruction (a live
24h L2 training run must not be disturbed by real-data/real-checkpoint I/O).
"""
from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
import pytest
from sb3_contrib import RecurrentPPO
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from scripts.eval_l2_n500 import (
    make_l2_policy_action_fn,
    paired_report,
    run_arm,
    run_wrapped_episode,
)
from scripts.phase2a_sanity_suite import TWAPPolicy, run_episode
from src.envs.lob_execution_env import LOBExecutionEnv
from src.envs.wrappers import FrozenL3Wrapper

HORIZON_TICKS = 20
LOOKBACK_TICKS = 2
TICKS_PER_L2_DECISION = 5


def _write_synthetic_day(path, n_rows: int, base_price: float, ts_start: int) -> None:
    best_bid = base_price - 0.05
    best_ask = base_price + 0.05
    bids = json.dumps([[best_bid, 10.0], [best_bid - 0.1, 5.0]])
    asks = json.dumps([[best_ask, 10.0], [best_ask + 0.1, 5.0]])
    rows = [
        {"ts": ts_start + i, "best_bid": best_bid, "best_ask": best_ask, "mid_price": base_price,
         "spread": best_ask - best_bid, "bids": bids, "asks": asks}
        for i in range(n_rows)
    ]
    pd.DataFrame(rows).to_parquet(path, index=False)


def _build_base_env(tmp_path) -> LOBExecutionEnv:
    data_dir = tmp_path / "BTCUSDT"
    data_dir.mkdir(exist_ok=True)
    _write_synthetic_day(data_dir / "l2-BTCUSDT-2024-01-01.parquet", n_rows=200, base_price=100.0, ts_start=1_000_000)
    return LOBExecutionEnv(
        data_dir=data_dir, horizon_ticks=HORIZON_TICKS, lookback_ticks=LOOKBACK_TICKS,
        tick_interval_s=1.0, date_range=("2024-01-01", "2024-01-01"),
    )


def _build_tiny_recurrent_ppo(env: LOBExecutionEnv) -> RecurrentPPO:
    venv = DummyVecEnv([lambda: env])
    return RecurrentPPO(
        "MlpLstmPolicy", venv,
        policy_kwargs=dict(lstm_hidden_size=8, n_lstm_layers=1, net_arch=dict(pi=[8], vf=[8])),
        n_steps=8, batch_size=8, device="cpu", seed=0,
    )


def _write_vecnormalize(path, env) -> str:
    venv = DummyVecEnv([lambda: env])
    vn = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=5.0)
    with open(path, "wb") as f:
        pickle.dump(vn, f)
    return str(path)


@pytest.fixture
def synthetic_setup(tmp_path):
    """Builds everything a real invocation would load from disk: a tiny frozen 'L3'
    checkpoint + VecNormalize, a FrozenL3Wrapper-wrapped env (to get L2's real obs/
    action space), a tiny/untrained SAC 'L2' model + its own VecNormalize, and a base
    (unwrapped) env for the pure-TWAP arm -- all on synthetic data, all CPU, all tiny."""
    base_env_for_l3 = _build_base_env(tmp_path)
    l3_model = _build_tiny_recurrent_ppo(base_env_for_l3)
    l3_vecnorm_path = tmp_path / "l3_vecnormalize.pkl"
    _write_vecnormalize(l3_vecnorm_path, base_env_for_l3)

    base_env_for_wrapper = _build_base_env(tmp_path)
    wrapped_env = FrozenL3Wrapper(
        base_env_for_wrapper, l3_model, str(l3_vecnorm_path),
        ticks_per_l2_decision=TICKS_PER_L2_DECISION, l3_deterministic=True,
    )

    l2_venv = DummyVecEnv([lambda: wrapped_env])
    l2_model = SAC("MlpPolicy", l2_venv, policy_kwargs=dict(net_arch=[8]), device="cpu", seed=0, learning_starts=0)
    l2_vecnorm = VecNormalize(l2_venv, norm_obs=True, norm_reward=True, clip_obs=5.0)
    l2_vecnorm.training = False

    base_env_for_twap = _build_base_env(tmp_path)

    return {
        "wrapped_env": wrapped_env, "l2_model": l2_model, "l2_vecnorm": l2_vecnorm,
        "base_env_for_twap": base_env_for_twap,
    }


def test_run_wrapped_episode_l2_policy_arm(synthetic_setup):
    action_fn = make_l2_policy_action_fn(synthetic_setup["l2_model"], synthetic_setup["l2_vecnorm"])
    max_decisions = HORIZON_TICKS // TICKS_PER_L2_DECISION + 1
    result = run_wrapped_episode(synthetic_setup["wrapped_env"], seed=5_000_000, action_fn=action_fn, max_decisions=max_decisions)
    assert "is_result" in result
    assert np.isfinite(result["is_result"].is_total_bps)
    assert 0.0 <= result["is_result"].fill_ratio <= 1.0


def test_run_wrapped_episode_twap_passthrough_arm(synthetic_setup):
    passthrough_action = np.array([1.0, 0.5], dtype=np.float32)
    max_decisions = HORIZON_TICKS // TICKS_PER_L2_DECISION + 1
    result = run_wrapped_episode(
        synthetic_setup["wrapped_env"], seed=5_000_000, action_fn=lambda obs: passthrough_action,
        max_decisions=max_decisions,
    )
    assert np.isfinite(result["is_result"].is_total_bps)


def test_run_arm_and_paired_report_end_to_end(synthetic_setup):
    n = 3
    seeds = [5_000_000 + i for i in range(n)]
    max_decisions = HORIZON_TICKS // TICKS_PER_L2_DECISION + 1
    l2_action_fn = make_l2_policy_action_fn(synthetic_setup["l2_model"], synthetic_setup["l2_vecnorm"])
    passthrough_action = np.array([1.0, 0.5], dtype=np.float32)

    arm1 = run_arm(
        "L2 (untrained)", seeds,
        lambda s: run_wrapped_episode(synthetic_setup["wrapped_env"], s, l2_action_fn, max_decisions),
    )
    arm2 = run_arm(
        "TWAP-passthrough", seeds,
        lambda s: run_wrapped_episode(synthetic_setup["wrapped_env"], s, lambda obs: passthrough_action, max_decisions),
    )
    twap_policy = TWAPPolicy(n_slices=4)
    arm3 = run_arm(
        "Pure TWAP", seeds,
        lambda s: run_episode(synthetic_setup["base_env_for_twap"], twap_policy, seed=s, horizon_ticks=HORIZON_TICKS),
    )

    for arm in (arm1, arm2, arm3):
        assert len(arm["is_bps"]) == n
        assert np.all(np.isfinite(arm["is_bps"]))

    cmp_1v2 = paired_report(arm2["label"], arm2, arm1["label"], arm1)
    cmp_1v3 = paired_report(arm3["label"], arm3, arm1["label"], arm1)
    for cmp in (cmp_1v2, cmp_1v3):
        assert "d_z" in cmp  # effect size present, not just p-values
        assert "t_p" in cmp and "w_p" in cmp
        assert np.isfinite(cmp["mean_diff"])


def test_paired_report_effect_size_matches_hand_computation():
    r_a = {"is_bps": np.array([1.0, 2.0, 3.0, 4.0, 5.0])}
    r_b = {"is_bps": np.array([2.0, 2.0, 4.0, 6.0, 5.0])}
    result = paired_report("A", r_a, "B", r_b)
    diff = r_b["is_bps"] - r_a["is_bps"]
    expected_d_z = diff.mean() / diff.std()
    assert result["d_z"] == pytest.approx(expected_d_z)
    assert result["mean_diff"] == pytest.approx(diff.mean())
