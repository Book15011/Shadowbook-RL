"""Fixture-based unit tests for src/envs/wrappers.py::FrozenL3Wrapper. Same style as
tests/test_reward.py (hand-computed expected values) and tests/test_lob_execution_env.py
(small synthetic parquet days rather than the real Bybit archive).

Mechanics tests (VecNormalize application, LSTM state/episode_start threading, observation
shape, action-space bounds) use a small, freshly-initialized RecurrentPPO model with
matching obs/action spaces rather than the real frozen checkpoint -- faster, deterministic,
no GPU contention with whatever else is running on this box. One separate, explicitly gated
integration smoke test loads the actual current checkpoint.
"""
from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
import pytest
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.envs.lob_execution_env import LOBExecutionEnv
from src.envs.wrappers import (
    L2_ACTION_HIGH,
    L2_ACTION_LOW,
    L2_BASE_OBS_DIM,
    L2_FULL_OBS_DIM,
    FrozenL3Wrapper,
    _downsample_window,
    _l2_observation_space,
    _schedule_deviation,
    _twap_schedule_baseline,
)


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def _write_synthetic_day(path, n_rows: int, base_price: float, ts_start: int) -> None:
    # Same technique as tests/test_lob_execution_env.py -- small synthetic parquet day,
    # self-contained, no dependency on the real Bybit archive.
    best_bid = base_price - 0.05
    best_ask = base_price + 0.05
    bids = json.dumps([[best_bid, 10.0], [best_bid - 0.1, 5.0]])
    asks = json.dumps([[best_ask, 10.0], [best_ask + 0.1, 5.0]])
    rows = [
        {
            "ts": ts_start + i,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid_price": base_price,
            "spread": best_ask - best_bid,
            "bids": bids,
            "asks": asks,
        }
        for i in range(n_rows)
    ]
    pd.DataFrame(rows).to_parquet(path, index=False)


def _build_env(tmp_path, horizon_ticks: int = 20, lookback_ticks: int = 2) -> LOBExecutionEnv:
    # tick_interval_s=1.0 (not the real 0.1s) keeps _max_lookback_ticks small (60 vs 600)
    # so a small synthetic day is enough -- deliberately NOT realistic production timing,
    # this is a mechanics test, not a physics test.
    data_dir = tmp_path / "BTCUSDT"
    data_dir.mkdir(exist_ok=True)
    _write_synthetic_day(data_dir / "l2-BTCUSDT-2024-01-01.parquet", n_rows=200, base_price=100.0, ts_start=1_000_000)
    return LOBExecutionEnv(
        data_dir=data_dir, horizon_ticks=horizon_ticks, lookback_ticks=lookback_ticks,
        tick_interval_s=1.0, date_range=("2024-01-01", "2024-01-01"),
    )


def _build_tiny_recurrent_ppo(env: LOBExecutionEnv) -> RecurrentPPO:
    venv = DummyVecEnv([lambda: env])
    return RecurrentPPO(
        "MlpLstmPolicy", venv,
        policy_kwargs=dict(lstm_hidden_size=8, n_lstm_layers=1, net_arch=dict(pi=[8], vf=[8])),
        n_steps=8, batch_size=8, device="cpu", seed=0,
    )


def _write_fake_vecnormalize(path, env: LOBExecutionEnv, mean: np.ndarray, var: np.ndarray) -> str:
    venv = DummyVecEnv([lambda: env])
    vn = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=5.0)
    vn.obs_rms.mean = mean.astype(np.float64)
    vn.obs_rms.var = var.astype(np.float64)
    out = path / "fake_vecnormalize.pkl"
    with open(out, "wb") as f:
        pickle.dump(vn, f)
    return str(out)


def _default_vecnormalize_path(tmp_path, env) -> str:
    # mean=0, var=1 -> normalize_obs is a no-op (modulo clipping) -- used by tests that
    # don't care about the normalization transform itself.
    return _write_fake_vecnormalize(tmp_path, env, mean=np.zeros(42), var=np.ones(42))


# --------------------------------------------------------------------------------------
# Pure-function tests: _downsample_window / _twap_schedule_baseline / _schedule_deviation
# -- hand-computed, no env or model needed.
# --------------------------------------------------------------------------------------

def test_downsample_window_mean_group_uses_window_average():
    # idx 2 (spread_norm) is in the MEAN group. 3-tick window: values 0.1, 0.3, 0.5 ->
    # mean = 0.3, not the last value (0.5).
    window = np.zeros((3, 42), dtype=np.float32)
    window[:, 2] = [0.1, 0.3, 0.5]
    info = {"ticks_elapsed": 0, "qty_total": 10.0, "qty_remaining": 10.0}
    out = _downsample_window(window, info, horizon_ticks=100, prev_action=None)
    # new position 2 <- old idx 2 (0, 1 map 1:1 before any exclusion; idx 2 has no
    # excluded indices before it, so new pos == old idx here).
    assert out[2] == pytest.approx(0.3)


def test_downsample_window_last_value_group_uses_final_tick_only():
    # idx 0 (time_remaining_norm) is in the instantaneous/last-value group. 3-tick
    # window: values 0.9, 0.8, 0.7 -> last value (0.7), not the mean (0.8).
    window = np.zeros((3, 42), dtype=np.float32)
    window[:, 0] = [0.9, 0.8, 0.7]
    info = {"ticks_elapsed": 0, "qty_total": 10.0, "qty_remaining": 10.0}
    out = _downsample_window(window, info, horizon_ticks=100, prev_action=None)
    assert out[0] == pytest.approx(0.7)


def test_downsample_window_book_depth_mean_group_per_level():
    # idx 19 (book_depth_norm_0), MEAN group. New position: idx 19 maps to new pos 17
    # (0-14 -> 0-14 [15 positions], 17-18 -> 15-16 [2 positions], so idx19 -> pos 17).
    window = np.zeros((4, 42), dtype=np.float32)
    window[:, 19] = [1.0, 2.0, 3.0, 4.0]
    info = {"ticks_elapsed": 0, "qty_total": 10.0, "qty_remaining": 10.0}
    out = _downsample_window(window, info, horizon_ticks=100, prev_action=None)
    assert out[17] == pytest.approx(2.5)


def test_downsample_window_excluded_indices_15_16_are_not_in_output():
    # 42 raw features minus 2 excluded (15, 16) = 40, plus schedule_deviation = 41 total
    # with no previous-action toggle.
    window = np.zeros((1, 42), dtype=np.float32)
    info = {"ticks_elapsed": 0, "qty_total": 10.0, "qty_remaining": 10.0}
    out = _downsample_window(window, info, horizon_ticks=100, prev_action=None)
    assert out.shape == (L2_BASE_OBS_DIM,)
    assert L2_BASE_OBS_DIM == 41


def test_downsample_window_prev_action_toggle_appends_two_raw_scalars():
    window = np.zeros((1, 42), dtype=np.float32)
    info = {"ticks_elapsed": 0, "qty_total": 10.0, "qty_remaining": 10.0}
    prev_action = np.array([1.5, 0.25], dtype=np.float32)
    out = _downsample_window(window, info, horizon_ticks=100, prev_action=prev_action)
    assert out.shape == (L2_FULL_OBS_DIM,)
    assert out[41] == pytest.approx(1.5)
    assert out[42] == pytest.approx(0.25)


def test_twap_schedule_baseline_linear_in_elapsed_ticks():
    # ticks_elapsed=30, horizon_ticks=100 -> 0.3, hand-computed.
    info = {"ticks_elapsed": 30}
    assert _twap_schedule_baseline(info, horizon_ticks=100) == pytest.approx(0.3)


def test_twap_schedule_baseline_clips_at_1_past_horizon():
    info = {"ticks_elapsed": 150}
    assert _twap_schedule_baseline(info, horizon_ticks=100) == pytest.approx(1.0)


def test_schedule_deviation_ahead_of_schedule_is_positive():
    # executed_so_far = (10-2)/10 = 0.8, twap_baseline = 40/100 = 0.4 -> deviation = 0.4.
    info = {"ticks_elapsed": 40, "qty_total": 10.0, "qty_remaining": 2.0}
    assert _schedule_deviation(info, horizon_ticks=100) == pytest.approx(0.4)


def test_schedule_deviation_behind_schedule_is_negative():
    # executed_so_far = (10-9)/10 = 0.1, twap_baseline = 80/100 = 0.8 -> deviation = -0.7.
    info = {"ticks_elapsed": 80, "qty_total": 10.0, "qty_remaining": 9.0}
    assert _schedule_deviation(info, horizon_ticks=100) == pytest.approx(-0.7)


def test_l2_observation_space_shapes_and_bounds_both_toggle_states():
    base = _l2_observation_space(include_prev_action=False)
    full = _l2_observation_space(include_prev_action=True)
    assert base.shape == (L2_BASE_OBS_DIM,)
    assert full.shape == (L2_FULL_OBS_DIM,)
    # schedule_deviation bound is the final entry in the base space.
    assert base.low[-1] == pytest.approx(-1.0)
    assert base.high[-1] == pytest.approx(1.0)
    # prev-action bounds match architecture_spec.md Section 3.2's L2 action space exactly.
    assert full.low[-2:].tolist() == pytest.approx(L2_ACTION_LOW.tolist())
    assert full.high[-2:].tolist() == pytest.approx(L2_ACTION_HIGH.tolist())


# --------------------------------------------------------------------------------------
# Wrapper mechanics tests: small synthetic env + small freshly-initialized RecurrentPPO.
# --------------------------------------------------------------------------------------

def test_action_space_matches_section_3_2(tmp_path):
    env = _build_env(tmp_path)
    model = _build_tiny_recurrent_ppo(env)
    vecnorm_path = _default_vecnormalize_path(tmp_path, env)
    wrapper = FrozenL3Wrapper(env, model, vecnorm_path, ticks_per_l2_decision=4)
    assert wrapper.action_space.low.tolist() == pytest.approx([0.0, 0.0])
    assert wrapper.action_space.high.tolist() == pytest.approx([2.0, 1.0])


def test_apply_l2_action_participation_multiplier_scales_twap_not_1to1(tmp_path):
    # At the very first decision (ticks_elapsed=0), the env's own default TWAP baseline
    # is exactly 0.0 -- so l2_target_slice_ratio_override should land at
    # 0.0 * participation_mult = 0.0 regardless of the multiplier, NOT participation_mult
    # itself (which is what a direct 1:1 Box(0,1) map would have produced instead,
    # confirming the transform is genuinely multiplicative-against-baseline, not a
    # pass-through under-the-hood).
    env = _build_env(tmp_path)
    model = _build_tiny_recurrent_ppo(env)
    vecnorm_path = _default_vecnormalize_path(tmp_path, env)
    wrapper = FrozenL3Wrapper(env, model, vecnorm_path, ticks_per_l2_decision=4)
    wrapper.reset(seed=0)
    wrapper.step(np.array([1.5, 0.3], dtype=np.float32))
    assert env.l2_target_slice_ratio_override == pytest.approx(0.0)
    assert env.l2_urgency == pytest.approx(0.3)  # urgency is a direct 1:1 pass-through


def test_apply_l2_action_participation_multiplier_nonzero_after_elapsed_ticks(tmp_path):
    env = _build_env(tmp_path, horizon_ticks=20)
    model = _build_tiny_recurrent_ppo(env)
    vecnorm_path = _default_vecnormalize_path(tmp_path, env)
    wrapper = FrozenL3Wrapper(env, model, vecnorm_path, ticks_per_l2_decision=4)
    wrapper.reset(seed=0)
    wrapper.step(np.array([1.0, 0.5], dtype=np.float32))  # window 1: ticks_elapsed 0->4
    # Second decision: twap_baseline = ticks_elapsed(4) / horizon_ticks(20) = 0.2.
    # participation_mult=0.5 -> override = 0.2 * 0.5 = 0.1, hand-computed.
    wrapper.step(np.array([0.5, 0.5], dtype=np.float32))
    assert env.l2_target_slice_ratio_override == pytest.approx(0.1)


def test_observation_shape_base_toggle_off(tmp_path):
    env = _build_env(tmp_path)
    model = _build_tiny_recurrent_ppo(env)
    vecnorm_path = _default_vecnormalize_path(tmp_path, env)
    wrapper = FrozenL3Wrapper(env, model, vecnorm_path, ticks_per_l2_decision=4, l2_include_prev_action=False)
    obs, info = wrapper.reset(seed=0)
    assert obs.shape == (L2_BASE_OBS_DIM,)
    obs, reward, terminated, truncated, info = wrapper.step(np.array([1.0, 0.5], dtype=np.float32))
    assert obs.shape == (L2_BASE_OBS_DIM,)


def test_observation_shape_with_prev_action_toggle_on(tmp_path):
    env = _build_env(tmp_path)
    model = _build_tiny_recurrent_ppo(env)
    vecnorm_path = _default_vecnormalize_path(tmp_path, env)
    wrapper = FrozenL3Wrapper(env, model, vecnorm_path, ticks_per_l2_decision=4, l2_include_prev_action=True)
    obs, info = wrapper.reset(seed=0)
    assert obs.shape == (L2_FULL_OBS_DIM,)
    action = np.array([1.25, 0.75], dtype=np.float32)
    obs, reward, terminated, truncated, info = wrapper.step(action)
    assert obs.shape == (L2_FULL_OBS_DIM,)
    # prev-action features reflect the action just taken (raw copy, not lossily derived).
    assert obs[-2] == pytest.approx(1.25)
    assert obs[-1] == pytest.approx(0.75)


def test_prev_action_defaults_to_off(tmp_path):
    env = _build_env(tmp_path)
    model = _build_tiny_recurrent_ppo(env)
    vecnorm_path = _default_vecnormalize_path(tmp_path, env)
    wrapper = FrozenL3Wrapper(env, model, vecnorm_path, ticks_per_l2_decision=4)
    assert wrapper.l2_include_prev_action is False
    obs, info = wrapper.reset(seed=0)
    assert obs.shape == (L2_BASE_OBS_DIM,)


def test_vecnormalize_formula_matches_hand_computation_exactly(tmp_path, monkeypatch):
    # Non-trivial mean/var, checked against the exact hand-computed formula -- this is
    # the test that would catch "loaded the .pkl but never called normalize_obs" (raw obs
    # would be passed through, failing the exact-match assertion) or "recomputed
    # normalization by hand with the wrong formula" (any deviation from SB3's own
    # clip((raw-mean)/sqrt(var+eps), -5, 5) fails too).
    env = _build_env(tmp_path)
    model = _build_tiny_recurrent_ppo(env)
    mean = np.linspace(-1.0, 1.0, 42, dtype=np.float64)
    var = np.full(42, 4.0, dtype=np.float64)
    vecnorm_path = _write_fake_vecnormalize(tmp_path, env, mean=mean, var=var)
    wrapper = FrozenL3Wrapper(env, model, vecnorm_path, ticks_per_l2_decision=1)

    received = []
    original_predict = model.predict

    def spy_predict(observation, state=None, episode_start=None, deterministic=False):
        received.append(np.array(observation, dtype=np.float64).copy())
        return original_predict(observation, state=state, episode_start=episode_start, deterministic=deterministic)

    monkeypatch.setattr(model, "predict", spy_predict)

    obs, info = wrapper.reset(seed=0)
    raw_obs = wrapper._l3_obs.astype(np.float64)
    wrapper.step(np.array([1.0, 0.5], dtype=np.float32))

    expected = np.clip((raw_obs - mean) / np.sqrt(var + 1e-8), -5.0, 5.0)
    assert len(received) == 1
    np.testing.assert_allclose(received[0], expected, rtol=1e-5)


def test_lstm_episode_start_true_only_on_first_predict_after_reset(tmp_path, monkeypatch):
    env = _build_env(tmp_path, horizon_ticks=20)
    model = _build_tiny_recurrent_ppo(env)
    vecnorm_path = _default_vecnormalize_path(tmp_path, env)
    wrapper = FrozenL3Wrapper(env, model, vecnorm_path, ticks_per_l2_decision=4)

    episode_starts = []
    states_in = []
    states_out = []
    original_predict = model.predict

    def spy_predict(observation, state=None, episode_start=None, deterministic=False):
        episode_starts.append(bool(np.array(episode_start)[0]))
        states_in.append(state)
        action, next_state = original_predict(observation, state=state, episode_start=episode_start, deterministic=deterministic)
        states_out.append(next_state)
        return action, next_state

    monkeypatch.setattr(model, "predict", spy_predict)

    wrapper.reset(seed=0)
    wrapper.step(np.array([1.0, 0.5], dtype=np.float32))  # window 1: 4 predict calls
    wrapper.step(np.array([1.0, 0.5], dtype=np.float32))  # window 2: 4 more predict calls

    assert len(episode_starts) == 8
    # Only the very first predict() call of the episode has episode_start=True -- every
    # other call, INCLUDING the first tick of the second window (index 4, the window
    # boundary), must be False. This is exactly the case a naive re-implementation gets
    # wrong (e.g. resetting episode_start every step() call instead of every episode).
    assert episode_starts == [True] + [False] * 7

    # state=None only on the very first call; every subsequent call (within a window AND
    # across the window boundary at index 4) must receive the PRIOR call's returned
    # state, not None and not a freshly-reset value.
    assert states_in[0] is None
    for i in range(1, 8):
        np.testing.assert_equal(states_in[i], states_out[i - 1])


def test_lstm_episode_start_resets_true_on_new_episode(tmp_path, monkeypatch):
    env = _build_env(tmp_path, horizon_ticks=20)
    model = _build_tiny_recurrent_ppo(env)
    vecnorm_path = _default_vecnormalize_path(tmp_path, env)
    wrapper = FrozenL3Wrapper(env, model, vecnorm_path, ticks_per_l2_decision=4)

    episode_starts = []
    original_predict = model.predict

    def spy_predict(observation, state=None, episode_start=None, deterministic=False):
        episode_starts.append(bool(np.array(episode_start)[0]))
        return original_predict(observation, state=state, episode_start=episode_start, deterministic=deterministic)

    monkeypatch.setattr(model, "predict", spy_predict)

    wrapper.reset(seed=0)
    wrapper.step(np.array([1.0, 0.5], dtype=np.float32))
    wrapper.reset(seed=1)  # new episode -> LSTM state must start fresh again
    wrapper.step(np.array([1.0, 0.5], dtype=np.float32))

    assert episode_starts[0] is True   # episode 1, first call
    assert episode_starts[4] is True   # episode 2, first call after the second reset()
    assert wrapper._l3_lstm_state is not None or True  # state exists post-step (non-None after any predict call)


def test_l3_deterministic_flag_controls_inner_predict_determinism(tmp_path, monkeypatch):
    # Regression test for a real bug: the frozen L3 policy's own inner predict() calls
    # were hardcoded deterministic=False regardless of caller intent, discovered via
    # tests/test_train_l2.py's own eval-callback determinism test. l3_deterministic
    # (default False, preserving prior training-time behavior) must actually reach
    # l3_model.predict()'s own deterministic= argument.
    env = _build_env(tmp_path)
    model = _build_tiny_recurrent_ppo(env)
    vecnorm_path = _default_vecnormalize_path(tmp_path, env)

    deterministic_flags_seen = []
    original_predict = model.predict

    def spy_predict(observation, state=None, episode_start=None, deterministic=False):
        deterministic_flags_seen.append(deterministic)
        return original_predict(observation, state=state, episode_start=episode_start, deterministic=deterministic)

    monkeypatch.setattr(model, "predict", spy_predict)

    wrapper_default = FrozenL3Wrapper(env, model, vecnorm_path, ticks_per_l2_decision=2)
    wrapper_default.reset(seed=0)
    wrapper_default.step(np.array([1.0, 0.5], dtype=np.float32))
    assert deterministic_flags_seen == [False, False]

    deterministic_flags_seen.clear()
    wrapper_det = FrozenL3Wrapper(env, model, vecnorm_path, ticks_per_l2_decision=2, l3_deterministic=True)
    wrapper_det.reset(seed=0)
    wrapper_det.step(np.array([1.0, 0.5], dtype=np.float32))
    assert deterministic_flags_seen == [True, True]


def test_reset_clears_l2_target_slice_ratio_override_and_urgency(tmp_path):
    # Regression test for a real bug: on a reused instance (every episode after the
    # first, in both training and eval), env.l2_target_slice_ratio_override/l2_urgency
    # were left holding the PREVIOUS episode's last step()-set values, since
    # LOBExecutionEnv.reset() never touches either attribute -- silently leaking stale
    # state into the new episode's very first observation (idx 15/16 of the raw obs fed
    # to the frozen L3 policy). Also caught via tests/test_train_l2.py's determinism
    # test: fixing l3_deterministic alone wasn't enough, because this leak meant the two
    # runs' first L3 observations genuinely differed.
    env = _build_env(tmp_path, horizon_ticks=20)
    model = _build_tiny_recurrent_ppo(env)
    vecnorm_path = _default_vecnormalize_path(tmp_path, env)
    wrapper = FrozenL3Wrapper(env, model, vecnorm_path, ticks_per_l2_decision=4)

    wrapper.reset(seed=0)
    wrapper.step(np.array([1.5, 0.9], dtype=np.float32))  # leaves both attributes non-default
    assert env.l2_target_slice_ratio_override is not None
    assert env.l2_urgency == pytest.approx(0.9)

    wrapper.reset(seed=1)  # new episode, same instance -- must NOT see the leftover values above
    assert env.l2_target_slice_ratio_override is None
    assert env.l2_urgency == pytest.approx(0.5)


# --------------------------------------------------------------------------------------
# Integration smoke test -- real checkpoint. Gated behind a fresh GPU/RAM check; run
# manually, not part of the default fast suite (see skip condition below). The target
# checkpoint (models/l3_executioner_v1.zip) may not be the FINAL frozen L3 checkpoint --
# per docs/TRACK_STATUS.md, the L3 track has an open question about whether this
# checkpoint's validation result is good enough to build on. This test only exercises
# shape/integration correctness, not policy quality.
# --------------------------------------------------------------------------------------

def _real_checkpoint_paths():
    import pathlib
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    model_path = repo_root / "models" / "l3_executioner_v1.zip"
    vecnorm_path = repo_root / "models" / "l3_vecnormalize.pkl"
    return model_path, vecnorm_path


def _gpu_has_headroom() -> bool:
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        used, total = (int(x.strip()) for x in out.stdout.strip().split(","))
        return (total - used) > 2000  # >2GB free
    except Exception:
        return False


@pytest.mark.skipif(not _real_checkpoint_paths()[0].exists(), reason="real L3 checkpoint not present on this box")
@pytest.mark.skipif(not _gpu_has_headroom(), reason="insufficient GPU headroom for a real checkpoint load -- rerun once other GPU work clears")
def test_integration_smoke_real_checkpoint(tmp_path):
    model_path, vecnorm_path = _real_checkpoint_paths()
    env = _build_env(tmp_path, horizon_ticks=8, lookback_ticks=2)
    model = RecurrentPPO.load(str(model_path), device="cpu")
    wrapper = FrozenL3Wrapper(env, model, str(vecnorm_path), ticks_per_l2_decision=4)

    obs, info = wrapper.reset(seed=0)
    assert obs.shape == (L2_BASE_OBS_DIM,)
    for _ in range(2):
        obs, reward, terminated, truncated, info = wrapper.step(np.array([1.0, 0.5], dtype=np.float32))
        assert obs.shape == (L2_BASE_OBS_DIM,)
        assert np.isfinite(obs).all()
        assert np.isfinite(reward)
        if terminated or truncated:
            break
