"""Tests for EXPERIMENTAL ADDITION 5 (reward.py module docstring):
RewardWeights.subtract_twap_baseline and LOBExecutionEnv's TWAP shadow
computation. Same fixture discipline as test_reward.py's hand-computed
tests, adapted for env-level physics: rather than hand-simulating the
matching engine in the test itself (error-prone, duplicates even more
logic), these verify exact ALGEBRAIC relationships between paired runs on
the same seed -- deterministic, exactly assertable, and self-checking
against the REAL TWAPPolicy as ground truth rather than a second
hand-written approximation of it.
"""
import json

import numpy as np
import pandas as pd
import pytest

from scripts.phase2a_sanity_suite import TWAPPolicy, run_episode
from src.envs.lob_execution_env import ORDER_TYPE_HOLD, ORDER_TYPE_MARKET, LOBExecutionEnv
from src.envs.reward import RewardWeights


def _write_constant_day(path, n_rows, bid_price, bid_size, ask_price, ask_size, ts_start=1):
    bids = json.dumps([[bid_price, bid_size]])
    asks = json.dumps([[ask_price, ask_size]])
    rows = [
        {
            "ts": ts_start + i, "best_bid": bid_price, "best_ask": ask_price,
            "mid_price": (bid_price + ask_price) / 2.0, "spread": ask_price - bid_price,
            "bids": bids, "asks": asks,
        }
        for i in range(n_rows)
    ]
    pd.DataFrame(rows).to_parquet(path, index=False)


def _make_env(tmp_path, **kwargs):
    data_dir = tmp_path / "BTCUSDT"
    data_dir.mkdir(exist_ok=True)
    day_path = data_dir / "l2-BTCUSDT-2024-01-01.parquet"
    if not day_path.exists():
        _write_constant_day(day_path, 400, 100.0, 1e8, 100.5, 1e8)
    defaults = dict(data_dir=data_dir, horizon_ticks=100, lookback_ticks=2)
    defaults.update(kwargs)
    return LOBExecutionEnv(**defaults)


def test_subtract_twap_baseline_defaults_to_inert(tmp_path):
    assert RewardWeights().subtract_twap_baseline is False

    env = _make_env(tmp_path)
    env.reset(seed=1)
    # Not computed at all when the flag is off -- costs nothing, not just "unused".
    assert env._twap_shadow_terminal_is_bps is None


def test_subtract_twap_baseline_does_not_change_reported_is(tmp_path):
    # info["implementation_shortfall"] must report the real, un-adjusted
    # execution outcome regardless of the flag -- only the scalar reward
    # returned from step() may differ. Same seed, same (trivial HOLD) policy,
    # only the flag differs.
    env_off = _make_env(tmp_path, reward_weights=RewardWeights(subtract_twap_baseline=False))
    env_off.reset(seed=1)
    for _ in range(100):
        obs, r, term, trunc, info = env_off.step(np.array([ORDER_TYPE_HOLD, 5, 0]))
        if term or trunc:
            break
    is_off = info["implementation_shortfall"]

    env_on = _make_env(tmp_path, reward_weights=RewardWeights(subtract_twap_baseline=True))
    env_on.reset(seed=1)
    for _ in range(100):
        obs, r, term, trunc, info = env_on.step(np.array([ORDER_TYPE_HOLD, 5, 0]))
        if term or trunc:
            break
    is_on = info["implementation_shortfall"]

    assert is_on.is_total_bps == pytest.approx(is_off.is_total_bps)
    assert is_on.fill_ratio == pytest.approx(is_off.fill_ratio)


def test_subtract_twap_baseline_arithmetic_matches_hand_derivation(tmp_path):
    # A deterministic, non-TWAP policy (single MARKET order on tick 1, then
    # HOLD) so total_reward is exactly reproducible. Every non-terminal
    # reward component (r_slip, r_inv, r_queue, r_spread, r_stale,
    # r_placement_stale) is IDENTICAL between the two runs below -- neither
    # depends on subtract_twap_baseline at all -- so they cancel exactly in
    # the difference, leaving a hand-derivable relationship:
    #   total_reward_on - total_reward_off == kappa * twap_shadow_is_bps
    # (from r_on's terminal term -kappa*(agent_is - twap_is) vs r_off's
    # -kappa*agent_is -- the agent_is terms cancel, leaving +kappa*twap_is).
    kappa = RewardWeights().kappa

    def policy_actions():
        yield np.array([ORDER_TYPE_MARKET, 5, 4])
        while True:
            yield np.array([ORDER_TYPE_HOLD, 5, 0])

    env_off = _make_env(tmp_path, reward_weights=RewardWeights(subtract_twap_baseline=False))
    env_off.reset(seed=1)
    total_off = 0.0
    actions = policy_actions()
    for _ in range(100):
        obs, r, term, trunc, info = env_off.step(next(actions))
        total_off += r
        if term or trunc:
            break

    env_on = _make_env(tmp_path, reward_weights=RewardWeights(subtract_twap_baseline=True))
    env_on.reset(seed=1)
    twap_shadow_is = env_on._twap_shadow_terminal_is_bps
    assert twap_shadow_is is not None
    total_on = 0.0
    actions = policy_actions()
    for _ in range(100):
        obs, r, term, trunc, info = env_on.step(next(actions))
        total_on += r
        if term or trunc:
            break

    assert (total_on - total_off) == pytest.approx(kappa * twap_shadow_is, abs=1e-6)


def test_subtract_twap_baseline_matches_real_twap_policy_exactly(tmp_path):
    # The key integration check: run the REAL TWAPPolicy (from
    # scripts/phase2a_sanity_suite.py) through the REAL env, flag on, same
    # seed. Since the real episode's own execution IS TWAP itself, the
    # shadow computed in reset() (a deliberately-duplicated, not-imported,
    # reimplementation of TWAP's decision logic) must produce a terminal IS
    # matching the real run's own outcome almost exactly -- this is what
    # actually catches drift between the two implementations, not code
    # review alone.
    env = _make_env(tmp_path, reward_weights=RewardWeights(subtract_twap_baseline=True))
    result = run_episode(env, TWAPPolicy(n_slices=10), seed=1, horizon_ticks=100)
    agent_is = result["is_result"].is_total_bps
    twap_shadow_is = env._twap_shadow_terminal_is_bps
    assert twap_shadow_is is not None
    assert agent_is == pytest.approx(twap_shadow_is, abs=1e-6)
