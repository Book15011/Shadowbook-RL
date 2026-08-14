"""Fixture-based unit tests for Phase 2b new observation-vector features
(idx 4-5, 10-19, 39-41), plus the full-vector completeness test. Same style
as tests/test_matching_engine.py: hand-computed expected values, tolerance-
based assertion.

Several tests deliberately pick lookback_ticks/tick_interval_s so that a
rolling window is FULLY populated regardless of where reset() randomly
starts the episode within the synthetic day (see each test's comment) --
this makes the hand-computed expected value robust to the seed draw rather
than needing to special-case a specific seed outcome.
"""
import json
import math

import numpy as np
import pandas as pd
import pytest

from src.envs.lob_execution_env import (
    ORDER_TYPE_HOLD,
    ORDER_TYPE_LIMIT,
    ORDER_TYPE_MARKET,
    _OBS_SPEC,
    LOBExecutionEnv,
)


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


def _write_book_depth_day(path, n_rows, bid_sizes, ask_sizes, price=100.0, ts_start=1):
    bid_levels = [[round(price - 0.1 * (i + 1), 2), s] for i, s in enumerate(bid_sizes)]
    ask_levels = [[round(price + 0.1 * (i + 1), 2), s] for i, s in enumerate(ask_sizes)]
    bids = json.dumps(bid_levels)
    asks = json.dumps(ask_levels)
    rows = [
        {
            "ts": ts_start + i, "best_bid": bid_levels[0][0], "best_ask": ask_levels[0][0],
            "mid_price": price, "spread": ask_levels[0][0] - bid_levels[0][0],
            "bids": bids, "asks": asks,
        }
        for i in range(n_rows)
    ]
    pd.DataFrame(rows).to_parquet(path, index=False)


def _write_decreasing_day(path, n_rows, bid_start, bid_step, ask_start, ask_step, price=100.0, ts_start=1):
    rows = []
    for i in range(n_rows):
        bs = bid_start - bid_step * i
        asz = ask_start - ask_step * i
        rows.append({
            "ts": ts_start + i, "best_bid": price - 0.05, "best_ask": price + 0.05,
            "mid_price": price, "spread": 0.1,
            "bids": json.dumps([[price - 0.05, bs]]),
            "asks": json.dumps([[price + 0.05, asz]]),
        })
    pd.DataFrame(rows).to_parquet(path, index=False)


def _write_funding_history(path, calc_times, rates):
    pd.DataFrame({"calc_time": calc_times, "last_funding_rate": rates}).to_parquet(path, index=False)


# ---- idx 19-38: book_depth_norm ----

def test_book_depth_norm_hand_computed_fixture(tmp_path):
    data_dir = tmp_path / "BTCUSDT"
    data_dir.mkdir()
    bid_sizes = [10.0, 8.0, 6.0, 4.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    ask_sizes = [5.0] * 10
    _write_book_depth_day(data_dir / "l2-BTCUSDT-2024-01-01.parquet", 20, bid_sizes, ask_sizes)

    env = LOBExecutionEnv(data_dir=data_dir, horizon_ticks=5, lookback_ticks=2)
    obs, info = env.reset(seed=1)

    # mean = 85/20 = 4.25, sum_sq = 475, mean_sq = 23.75, variance = 5.6875 = 91/16
    # std = sqrt(91)/4. z_bid_level1 (size=10) = 23/sqrt(91); z_bid_level10 (size=1) =
    # -13/sqrt(91); z_ask_level1 (size=5) = 3/sqrt(91).
    sqrt91 = math.sqrt(91)
    assert obs[19] == pytest.approx(23.0 / sqrt91, abs=1e-4)
    assert obs[28] == pytest.approx(-13.0 / sqrt91, abs=1e-4)
    assert obs[29] == pytest.approx(3.0 / sqrt91, abs=1e-4)


# ---- idx 15: l2_target_slice_ratio ----

def test_l2_target_slice_ratio_hand_computed(tmp_path):
    data_dir = tmp_path / "BTCUSDT"
    data_dir.mkdir()
    _write_constant_day(data_dir / "l2-BTCUSDT-2024-01-01.parquet", 200, 100.0, 1e8, 100.5, 1e8)

    env = LOBExecutionEnv(data_dir=data_dir, horizon_ticks=10, lookback_ticks=2)
    obs, info = env.reset(seed=1)
    assert obs[15] == pytest.approx(0.0)

    for expected_elapsed in range(1, 4):
        obs, r, term, trunc, info = env.step(np.array([ORDER_TYPE_HOLD, 5, 0]))
        assert obs[15] == pytest.approx(expected_elapsed / 10.0)


def test_l2_target_slice_ratio_override(tmp_path):
    data_dir = tmp_path / "BTCUSDT"
    data_dir.mkdir()
    _write_constant_day(data_dir / "l2-BTCUSDT-2024-01-01.parquet", 200, 100.0, 1e8, 100.5, 1e8)

    env = LOBExecutionEnv(
        data_dir=data_dir, horizon_ticks=10, lookback_ticks=2,
        l2_target_slice_ratio_override=0.75,
    )
    obs, info = env.reset(seed=1)
    assert obs[15] == pytest.approx(0.75)


# ---- idx 41: own_open_orders_norm ----

def test_own_open_orders_norm_reflects_resting_order(tmp_path):
    data_dir = tmp_path / "BTCUSDT"
    data_dir.mkdir()
    _write_constant_day(data_dir / "l2-BTCUSDT-2024-01-01.parquet", 200, 100.0, 1e8, 100.5, 1e8)

    env = LOBExecutionEnv(data_dir=data_dir, horizon_ticks=10, lookback_ticks=2)
    obs, info = env.reset(seed=1)
    assert obs[41] == pytest.approx(0.0)

    obs, r, term, trunc, info = env.step(np.array([ORDER_TYPE_LIMIT, 5, 2]))  # SIZE_FRACTIONS[2] = 0.6
    assert obs[41] == pytest.approx(0.6)


# ---- idx 10-11: cancel_add_ratio, always 0.0 (genuinely blocked, see module docstring) ----

def test_cancel_add_ratio_always_zero(tmp_path):
    data_dir = tmp_path / "BTCUSDT"
    data_dir.mkdir()
    _write_constant_day(data_dir / "l2-BTCUSDT-2024-01-01.parquet", 200, 100.0, 1e8, 100.5, 1e8)

    env = LOBExecutionEnv(data_dir=data_dir, horizon_ticks=10, lookback_ticks=2)
    obs, info = env.reset(seed=1)
    assert obs[10] == 0.0 and obs[11] == 0.0
    obs, r, term, trunc, info = env.step(np.array([ORDER_TYPE_MARKET, 5, 4]))
    assert obs[10] == 0.0 and obs[11] == 0.0


# ---- idx 14: ticks_since_own_fill_norm ----

def test_ticks_since_own_fill_norm_tracks_last_fill(tmp_path):
    data_dir = tmp_path / "BTCUSDT"
    data_dir.mkdir()
    _write_constant_day(data_dir / "l2-BTCUSDT-2024-01-01.parquet", 400, 100.0, 1e8, 100.5, 1e8)

    env = LOBExecutionEnv(data_dir=data_dir, horizon_ticks=100, lookback_ticks=2)
    obs, info = env.reset(seed=1)
    assert obs[14] == pytest.approx(1.0)  # no fill yet this episode

    obs, r, term, trunc, info = env.step(np.array([ORDER_TYPE_MARKET, 5, 0]))  # 20% market fill
    assert not term
    assert obs[14] == pytest.approx(1.0 / 100.0)

    for _ in range(5):
        obs, r, term, trunc, info = env.step(np.array([ORDER_TYPE_HOLD, 5, 0]))
    assert obs[14] == pytest.approx(6.0 / 100.0)


# ---- idx 39: funding_rate_z ----

def test_funding_rate_z_hand_computed(tmp_path):
    data_dir = tmp_path / "BTCUSDT"
    data_dir.mkdir()
    _write_constant_day(
        data_dir / "l2-BTCUSDT-2024-01-01.parquet", 200, 100.0, 1e8, 100.5, 1e8, ts_start=10_000,
    )

    funding_dir = tmp_path / "funding_rate"
    funding_dir.mkdir()
    _write_funding_history(
        funding_dir / "BTCUSDT-funding_rate-2024-01.parquet",
        calc_times=[1000, 2000, 3000, 4000, 5000],
        rates=[0.0001, 0.0002, 0.0003, 0.0004, 0.0005],
    )

    env = LOBExecutionEnv(
        data_dir=data_dir, horizon_ticks=5, lookback_ticks=2, funding_rate_dir=funding_dir,
    )
    obs, info = env.reset(seed=1)
    # mean=0.0003, deviations in units of 0.0001 are -2,-1,0,1,2 -> mean_sq_dev=2*(1e-4)^2
    # std = sqrt(2)*1e-4, current=0.0005 -> z = (0.0005-0.0003)/(sqrt(2)*1e-4) = 2/sqrt(2) = sqrt(2)
    assert obs[39] == pytest.approx(math.sqrt(2.0), abs=1e-6)


def test_funding_rate_z_defaults_to_zero_with_no_funding_data(tmp_path):
    data_dir = tmp_path / "BTCUSDT"
    data_dir.mkdir()
    _write_constant_day(data_dir / "l2-BTCUSDT-2024-01-01.parquet", 200, 100.0, 1e8, 100.5, 1e8)

    env = LOBExecutionEnv(
        data_dir=data_dir, horizon_ticks=5, lookback_ticks=2,
        funding_rate_dir=tmp_path / "does_not_exist",
    )
    obs, info = env.reset(seed=1)
    assert obs[39] == 0.0


# ---- idx 12 / 40: trade_flow_imbalance_5s / taker_buy_sell_ratio_1m ----

def test_trade_flow_and_taker_ratio_hand_computed(tmp_path):
    data_dir = tmp_path / "BTCUSDT"
    data_dir.mkdir()
    # constant per-tick depletion: bid loses 0.1/tick, ask loses 0.05/tick -> constant
    # imbalance ratio regardless of window length.
    _write_decreasing_day(
        data_dir / "l2-BTCUSDT-2024-01-01.parquet", n_rows=200,
        bid_start=50.0, bid_step=0.1, ask_start=50.0, ask_step=0.05,
    )
    # lookback_ticks=60 with tick_interval_s=1.0 guarantees episode_start >= 60, so both
    # the 5-tick and 60-tick rolling windows are always fully populated (deterministic,
    # independent of the random start draw) -- see module docstring Lookback buffer note.
    env = LOBExecutionEnv(
        data_dir=data_dir, horizon_ticks=5, lookback_ticks=60, tick_interval_s=1.0,
    )
    obs, info = env.reset(seed=1)
    # per-tick: ask_dep=0.05 (taker buy), bid_dep=0.1 (taker sell), signed=-0.05, abs=0.15
    # ratio = -0.05/0.15 = -1/3, identical for both window lengths since the per-tick
    # rate is constant.
    assert obs[12] == pytest.approx(-1.0 / 3.0, abs=1e-4)
    assert obs[40] == pytest.approx(-1.0 / 3.0, abs=1e-4)


# ---- full-vector completeness ----

def test_full_vector_completeness(tmp_path):
    data_dir = tmp_path / "BTCUSDT"
    data_dir.mkdir()
    _write_decreasing_day(
        data_dir / "l2-BTCUSDT-2024-01-01.parquet", n_rows=300,
        bid_start=50.0, bid_step=0.05, ask_start=50.0, ask_step=0.03,
    )
    env = LOBExecutionEnv(data_dir=data_dir, horizon_ticks=50, lookback_ticks=10)

    actions = [
        [ORDER_TYPE_HOLD, 5, 0], [ORDER_TYPE_LIMIT, 5, 2], [ORDER_TYPE_HOLD, 5, 0],
        [ORDER_TYPE_MARKET, 5, 1], [ORDER_TYPE_HOLD, 5, 0],
    ]
    for seed in range(5):
        obs, info = env.reset(seed=seed)
        for step_i in range(20):
            assert obs.shape == (42,)
            assert obs.dtype == np.float32
            assert not np.isnan(obs).any(), f"seed={seed} step={step_i}: NaN in obs"
            assert not np.isinf(obs).any(), f"seed={seed} step={step_i}: Inf in obs"
            for idx, name, (lo, hi) in _OBS_SPEC:
                assert lo - 1e-4 <= obs[idx] <= hi + 1e-4, (
                    f"seed={seed} step={step_i}: obs[{idx}] ({name})={obs[idx]} outside [{lo},{hi}]"
                )
            action = actions[step_i % len(actions)]
            obs, r, term, trunc, info = env.step(np.array(action))
            if term or trunc:
                break


# ---- regression: ref_depth must not leak the wider lookback buffer ----

def test_ref_depth_scoped_to_legacy_window_not_full_buffer(tmp_path):
    # Regression test for a bug caught during Phase 2b development: ref_depth (which
    # sizes qty_total) must be computed only from legacy_ticks (the original Phase 2a
    # lookback_ticks+horizon_ticks window), NOT the full (wider, up to 600-tick-buffered)
    # self._ticks -- otherwise qty_total silently drifts when the buffer widens for idx
    # 4/5/12/40 rolling windows, breaking already-validated Phase 2a scenario sizing.
    # Caught via a controlled old-vs-new comparison at an identical seed and date_range:
    # qty_total differed (23.9086 vs 22.7718) purely because ref_depth median was being
    # computed over more history than before.
    data_dir = tmp_path / "BTCUSDT"
    data_dir.mkdir()
    n_rows = 3000
    # seed=0 with horizon_ticks=100, lookback_ticks=10 deterministically draws start=2468
    # for this exact synthetic day (n_rows=3000, constant book) -- verified empirically
    # once, asserted below so an RNG algorithm change would fail loudly here rather than
    # silently invalidating the test.
    low_depth = 1.0
    high_depth = 1000.0
    boundary = 2458  # start(2468) - lookback_ticks(10): legacy window begins exactly here
    rows = []
    for i in range(n_rows):
        depth = high_depth if i >= boundary else low_depth
        rows.append({
            "ts": i, "best_bid": 99.95, "best_ask": 100.05, "mid_price": 100.0, "spread": 0.1,
            "bids": json.dumps([[99.95, depth]]), "asks": json.dumps([[100.05, depth]]),
        })
    pd.DataFrame(rows).to_parquet(data_dir / "l2-BTCUSDT-2024-01-01.parquet", index=False)

    env = LOBExecutionEnv(data_dir=data_dir, horizon_ticks=100, lookback_ticks=10)
    obs, info = env.reset(seed=0)
    assert env._episode_start == 600, "buffer did not reach full width -- fixture assumption broken"
    ts_at_start = env._ticks[env._episode_start].ts
    assert ts_at_start == 2468, "start drifted from the value this fixture was built around"

    ref_depth = env.qty_total / env._scenario_depth_ratio
    assert ref_depth == pytest.approx(2 * high_depth, rel=1e-6), (
        f"ref_depth={ref_depth} looks like it leaked the wide buffer low-depth rows "
        f"(expected ~{2 * high_depth}, the buffer-only region has depth {2 * low_depth})"
    )
