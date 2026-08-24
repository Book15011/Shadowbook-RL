"""Regression tests for the reset()-optimization vectorization (2026-08-24,
docs/reports/phase4_l2_reconciliation_and_plan.md's env.reset() investigation).

_rolling_sum/_rolling_rms/_rolling_mean_std and the new _vec_qty_at_price replaced
python range(n) loops with vectorized numpy -- same arithmetic, same array-order
first-match semantics. These tests pin that equivalence permanently: each helper
is checked against a REFERENCE implementation that is the ORIGINAL python-loop
version, reproduced verbatim here (the original no longer exists in source once
replaced) -- not against the new code's own logic restated differently, which
would prove nothing. A live-data, real-seed, full-env before/after comparison
(env.reset()/env.step() traces over 10 fixed seeds, both the initial seeded reset
and a subsequent unseeded reset exercising the day-cache-hit path) was also run
separately for this change and confirmed byte-identical (np.array_equal, not
np.allclose) -- see docs/reports/phase4_l2_reconciliation_and_plan.md for that
result; these tests are the permanent, hand-computed-fixture regression coverage
matching this project's established test convention (tests/test_lob_execution_env_features.py),
not a repeat of the live-data check.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from src.envs.lob_execution_env import (
    TICK_SIZE,
    LOBExecutionEnv,
    TickView,
    _rolling_mean_std,
    _rolling_rms,
    _rolling_sum,
    _vec_qty_at_price,
)


# ---- reference (original, pre-vectorization) implementations, reproduced verbatim ----

def _ref_rolling_sum(values: np.ndarray, window_ticks: int) -> np.ndarray:
    n = len(values)
    csum = np.concatenate([[0.0], np.cumsum(values.astype(float))])
    out = np.empty(n, dtype=float)
    for i in range(n):
        w = min(window_ticks, i + 1)
        out[i] = csum[i + 1] - csum[i + 1 - w]
    return out


def _ref_rolling_rms(values: np.ndarray, window_ticks: int) -> np.ndarray:
    n = len(values)
    sq = values.astype(float) ** 2
    csum = np.concatenate([[0.0], np.cumsum(sq)])
    out = np.empty(n, dtype=float)
    for i in range(n):
        w = min(window_ticks, i + 1)
        out[i] = math.sqrt((csum[i + 1] - csum[i + 1 - w]) / w) if w > 0 else 0.0
    return out


def _ref_rolling_mean_std(values: np.ndarray, window_ticks: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(values)
    v = values.astype(float)
    csum = np.concatenate([[0.0], np.cumsum(v)])
    csum_sq = np.concatenate([[0.0], np.cumsum(v * v)])
    mean_out = np.empty(n, dtype=float)
    std_out = np.empty(n, dtype=float)
    for i in range(n):
        w = min(window_ticks, i + 1)
        s = csum[i + 1] - csum[i + 1 - w]
        sq = csum_sq[i + 1] - csum_sq[i + 1 - w]
        m = s / w
        variance = max(0.0, sq / w - m * m)
        mean_out[i] = m
        std_out[i] = math.sqrt(variance)
    return mean_out, std_out


# ---- rolling-window helpers: vectorized vs. reference, several sizes/edge cases ----

_WINDOW_CASES = [
    (np.array([3.0, -1.0, 4.0, 1.0, -5.0, 9.0, 2.0, -6.0]), 3),  # ordinary window < n
    (np.array([3.0, -1.0, 4.0, 1.0, -5.0, 9.0, 2.0, -6.0]), 1),  # window=1 (degenerate, no averaging)
    (np.array([3.0, -1.0, 4.0, 1.0, -5.0, 9.0, 2.0, -6.0]), 100),  # window > n (always full-array-so-far)
    (np.array([7.0]), 5),  # n=1
    (np.array([]), 5),  # n=0
]


@pytest.mark.parametrize("values,window", _WINDOW_CASES)
def test_rolling_sum_matches_reference_loop(values, window):
    assert np.array_equal(_rolling_sum(values, window), _ref_rolling_sum(values, window))


@pytest.mark.parametrize("values,window", _WINDOW_CASES)
def test_rolling_rms_matches_reference_loop(values, window):
    assert np.array_equal(_rolling_rms(values, window), _ref_rolling_rms(values, window))


@pytest.mark.parametrize("values,window", _WINDOW_CASES)
def test_rolling_mean_std_matches_reference_loop(values, window):
    m_new, s_new = _rolling_mean_std(values, window)
    m_ref, s_ref = _ref_rolling_mean_std(values, window)
    assert np.array_equal(m_new, m_ref)
    assert np.array_equal(s_new, s_ref)


# ---- _vec_qty_at_price vs. TickView.qty_at_price (unchanged, still the ground truth) ----

def _make_tick(bid_prices, bid_sizes, ask_prices, ask_sizes):
    return TickView(
        ts=0, best_bid=bid_prices[0] if bid_prices else 0.0,
        best_ask=ask_prices[0] if ask_prices else 0.0,
        mid_price=0.0, spread=0.0,
        bid_prices=np.array(bid_prices), bid_sizes=np.array(bid_sizes),
        ask_prices=np.array(ask_prices), ask_sizes=np.array(ask_sizes),
    )


def test_vec_qty_at_price_matches_scalar_including_no_match_and_ragged_padding():
    # Row 0: normal single-level book. Row 1: price NOT present (no-match case, must
    # return 0.0). Row 2: shorter array than row 0/1 (ragged -- zero-padding must not
    # spuriously match: pad value 0.0 vs a real ~100-scale query price is never within
    # TICK_SIZE/2). Row 3: TWO levels both within atol of the query price (duplicate
    # tolerance-window case) -- first-in-array-order must win, matching sizes[matches][0].
    ticks = [
        _make_tick([100.0, 99.9], [5.0, 3.0], [], []),
        _make_tick([100.0, 99.9], [5.0, 3.0], [], []),
        _make_tick([100.0], [7.0], [], []),
        _make_tick([100.02, 100.04], [11.0, 22.0], [], []),  # both within atol=0.05 of query 100.0
    ]
    max_levels = max(len(t.bid_prices) for t in ticks)
    price_mat = np.zeros((len(ticks), max_levels))
    size_mat = np.zeros((len(ticks), max_levels))
    for i, t in enumerate(ticks):
        k = len(t.bid_prices)
        price_mat[i, :k] = t.bid_prices
        size_mat[i, :k] = t.bid_sizes

    queries = np.array([100.0, 50.0, 100.0, 100.0])  # row1: no level near 50.0 -> 0.0
    result = _vec_qty_at_price(price_mat, size_mat, queries)

    expected = np.array([
        ticks[0].qty_at_price(100.0, "bid"),
        ticks[1].qty_at_price(50.0, "bid"),
        ticks[2].qty_at_price(100.0, "bid"),
        ticks[3].qty_at_price(100.0, "bid"),
    ])
    assert np.array_equal(result, expected)
    assert expected[1] == 0.0  # confirms the no-match branch was actually exercised
    assert expected[3] == 11.0  # confirms first-in-array-order semantics, not e.g. nearest


# ---- full env.reset() integration, tiny synthetic day, hand-computed touch-depletion ----

def _write_tiny_day(path):
    # Degenerate-window construction (n_rows == lookback_ticks+horizon_ticks below):
    # reset()'s own fallback branch sets start=lookback_ticks, end=n_rows
    # deterministically (no RNG-dependent window placement) -- see reset()'s own
    # comment on this branch. self._ticks becomes exactly these 5 rows, in order.
    # Single bid/ask level each, chosen so touch-depletion (signed/absvol at idx
    # 12's trade_flow_imbalance_5s) is hand-computable tick by tick:
    #   row0->row1: bid 10.0->6.0 (dep 4.0), ask 8.0->8.0 (dep 0.0) => signed=-4, absvol=4
    #   row1->row2: bid 6.0->6.0  (dep 0.0), ask 8.0->3.0 (dep 5.0) => signed=+5, absvol=5
    #   row2->row3: bid 6.0->6.0  (dep 0.0), ask 3.0->3.0 (dep 0.0) => signed=0,  absvol=0
    #   row3->row4: bid 6.0->2.0  (dep 4.0), ask 3.0->9.0 (dep 0.0, size INCREASED,
    #               not depleted -- max(0, 3-9) clips to 0) => signed=-4, absvol=4
    bid_sizes = [10.0, 6.0, 6.0, 6.0, 2.0]
    ask_sizes = [8.0, 8.0, 3.0, 3.0, 9.0]
    rows = []
    for i in range(5):
        bids = json.dumps([[100.0, bid_sizes[i]]])
        asks = json.dumps([[100.2, ask_sizes[i]]])
        rows.append({
            "ts": i, "best_bid": 100.0, "best_ask": 100.2, "mid_price": 100.1, "spread": 0.2,
            "bids": bids, "asks": asks,
        })
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_touch_depletion_hand_computed_via_full_reset(tmp_path):
    data_dir = tmp_path / "BTCUSDT"
    data_dir.mkdir()
    _write_tiny_day(data_dir / "l2-BTCUSDT-2024-01-01.parquet")

    env = LOBExecutionEnv(data_dir=data_dir, horizon_ticks=3, lookback_ticks=2, tick_interval_s=1.0)
    obs, info = env.reset(seed=0)
    assert env._episode_start == 2  # degenerate-branch construction, deterministic (see _write_tiny_day)
    assert len(env._ticks) == 5

    ticks_5s = max(1, round(5.0 / env.tick_interval_s))  # =5, i.e. the whole 5-row window
    expected_signed = np.array([0.0, -4.0, 5.0, 0.0, -4.0])
    expected_absvol = np.array([0.0, 4.0, 5.0, 0.0, 4.0])
    expected_flow5s = np.clip(
        _ref_rolling_sum(expected_signed, ticks_5s) / (_ref_rolling_sum(expected_absvol, ticks_5s) + 1e-9),
        -1.0, 1.0,
    )
    assert np.allclose(env._flow5s_series, expected_flow5s, atol=1e-12)

    # obs at episode_start (idx 2, first decision) reads _flow5s_series[2] into idx 12.
    assert obs[12] == pytest.approx(expected_flow5s[2], abs=1e-6)
