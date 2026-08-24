"""Regression tests for the numeric-format storage conversion (2026-08-24,
docs/reports/phase4_l2_reconciliation_and_plan.md's storage-format round).
src/data/l2_numeric_format.py's write_day/read_day round-trip, and a full
env.reset()/step() comparison between the original JSON/parquet path and the
use_numeric_format=True path on a tiny synthetic day built in both formats --
matching this project's hand-computed-fixture convention, permanent coverage
alongside the separate real-data 10-seed check
(scripts/compare_formats_equivalence.py, run once, not committed as a test
since it needs real converted data files on disk).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.data.l2_numeric_format import read_day, write_day
from src.envs.lob_execution_env import LOBExecutionEnv


def test_write_read_day_round_trip_byte_identical(tmp_path):
    rng = np.random.default_rng(0)
    n = 37
    arrays = {
        "ts": np.arange(1000, 1000 + n, dtype=np.int64),
        "best_bid": rng.uniform(50000, 60000, n).astype(np.float64),
        "best_ask": rng.uniform(50000, 60000, n).astype(np.float64),
        "mid_price": rng.uniform(50000, 60000, n).astype(np.float64),
        "spread": rng.uniform(0, 1, n).astype(np.float64),
        "bid_prices": rng.uniform(50000, 60000, (n, 20)).astype(np.float64),
        "bid_sizes": rng.uniform(0, 10, (n, 20)).astype(np.float64),
        "ask_prices": rng.uniform(50000, 60000, (n, 20)).astype(np.float64),
        "ask_sizes": rng.uniform(0, 10, (n, 20)).astype(np.float64),
    }
    out_path = tmp_path / "test_day.npzst"
    write_day(arrays, out_path)
    loaded = read_day(out_path)

    for key in arrays:
        assert np.array_equal(arrays[key], loaded[key]), f"{key} not byte-identical after round-trip"
    assert len(loaded["ts"]) == n  # the n_rows the RNG draw sequence depends on


def test_write_day_rejects_missing_arrays(tmp_path):
    with pytest.raises(ValueError, match="missing required arrays"):
        write_day({"ts": np.array([1, 2, 3])}, tmp_path / "bad.npzst")


def _write_tiny_day_both_formats(tmp_path, n_rows=8):
    """Same tiny synthetic day (varying single-level book, deterministic values),
    written to BOTH the original JSON/parquet format and the numeric format, so
    reset()/step() can be compared directly against each other -- not against a
    hand-derived expected value (that's covered by
    tests/test_reset_vectorization_equivalence.py already), but against the
    OTHER format reading the identical underlying data."""
    orig_dir = tmp_path / "orig" / "BTCUSDT"
    orig_dir.mkdir(parents=True)
    numeric_dir = tmp_path / "numeric" / "BTCUSDT"
    numeric_dir.mkdir(parents=True)

    bid_sizes = [10.0 + i for i in range(n_rows)]
    ask_sizes = [8.0 + 0.5 * i for i in range(n_rows)]
    rows = []
    bid_prices_arr = np.zeros((n_rows, 20), dtype=np.float64)
    bid_sizes_arr = np.zeros((n_rows, 20), dtype=np.float64)
    ask_prices_arr = np.zeros((n_rows, 20), dtype=np.float64)
    ask_sizes_arr = np.zeros((n_rows, 20), dtype=np.float64)
    for i in range(n_rows):
        bid_levels = [[100.0 - 0.1 * k, bid_sizes[i] - k] for k in range(20)]
        ask_levels = [[100.2 + 0.1 * k, ask_sizes[i] + k] for k in range(20)]
        rows.append({
            "ts": i, "best_bid": bid_levels[0][0], "best_ask": ask_levels[0][0],
            "mid_price": (bid_levels[0][0] + ask_levels[0][0]) / 2.0,
            "spread": ask_levels[0][0] - bid_levels[0][0],
            "bids": json.dumps(bid_levels), "asks": json.dumps(ask_levels),
        })
        bid_prices_arr[i] = [lv[0] for lv in bid_levels]
        bid_sizes_arr[i] = [lv[1] for lv in bid_levels]
        ask_prices_arr[i] = [lv[0] for lv in ask_levels]
        ask_sizes_arr[i] = [lv[1] for lv in ask_levels]

    pd.DataFrame(rows).to_parquet(orig_dir / "l2-BTCUSDT-2024-01-01.parquet", index=False)
    write_day(
        {
            "ts": np.array([r["ts"] for r in rows], dtype=np.int64),
            "best_bid": np.array([r["best_bid"] for r in rows], dtype=np.float64),
            "best_ask": np.array([r["best_ask"] for r in rows], dtype=np.float64),
            "mid_price": np.array([r["mid_price"] for r in rows], dtype=np.float64),
            "spread": np.array([r["spread"] for r in rows], dtype=np.float64),
            "bid_prices": bid_prices_arr, "bid_sizes": bid_sizes_arr,
            "ask_prices": ask_prices_arr, "ask_sizes": ask_sizes_arr,
        },
        numeric_dir / "l2-BTCUSDT-2024-01-01.npzst",
    )
    return orig_dir, numeric_dir


def test_numeric_format_env_matches_original_format_env(tmp_path):
    orig_dir, numeric_dir = _write_tiny_day_both_formats(tmp_path)

    env_orig = LOBExecutionEnv(data_dir=orig_dir, horizon_ticks=3, lookback_ticks=2, tick_interval_s=1.0)
    env_numeric = LOBExecutionEnv(
        data_dir=numeric_dir, horizon_ticks=3, lookback_ticks=2, tick_interval_s=1.0,
        use_numeric_format=True,
    )
    assert env_numeric.use_numeric_format is True
    assert env_orig.use_numeric_format is False

    obs_orig, info_orig = env_orig.reset(seed=7)
    obs_numeric, info_numeric = env_numeric.reset(seed=7)
    assert np.array_equal(obs_orig, obs_numeric)
    assert env_orig.qty_total == env_numeric.qty_total
    assert env_orig.side == env_numeric.side
    assert len(env_orig._ticks) == len(env_numeric._ticks)

    for _ in range(3):
        o1, r1, t1, tr1, i1 = env_orig.step(np.array([0, 5, 0]))
        o2, r2, t2, tr2, i2 = env_numeric.step(np.array([0, 5, 0]))
        assert np.array_equal(o1, o2)
        assert r1 == r2
        assert t1 == t2 and tr1 == tr2


def test_numeric_format_glob_pattern_finds_npzst_not_parquet(tmp_path):
    orig_dir, numeric_dir = _write_tiny_day_both_formats(tmp_path)
    env_numeric = LOBExecutionEnv(
        data_dir=numeric_dir, horizon_ticks=3, lookback_ticks=2, tick_interval_s=1.0,
        use_numeric_format=True,
    )
    assert len(env_numeric._files) == 1
    assert env_numeric._files[0].suffix == ".npzst"
