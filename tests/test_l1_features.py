"""Tests for src/data/l1_features.py (architecture_spec.md Section 1.2) --
hand-computed fixtures matching the real column schemas found in
data/raw_l1 (see src/data/l1_features.py's verified inventory), not the
live archive itself, so this suite is self-contained and fast.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.data.l1_features import (
    _has_min_coverage,
    _load_klines_window,
    _load_oi_window,
    _realized_vol,
    _return_pct,
    _taker_flow_imbalance,
    build_l1_feature_summary,
)


def _to_ms(y, mo, d, h=0, mi=0, s=0) -> int:
    return int(datetime(y, mo, d, h, mi, s, tzinfo=timezone.utc).timestamp() * 1000)


def _write_klines(path, rows) -> None:
    pd.DataFrame(rows, columns=["open_time", "close", "volume", "taker_buy_volume"]).to_parquet(path, index=False)


def _write_funding(path, rows) -> None:
    pd.DataFrame(rows, columns=["calc_time", "last_funding_rate"]).to_parquet(path, index=False)


def _write_oi(path, rows) -> None:
    pd.DataFrame(
        rows,
        columns=["create_time", "sum_open_interest", "sum_toptrader_long_short_ratio", "sum_taker_long_short_vol_ratio"],
    ).to_parquet(path, index=False)


def test_return_pct_hand_computed():
    # closes 100 -> 110 -> 121 (10% growth twice); (121-100)/100 = 0.21 exactly
    closes = pd.Series([100.0, 110.0, 121.0])
    assert _return_pct(closes) == pytest.approx(0.21)


def test_return_pct_none_with_fewer_than_two_rows():
    assert _return_pct(pd.Series([100.0])) is None
    assert _return_pct(pd.Series([], dtype=float)) is None


def test_realized_vol_hand_computed():
    # constant per-step log return: closes[i] = 100 * exp(0.01 * i) -> every 1-step log
    # return is exactly 0.01, so RMS over any number of steps is exactly 0.01 too.
    closes = pd.Series([100.0 * np.exp(0.01 * i) for i in range(5)])
    assert _realized_vol(closes) == pytest.approx(0.01, abs=1e-9)


def test_realized_vol_none_with_fewer_than_two_rows():
    assert _realized_vol(pd.Series([100.0])) is None


def test_taker_flow_imbalance_hand_computed():
    # 3 rows, volume=10 each (total=30), taker_buy=7 each (total=21):
    # (2*21 - 30) / 30 = 12/30 = 0.4 exactly
    volume = pd.Series([10.0, 10.0, 10.0])
    taker_buy = pd.Series([7.0, 7.0, 7.0])
    assert _taker_flow_imbalance(volume, taker_buy) == pytest.approx(0.4)


def test_taker_flow_imbalance_none_with_zero_volume():
    assert _taker_flow_imbalance(pd.Series([0.0, 0.0]), pd.Series([0.0, 0.0])) is None


def test_has_min_coverage_boundary():
    window = timedelta(hours=24)
    full_span_ms = int(window.total_seconds() * 1000)
    assert _has_min_coverage(0, int(full_span_ms * 0.8), window) is True
    assert _has_min_coverage(0, int(full_span_ms * 0.8) - 1, window) is False


def test_load_klines_window_respects_boundary_and_falls_back_to_daily(tmp_path):
    klines_dir = tmp_path / "klines_1m"
    klines_dir.mkdir()

    # Jan 2024 monthly archive: last 2 hours of the month, 1-min bars.
    jan_rows = []
    t = _to_ms(2024, 1, 31, 22, 0)
    for i in range(120):  # 22:00 .. 23:59
        jan_rows.append({"open_time": t + i * 60_000, "close": 100.0 + i, "volume": 1.0, "taker_buy_volume": 0.5})
    _write_klines(klines_dir / "BTCUSDT-1m-2024-01.parquet", jan_rows)

    # Feb 2024: no monthly archive yet -- only a daily file for 2024-02-01.
    feb_rows = []
    t2 = _to_ms(2024, 2, 1, 0, 0)
    for i in range(31):  # 00:00 .. 00:30
        feb_rows.append({"open_time": t2 + i * 60_000, "close": 200.0 + i, "volume": 1.0, "taker_buy_volume": 0.5})
    _write_klines(klines_dir / "BTCUSDT-1m-2024-02-01.parquet", feb_rows)

    as_of_ms = t2 + 30 * 60_000  # 2024-02-01 00:30 (last Feb row)
    df = _load_klines_window(as_of_ms, timedelta(hours=1), klines_dir)

    # window is (as_of - 1h, as_of] = (2024-01-31 23:30, 2024-02-01 00:30]:
    # 29 Jan rows (23:31..23:59) + 31 Feb rows (00:00..00:30) = 60 rows total.
    assert len(df) == 60
    assert df["open_time"].iloc[0] == _to_ms(2024, 1, 31, 23, 31)
    assert df["open_time"].iloc[-1] == as_of_ms
    assert (df["open_time"].diff().dropna() == 60_000).all()


def test_load_oi_window_dedups_duplicate_create_time_rows(tmp_path):
    oi_dir = tmp_path / "open_interest"
    oi_dir.mkdir()
    # Reproduces the real 2020-09-01..2021-05-21 artifact: every row duplicated once,
    # byte-identical.
    rows = [
        {"create_time": "2024-01-01 00:00:00", "sum_open_interest": 100.0,
         "sum_toptrader_long_short_ratio": 1.2, "sum_taker_long_short_vol_ratio": 0.9},
        {"create_time": "2024-01-01 00:00:00", "sum_open_interest": 100.0,
         "sum_toptrader_long_short_ratio": 1.2, "sum_taker_long_short_vol_ratio": 0.9},
        {"create_time": "2024-01-01 00:05:00", "sum_open_interest": 101.0,
         "sum_toptrader_long_short_ratio": 1.3, "sum_taker_long_short_vol_ratio": 0.95},
    ]
    _write_oi(oi_dir / "BTCUSDT-open_interest-2024-01-01.parquet", rows)

    as_of_ms = _to_ms(2024, 1, 1, 0, 5)
    df = _load_oi_window(as_of_ms, timedelta(hours=1), oi_dir)

    assert len(df) == 2  # deduped from 3 raw rows to 2 unique timestamps
    assert df["sum_open_interest"].tolist() == [100.0, 101.0]


def test_build_l1_feature_summary_klines_fields_hand_computed(tmp_path):
    raw_dir = tmp_path / "raw_l1"
    klines_dir = raw_dir / "klines_1m"
    klines_dir.mkdir(parents=True)
    (raw_dir / "funding_rate").mkdir()
    (raw_dir / "open_interest").mkdir()

    n_rows = 1500  # 25h of 1-minute bars, comfortably covers the 24h window
    step_ms = 60_000
    start_ms = _to_ms(2024, 3, 1, 0, 0)
    log_step = 0.0001  # constant per-minute log return
    volume, taker_buy_frac = 10.0, 0.7  # imbalance = 2*0.7 - 1 = 0.4 exactly, constant

    open_times = [start_ms + i * step_ms for i in range(n_rows)]
    closes = [100.0 * np.exp(log_step * i) for i in range(n_rows)]
    rows = [
        {"open_time": open_times[i], "close": closes[i], "volume": volume, "taker_buy_volume": volume * taker_buy_frac}
        for i in range(n_rows)
    ]
    _write_klines(klines_dir / "BTCUSDT-1m-2024-03.parquet", rows)

    as_of_ms = open_times[-1]
    summary = build_l1_feature_summary(as_of_ms, raw_dir)

    open_times_arr = np.array(open_times)
    closes_arr = np.array(closes)

    for label, window_h in (("1h", 1), ("24h", 24)):
        window_ms = window_h * 3600_000
        mask = (open_times_arr > as_of_ms - window_ms) & (open_times_arr <= as_of_ms)
        sub_closes = closes_arr[mask]
        expected_return = sub_closes[-1] / sub_closes[0] - 1.0
        expected_vol = float(np.sqrt(np.mean(np.diff(np.log(sub_closes)) ** 2)))
        assert summary[f"return_{label}_pct"] == pytest.approx(expected_return)
        assert summary[f"realized_vol_{label}"] == pytest.approx(expected_vol, abs=1e-9)

    assert summary["taker_flow_imbalance_1h"] == pytest.approx(0.4)
    assert summary["as_of_ms"] == as_of_ms


def test_build_l1_feature_summary_funding_fields_hand_computed(tmp_path):
    raw_dir = tmp_path / "raw_l1"
    (raw_dir / "klines_1m").mkdir(parents=True)
    funding_dir = raw_dir / "funding_rate"
    funding_dir.mkdir()
    (raw_dir / "open_interest").mkdir()

    rates = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]
    base = _to_ms(2024, 6, 1, 0, 0)
    rows = [{"calc_time": base + i * 8 * 3600_000, "last_funding_rate": r} for i, r in enumerate(rates)]
    _write_funding(funding_dir / "BTCUSDT-funding_rate-2024-06.parquet", rows)

    as_of_ms = rows[-1]["calc_time"]
    summary = build_l1_feature_summary(as_of_ms, raw_dir)

    hist = pd.Series(rates)  # pandas .std() default ddof=1, same as the implementation
    expected_z = (0.0005 - hist.mean()) / hist.std()
    assert summary["funding_rate_current"] == pytest.approx(0.0005)
    assert summary["funding_rate_z"] == pytest.approx(expected_z)


def test_build_l1_feature_summary_funding_z_none_with_single_period(tmp_path):
    raw_dir = tmp_path / "raw_l1"
    (raw_dir / "klines_1m").mkdir(parents=True)
    funding_dir = raw_dir / "funding_rate"
    funding_dir.mkdir()
    (raw_dir / "open_interest").mkdir()

    as_of_ms = _to_ms(2024, 6, 1, 0, 0)
    _write_funding(
        funding_dir / "BTCUSDT-funding_rate-2024-06.parquet",
        [{"calc_time": as_of_ms, "last_funding_rate": 0.0001}],
    )

    summary = build_l1_feature_summary(as_of_ms, raw_dir)
    assert summary["funding_rate_current"] == pytest.approx(0.0001)
    assert summary["funding_rate_z"] is None


def test_build_l1_feature_summary_oi_fields_hand_computed(tmp_path):
    raw_dir = tmp_path / "raw_l1"
    (raw_dir / "klines_1m").mkdir(parents=True)
    (raw_dir / "funding_rate").mkdir()
    oi_dir = raw_dir / "open_interest"
    oi_dir.mkdir()

    day1, day2 = date(2024, 7, 1), date(2024, 7, 2)
    ts_ms, oi_vals = [], []

    def _make_day(day, oi_base, ratio, taker_ratio):
        rows = []
        for i in range(288):
            ts = _to_ms(day.year, day.month, day.day) + i * 5 * 60_000
            oi = oi_base + i
            rows.append({
                "create_time": datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                "sum_open_interest": oi, "sum_toptrader_long_short_ratio": ratio,
                "sum_taker_long_short_vol_ratio": taker_ratio,
            })
            ts_ms.append(ts)
            oi_vals.append(oi)
        return rows

    _write_oi(oi_dir / f"BTCUSDT-open_interest-{day1.isoformat()}.parquet", _make_day(day1, 1000.0, 1.5, 0.8))
    _write_oi(oi_dir / f"BTCUSDT-open_interest-{day2.isoformat()}.parquet", _make_day(day2, 1288.0, 1.6, 0.85))

    as_of_ms = _to_ms(2024, 7, 2, 12, 0)
    summary = build_l1_feature_summary(as_of_ms, raw_dir)

    ts_arr, oi_arr = np.array(ts_ms), np.array(oi_vals)
    current_mask = ts_arr <= as_of_ms
    current_pos = np.argmax(ts_arr[current_mask])
    current_oi = oi_arr[current_mask][current_pos]

    window_mask = (ts_arr > as_of_ms - 24 * 3600_000) & (ts_arr <= as_of_ms)
    first_pos = np.argmin(ts_arr[window_mask])
    first_oi = oi_arr[window_mask][first_pos]

    assert summary["open_interest_level"] == pytest.approx(float(current_oi))
    assert summary["open_interest_change_24h_pct"] == pytest.approx(float((current_oi - first_oi) / first_oi))
    assert summary["top_trader_long_short_ratio"] == pytest.approx(1.6)
    assert summary["taker_long_short_vol_ratio"] == pytest.approx(0.85)


def test_build_l1_feature_summary_oi_change_none_when_span_too_short(tmp_path):
    raw_dir = tmp_path / "raw_l1"
    (raw_dir / "klines_1m").mkdir(parents=True)
    (raw_dir / "funding_rate").mkdir()
    oi_dir = raw_dir / "open_interest"
    oi_dir.mkdir()

    # only 2 rows, 1 hour apart -- far short of the 24h*0.8=19.2h minimum coverage.
    rows = [
        {"create_time": "2024-08-01 10:00:00", "sum_open_interest": 500.0,
         "sum_toptrader_long_short_ratio": 1.1, "sum_taker_long_short_vol_ratio": 0.7},
        {"create_time": "2024-08-01 11:00:00", "sum_open_interest": 505.0,
         "sum_toptrader_long_short_ratio": 1.2, "sum_taker_long_short_vol_ratio": 0.75},
    ]
    _write_oi(oi_dir / "BTCUSDT-open_interest-2024-08-01.parquet", rows)

    as_of_ms = _to_ms(2024, 8, 1, 11, 0)
    summary = build_l1_feature_summary(as_of_ms, raw_dir)

    assert summary["open_interest_level"] == pytest.approx(505.0)  # current level still reported
    assert summary["open_interest_change_24h_pct"] is None  # but the 24h-labeled change is not
    assert summary["top_trader_long_short_ratio"] == pytest.approx(1.2)


def test_build_l1_feature_summary_all_none_and_valid_json_when_no_data_exists(tmp_path):
    raw_dir = tmp_path / "raw_l1"
    (raw_dir / "klines_1m").mkdir(parents=True)
    (raw_dir / "funding_rate").mkdir()
    (raw_dir / "open_interest").mkdir()

    as_of_ms = _to_ms(2019, 1, 1, 0, 0)  # long before any real archive would exist
    summary = build_l1_feature_summary(as_of_ms, raw_dir)

    assert summary["as_of_ms"] == as_of_ms
    for key, value in summary.items():
        if key != "as_of_ms":
            assert value is None, f"{key} should be None, got {value!r}"

    # the concrete failure mode this design avoids: json.dumps happily emits a literal
    # NaN token for a float NaN, which is not valid JSON -- None/null round-trips clean.
    payload = json.dumps(summary)
    assert "NaN" not in payload
    assert json.loads(payload) == summary
