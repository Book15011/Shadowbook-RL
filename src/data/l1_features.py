"""L1 feature_summary construction (architecture_spec.md Section 1.2) --
the missing link between data/raw_l1's collected Binance aggregates
(klines_1m, funding_rate, open_interest) and what
src.agents.l1_macro_analyst.L1MacroAnalyst.maybe_refresh() actually
consumes as its feature_summary argument.

Real-code-wins check (read directly from src/agents/l1_macro_analyst.py,
not inferred from the spec): maybe_refresh() takes a plain
feature_summary dict and json.dumps()'s it straight into the Ollama
prompt -- there is NO pydantic-validated input schema, unlike
MacroRiskContext on the output side. The only real contract is semantic,
from SYSTEM_PROMPT's own description: "rolling numeric features
(order-book imbalance, realized volatility, funding rate, recent trade
flow)". This module supplies everything in that list EXCEPT order-book
imbalance, deliberately: data/raw_l1 (Binance klines/funding/open interest
aggregates) has no book-depth data at all -- per architecture_spec.md
Section 2.1.1, real OBI is sourced from the separate Bybit L2 archive that
feeds L2/L3's observation space (src/envs/lob_execution_env.py), not this
Binance-aggregate pipeline. Inventing an OBI proxy from L1-only data would
misrepresent a genuine book-imbalance signal as one this module cannot
actually see, so it is omitted here, not faked. A future orchestrator
wiring this into the live loop should merge in a real OBI figure from the
L2/L3 side separately, if that pairing is ever wanted.

Real data inventory (verified directly against data/raw_l1 on disk, not
assumed):
  - klines_1m: 2019-12-31 through the latest collected day, zero missing
    months/days against the full expected range, zero duplicate open_time
    rows (full sweep across every file).
  - funding_rate: 2020-01 through the last fully-completed month, zero
    gaps, zero duplicate calc_time rows (full sweep). Monthly archive
    only, 8h cadence confirmed from calc_time deltas.
  - open_interest: 2020-09-01 through the latest collected day, zero
    missing DAYS, but 263 consecutive days from 2020-09-01 through
    2021-05-21 (100% of that span) have every row duplicated exactly once
    -- byte-identical across every non-timestamp column too, confirmed
    directly (not assumed) via a full groupby-and-compare sweep. An early
    collection-run artifact, not a real gap. Every file after 2021-05-21
    checked clean. _load_oi_window() below deduplicates on create_time
    unconditionally (keep-first) rather than special-casing the affected
    date range, since the fix is correct and free either way.

Point-in-time correctness: every window loaded here is filtered strictly
to timestamp <= as_of_ms -- build_l1_feature_summary() must never be
handed lookahead data, mirroring src/data/split.py's chronological-only
discipline for the same underlying reason (a signal trained or evaluated
partly on future information is not honestly evaluated).

Missing/insufficient-history handling: every derived feature is set to
None (-> JSON null in the eventual Ollama prompt) when its window has too
little real data to compute, rather than a silently-propagated NaN
(json.dumps would happily emit a literal NaN token for a float NaN, which
is not valid JSON and would corrupt the very payload L1MacroAnalyst
treats as trusted input). A window is only trusted once it covers at
least _MIN_WINDOW_COVERAGE_FRAC of its nominal span -- this guards
against a field labeled e.g. "24h change" silently reporting a
much-shorter, gap-shrunken span under that label. This mirrors the
shrinking-window convention already used by
LOBExecutionEnv._rolling_rms/_rolling_sum (partial windows near the start
of an episode), applied here to partial windows near the start/end of the
real collected date range instead.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

RAW_L1_DIR = Path("data/raw_l1")

_RETURN_VOL_WINDOWS: dict[str, timedelta] = {"1h": timedelta(hours=1), "24h": timedelta(hours=24)}
_TAKER_FLOW_WINDOW = timedelta(hours=1)

_FUNDING_Z_LOOKBACK_PERIODS = 90
_FUNDING_LOOKBACK_BUFFER = timedelta(days=35)

_OI_CHANGE_WINDOW = timedelta(hours=24)

_MIN_WINDOW_COVERAGE_FRAC = 0.8


def _month_start(d: date) -> date:
    return d.replace(day=1)


def _next_month(d: date) -> date:
    return date(d.year + (d.month == 12), d.month % 12 + 1, 1)


def _iter_month_starts(start: date, end: date):
    m = _month_start(start)
    while m <= end:
        yield m
        m = _next_month(m)


def _ms_to_utc_date(ms: int) -> date:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).date()


def _has_min_coverage(first_ts_ms: int, last_ts_ms: int, window: timedelta) -> bool:
    span_ms = last_ts_ms - first_ts_ms
    return span_ms >= window.total_seconds() * 1000 * _MIN_WINDOW_COVERAGE_FRAC


def _load_klines_window(as_of_ms: int, lookback: timedelta, klines_dir: Path) -> pd.DataFrame:
    """Loads klines_1m rows in (as_of_ms - lookback, as_of_ms], trying the monthly
    archive first and falling back to daily files for any month where it is absent --
    mirroring scripts/collect_l1_data.py's own monthly-first/daily-fallback logic for
    this same dataset."""
    cols = ["open_time", "close", "volume", "taker_buy_volume"]
    start_ms = as_of_ms - int(lookback.total_seconds() * 1000)
    start_d, end_d = _ms_to_utc_date(start_ms), _ms_to_utc_date(as_of_ms)

    frames = []
    for month in _iter_month_starts(start_d, end_d):
        monthly_path = klines_dir / f"BTCUSDT-1m-{month.year:04d}-{month.month:02d}.parquet"
        if monthly_path.exists():
            frames.append(pd.read_parquet(monthly_path, columns=cols))
            continue
        month_end = _next_month(month) - timedelta(days=1)
        d = max(month, start_d)
        while d <= min(month_end, end_d):
            daily_path = klines_dir / f"BTCUSDT-1m-{d.isoformat()}.parquet"
            if daily_path.exists():
                frames.append(pd.read_parquet(daily_path, columns=cols))
            d += timedelta(days=1)

    if not frames:
        return pd.DataFrame(columns=cols)
    df = pd.concat(frames, ignore_index=True)
    df = df[(df["open_time"] > start_ms) & (df["open_time"] <= as_of_ms)]
    return df.sort_values("open_time").drop_duplicates(subset=["open_time"]).reset_index(drop=True)


def _load_funding_window(as_of_ms: int, lookback: timedelta, funding_dir: Path) -> pd.DataFrame:
    """Loads funding_rate rows in (as_of_ms - lookback, as_of_ms]. Monthly archive only
    (scripts/collect_l1_data.py: funding rate has no daily fallback on Binance's side) --
    a month with no published archive yet (e.g. the current in-progress month)
    contributes no rows, which is an expected condition, not an error."""
    cols = ["calc_time", "last_funding_rate"]
    start_ms = as_of_ms - int(lookback.total_seconds() * 1000)
    start_d, end_d = _ms_to_utc_date(start_ms), _ms_to_utc_date(as_of_ms)

    frames = []
    for month in _iter_month_starts(start_d, end_d):
        path = funding_dir / f"BTCUSDT-funding_rate-{month.year:04d}-{month.month:02d}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path, columns=cols))

    if not frames:
        return pd.DataFrame(columns=cols)
    df = pd.concat(frames, ignore_index=True)
    df = df[(df["calc_time"] > start_ms) & (df["calc_time"] <= as_of_ms)]
    return df.sort_values("calc_time").drop_duplicates(subset=["calc_time"]).reset_index(drop=True)


def _load_oi_window(as_of_ms: int, lookback: timedelta, oi_dir: Path) -> pd.DataFrame:
    """Loads open_interest rows in (as_of_ms - lookback, as_of_ms]. Daily files only.
    Deduplicates on create_time unconditionally (keep-first) -- see module docstring
    for the confirmed 2020-09-01..2021-05-21 exact-duplicate-row artifact this guards
    against; the same dedup is a correct no-op on every clean file too."""
    cols = ["create_time", "sum_open_interest", "sum_toptrader_long_short_ratio", "sum_taker_long_short_vol_ratio"]
    start_ms = as_of_ms - int(lookback.total_seconds() * 1000)
    start_d, end_d = _ms_to_utc_date(start_ms), _ms_to_utc_date(as_of_ms)

    frames = []
    d = start_d
    while d <= end_d:
        path = oi_dir / f"BTCUSDT-open_interest-{d.isoformat()}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path, columns=cols))
        d += timedelta(days=1)

    out_cols = ["create_time_ms"] + cols[1:]
    if not frames:
        return pd.DataFrame(columns=out_cols)
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["create_time"], keep="first")
    df["create_time_ms"] = pd.to_datetime(df["create_time"], utc=True).astype("int64") // 1_000_000
    df = df[(df["create_time_ms"] > start_ms) & (df["create_time_ms"] <= as_of_ms)]
    return df.sort_values("create_time_ms")[out_cols].reset_index(drop=True)


def _return_pct(closes: pd.Series) -> float | None:
    if len(closes) < 2:
        return None
    first, last = float(closes.iloc[0]), float(closes.iloc[-1])
    if first <= 0:
        return None
    return (last - first) / first


def _realized_vol(closes: pd.Series) -> float | None:
    """RMS of 1-minute log returns -- same convention as
    LOBExecutionEnv._rolling_rms's realized_vol_60s_z (a near-zero-mean series, so RMS
    rather than STD around a possibly-nonzero mean)."""
    if len(closes) < 2:
        return None
    log_ret = np.diff(np.log(closes.to_numpy(dtype=float)))
    if len(log_ret) == 0:
        return None
    return float(np.sqrt(np.mean(log_ret ** 2)))


def _taker_flow_imbalance(volume: pd.Series, taker_buy_volume: pd.Series) -> float | None:
    """(taker_buy - taker_sell) / total volume over the window -- same [-1,1]-imbalance
    shape as architecture_spec.md Section 2.3's OBI formula, but built from Binance's
    own taker-side trade volume (klines_1m) rather than order-book depth, since
    data/raw_l1 has none -- see module docstring's OBI-omission note."""
    total = float(volume.sum())
    if total <= 0:
        return None
    taker_buy = float(taker_buy_volume.sum())
    return (2.0 * taker_buy - total) / total


def build_l1_feature_summary(as_of_ms: int, raw_l1_dir: Path = RAW_L1_DIR) -> dict:
    """Point-in-time feature_summary for L1MacroAnalyst.maybe_refresh(). Every window is
    filtered strictly to timestamp <= as_of_ms -- see module docstring's point-in-time
    correctness note. Any field that cannot be computed from real data at as_of_ms is
    explicitly None, not a silently-propagated NaN -- see module docstring."""
    klines_dir = raw_l1_dir / "klines_1m"
    funding_dir = raw_l1_dir / "funding_rate"
    oi_dir = raw_l1_dir / "open_interest"

    summary: dict = {"as_of_ms": as_of_ms}

    kl_lookback = max(*_RETURN_VOL_WINDOWS.values(), _TAKER_FLOW_WINDOW)
    kl_df = _load_klines_window(as_of_ms, kl_lookback, klines_dir)

    for label, window in _RETURN_VOL_WINDOWS.items():
        start_ms = as_of_ms - int(window.total_seconds() * 1000)
        sub = kl_df[kl_df["open_time"] > start_ms]
        if sub.empty or not _has_min_coverage(int(sub["open_time"].iloc[0]), as_of_ms, window):
            summary[f"return_{label}_pct"] = None
            summary[f"realized_vol_{label}"] = None
        else:
            summary[f"return_{label}_pct"] = _return_pct(sub["close"])
            summary[f"realized_vol_{label}"] = _realized_vol(sub["close"])

    tf_start_ms = as_of_ms - int(_TAKER_FLOW_WINDOW.total_seconds() * 1000)
    tf_sub = kl_df[kl_df["open_time"] > tf_start_ms]
    if tf_sub.empty or not _has_min_coverage(int(tf_sub["open_time"].iloc[0]), as_of_ms, _TAKER_FLOW_WINDOW):
        summary["taker_flow_imbalance_1h"] = None
    else:
        summary["taker_flow_imbalance_1h"] = _taker_flow_imbalance(tf_sub["volume"], tf_sub["taker_buy_volume"])

    fr_df = _load_funding_window(as_of_ms, _FUNDING_LOOKBACK_BUFFER, funding_dir)
    if fr_df.empty:
        summary["funding_rate_current"] = None
        summary["funding_rate_z"] = None
    else:
        current = float(fr_df["last_funding_rate"].iloc[-1])
        hist = fr_df["last_funding_rate"].tail(_FUNDING_Z_LOOKBACK_PERIODS)
        summary["funding_rate_current"] = current
        if len(hist) < 2:
            summary["funding_rate_z"] = None
        else:
            std = float(hist.std())
            summary["funding_rate_z"] = None if std == 0 else float((current - hist.mean()) / std)

    oi_df = _load_oi_window(as_of_ms, _OI_CHANGE_WINDOW, oi_dir)
    if oi_df.empty:
        summary["open_interest_level"] = None
        summary["open_interest_change_24h_pct"] = None
        summary["top_trader_long_short_ratio"] = None
        summary["taker_long_short_vol_ratio"] = None
    else:
        current_row, first_row = oi_df.iloc[-1], oi_df.iloc[0]
        first_oi = float(first_row["sum_open_interest"])
        summary["open_interest_level"] = float(current_row["sum_open_interest"])
        summary["open_interest_change_24h_pct"] = (
            None
            if first_oi <= 0
            or not _has_min_coverage(int(first_row["create_time_ms"]), as_of_ms, _OI_CHANGE_WINDOW)
            else float((current_row["sum_open_interest"] - first_oi) / first_oi)
        )
        summary["top_trader_long_short_ratio"] = float(current_row["sum_toptrader_long_short_ratio"])
        summary["taker_long_short_vol_ratio"] = float(current_row["sum_taker_long_short_vol_ratio"])

    return summary
