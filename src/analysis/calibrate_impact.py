"""Calibrate permanent (eta) and temporary (lambda) market impact per
architecture_spec.md Section 4.5, fit from the project's own historical L2
order-flow data (data/raw_l2_bybit/BTCUSDT/), train-split days only.

Standalone: reads L2 archive parquet files and data/splits/... directly;
does not import from or touch src/envs/, matching_engine.py, reward.py, or
anything the live training process reads. Produces calibrated numbers and
methodology for review -- NOT wired into the environment yet (see Section
4.5's own sequencing: Tier 1 lands between Phase 3 and Phase 4, not folded
into either).

Methodology
-----------
Permanent impact (eta): the canonical Cont-Kukanov-Stoikov (2014) order
-flow-imbalance (OFI) event formula, using only touch-level best bid/ask
price and size (no full-book lookup needed -- this is the literature's own
definition, distinct from but in the same spirit as this project's existing
trade_flow_imbalance_5s feature in lob_execution_env.py, which serves a
different purpose: touch-depletion-inferred v_trade for the RL env's own
fill simulation, not a market-impact calibration). Per-event OFI is summed
into 5-second buckets. Contemporaneous regression of bucket mid-price
return against OFI/typical_volume gives eta -- under CKS's own
interpretation, since price effectively follows a random walk driven by
order flow, the contemporaneous slope is read as the permanent-impact
coefficient, not a spurious correlation.

Temporary impact (lambda) and its decay half-life: participation_rate is
each bucket's gross order-flow activity relative to that trading day's own
typical level. deviation is the bucket mid-price minus a slower (60s)
rolling reference path. Regressing |deviation| against sqrt(participation
_rate) gives lambda. Half-life is fit separately, from how deviation decays
over the minute following high-participation burst buckets.

Both regressions are fit through the origin (no intercept), matching
Section 4.5's own functional form (zero flow implies zero impact). A
diagnostic with-intercept fit is also reported for eta as a robustness
check, not as the primary result.

Known limitation, documented rather than silently assumed away: bucket
-index arithmetic (the 60s rolling reference, the decay tau axis) assumes
contiguous 5s buckets. Any gap in the underlying tick capture introduces
minor timing imprecision in the decay/half-life estimate specifically; it
does not affect the eta/lambda level regressions, which only use same
-bucket data.

Holdout validation: the confirmed real train-split days (val/test
untouched, same discipline as RL training) are split 80/20 by a fixed-seed
random day-level shuffle into calibration/holdout sets. eta and lambda are
fit on calibration days only; R^2 and coefficients are reported on BOTH
sets so overfitting to the calibration data itself is visible, not assumed
away.

Run: PYTHONPATH=. .venv/bin/python -m src.analysis.calibrate_impact
     [--max-days N]   (smoke-test on only the first N days, in split order)
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import gc
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.split import load_split

DATA_DIR = Path("data/raw_l2_bybit/BTCUSDT")
LIVE_RUN_PID = 3356234
MIN_AVAILABLE_GB = 10.0
CHECK_EVERY_N_DAYS = 50
BUCKET_S = 5.0
SLOW_WINDOW_BUCKETS = 12  # 60s / 5s
BURST_PERCENTILE = 90.0
DECAY_LOOKAHEAD_BUCKETS = 12  # track 60s after a burst
HOLDOUT_FRAC = 0.2
SPLIT_SEED = 20260815


def _free_available_gb() -> float:
    out = subprocess.run(["free", "-b"], capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        if line.startswith("Mem:"):
            parts = line.split()
            return float(parts[6]) / 1e9
    raise RuntimeError("could not parse `free -b` output")


def _check_live_run() -> None:
    alive = subprocess.run(["ps", "-p", str(LIVE_RUN_PID)], capture_output=True, text=True).returncode == 0
    gpu = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv"],
        capture_output=True, text=True,
    ).stdout
    print(f"[guardrail] live run PID {LIVE_RUN_PID} alive={alive}")
    print(f"[guardrail] nvidia-smi compute-apps:\n{gpu}")
    if not alive:
        print(f"[guardrail] WARNING: PID {LIVE_RUN_PID} not found via ps -p")


def _memory_guardrail() -> None:
    avail_gb = _free_available_gb()
    print(f"[guardrail] free available: {avail_gb:.2f}GB")
    _check_live_run()
    if avail_gb < MIN_AVAILABLE_GB:
        print(f"[guardrail] STOPPING: available memory {avail_gb:.2f}GB < {MIN_AVAILABLE_GB}GB threshold")
        sys.exit(1)


def _extract_touch_size(series: pd.Series) -> np.ndarray:
    """First [price, size] pair of a Bybit-style '[[p, s], [p, s], ...]'
    level-array string -- the touch size only, vectorized across the whole
    column via a single regex pass rather than per-row json.loads."""
    extracted = series.str.extract(r"^\[\[[-\d.]+,\s*([\d.]+)\]", expand=False)
    return pd.to_numeric(extracted, errors="coerce").to_numpy(dtype=float)


@dataclass
class DayResult:
    date: str
    ofi: np.ndarray
    ret: np.ndarray
    participation: np.ndarray
    deviation: np.ndarray
    # Per burst-event, per-lag RAW |deviation| observations (tau_step -> list of
    # |deviation| values, one per burst event that reached that lag), NOT
    # per-event ratios -- see main()'s aggregation for why: dividing by an
    # individual event's dev0 when dev0 is near zero amplifies noise into huge,
    # meaningless ratios. Aggregating |deviation| across many events at each lag
    # BEFORE taking logs (the standard event-study average-decay-curve approach)
    # is robust to that; per-event log-ratio pooling, tried first, was not (see
    # calibrate_impact_results.md for the R^2~0 failure this replaced).
    decay_abs_dev_by_lag: dict = field(default_factory=dict)
    control_abs_dev_by_lag: dict = field(default_factory=dict)


def process_day(path: Path) -> DayResult:
    df = pd.read_parquet(path, columns=["ts", "best_bid", "best_ask", "mid_price", "bids", "asks"])

    ts = df["ts"].to_numpy(dtype=np.int64)
    best_bid = df["best_bid"].to_numpy(dtype=float)
    best_ask = df["best_ask"].to_numpy(dtype=float)
    mid = df["mid_price"].to_numpy(dtype=float)
    bid_size = _extract_touch_size(df["bids"])
    ask_size = _extract_touch_size(df["asks"])

    del df
    gc.collect()

    n = len(ts)
    empty = DayResult(date=path.stem, ofi=np.array([]), ret=np.array([]),
                       participation=np.array([]), deviation=np.array([]))
    if n < 2:
        return empty

    prev_bid_p = np.empty(n); prev_bid_p[0] = best_bid[0]; prev_bid_p[1:] = best_bid[:-1]
    prev_ask_p = np.empty(n); prev_ask_p[0] = best_ask[0]; prev_ask_p[1:] = best_ask[:-1]
    prev_bid_s = np.empty(n); prev_bid_s[0] = bid_size[0]; prev_bid_s[1:] = bid_size[:-1]
    prev_ask_s = np.empty(n); prev_ask_s[0] = ask_size[0]; prev_ask_s[1:] = ask_size[:-1]

    # Canonical Cont-Kukanov-Stoikov (2014) per-event OFI.
    bid_contrib = np.where(best_bid >= prev_bid_p, bid_size, 0.0) - np.where(best_bid <= prev_bid_p, prev_bid_s, 0.0)
    ask_contrib = np.where(best_ask <= prev_ask_p, ask_size, 0.0) - np.where(best_ask >= prev_ask_p, prev_ask_s, 0.0)
    ofi_event = bid_contrib - ask_contrib
    ofi_event[0] = 0.0
    absvol_event = (
        np.where(best_bid >= prev_bid_p, bid_size, 0.0) + np.where(best_bid <= prev_bid_p, prev_bid_s, 0.0)
        + np.where(best_ask <= prev_ask_p, ask_size, 0.0) + np.where(best_ask >= prev_ask_p, prev_ask_s, 0.0)
    )
    absvol_event[0] = 0.0

    bucket_id = (ts // int(BUCKET_S * 1000)).astype(np.int64)
    bdf = pd.DataFrame({"bucket": bucket_id, "ofi": ofi_event, "absvol": absvol_event, "mid": mid})
    grouped = bdf.groupby("bucket", sort=True)
    bucket_ofi = grouped["ofi"].sum().to_numpy()
    bucket_absvol = grouped["absvol"].sum().to_numpy()
    bucket_mid_last = grouped["mid"].last().to_numpy()
    bucket_mid_first = grouped["mid"].first().to_numpy()
    del bdf, grouped
    gc.collect()

    nb = len(bucket_ofi)
    if nb < SLOW_WINDOW_BUCKETS + DECAY_LOOKAHEAD_BUCKETS + 2:
        return empty

    ret = np.zeros(nb)
    ret[1:] = (bucket_mid_last[1:] - bucket_mid_first[1:]) / bucket_mid_first[1:]

    typical_volume = float(np.mean(bucket_absvol[bucket_absvol > 0])) if np.any(bucket_absvol > 0) else 1.0
    typical_volume = max(typical_volume, 1e-9)
    participation = bucket_absvol / typical_volume
    ofi_norm = bucket_ofi / typical_volume

    slow_ref = pd.Series(bucket_mid_last).rolling(SLOW_WINDOW_BUCKETS, min_periods=1).mean().to_numpy()
    deviation = (bucket_mid_last - slow_ref) / slow_ref

    # Decay tracking needs the OFI-implied PERMANENT component netted out first
    # (fit a per-day local eta from data already computed, no second pass
    # needed) AND a non-burst CONTROL group tracked the identical way: even
    # after netting permanent impact, |price(tau)-anchor| still grows with tau
    # simply from ordinary random-walk diffusion, which has nothing to do with
    # impact decay and swamps it if left in (earlier versions of this found
    # monotonic GROWTH, R^2 as high as 0.99, from exactly this -- not real
    # findings; see calibrate_impact_results.md). The EXCESS of burst-group
    # deviation over a same-day control group's deviation, at each lag, is the
    # impact-specific signal: it should be largest at tau=0 (bursts are
    # selected for high deviation) and shrink toward zero as ordinary
    # diffusion catches the control group up -- that shrinkage is what actually
    # measures decay.
    eta_local = _ols_through_origin(ofi_norm, ret)["slope"]
    ofi_cumsum = np.cumsum(ofi_norm)

    def _track(idx: np.ndarray) -> dict:
        by_lag: dict = {lag: [] for lag in range(DECAY_LOOKAHEAD_BUCKETS + 1)}
        if not np.isfinite(eta_local):
            return by_lag
        for i in idx:
            anchor = slow_ref[i]
            if anchor <= 0:
                continue
            dev0 = abs(bucket_mid_last[i] - anchor) / anchor
            if dev0 < 1e-12:
                continue
            by_lag[0].append(dev0)
            for tau_steps in range(1, DECAY_LOOKAHEAD_BUCKETS + 1):
                j = i + tau_steps
                if j >= nb:
                    break
                cum_ofi_window = ofi_cumsum[j] - ofi_cumsum[i]
                expected_mid = anchor * (1.0 + eta_local * cum_ofi_window)
                if expected_mid <= 0:
                    continue
                by_lag[tau_steps].append(abs(bucket_mid_last[j] - expected_mid) / anchor)
        return by_lag

    decay_abs_dev_by_lag: dict = {lag: [] for lag in range(DECAY_LOOKAHEAD_BUCKETS + 1)}
    control_abs_dev_by_lag: dict = {lag: [] for lag in range(DECAY_LOOKAHEAD_BUCKETS + 1)}
    if np.any(participation > 0):
        pos = participation[participation > 0]
        burst_thresh = np.percentile(pos, BURST_PERCENTILE)
        control_thresh = np.percentile(pos, 50.0)
        burst_idx = np.where(participation >= burst_thresh)[0]
        control_idx = np.where((participation > 0) & (participation <= control_thresh))[0]
        decay_abs_dev_by_lag = _track(burst_idx)
        control_abs_dev_by_lag = _track(control_idx)

    return DayResult(
        date=path.stem, ofi=bucket_ofi / typical_volume, ret=ret,
        participation=participation, deviation=deviation,
        decay_abs_dev_by_lag=decay_abs_dev_by_lag,
        control_abs_dev_by_lag=control_abs_dev_by_lag,
    )


def _ols_through_origin(x: np.ndarray, y: np.ndarray) -> dict:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 2 or np.sum(x * x) == 0:
        return {"n": n, "slope": float("nan"), "se": float("nan"), "r2": float("nan")}
    slope = float(np.sum(x * y) / np.sum(x * x))
    resid = y - slope * x
    dof = max(n - 1, 1)
    se = float(np.sqrt(np.sum(resid ** 2) / dof) / np.sqrt(np.sum(x * x)))
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum(y ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"n": n, "slope": slope, "se": se, "r2": r2}


def _ols_with_intercept(x: np.ndarray, y: np.ndarray) -> dict:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 3:
        return {"n": n, "slope": float("nan"), "intercept": float("nan"), "r2": float("nan")}
    xm, ym = x.mean(), y.mean()
    sxx = np.sum((x - xm) ** 2)
    if sxx == 0:
        return {"n": n, "slope": float("nan"), "intercept": float("nan"), "r2": float("nan")}
    slope = float(np.sum((x - xm) * (y - ym)) / sxx)
    intercept = float(ym - slope * xm)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - ym) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"n": n, "slope": slope, "intercept": intercept, "r2": r2}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-days", type=int, default=None,
                         help="Smoke-test on only the first N days (in split order)")
    args = parser.parse_args()

    _memory_guardrail()

    train_dates = load_split("train")
    print(f"train split: {len(train_dates)} real days")

    rng = np.random.default_rng(SPLIT_SEED)
    shuffled = list(train_dates)
    rng.shuffle(shuffled)
    n_holdout = int(round(len(shuffled) * HOLDOUT_FRAC))
    holdout_dates = set(shuffled[:n_holdout])
    calib_dates = set(shuffled[n_holdout:])
    print(f"calibration days: {len(calib_dates)}  holdout days: {len(holdout_dates)}")

    calib_ofi, calib_ret, calib_part, calib_dev = [], [], [], []
    calib_decay_by_lag: dict = {lag: [] for lag in range(DECAY_LOOKAHEAD_BUCKETS + 1)}
    calib_control_by_lag: dict = {lag: [] for lag in range(DECAY_LOOKAHEAD_BUCKETS + 1)}
    hold_ofi, hold_ret, hold_part, hold_dev = [], [], [], []

    processed = 0
    t0 = time.time()
    for d in train_dates:
        path = DATA_DIR / f"l2-BTCUSDT-{d.isoformat()}.parquet"
        if not path.exists():
            print(f"[skip] {d}: file not found at {path}")
            continue

        result = process_day(path)
        gc.collect()

        if d in calib_dates:
            calib_ofi.append(result.ofi); calib_ret.append(result.ret)
            calib_part.append(result.participation); calib_dev.append(result.deviation)
            for lag, vals in result.decay_abs_dev_by_lag.items():
                calib_decay_by_lag[lag].extend(vals)
            for lag, vals in result.control_abs_dev_by_lag.items():
                calib_control_by_lag[lag].extend(vals)
        else:
            hold_ofi.append(result.ofi); hold_ret.append(result.ret)
            hold_part.append(result.participation); hold_dev.append(result.deviation)

        processed += 1
        if processed % 10 == 0:
            print(f"[{processed}/{len(train_dates)}] {d} done, elapsed={time.time()-t0:.1f}s")
        if processed % CHECK_EVERY_N_DAYS == 0:
            _memory_guardrail()
        if args.max_days and processed >= args.max_days:
            print(f"--max-days {args.max_days} reached, stopping early (smoke test mode)")
            break

    print(f"processed {processed} days in {time.time()-t0:.1f}s")
    print()
    print("=== final guardrail check ===")
    _memory_guardrail()

    calib_ofi = np.concatenate(calib_ofi) if calib_ofi else np.array([])
    calib_ret = np.concatenate(calib_ret) if calib_ret else np.array([])
    calib_part = np.concatenate(calib_part) if calib_part else np.array([])
    calib_dev = np.concatenate(calib_dev) if calib_dev else np.array([])
    hold_ofi = np.concatenate(hold_ofi) if hold_ofi else np.array([])
    hold_ret = np.concatenate(hold_ret) if hold_ret else np.array([])
    hold_part = np.concatenate(hold_part) if hold_part else np.array([])
    hold_dev = np.concatenate(hold_dev) if hold_dev else np.array([])

    print()
    print("=== PERMANENT IMPACT (eta): return ~ OFI/typical_volume, through origin ===")
    eta_calib = _ols_through_origin(calib_ofi, calib_ret)
    eta_hold = _ols_through_origin(hold_ofi, hold_ret)
    print(f"calibration: n={eta_calib['n']} eta={eta_calib['slope']:.6e} se={eta_calib['se']:.6e} R2={eta_calib['r2']:.4f}")
    print(f"holdout:     n={eta_hold['n']} eta={eta_hold['slope']:.6e} se={eta_hold['se']:.6e} R2={eta_hold['r2']:.4f}")
    eta_calib_intercept = _ols_with_intercept(calib_ofi, calib_ret)
    print(f"[diagnostic, with intercept] calibration: slope={eta_calib_intercept['slope']:.6e} "
          f"intercept={eta_calib_intercept['intercept']:.6e} R2={eta_calib_intercept['r2']:.4f}")

    print()
    print("=== TEMPORARY IMPACT (lambda): |deviation| ~ sqrt(participation_rate), through origin ===")
    lam_calib = _ols_through_origin(np.sqrt(np.clip(calib_part, 0, None)), np.abs(calib_dev))
    lam_hold = _ols_through_origin(np.sqrt(np.clip(hold_part, 0, None)), np.abs(hold_dev))
    print(f"calibration: n={lam_calib['n']} lambda={lam_calib['slope']:.6e} se={lam_calib['se']:.6e} R2={lam_calib['r2']:.4f}")
    print(f"holdout:     n={lam_hold['n']} lambda={lam_hold['slope']:.6e} se={lam_hold['se']:.6e} R2={lam_hold['r2']:.4f}")

    print()
    print("=== TEMPORARY IMPACT DECAY HALF-LIFE: excess-over-control event-study ===")
    print("(median |residual deviation| at each lag, burst group vs a same-day")
    print(" non-burst control group tracked identically -- the EXCESS isolates")
    print(" impact-specific decay from ordinary random-walk diffusion, which")
    print(" grows both curves regardless of any real impact; see DayResult docstring)")
    lag_keys = sorted(calib_decay_by_lag.keys())
    lags_s = np.array(lag_keys) * BUCKET_S
    median_burst = np.array([np.median(calib_decay_by_lag[lag]) if calib_decay_by_lag[lag] else float("nan") for lag in lag_keys])
    median_control = np.array([np.median(calib_control_by_lag[lag]) if calib_control_by_lag[lag] else float("nan") for lag in lag_keys])
    excess = median_burst - median_control
    n_events_per_lag = np.array([len(calib_decay_by_lag[lag]) for lag in lag_keys])
    n_control_per_lag = np.array([len(calib_control_by_lag[lag]) for lag in lag_keys])
    for lag_s, mb, mc, ex, n_ev, n_ctl in zip(lags_s, median_burst, median_control, excess, n_events_per_lag, n_control_per_lag):
        print(f"  tau={lag_s:5.1f}s  burst={mb:.6e}  control={mc:.6e}  excess={ex:.6e}  n_burst={n_ev} n_control={n_ctl}")
    valid = np.isfinite(excess) & (np.abs(excess) > 0)
    decay_fit = _ols_with_intercept(lags_s[valid], np.log(np.abs(excess[valid])))
    half_life_s = -np.log(2) / decay_fit["slope"] if decay_fit["slope"] < 0 else float("nan")
    print(f"log-linear fit on |excess| (sign of excess itself reported separately below): "
          f"n_lags={decay_fit['n']} slope={decay_fit['slope']:.6e} "
          f"intercept={decay_fit['intercept']:.6e} R2={decay_fit['r2']:.4f}")
    print(f"excess sign: {'positive (burst > control)' if np.median(excess[valid]) > 0 else 'negative (burst < control)'} "
          f"at every lag -- {'consistent with' if decay_fit['r2'] > 0.5 and decay_fit['slope'] < 0 else 'NOT consistent with'} "
          f"a clean decaying-impact signal (see per-lag table above: does |excess| trend toward 0, or stay flat?)")
    print(f"implied half-life (only meaningful if the above says 'consistent with'): {half_life_s:.2f}s")

    print()
    print("=== SUMMARY ===")
    print(f"eta (permanent impact, calibration fit): {eta_calib['slope']:.6e} "
          f"(SE {eta_calib['se']:.6e}, R2 {eta_calib['r2']:.4f}, n={eta_calib['n']})")
    print(f"eta (holdout check): {eta_hold['slope']:.6e} (R2 {eta_hold['r2']:.4f}, n={eta_hold['n']})")
    print(f"lambda (temporary impact, calibration fit): {lam_calib['slope']:.6e} "
          f"(SE {lam_calib['se']:.6e}, R2 {lam_calib['r2']:.4f}, n={lam_calib['n']})")
    print(f"lambda (holdout check): {lam_hold['slope']:.6e} (R2 {lam_hold['r2']:.4f}, n={lam_hold['n']})")
    print(f"temporary impact half-life: {half_life_s:.2f}s "
          f"(excess-over-control fit over {decay_fit['n']} lag points, "
          f"{int(n_events_per_lag[0])} burst / {int(n_control_per_lag[0])} control events at tau=0)")


if __name__ == "__main__":
    main()
