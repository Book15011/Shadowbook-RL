"""L1 aggregate-feature collection: klines, funding rate, open interest.

Pulls whatever is genuinely available from Binance USDT-margined futures for a
single symbol and writes Parquet under data/raw_l1/{klines_1m,funding_rate,open_interest}/.

All three datasets are pulled from data.binance.vision archives (zip+CHECKSUM,
same mechanism bulk_backfill.py uses) -- NOT from fapi.binance.com/api.binance.com/
www.binance.com REST endpoints. Those domains are blocked from this host's
network (verified: DNS for fapi.binance.com resolves to a non-routing address
through the configured proxy, api.binance.com/www.binance.com connection-reset
immediately), so an original REST-pagination design for funding rate and open
interest was replaced with archive-file pulls that only ever talk to
data.binance.vision, which is reachable.

Real availability differs per dataset (verified against the live archive, not
assumed):
  - klines: monthly archives from 2020-01 onward, plus a single leftover daily
    file for 2019-12-31 (before the monthly pipeline existed) and daily files
    for the current, not-yet-monthly-rolled-up month. Tries monthly first for
    each requested month and falls back to daily files for any month where
    the monthly archive 404s or the request only partially covers it.
  - funding rate: Binance's "fundingRate" archive, monthly-only (no daily
    variant exists), real earliest month is 2020-01. Because there is no daily
    fallback, the current in-progress month is unavailable until Binance
    publishes it after month-end -- default --funding-end is therefore the
    last fully-completed month, not "today".
  - open interest: Binance's "metrics" archive, daily-only (no monthly
    variant), real earliest day is 2020-09-01. This archive bundles open
    interest (sum_open_interest, sum_open_interest_value) together with
    top-trader and taker long/short ratios in the same file -- all columns are
    kept since they come for free. This replaces the old openInterestHist
    REST call (which additionally only had a ~30-day retention window; the
    archive has none -- real history back to 2020-09-01).

Usage:
    python scripts/collect_l1_data.py [--symbol BTCUSDT]
        [--klines-start 2019-12-31] [--klines-end 2026-08-11]
        [--funding-start 2020-01] [--funding-end 2026-07]
        [--oi-start 2020-09-01] [--oi-end 2026-08-11]
        [--out-dir data/raw_l1] [--max-retries 5] [--request-delay-s 0.3]
        [--datasets klines,funding_rate,open_interest]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import signal
import sys
import time
import zipfile
from calendar import monthrange
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

_LOG_FILE = Path("logs/l1_collection.log")


def _setup_logging(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)s %(message)s"
    logger = logging.getLogger("l1_collect")
    logger.setLevel(logging.DEBUG)
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(fmt))
    fh.stream.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter(fmt))
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


log = logging.getLogger("l1_collect")

_SHUTDOWN = {"requested": False}


def _handle_signal(sig, frame):  # noqa: ANN001
    log.warning("Signal %s received -- finishing current chunk then exiting", sig)
    _SHUTDOWN["requested"] = True


class Manifest:
    _NAME = "_manifest.jsonl"

    def __init__(self, out_dir: Path) -> None:
        self._path = out_dir / self._NAME
        self._entries: dict[tuple[str, str], dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        with open(self._path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    key = (entry["period"], entry["granularity"])
                    self._entries[key] = entry
                except (json.JSONDecodeError, KeyError):
                    pass

    def get(self, period: str, granularity: str) -> dict[str, Any] | None:
        return self._entries.get((period, granularity))

    def record(
        self,
        period: str,
        granularity: str,
        status: str,
        sha256: str = "",
        bytes_written: int = 0,
    ) -> None:
        entry = {
            "period": period,
            "granularity": granularity,
            "status": status,
            "sha256": sha256,
            "bytes": bytes_written,
            "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        self._entries[(period, granularity)] = entry
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a", encoding="utf-8", buffering=1) as fh:
            fh.write(json.dumps(entry) + "\n")

    def all_entries(self) -> list[dict[str, Any]]:
        return list(self._entries.values())


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_checksum(checksum_text: str) -> str:
    for token in checksum_text.split():
        if len(token) == 64 and all(c in "0123456789abcdef" for c in token):
            return token
    first = checksum_text.strip().splitlines()[0]
    return first.strip().split()[-1]


def _get_with_retry(url: str, max_retries: int, delay_s: float, timeout: int = 90) -> requests.Response:
    backoff = 1.0
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        if attempt > 0:
            sleep_time = backoff * (2 ** (attempt - 1))
            log.debug("Retry %d/%d in %.1fs for %s", attempt, max_retries, sleep_time, url)
            time.sleep(sleep_time)
        time.sleep(delay_s)
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 404:
                return resp
            if resp.status_code >= 500 or resp.status_code == 429:
                log.warning("HTTP %d for %s", resp.status_code, url)
                last_exc = requests.HTTPError(f"HTTP {resp.status_code}", response=resp)
                continue
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            log.warning("Request error for %s: %s", url, exc)
            last_exc = exc
    raise last_exc or RuntimeError(f"Failed to GET {url}")


def _archive_checksum_url(zip_url: str) -> str:
    return zip_url + ".CHECKSUM"


def _download_archive_zip(zip_url: str, max_retries: int, delay_s: float) -> bytes | None:
    csum_resp = _get_with_retry(_archive_checksum_url(zip_url), max_retries, delay_s)
    if csum_resp.status_code == 404:
        return None
    expected_sha = _parse_checksum(csum_resp.text)

    zip_resp = _get_with_retry(zip_url, max_retries, delay_s)
    if zip_resp.status_code == 404:
        return None

    zip_bytes = zip_resp.content
    actual_sha = _sha256_bytes(zip_bytes)
    if actual_sha != expected_sha:
        raise ValueError(f"checksum mismatch for {zip_url}")
    return zip_bytes


def _extract_single_csv(zip_bytes: bytes) -> bytes:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
        return zf.read(csv_name)


_KLINES_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "count",
    "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]


def _klines_monthly_url(symbol: str, interval: str, month: str) -> str:
    return (
        f"https://data.binance.vision/data/futures/um/monthly/klines/"
        f"{symbol}/{interval}/{symbol}-{interval}-{month}.zip"
    )


def _klines_daily_url(symbol: str, interval: str, day: str) -> str:
    return (
        f"https://data.binance.vision/data/futures/um/daily/klines/"
        f"{symbol}/{interval}/{symbol}-{interval}-{day}.zip"
    )


def _read_klines_csv(csv_bytes: bytes) -> pd.DataFrame:
    first_line = csv_bytes.split(b"\n", 1)[0]
    first_field = first_line.split(b",", 1)[0].decode("utf-8", errors="replace")
    has_header = not first_field.strip().lstrip("-").isdigit()
    df = pd.read_csv(
        io.BytesIO(csv_bytes),
        header=0 if has_header else None,
        names=None if has_header else _KLINES_COLUMNS,
    )
    if has_header:
        df.columns = _KLINES_COLUMNS
    return df


def _write_klines_parquet(zip_bytes: bytes, out_path: Path) -> int:
    csv_bytes = _extract_single_csv(zip_bytes)
    df = _read_klines_csv(csv_bytes)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return out_path.stat().st_size


def _month_range(start: date, end: date) -> list[str]:
    months = []
    cur = date(start.year, start.month, 1)
    while cur <= end:
        months.append(cur.strftime("%Y-%m"))
        year, month = (cur.year + 1, 1) if cur.month == 12 else (cur.year, cur.month + 1)
        cur = date(year, month, 1)
    return months


def _dates_in_range(start: date, end: date) -> list[date]:
    out = []
    cur = start
    while cur <= end:
        out.append(cur)
        cur += timedelta(days=1)
    return out


def collect_klines(
    symbol: str,
    interval: str,
    start: date,
    end: date,
    out_dir: Path,
    max_retries: int,
    delay_s: float,
) -> dict[str, Any]:
    log.info("Starting klines collection: symbol=%s interval=%s start=%s end=%s", symbol, interval, start, end)
    dataset_dir = out_dir / "klines_1m"
    manifest = Manifest(dataset_dir)

    counters = {"ok": 0, "missing": 0, "failed": 0}
    covered_days: list[date] = []

    for month in _month_range(start, end):
        if _SHUTDOWN["requested"]:
            break
        year, mon = (int(x) for x in month.split("-"))
        month_first = date(year, mon, 1)
        month_last = date(year, mon, monthrange(year, mon)[1])
        req_first = max(month_first, start)
        req_last = min(month_last, end)

        existing = manifest.get(month, "monthly")
        parquet_path = dataset_dir / f"{symbol}-{interval}-{month}.parquet"
        if existing and existing.get("status") == "ok" and parquet_path.exists() and _sha256_file(parquet_path) == existing.get("sha256"):
            log.info("SKIP klines monthly %s (already ok)", month)
            counters["ok"] += 1
            covered_days.extend(_dates_in_range(req_first, req_last))
            continue
        if existing and existing.get("status") == "missing_use_daily":
            log.info("SKIP klines monthly %s (previously confirmed no monthly archive; using daily)", month)
            c, days = _collect_klines_daily_range(symbol, interval, req_first, req_last, dataset_dir, manifest, max_retries, delay_s)
            for k, v in c.items():
                counters[k] += v
            covered_days.extend(days)
            continue

        if req_first != month_first or req_last != month_last:
            log.info("Month %s is a partial request range (%s..%s) -- using daily files", month, req_first, req_last)
            c, days = _collect_klines_daily_range(symbol, interval, req_first, req_last, dataset_dir, manifest, max_retries, delay_s)
            for k, v in c.items():
                counters[k] += v
            covered_days.extend(days)
            continue

        log.info("Starting download for klines monthly %s", month)
        try:
            zip_bytes = _download_archive_zip(_klines_monthly_url(symbol, interval, month), max_retries, delay_s)
        except Exception as exc:
            log.error("Failed to fetch monthly klines %s: %s", month, exc)
            manifest.record(month, "monthly", "failed")
            counters["failed"] += 1
            continue

        if zip_bytes is None:
            log.info("No monthly archive for %s (404) -- falling back to daily files for this month", month)
            manifest.record(month, "monthly", "missing_use_daily")
            c, days = _collect_klines_daily_range(symbol, interval, req_first, req_last, dataset_dir, manifest, max_retries, delay_s)
            for k, v in c.items():
                counters[k] += v
            covered_days.extend(days)
            continue

        try:
            nbytes = _write_klines_parquet(zip_bytes, parquet_path)
        except Exception as exc:
            log.error("Failed to write parquet for monthly klines %s: %s", month, exc)
            manifest.record(month, "monthly", "failed")
            counters["failed"] += 1
            continue

        parquet_sha = _sha256_file(parquet_path)
        manifest.record(month, "monthly", "ok", sha256=parquet_sha, bytes_written=nbytes)
        log.info("Successfully wrote Parquet for klines monthly %s (%d bytes)", month, nbytes)
        counters["ok"] += 1
        covered_days.extend(_dates_in_range(req_first, req_last))

    real_start = min(covered_days) if covered_days else None
    real_end = max(covered_days) if covered_days else None
    log.info(
        "Klines collection done: ok=%d missing=%d failed=%d real_range=%s..%s",
        counters["ok"], counters["missing"], counters["failed"], real_start, real_end,
    )
    return {"counters": counters, "real_start": real_start, "real_end": real_end, "dataset_dir": dataset_dir}


def _collect_klines_daily_range(
    symbol: str,
    interval: str,
    start: date,
    end: date,
    dataset_dir: Path,
    manifest: Manifest,
    max_retries: int,
    delay_s: float,
) -> tuple[dict[str, int], list[date]]:
    counters = {"ok": 0, "missing": 0, "failed": 0}
    covered: list[date] = []
    cur = start
    while cur <= end:
        if _SHUTDOWN["requested"]:
            break
        day_str = cur.strftime("%Y-%m-%d")
        parquet_path = dataset_dir / f"{symbol}-{interval}-{day_str}.parquet"

        existing = manifest.get(day_str, "daily")
        if existing and existing.get("status") == "ok" and parquet_path.exists() and _sha256_file(parquet_path) == existing.get("sha256"):
            log.info("SKIP klines daily %s (already ok)", day_str)
            counters["ok"] += 1
            covered.append(cur)
            cur += timedelta(days=1)
            continue
        if existing and existing.get("status") == "missing":
            counters["missing"] += 1
            cur += timedelta(days=1)
            continue

        log.info("Starting download for klines daily %s", day_str)
        try:
            zip_bytes = _download_archive_zip(_klines_daily_url(symbol, interval, day_str), max_retries, delay_s)
        except Exception as exc:
            log.error("Failed to fetch daily klines %s: %s", day_str, exc)
            manifest.record(day_str, "daily", "failed")
            counters["failed"] += 1
            cur += timedelta(days=1)
            continue

        if zip_bytes is None:
            log.info("MISSING klines daily %s (404)", day_str)
            manifest.record(day_str, "daily", "missing")
            counters["missing"] += 1
            cur += timedelta(days=1)
            continue

        try:
            nbytes = _write_klines_parquet(zip_bytes, parquet_path)
        except Exception as exc:
            log.error("Failed to write parquet for daily klines %s: %s", day_str, exc)
            manifest.record(day_str, "daily", "failed")
            counters["failed"] += 1
            cur += timedelta(days=1)
            continue

        parquet_sha = _sha256_file(parquet_path)
        manifest.record(day_str, "daily", "ok", sha256=parquet_sha, bytes_written=nbytes)
        log.info("Successfully wrote Parquet for klines daily %s (%d bytes)", day_str, nbytes)
        counters["ok"] += 1
        covered.append(cur)
        cur += timedelta(days=1)

    return counters, covered


_FUNDING_RATE_COLUMNS = ["calc_time", "funding_interval_hours", "last_funding_rate"]


def _funding_rate_monthly_url(symbol: str, month: str) -> str:
    return (
        f"https://data.binance.vision/data/futures/um/monthly/fundingRate/"
        f"{symbol}/{symbol}-fundingRate-{month}.zip"
    )


def collect_funding_rate(
    symbol: str, start: date, end: date, out_dir: Path, max_retries: int, delay_s: float
) -> dict[str, Any]:
    log.info("Starting funding rate collection (data.binance.vision fundingRate archive): symbol=%s start=%s end=%s", symbol, start, end)
    dataset_dir = out_dir / "funding_rate"
    manifest = Manifest(dataset_dir)

    counters = {"ok": 0, "missing": 0, "failed": 0}
    covered_months: list[str] = []

    for month in _month_range(start, end):
        if _SHUTDOWN["requested"]:
            break
        parquet_path = dataset_dir / f"{symbol}-funding_rate-{month}.parquet"

        existing = manifest.get(month, "monthly")
        if existing and existing.get("status") == "ok" and parquet_path.exists() and _sha256_file(parquet_path) == existing.get("sha256"):
            log.info("SKIP funding_rate monthly %s (already ok)", month)
            counters["ok"] += 1
            covered_months.append(month)
            continue
        if existing and existing.get("status") == "missing":
            counters["missing"] += 1
            continue

        log.info("Starting download for funding_rate monthly %s", month)
        try:
            zip_bytes = _download_archive_zip(_funding_rate_monthly_url(symbol, month), max_retries, delay_s)
        except Exception as exc:
            log.error("Failed to fetch funding_rate monthly %s: %s", month, exc)
            manifest.record(month, "monthly", "failed")
            counters["failed"] += 1
            continue

        if zip_bytes is None:
            log.info("MISSING funding_rate monthly %s (404 -- not yet published or before real archive start)", month)
            manifest.record(month, "monthly", "missing")
            counters["missing"] += 1
            continue

        try:
            csv_bytes = _extract_single_csv(zip_bytes)
            df = pd.read_csv(io.BytesIO(csv_bytes))
            df.columns = _FUNDING_RATE_COLUMNS
            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(parquet_path, index=False)
            nbytes = parquet_path.stat().st_size
        except Exception as exc:
            log.error("Failed to write parquet for funding_rate monthly %s: %s", month, exc)
            manifest.record(month, "monthly", "failed")
            counters["failed"] += 1
            continue

        parquet_sha = _sha256_file(parquet_path)
        manifest.record(month, "monthly", "ok", sha256=parquet_sha, bytes_written=nbytes)
        log.info("Successfully wrote Parquet for funding_rate monthly %s (%d bytes)", month, nbytes)
        counters["ok"] += 1
        covered_months.append(month)

    real_start = min(covered_months) if covered_months else None
    real_end = max(covered_months) if covered_months else None
    log.info(
        "Funding rate collection done: ok=%d missing=%d failed=%d real_range=%s..%s",
        counters["ok"], counters["missing"], counters["failed"], real_start, real_end,
    )
    return {"counters": counters, "real_start": real_start, "real_end": real_end, "dataset_dir": dataset_dir}


_METRICS_COLUMNS = [
    "create_time", "symbol", "sum_open_interest", "sum_open_interest_value",
    "count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio",
    "count_long_short_ratio", "sum_taker_long_short_vol_ratio",
]


def _metrics_daily_url(symbol: str, day: str) -> str:
    return (
        f"https://data.binance.vision/data/futures/um/daily/metrics/"
        f"{symbol}/{symbol}-metrics-{day}.zip"
    )


def collect_open_interest(
    symbol: str, start: date, end: date, out_dir: Path, max_retries: int, delay_s: float
) -> dict[str, Any]:
    log.info("Starting open interest collection (data.binance.vision metrics archive): symbol=%s start=%s end=%s", symbol, start, end)
    dataset_dir = out_dir / "open_interest"
    manifest = Manifest(dataset_dir)

    counters = {"ok": 0, "missing": 0, "failed": 0}
    covered_days: list[date] = []

    cur = start
    while cur <= end:
        if _SHUTDOWN["requested"]:
            break
        day_str = cur.strftime("%Y-%m-%d")
        parquet_path = dataset_dir / f"{symbol}-open_interest-{day_str}.parquet"

        existing = manifest.get(day_str, "daily")
        if existing and existing.get("status") == "ok" and parquet_path.exists() and _sha256_file(parquet_path) == existing.get("sha256"):
            log.info("SKIP open_interest daily %s (already ok)", day_str)
            counters["ok"] += 1
            covered_days.append(cur)
            cur += timedelta(days=1)
            continue
        if existing and existing.get("status") == "missing":
            counters["missing"] += 1
            cur += timedelta(days=1)
            continue

        log.info("Starting download for open_interest daily %s", day_str)
        try:
            zip_bytes = _download_archive_zip(_metrics_daily_url(symbol, day_str), max_retries, delay_s)
        except Exception as exc:
            log.error("Failed to fetch open_interest daily %s: %s", day_str, exc)
            manifest.record(day_str, "daily", "failed")
            counters["failed"] += 1
            cur += timedelta(days=1)
            continue

        if zip_bytes is None:
            log.info("MISSING open_interest daily %s (404)", day_str)
            manifest.record(day_str, "daily", "missing")
            counters["missing"] += 1
            cur += timedelta(days=1)
            continue

        try:
            csv_bytes = _extract_single_csv(zip_bytes)
            df = pd.read_csv(io.BytesIO(csv_bytes))
            df.columns = _METRICS_COLUMNS
            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(parquet_path, index=False)
            nbytes = parquet_path.stat().st_size
        except Exception as exc:
            log.error("Failed to write parquet for open_interest daily %s: %s", day_str, exc)
            manifest.record(day_str, "daily", "failed")
            counters["failed"] += 1
            cur += timedelta(days=1)
            continue

        parquet_sha = _sha256_file(parquet_path)
        manifest.record(day_str, "daily", "ok", sha256=parquet_sha, bytes_written=nbytes)
        log.info("Successfully wrote Parquet for open_interest daily %s (%d bytes)", day_str, nbytes)
        counters["ok"] += 1
        covered_days.append(cur)
        cur += timedelta(days=1)

    real_start = min(covered_days) if covered_days else None
    real_end = max(covered_days) if covered_days else None
    log.info(
        "Open interest collection done: ok=%d missing=%d failed=%d real_range=%s..%s",
        counters["ok"], counters["missing"], counters["failed"], real_start, real_end,
    )
    return {"counters": counters, "real_start": real_start, "real_end": real_end, "dataset_dir": dataset_dir}


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _print_summary(results: dict[str, Any], out_dir: Path) -> None:
    print("\n" + "=" * 70)
    print("L1 DATA COLLECTION SUMMARY")
    print("=" * 70)

    total_bytes = 0
    for name, res in results.items():
        if res is None:
            continue
        size = _dir_size_bytes(res["dataset_dir"])
        total_bytes += size
        c = res["counters"]
        print(f"\n[{name}]")
        print(f"  ok={c['ok']} missing={c['missing']} failed={c['failed']}")
        print(f"  real covered range: {res['real_start']} .. {res['real_end']}")
        print(f"  bytes on disk: {size:,} ({size / 1e6:.2f} MB)")
        if size > 500_000_000:
            print(f"  *** FLAG: {name} exceeds 500MB ({size/1e6:.1f} MB) -- unexpected for L1 aggregate data, investigate ***")
            log.warning("FLAG: %s directory exceeds 500MB (%d bytes)", name, size)

    print(f"\nTOTAL bytes on disk (data/raw_l1): {total_bytes:,} ({total_bytes / 1e6:.2f} MB)")
    if total_bytes > 500_000_000:
        print("*** FLAG: total L1 data exceeds 500MB -- unexpected, investigate before trusting the pull ***")
        log.warning("FLAG: total L1 data exceeds 500MB (%d bytes)", total_bytes)
    print("=" * 70 + "\n")


def _last_complete_month(today: date) -> str:
    first_of_this_month = today.replace(day=1)
    last_complete = first_of_this_month - timedelta(days=1)
    return last_complete.strftime("%Y-%m")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    today = datetime.now(timezone.utc).date()
    yesterday = (today - timedelta(days=1)).isoformat()
    default_funding_end = _last_complete_month(today)

    p = argparse.ArgumentParser(description="Collect L1 aggregate data: klines, funding rate, open interest")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--klines-interval", default="1m", dest="klines_interval")
    p.add_argument("--klines-start", default="2019-12-31", dest="klines_start", metavar="YYYY-MM-DD")
    p.add_argument("--klines-end", default=yesterday, dest="klines_end", metavar="YYYY-MM-DD")
    p.add_argument("--funding-start", default="2020-01", dest="funding_start", metavar="YYYY-MM")
    p.add_argument("--funding-end", default=default_funding_end, dest="funding_end", metavar="YYYY-MM")
    p.add_argument("--oi-start", default="2020-09-01", dest="oi_start", metavar="YYYY-MM-DD")
    p.add_argument("--oi-end", default=yesterday, dest="oi_end", metavar="YYYY-MM-DD")
    p.add_argument("--out-dir", default="data/raw_l1", dest="out_dir")
    p.add_argument("--max-retries", type=int, default=5, dest="max_retries")
    p.add_argument("--request-delay-s", type=float, default=0.3, dest="request_delay_s")
    p.add_argument("--datasets", default="klines,funding_rate,open_interest")
    p.add_argument("--log-file", default=str(_LOG_FILE), dest="log_file")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _setup_logging(Path(args.log_file))

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = {d.strip() for d in args.datasets.split(",") if d.strip()}

    log.info("=== L1 collection run starting: symbol=%s datasets=%s ===", args.symbol, sorted(datasets))

    results: dict[str, Any] = {"klines": None, "funding_rate": None, "open_interest": None}

    if "klines" in datasets and not _SHUTDOWN["requested"]:
        start = datetime.strptime(args.klines_start, "%Y-%m-%d").date()
        end = datetime.strptime(args.klines_end, "%Y-%m-%d").date()
        try:
            results["klines"] = collect_klines(
                args.symbol, args.klines_interval, start, end, out_dir, args.max_retries, args.request_delay_s
            )
        except Exception:
            log.exception("Klines collection crashed")

    if "funding_rate" in datasets and not _SHUTDOWN["requested"]:
        start = datetime.strptime(args.funding_start + "-01", "%Y-%m-%d").date()
        end = datetime.strptime(args.funding_end + "-01", "%Y-%m-%d").date()
        try:
            results["funding_rate"] = collect_funding_rate(
                args.symbol, start, end, out_dir, args.max_retries, args.request_delay_s
            )
        except Exception:
            log.exception("Funding rate collection crashed")

    if "open_interest" in datasets and not _SHUTDOWN["requested"]:
        start = datetime.strptime(args.oi_start, "%Y-%m-%d").date()
        end = datetime.strptime(args.oi_end, "%Y-%m-%d").date()
        try:
            results["open_interest"] = collect_open_interest(
                args.symbol, start, end, out_dir, args.max_retries, args.request_delay_s
            )
        except Exception:
            log.exception("Open interest collection crashed")

    _print_summary(results, out_dir)
    log.info("=== L1 collection run finished ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
