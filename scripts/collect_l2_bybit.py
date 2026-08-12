"""Byte-budget-targeted Bybit L2 order book backfill.

Downloads Bybit's public daily order-book archives
(https://quote-saver.bycsi.com/orderbook/linear/{symbol}/{date}_{symbol}_ob500.data.zip),
reconstructs book state from the snapshot+delta+seq stream using the same
L2Book/apply_diff logic src/data/l2_capture_daemon.py uses for live Binance
capture, truncates to the top N bid/ask levels, writes one compressed Parquet
file per day, and deletes the raw zip immediately after processing.

Walks backward day-by-day from the most recent day with confirmed data (found
by probing quote-saver.bycsi.com — no hardcoded date, since archive currency
drifts over time) until either the cumulative Parquet byte budget or the
calendar-day cap is reached, whichever comes first. The byte budget is the
real target; the day cap is only a sanity backstop.

Note: Bybit does not publish a checksum for these archives (unlike Binance
Vision's .CHECKSUM files), so resumability is verified against our own output
Parquet's sha256 rather than an upstream-published hash.

Usage:
    python scripts/collect_l2_bybit.py [--symbol BTCUSDT] [--out-dir data/raw_l2_bybit]
        [--top-levels 20] [--byte-budget-gb 35] [--max-days 2000]
        [--start-day 2025-08-20] [--max-retries 5] [--request-delay-s 0.5]
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
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.l2_capture_daemon import L2Book, apply_diff  # noqa: E402

# ---------------------------------------------------------------------------
# Logging setup — milestone-level only. Never log per-tick/per-line data.
# ---------------------------------------------------------------------------

_LOG_FILE = Path("logs/l2_bybit_collection.log")


def _setup_logging(log_file: Path) -> logging.Logger:
    """Configure the module logger's handlers to point at log_file. Safe to
    call multiple times (e.g. once main() knows the real --log-file) — it
    replaces any previously attached handlers rather than skipping setup."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)s %(message)s"
    logger = logging.getLogger("l2_bybit_collect")
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


log = logging.getLogger("l2_bybit_collect")

_SHUTDOWN = {"requested": False}


def _handle_signal(sig, frame):  # noqa: ANN001
    log.warning("Signal %s received — finishing current day then exiting…", sig)
    _SHUTDOWN["requested"] = True


# ---------------------------------------------------------------------------
# Manifest (mirrors bulk_backfill.py's resumable JSONL pattern; keyed by date
# only since there is one dataset here. sha256 is of OUR OWN output parquet,
# not an upstream checksum — Bybit doesn't publish one for these archives).
# ---------------------------------------------------------------------------


class Manifest:
    _NAME = "_manifest.jsonl"

    def __init__(self, out_dir: Path) -> None:
        self._path = out_dir / self._NAME
        self._entries: dict[str, dict[str, Any]] = {}
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
                    self._entries[entry["date"]] = entry
                except (json.JSONDecodeError, KeyError):
                    pass

    def get(self, date_str: str) -> dict[str, Any] | None:
        return self._entries.get(date_str)

    def record(self, date_str: str, status: str, sha256: str = "", bytes_written: int = 0, rows: int = 0) -> None:
        entry = {
            "date": date_str,
            "status": status,
            "sha256": sha256,
            "bytes": bytes_written,
            "rows": rows,
            "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        self._entries[date_str] = entry
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a", encoding="utf-8", buffering=1) as fh:
            fh.write(json.dumps(entry) + "\n")

    def all_entries(self) -> list[dict[str, Any]]:
        return list(self._entries.values())


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Bybit archive access
# ---------------------------------------------------------------------------


def _archive_url(symbol: str, day: str) -> str:
    return f"https://quote-saver.bycsi.com/orderbook/linear/{symbol}/{day}_{symbol}_ob500.data.zip"


def _head_available(url: str, timeout: int = 20) -> bool:
    try:
        resp = requests.head(url, timeout=timeout)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def _discover_most_recent_day(symbol: str) -> date:
    """Probe backward from today with exponential step growth to find the
    boundary fast, then binary search to pin the exact last-available day.
    No fixed lag is assumed — archive currency can drift over time."""
    today = datetime.now(timezone.utc).date()

    if _head_available(_archive_url(symbol, today.isoformat())):
        return today

    step = 1
    lo = None
    probe = today
    while step <= 4096:
        probe = today - timedelta(days=step)
        if _head_available(_archive_url(symbol, probe.isoformat())):
            lo = probe
            break
        step *= 2
    if lo is None:
        raise RuntimeError(
            f"Could not find any available Bybit L2 archive day within {step} days back from {today}"
        )

    # Binary search between lo (available) and hi (today, unavailable) for the boundary.
    hi_bound = today
    lo_bound = lo
    while (hi_bound - lo_bound).days > 1:
        mid = lo_bound + (hi_bound - lo_bound) // 2
        if _head_available(_archive_url(symbol, mid.isoformat())):
            lo_bound = mid
        else:
            hi_bound = mid
    log.info("Discovered most recent available Bybit L2 day: %s", lo_bound.isoformat())
    return lo_bound


def _get_with_retry(url: str, max_retries: int, delay_s: float, timeout: int = 120) -> requests.Response | None:
    """Streaming-safe GET with retry. Returns the response (stream=True) on
    success, or None if the resource is genuinely missing (404). Raises after
    exhausting retries on transient errors."""
    backoff = 2.0
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        if attempt > 0:
            sleep_time = backoff * (2 ** (attempt - 1))
            log.warning("Retry %d/%d in %.1fs for %s", attempt, max_retries, sleep_time, url)
            time.sleep(sleep_time)
        time.sleep(delay_s)
        try:
            resp = requests.get(url, timeout=timeout, stream=True)
            if resp.status_code == 404:
                return None
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


def _download_to_file(resp: requests.Response, dest: Path) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    nbytes = 0
    with open(dest, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            if chunk:
                fh.write(chunk)
                nbytes += len(chunk)
    return nbytes


# ---------------------------------------------------------------------------
# Book reconstruction + truncation (reuses L2Book/apply_diff as-is for the
# actual level-by-level bid/ask updates — do not reimplement that part).
#
# NOTE: this deployment's apply_diff() has no built-in sequence-gap check —
# it unconditionally applies whatever bids/asks it's given (confirmed by
# reading the real src/data/l2_capture_daemon.py on this machine; an earlier
# draft of this script assumed a gap-check existed inside apply_diff itself,
# which was wrong for this codebase). Gap detection for Bybit's per-message
# incrementing "u" is therefore done here, one layer up, before calling
# apply_diff: if u jumps by more than 1, the book is reset via
# apply_snapshot({"bids": [], "asks": []}, ...) — the same reset apply_diff
# would have done internally on the daemon's live Binance path — and lets
# subsequent deltas rebuild it. In practice Bybit's archived "u" increments
# by exactly 1 for every message with no gaps (confirmed against a real
# sample day), so this branch is not expected to fire; it exists as a safety
# net against corrupted/truncated archive files.
# ---------------------------------------------------------------------------


def _top_levels(side_dict: dict[float, float], top_n: int, descending: bool) -> list[list[float]]:
    levels = sorted(side_dict.items(), key=lambda kv: kv[0], reverse=descending)[:top_n]
    return [[price, qty] for price, qty in levels]


def _process_day_zip(zip_path: Path, symbol: str, top_levels: int, day_str: str) -> pd.DataFrame:
    book = L2Book()
    rows: list[dict[str, Any]] = []
    gap_count = 0

    with zipfile.ZipFile(zip_path) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".data"))
        with zf.open(name) as raw_fh:
            text_stream = io.TextIOWrapper(raw_fh, encoding="utf-8")
            for line in text_stream:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue  # tolerate an occasional malformed line, don't abort the day

                data = obj.get("data", {})
                u = data.get("u")
                bids = data.get("b", [])
                asks = data.get("a", [])
                if u is None:
                    continue

                if obj.get("type") == "snapshot":
                    book.apply_snapshot({"bids": bids, "asks": asks}, seq=u)
                else:
                    if book.last_u is not None and u <= book.last_u:
                        continue  # stale/duplicate message, ignore
                    if book.last_u is not None and u > book.last_u + 1:
                        gap_count += 1
                        book.apply_snapshot({"bids": [], "asks": []}, seq=u)
                    apply_diff(book, {"u": u, "bids": bids, "asks": asks})

                best_bid = book.best_bid()
                best_ask = book.best_ask()
                if best_bid is None or best_ask is None:
                    continue

                rows.append(
                    {
                        "symbol": symbol,
                        "ts": int(obj.get("ts", 0)),
                        "update_id": int(u),
                        "seq": int(data.get("seq", 0)),
                        "best_bid": float(best_bid),
                        "best_ask": float(best_ask),
                        "mid_price": float((best_bid + best_ask) / 2.0),
                        "spread": float(best_ask - best_bid),
                        "bids": json.dumps(_top_levels(book.bids, top_levels, descending=True)),
                        "asks": json.dumps(_top_levels(book.asks, top_levels, descending=False)),
                    }
                )

    if gap_count:
        log.warning("Detected %d sequence gap(s) while reconstructing %s (book was reset and rebuilt each time)", gap_count, day_str)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-day pipeline
# ---------------------------------------------------------------------------


def _process_one_day(
    symbol: str,
    day: date,
    out_dir: Path,
    manifest: Manifest,
    top_levels: int,
    max_retries: int,
    delay_s: float,
) -> tuple[str, int]:
    """Returns (status, parquet_bytes). status in {'ok','missing','failed','skipped'}."""
    day_str = day.isoformat()
    symbol_dir = out_dir / symbol
    parquet_path = symbol_dir / f"l2-{symbol}-{day_str}.parquet"

    existing = manifest.get(day_str)
    if existing and existing.get("status") == "ok" and parquet_path.exists() and _sha256_file(parquet_path) == existing.get("sha256"):
        log.info("SKIP %s (already ok, %d bytes)", day_str, existing.get("bytes", 0))
        return "skipped", int(existing.get("bytes", 0))
    if existing and existing.get("status") == "missing":
        log.info("SKIP %s (previously confirmed missing)", day_str)
        return "missing", 0

    log.info("Starting download for %s", day_str)
    url = _archive_url(symbol, day_str)
    zip_path = symbol_dir / f".tmp-{day_str}.zip"

    try:
        resp = _get_with_retry(url, max_retries, delay_s)
    except Exception as exc:
        log.error("Failed to fetch URL for %s: %s", day_str, exc)
        manifest.record(day_str, "failed")
        return "failed", 0

    if resp is None:
        log.info("MISSING %s (404 — no archive published for this day)", day_str)
        manifest.record(day_str, "missing")
        return "missing", 0

    try:
        zip_bytes = _download_to_file(resp, zip_path)
        log.info("Downloaded %s: %d bytes, reconstructing book state", day_str, zip_bytes)
    except Exception as exc:
        log.error("Download failed for %s: %s", day_str, exc)
        zip_path.unlink(missing_ok=True)
        manifest.record(day_str, "failed")
        return "failed", 0

    try:
        df = _process_day_zip(zip_path, symbol, top_levels, day_str)
    except Exception as exc:
        log.error("Failed to parse/reconstruct book for %s: %s", day_str, exc)
        zip_path.unlink(missing_ok=True)
        manifest.record(day_str, "failed")
        return "failed", 0

    zip_path.unlink(missing_ok=True)  # raw archive deleted immediately, per requirement

    if df.empty:
        log.error("No usable rows reconstructed for %s (corrupted or empty archive)", day_str)
        manifest.record(day_str, "failed")
        return "failed", 0

    try:
        symbol_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_path, index=False, compression="snappy")
        nbytes = parquet_path.stat().st_size
        parquet_sha = _sha256_file(parquet_path)
    except Exception as exc:
        log.error("Failed to write Parquet for %s: %s", day_str, exc)
        manifest.record(day_str, "failed")
        return "failed", 0

    manifest.record(day_str, "ok", sha256=parquet_sha, bytes_written=nbytes, rows=len(df))
    log.info("Successfully wrote Parquet for %s: %d bytes, %d rows", day_str, nbytes, len(df))
    return "ok", nbytes


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _print_summary(
    covered_days: list[date],
    missing_days: list[date],
    failed_days: list[date],
    total_bytes: int,
    ok_days: int,
    stop_reason: str,
) -> None:
    print("\n" + "=" * 70)
    print("L2 BYBIT COLLECTION SUMMARY")
    print("=" * 70)
    if covered_days:
        print(f"  Real day range covered: {min(covered_days)} .. {max(covered_days)}")
    else:
        print("  Real day range covered: none")
    print(f"  Days OK:      {ok_days}")
    print(f"  Days missing: {len(missing_days)}")
    print(f"  Days failed:  {len(failed_days)}")
    avg = (total_bytes / ok_days) if ok_days else 0
    print(f"  Total Parquet bytes: {total_bytes:,} ({total_bytes / 1e9:.3f} GB)")
    print(f"  Average bytes/day (real, measured): {avg:,.0f} ({avg / 1e6:.1f} MB)")
    print(f"  Stop reason: {stop_reason}")
    if missing_days:
        print(f"\n  Missing days: {[d.isoformat() for d in missing_days]}")
    if failed_days:
        print(f"\n  Failed days (rerun the script to retry — manifest will skip completed days):")
        print(f"    {[d.isoformat() for d in failed_days]}")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Byte-budget-targeted Bybit L2 order book backfill")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--out-dir", default="data/raw_l2_bybit", dest="out_dir")
    p.add_argument("--top-levels", type=int, default=20, dest="top_levels")
    p.add_argument("--byte-budget-gb", type=float, default=35.0, dest="byte_budget_gb")
    p.add_argument("--max-days", type=int, default=2000, dest="max_days")
    p.add_argument("--max-consecutive-missing", type=int, default=14, dest="max_consecutive_missing",
                    help="Safety backstop: stop early if this many consecutive days are missing/failed in a row")
    p.add_argument("--start-day", default=None, dest="start_day", metavar="YYYY-MM-DD",
                    help="Most recent day to start from; auto-discovered via probing if omitted")
    p.add_argument("--max-retries", type=int, default=5, dest="max_retries")
    p.add_argument("--request-delay-s", type=float, default=0.5, dest="request_delay_s")
    p.add_argument("--log-file", default=str(_LOG_FILE), dest="log_file")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _setup_logging(Path(args.log_file))

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    byte_budget = int(args.byte_budget_gb * (1024 ** 3))

    if args.start_day:
        start_day = datetime.strptime(args.start_day, "%Y-%m-%d").date()
        log.info("Using explicit start day: %s", start_day.isoformat())
    else:
        log.info("Discovering most recent available Bybit L2 archive day…")
        start_day = _discover_most_recent_day(args.symbol)

    manifest = Manifest(out_dir)

    log.info(
        "=== L2 Bybit collection starting: symbol=%s start_day=%s byte_budget=%.1fGB max_days=%d ===",
        args.symbol, start_day, args.byte_budget_gb, args.max_days,
    )

    covered_days: list[date] = []
    missing_days: list[date] = []
    failed_days: list[date] = []
    total_bytes = 0
    ok_days = 0
    consecutive_missing = 0
    stop_reason = "exhausted loop unexpectedly"

    day = start_day
    days_attempted = 0
    while True:
        if _SHUTDOWN["requested"]:
            stop_reason = "interrupted by signal"
            break
        if days_attempted >= args.max_days:
            stop_reason = f"day cap reached ({args.max_days} days)"
            break
        if total_bytes >= byte_budget:
            stop_reason = f"byte budget reached ({args.byte_budget_gb} GB)"
            break
        if consecutive_missing >= args.max_consecutive_missing:
            stop_reason = f"{consecutive_missing} consecutive missing/failed days — likely reached start of archive"
            break

        status, nbytes = _process_one_day(
            args.symbol, day, out_dir, manifest, args.top_levels, args.max_retries, args.request_delay_s
        )
        days_attempted += 1

        if status in ("ok", "skipped"):
            covered_days.append(day)
            total_bytes += nbytes
            ok_days += 1
            consecutive_missing = 0
        elif status == "missing":
            missing_days.append(day)
            consecutive_missing += 1
        elif status == "failed":
            failed_days.append(day)
            consecutive_missing += 1

        day -= timedelta(days=1)

    log.info("=== L2 Bybit collection stopping: %s ===", stop_reason)
    _print_summary(covered_days, missing_days, failed_days, total_bytes, ok_days, stop_reason)
    return 0


if __name__ == "__main__":
    sys.exit(main())
