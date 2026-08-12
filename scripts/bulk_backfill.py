"""Overnight bulk historical backfill for Binance futures data."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import signal
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any

import requests

_LOG_FILE = Path("logs/backfill.log")


def _setup_logging() -> logging.Logger:
    _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)s %(message)s"
    logger = logging.getLogger("backfill")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        fh = logging.FileHandler(_LOG_FILE, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(fmt))
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.INFO)
        sh.setFormatter(logging.Formatter(fmt))
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger


log = _setup_logging()

_MANIFEST_NAME = "_manifest.jsonl"


class Manifest:
    """Thread-safe JSONL manifest tracking (date, dataset) status."""

    def __init__(self, out_dir: Path) -> None:
        self._path = out_dir / _MANIFEST_NAME
        self._lock = Lock()
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
                    key = (entry["date"], entry["dataset"])
                    self._entries[key] = entry
                except (json.JSONDecodeError, KeyError):
                    pass

    def _append(self, entry: dict[str, Any]) -> None:
        # buffering=1 = line-buffered so tail -f works live
        with open(self._path, "a", encoding="utf-8", buffering=1) as fh:
            fh.write(json.dumps(entry) + "\n")

    def get(self, date_str: str, dataset: str) -> dict[str, Any] | None:
        with self._lock:
            return self._entries.get((date_str, dataset))

    def record(
        self,
        date_str: str,
        dataset: str,
        status: str,
        sha256: str = "",
        bytes_downloaded: int = 0,
    ) -> None:
        entry = {
            "date": date_str,
            "dataset": dataset,
            "status": status,
            "sha256": sha256,
            "bytes": bytes_downloaded,
            "checked_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
        with self._lock:
            self._entries[(date_str, dataset)] = entry
            self._append(entry)

    def all_entries(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._entries.values())


_DATASET_PATH = {
    "trades": "trades",
    "aggTrades": "aggTrades",
    "bookDepth": "bookDepth",
}


def _zip_url(symbol: str, dataset: str, day: str) -> str:
    dp = _DATASET_PATH[dataset]
    return (
        f"https://data.binance.vision/data/futures/um/daily/{dp}/{symbol}/"
        f"{symbol}-{dataset}-{day}.zip"
    )

def _checksum_url(symbol: str, dataset: str, day: str) -> str:
    dp = _DATASET_PATH[dataset]
    return (
        f"https://data.binance.vision/data/futures/um/daily/{dp}/{symbol}/"
        f"{symbol}-{dataset}-{day}.zip.CHECKSUM"
    )

def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_checksum(checksum_text: str) -> str:
    """Extract hex digest from Binance CHECKSUM file."""
    for token in checksum_text.split():
        if len(token) == 64 and all(c in "0123456789abcdef" for c in token):
            return token
    return checksum_text.strip().splitlines()[0].strip().split()[-1]


def _get_with_retry(url: str, max_retries: int, delay_s: float) -> requests.Response:
    """GET url, retrying on transient errors. Returns immediately on 404."""
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        if attempt > 0:
            sleep_time = 1.0 * (2 ** (attempt - 1))
            log.debug("Retry %d/%d in %.1fs for %s", attempt, max_retries, sleep_time, url)
            time.sleep(sleep_time)
        time.sleep(delay_s)
        try:
            resp = requests.get(url, timeout=90)
            if resp.status_code == 404:
                return resp
            if resp.status_code >= 500:
                log.warning("HTTP %d for %s", resp.status_code, url)
                last_exc = requests.HTTPError(f"HTTP {resp.status_code}", response=resp)
                continue
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            log.warning("Request error for %s: %s", url, exc)
            last_exc = exc
    raise last_exc or RuntimeError(f"Failed to GET {url}")


def _download_one(
    symbol: str,
    dataset: str,
    date_str: str,
    out_dir: Path,
    manifest: Manifest,
    max_retries: int,
    delay_s: float,
) -> str:
    """Download one (date, dataset). Returns 'ok' | 'missing' | 'failed'."""
    t0 = time.monotonic()

    entry = manifest.get(date_str, dataset)
    if entry and entry.get("status") == "ok":
        parquet = out_dir / dataset / f"{symbol}-{dataset}-{date_str}.parquet"
        if parquet.exists() and _sha256_file(parquet) == entry.get("sha256", ""):
            log.info("SKIP   %s %s (already ok)", date_str, dataset)
            return "ok"

    if entry and entry.get("status") == "missing":
        log.info("SKIP   %s %s (previously missing)", date_str, dataset)
        return "missing"

    csum_url = _checksum_url(symbol, dataset, date_str)
    try:
        csum_resp = _get_with_retry(csum_url, max_retries, delay_s)
    except Exception as exc:
        log.error("FAILED %s %s checksum: %s", date_str, dataset, exc)
        manifest.record(date_str, dataset, "failed")
        return "failed"

    if csum_resp.status_code == 404:
        log.info("MISSING %s %s (404 on CHECKSUM)", date_str, dataset)
        manifest.record(date_str, dataset, "missing")
        return "missing"

    expected_sha = _parse_checksum(csum_resp.text)

    zip_url = _zip_url(symbol, dataset, date_str)
    try:
        zip_resp = _get_with_retry(zip_url, max_retries, delay_s)
    except Exception as exc:
        log.error("FAILED %s %s zip: %s", date_str, dataset, exc)
        manifest.record(date_str, dataset, "failed")
        return "failed"

    if zip_resp.status_code == 404:
        log.info("MISSING %s %s (404 on zip)", date_str, dataset)
        manifest.record(date_str, dataset, "missing")
        return "missing"

    zip_bytes = zip_resp.content
    if _sha256_bytes(zip_bytes) != expected_sha:
        log.error("FAILED %s %s checksum mismatch", date_str, dataset)
        manifest.record(date_str, dataset, "failed")
        return "failed"

    try:
        import pandas as pd

        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
            csv_bytes = zf.read(csv_name)

        df = pd.read_csv(io.StringIO(csv_bytes.decode("utf-8")))
        parquet_path = out_dir / f"{symbol}-{dataset}-{date_str}.parquet"
        df.to_parquet(parquet_path, index=False)
        parquet_sha = _sha256_file(parquet_path)
    except Exception as exc:
        log.error("FAILED %s %s extract/parquet: %s", date_str, dataset, exc)
        manifest.record(date_str, dataset, "failed")
        return "failed"

    elapsed = time.monotonic() - t0
    log.info("OK     %s %s  %d bytes  %.1fs", date_str, dataset, len(zip_bytes), elapsed)
    manifest.record(date_str, dataset, "ok", sha256=parquet_sha, bytes_downloaded=len(zip_bytes))
    return "ok"


def _print_summary(manifest: Manifest) -> None:
    entries = manifest.all_entries()
    ok = [e for e in entries if e["status"] == "ok"]
    missing = [e for e in entries if e["status"] == "missing"]
    failed = [e for e in entries if e["status"] == "failed"]
    total_bytes = sum(e.get("bytes", 0) for e in ok)

    print("\n" + "=" * 60)
    print("BACKFILL SUMMARY")
    print("=" * 60)
    print(f"  OK:      {len(ok):>6}")
    print(f"  Missing: {len(missing):>6}  (legitimate gaps on Binance Vision)")
    print(f"  Failed:  {len(failed):>6}  (need rerun)")
    print(f"  Bytes:   {total_bytes:>12,}")
    if failed:
        print("\nFailed (date, dataset) pairs to rerun:")
        for e in sorted(failed, key=lambda x: (x["date"], x["dataset"])):
            print(f"  {e['date']}  {e['dataset']}")
    print("=" * 60 + "\n")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_start = date(2021, 8, 10).isoformat()
    default_end = date(2026, 8, 10).isoformat()

    p = argparse.ArgumentParser(
        description="Overnight bulk historical backfill from data.binance.vision"
    )
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--datasets", default="trades,aggTrades,bookDepth",
                   help="Comma-separated list of dataset names")
    p.add_argument("--start", default=default_start, metavar="YYYY-MM-DD")
    p.add_argument("--end", default=default_end, metavar="YYYY-MM-DD")
    p.add_argument("--out-dir", default="data/raw", dest="out_dir")
    p.add_argument("--max-retries", type=int, default=5, dest="max_retries")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--request-delay-s", type=float, default=0.2, dest="request_delay_s",
                   help="Politeness delay (seconds) between request starts per worker")
    return p.parse_args(argv)



def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()

    work: list[tuple[str, str]] = []
    current = start
    while current <= end:
        day = current.strftime("%Y-%m-%d")
        for ds in datasets:
            work.append((day, ds))
        current += timedelta(days=1)

    manifest = Manifest(out_dir)
    log.info(
        "Starting backfill: symbol=%s datasets=%s start=%s end=%s tasks=%d concurrency=%d",
        args.symbol, datasets, args.start, args.end, len(work), args.concurrency,
    )

    counters: dict[str, int] = {"ok": 0, "missing": 0, "failed": 0}
    _shutdown = {"requested": False}

    def _handle_signal(sig, frame):  # noqa: ANN001
        log.warning("Signal %s received — finishing in-flight tasks then exiting…", sig)
        _shutdown["requested"] = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(
                _download_one,
                args.symbol, ds, day, out_dir, manifest, args.max_retries, args.request_delay_s,
            ): (day, ds)
            for day, ds in work
        }
        for fut in as_completed(futures):
            if _shutdown["requested"]:
                pool.shutdown(wait=False, cancel_futures=True)
                break
            day, ds = futures[fut]
            try:
                status = fut.result()
            except Exception as exc:
                log.error("Unhandled exception for %s %s: %s", day, ds, exc)
                status = "failed"
            counters[status] = counters.get(status, 0) + 1

    _print_summary(manifest)
    log.info("Done. ok=%d missing=%d failed=%d", counters["ok"], counters["missing"], counters["failed"])
    return 0 if counters["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
