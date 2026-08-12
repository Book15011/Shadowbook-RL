"""Tests for scripts/bulk_backfill.py."""
from __future__ import annotations

import hashlib
import io
import sys
import zipfile
from pathlib import Path

import pytest
import responses as resp_lib

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from bulk_backfill import Manifest, _download_one, _sha256_file

SYMBOL = "BTCUSDT"
DATASET = "trades"
DAY = "2024-01-15"
_CSV = "timestamp,price,qty\n1705276800000,42000.0,0.1\n"
_ZIP_URL = (
    f"https://data.binance.vision/data/futures/um/daily/trades/"
    f"{SYMBOL}/{DAY}/{SYMBOL}-{DATASET}-{DAY}.zip"
)
_CSUM_URL = (
    f"https://data.binance.vision/data/futures/um/daily/trades/"
    f"{SYMBOL}/{DAY}/.CHECKSUM"
)


def _make_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(f"{SYMBOL}-{DATASET}-{DAY}.csv", _CSV)
    return buf.getvalue()


def _zip_sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@resp_lib.activate
def test_manifest_skips_already_ok_entry(tmp_path: Path) -> None:
    manifest = Manifest(tmp_path)
    parquet_path = tmp_path / f"{SYMBOL}-{DATASET}-{DAY}.parquet"
    parquet_path.write_bytes(b"fake-parquet-content")
    sha = _sha256_file(parquet_path)
    manifest.record(DAY, DATASET, "ok", sha256=sha, bytes_downloaded=1234)

    status = _download_one(SYMBOL, DATASET, DAY, tmp_path, manifest, max_retries=3, delay_s=0.0)

    assert status == "ok"
    assert len(resp_lib.calls) == 0


@resp_lib.activate
def test_retries_on_transient_5xx_then_succeeds(tmp_path: Path) -> None:
    manifest = Manifest(tmp_path)
    zip_bytes = _make_zip()
    sha = _zip_sha(zip_bytes)
    checksum_body = f"sha256  {SYMBOL}-{DATASET}-{DAY}.csv  {sha}\n"

    resp_lib.add(resp_lib.GET, _CSUM_URL, status=503, body="server error")
    resp_lib.add(resp_lib.GET, _CSUM_URL, body=checksum_body, status=200, content_type="text/plain")
    resp_lib.add(resp_lib.GET, _ZIP_URL, body=zip_bytes, status=200, content_type="application/zip")

    status = _download_one(SYMBOL, DATASET, DAY, tmp_path, manifest, max_retries=3, delay_s=0.0)

    assert status == "ok"
    entry = manifest.get(DAY, DATASET)
    assert entry is not None and entry["status"] == "ok"
    assert entry["bytes"] == len(zip_bytes)


@resp_lib.activate
def test_404_on_checksum_marks_missing_and_not_retried(tmp_path: Path) -> None:
    manifest = Manifest(tmp_path)
    resp_lib.add(resp_lib.GET, _CSUM_URL, status=404)

    status = _download_one(SYMBOL, DATASET, DAY, tmp_path, manifest, max_retries=5, delay_s=0.0)

    assert status == "missing"
    entry = manifest.get(DAY, DATASET)
    assert entry is not None and entry["status"] == "missing"
    zip_calls = [c for c in resp_lib.calls if _ZIP_URL in c.request.url]
    assert len(zip_calls) == 0

    # re-run must skip without any HTTP calls
    resp_lib.reset()
    status2 = _download_one(SYMBOL, DATASET, DAY, tmp_path, manifest, max_retries=5, delay_s=0.0)
    assert status2 == "missing"
    assert len(resp_lib.calls) == 0


def test_manifest_persists_and_reloads(tmp_path: Path) -> None:
    m1 = Manifest(tmp_path)
    m1.record("2024-01-01", "trades", "ok", sha256="abc123", bytes_downloaded=500)
    m1.record("2024-01-02", "trades", "missing")

    m2 = Manifest(tmp_path)
    e1 = m2.get("2024-01-01", "trades")
    e2 = m2.get("2024-01-02", "trades")

    assert e1 is not None and e1["status"] == "ok" and e1["sha256"] == "abc123"
    assert e2 is not None and e2["status"] == "missing"
