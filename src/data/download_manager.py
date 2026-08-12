from __future__ import annotations

import hashlib
import io
import os
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests


def _build_url(symbol: str, dataset: str, day: str) -> str:
    dataset_path = {"trades": "trades", "aggTrades": "aggTrades", "bookDepth": "depth"}[dataset]
    return f"https://data.binance.vision/data/futures/um/daily/{dataset_path}/{symbol}/{day}/{symbol}-{dataset}-{day}.zip"


def _build_checksum_url(symbol: str, dataset: str, day: str) -> str:
    return _build_url(symbol, dataset, day).replace(f"/{symbol}-{dataset}-{day}.zip", "/.CHECKSUM")


def _checksum_matches(path: Path, expected: str) -> bool:
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return expected.strip().split()[1] == sha


def _download_bytes(url: str) -> bytes:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.content


def download_and_verify(dataset: str, symbol: str, date_str: str, out_dir: str | os.PathLike[str]) -> pd.DataFrame:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    url = _build_url(symbol, dataset, date_str)
    checksum_url = _build_checksum_url(symbol, dataset, date_str)
    archive_bytes = _download_bytes(url)
    checksum_text = _download_bytes(checksum_url).decode("utf-8")

    archive_path = out_path / f"{symbol}-{dataset}-{date_str}.zip"
    archive_path.write_bytes(archive_bytes)

    if not _checksum_matches(archive_path, checksum_text):
        raise ValueError("Checksum mismatch")

    with zipfile.ZipFile(archive_path) as zf:
        csv_name = next(name for name in zf.namelist() if name.endswith(".csv"))
        with zf.open(csv_name) as fh:
            data = fh.read().decode("utf-8")

    df = pd.read_csv(io.StringIO(data))
    df.to_parquet(out_path / f"{symbol}-{dataset}-{date_str}.parquet", index=False)
    return df


def bulk_download(symbol: str, dataset: str, start_date: str, end_date: str, out_dir: str | os.PathLike[str]) -> pd.DataFrame:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    current = start
    while current <= end:
        day = current.strftime("%Y-%m-%d")
        try:
            frames.append(download_and_verify(dataset, symbol, day, out_path))
        except Exception:
            pass
        current += timedelta(days=1)

    if not frames:
        return pd.DataFrame(columns=["timestamp", "price", "qty"])
    return pd.concat(frames, ignore_index=True)
