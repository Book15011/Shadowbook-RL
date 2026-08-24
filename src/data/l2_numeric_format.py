"""Shared read/write logic for the numeric (converted) L2 day-file format --
single source of truth for both the conversion script and the env's own
loading path, so write and read can never silently drift apart.

Format (2026-08-24, this round's storage-format investigation): every real
L2 day file has EXACTLY 20 bid levels and 20 ask levels per row (verified
exhaustively across 31 real days, not sampled -- see
scripts/check_level_counts.py). This module stores that fixed shape as flat
float64/int64 binary arrays, zstd-compressed (level 9 -- chosen for a strong
speed/size tradeoff measured directly: ~46.8MB/day, 0.45x the original
parquet's size, vs. ~587.5MB/day uncompressed and its 5.61x blowup that would
not fit this box's available disk for the full 441-day train+val+test set).

Values are parsed via the exact same path the original JSON pipeline uses
(json.loads -> float64), so the stored bit patterns match what the original
per-reset parsing produces -- this, not the format choice itself, is what
makes byte-identical seed-equivalence achievable. float32 was tested and
rejected: 87% of real price/size values do not round-trip exactly through
float32 (measured on 1.6M real values, not assumed) -- every array here stays
float64.

File layout: [4-byte big-endian header length][JSON header, UTF-8][zstd-
compressed raw concatenated array bytes, in the header's own array order].
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import zstandard as zstd

ZSTD_LEVEL = 9
ARRAY_ORDER = [
    "ts", "best_bid", "best_ask", "mid_price", "spread",
    "bid_prices", "bid_sizes", "ask_prices", "ask_sizes",
]


def write_day(arrays: dict[str, np.ndarray], out_path: Path | str) -> None:
    """arrays must contain exactly ARRAY_ORDER's keys, each already the dtype/
    shape the env expects (ts: (n,) int64; best_bid/best_ask/mid_price/spread:
    (n,) float64; bid_prices/bid_sizes/ask_prices/ask_sizes: (n, 20) float64)."""
    missing = set(ARRAY_ORDER) - set(arrays)
    if missing:
        raise ValueError(f"missing required arrays: {missing}")
    header = {name: {"dtype": str(arrays[name].dtype), "shape": list(arrays[name].shape)} for name in ARRAY_ORDER}
    header_bytes = json.dumps(header).encode("utf-8")
    blob = b"".join(np.ascontiguousarray(arrays[name]).tobytes() for name in ARRAY_ORDER)
    cctx = zstd.ZstdCompressor(level=ZSTD_LEVEL)
    compressed = cctx.compress(blob)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        f.write(struct.pack(">I", len(header_bytes)))
        f.write(header_bytes)
        f.write(compressed)
    tmp_path.replace(out_path)  # atomic on the same filesystem -- no partial-file readers


def read_day(path: Path | str) -> dict[str, np.ndarray]:
    with open(path, "rb") as f:
        raw = f.read()
    header_len = struct.unpack(">I", raw[:4])[0]
    header = json.loads(raw[4 : 4 + header_len].decode("utf-8"))
    dctx = zstd.ZstdDecompressor()
    blob = dctx.decompress(raw[4 + header_len :])

    out = {}
    offset = 0
    for name in ARRAY_ORDER:
        meta = header[name]
        dtype = np.dtype(meta["dtype"])
        shape = tuple(meta["shape"])
        count = int(np.prod(shape)) if shape else 1
        nbytes = count * dtype.itemsize
        out[name] = np.frombuffer(blob, dtype=dtype, count=count, offset=offset).reshape(shape)
        offset += nbytes
    return out
