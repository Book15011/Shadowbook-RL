"""Parallel full-dataset conversion (train+val+test -- all 441 files live in
one physical directory, data/raw_l2_bybit/BTCUSDT/, partitioned into splits
only by date range, so converting every file there covers all three splits in
one pass). Each day's conversion is independent CPU+I/O work (JSON parse +
zstd compress), so process-level parallelism scales close to core count on
this pure-CPU, no-GPU-contention job. Skips a day if its output already
exists (the 10 benchmark-pool days converted during the equivalence-gate
round), so re-running this is idempotent -- does not re-convert unnecessarily.

The ORIGINAL parquet files are only ever opened for reading (read_parquet) --
never written, moved, or deleted, per hard boundary.

Run: PYTHONPATH=. .venv/bin/python scripts/convert_l2_to_numeric_parallel.py
"""
from __future__ import annotations

import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.l2_numeric_format import write_day

SRC_DIR = Path("data/raw_l2_bybit/BTCUSDT")
OUT_DIR = Path("data/raw_l2_bybit_numeric/BTCUSDT")
N_LEVELS = 20
N_WORKERS = 4  # reduced from 8 (2026-08-24): the original 8-worker run OOM-killed 6
# workers within its first ~3.5 minutes (dmesg confirmed, each holding 6.1-7.8GB RSS at
# time of death) -- per-worker memory scales with file size via the row-by-row
# json.loads() parsing approach, and 8 concurrent workers on this box's available
# memory (~18-36GB free depending on what else is running) can exceed it, especially
# since the files remaining after the crash skew toward the largest in the dataset.
# multiprocessing.Pool silently drops the result for any task whose worker is
# SIGKILLed (no retry, no exception) -- the run hung forever waiting for that lost
# result once all other dispatchable work was exhausted. 4 workers is conservative
# given the remaining files skew large; write_day() writes atomically, so no
# partial/corrupt output risk from re-running this.


def convert_one(src_path: Path) -> tuple[str, float, int, int]:
    out_path = OUT_DIR / (src_path.stem + ".npzst")
    if out_path.exists():
        return (src_path.name, 0.0, out_path.stat().st_size, -1)  # -1 = skipped, already converted

    t0 = time.perf_counter()
    df = pd.read_parquet(src_path, columns=["ts", "best_bid", "best_ask", "mid_price", "spread", "bids", "asks"])
    n = len(df)

    bid_prices = np.empty((n, N_LEVELS), dtype=np.float64)
    bid_sizes = np.empty((n, N_LEVELS), dtype=np.float64)
    ask_prices = np.empty((n, N_LEVELS), dtype=np.float64)
    ask_sizes = np.empty((n, N_LEVELS), dtype=np.float64)
    for i, (bids_str, asks_str) in enumerate(zip(df["bids"].to_numpy(), df["asks"].to_numpy())):
        bid_levels = json.loads(bids_str)
        ask_levels = json.loads(asks_str)
        if len(bid_levels) != N_LEVELS or len(ask_levels) != N_LEVELS:
            raise ValueError(
                f"row {i} in {src_path}: expected exactly {N_LEVELS} levels/side, "
                f"got bid={len(bid_levels)} ask={len(ask_levels)} -- stopping rather than "
                "silently padding, this file needs manual handling before conversion."
            )
        bid_arr = np.asarray(bid_levels, dtype=np.float64)
        ask_arr = np.asarray(ask_levels, dtype=np.float64)
        bid_prices[i] = bid_arr[:, 0]
        bid_sizes[i] = bid_arr[:, 1]
        ask_prices[i] = ask_arr[:, 0]
        ask_sizes[i] = ask_arr[:, 1]

    arrays = {
        "ts": df["ts"].to_numpy(dtype=np.int64),
        "best_bid": df["best_bid"].to_numpy(dtype=np.float64),
        "best_ask": df["best_ask"].to_numpy(dtype=np.float64),
        "mid_price": df["mid_price"].to_numpy(dtype=np.float64),
        "spread": df["spread"].to_numpy(dtype=np.float64),
        "bid_prices": bid_prices, "bid_sizes": bid_sizes,
        "ask_prices": ask_prices, "ask_sizes": ask_sizes,
    }
    write_day(arrays, out_path)
    elapsed = time.perf_counter() - t0
    return (src_path.name, elapsed, out_path.stat().st_size, n)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    src_files = sorted(SRC_DIR.glob("*.parquet"))
    print(f"converting {len(src_files)} files from {SRC_DIR} -> {OUT_DIR}, {N_WORKERS} workers")

    t0 = time.perf_counter()
    total_bytes = 0
    n_converted = 0
    n_skipped = 0
    with Pool(N_WORKERS) as pool:
        for i, (name, elapsed, out_size, n_rows) in enumerate(pool.imap_unordered(convert_one, src_files)):
            total_bytes += out_size
            if n_rows == -1:
                n_skipped += 1
            else:
                n_converted += 1
            if (i + 1) % 25 == 0 or i == len(src_files) - 1:
                elapsed_total = time.perf_counter() - t0
                print(f"  [{i+1}/{len(src_files)}] elapsed={elapsed_total/60:.1f}min "
                      f"converted={n_converted} skipped={n_skipped}")

    total_elapsed = time.perf_counter() - t0
    print(f"\ndone: {len(src_files)} files ({n_converted} converted, {n_skipped} already present), "
          f"{total_elapsed/60:.1f}min total, {total_bytes/1e9:.2f}GB output")


if __name__ == "__main__":
    main()
