"""Isolates where parquet decode cost actually concentrates: bids/asks (JSON
strings) vs the 5 small numeric/ts columns, and checks whether explicit
multi-threaded decode (pyarrow use_threads) helps on a single read (relevant
before deciding whether to touch it, given n_envs=8 workers already run with
threads capped to 1 each to avoid oversubscription -- turning pyarrow threading
back on could reintroduce exactly that problem at the SubprocVecEnv scale).

Run: PYTHONPATH=. .venv/bin/python scripts/measure_column_split.py
"""
from __future__ import annotations

import statistics
import time

import pandas as pd
import pyarrow.parquet as pq

from src.data.split import load_split

NUMERIC_ONLY = ["ts", "best_bid", "best_ask", "mid_price", "spread"]
BOOK_ONLY = ["bids", "asks"]
ALL_NEEDED = NUMERIC_ONLY + BOOK_ONLY
N_TRIALS = 5


def time_pandas_read(path, columns) -> float:
    t0 = time.perf_counter()
    df = pd.read_parquet(path, columns=columns)
    elapsed = time.perf_counter() - t0
    del df
    return elapsed


def time_pyarrow_read(path, columns, use_threads) -> float:
    t0 = time.perf_counter()
    table = pq.read_table(path, columns=columns, use_threads=use_threads)
    df = table.to_pandas()
    elapsed = time.perf_counter() - t0
    del df, table
    return elapsed


def main() -> None:
    pool = load_split("train")[:3]
    for d in pool:
        path = f"data/raw_l2_bybit/BTCUSDT/l2-BTCUSDT-{d.isoformat()}.parquet"
        pd.read_parquet(path)  # warm page cache once before all variants below

        numeric_times = [time_pandas_read(path, NUMERIC_ONLY) for _ in range(N_TRIALS)]
        book_times = [time_pandas_read(path, BOOK_ONLY) for _ in range(N_TRIALS)]
        all_needed_times = [time_pandas_read(path, ALL_NEEDED) for _ in range(N_TRIALS)]
        pyarrow_mt_times = [time_pyarrow_read(path, ALL_NEEDED, True) for _ in range(N_TRIALS)]
        pyarrow_st_times = [time_pyarrow_read(path, ALL_NEEDED, False) for _ in range(N_TRIALS)]

        print(f"=== {d.isoformat()} ===")
        print(f"  5 numeric cols only:      {statistics.mean(numeric_times)*1000:.1f}ms")
        print(f"  bids+asks only:           {statistics.mean(book_times)*1000:.1f}ms")
        print(f"  all 7 needed (pandas):    {statistics.mean(all_needed_times)*1000:.1f}ms")
        print(f"  all 7 needed (pyarrow, use_threads=True):  {statistics.mean(pyarrow_mt_times)*1000:.1f}ms")
        print(f"  all 7 needed (pyarrow, use_threads=False): {statistics.mean(pyarrow_st_times)*1000:.1f}ms")


if __name__ == "__main__":
    main()
