"""Measures the real cost of pd.read_parquet with all columns vs. only the 7
columns _build_ticks/_precompute_feature_series actually touch (ts, best_bid,
best_ask, mid_price, spread, bids, asks -- symbol/update_id/seq confirmed unused
anywhere in lob_execution_env.py). Real files, several distinct days (not the
same file repeated, to avoid OS-page-cache masking the comparison), several
trials each with the process's page cache pre-warmed identically for both
variants (read once, discard, then time N repeats) so this isolates parquet
decode cost, not disk I/O variance.

Run: PYTHONPATH=. .venv/bin/python scripts/measure_column_pruning.py
"""
from __future__ import annotations

import statistics
import time

import pandas as pd

from src.data.split import load_split

NEEDED_COLUMNS = ["ts", "best_bid", "best_ask", "mid_price", "spread", "bids", "asks"]
N_TRIALS = 5


def time_read(path, columns=None) -> float:
    t0 = time.perf_counter()
    df = pd.read_parquet(path, columns=columns)
    elapsed = time.perf_counter() - t0
    del df
    return elapsed


def main() -> None:
    pool = load_split("train")[:5]
    for d in pool:
        path = f"data/raw_l2_bybit/BTCUSDT/l2-BTCUSDT-{d.isoformat()}.parquet"
        # warm the OS page cache identically before each variant's timed trials
        pd.read_parquet(path)

        full_times = [time_read(path) for _ in range(N_TRIALS)]
        pruned_times = [time_read(path, columns=NEEDED_COLUMNS) for _ in range(N_TRIALS)]

        mean_full = statistics.mean(full_times)
        mean_pruned = statistics.mean(pruned_times)
        pct = 100 * (mean_full - mean_pruned) / mean_full
        print(
            f"{d.isoformat()}: full={mean_full*1000:.1f}ms (stdev {statistics.stdev(full_times)*1000:.1f}) "
            f"pruned={mean_pruned*1000:.1f}ms (stdev {statistics.stdev(pruned_times)*1000:.1f}) "
            f"-> {pct:.1f}% faster"
        )


if __name__ == "__main__":
    main()
