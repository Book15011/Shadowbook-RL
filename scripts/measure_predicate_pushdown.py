"""Tests whether pyarrow can skip decoding rows outside a ts-range filter, even
within this file's single row group (parquet page-index / column-index can, in
principle, allow sub-row-group skipping if the file was written with it and the
reader exploits it -- empirical test, not assumed from format knowledge alone).
If a ~3600-row-equivalent filtered window reads meaningfully faster than the
full 864k-row file, real skipping is happening; if it's roughly as slow as a
full read, it isn't (pyarrow decoded everything then filtered in memory).

Run: PYTHONPATH=. .venv/bin/python scripts/measure_predicate_pushdown.py
"""
from __future__ import annotations

import time

import pyarrow.parquet as pq

from src.data.split import load_split

ALL_NEEDED = ["ts", "best_bid", "best_ask", "mid_price", "spread", "bids", "asks"]


def main() -> None:
    pool = load_split("train")[:1]
    path = f"data/raw_l2_bybit/BTCUSDT/l2-BTCUSDT-{pool[0].isoformat()}.parquet"

    pf = pq.ParquetFile(path)
    # check for page/column index presence directly, not just inferred from timing
    has_column_index = False
    try:
        row_group_meta = pf.metadata.row_group(0)
        col_meta = row_group_meta.column(0)
        has_column_index = col_meta.has_index_page if hasattr(col_meta, "has_index_page") else "unknown (attr not exposed)"
    except Exception as e:
        has_column_index = f"check failed: {e}"
    print(f"column index metadata check: {has_column_index}")

    full_table = pq.read_table(path, columns=["ts"])
    ts_values = full_table.column("ts").to_pylist()
    n = len(ts_values)
    print(f"n_rows={n}")

    mid = n // 2
    lo, hi = ts_values[mid], ts_values[mid + 3600]  # a ~3600-row window, mid-file

    pq.read_table(path, columns=ALL_NEEDED)  # warm page cache once before both timed variants

    t0 = time.perf_counter()
    full = pq.read_table(path, columns=ALL_NEEDED)
    t_full = time.perf_counter() - t0
    print(f"full read (864k rows): {t_full*1000:.1f}ms")

    t0 = time.perf_counter()
    filtered = pq.read_table(path, columns=ALL_NEEDED, filters=[("ts", ">=", lo), ("ts", "<", hi)])
    t_filtered = time.perf_counter() - t0
    print(f"filtered read (~3600 rows via ts filter): {t_filtered*1000:.1f}ms, "
          f"returned {filtered.num_rows} rows")
    print(f"speedup: {t_full/t_filtered:.2f}x" if t_filtered > 0 else "N/A")


if __name__ == "__main__":
    main()
