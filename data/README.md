# Data Regeneration

Do not commit raw data, processed data, model artifacts, or logs to Git.
All data output directories (data/raw*) are gitignored.

## Historical trades / aggTrades / bookDepth (Binance futures/um, daily)

    python3 scripts/bulk_backfill.py --start 2021-08-10 --end 2026-08-10

Notes:
- Output path defaults to data/raw.
- The backfill manifest file is data/raw/_manifest.jsonl.
- Use scripts/clean_manifest_bug.py once if stale missing statuses need cleanup.

## L1 aggregate data: klines, funding rate, open interest (Binance futures/um)

    python3 scripts/collect_l1_data.py

Notes:
- Output path defaults to data/raw_l1/{klines_1m,funding_rate,open_interest}/.
- All three datasets are pulled from data.binance.vision archive files
  (zip + per-file CHECKSUM), not the fapi.binance.com REST API. That domain
  is blocked from some hosts (e.g. behind certain proxies); the archive path
  works wherever data.binance.vision is reachable.
- Confirmed real ranges at time of writing: klines daily archive from
  2019-12-31, funding rate monthly archive from 2020-01, open interest
  (Binance "metrics" archive) daily from 2020-09-01. These are the script
  defaults; override with --klines-start/--klines-end,
  --funding-start/--funding-end, --oi-start/--oi-end if needed.
- Resumable via a JSONL manifest per dataset directory; safe to Ctrl+C and
  rerun.
- Run a subset with --datasets klines,funding_rate,open_interest
  (comma-separated).
- Typical total size: well under 1GB (klines is the largest piece at
  roughly 200MB for the full 5+ year range; funding rate and open interest
  are a few hundred KB and tens of MB respectively).

## L2 order book, Bybit linear perpetual (500-level snapshots and deltas)

    python3 scripts/collect_l2_bybit.py

Notes:
- Output path defaults to data/raw_l2_bybit/{symbol}/.
- Auto-discovers the most recent day with real published data on Bybit's
  archive -- its currency drifts over time, so do not assume it matches
  "today".
- Walks backward day by day, reconstructing book state from Bybit's
  snapshot+delta stream using src/data/l2_capture_daemon.py's L2Book and
  apply_diff (reused, not reimplemented), truncating to the top 20 bid/ask
  levels, and deleting each raw .zip immediately after writing its Parquet.
- Stops at --byte-budget-gb (default 35GB of Parquet output) or --max-days
  (default 2000), whichever comes first -- the byte budget is the real
  target, the day cap is a sanity backstop.
- Resumable via a JSONL manifest; safe to Ctrl+C and rerun.
- This is a slow, long-running job -- expect on the order of days rather
  than hours depending on network throughput to Bybit's CDN. Run it in the
  background:

      nohup .venv/bin/python scripts/collect_l2_bybit.py > logs/l2_stdout.log 2>&1 &
      disown

## Live L2 capture (Binance futures/um, websocket, real time)

    python3 scripts/run_l2_capture.py --symbol BTCUSDT --out-dir data/raw_l2

Notes:
- Requires network access to Binance Futures; stop with Ctrl+C.
- Output path defaults to data/raw_l2/{symbol}/{date}/.
- This captures live data going forward only; it does not backfill history.
