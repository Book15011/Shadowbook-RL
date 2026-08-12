# Data Regeneration

Do not commit raw data, processed data, model artifacts, or logs to Git.

Regenerate historical market data with:

python3 scripts/bulk_backfill.py --start 2021-08-10 --end 2026-08-10

Notes:
- Output path defaults to data/raw.
- The backfill manifest file is data/raw/_manifest.jsonl.
- Use scripts/clean_manifest_bug.py once if stale missing statuses need cleanup.
