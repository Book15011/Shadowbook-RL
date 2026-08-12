# scripts/clean_manifest_bug.py — run once, before resuming
import json, collections
from pathlib import Path

path = Path("data/raw/_manifest.jsonl")
entries = [json.loads(l) for l in open(path) if l.strip()]

by_date = collections.defaultdict(list)
for e in entries:
    by_date[e["date"]].append(e)

kept, dropped = [], 0
for date, group in by_date.items():
    statuses = {e["dataset"]: e["status"] for e in group}
    both_core_missing = statuses.get("trades") == "missing" and statuses.get("aggTrades") == "missing"
    for e in group:
        if e["dataset"] == "bookDepth" and e["status"] == "missing":
            dropped += 1; continue          # known URL-path bug
        if both_core_missing:
            dropped += 1; continue          # whole day looks bogus — BTCUSDT trades every day
        kept.append(e)

with open(path, "w") as f:
    for e in kept:
        f.write(json.dumps(e) + "\n")
print(f"kept={len(kept)} dropped={dropped}")