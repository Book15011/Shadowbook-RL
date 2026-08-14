"""Chronological train/val/test split (architecture_spec.md Section 2.5).

Splits are chronological only -- train on the oldest days, val on the next
block, test on the most recent block. Never shuffle across dates: that
leaks lookahead (a policy trained partly on days AFTER its validation
window is not honestly validated) and silently invalidates every IS/
mark-out number reported downstream, including Section 4.4's ablation.

Boundary heuristic (Section 2.5): test = the most recent ~15-20 REAL
(present-on-disk) day-files; val = the next ~15-20 real day-files back
from test's boundary; train = everything older.

Interpretation flag: "most recent ~15-20 calendar days actually present"
is read here as a fixed REAL-FILE COUNT (SPLIT_SIZE), not a fixed
calendar-day window. Checked directly against data/raw_l2_bybit/BTCUSDT/
during Phase 3 preflight: a naive most-recent-18-CALENDAR-day window
contained only 7 real files (11 of 18 days missing) -- an unacceptable
shrink for a set meant to be "held out entirely" for the final backtest.
Picking by real-file count guarantees the intended day count, at the cost
of a wider (gappier) calendar span -- e.g. test's 18 real files currently
span 34 calendar days, not 18. That span, and every known gap date, is
recorded in the artifact (known_gap_dates) rather than silently absorbed.

Regenerating: run `python -m src.data.split` after backfill adds more
OLDER days. Per Section 2.5 this should not be necessary to keep val/test
valid, since backfill only appends days older than train's boundary --
but re-running is harmless and will refuse (raise) if any val/test date
present in an existing artifact would change, since that is a genuine
anomaly (files in the supposedly-frozen recent window changed) worth
investigating, not something to silently regenerate past.
"""
from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

DATA_DIR = Path("data/raw_l2_bybit/BTCUSDT")
SPLIT_PATH = Path("data/splits/l2_bybit_btcusdt_split.json")  # NOT under data/raw* (gitignored) -- this small artifact is meant to be tracked
SPLIT_SIZE = 18  # per split (val, test) -- Section 2.5's "~15-20" heuristic

_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def _discover_present_dates(data_dir: Path = DATA_DIR) -> list[date]:
    """Real day-files actually present on disk, sorted ascending. Filename
    convention l2-{symbol}-{YYYY-MM-DD}.parquet, matching
    scripts/collect_l2_bybit.py's day_str = day.isoformat()."""
    dates = []
    for p in sorted(data_dir.glob("*.parquet")):
        m = _DATE_RE.search(p.stem)
        if m is None:
            raise ValueError(f"Could not parse a YYYY-MM-DD date out of filename: {p.name}")
        dates.append(date.fromisoformat(m.group(1)))
    return sorted(dates)


def _known_gap_dates(present: list[date]) -> list[date]:
    """Every calendar date strictly between present[0] and present[-1]
    (inclusive) that has no real file -- see architecture_spec.md Section
    2.1.1 for why these gaps exist (failed downloads, genuine 404s on
    Bybit's archive for a given day)."""
    if not present:
        return []
    present_set = set(present)
    gaps = []
    d = present[0]
    while d <= present[-1]:
        if d not in present_set:
            gaps.append(d)
        d += timedelta(days=1)
    return gaps


def generate_split(data_dir: Path = DATA_DIR, split_size: int = SPLIT_SIZE) -> dict:
    """Computes the split boundaries fresh from whatever is actually present
    on disk right now. Does not read or write the persisted artifact --
    see write_split() for that."""
    present = _discover_present_dates(data_dir)
    if len(present) < 2 * split_size + 1:
        raise ValueError(
            f"Only {len(present)} real day-files in {data_dir}, need at least "
            f"{2 * split_size + 1} for a {split_size}/{split_size} val/test split "
            "with a non-empty train set."
        )

    test_dates = present[-split_size:]
    val_dates = present[-2 * split_size : -split_size]
    train_dates = present[: -2 * split_size]
    gap_dates = _known_gap_dates(present)

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_day_count": len(present),
        "train_dates": [d.isoformat() for d in train_dates],
        "val_dates": [d.isoformat() for d in val_dates],
        "test_dates": [d.isoformat() for d in test_dates],
        "known_gap_dates": [d.isoformat() for d in gap_dates],
    }


def write_split(
    data_dir: Path = DATA_DIR, split_path: Path = SPLIT_PATH, split_size: int = SPLIT_SIZE
) -> dict:
    """Generates the split and persists it to split_path. Refuses (raises)
    if an artifact already exists there and its val_dates/test_dates would
    change -- per the module docstring, that is a signal something
    unexpected happened to the supposedly-frozen recent window, not
    something to silently regenerate past."""
    artifact = generate_split(data_dir=data_dir, split_size=split_size)

    if split_path.exists():
        existing = json.loads(split_path.read_text())
        for key in ("val_dates", "test_dates"):
            if existing.get(key) != artifact[key]:
                raise RuntimeError(
                    f"{key} in the existing artifact at {split_path} does not match "
                    f"what would be generated now -- per architecture_spec.md Section "
                    f"2.5, val/test should be stable once generated (backfill only adds "
                    f"OLDER days). Investigate before overwriting; existing={existing.get(key)!r} "
                    f"new={artifact[key]!r}"
                )

    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(artifact, indent=2) + "\n")
    return artifact


def load_split(name: str) -> list[date]:
    """name in {'train', 'val', 'test'}. Reads the persisted split artifact;
    raises if the artifact doesn't exist yet (must be generated once, explicitly,
    not implicitly on first use)."""
    if name not in ("train", "val", "test"):
        raise ValueError(f"name must be one of 'train', 'val', 'test', got {name!r}")
    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"Split artifact not found at {SPLIT_PATH} -- generate it once via "
            "write_split() (or `python -m src.data.split`) before calling load_split()."
        )
    artifact = json.loads(SPLIT_PATH.read_text())
    return [date.fromisoformat(s) for s in artifact[f"{name}_dates"]]


if __name__ == "__main__":
    result = write_split()
    print(f"Wrote split artifact to {SPLIT_PATH}")
    for split_name in ("train", "val", "test"):
        dates = result[f"{split_name}_dates"]
        print(f"  {split_name:5s}: {len(dates):4d} dates ({dates[0]} .. {dates[-1]})")
    print(f"  known_gap_dates: {len(result['known_gap_dates'])}")
