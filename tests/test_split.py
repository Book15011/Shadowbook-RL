"""Tests for src/data/split.py (architecture_spec.md Section 2.5).

Uses both the REAL data/raw_l2_bybit/BTCUSDT/ directory (to validate the
actual persisted artifact against what is really on disk, per the Phase 3
preflight task's explicit requirement) and small synthetic fixtures (for
hand-computed, disk-state-independent checks of the boundary logic
itself).
"""
import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from src.data.split import (
    DATA_DIR,
    SPLIT_PATH,
    generate_split,
    load_split,
    write_split,
)


def _dates_ascending_no_overlap(name_a, dates_a, name_b, dates_b):
    assert set(dates_a).isdisjoint(dates_b), f"{name_a} and {name_b} overlap"
    assert dates_a[-1] < dates_b[0], (
        f"{name_a}'s latest ({dates_a[-1]}) should be < {name_b}'s earliest ({dates_b[0]})"
    )


def test_real_split_disjoint_and_chronologically_ordered():
    artifact = generate_split(data_dir=DATA_DIR)
    train = [date.fromisoformat(s) for s in artifact["train_dates"]]
    val = [date.fromisoformat(s) for s in artifact["val_dates"]]
    test = [date.fromisoformat(s) for s in artifact["test_dates"]]

    assert train and val and test
    _dates_ascending_no_overlap("train", train, "val", val)
    _dates_ascending_no_overlap("val", val, "test", test)
    # each split internally sorted ascending
    assert train == sorted(train)
    assert val == sorted(val)
    assert test == sorted(test)


def test_real_split_total_matches_disk_right_now():
    artifact = generate_split(data_dir=DATA_DIR)
    real_files_now = len(list(DATA_DIR.glob("*.parquet")))
    total_in_artifact = (
        len(artifact["train_dates"]) + len(artifact["val_dates"]) + len(artifact["test_dates"])
    )
    assert total_in_artifact == artifact["source_day_count"]
    assert total_in_artifact == real_files_now


def _touch(data_dir: Path, day: date) -> None:
    (data_dir / f"l2-BTCUSDT-{day.isoformat()}.parquet").write_bytes(b"")


def test_generate_split_synthetic_hand_computed(tmp_path):
    data_dir = tmp_path / "BTCUSDT"
    data_dir.mkdir()
    # 10 consecutive real dates, split_size=3 -> test=last 3, val=next 3 back, train=rest.
    start = date(2024, 1, 1)
    all_dates = [start + timedelta(days=i) for i in range(10)]
    for d in all_dates:
        _touch(data_dir, d)

    artifact = generate_split(data_dir=data_dir, split_size=3)
    assert artifact["source_day_count"] == 10
    assert artifact["train_dates"] == [d.isoformat() for d in all_dates[:4]]
    assert artifact["val_dates"] == [d.isoformat() for d in all_dates[4:7]]
    assert artifact["test_dates"] == [d.isoformat() for d in all_dates[7:10]]
    assert artifact["known_gap_dates"] == []


def test_generate_split_synthetic_hand_computed_with_gaps(tmp_path):
    data_dir = tmp_path / "BTCUSDT"
    data_dir.mkdir()
    # dates 2024-01-01 .. 2024-01-10 with 01-04 and 01-07 missing (8 real files)
    present = [date(2024, 1, 1) + timedelta(days=i) for i in range(10) if i not in (3, 6)]
    for d in present:
        _touch(data_dir, d)

    artifact = generate_split(data_dir=data_dir, split_size=3)
    assert artifact["source_day_count"] == 8
    assert artifact["known_gap_dates"] == ["2024-01-04", "2024-01-07"]
    assert artifact["test_dates"] == [d.isoformat() for d in present[-3:]]
    assert artifact["val_dates"] == [d.isoformat() for d in present[-6:-3]]
    assert artifact["train_dates"] == [d.isoformat() for d in present[:-6]]


def test_generate_split_raises_when_not_enough_real_files(tmp_path):
    data_dir = tmp_path / "BTCUSDT"
    data_dir.mkdir()
    for i in range(5):
        _touch(data_dir, date(2024, 1, 1) + timedelta(days=i))
    with pytest.raises(ValueError):
        generate_split(data_dir=data_dir, split_size=3)  # needs >= 7 for a non-empty train


def test_load_split_raises_if_artifact_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("src.data.split.SPLIT_PATH", tmp_path / "does_not_exist.json")
    with pytest.raises(FileNotFoundError):
        load_split("train")


def test_load_split_rejects_bad_name(tmp_path, monkeypatch):
    path = tmp_path / "split.json"
    path.write_text(json.dumps({"train_dates": [], "val_dates": [], "test_dates": []}))
    monkeypatch.setattr("src.data.split.SPLIT_PATH", path)
    with pytest.raises(ValueError):
        load_split("bogus")


def test_write_split_then_load_split_round_trip(tmp_path, monkeypatch):
    data_dir = tmp_path / "BTCUSDT"
    data_dir.mkdir()
    all_dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(10)]
    for d in all_dates:
        _touch(data_dir, d)
    split_path = tmp_path / "_split.json"
    monkeypatch.setattr("src.data.split.SPLIT_PATH", split_path)

    write_split(data_dir=data_dir, split_path=split_path, split_size=3)
    assert split_path.exists()

    assert load_split("train") == all_dates[:4]
    assert load_split("val") == all_dates[4:7]
    assert load_split("test") == all_dates[7:10]


def test_write_split_refuses_if_val_or_test_would_change(tmp_path):
    data_dir = tmp_path / "BTCUSDT"
    data_dir.mkdir()
    split_path = tmp_path / "_split.json"
    all_dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(10)]
    for d in all_dates:
        _touch(data_dir, d)
    write_split(data_dir=data_dir, split_path=split_path, split_size=3)

    # simulate an unexpected change to what should be the frozen val/test window: a new,
    # more-recent file appears, shifting what a fresh generate_split() would compute for
    # both val_dates and test_dates relative to what is already persisted.
    _touch(data_dir, all_dates[-1] + timedelta(days=1))
    with pytest.raises(RuntimeError):
        write_split(data_dir=data_dir, split_path=split_path, split_size=3)
