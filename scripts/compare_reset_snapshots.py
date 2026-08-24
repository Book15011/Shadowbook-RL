"""Diffs /tmp/reset_snapshot_before.pkl vs /tmp/reset_snapshot_after.pkl -- asserts
byte-identical (np.array_equal, not np.allclose) observations, rewards, and
terminal IS across every seed. Prints PASS/FAIL per field; on any mismatch,
prints the first divergent index/value pair rather than just "not equal", so a
real regression is diagnosable, not just detected.

Run: PYTHONPATH=. .venv/bin/python scripts/compare_reset_snapshots.py
"""
from __future__ import annotations

import pickle

import numpy as np

with open("/tmp/reset_snapshot_before.pkl", "rb") as f:
    before = pickle.load(f)
with open("/tmp/reset_snapshot_after.pkl", "rb") as f:
    after = pickle.load(f)

assert set(before.keys()) == set(after.keys()), "seed sets differ"

all_ok = True
for seed in sorted(before.keys()):
    b, a = before[seed], after[seed]
    for key in b:
        bv, av = b[key], a[key]
        if isinstance(bv, np.ndarray):
            ok = np.array_equal(bv, av)
            if not ok:
                diff_idx = np.argwhere(bv != av)
                first = tuple(diff_idx[0]) if len(diff_idx) else None
                print(f"seed={seed} key={key}: MISMATCH at {len(diff_idx)} positions, "
                      f"first={first} before={bv[first] if first else None} after={av[first] if first else None}")
                all_ok = False
        else:
            ok = (bv == av) or (bv is None and av is None)
            if not ok:
                print(f"seed={seed} key={key}: MISMATCH before={bv!r} after={av!r}")
                all_ok = False
    print(f"seed={seed}: {'PASS' if all(np.array_equal(b[k], a[k]) if isinstance(b[k], np.ndarray) else b[k] == a[k] for k in b) else 'FAIL'}")

print()
print("=== OVERALL:", "PASS -- byte-identical across all seeds/fields" if all_ok else "FAIL -- see mismatches above", "===")
