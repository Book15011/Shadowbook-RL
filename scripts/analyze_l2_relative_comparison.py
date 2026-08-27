"""Cross-split RELATIVE comparison (2026-08-28, next-round diagnostic battery item).

Re-implements the same methodology from this project's earlier Diagnostic-2 correction
(see docs/TRACK_STATUS.md's L2 section, prior entries): the comparison that matters when
asking "does this checkpoint's edge over baseline differ between train and val" is NOT the
two splits' absolute IS_total_bps numbers -- it is each split's own L2-minus-baseline
DIFFERENCE, compared across splits with an INDEPENDENT-samples test (the two difference
distributions come from disjoint episode pools, so this is Welch's t-test + Mann-Whitney U,
not a paired test).

Sign convention: diff = L2 IS_total_bps - TWAP-passthrough IS_total_bps, per seed, per
split. Positive = L2 worse than baseline on that episode; negative = L2 better.

Input: two --episodes-csv files produced by scripts/eval_l2_diagnostics.py's
--output-episodes-csv (arm,seed,day,is_total_bps,fill_ratio columns, 3 arms x n rows each).

Run:
  PYTHONPATH=. .venv/bin/python scripts/analyze_l2_relative_comparison.py \\
    --val-episodes-csv models/l2_n500_<label>_val_episodes.csv \\
    --train-episodes-csv models/l2_n500_<label>_train_episodes.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import stats


def split_diffs(episodes_csv: str) -> np.ndarray:
    df = pd.read_csv(episodes_csv)
    l2 = df[df["arm"] == "l2"].set_index("seed")["is_total_bps"]
    base = df[df["arm"] == "twap_passthrough"].set_index("seed")["is_total_bps"]
    diff = (l2 - base).dropna()
    return diff.values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cross-split relative (L2-minus-baseline) comparison.")
    parser.add_argument("--val-episodes-csv", type=str, required=True)
    parser.add_argument("--train-episodes-csv", type=str, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    val_diff = split_diffs(args.val_episodes_csv)
    train_diff = split_diffs(args.train_episodes_csv)

    print(f"val   n={len(val_diff)}  L2-minus-baseline mean={val_diff.mean():.4f}bps  std={val_diff.std():.4f}")
    print(f"train n={len(train_diff)}  L2-minus-baseline mean={train_diff.mean():.4f}bps  std={train_diff.std():.4f}")
    swing = val_diff.mean() - train_diff.mean()
    print(f"\nswing (val_diff_mean - train_diff_mean) = {swing:.4f}bps")

    t_stat, t_p = stats.ttest_ind(val_diff, train_diff, equal_var=False)  # Welch's t-test
    u_stat, u_p = stats.mannwhitneyu(val_diff, train_diff, alternative="two-sided")
    pooled_std = np.sqrt((val_diff.std() ** 2 + train_diff.std() ** 2) / 2)
    cohens_d = float(swing / pooled_std) if pooled_std > 0 else float("nan")

    print(f"\nWelch's t-test (independent, unequal variance): t={t_stat:.4f}  p={t_p:.4f}")
    print(f"Mann-Whitney U:                                 U={u_stat:.1f}  p={u_p:.4f}")
    print(f"Cohen's d (independent-samples, pooled std):    d={cohens_d:.4f}")
    print(
        "\n(p<0.05 on both AND |d| not negligible -> the split difference itself is real, "
        "not noise; p>=0.05 on either -> cannot distinguish this swing from a regime/sampling "
        "artifact at this n)"
    )


if __name__ == "__main__":
    main()
