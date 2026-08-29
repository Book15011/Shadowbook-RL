"""Direct paired checkpoint-vs-checkpoint comparison: l2v3 final (gamma=0.983) vs.
l2v2 pre-divergence (gamma=0.995, step 1,599,936) -- same reward, same paired seeds,
gamma the only difference. Isolates whether the stable critic under gamma=0.983
produced a better policy or just a numerically calmer one converging to the same
place. Reuses paired_report's exact methodology (paired t-test + Wilcoxon + Cohen's
d_z) via the same two episodes CSVs' arm='l2' rows, matched by seed -- no new
episodes are run, this is pure post-hoc analysis of two already-completed n=500 evals.

Run: PYTHONPATH=. .venv/bin/python scripts/compare_l2v3_vs_l2v2predivergence.py \\
  --a models/l2_diagnostics_l2v3final_val_episodes.csv --a-label l2v3_final \\
  --b models/l2_n500_l2v2predivergence_val_episodes.csv --b-label l2v2_predivergence
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import stats


def load_l2_arm(path: str) -> pd.Series:
    df = pd.read_csv(path)
    df = df[df["arm"] == "l2"].set_index("seed")["is_total_bps"]
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True)
    parser.add_argument("--a-label", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--b-label", required=True)
    args = parser.parse_args()

    a = load_l2_arm(args.a)
    b = load_l2_arm(args.b)
    common_seeds = sorted(set(a.index) & set(b.index))
    assert len(common_seeds) > 0, "no overlapping seeds -- cannot pair"
    if len(common_seeds) != len(a) or len(common_seeds) != len(b):
        print(f"WARNING: seed sets not identical -- a has {len(a)}, b has {len(b)}, "
              f"{len(common_seeds)} common. Restricting to common seeds.")
    a = a.loc[common_seeds]
    b = b.loc[common_seeds]

    print(f"n={len(common_seeds)} paired seeds {common_seeds[0]}..{common_seeds[-1]}")
    print(f"{args.a_label}: mean={a.mean():.4f} std={a.std():.4f}")
    print(f"{args.b_label}: mean={b.mean():.4f} std={b.std():.4f}")

    diff = b.values - a.values  # b - a: positive means a (l2v3) is lower/better
    t_stat, t_p = stats.ttest_rel(b.values, a.values)
    w_stat, w_p = stats.wilcoxon(b.values, a.values)
    d_z = float(diff.mean() / diff.std()) if diff.std() > 0 else float("nan")

    print(f"\n{args.b_label} vs {args.a_label}: mean diff ({args.b_label}-{args.a_label})={diff.mean():.4f}bps "
          f"std={diff.std():.4f}  Cohen's d_z={d_z:.4f}")
    print(f"  paired t-test:        t={t_stat:.4f}  p={t_p:.4f}")
    print(f"  Wilcoxon signed-rank: W={w_stat:.4f}  p={w_p:.4f}")

    a_better = diff.mean() > 0 and t_p < 0.05 and w_p < 0.05
    b_better = diff.mean() < 0 and t_p < 0.05 and w_p < 0.05
    if a_better:
        print(f"\n{args.a_label} beats {args.b_label} (both tests agree, p<0.05).")
    elif b_better:
        print(f"\n{args.b_label} beats {args.a_label} (both tests agree, p<0.05).")
    else:
        print(f"\nNo significant difference at the pre-registered bar (both tests must agree).")


if __name__ == "__main__":
    main()
