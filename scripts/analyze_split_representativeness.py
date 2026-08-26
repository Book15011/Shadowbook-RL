"""Split representativeness check (2026-08-27, built to resolve a specific
question during the L2 post-mortem: is the negative measured on val a
regime artifact of val's own 18 days being systematically choppier/
wider-spread than train, or is val broadly representative of train's
conditions? -- but written generally: any future round measuring anything
on val (or wanting to sanity-check test before spending it) can reuse this
without re-deriving the day-conditions computation from scratch).

Pure market-data descriptive statistics -- day_return_bps/realized_vol_bps/
mean_spread computed directly from each day's mid_price/spread series via
src.data.l2_numeric_format.read_day(). No model inference, no episode
evaluation, nothing touches FrozenL3Wrapper/LOBExecutionEnv/model.predict.
This is why it is safe to include the 18 test days here even though the
project's own discipline is to never evaluate against test: reading a raw
price series to compute its own volatility is not "spending" the holdout in
the sense that matters (no model/policy conclusion is drawn from test data),
it is the same category of housekeeping as checking file counts or date
ranges. Actually running any of the eval harnesses (eval_l2_n500.py,
eval_l2_diagnostics.py) against test_dates would be spending it; this script
never does that.

Run: PYTHONPATH=. .venv/bin/python scripts/analyze_split_representativeness.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.data.l2_numeric_format import read_day
from src.data.split import load_split

DATA_DIR = "data/raw_l2_bybit_numeric/BTCUSDT"


def day_conditions(dates, split_name: str) -> pd.DataFrame:
    rows = []
    for d in dates:
        path = f"{DATA_DIR}/l2-BTCUSDT-{d.isoformat()}.npzst"
        data = read_day(path)
        mid = data["mid_price"]
        spread = data["spread"]
        day_return_bps = (mid[-1] - mid[0]) / mid[0] * 1e4
        rets = np.diff(mid) / mid[:-1]
        realized_vol_bps = float(np.std(rets) * 1e4)
        mean_spread = float(np.mean(spread))
        rows.append(
            {
                "split": split_name, "day": d.isoformat(),
                "day_return_bps": day_return_bps, "abs_return_bps": abs(day_return_bps),
                "realized_vol_bps": realized_vol_bps, "mean_spread": mean_spread,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    train_dates = load_split("train")
    val_dates = load_split("val")
    test_dates = load_split("test")
    print(f"train: {len(train_dates)} days, val: {len(val_dates)} days, test: {len(test_dates)} days")

    train_df = day_conditions(train_dates, "train")
    val_df = day_conditions(val_dates, "val")
    test_df = day_conditions(test_dates, "test")

    train_df.to_csv("models/l2_day_conditions_train.csv", index=False)
    val_df.to_csv("models/l2_day_conditions_val.csv", index=False)
    test_df.to_csv("models/l2_day_conditions_test.csv", index=False)

    print("\n" + "=" * 70)
    print("SUMMARY STATS BY SPLIT")
    print("=" * 70)
    for metric in ["realized_vol_bps", "mean_spread", "abs_return_bps"]:
        print(f"\n{metric}:")
        for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            s = df[metric]
            print(
                f"  {name:5s}: mean={s.mean():9.4f}  std={s.std():9.4f}  "
                f"median={s.median():9.4f}  min={s.min():9.4f}  max={s.max():9.4f}"
            )

    print("\n" + "=" * 70)
    print("WHERE VAL SITS WITHIN TRAIN'S DISTRIBUTION")
    print("=" * 70)
    for metric in ["realized_vol_bps", "mean_spread", "abs_return_bps"]:
        train_vals = train_df[metric].values
        val_mean = val_df[metric].mean()
        pct_of_mean = float(stats.percentileofscore(train_vals, val_mean))
        per_day_pct = [float(stats.percentileofscore(train_vals, v)) for v in val_df[metric].values]
        print(f"\n{metric}: val's MEAN sits at train percentile {pct_of_mean:.1f}")
        print(f"  individual val-day percentiles within train: {sorted(round(p, 1) for p in per_day_pct)}")
        print(
            f"  median of those percentiles: {np.median(per_day_pct):.1f}  "
            f"(50=typical, >50=val days skew choppier/wider than train, <50=calmer)"
        )
        u, p = stats.mannwhitneyu(val_df[metric].values, train_vals, alternative="two-sided")
        print(f"  Mann-Whitney U (val vs train, are the two distributions different): U={u:.1f} p={p:.4f}")

    print("\n" + "=" * 70)
    print("TEST SPLIT (descriptive only, for context -- not evaluated)")
    print("=" * 70)
    for metric in ["realized_vol_bps", "mean_spread", "abs_return_bps"]:
        train_vals = train_df[metric].values
        test_mean = test_df[metric].mean()
        pct = float(stats.percentileofscore(train_vals, test_mean))
        print(f"{metric}: test's MEAN sits at train percentile {pct:.1f}")


if __name__ == "__main__":
    main()
