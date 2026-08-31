"""Execution predictability: frozen L3 vs. pure TWAP, on the same 500 paired val episodes
already used for the project's n=500 evaluations (2026-08-31).

MEASURES A PROPERTY, NOT A PAYOFF -- READ THIS BEFORE TRUSTING ANY CONCLUSION DRAWN FROM IT.
TWAP's known real-world weakness is that it is trivially predictable (fixed schedule, uniform
slices), which makes it detectable and exploitable by adversarial flow in a real market. This
script asks only "is the trained L3 policy measurably less predictable than TWAP, and by how
much" -- it does NOT and CANNOT show that lower predictability is beneficial, because
LOBExecutionEnv has no adversarial participants and no market impact from the agent's own
orders (confirmed directly: nothing in lob_execution_env.py's fill simulation reads the
agent's own order history to move the book against it). A predictability gap found here is real
evidence about a property of the two policies. It is not evidence that the gap pays off in this
environment, and it should never be reported as if it were.

Two tasks:

TASK 1 -- descriptive regularity metrics (always run). Reconstructs discrete child orders
(scripts/replay_episode.py's reconstruct_child_orders(), imported not reimplemented -- same
already-verified state machine used throughout this project) for both arms on all n episodes,
then compares inter-placement tick gaps, placed order sizes, and price offsets (LIMIT/
CANCEL_AND_REPLACE placements that actually rested, i.e. kind="resting" -- see
_offset_distribution()'s own docstring for why market-kind orders are excluded from this
specific metric) -- distributions, not just means, plus coefficient of variation (CoV = std/mean)
computed two ways: pooled across all episodes (conflates within-episode regularity with
between-episode variation, e.g. different order sizes/arrival conditions) and per-episode-then-
averaged (the more precise "how metronomic is EACH episode's own rhythm" measure). Both are
reported; they answer slightly different questions.

TASK 2 -- direct predictability test (conditional on Task 1 showing a real gap -- if L3's
regularity already looks similar to TWAP's, running a classifier on top adds cost without adding
information, and this script says so and stops there). A small RandomForestClassifier
(n_estimators=50, max_depth=8 -- deliberately shallow; the point is comparative predictability,
not the best possible predictor) predicts the NEXT TICK's order_type (the coarsest, most
behaviorally meaningful "what happens next" question) from a small, symmetric feature set built
identically for both arms: recent mid-price returns, spread level/change, own qty-remaining
fraction, own elapsed-time fraction, ticks since the policy's own last fill/placement, and the
PREVIOUS tick's order_type. Every feature is computed the same way, from the same tick_records
schema, for both arms -- see build_features() for the exact list.

ENCODING FAIRNESS -- READ BEFORE TRUSTING THE NUMBER. order_type (not the full 3-tuple
including price offset and size) is the PRIMARY classification target specifically because
TWAP's own price offset is a hardcoded constant (scripts/phase2a_sanity_suite.py's TWAPPolicy
always emits offset_idx=5, i.e. offset=0, on every single placement -- confirmed directly from
its source, not assumed). Including offset in the primary target would make TWAP's "predictive
accuracy" partly an artifact of a zero-entropy label component baked in by construction, not a
finding about flow predictability -- exactly the kind of encoding choice that could manufacture
the result in either direction, which is why order_type alone is the headline number. Offset and
size are still reported as SEPARATE, secondary classifiers (conditional on a placement actually
happening) specifically so that mechanical, by-construction predictability (TWAP's offset) is
visible and separated from genuine timing/flow predictability (order_type), not hidden by
folding them into one number.

Both arms run through the SAME harness used throughout this project: frozen L3 = the constant
TWAP-passthrough L2 action ([1.0, 0.5]) through FrozenL3Wrapper, i.e. the same "frozen L3,
unsteered" arm as every prior n=500 evaluation's Arm 2 -- NOT a bare, unwrapped L3 (L3's own
observation space includes L2-related dims 15/16 it was trained with; there is no supported way
to run it without SOME wrapper). Pure TWAP = TWAPPolicy(n_slices=10) on the base env, same as
every prior evaluation's Arm 3. Same val date_range, same paired seeds (EVAL_SEED_BASE=5,000,000
.. +n-1). Test split untouched -- val only, per every prior round in this project.

Thread-capping is MANDATORY -- see eval_l2_n500.py's own comment at this same location for the
measured 1,353%-CPU/33-minute/zero-output incident this fixes.

Run (mechanics only, small n):
  PYTHONPATH=. .venv/bin/python -m scripts.analyze_predictability \\
    --l3-checkpoint models/l3_frozen_backup/l3_executioner_v1_frozen.zip \\
    --l3-vecnormalize models/l3_frozen_backup/l3_vecnormalize_frozen.pkl \\
    --n 20 --use-numeric-format --output-dir models/predictability_smoke

Run (real, n=500, matching every other n=500 round in this project):
  PYTHONPATH=. .venv/bin/python -m scripts.analyze_predictability \\
    --l3-checkpoint models/l3_frozen_backup/l3_executioner_v1_frozen.zip \\
    --l3-vecnormalize models/l3_frozen_backup/l3_vecnormalize_frozen.pkl \\
    --n 500 --use-numeric-format --output-dir models/predictability_n500
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import json
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from sb3_contrib import RecurrentPPO
from sklearn.ensemble import RandomForestClassifier
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

torch.set_num_threads(1)  # defense-in-depth -- see comment above the env vars

from scripts.phase2a_sanity_suite import TWAPPolicy
from scripts.replay_episode import (
    ChildOrder,
    install_capture,
    install_tick_capture,
    reconstruct_child_orders,
)
from src.data.split import load_split
from src.envs.lob_execution_env import (
    ORDER_TYPE_CANCEL_REPLACE,
    ORDER_TYPE_HOLD,
    ORDER_TYPE_LIMIT,
    ORDER_TYPE_MARKET,
    LOBExecutionEnv,
)
from src.train.train_l2 import make_l2_wrapped_env

EVAL_SEED_BASE = 5_000_000  # same base as every other n=500 round in this project
HORIZON_TICKS = 3000
LOOKBACK_TICKS = 10
_TWAP_PASSTHROUGH_ACTION = np.array([1.0, 0.5], dtype=np.float32)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="L3 vs TWAP execution predictability (property, not payoff).")
    parser.add_argument("--l3-checkpoint", type=str, required=True)
    parser.add_argument("--l3-vecnormalize", type=str, required=True)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--ticks-per-l2-decision", type=int, default=50)
    parser.add_argument("--l2-include-prev-action", action="store_true")
    parser.add_argument("--data-dir", type=str, default="data/raw_l2_bybit_numeric/BTCUSDT")
    parser.add_argument("--use-numeric-format", action="store_true")
    parser.add_argument("--device", default="cpu", choices=["cuda", "cpu"])
    parser.add_argument("--test-episodes", type=int, default=100,
                         help="Last this-many (by seed order) of the n episodes are held out for "
                         "the Task 2 classifier's test split; the rest are its train split.")
    parser.add_argument("--classifier-seed", type=int, default=42)
    parser.add_argument("--force-task2", action="store_true",
                         help="Run Task 2 even if Task 1's regularity gap looks small (default: "
                         "skip Task 2 and say why, per this round's own instruction).")
    parser.add_argument("--output-dir", type=str, required=True)
    return parser


# ---------------------------------------------------------------------------
# Task 1: reconstruct child orders for both arms, same 500 episodes.
# ---------------------------------------------------------------------------

def make_l3_capture_env(l3_model, l3_vecnormalize_path: str, args, date_range):
    """Built ONCE, reused across all n episodes via .reset(seed=...) -- same pattern
    every other n=500 script in this project uses (day-cache reuse across resets,
    real throughput benefit, not just tidiness). install_capture() must also be
    called exactly ONCE per env: it patches base_env.step/wrapped_env.step by
    wrapping whatever is CURRENTLY installed, so calling it again on the same env
    would nest a second instrumented layer around the first instead of replacing
    it -- silently double-recording and leaking closures. Call capture.tick_records
    .clear() before each episode instead (see run_l3_episode)."""
    wrapped_env = make_l2_wrapped_env(
        date_range, HORIZON_TICKS, LOOKBACK_TICKS, l3_model, l3_vecnormalize_path,
        args.ticks_per_l2_decision, args.l2_include_prev_action,
        data_dir=args.data_dir, l3_deterministic=True, use_numeric_format=args.use_numeric_format,
    )
    capture = install_capture(wrapped_env)
    return wrapped_env, capture


def run_l3_episode(wrapped_env, capture, seed: int, args) -> dict[str, Any]:
    capture.tick_records.clear()
    capture.l2_decision_records.clear()
    obs, info = wrapped_env.reset(seed=seed)
    max_decisions = HORIZON_TICKS // args.ticks_per_l2_decision + 1
    for _ in range(max_decisions):
        obs, r, term, trunc, info = wrapped_env.step(_TWAP_PASSTHROUGH_ACTION)
        if term or trunc:
            break
    base_env: LOBExecutionEnv = wrapped_env.env
    return {
        "tick_records": list(capture.tick_records),  # copy -- next episode clears the original
        "side": base_env.side, "qty_total": base_env.qty_total,
        "is_result": info["implementation_shortfall"],
    }


def make_twap_capture_env(args, date_range):
    """Same one-time-build, one-time-install-capture discipline as make_l3_capture_env
    above, for the identical reason."""
    base_env = LOBExecutionEnv(
        data_dir=args.data_dir, date_range=date_range, horizon_ticks=HORIZON_TICKS,
        lookback_ticks=LOOKBACK_TICKS, use_numeric_format=args.use_numeric_format,
    )
    capture = install_tick_capture(base_env)
    return base_env, capture


def run_twap_episode(base_env, capture, seed: int, args) -> dict[str, Any]:
    capture.tick_records.clear()
    policy = TWAPPolicy(n_slices=10)  # fresh instance per episode == policy.reset()'s own
    # effect (re-zeroes _current_slice/_qty_remaining_at_slice_start in __init__) --
    # sidesteps TWAPPolicy.reset()'s no-args signature (unlike NoOpPolicy/OraclePolicy's
    # reset(env), confirmed from phase2a_sanity_suite.py's own source) without needing
    # the try/except signature-dispatch run_episode() there uses.
    obs, info = base_env.reset(seed=seed)
    for _ in range(HORIZON_TICKS + 1):
        action = policy.act(base_env, info)
        obs, r, term, trunc, info = base_env.step(action)
        if term or trunc:
            break
    return {
        "tick_records": list(capture.tick_records),
        "side": base_env.side, "qty_total": base_env.qty_total,
        "is_result": info["implementation_shortfall"],
    }


def aggregate_placement_events(child_orders: list[ChildOrder]) -> list[dict]:
    """Groups same-tick "market"-kind ChildOrder fragments into ONE placement
    event. reconstruct_child_orders() creates one ChildOrder per FILL EVENT for a
    market action (by design -- useful for its own replay-visualization use case,
    where each individual fill is worth its own marker on a price chart), so a
    single forced-completion market order that walks several thin book levels
    shows up as many same-tick "orders" -- confirmed directly: one real episode's
    single slice-end forced completion produced 13 separate same-tick ChildOrder
    fragments (sizes as small as 0.001-0.13 units each) for what was one policy
    decision. For placement-PATTERN metrics (gaps between decisions, size of each
    decision) that fragmentation is a matching-engine artifact of book depth at
    the moment of a market sweep, not a separate policy decision -- left
    unaggregated it inflates order counts and distorts size/gap distributions,
    and asymmetrically so between arms, since TWAP's forced slice-end completions
    hit this far more often than L3's own near-0% MARKET usage. "resting" orders
    are never fragmented this way (one placement = one ChildOrder already) and
    pass through unchanged."""
    events: list[dict] = []
    for o in child_orders:
        if (o.kind == "market" and events and events[-1]["kind"] == "market"
                and events[-1]["placement_tick"] == o.placement_tick):
            events[-1]["total_size"] += (o.placed_size or 0.0)
            events[-1]["n_fragments"] += 1
        else:
            events.append({
                "placement_tick": o.placement_tick, "kind": o.kind,
                "total_size": o.placed_size or 0.0, "offset_from_touch": o.offset_from_touch,
                "n_fragments": 1,
            })
    return events


def _gaps(events: list[dict]) -> np.ndarray:
    ticks = sorted(e["placement_tick"] for e in events)
    return np.diff(ticks).astype(float) if len(ticks) > 1 else np.array([])


def _sizes(events: list[dict]) -> np.ndarray:
    return np.array([e["total_size"] for e in events if e["total_size"] > 0])


def _offsets(child_orders: list[ChildOrder]) -> np.ndarray:
    """Only kind="resting" orders -- i.e. LIMIT/CANCEL_AND_REPLACE placements that
    actually rested rather than crossing or being a true MARKET action. A true
    ORDER_TYPE_MARKET action's offset isn't a chosen/meaningful value (the book-walk
    ignores it regardless of what was passed -- both TWAP's own forced-completion
    MARKET orders and any MARKET action L3 might take always pass offset_idx=5
    without that value doing anything), so including it would dilute this metric
    with a constant that was never actually "aimed" anywhere. This is a real,
    stated scoping choice, not an oversight -- see this module's own docstring."""
    return np.array([o.offset_from_touch for o in child_orders if o.kind == "resting"], dtype=float)


def _dist_stats(x: np.ndarray) -> dict[str, float]:
    if len(x) == 0:
        return {"n": 0, "mean": None, "std": None, "cov": None,
                "p10": None, "p25": None, "p50": None, "p75": None, "p90": None}
    mean = float(x.mean())
    std = float(x.std())
    pct = np.percentile(x, [10, 25, 50, 75, 90])
    return {
        "n": int(len(x)), "mean": mean, "std": std,
        "cov": (std / mean) if abs(mean) > 1e-12 else None,
        "p10": float(pct[0]), "p25": float(pct[1]), "p50": float(pct[2]),
        "p75": float(pct[3]), "p90": float(pct[4]),
    }


def _per_episode_cov(per_episode_arrays: list[np.ndarray]) -> dict[str, float]:
    covs = [a.std() / a.mean() for a in per_episode_arrays if len(a) >= 2 and abs(a.mean()) > 1e-12]
    if not covs:
        return {"n_episodes": 0, "mean_cov": None, "std_cov": None}
    covs = np.array(covs)
    return {"n_episodes": int(len(covs)), "mean_cov": float(covs.mean()), "std_cov": float(covs.std())}


def task1_regularity(l3_orders_by_ep: list[list[ChildOrder]], twap_orders_by_ep: list[list[ChildOrder]]) -> dict:
    result = {}
    for label, orders_by_ep in [("l3", l3_orders_by_ep), ("twap", twap_orders_by_ep)]:
        events_by_ep = [aggregate_placement_events(o) for o in orders_by_ep]
        gaps_pooled = np.concatenate([_gaps(e) for e in events_by_ep]) if events_by_ep else np.array([])
        sizes_pooled = np.concatenate([_sizes(e) for e in events_by_ep]) if events_by_ep else np.array([])
        offsets_pooled = np.concatenate([_offsets(o) for o in orders_by_ep]) if orders_by_ep else np.array([])
        n_fragments_total = sum(len(o) for o in orders_by_ep)
        n_events_total = sum(len(e) for e in events_by_ep)
        result[label] = {
            "n_episodes": len(orders_by_ep),
            "n_placement_events_total": n_events_total,
            "n_fill_fragments_total": n_fragments_total,  # >= n_placement_events_total; see
            # aggregate_placement_events()'s own docstring for why these differ
            "events_per_episode_mean": float(np.mean([len(e) for e in events_by_ep])) if events_by_ep else None,
            "gaps_pooled": _dist_stats(gaps_pooled),
            "gaps_per_episode_cov": _per_episode_cov([_gaps(e) for e in events_by_ep]),
            "sizes_pooled": _dist_stats(sizes_pooled),
            "sizes_per_episode_cov": _per_episode_cov([_sizes(e) for e in events_by_ep]),
            "offsets_pooled": _dist_stats(offsets_pooled),
            "offsets_per_episode_cov": _per_episode_cov([_offsets(o) for o in orders_by_ep]),
        }
    return result


# ---------------------------------------------------------------------------
# Task 2: fair, symmetric feature set + next-order_type classifier.
# ---------------------------------------------------------------------------

_MAX_CAP_TICKS = 200.0  # cap for the two "ticks since own last X" features, see build_features()


def build_features(tick_records: list[dict], qty_total: float, seed: int) -> pd.DataFrame:
    """One row per tick t, features computed from ticks <= t only (no lookahead),
    label = order_type at t+1 (see label columns added by the caller). Every
    feature here is computed IDENTICALLY for both arms from the same tick_records
    schema (scripts/replay_episode.py's install_tick_capture()) -- this symmetry
    is what makes the L3-vs-TWAP comparison fair; see this module's own docstring
    for the encoding-fairness discussion of the LABEL side of this."""
    n = len(tick_records)
    mid = np.array([r["mid_price"] for r in tick_records])
    bid = np.array([r["best_bid"] for r in tick_records])
    ask = np.array([r["best_ask"] for r in tick_records])
    qty_rem = np.array([r["qty_remaining"] for r in tick_records])
    order_type = np.array([r["order_type"] for r in tick_records])
    offset = np.array([r["offset"] for r in tick_records])
    size_frac = np.array([r["size_frac"] for r in tick_records])
    n_fills = np.array([len(r["fills"]) for r in tick_records])
    is_placement = (order_type != ORDER_TYPE_HOLD)

    mid_ret_1 = np.zeros(n)
    mid_ret_1[1:] = (mid[1:] - mid[:-1]) / mid[:-1]
    mid_ret_5 = np.zeros(n)
    mid_ret_5[5:] = (mid[5:] - mid[:-5]) / mid[:-5]
    spread = ask - bid
    spread_chg_5 = np.zeros(n)
    spread_chg_5[5:] = spread[5:] - spread[:-5]

    ticks_since_fill = np.full(n, _MAX_CAP_TICKS)
    ticks_since_placement = np.full(n, _MAX_CAP_TICKS)
    last_fill, last_place = -np.inf, -np.inf
    for i in range(n):
        ticks_since_fill[i] = min(_MAX_CAP_TICKS, i - last_fill)
        ticks_since_placement[i] = min(_MAX_CAP_TICKS, i - last_place)
        if n_fills[i] > 0:
            last_fill = i
        if is_placement[i]:
            last_place = i

    df = pd.DataFrame({
        "seed": seed,
        "tick_idx": np.arange(n),
        "mid_ret_1": mid_ret_1,
        "mid_ret_5": mid_ret_5,
        "spread": spread,
        "spread_chg_5": spread_chg_5,
        "qty_remaining_frac": qty_rem / qty_total,
        "ticks_elapsed_frac": np.arange(n) / HORIZON_TICKS,
        "ticks_since_fill_norm": ticks_since_fill / _MAX_CAP_TICKS,
        "ticks_since_placement_norm": ticks_since_placement / _MAX_CAP_TICKS,
        "cur_order_type": order_type,  # the action taken AT t -- known before predicting t+1
        "next_order_type": np.roll(order_type, -1),
        "next_offset": np.roll(offset, -1),
        # size_frac's 5 values (SIZE_FRACTIONS = 0.2/0.4/0.6/0.8/1.0) are a discrete
        # action-space choice, but as raw floats sklearn's type_of_target() detects
        # "continuous" and RandomForestClassifier refuses to fit -- rescaled to
        # {2,4,6,8,10} (int) so it's unambiguously read as a multiclass target.
        "next_size_frac": np.round(np.roll(size_frac, -1) * 10).astype(int),
    })
    return df.iloc[:-1]  # drop the last row: its "next" columns wrapped around from roll()


def _majority_baseline_acc(y_train: np.ndarray, y_test: np.ndarray) -> float:
    majority = pd.Series(y_train).mode().iloc[0]
    return float((y_test == majority).mean())


FEATURE_COLS = [
    "mid_ret_1", "mid_ret_5", "spread", "spread_chg_5", "qty_remaining_frac",
    "ticks_elapsed_frac", "ticks_since_fill_norm", "ticks_since_placement_norm", "cur_order_type",
]


def train_and_eval(df_train: pd.DataFrame, df_test: pd.DataFrame, label_col: str, seed: int) -> dict:
    Xtr, ytr = df_train[FEATURE_COLS].values, df_train[label_col].values
    Xte, yte = df_test[FEATURE_COLS].values, df_test[label_col].values
    clf = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=seed, n_jobs=1)
    clf.fit(Xtr, ytr)
    train_acc = float(clf.score(Xtr, ytr))
    test_acc = float(clf.score(Xte, yte))
    base_acc = _majority_baseline_acc(ytr, yte)
    return {
        "n_train": int(len(ytr)), "n_test": int(len(yte)),
        "majority_baseline_test_acc": base_acc,
        "train_acc": train_acc, "test_acc": test_acc,
        "test_acc_minus_baseline": test_acc - base_acc,
    }


def task2_predictability(all_rows: dict[str, pd.DataFrame], train_seeds: set[int], test_seeds: set[int],
                          classifier_seed: int) -> dict:
    result = {}
    for arm, df in all_rows.items():
        df_train = df[df["seed"].isin(train_seeds)]
        df_test = df[df["seed"].isin(test_seeds)]
        arm_result = {"order_type": train_and_eval(df_train, df_test, "next_order_type", classifier_seed)}

        # Secondary, conditional classifiers -- offset only meaningful for LIMIT/REPLACE
        # placements; size only meaningful for any real placement (order_type != HOLD).
        placement_mask_tr = df_train["next_order_type"] != ORDER_TYPE_HOLD
        placement_mask_te = df_test["next_order_type"] != ORDER_TYPE_HOLD
        limit_mask_tr = df_train["next_order_type"].isin([ORDER_TYPE_LIMIT, ORDER_TYPE_CANCEL_REPLACE])
        limit_mask_te = df_test["next_order_type"].isin([ORDER_TYPE_LIMIT, ORDER_TYPE_CANCEL_REPLACE])

        if limit_mask_tr.sum() >= 20 and limit_mask_te.sum() >= 20:
            arm_result["offset_given_limit"] = train_and_eval(
                df_train[limit_mask_tr], df_test[limit_mask_te], "next_offset", classifier_seed)
        else:
            arm_result["offset_given_limit"] = {"note": "too few LIMIT/REPLACE rows to fit"}

        if placement_mask_tr.sum() >= 20 and placement_mask_te.sum() >= 20:
            arm_result["size_given_placement"] = train_and_eval(
                df_train[placement_mask_tr], df_test[placement_mask_te], "next_size_frac", classifier_seed)
        else:
            arm_result["size_given_placement"] = {"note": "too few placement rows to fit"}

        result[arm] = arm_result
    return result


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    seeds = [EVAL_SEED_BASE + i for i in range(args.n)]
    test_seeds = set(seeds[-args.test_episodes:]) if args.test_episodes > 0 else set()
    train_seeds = set(seeds) - test_seeds

    val_dates = load_split("val")
    date_range = (val_dates[0].isoformat(), val_dates[-1].isoformat())
    print(f"val date_range: {date_range} ({len(val_dates)} real days)")
    print(f"n={args.n} paired seeds {seeds[0]}..{seeds[-1]}")
    print(f"classifier split: {len(train_seeds)} train episodes, {len(test_seeds)} test episodes "
          f"(seeds {min(test_seeds) if test_seeds else 'n/a'}..{max(test_seeds) if test_seeds else 'n/a'} held out)")

    l3_model = RecurrentPPO.load(args.l3_checkpoint, device="cpu")
    l3_wrapped_env, l3_capture = make_l3_capture_env(l3_model, args.l3_vecnormalize, args, date_range)
    twap_base_env, twap_capture = make_twap_capture_env(args, date_range)

    l3_orders_by_ep, twap_orders_by_ep = [], []
    l3_rows, twap_rows = [], []
    t0 = time.time()
    for i, seed in enumerate(seeds):
        l3_ep = run_l3_episode(l3_wrapped_env, l3_capture, seed, args)
        twap_ep = run_twap_episode(twap_base_env, twap_capture, seed, args)

        l3_orders_by_ep.append(reconstruct_child_orders(l3_ep["tick_records"], l3_ep["side"]))
        twap_orders_by_ep.append(reconstruct_child_orders(twap_ep["tick_records"], twap_ep["side"]))

        l3_rows.append(build_features(l3_ep["tick_records"], l3_ep["qty_total"], seed))
        twap_rows.append(build_features(twap_ep["tick_records"], twap_ep["qty_total"], seed))

        if (i + 1) % 50 == 0 or (i + 1) == args.n:
            dt = time.time() - t0
            print(f"  {i + 1}/{args.n} episode pairs done ({dt:.1f}s elapsed, {dt / (i + 1):.2f}s/pair)")

    print("\n" + "=" * 70)
    print("TASK 1: DESCRIPTIVE REGULARITY METRICS")
    print("=" * 70)
    t1 = task1_regularity(l3_orders_by_ep, twap_orders_by_ep)
    for label in ("l3", "twap"):
        r = t1[label]
        print(f"\n{label.upper()}: {r['n_placement_events_total']} placement events "
              f"({r['n_fill_fragments_total']} raw fill fragments before aggregation) across "
              f"{r['n_episodes']} episodes ({r['events_per_episode_mean']:.2f} events/episode)")
        for metric in ("gaps", "sizes", "offsets"):
            pooled = r[f"{metric}_pooled"]
            per_ep = r[f"{metric}_per_episode_cov"]
            if pooled["n"] == 0:
                print(f"  {metric}: no data")
                continue
            cov_str = f"{pooled['cov']:.4f}" if pooled["cov"] is not None else "n/a"
            print(f"  {metric}: pooled mean={pooled['mean']:.4f} std={pooled['std']:.4f} "
                  f"CoV={cov_str} | p10={pooled['p10']:.3f} p50={pooled['p50']:.3f} p90={pooled['p90']:.3f} "
                  f"(n={pooled['n']})")
            print(f"    per-episode CoV: mean={per_ep['mean_cov']:.4f} (n_episodes={per_ep['n_episodes']})"
                  if per_ep["mean_cov"] is not None else "    per-episode CoV: n/a")

    gap_cov_l3 = t1["l3"]["gaps_per_episode_cov"]["mean_cov"]
    gap_cov_twap = t1["twap"]["gaps_per_episode_cov"]["mean_cov"]
    similar = (
        gap_cov_l3 is not None and gap_cov_twap is not None
        and abs(gap_cov_l3 - gap_cov_twap) < 0.15 * max(gap_cov_l3, gap_cov_twap, 1e-9)
    )
    print(f"\nGap-timing CoV: L3={gap_cov_l3}, TWAP={gap_cov_twap} -> "
          f"{'SIMILAR (within 15%)' if similar else 'DIFFERENT'}")

    result = {"n": args.n, "task1": t1, "task2_run": False, "task2": None}

    run_task2 = args.force_task2 or not similar
    if not run_task2:
        print("\nTask 1 shows similar regularity between L3 and TWAP -- per this round's own "
              "instruction, Task 2 is skipped (pass --force-task2 to override).")
    else:
        print("\n" + "=" * 70)
        print("TASK 2: DIRECT PREDICTABILITY TEST")
        print("=" * 70)
        df_l3 = pd.concat(l3_rows, ignore_index=True)
        df_twap = pd.concat(twap_rows, ignore_index=True)
        t2 = task2_predictability({"l3": df_l3, "twap": df_twap}, train_seeds, test_seeds, args.classifier_seed)
        result["task2_run"] = True
        result["task2"] = t2
        for arm in ("l3", "twap"):
            ot = t2[arm]["order_type"]
            print(f"\n{arm.upper()} order_type: n_train={ot['n_train']} n_test={ot['n_test']} "
                  f"majority_baseline={ot['majority_baseline_test_acc']:.4f} "
                  f"train_acc={ot['train_acc']:.4f} test_acc={ot['test_acc']:.4f} "
                  f"(test-baseline={ot['test_acc_minus_baseline']:+.4f})")
            for sub in ("offset_given_limit", "size_given_placement"):
                s = t2[arm][sub]
                if "note" in s:
                    print(f"  {sub}: {s['note']}")
                else:
                    print(f"  {sub}: majority_baseline={s['majority_baseline_test_acc']:.4f} "
                          f"test_acc={s['test_acc']:.4f} (test-baseline={s['test_acc_minus_baseline']:+.4f})")
        gap = t2["l3"]["order_type"]["test_acc"] - t2["twap"]["order_type"]["test_acc"]
        print(f"\nHeadline gap (L3 test_acc - TWAP test_acc, order_type): {gap:+.4f}")

    out_path = os.path.join(args.output_dir, "predictability_result.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
