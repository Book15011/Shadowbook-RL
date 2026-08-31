"""Episode replay visualizer -- makes one episode of the trained L2 policy legible to
a trading/finance reader, not a codebase reader.

Runs the REAL FrozenL3Wrapper/LOBExecutionEnv unmodified (wrappers.py and
lob_execution_env.py are not touched by this script) and captures tick-level detail
via monkeypatching -- the same instrumentation pattern used throughout this project
(scripts/profile_l2_throughput.py, scripts/profile_reset.py): the patched step()
methods call the real, unmodified originals and return their real result unchanged,
only recording additional data as a side channel. This capture exists ONLY inside
this script's own process -- it is never installed unless this script runs, so it is
inert by construction for every other caller (training, eval, everything else).

Two episodes are run at the SAME seed for a same-window comparison: (1) the real L2
policy steering frozen L3 through FrozenL3Wrapper, (2) a scripted TWAP baseline on the
base env (scripts/phase2a_sanity_suite.py's TWAPPolicy, unmodified) -- identical
day/window/side/size draw, since the RNG sequence is fixed by the seed, so the two
IS_total_bps numbers are directly comparable, not just similar in spirit.

CLI takes explicit --l2-checkpoint/--l2-vecnormalize/--l3-checkpoint/
--l3-vecnormalize/--seed -- no defaults, same discipline as train_l2.py/eval_l2_n500.py.
--data-dir auto-resolves to the numeric or original archive based on --use-numeric-format
(same convention as train_l2.py's own NUMERIC_DATA_DIR/PARQUET_DATA_DIR) unless overridden.

--frozen-l3-only: skips --l2-checkpoint/--l2-vecnormalize entirely and drives the episode
with the constant TWAP-passthrough action [1.0, 0.5] instead -- frozen L3, completely
unsteered. Exists specifically as a correctness check on THIS TOOL, not on L2: L3's own
n=500 evaluation already measured its action-type distribution (HOLD 47.6% / LIMIT 52.0%
/ MARKET 0.02% / REPLACE 0.36%) -- a --frozen-l3-only replay's printed action-type
breakdown should land close to those numbers (one episode vs. n=500, so exact match isn't
expected, but a wildly different shape -- e.g. mostly MARKET orders -- would mean this
visualizer has a bug, not that frozen L3 changed).

reconstruct_child_orders()'s only flagged limitation (a crossing placement that partially
fills and rests the remainder on the same tick) is RESOLVED, not open: read directly from
_place_limit()'s source -- the `crossed` branch's `return fills` is unconditional (fires
whether walk_market_fill() fully or only partially fills the crossing size), so the
resting-order code below it never executes when crossed=True. That function's own comment
confirms it: "Prices that cross the opposing side never reach here." The scenario this
limitation described cannot happen -- 5 tests already covered the reachable cases; a 6th
(added this round) confirms a partial crossing fill is handled correctly too (marked
filled with the partial amount, never left looking like it is still resting).

Run (synthetic data, mechanics only):
  PYTHONPATH=. .venv/bin/python scripts/replay_episode.py --help
Run (real, frozen-L3-only sanity check -- do this FIRST, before any L2 checkpoint exists
or is trusted, per the reasoning above):
  PYTHONPATH=. .venv/bin/python scripts/replay_episode.py \\
    --frozen-l3-only \\
    --l3-checkpoint models/l3_frozen_backup/l3_executioner_v1_frozen.zip \\
    --l3-vecnormalize models/l3_frozen_backup/l3_vecnormalize_frozen.pkl \\
    --seed 5000000 --use-numeric-format --output replay_frozen_l3_seed5000000.png
Run (real, full L2 policy, only after training completes):
  PYTHONPATH=. .venv/bin/python scripts/replay_episode.py \\
    --l2-checkpoint models/l2_strategist_v1.zip --l2-vecnormalize models/l2_vecnormalize.pkl \\
    --l3-checkpoint models/l3_frozen_backup/l3_executioner_v1_frozen.zip \\
    --l3-vecnormalize models/l3_frozen_backup/l3_vecnormalize_frozen.pkl \\
    --seed 5000000 --use-numeric-format --output replay_seed5000000.png
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any

import matplotlib
matplotlib.use("Agg")  # no interactive display required
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from scripts.phase2a_sanity_suite import TWAPPolicy, run_episode
from src.data.split import load_split
from src.envs.lob_execution_env import (
    ORDER_TYPE_CANCEL_REPLACE, ORDER_TYPE_LIMIT, ORDER_TYPE_MARKET, SIZE_FRACTIONS, TICK_SIZE,
)
from src.envs.lob_execution_env import LOBExecutionEnv
from src.envs.wrappers import FrozenL3Wrapper
from src.train.train_l2 import make_l2_wrapped_env

HORIZON_TICKS = 3000
LOOKBACK_TICKS = 10


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize one L2 episode for a trading audience.")
    parser.add_argument(
        "--l2-checkpoint", type=str, default=None,
        help="Required unless --frozen-l3-only (then must be omitted).",
    )
    parser.add_argument(
        "--l2-vecnormalize", type=str, default=None,
        help="Required unless --frozen-l3-only (then must be omitted).",
    )
    parser.add_argument("--l3-checkpoint", type=str, required=True)
    parser.add_argument("--l3-vecnormalize", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True, help="Episode seed -- no default, pick one deliberately.")
    parser.add_argument(
        "--frozen-l3-only", action="store_true",
        help="Drive the episode with the constant TWAP-passthrough action [1.0, 0.5] instead "
        "of an L2 policy -- frozen L3, completely unsteered. --l2-checkpoint/--l2-vecnormalize "
        "must be omitted when this is set. A correctness check on THIS TOOL: compare the "
        "printed action-type breakdown against L3's own n=500 numbers (HOLD 47.6% / LIMIT "
        "52.0% / MARKET 0.02% / REPLACE 0.36%).",
    )
    parser.add_argument("--ticks-per-l2-decision", type=int, default=50)
    parser.add_argument("--l2-include-prev-action", action="store_true")
    parser.add_argument("--n-slices", type=int, default=10, help="TWAP comparison baseline's own slice count.")
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Defaults to the numeric or original archive based on --use-numeric-format "
        "(same convention as train_l2.py's own NUMERIC_DATA_DIR/PARQUET_DATA_DIR) -- set "
        "explicitly only to point at something else (e.g. synthetic test data).",
    )
    parser.add_argument("--use-numeric-format", action="store_true")
    parser.add_argument("--device", default="cpu", choices=["cuda", "cpu"])
    parser.add_argument("--output", type=str, default=None, help="PNG path; default replay_seed<seed>.png")
    return parser


NUMERIC_DATA_DIR = "data/raw_l2_bybit_numeric/BTCUSDT"
PARQUET_DATA_DIR = "data/raw_l2_bybit/BTCUSDT"


# ---------------------------------------------------------------------------
# Capture: monkeypatch instrumentation, inert unless this script installs it.
# ---------------------------------------------------------------------------

@dataclass
class EpisodeCapture:
    tick_records: list[dict[str, Any]] = field(default_factory=list)
    l2_decision_records: list[dict[str, Any]] = field(default_factory=list)


def install_tick_capture(base_env: LOBExecutionEnv, capture: EpisodeCapture | None = None) -> EpisodeCapture:
    """The tick-level half of install_capture() below, factored out so it can be
    installed on ANY LOBExecutionEnv directly -- not just one sitting inside a
    FrozenL3Wrapper. Used by scripts/analyze_predictability.py to capture a pure
    TWAPPolicy's own tick-level actions on the bare base env, where there is no
    wrapper (and no L2 decisions) to speak of -- same instrumentation, same
    tick_records schema, so downstream analysis (reconstruct_child_orders() etc.)
    works identically on either capture's output."""
    capture = capture if capture is not None else EpisodeCapture()
    orig_env_step = base_env.step

    def instrumented_env_step(action):
        tick_idx = base_env._tick_idx
        tick = base_env._current_tick()
        order_type, offset_idx, size_frac_idx = int(action[0]), int(action[1]), int(action[2])
        obs, r, term, trunc, info = orig_env_step(action)
        capture.tick_records.append({
            "tick_idx": tick_idx,
            "ts": tick.ts,
            "mid_price": tick.mid_price,
            "best_bid": tick.best_bid,
            "best_ask": tick.best_ask,
            "order_type": order_type,
            "offset": offset_idx - 5,
            "size_frac": SIZE_FRACTIONS[size_frac_idx],
            "fills": list(info["fills_this_step"]),
            "canceled_via_market": info["canceled_via_market"],
            "canceled_via_replace": info["canceled_via_replace"],
            "qty_remaining": info["qty_remaining"],
            "resting_own_remaining": info["resting_own_remaining"],
        })
        return obs, r, term, trunc, info

    base_env.step = instrumented_env_step
    return capture


def install_capture(wrapped_env: FrozenL3Wrapper) -> EpisodeCapture:
    base_env: LOBExecutionEnv = wrapped_env.env
    capture = install_tick_capture(base_env)

    orig_wrapped_step = wrapped_env.step

    def instrumented_wrapped_step(l2_action):
        tick_idx_at_decision = base_env._tick_idx
        result = orig_wrapped_step(l2_action)
        capture.l2_decision_records.append({
            "tick_idx": tick_idx_at_decision,
            "participation_mult": float(np.asarray(l2_action)[0]),
            "urgency": float(np.asarray(l2_action)[1]),
            "l2_target_slice_ratio_override": base_env.l2_target_slice_ratio_override,
        })
        return result

    wrapped_env.step = instrumented_wrapped_step
    return capture


# ---------------------------------------------------------------------------
# Reconstruct discrete child orders from the tick-level capture.
# ---------------------------------------------------------------------------

@dataclass
class ChildOrder:
    kind: str  # "resting" or "market"
    placement_tick: int
    placement_price: float
    offset_from_touch: int
    outcome: str  # "filled", "replaced", "open_at_episode_end" (resting) / "filled" (market)
    fill_ticks: list[int] = field(default_factory=list)
    fill_qtys: list[float] = field(default_factory=list)
    fill_prices: list[float] = field(default_factory=list)
    placed_size: float | None = None  # the quantity requested AT PLACEMENT, not the
    # (possibly smaller) filled quantity -- for "resting" orders this is the env's own
    # info["resting_own_remaining"] captured on the placement tick (ground truth,
    # confirmed a fresh non-crossing placement can never absorb a same-tick maker
    # fill, so this value is uncontaminated by same-tick fills -- see this function's
    # own docstring); for "market"-kind orders (both ORDER_TYPE_MARKET and a crossing
    # LIMIT/CANCEL_AND_REPLACE) it is sum(fill_qtys), since those fill immediately and
    # in full against available depth by construction.


def reconstruct_child_orders(tick_records: list[dict], side: int) -> list[ChildOrder]:
    """Walks the tick-level capture and regroups it into discrete child-order
    lifetimes (placement -> outcome), mirroring the real env's own resting-order
    state machine (LOBExecutionEnv.step(), in that exact order: evolve the
    existing resting order against market activity, apply an explicit cancel,
    THEN decide whether the tick's action places something new) instead of
    inferring placements from raw action types alone.

    Real bug this replaced, found by comparing a real episode's reconstructed
    order count (1,693 orders from 3,000 ticks, ~56% of which were LIMIT/
    CANCEL_AND_REPLACE actions) against how rarely resting orders should
    actually turn over: the previous version created a new ChildOrder for
    EVERY recorded LIMIT/CANCEL_AND_REPLACE tick, regardless of whether the
    real env's self._resting was already occupied. But step()'s own dispatch
    (`elif order_type in (LIMIT, CANCEL_REPLACE): if self._resting is None and
    self.qty_remaining > 0: ...`) makes a LIMIT action issued while an order is
    ALREADY resting a silent no-op there -- not a new placement, and not a
    replace. The old code both fabricated phantom placements and mislabeled
    still-resting orders "replaced" every time L3 kept emitting LIMIT while
    already resting (which this session's own action-distribution checks show
    is common: L3 was never trained to prefer HOLD once resting). Fixed by
    shadowing self._resting.own_qty_remaining directly from the env's own
    info["resting_own_remaining"] (ground truth, captured every tick -- not
    re-derived from size_frac/qty_remaining): maker fills deplete it first
    (a same-tick fresh placement can never receive a same-tick maker fill --
    _place_limit()'s non-crossing branch only ever creates a QueueState, no
    fill), an explicit cancel (canceled_via_market/canceled_via_replace)
    clears it, and only THEN does a LIMIT/CANCEL_AND_REPLACE tick get treated
    as a genuine new placement -- exactly matching step()'s own precondition.

    Crossing placements: _place_limit()'s `crossed` branch returns
    unconditionally after walk_market_fill(), so a crossing LIMIT/CANCEL_AND_
    REPLACE never rests any remainder, partial or otherwise -- confirmed from
    that function's own source, not assumed. Handled here as an immediate
    non-maker fill on the placement's own tick, relabeled kind="market"."""
    orders: list[ChildOrder] = []
    open_order: ChildOrder | None = None
    resting_remaining: float | None = None  # shadows self._resting.own_qty_remaining

    for rec in tick_records:
        # 1. Evolve: a maker fill this tick can only belong to an order that
        # was ALREADY resting entering this tick (see docstring) -- so if
        # nothing was resting, there is nothing to evolve.
        if resting_remaining is not None:
            for f in rec["fills"]:
                if f.get("is_maker"):
                    open_order.fill_ticks.append(rec["tick_idx"])
                    open_order.fill_qtys.append(f["qty"])
                    open_order.fill_prices.append(f["price"])
                    resting_remaining = max(0.0, resting_remaining - f["qty"])
            if resting_remaining is not None and resting_remaining <= 1e-9:
                open_order.outcome = "filled"
                resting_remaining = None

        # 2. Explicit cancel -- matches step()'s own MARKET/CANCEL_AND_REPLACE
        # teardown branches, which both run before the placement check.
        if rec["canceled_via_market"] or rec["canceled_via_replace"]:
            if resting_remaining is not None and open_order is not None and open_order.outcome == "open_at_episode_end":
                open_order.outcome = "replaced"
            resting_remaining = None

        # 3. Apply the action.
        if rec["order_type"] in (ORDER_TYPE_LIMIT, ORDER_TYPE_CANCEL_REPLACE):
            if resting_remaining is None:
                touch = rec["best_bid"] if side == 1 else rec["best_ask"]
                price = round(touch + rec["offset"] * TICK_SIZE, 1) if side == 1 else round(touch - rec["offset"] * TICK_SIZE, 1)
                open_order = ChildOrder(
                    kind="resting", placement_tick=rec["tick_idx"], placement_price=price,
                    offset_from_touch=rec["offset"], outcome="open_at_episode_end",
                )
                orders.append(open_order)
                resting_remaining = rec["resting_own_remaining"]
                open_order.placed_size = resting_remaining
                # Crossed-price immediate fill: same-tick non-maker fill means this
                # "placement" never actually rested -- it executed immediately, like a
                # market order.
                for f in rec["fills"]:
                    if not f.get("is_maker"):
                        open_order.fill_ticks.append(rec["tick_idx"])
                        open_order.fill_qtys.append(f["qty"])
                        open_order.fill_prices.append(f["price"])
                        open_order.outcome = "filled"
                        open_order.kind = "market"
                        resting_remaining = None
                if open_order.kind == "market":
                    open_order.placed_size = sum(open_order.fill_qtys)
            # else: already resting -- a real no-op in the env, not a new order.

        elif rec["order_type"] == ORDER_TYPE_MARKET:
            for f in rec["fills"]:
                orders.append(ChildOrder(
                    kind="market", placement_tick=rec["tick_idx"], placement_price=f["price"],
                    offset_from_touch=0, outcome="filled",
                    fill_ticks=[rec["tick_idx"]], fill_qtys=[f["qty"]], fill_prices=[f["price"]],
                    placed_size=f["qty"],
                ))

    return orders


# ---------------------------------------------------------------------------
# Run the two episodes.
# ---------------------------------------------------------------------------

_TWAP_PASSTHROUGH_ACTION = np.array([1.0, 0.5], dtype=np.float32)  # same as train_l2.py's
                                                                     # ValISEvalCallback


def run_l2_episode(args, val_date_range: tuple[str, str]) -> dict[str, Any]:
    l3_model = RecurrentPPO.load(args.l3_checkpoint, device="cpu")
    wrapped_env = make_l2_wrapped_env(
        val_date_range, HORIZON_TICKS, LOOKBACK_TICKS, l3_model, args.l3_vecnormalize,
        args.ticks_per_l2_decision, args.l2_include_prev_action,
        data_dir=args.data_dir, l3_deterministic=True, use_numeric_format=args.use_numeric_format,
    )
    capture = install_capture(wrapped_env)

    if args.frozen_l3_only:
        action_fn = lambda obs: _TWAP_PASSTHROUGH_ACTION
    else:
        l2_model = SAC.load(args.l2_checkpoint, device=args.device)
        l2_vec_normalize = VecNormalize.load(args.l2_vecnormalize, DummyVecEnv([lambda: wrapped_env]))
        l2_vec_normalize.training = False

        def action_fn(obs):
            obs_for_policy = l2_vec_normalize.normalize_obs(obs[None, :])
            action, _ = l2_model.predict(obs_for_policy, deterministic=True)
            return action[0]

    obs, info = wrapped_env.reset(seed=args.seed)
    max_decisions = HORIZON_TICKS // args.ticks_per_l2_decision + 1
    for _ in range(max_decisions):
        obs, r, term, trunc, info = wrapped_env.step(action_fn(obs))
        if term or trunc:
            break

    base_env: LOBExecutionEnv = wrapped_env.env
    return {
        "is_result": info["implementation_shortfall"],
        "side": base_env.side, "qty_total": base_env.qty_total,
        "arrival_price": base_env.arrival_price,
        "tick_records": capture.tick_records, "l2_decision_records": capture.l2_decision_records,
        "child_orders": reconstruct_child_orders(capture.tick_records, base_env.side),
    }


def run_twap_baseline(args, val_date_range: tuple[str, str]) -> dict[str, Any]:
    base_env = LOBExecutionEnv(
        data_dir=args.data_dir, date_range=val_date_range, horizon_ticks=HORIZON_TICKS,
        lookback_ticks=LOOKBACK_TICKS, use_numeric_format=args.use_numeric_format,
    )
    result = run_episode(base_env, TWAPPolicy(n_slices=args.n_slices), seed=args.seed, horizon_ticks=HORIZON_TICKS)
    return result


# ---------------------------------------------------------------------------
# Plain-language labels.
# ---------------------------------------------------------------------------

def _side_label(side: int) -> str:
    return "BUY" if side == 1 else "SELL"


def _price_formatter(prices: list[float]) -> FuncFormatter:
    """Full number, thousands-separated -- no scientific/offset notation (e.g.
    matplotlib's default '+1.189e5' axis label), which a finance reader should not
    have to decode. Decimal precision adapts to how much the price actually moves
    in THIS episode: whole dollars would collapse a calm, sub-dollar episode's
    y-axis ticks to a single repeated label (e.g. every tick reading '118,980'),
    which happened during review and is a real loss of information, not just
    cosmetic -- a reader could no longer see that the price moved at all."""
    span = (max(prices) - min(prices)) if prices else 0.0
    decimals = 0 if span >= 50 else 1 if span >= 5 else 2 if span >= 0.5 else 3
    return FuncFormatter(lambda x, _pos=None: f"{x:,.{decimals}f}")


# ---------------------------------------------------------------------------
# Figure.
# ---------------------------------------------------------------------------

def _episode_title(l2_result: dict) -> str:
    side = l2_result["side"]
    mode_label = "frozen L3 (unsteered, no L2 policy)" if l2_result.get("frozen_l3_only") else "L2 policy"
    return (
        f"Episode replay -- {mode_label} -- seed={l2_result.get('seed', '?')} -- {_side_label(side)} "
        f"{l2_result['qty_total']:.3f} units, arrival price {l2_result['arrival_price']:,.1f}"
    )


def _extract_series(l2_result: dict) -> dict[str, Any]:
    """Shared per-tick/per-decision series, computed once and reused by both the
    combined overview figure and the separate per-panel figures below, so the two
    can never silently drift apart."""
    tick_records = l2_result["tick_records"]
    l2_decisions = l2_result["l2_decision_records"]
    qty_total = l2_result["qty_total"]

    ticks = [r["tick_idx"] for r in tick_records]
    t0 = ticks[0] if ticks else 0
    rel_ticks = [t - t0 for t in ticks]
    qty_remaining_series = [r["qty_remaining"] for r in tick_records]

    return {
        "t0": t0,
        "rel_ticks": rel_ticks,
        "mids": [r["mid_price"] for r in tick_records],
        "bids": [r["best_bid"] for r in tick_records],
        "asks": [r["best_ask"] for r in tick_records],
        "dec_ticks": [d["tick_idx"] - t0 for d in l2_decisions],
        "part_mult": [d["participation_mult"] for d in l2_decisions],
        "urgency": [d["urgency"] for d in l2_decisions],
        "executed_frac": [1.0 - qr / qty_total for qr in qty_remaining_series],
        "schedule_frac": [min(1.0, rt / HORIZON_TICKS) for rt in rel_ticks],
    }


def _mark_decisions(ax, dec_ticks: list[int], label: bool = True) -> None:
    """Thin vertical line at every L2 decision boundary, drawn on all three panels so
    a reader can line up 'what happened to price/fills right after decision N' across
    them -- this is the direct answer to 'let me see the agent choose at a specific
    stage,' rather than requiring the reader to cross-reference tick numbers by eye.
    Numbered D1/D2/... labels only when there are few enough to stay legible; past
    that the lines alone (a longer episode, up to the ~60-decision max) still mark
    every decision without cluttering the panel with unreadable overlapping text."""
    for dt in dec_ticks:
        ax.axvline(dt, color="#999999", ls=":", lw=0.7, alpha=0.6, zorder=1)
    if label and 0 < len(dec_ticks) <= 20:
        # Bottom, not top -- the legend defaults to "upper left" on every panel in
        # this script, so a top-anchored label collides with it whenever an early
        # decision sits near the left edge (as D1/D2 typically do).
        ylim = ax.get_ylim()
        y = ylim[0] + 0.03 * (ylim[1] - ylim[0])
        for i, dt in enumerate(dec_ticks):
            ax.text(dt, y, f"D{i + 1}", fontsize=7, color="#666666", ha="left", va="bottom")


def _draw_price_panel(ax, l2_result: dict, s: dict) -> None:
    side = l2_result["side"]
    arrival_price = l2_result["arrival_price"]
    rel_ticks, mids, bids, asks = s["rel_ticks"], s["mids"], s["bids"], s["asks"]

    ax.plot(rel_ticks, mids, color="#333333", lw=1.1, label="mid price")
    ax.fill_between(rel_ticks, bids, asks, color="#cccccc", alpha=0.4, label="bid/ask spread")
    ax.axhline(arrival_price, color="#1f77b4", ls="--", lw=1.2, label=f"arrival price ({arrival_price:,.1f})")
    if mids:
        ax.scatter([rel_ticks[-1]], [mids[-1]], color="#d62728", zorder=5, s=40,
                   label=f"terminal mid ({mids[-1]:,.1f})")

    outcome_style = {
        "filled": dict(marker="^" if side == 1 else "v", color="#2ca02c", label="resting order -- filled"),
        "replaced": dict(marker="x", color="#ff7f0e", label="resting order -- replaced before filling"),
        "open_at_episode_end": dict(marker="o", color="#7f7f7f", label="resting order -- still open at end"),
    }
    seen_labels: set[str] = set()
    for order in l2_result["child_orders"]:
        if order.kind == "resting":
            style = outcome_style[order.outcome]
            lbl = style["label"] if style["label"] not in seen_labels else None
            seen_labels.add(style["label"])
            ax.scatter([order.placement_tick - s["t0"]], [order.placement_price], marker=style["marker"],
                       color=style["color"], s=70, zorder=6, label=lbl, edgecolors="black", linewidths=0.5)
        else:
            lbl = "market/crossing fill" if "market/crossing fill" not in seen_labels else None
            seen_labels.add("market/crossing fill")
            ax.scatter([order.placement_tick - s["t0"]], [order.placement_price], marker="*",
                       color="#9467bd", s=110, zorder=7, label=lbl, edgecolors="black", linewidths=0.5)

    _mark_decisions(ax, s["dec_ticks"])
    ax.yaxis.set_major_formatter(_price_formatter(bids + asks + mids))
    ax.set_ylabel("price (USDT)")
    ax.set_title(
        "Price path and child order placements\n"
        "(dotted lines = each L2 decision point, labeled D1, D2, ... -- see the steering panel for what was chosen at each)",
        fontsize=10,
    )
    ax.legend(loc="upper left", fontsize=7, ncol=2)


def _draw_execution_panel(ax, s: dict) -> None:
    rel_ticks, executed_frac, schedule_frac = s["rel_ticks"], s["executed_frac"], s["schedule_frac"]
    ax.plot(rel_ticks, schedule_frac, color="#1f77b4", ls="--", lw=1.3, label="linear TWAP schedule")
    ax.plot(rel_ticks, executed_frac, color="#2ca02c", lw=1.6, label="actual fraction executed")
    ax.fill_between(rel_ticks, executed_frac, schedule_frac,
                     where=[e >= sch for e, sch in zip(executed_frac, schedule_frac)],
                     color="#2ca02c", alpha=0.15, interpolate=True, label="ahead of schedule")
    ax.fill_between(rel_ticks, executed_frac, schedule_frac,
                     where=[e < sch for e, sch in zip(executed_frac, schedule_frac)],
                     color="#d62728", alpha=0.15, interpolate=True, label="behind schedule")
    _mark_decisions(ax, s["dec_ticks"], label=False)
    ax.set_ylabel("fraction of order filled")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("Execution progress vs. on-schedule TWAP pace", fontsize=10)
    ax.legend(loc="upper left", fontsize=7)


def _draw_steering_panel(ax, s: dict, annotate_values: bool = True) -> None:
    dec_ticks, part_mult, urgency = s["dec_ticks"], s["part_mult"], s["urgency"]
    rel_ticks = s["rel_ticks"]
    x_end = rel_ticks[-1] if rel_ticks else (dec_ticks[-1] if dec_ticks else 1)

    ax.step(dec_ticks, part_mult, where="post", color="#1f77b4", lw=1.6, label="participation-rate multiplier")
    ax.step(dec_ticks, urgency, where="post", color="#e377c2", lw=1.6, label="urgency")
    ax.axhline(1.0, color="#1f77b4", ls=":", lw=0.8, alpha=0.6)
    ax.axhline(0.5, color="#e377c2", ls=":", lw=0.8, alpha=0.6)

    # Extra x/y headroom so the value labels -- placed just above/below each step,
    # including right at the data's own min/max -- don't get clipped by the
    # axes/figure boundary (seen concretely: a value of 2.0, the participation-rate
    # ceiling, is a common real value here, and its "+6 points" label offset was
    # landing right on the top border without this).
    x_lo = dec_ticks[0] if dec_ticks else 0
    ax.set_xlim(x_lo - 0.02 * (x_end - x_lo), x_end + 0.05 * (x_end - x_lo))
    y_hi = max([*part_mult, *urgency, 1.0], default=1.0)
    y_lo = min([*part_mult, *urgency, 0.0], default=0.0)
    ax.set_ylim(y_lo - 0.12 * (y_hi - y_lo), y_hi + 0.12 * (y_hi - y_lo))

    # Label the exact chosen value at every decision directly on its own segment --
    # answers "what did the agent actually choose at stage N" without making the
    # reader eyeball a y-axis position, which is the whole point of this panel. Only
    # in the standalone figure (annotate_values=True, the default): the combined
    # overview's panel is short enough that the fixed-size legend box covers a large
    # share of it regardless of where a label is placed, so the overview shows shape
    # only and leaves exact values to the larger, dedicated figure.
    if annotate_values:
        n = len(dec_ticks)
        for i, dt in enumerate(dec_ticks):
            seg_end = dec_ticks[i + 1] if i + 1 < n else x_end
            mid_x = (dt + seg_end) / 2
            ax.annotate(f"{part_mult[i]:.2f}", (mid_x, part_mult[i]), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=7, color="#1f77b4", fontweight="bold",
                        annotation_clip=False)
            ax.annotate(f"{urgency[i]:.2f}", (mid_x, urgency[i]), textcoords="offset points",
                        xytext=(0, -11), ha="center", fontsize=7, color="#e377c2", fontweight="bold",
                        annotation_clip=False)

    _mark_decisions(ax, dec_ticks, label=False)
    ax.set_ylabel("L2 action value")
    ax.set_title(
        "L2's steering, one decision at a time (labels = exact value chosen)\n"
        "1.0=on-schedule pace, 2.0=max catch-up burst, 0=defer  |  urgency: 0.5=neutral",
        fontsize=10,
    )
    ax.legend(loc="upper left", fontsize=7)


def _summary_lines(l2_result: dict, twap_result: dict) -> list[str]:
    """Short explicit lines, none wide enough to overflow the figure at this
    fontsize -- a single long f-string wrapped only at spaces was found to run off
    the right edge of the saved PNG, fixed by splitting deliberately rather than
    relying on matplotlib's own wrapping."""
    is_result = l2_result["is_result"]
    twap_is = twap_result["is_result"]
    exec_str = f"{is_result.is_exec_bps:+.2f}bps" if is_result.is_exec_bps is not None else "n/a (no fills)"
    verdict = "BEAT" if is_result.is_total_bps < twap_is.is_total_bps else "LOST TO"
    verdict_gap = abs(is_result.is_total_bps - twap_is.is_total_bps)
    arm_label = "Frozen L3 (unsteered)" if l2_result.get("frozen_l3_only") else "L2 policy"
    return [
        f"{arm_label}:  fill_ratio={is_result.fill_ratio:.1%}   IS_total={is_result.is_total_bps:+.2f}bps",
        f"  (execution={exec_str}, opportunity={is_result.is_opp_bps:+.2f}bps, fees={is_result.fees_bps:+.2f}bps)",
        f"TWAP, same window/seed:  fill_ratio={twap_is.fill_ratio:.1%}   IS_total={twap_is.is_total_bps:+.2f}bps",
        f"{arm_label} {verdict} TWAP by {verdict_gap:.2f}bps on this single episode "
        f"(not a significance test -- see scripts/eval_l2_n500.py).",
    ]


def _add_summary_box(fig, l2_result: dict, twap_result: dict, fontsize: int) -> None:
    fig.text(0.5, 0.01, "\n".join(_summary_lines(l2_result, twap_result)), ha="center", va="bottom",
              fontsize=fontsize, family="monospace",
              bbox=dict(boxstyle="round", facecolor="#f0f0f0", edgecolor="#999999"))


def build_figure(l2_result: dict, twap_result: dict, output_path: str) -> None:
    """Combined 3-panel overview -- one at-a-glance read. Each panel is cramped by
    sharing vertical space with the other two; see build_separate_figures() for a
    larger, individually detailed version of each, which is the better read when the
    price axis's full numbers or the steering panel's per-decision labels matter."""
    s = _extract_series(l2_result)

    fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True, height_ratios=[2.2, 1.3, 1.2])
    fig.suptitle(_episode_title(l2_result), fontsize=13, fontweight="bold")

    _draw_price_panel(axes[0], l2_result, s)
    _draw_execution_panel(axes[1], s)
    _draw_steering_panel(axes[2], s, annotate_values=False)
    axes[2].set_xlabel("ticks since episode start")

    _add_summary_box(fig, l2_result, twap_result, fontsize=8)
    fig.tight_layout(rect=[0, 0.12, 1, 0.95])
    fig.savefig(output_path, dpi=130)
    plt.close(fig)


def build_separate_figures(l2_result: dict, twap_result: dict, output_stem: str) -> list[str]:
    """One larger, standalone figure per panel -- easier to read in isolation,
    especially the price axis's comma-formatted full numbers and the steering
    panel's per-decision value labels, both cramped in the combined overview above.
    Each file stands alone (own title, own summary box) since a reader may only
    open one of the three."""
    s = _extract_series(l2_result)
    title = _episode_title(l2_result)
    panels = [
        ("price", _draw_price_panel, (12, 6.5)),
        ("execution", _draw_execution_panel, (12, 5.5)),
        ("steering", _draw_steering_panel, (12, 5.5)),
    ]
    paths = []
    for name, draw_fn, figsize in panels:
        fig, ax = plt.subplots(figsize=figsize)
        fig.suptitle(title, fontsize=12, fontweight="bold")
        if draw_fn is _draw_price_panel:
            draw_fn(ax, l2_result, s)
        else:
            draw_fn(ax, s)
        ax.set_xlabel("ticks since episode start")
        _add_summary_box(fig, l2_result, twap_result, fontsize=9)
        fig.tight_layout(rect=[0, 0.15, 1, 0.92])
        path = f"{output_stem}_{name}.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(path)
    return paths


# L3's own n=500 action-type distribution (docs/TRACK_STATUS.md / L3's evaluation
# report) -- the reference a --frozen-l3-only replay's printed breakdown should land
# close to. One episode vs. n=500: not an exact-match bar, a shape check.
_L3_N500_ACTION_DISTRIBUTION = {"HOLD": 0.476, "LIMIT": 0.520, "MARKET": 0.0002, "REPLACE": 0.0036}


def _action_type_distribution(tick_records: list[dict]) -> dict[str, float]:
    from src.envs.lob_execution_env import ORDER_TYPE_HOLD
    counts = {"HOLD": 0, "LIMIT": 0, "MARKET": 0, "REPLACE": 0}
    label_by_type = {
        ORDER_TYPE_HOLD: "HOLD", ORDER_TYPE_LIMIT: "LIMIT",
        ORDER_TYPE_MARKET: "MARKET", ORDER_TYPE_CANCEL_REPLACE: "REPLACE",
    }
    for rec in tick_records:
        counts[label_by_type[rec["order_type"]]] += 1
    n = len(tick_records)
    return {k: v / n for k, v in counts.items()} if n else counts


def main() -> None:
    args = build_parser().parse_args()
    if args.frozen_l3_only:
        if args.l2_checkpoint or args.l2_vecnormalize:
            raise SystemExit("--frozen-l3-only and --l2-checkpoint/--l2-vecnormalize are mutually exclusive.")
    else:
        if not args.l2_checkpoint or not args.l2_vecnormalize:
            raise SystemExit("--l2-checkpoint and --l2-vecnormalize are both required unless --frozen-l3-only.")
    if args.data_dir is None:
        args.data_dir = NUMERIC_DATA_DIR if args.use_numeric_format else PARQUET_DATA_DIR

    output_path = args.output or f"replay_seed{args.seed}.png"
    output_stem = output_path[:-4] if output_path.lower().endswith(".png") else output_path

    val_dates = load_split("val")
    val_date_range = (val_dates[0].isoformat(), val_dates[-1].isoformat())
    print(f"val date_range: {val_date_range} ({len(val_dates)} real days)")
    print(f"data_dir: {args.data_dir}")
    print(f"seed={args.seed}")
    print(f"mode: {'frozen L3 only (TWAP-passthrough action)' if args.frozen_l3_only else 'L2 policy'}")

    l2_result = run_l2_episode(args, val_date_range)
    l2_result["seed"] = args.seed
    l2_result["frozen_l3_only"] = args.frozen_l3_only
    twap_result = run_twap_baseline(args, val_date_range)

    is_r = l2_result["is_result"]
    twap_is = twap_result["is_result"]
    arm_label = "Frozen L3 (unsteered)" if args.frozen_l3_only else "L2 policy"
    print(f"\n{arm_label}: fill_ratio={is_r.fill_ratio:.3f}  IS_total_bps={is_r.is_total_bps:+.4f}")
    print(f"TWAP (same window/seed): fill_ratio={twap_is.fill_ratio:.3f}  IS_total_bps={twap_is.is_total_bps:+.4f}")
    print(f"Child orders reconstructed: {len(l2_result['child_orders'])} "
          f"({sum(1 for o in l2_result['child_orders'] if o.kind == 'resting')} resting, "
          f"{sum(1 for o in l2_result['child_orders'] if o.kind == 'market')} market)")
    print(f"L2 decisions: {len(l2_result['l2_decision_records'])}")

    dist = _action_type_distribution(l2_result["tick_records"])
    print(f"\nAction-type distribution, this episode ({len(l2_result['tick_records'])} ticks):")
    for label in ("HOLD", "LIMIT", "MARKET", "REPLACE"):
        ref = _L3_N500_ACTION_DISTRIBUTION[label]
        print(f"  {label:8s} {dist.get(label, 0.0):6.1%}   (L3's own n=500: {ref:6.1%})")
    if args.frozen_l3_only:
        print("  ^ sanity check: this episode's shape should land reasonably close to the")
        print("    n=500 reference above -- a big divergence (e.g. mostly MARKET here) means")
        print("    this visualizer has a bug, not that frozen L3's behavior changed.")

    build_figure(l2_result, twap_result, output_path)
    print(f"\nWrote {output_path} (combined overview)")
    for p in build_separate_figures(l2_result, twap_result, output_stem):
        print(f"Wrote {p} (single-panel detail)")


if __name__ == "__main__":
    main()
