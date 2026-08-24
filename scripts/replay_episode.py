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

Run (synthetic data, mechanics only):
  PYTHONPATH=. .venv/bin/python scripts/replay_episode.py --help
Run (real, only after training completes):
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
    parser.add_argument("--l2-checkpoint", type=str, required=True)
    parser.add_argument("--l2-vecnormalize", type=str, required=True)
    parser.add_argument("--l3-checkpoint", type=str, required=True)
    parser.add_argument("--l3-vecnormalize", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True, help="Episode seed -- no default, pick one deliberately.")
    parser.add_argument("--ticks-per-l2-decision", type=int, default=50)
    parser.add_argument("--l2-include-prev-action", action="store_true")
    parser.add_argument("--n-slices", type=int, default=10, help="TWAP comparison baseline's own slice count.")
    parser.add_argument("--data-dir", type=str, default="data/raw_l2_bybit/BTCUSDT")
    parser.add_argument("--use-numeric-format", action="store_true")
    parser.add_argument("--device", default="cpu", choices=["cuda", "cpu"])
    parser.add_argument("--output", type=str, default=None, help="PNG path; default replay_seed<seed>.png")
    return parser


# ---------------------------------------------------------------------------
# Capture: monkeypatch instrumentation, inert unless this script installs it.
# ---------------------------------------------------------------------------

@dataclass
class EpisodeCapture:
    tick_records: list[dict[str, Any]] = field(default_factory=list)
    l2_decision_records: list[dict[str, Any]] = field(default_factory=list)


def install_capture(wrapped_env: FrozenL3Wrapper) -> EpisodeCapture:
    capture = EpisodeCapture()
    base_env: LOBExecutionEnv = wrapped_env.env

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
        })
        return obs, r, term, trunc, info

    base_env.step = instrumented_env_step

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


def reconstruct_child_orders(tick_records: list[dict], side: int) -> list[ChildOrder]:
    """Walks the tick-level capture and regroups it into discrete child-order
    lifetimes (placement -> outcome), matching _place_limit()'s own real pricing
    (tick.best_bid/ask +/- offset*TICK_SIZE, verified against that function's source,
    not guessed). One case handled explicitly: _place_limit() routes a price that
    crosses the opposing side through walk_market_fill() (same mechanism as an
    ORDER_TYPE_MARKET action), so a LIMIT/CANCEL_AND_REPLACE placement can show a
    non-maker fill on its OWN placement tick -- that is an immediate crossing fill,
    not a later maker fill, and is attributed to the order accordingly rather than
    left looking "still open." Not handled: a crossing placement that fills only
    PART of the order and rests the remainder on the same tick -- not observed in
    the one episode this was verified against, flagged as a known limitation rather
    than silently assumed away."""
    orders: list[ChildOrder] = []
    open_order: ChildOrder | None = None

    for rec in tick_records:
        if rec["order_type"] in (ORDER_TYPE_LIMIT, ORDER_TYPE_CANCEL_REPLACE):
            if open_order is not None and open_order.outcome == "open_at_episode_end":
                open_order.outcome = "replaced"
            touch = rec["best_bid"] if side == 1 else rec["best_ask"]
            price = round(touch + rec["offset"] * TICK_SIZE, 1) if side == 1 else round(touch - rec["offset"] * TICK_SIZE, 1)
            open_order = ChildOrder(
                kind="resting", placement_tick=rec["tick_idx"], placement_price=price,
                offset_from_touch=rec["offset"], outcome="open_at_episode_end",
            )
            orders.append(open_order)
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
            continue

        if rec["order_type"] == ORDER_TYPE_MARKET:
            for f in rec["fills"]:
                orders.append(ChildOrder(
                    kind="market", placement_tick=rec["tick_idx"], placement_price=f["price"],
                    offset_from_touch=0, outcome="filled",
                    fill_ticks=[rec["tick_idx"]], fill_qtys=[f["qty"]], fill_prices=[f["price"]],
                ))
            continue

        if open_order is not None:
            for f in rec["fills"]:
                if f.get("is_maker"):
                    open_order.fill_ticks.append(rec["tick_idx"])
                    open_order.fill_qtys.append(f["qty"])
                    open_order.fill_prices.append(f["price"])
                    open_order.outcome = "filled"

    return orders


# ---------------------------------------------------------------------------
# Run the two episodes.
# ---------------------------------------------------------------------------

def run_l2_episode(args, val_date_range: tuple[str, str]) -> dict[str, Any]:
    l3_model = RecurrentPPO.load(args.l3_checkpoint, device="cpu")
    wrapped_env = make_l2_wrapped_env(
        val_date_range, HORIZON_TICKS, LOOKBACK_TICKS, l3_model, args.l3_vecnormalize,
        args.ticks_per_l2_decision, args.l2_include_prev_action,
        data_dir=args.data_dir, l3_deterministic=True, use_numeric_format=args.use_numeric_format,
    )
    capture = install_capture(wrapped_env)

    l2_model = SAC.load(args.l2_checkpoint, device=args.device)
    l2_vec_normalize = VecNormalize.load(args.l2_vecnormalize, DummyVecEnv([lambda: wrapped_env]))
    l2_vec_normalize.training = False

    obs, info = wrapped_env.reset(seed=args.seed)
    max_decisions = HORIZON_TICKS // args.ticks_per_l2_decision + 1
    for _ in range(max_decisions):
        obs_for_policy = l2_vec_normalize.normalize_obs(obs[None, :])
        action, _ = l2_model.predict(obs_for_policy, deterministic=True)
        obs, r, term, trunc, info = wrapped_env.step(action[0])
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


# ---------------------------------------------------------------------------
# Figure.
# ---------------------------------------------------------------------------

def build_figure(l2_result: dict, twap_result: dict, output_path: str) -> None:
    tick_records = l2_result["tick_records"]
    l2_decisions = l2_result["l2_decision_records"]
    child_orders = l2_result["child_orders"]
    side = l2_result["side"]
    qty_total = l2_result["qty_total"]
    arrival_price = l2_result["arrival_price"]
    is_result = l2_result["is_result"]
    twap_is = twap_result["is_result"]

    ticks = [r["tick_idx"] for r in tick_records]
    mids = [r["mid_price"] for r in tick_records]
    bids = [r["best_bid"] for r in tick_records]
    asks = [r["best_ask"] for r in tick_records]
    t0 = ticks[0] if ticks else 0
    rel_ticks = [t - t0 for t in ticks]

    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True, height_ratios=[2.2, 1.3, 1.0])
    fig.suptitle(
        f"L2 episode replay -- seed={l2_result.get('seed', '?')} -- {_side_label(side)} "
        f"{qty_total:.3f} units, arrival price {arrival_price:.1f}",
        fontsize=13, fontweight="bold",
    )

    # --- Panel 1: price path + child orders ---
    ax = axes[0]
    ax.plot(rel_ticks, mids, color="#333333", lw=1.1, label="mid price")
    ax.fill_between(rel_ticks, bids, asks, color="#cccccc", alpha=0.4, label="bid/ask spread")
    ax.axhline(arrival_price, color="#1f77b4", ls="--", lw=1.2, label=f"arrival price ({arrival_price:.1f})")
    if mids:
        ax.scatter([rel_ticks[-1]], [mids[-1]], color="#d62728", zorder=5, s=40,
                   label=f"terminal mid ({mids[-1]:.1f})")

    outcome_style = {
        "filled": dict(marker="^" if side == 1 else "v", color="#2ca02c", label="resting order -- filled"),
        "replaced": dict(marker="x", color="#ff7f0e", label="resting order -- replaced before filling"),
        "open_at_episode_end": dict(marker="o", color="#7f7f7f", label="resting order -- still open at end"),
    }
    seen_labels = set()
    for order in child_orders:
        if order.kind == "resting":
            style = outcome_style[order.outcome]
            lbl = style["label"] if style["label"] not in seen_labels else None
            seen_labels.add(style["label"])
            ax.scatter([order.placement_tick - t0], [order.placement_price], marker=style["marker"],
                       color=style["color"], s=70, zorder=6, label=lbl, edgecolors="black", linewidths=0.5)
        else:
            lbl = "market/crossing fill" if "market/crossing fill" not in seen_labels else None
            seen_labels.add("market/crossing fill")
            ax.scatter([order.placement_tick - t0], [order.placement_price], marker="*",
                       color="#9467bd", s=110, zorder=7, label=lbl, edgecolors="black", linewidths=0.5)

    ax.set_ylabel("price")
    ax.set_title("Price path and child order placements", fontsize=10)
    ax.legend(loc="upper left", fontsize=7, ncol=2)

    # --- Panel 2: inventory vs TWAP schedule ---
    ax = axes[1]
    qty_remaining_series = [r["qty_remaining"] for r in tick_records]
    executed_frac = [1.0 - qr / qty_total for qr in qty_remaining_series]
    schedule_frac = [min(1.0, rt / HORIZON_TICKS) for rt in rel_ticks]
    ax.plot(rel_ticks, schedule_frac, color="#1f77b4", ls="--", lw=1.3, label="linear TWAP schedule")
    ax.plot(rel_ticks, executed_frac, color="#2ca02c", lw=1.6, label="actual fraction executed")
    ax.fill_between(rel_ticks, executed_frac, schedule_frac,
                     where=[e >= s for e, s in zip(executed_frac, schedule_frac)],
                     color="#2ca02c", alpha=0.15, interpolate=True, label="ahead of schedule")
    ax.fill_between(rel_ticks, executed_frac, schedule_frac,
                     where=[e < s for e, s in zip(executed_frac, schedule_frac)],
                     color="#d62728", alpha=0.15, interpolate=True, label="behind schedule")
    ax.set_ylabel("fraction of order filled")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("Execution progress vs. on-schedule TWAP pace", fontsize=10)
    ax.legend(loc="upper left", fontsize=7)

    # --- Panel 3: L2's own steering decisions ---
    ax = axes[2]
    dec_ticks = [d["tick_idx"] - t0 for d in l2_decisions]
    part_mult = [d["participation_mult"] for d in l2_decisions]
    urgency = [d["urgency"] for d in l2_decisions]
    ax.step(dec_ticks, part_mult, where="post", color="#1f77b4", lw=1.4, label="participation-rate multiplier")
    ax.step(dec_ticks, urgency, where="post", color="#e377c2", lw=1.4, label="urgency")
    ax.axhline(1.0, color="#1f77b4", ls=":", lw=0.8, alpha=0.6)
    ax.axhline(0.5, color="#e377c2", ls=":", lw=0.8, alpha=0.6)
    ax.set_ylabel("L2 action value")
    ax.set_xlabel("ticks since episode start")
    ax.set_title("L2's steering: participation-rate multiplier (1.0=on-schedule) and urgency (0.5=neutral)", fontsize=10)
    ax.legend(loc="upper left", fontsize=7)

    # --- Summary text box --- (short explicit lines, none wide enough to overflow
    # the figure at this fontsize -- a single long f-string wrapped only at spaces
    # was found to run off the right edge of the saved PNG, fixed by splitting
    # deliberately rather than relying on matplotlib's own wrapping.)
    exec_str = f"{is_result.is_exec_bps:+.2f}bps" if is_result.is_exec_bps is not None else "n/a (no fills)"
    verdict = "BEAT" if is_result.is_total_bps < twap_is.is_total_bps else "LOST TO"
    verdict_gap = abs(is_result.is_total_bps - twap_is.is_total_bps)
    summary_lines = [
        f"L2 policy:  fill_ratio={is_result.fill_ratio:.1%}   IS_total={is_result.is_total_bps:+.2f}bps",
        f"  (execution={exec_str}, opportunity={is_result.is_opp_bps:+.2f}bps, fees={is_result.fees_bps:+.2f}bps)",
        f"TWAP, same window/seed:  fill_ratio={twap_is.fill_ratio:.1%}   IS_total={twap_is.is_total_bps:+.2f}bps",
        f"L2 {verdict} TWAP by {verdict_gap:.2f}bps on this single episode "
        f"(not a significance test -- see scripts/eval_l2_n500.py).",
    ]
    fig.text(0.5, 0.01, "\n".join(summary_lines), ha="center", va="bottom", fontsize=8, family="monospace",
              bbox=dict(boxstyle="round", facecolor="#f0f0f0", edgecolor="#999999"))

    fig.tight_layout(rect=[0, 0.11, 1, 0.97])
    fig.savefig(output_path, dpi=130)
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    output_path = args.output or f"replay_seed{args.seed}.png"

    val_dates = load_split("val")
    val_date_range = (val_dates[0].isoformat(), val_dates[-1].isoformat())
    print(f"val date_range: {val_date_range} ({len(val_dates)} real days)")
    print(f"seed={args.seed}")

    l2_result = run_l2_episode(args, val_date_range)
    l2_result["seed"] = args.seed
    twap_result = run_twap_baseline(args, val_date_range)

    is_r = l2_result["is_result"]
    twap_is = twap_result["is_result"]
    print(f"\nL2 policy:    fill_ratio={is_r.fill_ratio:.3f}  IS_total_bps={is_r.is_total_bps:+.4f}")
    print(f"TWAP (same window/seed): fill_ratio={twap_is.fill_ratio:.3f}  IS_total_bps={twap_is.is_total_bps:+.4f}")
    print(f"Child orders reconstructed: {len(l2_result['child_orders'])} "
          f"({sum(1 for o in l2_result['child_orders'] if o.kind == 'resting')} resting, "
          f"{sum(1 for o in l2_result['child_orders'] if o.kind == 'market')} market)")
    print(f"L2 decisions: {len(l2_result['l2_decision_records'])}")

    build_figure(l2_result, twap_result, output_path)
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
