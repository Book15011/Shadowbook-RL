"""Queue-position-aware fill simulation (architecture_spec.md Section 2.4).

Key modeling notes -- flagged explicitly rather than silently assumed:

1. q_p(t) in the cancel-proportional term is read as the FULL resting size at
   that price level (everyone there, ahead of and behind/including our own
   order) -- matches the "uniform-random cancel position across the level"
   framing in the spec.

2. Section 2.4 specifies the Q_ahead ESTIMATOR formula but not how actual
   fills against your own resting order are determined once Q_ahead reaches
   zero. update_queue() extends it: cancels deplete Q_ahead without ever
   generating a fill (canceled volume never traded through anyone), trade
   volume depletes remaining Q_ahead first, and any leftover trade volume
   (once Q_ahead hits zero) fills your own order up to its remaining size.
   This sequential cancel-then-trade decomposition produces an IDENTICAL
   Q_ahead(t+1) to the spec's single combined formula
   max(0, Q_ahead(t) - V_trade(t) - V_cancel(t)*Q_ahead(t)/q_p(t))
   -- it is a strict extension for fill bookkeeping, not a deviation. See
   tests/test_matching_engine.py for the hand-worked proof-by-fixture.

3. expected_wait_time() returns math.inf when the trailing average trade
   rate is zero (no trading at that level in the window) -- an infinite
   expected wait is the mathematically correct answer for "nothing is
   trading through this level", not an error condition.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class QueueState:
    """State of a single resting limit order's position in the price-time
    priority queue at its price level."""

    q_ahead: float
    own_qty_remaining: float
    filled_qty: float = 0.0

    @property
    def is_resolved(self) -> bool:
        """True once the order is either fully filled or has nothing left
        to wait through (both q_ahead and own remaining qty are zero)."""
        return self.own_qty_remaining <= 0.0


def queue_position_ratio(state: QueueState) -> float:
    """architecture_spec.md Section 3.1, obs index 13: -1 = no resting
    order; else Q_ahead / (Q_ahead + own_qty)."""
    if state.own_qty_remaining <= 0.0 and state.q_ahead <= 0.0:
        return -1.0
    denom = state.q_ahead + state.own_qty_remaining
    if denom <= 0.0:
        return -1.0
    return state.q_ahead / denom


def update_queue(state: QueueState, *, v_trade: float, v_cancel: float, q_p_before: float) -> QueueState:
    """One discrete tick update per architecture_spec.md Section 2.4, plus
    the fill-determination extension documented in the module docstring.

    v_trade: trade volume executed at this price level this tick.
    v_cancel: canceled resting volume at this price level this tick.
    q_p_before: total resting size at this price level immediately before
        this tick's trade/cancel activity (includes our own order).
    """
    if state.own_qty_remaining <= 0.0:
        return state  # already fully filled, nothing left to track

    q_ahead = max(0.0, state.q_ahead)
    v_trade = max(0.0, v_trade)
    v_cancel = max(0.0, v_cancel)

    cancel_effective = v_cancel * (q_ahead / q_p_before) if q_p_before > 0 else 0.0

    # Cancels first: canceled resting volume leaves the book without ever
    # trading through anyone, so it reduces q_ahead but never fills us.
    q_after_cancel = max(0.0, q_ahead - cancel_effective)

    # Trade volume next: depletes whatever queue remains ahead of us; any
    # leftover, once the queue ahead is exhausted, reaches our own order.
    consumed_from_queue = min(q_after_cancel, v_trade)
    new_q_ahead = q_after_cancel - consumed_from_queue
    leftover_trade = v_trade - consumed_from_queue

    fill_qty = min(leftover_trade, state.own_qty_remaining)
    new_own_remaining = state.own_qty_remaining - fill_qty
    new_filled = state.filled_qty + fill_qty

    return QueueState(q_ahead=new_q_ahead, own_qty_remaining=new_own_remaining, filled_qty=new_filled)


def walk_market_fill(
    qty: float, prices: Sequence[float], sizes: Sequence[float]
) -> tuple[list[tuple[float, float]], float]:
    """Level-by-level market-order fill against visible resting depth
    (architecture_spec.md Section 2.3's top-N=20 levels, best-to-worst order --
    matches the ordering TickView already stores, see lob_execution_env.py).

    Consumes qty starting at prices[0] (the touch) and walking outward one
    level at a time until qty is fully consumed or the visible levels run
    out. Never fills beyond what prices/sizes actually show: if the visible
    book doesn't have enough depth, the unconsumed remainder is returned as
    qty_unfilled rather than invented at a synthetic price -- callers are
    responsible for leaving that remainder in the caller's own qty_remaining
    so it can be picked up on a later tick (or fall through to the terminal
    opportunity-cost IS component if the episode ends first).

    Returns (fills, qty_unfilled): fills is a list of (price, qty) tuples,
    one entry per level touched (a partial fill at the last level touched
    gets its own entry with just the partial qty, not the level's full size).
    """
    remaining = max(0.0, qty)
    fills: list[tuple[float, float]] = []
    for price, size in zip(prices, sizes):
        if remaining <= 0.0:
            break
        size = max(0.0, float(size))
        if size <= 0.0:
            continue
        take = min(remaining, size)
        fills.append((float(price), take))
        remaining -= take
    return fills, remaining


def expected_wait_time(q_ahead: float, avg_trade_rate: float) -> float:
    """architecture_spec.md Section 2.4: T_hat_fill = Q_ahead(t) / avg_trade_rate.

    Returns math.inf when avg_trade_rate <= 0 (no trading at this level in
    the trailing window) -- an infinite expected wait is the correct answer,
    not an error.
    """
    if avg_trade_rate <= 0:
        return math.inf
    return max(0.0, q_ahead) / avg_trade_rate
