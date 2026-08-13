"""Reward function (architecture_spec.md Section 3.3) and Implementation
Shortfall decomposition (Section 5.1), adapted from the spec's reference
code rather than redesigned from scratch.

Flagged edge case: when fill_ratio == 0 (no fills at all -- e.g. a no-op
policy), IS_exec_bps is mathematically undefined (0/0 average fill price).
The total formula is fill_ratio * IS_exec_bps + IS_opp_bps + fees_bps; with
fill_ratio == 0 the first term must be treated as exactly 0.0 by
construction, NOT computed as 0 * nan (which is nan in IEEE float
arithmetic, not 0). compute_implementation_shortfall() special-cases this
explicitly rather than silently producing NaN.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class RewardWeights:
    alpha: float = 1.0     # slippage
    lam: float = 0.02      # inventory holding
    beta: float = 0.5      # unfilled cancel penalty (flat)
    gamma: float = 0.3     # unfilled cancel penalty (queue-position-weighted)
    delta: float = 0.8     # spread capture bonus
    kappa: float = 1.0     # terminal IS


def step_reward(
    w: RewardWeights,
    *,
    side: int,
    fills: list[dict],
    arrival_price: float,
    mid_price: float,
    qty_remaining: float,
    qty_total: float,
    dt: float,
    l1_risk_score: float,
    canceled_unfilled: bool,
    queue_ahead_at_cancel: float | None,
    queue_at_level: float | None,
) -> float:
    """Section 3.3, components 1-4 (per-step; the terminal IS component is
    computed separately via compute_implementation_shortfall() and added
    once at episode end -- see LOBExecutionEnv.step())."""
    r_slip = 0.0
    r_spread = 0.0
    for f in fills:
        r_slip += -w.alpha * side * (f["price"] - arrival_price) / arrival_price * (f["qty"] / qty_total) * 1e4
        if f.get("is_maker"):
            r_spread += w.delta * side * (mid_price - f["price"]) / mid_price * (f["qty"] / qty_total) * 1e4

    r_inv = -w.lam * (1 + max(0.0, l1_risk_score)) * (qty_remaining / qty_total) ** 2 * dt

    r_queue = 0.0
    if canceled_unfilled:
        r_queue -= w.beta
        if queue_ahead_at_cancel is not None and queue_at_level:
            r_queue -= w.gamma * (queue_ahead_at_cancel / queue_at_level)

    return r_slip + r_inv + r_queue + r_spread


@dataclass
class ImplementationShortfall:
    """Section 5.1 Perold decomposition, in basis points."""

    fill_ratio: float
    is_exec_bps: float | None  # None when fill_ratio == 0 (undefined, no fills to average)
    is_opp_bps: float
    fees_bps: float
    is_total_bps: float
    p_avg: float | None  # None when fill_ratio == 0


def compute_implementation_shortfall(
    *,
    side: int,
    fills: list[dict],
    qty_total: float,
    arrival_price: float,
    terminal_mid_price: float,
    fee_bps_per_fill: float = 0.0,
) -> ImplementationShortfall:
    """architecture_spec.md Section 5.1.

    fills: list of {"price": float, "qty": float} realized fills.
    fee_bps_per_fill: flat fee in bps charged on every filled unit of qty_total
        (kept simple and explicit rather than modeling maker/taker fee tiers,
        which Section 5.1 doesn't specify -- flagged simplification).
    """
    filled_qty = sum(f["qty"] for f in fills)
    fill_ratio = filled_qty / qty_total if qty_total > 0 else 0.0

    if filled_qty > 0:
        p_avg = sum(f["price"] * f["qty"] for f in fills) / filled_qty
        is_exec_bps = side * (p_avg - arrival_price) / arrival_price * 1e4
        exec_contribution = fill_ratio * is_exec_bps
    else:
        p_avg = None
        is_exec_bps = None
        exec_contribution = 0.0  # fill_ratio * undefined IS_exec, by construction -- see module docstring

    is_opp_bps = (1.0 - fill_ratio) * side * (terminal_mid_price - arrival_price) / arrival_price * 1e4
    fees_bps = fee_bps_per_fill * fill_ratio

    is_total_bps = exec_contribution + is_opp_bps + fees_bps

    return ImplementationShortfall(
        fill_ratio=fill_ratio,
        is_exec_bps=is_exec_bps,
        is_opp_bps=is_opp_bps,
        fees_bps=fees_bps,
        is_total_bps=is_total_bps,
        p_avg=p_avg,
    )
