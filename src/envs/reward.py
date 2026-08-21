"""Reward function (architecture_spec.md Section 3.3) and Implementation
Shortfall decomposition (Section 5.1), adapted from the spec reference
code rather than redesigned from scratch.

Flagged edge case: when fill_ratio == 0 (no fills at all -- e.g. a no-op
policy), IS_exec_bps is mathematically undefined (0/0 average fill price).
The total formula is fill_ratio * IS_exec_bps + IS_opp_bps + fees_bps; with
fill_ratio == 0 the first term must be treated as exactly 0.0 by
construction, NOT computed as 0 * nan (which is nan in IEEE float
arithmetic, not 0). compute_implementation_shortfall() special-cases this
explicitly rather than silently producing NaN.

EXPERIMENTAL ADDITION (not in Section 3.3): r_stale, added after the
finished 20M-step baseline (models/l3_executioner_v1.zip) was found to use
CANCEL_AND_REPLACE/MARKET 0% of the time -- see
docs/reports/phase3_l3_baseline_milestone.md. The original four components
only penalize an ACTIVE cancel (r_queue, via the cancel flags); leaving a
stale, unfilled order in place costs almost nothing per tick beyond r_inv,
so a single correction attempt is always a worse certain cost than doing
nothing (break-even was about 400 idle ticks under r_inv alone -- see the
report). r_stale scales with ticks_since_own_fill_norm (obs idx 14,
already computed by the env, not a new feature) while an order is resting
and unfilled, so accumulated staleness outpaces a correction one-time cost
well before that break-even point. See RewardWeights.zeta for the
coefficient derivation.

EXPERIMENTAL ADDITION 3 (modifies Section 3.3): r_queue is no longer
charged identically for MARKET and CANCEL_AND_REPLACE. Motivation,
verified from source rather than hypothesized: LOBExecutionEnv.step()
previously set a single canceled_unfilled flag for both actions, so both
paid -beta - gamma*queue_ratio. Since only MARKET guarantees a fill,
MARKET strictly economically dominated CANCEL_AND_REPLACE at every
coefficient -- the structural reason CANCEL_AND_REPLACE stayed at exactly
0% usage across every zeta AND every eta_replace probed (see
docs/reports/phase3_l3_baseline_milestone.md). No coefficient search can
rescue a dominated action; the price itself had to change. The split:
MARKET keeps both components (walking the spread on top of discarding
queue position is full abandonment), CANCEL_AND_REPLACE keeps only the
queue-weighted -gamma*queue_ratio (it stays in the limit-order system with
a fresh order, so the flat abandonment charge does not apply, but
discarded queue position is a real cost).

IMPORTANT, and contrary to the intuition this split was designed around:
queue_ratio = q_ahead / (q_ahead + own_qty_remaining) is LARGEST for a
freshly placed order sitting behind a deep level, and tends to 0 for an
order that has already waited its way to the front. _place_limit()
initializes q_ahead to the full visible resting volume at the price, and
update_queue() only ever decreases it. So this term does NOT "still sting
for an order that has already accumulated priority" -- it stings for the
opposite case, and it is exactly 0.0 for a price outside the visible
top-20 book (where _place_limit() sets q_ahead = 0.0). Any future
placement-staleness reward term (rewarding CANCEL_AND_REPLACE for
correcting a stale order) needs to account for this: a zero-cost regime
at prices outside the visible book means such a term could be gamed by
spam-replacing there for free -- this is not benign, work it out
explicitly before adding one.

EXPERIMENTAL ADDITION 5 (variance reduction, NOT an objective change --
this distinction matters and is stated explicitly so it is not later
mistaken for one): IS_total_bps variance across episodes is dominated by
market drift (observed std ~4-5bps across every eval this project has run)
while the agent's own execution-quality contribution is fractions of a
basis point. The critic cannot learn the drift component away -- the agent
has no way to observe FUTURE price drift over its own episode window, so
the value function is structurally blind to the single largest source of
variance in the terminal reward signal it is trying to predict. TWAP's own
terminal IS_total_bps over that SAME window is a natural proxy for "how
much of this episode's outcome was just the market moving," because TWAP
has no timing or venue-selection skill of its own (it slices blindly on a
fixed schedule) -- what's left in (agent_IS - TWAP_IS) is closer to the
agent's own execution-quality contribution than agent_IS alone is.

Critically: TWAP's shadow run for a given episode NEVER observes or reacts
to what the real policy does that episode -- it is a fixed function of the
market window alone (same day, same start tick, same side, same
qty_total), computed once in reset() before the real episode's own step()
calls begin (see LOBExecutionEnv._compute_twap_shadow_terminal_is()). That
makes it a per-episode CONSTANT with respect to the policy's actions,
which is what makes this a baseline subtraction rather than an objective
change: subtracting a constant from the reward at every point in the
policy's action space shifts the SCALE of the return but not the ARGMAX --
it cannot change which policy is optimal, and it creates no new incentive
to "beat" TWAP or anything else. It only changes how much of the reward
signal's variance is attributable to something the critic could never have
learned to predict in the first place. Gated behind
RewardWeights.subtract_twap_baseline, default False (inert) -- same
opt-in convention as zeta above, so nothing changes until deliberately
tested. See that field's docstring for the mechanism and docs/reports/ for
the measured per-reset cost of computing it.
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
    # EXPERIMENTAL (Part B, docs/reports/phase3_l3_baseline_milestone.md):
    # staleness penalty on ticks_since_own_fill_norm while resting and
    # unfilled. Derivation: r_inv alone gives a flat -lam*dt = -0.002/tick
    # idle-holding cost (near-full inventory, l1_risk_score=0); a single
    # CANCEL_AND_REPLACE costs up to r_queue = -beta - gamma = -0.8; the
    # r_inv-only break-even is about 400 idle ticks (40s). zeta is picked so
    # cumulative(r_inv + r_stale) reaches -0.8 by about 200 ticks instead:
    # sum_{k=1..200} zeta*(k/3000) = zeta*200^2/6000 = zeta*6.667 needs to
    # supply the remaining -0.4 (r_inv alone supplies -0.4 of the -0.8 over
    # the same 200 ticks) -> zeta = 0.4/6.667 = 0.06. At full-horizon
    # staleness (norm=1.0) the per-tick term is -0.06, well under a typical
    # single fill r_slip magnitude (order 1-2 for a couple bps on a full
    # fill) -- see module docstring.
    zeta: float = 0.06
    # EXPERIMENTAL 5 (see module docstring): variance reduction, NOT an
    # objective change -- read the module docstring's EXPERIMENTAL 5 section
    # before touching this. Defaults to False -- inert unless explicitly
    # overridden, same opt-in convention as zeta above. When True,
    # LOBExecutionEnv subtracts a TWAP shadow-execution's terminal
    # IS_total_bps (computed once per episode in reset(), over the identical
    # window -- see LOBExecutionEnv._compute_twap_shadow_terminal_is()) from
    # the terminal reward's kappa*IS_total_bps term. Because that subtracted
    # quantity does not depend on the agent's actions at all (TWAP's shadow
    # run never sees or reacts to what the real episode's policy does), it
    # is a per-episode CONSTANT with respect to the policy -- it cannot
    # change which policy is optimal, only reduce the variance of the signal
    # the critic has to learn from. This field does not affect
    # info["implementation_shortfall"] anywhere -- that always reports the
    # real, un-adjusted execution outcome; only the scalar reward changes.
    subtract_twap_baseline: bool = False


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
    canceled_via_market: bool,
    canceled_via_replace: bool,
    queue_ahead_at_cancel: float | None,
    queue_at_level: float | None,
    resting: bool = False,
    ticks_since_own_fill_norm: float = 0.0,
) -> float:
    """Section 3.3, components 1-4 (per-step; the terminal IS component is
    computed separately via compute_implementation_shortfall() and added
    once at episode end -- see LOBExecutionEnv.step()). Component 5
    (r_stale) is an experimental addition, not in Section 3.3 -- see
    module docstring.

    r_queue is priced differently for the two cancel paths (EXPERIMENTAL 3,
    module docstring). canceled_via_market and canceled_via_replace are
    mutually exclusive by construction in LOBExecutionEnv.step(); the flat
    -beta abandonment charge applies only to the MARKET path."""
    r_slip = 0.0
    r_spread = 0.0
    for f in fills:
        r_slip += -w.alpha * side * (f["price"] - arrival_price) / arrival_price * (f["qty"] / qty_total) * 1e4
        if f.get("is_maker"):
            r_spread += w.delta * side * (mid_price - f["price"]) / mid_price * (f["qty"] / qty_total) * 1e4

    r_inv = -w.lam * (1 + max(0.0, l1_risk_score)) * (qty_remaining / qty_total) ** 2 * dt

    # EXPERIMENTAL 3 (module docstring): the two cancel paths are priced
    # differently, because charging them identically made MARKET strictly
    # dominate CANCEL_AND_REPLACE (only MARKET guarantees a fill) and drove
    # CANCEL_AND_REPLACE usage to exactly 0% at every coefficient tested.
    r_queue = 0.0
    if canceled_via_market:
        # Full abandonment: walks the spread AND discards queue position.
        # Charged both components, exactly as in Section 3.3.
        r_queue -= w.beta
        if queue_ahead_at_cancel is not None and queue_at_level:
            r_queue -= w.gamma * (queue_ahead_at_cancel / queue_at_level)
    elif canceled_via_replace:
        # Not abandonment -- a fresh limit order replaces the old one, so the
        # flat -beta is dropped and REPLACE stops being dominated by MARKET.
        # The queue-weighted charge stays: discarded queue position is real.
        #
        # NOTE this charge does NOT behave the way the split was originally
        # motivated. queue_ahead_at_cancel / queue_at_level is LARGEST for a
        # fresh order behind a deep level and tends to 0 for an order that
        # has already waited its way to the front -- so it does not "still
        # sting for an already-waited order", it stings for the opposite
        # case, and it is exactly 0.0 when the price sits outside the visible
        # top-20 book (_place_limit() initializes q_ahead = 0.0 there) -- a
        # future placement-staleness reward term would need to account for
        # this zero-cost regime explicitly (see the module docstring).
        if queue_ahead_at_cancel is not None and queue_at_level:
            r_queue -= w.gamma * (queue_ahead_at_cancel / queue_at_level)

    r_stale = -w.zeta * ticks_since_own_fill_norm if resting else 0.0

    return r_slip + r_inv + r_queue + r_spread + r_stale


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
        which Section 5.1 does not specify -- flagged simplification).
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
