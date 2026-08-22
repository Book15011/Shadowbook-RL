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

EXPERIMENTAL ADDITION 2 (not in Section 3.3): r_placement_stale, added
after the coefficient sweep confirmed r_stale is structurally incapable of
ever rewarding CANCEL_AND_REPLACE at any zeta -- confirmed directly from
source, not hypothesized: _ticks_since_own_fill_norm() (and therefore
r_stale) depends solely on self._last_fill_tick_idx, which is set ONLY
inside `if step_fills:` in LOBExecutionEnv.step(). A CANCEL_AND_REPLACE
never touches it, so r_stale cannot distinguish "sitting on one stale
order for 1000 ticks" from "replaced 5 times in the last 10 ticks, still
unfilled" -- both look identical to r_stale. r_placement_stale is anchored
to self._resting_placed_tick_idx instead (stamped on every new placement,
including the fresh order from a CANCEL_AND_REPLACE), so replacing a
stale order genuinely resets this specific clock in a way r_stale never
could. Normalized against a much shorter window than r_stale (see
LOBExecutionEnv._PLACEMENT_STALENESS_WINDOW_TICKS) -- this is deliberately
a faster, more targeted signal, not a duplicate of r_stale at a different
scale. See RewardWeights.eta_replace for the loophole check (does spamming
CANCEL_AND_REPLACE every tick game this term) worked out explicitly.
Defaults to 0.0 (inert) until deliberately overridden for a probe.

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

SUPERSEDED by EXPERIMENTAL ADDITION 4 below -- the paragraph immediately
following is the original (backward, as it turned out) rationale for the
queue-weighted charge's direction. Kept verbatim as the trail that led to
the correction, not deleted.

IMPORTANT, and contrary to the intuition this split was designed around:
queue_ratio = q_ahead / (q_ahead + own_qty_remaining) is LARGEST for a
freshly placed order sitting behind a deep level, and tends to 0 for an
order that has already waited its way to the front. _place_limit()
initializes q_ahead to the full visible resting volume at the price, and
update_queue() only ever decreases it. So this term does NOT "still sting
for an order that has already accumulated priority" -- it stings for the
opposite case, and it is exactly 0.0 for a price outside the visible
top-20 book (where _place_limit() sets q_ahead = 0.0). The consequences
for the r_placement_stale loophole bound are worked out in full in
RewardWeights.eta_replace; they are not benign.

EXPERIMENTAL ADDITION 4 (corrects EXPERIMENTAL ADDITION 3's direction):
v1 (models/l3_executioner_v1.zip, the fixed-physics 2M-step checkpoint)
trained under the split above and, per the reproduced final-eval action
distribution, used CANCEL_AND_REPLACE on only 0.36% of actions and MARKET
on only 0.02% -- see docs/reports/phase3_l3_baseline_milestone.md. The
queue-weighted term's direction, described in the now-superseded paragraph
above, is the suspected structural cause: it charges LESS for discarding a
fresh, barely-waited order and MORE for discarding one that has already
earned priority through real queue consumption -- backward from any
sensible cost model of "wasted patience." This is a probe of that
direction, not a confirmed fix -- see the probe report for the outcome.

The queue-weighted term is inverted for BOTH canceled_via_market and
canceled_via_replace: gamma * (queue_ahead_at_cancel / queue_at_level)
becomes gamma * (1 - queue_ahead_at_cancel / queue_at_level). Since
queue_at_level = queue_ahead_at_cancel + own_qty_remaining, this is
algebraically gamma * own_qty_remaining / queue_at_level -- it grows
toward gamma (maximum) as q_ahead shrinks toward 0 (the order has waited
its way to the front, genuine earned priority, now expensive to discard)
and shrinks toward 0 as q_ahead grows large relative to the own resting
size (a fresh order behind a deep level, or a price that has stopped
seeing trade-through volume so q_ahead never shrinks -- both cheap to
discard, correctly this time). A price outside the visible top-20 book,
where q_ahead = 0.0 exactly, now costs the FULL gamma, not 0.0 -- the
opposite of before. See RewardWeights.eta_replace for the loophole bound
re-derived under this new formula.

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
opt-in convention as zeta/eta_replace above, so nothing changes until
deliberately tested. See that field's docstring for the mechanism and
docs/reports/ for the measured per-reset cost of computing it.
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
    # EXPERIMENTAL 2 (see module docstring): weight on r_placement_stale.
    # Defaults to 0.0 -- inert unless explicitly overridden (e.g. via
    # train_l3.py's --reward-zeta-style CLI mechanism), so nothing changes
    # until deliberately tested.
    #
    # LOOPHOLE CHECK -- REDERIVED after the EXPERIMENTAL 3 r_queue split.
    # The conclusion CHANGED: the loophole is no longer closed.
    #
    # BEFORE the split, every CANCEL_AND_REPLACE paid r_queue <= -beta =
    # -0.5 unconditionally, so spam-replacing every tick to keep
    # ticks_since_placement_norm pinned near 0 saved at most +eta_replace
    # per tick against a guaranteed -0.5. Strictly negative for any
    # eta_replace < 0.5, so the exploit was not viable at 0.02, 0.06 or
    # 0.15 (margins 25x, 8.3x, 3.3x below the 0.5 ceiling respectively --
    # note 0.15 is only 3.3x, NOT "an order of magnitude" as an earlier
    # revision of this comment claimed).
    #
    # AFTER the split, a CANCEL_AND_REPLACE pays only
    # -gamma * q_ahead/(q_ahead + own_qty_remaining), which is NOT bounded
    # away from zero:
    #   deep level, q_ahead >> own_qty  -> ratio -> 1, cost -> -gamma = -0.30/tick
    #   thin level, q_ahead ~ own_qty   -> ratio ~ 0.5, cost ~ -0.15/tick
    #   price outside visible top-20 book -> _place_limit() sets q_ahead =
    #       0.0 exactly, so ratio = 0 and the cost is EXACTLY 0.00/tick
    # and the policy chooses the price offset, so it can steer itself into
    # that zero-cost regime deliberately rather than stumbling into it.
    # Spam-replacing then nets up to +eta_replace per tick for zero charge:
    # the exploit is viable at ANY eta_replace > 0, not merely above some
    # threshold. beta no longer bounds it at all.
    #
    # CONSEQUENCE: r_placement_stale and the EXPERIMENTAL 3 r_queue split
    # must NOT be enabled together as they currently stand -- the two
    # interact and one of them needs redesigning first. r_placement_stale
    # should be revisited, or removed outright, alongside this change.
    # The isolation probe for the split is therefore run with BOTH
    # zeta = 0.0 and eta_replace = 0.0.
    #
    # LOOPHOLE CHECK -- REDERIVED AGAIN after EXPERIMENTAL 4's direction
    # inversion (module docstring). The section immediately above is now
    # SUPERSEDED -- kept verbatim as history, not deleted, since it is the
    # actual reasoning that led to the inversion.
    #
    # Under EXPERIMENTAL 4, a CANCEL_AND_REPLACE pays
    #   -gamma * (1 - q_ahead/(q_ahead + own_qty_remaining))
    #   = -gamma * own_qty_remaining/(q_ahead + own_qty_remaining).
    # The specific exploit named above -- steer to a price outside the
    # visible top-20 book, where q_ahead = 0.0 exactly, for a guaranteed
    # zero charge -- is CLOSED, and not just closed but inverted: q_ahead=0
    # now makes the ratio exactly 1.0, so the charge is the FULL -gamma,
    # the single most expensive outcome this term can produce. Off-book is
    # now the worst possible price to spam-replace at, not the best.
    #
    # This is NOT the same as saying the term is bounded away from zero in
    # every case -- it is not, and that residual needs stating plainly
    # rather than glossed over. own_qty_remaining has a floor: it is
    # size_frac * qty_remaining_at_placement with size_frac drawn from
    # SIZE_FRACTIONS = (0.2, 0.4, 0.6, 0.8, 1.0) -- never exactly 0, always
    # at least 20% of whatever qty_remaining was at the moment of that
    # placement. So the ratio -> 0 only if q_ahead is made large RELATIVE
    # TO a resting size that itself can never be driven below that 20%
    # floor -- i.e. only by choosing a price level with real, substantial
    # ambient resting depth already sitting there (several multiples of
    # the agent's own minimum order size). That is a genuine market
    # condition, not a free, policy-controlled knob the way "any off-book
    # price" was before: it depends on actual depth at the chosen level
    # and tick, is not guaranteed available every tick the way off-book
    # always was, and even when available only drives the ratio toward 0
    # asymptotically (never exactly 0 the way q_ahead=0 gave exactly 0
    # before).
    #
    # CONCLUSION: the specific, deterministic, always-available off-book
    # exploit is closed. A weaker, market-depth-dependent near-zero-cost
    # regime (thin own-size behind a very deep level) remains possible in
    # principle and is not mathematically ruled out -- harder to exploit
    # reliably than before, not provably impossible to exploit at all.
    # eta_replace stays at 0.0 (inert) for this probe regardless, so this
    # residual is not being tested here; it needs weighing explicitly, not
    # assumed away, before eta_replace is ever set above 0.0 again.
    eta_replace: float = 0.0
    # EXPERIMENTAL 5 (see module docstring): variance reduction, NOT an
    # objective change -- read the module docstring's EXPERIMENTAL 5 section
    # before touching this. Defaults to False -- inert unless explicitly
    # overridden, same opt-in convention as zeta/eta_replace above. When
    # True, LOBExecutionEnv subtracts a TWAP shadow-execution's terminal
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
    ticks_since_placement_norm: float = 0.0,
) -> float:
    """Section 3.3, components 1-4 (per-step; the terminal IS component is
    computed separately via compute_implementation_shortfall() and added
    once at episode end -- see LOBExecutionEnv.step()). Components 5-6
    (r_stale, r_placement_stale) are experimental additions, not in
    Section 3.3 -- see module docstring.

    r_queue is priced differently for the two cancel paths (EXPERIMENTAL 3,
    module docstring), and the queue-weighted component's direction is
    inverted from the original split (EXPERIMENTAL 4, module docstring).
    canceled_via_market and canceled_via_replace are mutually exclusive by
    construction in LOBExecutionEnv.step(); the flat -beta abandonment
    charge applies only to the MARKET path."""
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
        # EXPERIMENTAL 4 (module docstring): inverted direction -- charges
        # more for discarding queue position that has actually been earned
        # (q_ahead has shrunk toward 0 via real trade-through volume), less
        # for a fresh order that never waited. See RewardWeights.eta_replace
        # for the re-derived loophole bound.
        if queue_ahead_at_cancel is not None and queue_at_level:
            r_queue -= w.gamma * (1.0 - queue_ahead_at_cancel / queue_at_level)
    elif canceled_via_replace:
        # Not abandonment -- a fresh limit order replaces the old one, so the
        # flat -beta is dropped and REPLACE stops being dominated by MARKET.
        # The queue-weighted charge stays: discarded queue position is real.
        #
        # EXPERIMENTAL 4 (module docstring): corrected direction. The charge
        # scales with 1 - q_ahead/queue_at_level, i.e. with how much of the
        # level's current depth is the agent's own resting size -- q_ahead
        # shrinks toward 0 as trade-through volume consumes the orders ahead
        # of it, so this ratio grows toward 1 (max charge, -gamma) as the
        # order approaches the front through genuine earned priority. A
        # fresh order behind a deep level, or a price that has stopped
        # seeing trade-through volume (q_ahead never shrinks), costs close
        # to 0 instead. A price outside the visible top-20 book (_place_
        # limit() sets q_ahead = 0.0 there) now costs close to -gamma, not
        # 0.0 -- see RewardWeights.eta_replace for why this closes the
        # specific off-book spam-replace exploit the old direction left
        # open, and for the weaker residual that remains.
        if queue_ahead_at_cancel is not None and queue_at_level:
            r_queue -= w.gamma * (1.0 - queue_ahead_at_cancel / queue_at_level)

    r_stale = -w.zeta * ticks_since_own_fill_norm if resting else 0.0

    r_placement_stale = -w.eta_replace * ticks_since_placement_norm if resting else 0.0

    return r_slip + r_inv + r_queue + r_spread + r_stale + r_placement_stale


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
