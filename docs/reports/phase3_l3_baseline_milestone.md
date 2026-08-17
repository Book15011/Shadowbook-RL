# Phase 3 L3 baseline: training results and execution-behavior review

lob-execution-hma, RecurrentPPO, MlpLstmPolicy, 20,000,000 steps. This document
combines the training-run milestone report with a follow-up investigation into
one of its own flagged caveats, and the open questions that investigation
raised in turn. It supersedes treating either piece in isolation.

## Headline

L3 beat the TWAP baseline at 76 of 80 held-out evaluation checkpoints (95% of
the run), averaging 0.64 bps better implementation shortfall across all 80
evals (last-20 average: 0.54 bps, still solidly positive). Final eval: L3
IS_total_bps was 0.632 vs the TWAP fixed baseline of 1.265 bps.

That headline holds up under scrutiny. What follows is the full picture,
including the parts that do not look as clean.

## Training run summary

Full run: total_timesteps reached 20,001,776, in two segments joined by one
mid-flight recovery (an OOM interruption at step 1,750,000, resumed to
completion). 80 held-out evaluations against a fixed, paired seed set (n=50
episodes each), the same seeds used at every evaluation across both segments.
TWAP baseline is a fixed reference (same passive, same-level policy),
evaluated once and held constant across the run at 1.2652 bps IS_total and
0.9986 fill ratio.

Four things stand out in the training curves:

1. IS_total_bps vs TWAP tracks positive (L3 beating TWAP) for the large
   majority of the run, with real point-to-point noise (individual evals
   range roughly -0.5 to +1.8 bps beats-TWAP) but a consistently positive
   mean.
2. Fill ratio declined steadily, from about 0.96 near step 250,000 to a
   0.60-0.75 band for most of the remainder, and this decline started well
   before the OOM crash -- not a resume artifact. This was flagged in the
   original milestone report as worth a closer look, rather than accepted as
   fine because the headline number is positive. The follow-up investigation
   below is that closer look.
3. Cost decomposition (is_exec_bps vs is_opp_bps), logged from step 1,750,008
   onward: both terms are noisy and roughly centered near zero, with
   is_opp_bps (cost of the unfilled remainder, marked at the terminal price)
   showing the larger swings -- consistent with a meaningful and growing
   unfilled fraction per episode.
4. Episode length grew from 37.7 to 2,668.9 ticks against a 3,000-tick
   horizon cap, organically over training and continuing the same trajectory
   across the resume. Combined with point 2, this means most late-training
   episodes are running out the full horizon rather than terminating early
   via complete fill.


## Follow-up investigation: is the fill-ratio decline a real trade-off?

The original report described the fill-ratio decline as reading like a real
behavioral trade-off (holding out for better fills, accepting more terminal
opportunity cost) rather than a bug. That reading was tested directly: the
final checkpoint (models/l3_executioner_v1.zip plus l3_vecnormalize.pkl,
deterministic policy) was run for 15 full episodes across the val split,
capturing every raw per-step action (order_type, price_offset_idx,
size_frac_idx) plus queue-position state (obs idx 13, resting_q_ahead from
the info dict).

It does not hold up. Across all 40,367 recorded ticks in all 15 episodes,
MARKET and CANCEL_AND_REPLACE were used zero times. LIMIT while already
resting is a documented no-op in lob_execution_env.py, so filtering to
genuine placement events (order_type in LIMIT or CANCEL_REPLACE, no order
already resting) leaves only 26 real placements across 15 episodes -- most
episodes place once, early, and the resulting order then either fills or
does not, with no further agent action of any kind for the rest of the
horizon.

Of the 9 sampled low-fill (under 0.7) episodes, only 3 of 12 placement
events ever received a fill at all; the 3 that did took a median of 2,690 of
the 3,000-tick horizon. queue_position_ratio (obs idx 13) while resting sat
at a median of exactly 0.0 in low-fill episodes -- the order is at, or has
decayed to, the front of the queue -- yet mostly still does not fill. Per
the expected_wait_time formula in Section 2.4 (expected_wait_time equals
q_ahead divided by avg_trade_rate), a near-zero q_ahead should predict a
short wait if the level is still trading. Front-of-queue orders sitting
unfilled for thousands of ticks is more consistent with the trading rate at
that specific, now-stale price having collapsed to near zero (the market
moved away) than with a deep, active queue the agent is patiently waiting
out.

By contrast, the 2 high-fill (over 0.9) episodes had all 7 of their
placement streaks eventually fill, at a median of 125 ticks.

Revised verdict: this looks like under-execution from a missing corrective
mechanism, not deliberate patient, price-improving execution. The policy
has apparently learned to never use two of its four available order types.


## Further review

The above is itself not the end of the story. Three points, stated plainly
rather than smoothed over:

1. The "places once and never touches it again" summary does not hold for
   the high-fill bucket, and this was an open question as of this writing.
   The 7 placement streaks across only 2 high-fill episodes average 3.5
   placements per episode, not once -- an apparent contradiction with the
   low-fill bucket average of about 1.3 per episode that this document did
   not yet resolve. Two readings are possible without further work: either
   the high-fill episodes see more genuine corrective re-placements (which
   would undercut the "zero correction usage" framing above), or they see
   more full-fill-then-fresh-tranche cycles (which would be consistent with
   it). Pending disambiguation -- see Part A below, tracked to resolve this
   specific question against the existing raw data rather than leaving it
   as an unresolved tension in a document meant to be read as a coherent
   record.

2. The "avg_trade_rate collapsed to near zero" explanation is an inference,
   not a measurement. The expected_wait_time formula from Section 2.4 was
   never evaluated against a directly logged trade-rate-at-price series --
   no such series is logged anywhere in this environment. The claim rests
   on: q_ahead is near zero (measured), the order still does not fill
   (measured), and the only remaining free variable in the formula is
   avg_trade_rate (not measured) -- so a collapsed trade rate is the
   explanation consistent with the two things actually observed, not a
   third directly observed fact. A directly logged per-price trade-rate
   series would turn this into a measurement instead of an inference; none
   currently exists in this environment info dict.

3. The reward-structure explanation, with the arithmetic shown.
   step_reward() in src/envs/reward.py currently penalizes an active
   correction (r_queue, gated on canceled_unfilled) but nothing about simply
   leaving a stale, unfilled order in place. Concretely, using the deployed
   weights (alpha=1.0, lam=0.02, beta=0.5, gamma=0.3, delta=0.8, kappa=1.0)
   and this environment dt=0.1s, horizon_ticks=3000:

   - Holding an unfilled, near-full-inventory position costs r_inv per tick:
     r_inv = -lam * (1 + max(0, l1_risk_score)) * (qty_remaining/qty_total)^2
     * dt = -0.02 * 1 * 1^2 * 0.1 = -0.002 per idle tick (l1_risk_score=0 by
     default, the L1 stub is a no-op in Phase 2b and Phase 3).
   - A single CANCEL_AND_REPLACE on an order with almost nothing filled
     costs up to r_queue = -beta - gamma * (queue_ahead_at_cancel /
     queue_at_level) = -0.5 - 0.3 * 1 = -0.8 in the worst case (queue ratio
     up to 1).
   - Break-even: 0.8 / 0.002 = 400 idle ticks (about 40 seconds of simulated
     time) before a correction becomes cheaper than continuing to hold.
     Below that, the policy gradient always favors doing nothing.
   - Separately, the RL discount gamma=0.995 (not to be confused with the
     reward function queue-weight gamma above) over episodes averaging about
     2,700 ticks gives an effective discount horizon of 1/(1-0.995) = 200
     ticks. The terminal IS penalty (kappa times is_total_bps, applied once
     per episode) that is meant to eventually punish leaving inventory
     unexecuted is credited back to an early "should I correct" decision at
     a discount of roughly 0.995^2700, about 1.33e-6 -- six orders of
     magnitude smaller than the immediate, certain per-tick and
     per-correction costs above. The one reward term that should push
     against staleness is discounted into near irrelevance for any single
     early-episode decision.

This is presented as the most likely explanation for the observed zero
percent CANCEL_AND_REPLACE and MARKET usage rate, not a proven certainty --
it is consistent with everything measured above, but has not yet been
tested by actually changing the reward and observing the effect (see Part B
and Part C).

Verdict on this checkpoint: it should not be treated as final, and should
not be wired into Phase 3.5 as-is, until the staleness gap above is
addressed and the fix is validated against real training data, not just
argued from the reward-function arithmetic.
