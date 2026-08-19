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

1. RESOLVED (see Part A below for the analysis). The "places once and never
   touches it again" summary does not actually contradict the high-fill
   bucket. Classifying all 26 real placements against the raw per-step data:
   every single one is either the first placement in its episode (15 of 26)
   or a fresh tranche placed only after the prior order in that same episode
   fully resolved via a complete fill (11 of 26) -- zero were placed while a
   prior order was still outstanding. This is not a coincidence: it is
   mechanically guaranteed by the already-measured 0% CANCEL_AND_REPLACE
   usage rate, since nothing else can free up the resting-order slot
   mid-episode. The high-fill bucket average of 3.5 placements per episode
   (5 of its 7 events are fresh-tranche-after-full-fill) reflects orders
   resolving quickly and completely, cycling through several tranches in a
   row -- not corrective behavior. The low-fill bucket averages 1.3 per
   episode (only 3 of 12 events are fresh-tranche-after-full-fill, and 2 of
   those 3 occur in the final 10% of the horizon, at l2_target_slice_ratio
   0.90 and 0.96, leaving almost no runway to matter). Six of the 9 low-fill
   episodes place exactly once and never place again for the rest of the
   horizon. Separately: l2_target_slice_ratio (obs idx 15) is a continuous
   per-tick ramp (ticks_elapsed divided by horizon_ticks), not a discretized
   N-slice schedule, so it changes by construction on every single tick --
   "how many times did it change" is not a meaningful question in this
   environment. The meaningful question, fresh opportunity versus stuck-
   order correction, is answered structurally above without needing that
   quantity.

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

## Part A: high-fill placement-frequency disambiguation (resolved)

Classified all 26 real placements from the rollout investigation against
the raw per-step data (/tmp/rollout_analysis_raw.json), using the already-
confirmed fact that CANCEL_AND_REPLACE/MARKET are never used, so the only
way a new placement can occur after the first is if the prior order in that
same episode fully resolved via a complete fill.

Totals: 15 of 26 are the first placement in their episode; 11 of 26 are a
fresh tranche placed after the prior order fully filled; 0 of 26 occurred
while a prior order was still outstanding (structurally impossible given
0% CANCEL_AND_REPLACE usage, and confirmed directly rather than assumed).

HIGH-FILL bucket (7 events, 2 episodes): 2 first-in-episode, 5 fresh-
tranche-after-full-fill, 0 corrections. This is what produces the 3.5
placements/episode average -- orders resolving quickly and completely,
repeatedly, not corrective re-pricing.

LOW-FILL bucket (12 events, 9 episodes): 9 first-in-episode, 3 fresh-
tranche-after-full-fill, 0 corrections. Two of those three fresh tranches
land at l2_target_slice_ratio 0.90 and 0.96 -- in the final 10% of the
horizon, essentially out of runway. Six of the 9 low-fill episodes place
exactly once and take no further action of any kind for the remainder of
the episode, regardless of how far l2_target_slice_ratio (the L2 stub
pacing schedule) climbs above their actual, stalled progress.

This closes the open question from further-review point 1 above: the
high-fill bucket placement frequency and the "places once and never
touches it again" low-fill characterization are both correct descriptions
of the same underlying mechanism (fresh placements only follow full
resolution, never a correction), not a contradiction between them.

## Part B: the staleness reward term

Implemented in src/envs/reward.py: a new RewardWeights.zeta = 0.06 weight
and an r_stale = -zeta * ticks_since_own_fill_norm term (zero unless an
order is resting and unfilled), added to step_reward()'s return value
alongside the existing four components. Coefficient derivation is in the
further-review section above. src/envs/lob_execution_env.py required a
small, necessary wiring change: the existing ticks_since_own_fill_norm
computation (already feeding obs idx 14) was factored into a reusable
_ticks_since_own_fill_norm() helper, called from both _build_obs() (value
unchanged) and step()'s step_reward() call (new). No changes to
matching_engine.py or the observation/action space definitions. Two new
isolated tests were added to tests/test_reward.py, matching the file
existing one-test-per-component style; the full test suite was re-run
afterward -- all 38 previously-passing tests plus the 2 new ones pass (4
pre-existing failures in test_bulk_backfill.py and test_l2_capture.py are
unrelated: network-mock URL drift and an L2 gap-resync edge case, neither
file importing reward.py or lob_execution_env.py).

## Part C: validation probe results

Warm-started from the 20M-step baseline, resumed 2,000,000 steps under the
new reward (total_timesteps 20,001,776 to 22,004,720), same n_envs=8,
logged to logs/l3_train_staleness_probe.log.

Real, measured effect across the probe 8 held-out eval firings (paired
seeds 5000000-5000049, same set the baseline itself used, n=50 episodes
each):

fill_ratio: 0.582, 0.641, 0.595, 0.632, 0.625, 0.648, 0.647, 0.803, then
0.969 at the final firing -- a clear, substantial recovery from the
baseline own ~0.59-0.65 band (its last eval before this probe, at step
20,000,008, was 0.590). Corroborated independently: a fresh 8-episode
deterministic rollout on the finished probe checkpoint (different seeds
than the paired eval set, same seed pattern as the original rollout
investigation) found 6 of 8 episodes reaching fill_ratio=1.0 exactly, and
MARKET orders used 24 times out of 15,069 recorded ticks (0.16%) -- up
from exactly 0% in the baseline across 40,367 ticks. CANCEL_AND_REPLACE
remained at exactly 0% in this spot check.

IS_total_bps over the same 8 firings: 0.509, 0.331, 0.298, 0.541, 0.831,
0.849, 1.343, 1.448, then 1.101 at the final firing -- trending worse, not
better, and briefly underperforming the TWAP baseline of 1.2652 at two
firings (1.343 and 1.448). By the final firing the L3-vs-TWAP margin had
shrunk to about 0.16 bps, down from the baseline own roughly 0.63 bps at a
comparable step count. This is the real, measured cost of the fix: pushing
the policy to actually complete orders trades away some of the
price-improvement edge that came from its previous willingness to sit
passively and only fill on favorable terms.

Verdict: the reward-structure hypothesis in the further-review section
above is validated, not just argued -- changing one reward term measurably
changed the exact policy behavior it was designed to change, within
2,000,000 steps (10% of the original run length). This is a genuine
result, not a free one: zeta=0.06 clearly overcorrects within this short
probe, trading most of the IS edge for fill-ratio recovery rather than
preserving both. Whether a smaller zeta, or the same zeta given more
training time to re-balance, recovers more of that edge is an open
question this probe was not designed to answer, and is left for a
deliberate follow-up rather than decided here.

Both the original 20M-step baseline (restored to its canonical path,
models/l3_executioner_v1.zip and models/l3_vecnormalize.pkl, verified via
checksum against models/baseline_20M_backup/) and this probe own
checkpoint (models/l3_executioner_v1_staleness_probe.zip and
models/l3_executioner_v1_staleness_probe_vecnormalize.pkl) are preserved on
disk for comparison.


## Coefficient sweep (zeta=0.06 vs 0.006 vs 0.002)

Following the zeta=0.06 probe (fill_ratio 0.582 to 0.969, IS-vs-TWAP margin
0.63 to 0.16 bps), two questions needed answering before treating 0.06 as
the answer: what mechanism actually drove the fill_ratio recovery, and
would a gentler coefficient recover more of the IS edge while still fixing
the problem.

### Mechanism diagnosis (on the zeta=0.06 checkpoint)

A 15-episode rollout (same protocol as the original investigation) on the
finished zeta=0.06 checkpoint found:

- Placements per episode: 4.47, versus about 1.7 in the original baseline
  -- the dominant factor.
- price_offset_ticks at true placements: 65.67% passive, 25.37%
  aggressive, 8.96% at-touch -- a real but partial shift toward
  aggression (the baseline was 75-86% passive across its low/high-fill
  buckets), not a full reversal.
- MARKET usage: 0.26% of ticks (up from exactly 0%), and for the first
  time this actually cancels a still-resting unfilled order before
  acting (14 of 67 placement streaks in this sample ended this way,
  something structurally impossible at 0% MARKET usage). CANCEL_AND_REPLACE
  stayed at exactly 0%.
- Time-to-fill for streaks that did fill: median 195.5 ticks, versus
  2,690 in the original low-fill bucket -- about a 14x collapse.

Direct check on the ticks_since_own_fill_norm signal itself (obs idx 14,
also the new reward term input) while resting: only 2.77% of resting-ticks
sit at the fully saturated value of 1.0. Most sit near 0, because orders
are cycling and refilling far more often now.

Conclusion: the fill_ratio recovery is not explained by the small
MARKET-usage increase alone, nor primarily by a full reversal to
aggressive pricing. It is driven mainly by dramatically more frequent
re-engagement (2.6x more placement attempts per episode), combined with a
modest aggressiveness shift, with the new (if still small) MARKET-cancel
pathway playing a real but secondary supporting role.

### The saturation behavior changes the coefficient arithmetic

Direct source inspection (src/envs/lob_execution_env.py,
_ticks_since_own_fill_norm()) confirms: this value returns exactly 1.0
immediately whenever an episode has had zero fills so far -- it is a hard
floor, not a ramp building up from 0 to 1 over the episode. The original
break-even derivation in this document Part B assumed a linear 0-to-1 ramp
(giving a quadratic cumulative cost, sum_{k=1..K} zeta*(k/3000)). For the
dominant case (an order resting, never yet filled this episode), the real
cumulative cost is instead linear: -(0.002 + zeta) * K per K ticks held.

Recomputing the real break-even (ticks before a single correction, costing
up to 0.8, becomes cheaper than continuing to hold):

- zeta=0.06 (probed): real break-even is about 12.9 ticks, not the
  originally intended ~400 -- this is the actual reason 0.06 overcorrected
  as hard as it did.
- Naive "roughly 1/3 and 2/3 of 0.06" (0.02, 0.04): break-even about 36.4
  and 19.1 ticks respectively -- still far too aggressive, confirming the
  naive linear-fraction guess would not have meaningfully fixed the
  overcorrection.
- zeta=0.006 (chosen): break-even about 100 ticks.
- zeta=0.002 (chosen): break-even about 200 ticks, matching this
  document original intended target.

### Sweep results

Both candidates were warm-started from the ORIGINAL 20M-step baseline
(not from the zeta=0.06 checkpoint), 2,000,000 steps each, same n_envs=8.
A --reward-zeta CLI override was added to src/train/train_l3.py (threaded
through make_env() and ValISEvalCallback, both accepting an optional
RewardWeights) so each candidate could run without editing the shared
src/envs/reward.py default per run -- necessary infrastructure for the
sweep itself, not a change to matching_engine.py or the observation/action
space.

Paired eval (n=50, seeds 5000000-5000049, the reliable comparison basis),
fill_ratio then IS_total_bps, first firing to last:

- Baseline (no staleness term): flat around 0.59-0.65; IS_total_bps
  around 0.63 bps margin over TWAP (1.2652) at a comparable step count.
- zeta=0.06: fill_ratio 0.582 to 0.969 (strong recovery); IS_total_bps
  0.509 to 1.101 (margin over TWAP shrinks from about 0.63 to 0.16 bps).
- zeta=0.006: fill_ratio 0.582 to 0.527 (flat, no recovery across the
  9 firings, actual sequence 0.582, 0.563, 0.568, 0.53, 0.615, 0.587,
  0.618, 0.606, 0.527); IS_total_bps 0.509 to 0.637 (margin over TWAP
  preserved at about 0.63 bps, matching the baseline).
- zeta=0.002: fill_ratio 0.582 to 0.625 (flat, no recovery; sequence
  0.582, 0.642, 0.629, 0.645, 0.624, 0.61, 0.625 -- two mid-sequence
  firings were lost to log corruption during the earlier parallel-run
  incident, see methodology note below); IS_total_bps 0.509 to 0.652
  (margin over TWAP about 0.61 bps, also close to baseline).

Independent 8-episode direct rollout check (own seed convention, order-type
usage rates the eval log does not report):

- zeta=0.06: MARKET 0.16-0.26% (measured twice, 15-episode and 8-episode
  samples), CANCEL_AND_REPLACE 0%.
- zeta=0.006: MARKET 0.004% (1 of 23,996 ticks -- noise, not a real
  signal), CANCEL_AND_REPLACE 0%. mean_fill_ratio on this 8-episode sample
  was 0.2356, notably lower than the paired-eval trend -- consistent with
  this system own known high per-episode variance on a small, differently
  seeded sample, not treated as more reliable than the n=50 paired result.
- zeta=0.002: MARKET 0%, CANCEL_AND_REPLACE 0%. mean_fill_ratio on this
  sample was 0.2641, same caveat as above.

### Finding

Neither gentler candidate shows the fix working within this 2,000,000-step
window -- both remain statistically indistinguishable from the baseline
own 0% MARKET/CANCEL_AND_REPLACE usage pattern and its ~0.59-0.65
fill_ratio band. Only zeta=0.06 own much stronger, near-immediate penalty
(about 13-tick break-even versus 100-200 for the gentler candidates) was
enough to break the place-once-and-never-touch-it-again pattern within a
short probe. This does not prove the gentler candidates cannot work --
they may simply need substantially more than 2,000,000 steps for a weaker
per-tick signal to accumulate enough gradient pressure -- but that is
untested, not confirmed, by this sweep.

Of the three tested, zeta=0.06 is the only one that demonstrably fixes the
under-execution problem, at a real, not-yet-shown-to-recover cost to the
IS-vs-TWAP edge. The gentler candidates preserve the edge but do not yet
fix the problem within the tested window. No candidate has been picked as
a winner; this is deliberately left for review rather than decided here.
Both src/envs/reward.py (zeta=0.06 default, already committed in the prior
revision) and the new src/train/train_l3.py --reward-zeta override
(uncommitted) remain in their current state pending that decision.

### Methodology note: a resource-contention incident during the sweep

The two lower-zeta probes were originally launched in parallel (the GPU
was fully idle at the time, unlike earlier in this session when it was
shared with an unrelated project). A duplicate zeta=0.006 process appears
to have been launched alongside the intended one, most likely when a
launch command that timed out at the tool level and was moved to
background caused its underlying nohup invocation to fire twice. With
three n_envs=8 jobs competing for RAM instead of two, the kernel OOM
killer fired twice (confirmed via dmesg), killing the tracked zeta=0.006
process and, 23 minutes later, the unexplained duplicate. zeta=0.002
survived only because the kernel OOM heuristic picked the other two
processes both times, not by any protection. zeta=0.006 was relaunched
solo afterward (verified via a fresh connection that exactly one process
was running before proceeding) and completed cleanly. All further probes
in this sweep ran strictly sequentially, not in parallel, given this
confirmed real (not just theoretical) resource-contention risk. The
zeta=0.002 log shows minor corruption (two mid-sequence eval firings
missing) likely from the same period of concurrent writes; the remaining
7 of 9 firings are intact and used above.


## Extended coefficient sweep: filling the middle zone

The first sweep tested only two break-even points: 12.9 ticks (zeta=0.06,
full fix, high cost) and 100/200 ticks (zeta=0.006/0.002, zero measurable
effect) -- skipping the zone where the actual transition most likely
lives. Four new probes filled that gap, each warm-started from the
ORIGINAL 20M-step baseline (not any zeta probe checkpoint), 2,000,000
steps, same n_envs=8: zeta=0.0 (a genuine same-window control with the
untouched reward), zeta=0.008 (break-even about 80 ticks), zeta=0.015
(about 47 ticks), zeta=0.03 (about 25 ticks).

### The zeta=0.0 control validates the historical comparison

fill_ratio across its 9 paired-eval firings: 0.582, 0.595, 0.608, 0.604,
0.615, 0.636, 0.672, 0.642, 0.599 -- a 0.58-0.67 band, matching the
historical baseline range (about 0.59-0.65) this document has been citing
throughout. This confirms that band is a real, reproducible property of
the un-modified reward on a matched 2,000,000-step window, not an
artifact of a different time period or a stale comparison.

### None of the three new non-zero candidates show any effect

fill_ratio, paired eval (n=50, seeds 5000000-5000049), first firing to
last:

- zeta=0.008 (break-even 80 ticks): 0.582, 0.61, 0.653, 0.651, 0.63,
  0.615, 0.631, 0.616, 0.672 -- flat, same band as the control.
- zeta=0.015 (break-even 47 ticks): 0.582, 0.633, 0.58, 0.627, 0.595,
  0.611, 0.607, 0.643, 0.638 -- flat, same band.
- zeta=0.03 (break-even 25 ticks): 0.582, 0.628, 0.581, 0.589, 0.622,
  0.615, 0.583, 0.582, 0.569 -- flat, if anything slightly below the
  control by the final firing.

An independent 8-episode direct rollout check on each finished checkpoint
confirms this at the order-type level: MARKET and CANCEL_AND_REPLACE both
sit at exactly 0.0% for all three (0.0, 0.008, 0.015, 0.03), identical to
the control -- no measurable behavioral change of any kind, not just a
fill_ratio coincidence.

### The transition is a threshold, not a gradient

Six break-even points have now been tested, spanning 400 ticks down to
12.9 ticks: 400 (zeta=0.0), 100 (zeta=0.006), 80 (zeta=0.008), 47
(zeta=0.015), 25 (zeta=0.03), and 12.9 (zeta=0.06). Every single one is
flat -- statistically indistinguishable from the untouched control --
except 12.9. This rules out a smooth dose-response relationship across
the tested range: the real transition sits somewhere between a 25-tick
and a 12.9-tick break-even (zeta between 0.03 and 0.06), not spread
gradually across the wider range that was the working assumption when
this extended sweep was designed. Narrowing that specific window (e.g.
zeta around 0.04-0.05) would be the natural next probe if a precise
threshold is wanted, but has not been run -- no candidate is picked as a
winner here, per instruction to stop for review.


## Part A/B/C follow-up: a placement-anchored signal, tested

The coefficient sweep confirmed r_stale is structurally incapable of ever
rewarding CANCEL_AND_REPLACE at any zeta: _ticks_since_own_fill_norm()
depends solely on self._last_fill_tick_idx, set ONLY inside
`if step_fills:` in step() -- a CANCEL_AND_REPLACE never touches it, so
r_stale cannot tell "sitting on one stale order for 1000 ticks" apart from
"replaced 5 times in the last 10 ticks, still unfilled." This section
covers a direct fix for that gap, a reward-component decomposition to
check an existing assumption, and three probes of the fix in isolation.

### Part A: a placement-anchored staleness signal

Added self._resting_placed_tick_idx (mirrors self._last_fill_tick_idx's
init-only pattern exactly, in __init__ and reset()), stamped to the
current tick on every new placement in _place_limit() -- including the
fresh order from a CANCEL_AND_REPLACE, which is the whole point: replacing
a stale order now genuinely resets this specific clock. A new getter,
_ticks_since_placement_norm(), mirrors the existing getter structure but
normalizes against a new _PLACEMENT_STALENESS_WINDOW_TICKS constant (300
ticks, 30s) rather than horizon_ticks -- chosen because horizon_ticks is
already covered by the existing signal; 300 ticks comes from this project
own measured fast-fill timescale (successful fills clustered at a 125-195
tick median across every checkpoint tested in the coefficient sweep
above), not Section 2.4's expected_wait_time directly (that formula needs
a live per-price trade-rate estimate this environment does not compute --
the same gap flagged in the original rollout investigation). This value is
reward-internal only -- not added to the 42-dim observation vector, so the
existing baseline checkpoint stays warm-start compatible, and
matching_engine.py was not touched.

Wired into step_reward() as a new r_placement_stale term with its own
independent weight, RewardWeights.eta_replace, defaulting to 0.0 (inert)
so nothing changes unless explicitly overridden. Loophole check, worked
out explicitly rather than assumed: spamming CANCEL_AND_REPLACE every tick
keeps ticks_since_placement_norm pinned near 0, saving at most
eta_replace*1.0 per tick -- but every such cancel also sets
canceled_unfilled=True, costing r_queue at least -beta = -0.5 per tick
unconditionally. Net effect of spamming: at most +eta_replace - 0.5 per
tick, strictly negative for any eta_replace below 0.5 -- every coefficient
tested below has more than 3x margin under that ceiling.

44 tests pass (3 new isolated reward.py tests, 1 new env-level test
directly proving the CANCEL_AND_REPLACE-resets-the-clock behavior r_stale
could never have). A --reward-eta-replace CLI override was added to
train_l3.py, mirroring --reward-zeta, so it can be combined with
--reward-zeta 0.0 to isolate this term cleanly.

### Part B: does zeta=0.06 actually trade slippage for spread capture?

A 15-episode rollout on both the original baseline and the zeta=0.06
checkpoint, with src.envs.lob_execution_env's own step_reward reference
monkey-patched (observation only, no source files modified, the real
step_reward() still computes the actual returned value) to capture and
sum each reward component separately:

  component     baseline (40,367 ticks)   zeta=0.06 (24,603 ticks)
  r_slip        0.82                      11.39
  r_inv         -39.48                    -13.68
  r_queue       0.00                      -15.68
  r_spread      1.65                      1.62
  r_stale       -876.96 (retroactive)     -141.98

(both scored at the same RewardWeights(zeta=0.06) for an apples-to-apples
comparison, since the baseline checkpoint never actually experienced
r_stale during its own training.)

The raw sums are confounded by fill_ratio itself (0.52 baseline vs 0.98
under zeta=0.06 on this sample -- nearly double the filled quantity
contributing to every per-fill term). Normalizing r_slip by fill_ratio as
a volume proxy: baseline about 1.56, zeta=0.06 about 11.6 -- per-unit
slippage looks MORE favorable under zeta=0.06, not worse, opposite of the
naive prediction. r_spread's raw sum stayed nearly flat (1.65 to 1.62)
despite close to double the filled volume -- that flatness itself implies
per-unit spread capture did erode, just masked by higher volume in the raw
total. Conclusion: partially confirmed, partially refuted, not a clean
result either way -- r_spread-per-unit looks to have dropped as
hypothesized, but r_slip-per-unit improved, likely because a much larger
and more representative set of successful fills dilutes the cost of any
individual aggressive fill, rather than execution quality degrading
uniformly. r_queue's new nonzero cost (-15.68) directly confirms
MARKET-triggered cancellations are real and paid for under zeta=0.06.

### Part C: three isolated probes of eta_replace

Break-even ticks under the correct model for this new signal: unlike
r_stale (a hard floor at exactly 1.0 until the first fill, so the earlier
sweep linear cost model applied), ticks_since_placement_norm is a genuine
ramp from 0 (every placement, including a replace, resets it), so the
ORIGINAL quadratic accumulation model applies here correctly:
sum_{k=1..K} eta_replace*(k/300) for K under 300. Solving for K at the
0.8 correction-cost threshold: eta_replace=0.02 gives K about 128 ticks,
0.06 gives about 80, 0.15 gives about 53 -- all comfortably under the
0.5 loophole ceiling with real margin.

All three probes were warm-started from the ORIGINAL 20M baseline (not
any zeta checkpoint), 2,000,000 steps each, zeta=0.0 explicitly to isolate
this term own effect, run strictly sequentially (a user request to
parallelize all three was declined -- a single job alone was confirmed
using about 25GB of the system 50GB RAM, so three at once was not
physically possible without a near-certain repeat of the earlier
OOM-kill incident).

Paired eval (n=50) fill_ratio, first firing to last:
  eta_replace=0.02: 0.582, 0.598, 0.614, 0.955, 1.0, 1.0, 1.0, 1.0, 1.0
  eta_replace=0.06: 0.582, 0.644, 0.828, 1.0, 1.0, 0.985, 0.962, 0.972, 0.79
  eta_replace=0.15: 0.582, 0.633, 0.963, 1.0, 1.0, 1.0, 1.0, 0.999, 0.992

All three reach essentially complete fill_ratio recovery, and much faster
than zeta=0.06 ever did (by the 3rd-4th firing, versus a gradual climb
across the full 2,000,000 steps). Final IS_total_bps margin over TWAP
(1.2652): eta_replace=0.02 about 0.34 bps, eta_replace=0.06 about -0.69
bps (the only candidate anywhere in this sweep to finish WORSE than TWAP),
eta_replace=0.15 about 0.44 bps -- the best margin of every candidate
tested in this document, original baseline included.

An independent 8-episode rollout check on each finished checkpoint,
specifically for CANCEL_AND_REPLACE usage (the direct test of whether this
new signal does what r_stale structurally could not): CANCEL_AND_REPLACE
stayed at EXACTLY 0.0% for all three candidates. MARKET usage rose
substantially instead (2.91% for eta_replace=0.02, 1.06% for 0.15, 0.36%
for 0.06 -- all higher than zeta=0.06's own 0.16-0.26%). The fill_ratio
recovery is real, but it is not happening through the mechanism Part A was
built to unlock: the policy is escaping staleness by forcing immediate
market execution (which, if a resting order was present, cancels it as a
side effect of MARKET's own cancel-then-execute path -- see step()), not
by replacing a stale limit order with a fresh, better-priced one.

### Verdict

Part A's structural fix is real and correctly implemented -- confirmed
directly by the new env-level test, not just argued. It measurably changes
policy behavior (fill_ratio recovery, higher MARKET usage) at coefficients
an order of magnitude smaller and faster-acting than zeta ever needed. But
the specific hypothesis -- that removing the structural block on rewarding
CANCEL_AND_REPLACE would cause the policy to actually use it -- is not
confirmed by this data. The policy found a different, MARKET-based escape
route instead. eta_replace=0.15 is the strongest candidate across every
row in this document by both fill_ratio and IS-margin, but it achieves
that without engaging the mechanism this whole investigation set out to
unlock, and this is reported plainly rather than reframed as a win on the
original hypothesis. No zeta/eta_replace combination is picked here; left
for review.
