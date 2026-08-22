# TWAP-baseline (variance-reduction) reward: implementation

**Date:** 2026-08-21 (implementation), 2026-08-22 (A/B test result, see bottom)
**Status:** Implemented, tested, gated OFF by default. Matched A/B test run
2026-08-22 -- see "A/B test result" section at the bottom for the outcome:
a clean negative (Arm B, the treatment, is significantly WORSE than the
matched control).

## What this is, stated explicitly so it is not later mistaken for something else

This is a **baseline subtraction**, not an objective change. The terminal
reward term changes from `-kappa * IS_total_bps` to
`-kappa * (IS_total_bps - twap_IS_total_bps)`, where `twap_IS_total_bps` is
computed once per episode, over the identical market window, and **never
observes or reacts to what the real policy does that episode**. Because the
subtracted quantity is a per-episode constant with respect to the agent's own
actions, this shifts the *scale* of the return but not its *argmax* -- it
cannot change which policy is optimal, and creates no new incentive to "beat"
TWAP or anything else. It is pure variance reduction, aimed at a specific,
previously-stated structural problem: IS_total_bps variance is dominated by
market drift (observed std ~4-5bps across every eval this project has run)
while the agent's own execution-quality contribution is fractions of a bp,
and the critic has no way to predict the drift component -- it cannot observe
the future. Subtracting TWAP's own outcome on the same window removes exactly
the part of the signal the critic was structurally blind to.

Full rationale is also in the `reward.py` module docstring (EXPERIMENTAL
ADDITION 5) and `RewardWeights.subtract_twap_baseline`'s own docstring --
written there deliberately, not only here, so nobody reading the code cold
mistakes this for an objective change later.

## Design decision: computed in `reset()`, not cached

The task named two options: compute fresh in `reset()` (adds per-episode
cost) or precompute/cache per (seed, window). **Recommendation: compute in
reset(), and caching would not actually help training resets regardless of
which option is chosen for the reward term itself.**

Reasoning, not just a pick: TWAP's IS for a window is a per-episode constant
*with respect to the policy*, but it is emphatically NOT a constant with
respect to *which window gets drawn* -- and during training, essentially every
reset draws a genuinely novel window. `reset()` selects `file_idx` and
`start_tick` via `self.np_random.integers(...)` over ~405 train days x
~861,000 valid start-ticks each (n_rows per day minus horizon/lookback
buffer) -- on the order of 349 million distinct (file, start) combinations.
Over any realistic training budget (millions of steps, hundreds of thousands
of episodes), the probability of the *same* window recurring is negligible
(birthday-paradox territory against a 349-million-slot space). A cache keyed
by (seed, window) would sit at essentially a 0% hit rate for training
resets specifically -- it would only help the periodic eval callback, which
legitimately does reuse the same fixed 50 (or 500) seeds many times across a
training run's eval firings. That is a real but separate optimization
opportunity, not attempted here since it doesn't change the training-cost
picture this task is asking about.

Given that, `reset()`-time computation is not really "one of two options" for
training -- it is the only one that does anything for the case that matters
most. The implementation: `LOBExecutionEnv._compute_twap_shadow_terminal_is()`,
called from `reset()` only when `reward_weights.subtract_twap_baseline` is
True, storing the result in `self._twap_shadow_terminal_is_bps` (`None`
when the flag is off, so nothing is computed at all unless deliberately
enabled). It operates entirely on local variables -- never reads or writes
`self._resting`/`self.qty_remaining`/`self._tick_idx` -- so it cannot
interfere with the real episode that follows, and reuses the exact matching-
engine primitives `step()` itself uses (`walk_market_fill`, `update_queue`,
`TickView.qty_at_price`, `compute_implementation_shortfall`,
`self._estimate_trade_volume`) rather than reimplementing the underlying
physics. Only the TWAP-specific slicing/routing decision is duplicated
in-line (not imported from `scripts/phase2a_sanity_suite.py`, which is
documented as a throwaway evaluation script that core env code should not
depend on).

## Measured cost, not guessed

Measured directly on real market data (30 resets each, `train_date_range`,
`horizon_ticks=3000`), not the trivial synthetic test fixture:

| | flag OFF | flag ON |
|---|---|---|
| mean reset() time | 2026.7ms | 2074.8ms |

**Added cost: ~48ms/reset, a 2.4% overhead on reset() itself.** This is far
smaller than an earlier back-of-envelope worry (that a full 3,000-tick shadow
simulation might roughly double per-episode cost) -- measuring instead of
assuming caught that concern being wrong, not just imprecise. The reason:
`reset()`'s own baseline cost (~2 seconds) is dominated by day-file loading
and feature-precomputation (I/O and pandas/numpy work over the whole day),
which vastly exceeds a 3,000-tick shadow simulation's cost (pure in-memory
arithmetic over already-loaded book data, no I/O, no feature computation, no
model inference). Relative to a full episode's own wall-clock budget at v1's
measured ~350 aggregate fps / 8 workers (~1800 ticks/episode observed this
session, ~41s/episode/worker at that rate), the added 48ms is roughly **0.1%**
of that budget -- not a meaningful training-throughput concern at n_envs=8.
This has NOT been measured under actual multi-worker SubprocVecEnv training
load (only single-process, sequential resets) -- flagging that as the
one thing this measurement doesn't directly confirm, though there's no
obvious mechanism (no shared resource contention introduced by this change)
by which parallel execution would change the per-worker overhead percentage.

## Gating

`RewardWeights.subtract_twap_baseline: bool = False` -- same opt-in
convention as `zeta`/`eta_replace`. Defaults to inert; `reset()` skips the
shadow computation entirely when off (verified by test, not just by reading
the code -- `env._twap_shadow_terminal_is_bps is None` when the flag is
False). `info["implementation_shortfall"]` is never altered by this flag --
it always reports the real, un-adjusted execution outcome; only the scalar
reward returned from `step()` changes when the flag is on.

## Tests: `tests/test_twap_baseline_reward.py`

Same fixture discipline as `test_reward.py`, adapted for env-level physics:
rather than hand-simulating the matching engine inside the test (error-prone,
duplicates even more logic than the implementation already does), these
verify exact algebraic relationships between paired runs on the same seed --
deterministic and exactly assertable, self-checked against the *real*
`TWAPPolicy` as ground truth rather than a second hand-written approximation
of it.

- `test_subtract_twap_baseline_defaults_to_inert` -- `RewardWeights()` default
  is False; confirms `_twap_shadow_terminal_is_bps is None` after reset with
  default weights (not computed, not just unused).
- `test_subtract_twap_baseline_does_not_change_reported_is` -- same seed, same
  (trivial) policy, flag on vs off: `info["implementation_shortfall"]` is
  identical either way. Guards the "reward changes, reported outcome doesn't"
  invariant directly.
- `test_subtract_twap_baseline_arithmetic_matches_hand_derivation` -- since
  every non-terminal reward component is identical between flag-on/flag-off
  runs on the same seed and policy, `total_reward_on - total_reward_off`
  must equal exactly `kappa * twap_shadow_is_bps`. Verifies the subtraction
  wiring itself, independent of whether the shadow's own number is "correct."
- `test_subtract_twap_baseline_matches_real_twap_policy_exactly` -- runs the
  *real* `TWAPPolicy` through the *real* env with the flag on; since the real
  episode's own execution IS TWAP, the shadow's number must match the real
  run's own reported IS almost exactly. This is the test that actually
  guards against drift between the duplicated decision logic and the
  canonical policy, not code review alone.

That last test earned its place during implementation: it caught two real
bugs before either could reach committed code.

1. **Rounding.** `TWAPPolicy.act()` computes a continuous "ideal" size
   fraction each slice but the action space only has 5 discrete
   `SIZE_FRACTIONS` (0.2/0.4/0.6/0.8/1.0) -- the real system snaps to the
   nearest one before `step()` ever sizes an order. The first draft of the
   shadow used the continuous fraction directly. On a test scenario where
   the drawn `qty_total` happened to exceed the fixture's single-level book
   depth (so fills were genuinely partial, not saturating at 100%), this
   produced a ~0.02bps drift between the shadow and the real run --
   small, but real, and exactly what this test exists to catch.
2. **Decision/evolution ordering.** `TWAPPolicy.act()` is always called
   *before* `step()`'s own resting-order evolution for that tick runs, so
   its slice-accounting decision uses `env.qty_remaining`/`env._resting` as
   of the END of the *previous* tick, not the live, about-to-be-updated
   value. The first draft evolved-then-decided within the same loop
   iteration, using post-evolution state a tick early. This did not affect
   the specific fixture used here (a constant, unchanging book makes
   organic queue evolution a no-op regardless of ordering) but is a real
   discrepancy in general and is fixed to match the true sequencing
   exactly, with the reasoning recorded in the method's own comments.

Full suite after both fixes: `tests/test_twap_baseline_reward.py` 4/4 pass;
full project suite 104 passed (same 4 pre-existing, unrelated failures in
`test_bulk_backfill.py`/`test_l2_capture.py` as every prior round this
session -- confirmed untouched by this change).

## A/B test result (step 3): does the reward change improve the trained policy?

**Date:** 2026-08-22
**Design:** matched A/B, single seed per arm. Both arms warm-started (weights
only, VecNormalize fresh, step counter reset to 0) from the same canonical
checkpoint (`models/l3_executioner_v1.zip`, checksum `27afa91e...`, the
step-2,000,000 stand-in -- re-verified immediately before each launch), same
`--n-envs 8`, same `--total-timesteps 1000000`, run sequentially on the same
box. The ONLY difference between the two runs is `--subtract-twap-baseline`.

- **ARM A (control, flag off):** 21:39-23:09 HKT, ~90 min wall-clock, fps
  ~193-208 throughout. Saved to
  `models/l3_executioner_v1_twap_ab_armA_control.zip`.
- **ARM B (treatment, flag on):** 23:10-01:00 HKT, ~110 min wall-clock (the
  extra ~20 min is consistent with the measured ~48ms/reset TWAP-shadow
  overhead accumulating over ~1M/[ep_len] resets), fps ~152-208, trending
  down as episodes shortened over training (see below). Saved to
  `models/l3_executioner_v1_twap_ab_armB_treatment.zip`. Canonical checkpoint
  confirmed byte-for-byte untouched (checksum unchanged) after both runs.

**CORRECTION (added 2026-08-22, discovered while prepping a follow-up run):**
both arms trained with the r_queue queue-weighted term in the INVERTED
direction (EXPERIMENTAL 4 in reward.py's module docstring, commit `4d81a96`),
not the ORIGINAL direction implied above and elsewhere in this session's
reporting. This was not a deliberate choice for this A/B test -- it was
inline, unconditional code left uncommitted in the working tree since the
separate direction-inversion probe (~22:00 HKT 2026-08-20) and never
reverted, so every run launched afterward silently inherited it. v1's own
original training (the `27afa91e...` checkpoint both arms warm-started from)
DID use the original direction -- confirmed via file mtimes and training
logs, see commit `4d81a96`'s message for the full reconstruction.

**What this does and doesn't affect:** the ARM B vs ARM A comparison below
is NOT confounded by this -- both arms shared the identical (inverted)
r_queue formula, so it remains a clean, isolated test of
`subtract_twap_baseline` alone. What IS affected: the framing of "Arm A
reaches parity with TWAP, up from v1's 1.261" as "same reward config as v1,
just more training" is wrong -- it is "more training AND r_queue direction
flipped from original to inverted," not disentangled. Some unknown share of
Arm A's apparent improvement over v1 could be attributable to the r_queue
change rather than training budget alone. This was not tested cleanly by
anything run so far: the original direction-inversion probe compared
inverted-vs-TWAP and original-vs-TWAP separately (both n=50, neither
significant), never inverted-vs-original directly and never at n=500.

1M steps per arm was chosen over matching v1's own 2M-step run: this is a
probe of whether the reward change helps at all, not a from-scratch training
commitment, and 1M steps still gives multiple in-training eval firings to see
a trajectory, comfortably past the expected initial value-loss recalibration
window. Both arms' `value_loss` had settled to the same order of magnitude
(0.015-0.03) by step 1M with no sign of an unresolved recalibration in
progress -- so the step budget was not obviously too short for a fair read.

### Results at n=500

Same paired seeds (5,000,000-5,000,499) as every prior n=500 eval this
session; TWAP's per-episode numbers reused byte-for-byte from
`scripts/replace_value_probe_n500.py`'s output, not recomputed.

| | IS_total_bps mean | std | fill_ratio | vs TWAP (paired t / Wilcoxon) |
|---|---|---|---|---|
| TWAP | 0.889 | 4.353 | 0.994 | -- |
| **Arm A (control)** | **0.994** | 3.570 | 0.919 | diff +0.105bps, t p=0.534, W p=0.653 -- **not significant** |
| best-B (REPLACE probe, unrelated track) | 1.103 | -- | -- | (from `l3_replace_value_probe.md`, pooled for context) |
| v1 (2M-step, own checkpoint) | 1.261 | -- | 0.892 | (from earlier this session, pooled for context) |
| **Arm B (treatment)** | **1.341** | 2.405 | 0.990 | diff +0.452bps, t p=0.0092, W p=0.0140 -- **significantly worse** |

Pooled ordering at n=500: TWAP (0.889) < **Arm A (0.994)** < best-B (1.103) <
v1 (1.261) < **Arm B (1.341)**. Arm A -- plain continued training from the
v1 checkpoint, no reward change at all -- is now the best-performing RL
variant measured this session, closest to TWAP of any of them. Arm B is the
worst.

### The comparison that actually answers the question: Arm B vs Arm A, direct paired test

Arm-B-vs-TWAP and Arm-A-vs-TWAP separately can't distinguish "the reward
change helped" from "more training helped" -- both arms got 1M more steps
than v1's stand-in checkpoint had at warm-start. This is the test that
isolates the reward change specifically, both arms having received identical
training otherwise:

- **mean diff (B - A) = +0.347 bps** (B worse), std of the paired
  differences = 2.991 bps
- **paired t-test: t=2.595, p=0.0097** -- significant
- **Wilcoxon signed-rank: W=54561, p=0.0224** -- significant, agrees with
  the t-test on direction (unlike the v1-vs-TWAP case earlier this session,
  where the two tests disagreed)
- **Effect size:** Cohen's d_z (paired) = 0.116 (small); the mean difference
  is 7.98% of TWAP's own std -- comparable in magnitude to the v1-vs-TWAP
  effect size found earlier (~8.5% of TWAP's std), i.e. a real but modest
  effect, not a large one
- **Win/loss:** Arm A better (lower IS) in 271/500 episodes (54.2%), Arm B
  better in 226/500 (45.2%), 3 ties -- not a lopsided split

### Diagnosing the effect: broad shift or tail-driven?

Same diagnostic used for the earlier v1-vs-TWAP disagreement, applied here
even though both tests already agree, because it changes what the result
means:

- **Median diff (B - A) = 0.022 bps** -- essentially zero. The *typical*
  episode shows no meaningful difference between arms.
- **The worst 10 of 500 episodes for Arm B (2% of the sample) account for
  119.4 bps of the 173.7 bps total net difference (69%).** Gross positive
  (B worse) sum = 504.6 bps over 271 episodes; gross negative (B better) sum
  = -330.9 bps over 226 episodes -- both sides are large and mostly cancel;
  the net comes from a heavier right tail on Arm B's side, not a uniform
  shift.

So: Arm B is **not** uniformly worse than Arm A -- most episodes are at or
near parity. What moved the mean (and drove both tests to significance) is a
minority of episodes where Arm B does substantially worse. This matters for
interpreting the mechanism below.

### The variance-reduction mechanism did work, mechanically -- it just didn't help

Arm B's own outcome distribution has **lower** std (2.405 bps) than both Arm
A's (3.570 bps) and TWAP's own (4.353 bps) -- Levene's test confirms this
variance difference is highly significant (stat=27.48, p<0.0001). The
mechanism did exactly what `docs/reports/l3_twap_baseline_reward.md`'s design
section said it would do: reduce the variance of the terminal outcome the
critic has to predict. **That reduction did not translate into a better, or
even equal, mean outcome** -- the mean got significantly worse instead.

### A converged policy that looks behaviorally different, not just "same policy, cleaner"

The baseline-subtraction design is explicitly built on the premise that
subtracting a per-episode constant cannot change which policy is optimal --
only how much reward-signal variance the critic has to fight through to find
it. The behavioral numbers don't fully match a "same policy, found faster or
cleaner" story:

- **fill_ratio: Arm A 0.919 -> Arm B 0.990** -- Arm B completes far more of
  its target quantity
- **mean episode length: Arm A 1,572 ticks -> Arm B 811 ticks** -- Arm B
  finishes (or gives up) roughly twice as fast
- **terminated-early rate (order fully completed before horizon): Arm A
  384/500 (76.8%) -> Arm B 473/500 (94.6%)**

Read together with the tail-driven mean difference above, a plausible (not
proven) story: Arm B converged toward a more aggressive completion style --
finish the order fully and quickly -- which costs little extra on most
episodes but produces occasional expensive fills when that aggression meets
unfavorable conditions, and the occasional expensive tail outweighs the
frequent small gains from fuller/faster completion. This is offered as a
hypothesis for why the numbers look the way they do, not as a confirmed
causal mechanism -- distinguishing it from pure single-run training noise
would need a multi-seed replication (see caveat below).

### Caveat, stated plainly

This is a **single seed per arm**, as the task specified (a probe, not a
multi-seed study). A single matched pair cannot fully separate "the reward
mechanism is causally responsible" from "this particular stochastic training
run landed somewhere worse." What argues against pure noise: the effect
reaches significance on both a mean-sensitive test (paired t) and a
rank-sensitive test (Wilcoxon) that disagree when a result is fragile (as
they did for v1-vs-TWAP) and agree here; and the behavioral shift
(fill_ratio, episode length, termination rate) is large and directionally
coherent with the outcome shift, not just a noisy IS number moving on its
own. But it remains one training run per condition, and that should not be
overstated into more certainty than the design supports.

### Answer, plainly

**The TWAP-baseline (variance-reduction) reward did not improve the trained
policy over the matched control at this step budget -- it produced a policy
that is significantly worse, both against the control (Arm B vs Arm A:
p=0.0097 / p=0.0224) and against TWAP itself (p=0.0092 / p=0.0140), where the
control was statistically indistinguishable from TWAP.** Neither arm beats
TWAP: Arm A ties with it, Arm B loses to it. The reward change did achieve
its stated mechanical goal -- materially lower outcome variance (Levene
p<0.0001) -- but that did not carry through to better, or even equal,
execution quality, and the resulting policy looks behaviorally different
(fuller, faster, tail-costlier completions) rather than simply a
faster-converged version of the control's policy. This is a clean negative
result for the variance-reduction hypothesis as implemented and tested here,
reported as such rather than reframed around the one thing (variance) that
did move in the predicted direction.

### Not done this round

No multi-seed replication (would be needed to fully separate the reward
mechanism's effect from single-run training variance, per the caveat above).
No investigation of *why* Arm B's fill/episode-length behavior shifted
beyond the hypothesis offered above. No attempt to fix or iterate on the
reward formulation in this pass -- reported as a negative result for
direction, not patched in place.
