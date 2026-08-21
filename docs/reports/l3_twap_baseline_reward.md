# TWAP-baseline (variance-reduction) reward: implementation

**Date:** 2026-08-21
**Status:** Implemented, tested, gated OFF by default. No training run launched
this pass -- implementation and design review only, per instruction.

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

## Not done this pass

No training run launched, per explicit instruction -- this is implementation
and design review, not a probe result. The actual test of whether this
reward change helps training is a separate, future GPU run once this
implementation is reviewed.
