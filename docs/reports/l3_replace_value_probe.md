# Does CANCEL_AND_REPLACE actually help? A scripted-heuristic value probe

**Date:** 2026-08-21
**Script:** `scripts/replace_value_probe.py`
**Raw results:** `/tmp/replace_value_probe_results.json` (not committed -- regenerable by re-running the script; see Reproducibility)

## Why this report exists

Four rounds of reward engineering on the L3 track (r_stale, r_placement_stale,
the MARKET/REPLACE r_queue split, and the split's direction inversion) all
treated "CANCEL_AND_REPLACE usage is ~0%" as a problem to fix. None of them
established that higher REPLACE usage would actually improve implementation
shortfall (IS) in this environment. The direction-inversion probe was weak
evidence it might not (REPLACE usage nudged up by a statistically
indistinguishable amount, IS got numerically worse). This report answers the
prior question directly, with hand-written heuristic policies run through the
real environment -- no RL, no training, no GPU, no model loading anywhere in
this analysis.

**Bottom line: the evidence supports (b), and this has since been confirmed
at adequate statistical power (n=500; see "Higher-n follow-up" below).** No
REPLACE-based heuristic, given a genuinely fair and reasonably wide parameter
sweep, beats a fair comparison point (TWAP, or a comparably-filling
alternative) by a margin that survives scrutiny -- not even the single
best-of-18 configuration, selected post hoc, which should if anything be an
optimistic estimate. The near-0% REPLACE usage the trained agent converged to
looks like correct behavior on this data, not a training failure. **Update
(n=500):** the original n=50 result even had the WRONG SIGN -- best-B looked
numerically better than TWAP at n=50 (a selection artifact from screening 18
configs), but at n=500 it is numerically worse, still not significant
(p=0.10), and TWAP itself beats v1's trained RL policy's reported IS outright.
See "Higher-n follow-up" for the full account -- this is now a more strongly
supported (b), not merely a repeated one. See "Answering the actual question"
below for the original full reasoning and its scope.

## Step 0 -- Resolving the near-v1 action-distribution inconsistency

Before running anything new, a specific loose end from the last report needed
closing: the near-v1 stand-in checkpoint (restored after the prior probe's
checkpoint-overwrite incident) reproduces v1's own reported IS_total_bps and
fill_ratio exactly (1.245 / 0.918, bit-for-bit), but its CANCEL_AND_REPLACE
usage was reported at 0.298%, against v1's own reported 0.36%. For a
genuinely identical policy on identical seeds, both numbers cannot hold at
once.

**Resolution: these are not the identical policy.** They are two different,
though very close, checkpoints from the same training run:

- The "near-v1" stand-in is `models/l3_checkpoints/l3_ppo_2000000_steps.zip`
  -- v1's own periodic `CheckpointCallback` save at **step 2,000,000**.
- The true v1 (`973b2883...`, no longer recoverable) was v1's run's actual
  **final** save at **step 2,002,944** -- 2,944 additional training steps
  later, in the same run.

Both the 0.298% and 0.36% figures came from the same script
(`analyze_final_eval.py`), the same 50 seeds, and the same denominator (% of
all actions across all steps of all 50 episodes) -- this was checked
directly, not assumed, and it rules out a measurement-scope or
denominator-definition explanation. The only thing that differed between the
two runs of the script was which checkpoint file was loaded.

Why would 2,944 steps move one number but not the other? IS_total_bps and
fill_ratio are stable aggregate outcomes averaged over 50 full episodes --
they don't swing much on a further 0.15% increment of training near the end
of a long run. CANCEL_AND_REPLACE usage is a rare event (well under 0.4% of
all actions); a low-frequency count is far more sensitive to a small amount
of continued policy drift than a robust 50-episode aggregate is. A raw-count
two-proportion check on this specific pair (263/88141 vs. an estimated count
at v1's true rate) is not needed to see the pattern -- it is directly useful
context for this report's own result, in fact: **ordinary continued training,
with no reward change at all, moved the REPLACE usage rate by roughly the
same order of magnitude (0.298% -> 0.36%, over just 2,944 steps) as the
direction-inversion probe's reward change did (0.298% -> 0.336%, over
500,000 steps).** A reward change that needed ~170x more training steps to
produce a comparable-or-smaller shift than ordinary drift produces for free
is independent corroboration that the probe's own result was very plausibly
noise, not signal -- consistent with, and reinforcing, that report's own
conclusion.

This is not a concern for the checkpoint's validity as a v1 stand-in (already
verified via the exact IS/fill match) -- it is a genuine, small, expected
divergence between two checkpoints 2,944 steps apart, not a bug.

## Reward-independence, verified from source

This task's hard boundary states that reward config shouldn't matter to
IS/fill_ratio results, and asks that this be confirmed by reading the code
rather than assumed. Confirmed:

- `compute_implementation_shortfall()`'s signature (`src/envs/reward.py`)
  takes `side, fills, qty_total, arrival_price, terminal_mid_price,
  fee_bps_per_fill` -- no `reward_weights` argument at all.
- `self.reward_weights` appears in exactly two places in
  `LOBExecutionEnv.step()`: passed into `step_reward()` to compute the
  scalar reward `r`, and `r += -self.reward_weights.kappa *
  terminal_is.is_total_bps` -- both only ever *add to* the reward value `r`.
  Neither touches `terminal_is` itself (the `ImplementationShortfall` object
  placed in `info["implementation_shortfall"]`), which is computed via
  `compute_implementation_shortfall()` earlier in the same method, before
  either of these lines runs.
- `matching_engine.py` (fills, queue mechanics, crossing detection) has zero
  references to "reward" anywhere -- grepped directly, no matches.

IS_total_bps and fill_ratio are therefore fully independent of
`RewardWeights`/`reward.py`'s current state. **Working-tree state at the time
this was run, disclosed per instruction though it does not affect these
results:** `src/envs/reward.py` still carries the uncommitted r_queue
direction-inversion change from the prior probe (`gamma * (1 -
queue_ahead/queue_at_level)` in both cancel branches) -- confirmed present
via `grep` immediately before running. It was left as-is rather than
reverted, since reverting would have touched a file this task's hard
boundaries say not to modify, and since it provably cannot affect the
result.

## Methodology

Reused, not rebuilt: `LOBExecutionEnv`, `load_split("val")`, `TWAPPolicy` and
`run_episode()` from `scripts/phase2a_sanity_suite.py` -- the same 50 paired
seeds (5,000,000..5,000,049) and the same held-out val population
(2025-07-16..2025-08-02, 18 days) as the milestone report and every
reproduction script since. No model loading, no GPU, no training loop
anywhere in `scripts/replace_value_probe.py`.

Two new hand-written policy classes:

- **A. `PassivePolicy(offset)`** -- place one LIMIT at a fixed `offset`, full
  remaining size (no repricing, so no reason to hold size back), then HOLD
  every subsequent tick until filled or horizon.
- **B. `ReplaceActivePolicy(initial_offset, staleness_n, step)`** -- place a
  LIMIT at `initial_offset`, full size. Every tick, if an order is resting
  and unfilled for >= `staleness_n` ticks, CANCEL_AND_REPLACE at a more
  aggressive offset (`current_offset += step`, capped at the
  guaranteed-crossing/marketable end of the range). If nothing is resting and
  quantity remains (first placement, or a prior crossing replace only
  partially filled), place fresh at the current offset.

`offset` follows `_place_limit()`'s own convention, verified from source
before use, not assumed: for both sides, a *higher* `offset` value is *more
aggressive* (`price = best_bid + offset*TICK` for buy, `best_ask -
offset*TICK` for sell; `price_offset_idx = offset + 5`), so both policies can
be written side-agnostically.

**Grid, kept small enough to run quickly but wide enough to give B a
genuinely fair shot:**
- A: `offset` in {-5, -4, -3, -2, -1, 0, +1} (7 configurations)
- B: `initial_offset` in {-5, -3, -1} x `staleness_n` in {20, 100, 300} ticks
  x `step` in {1, 2} (18 configurations)
- C: TWAP (`n_slices=10`), run once as the existing baseline.

25 configurations x 50 episodes = 1,250 heuristic-policy episodes, plus 50
TWAP episodes. Total wall-clock: ~42 minutes (dominated by episode length --
poorly-filling PASSIVE configurations run close to the full 3,000-tick
horizon on almost every seed; REPLACE/TWAP configurations that reliably fill
terminate much earlier and are correspondingly cheaper per episode).

## Results

### A: PASSIVE(offset) sweep

| offset | IS_total_bps mean | std | fill_ratio |
|---|---|---|---|
| -5 | 0.406 | 10.497 | 0.048 |
| -4 | 0.042 | 10.930 | 0.035 |
| -3 | 0.267 | 10.574 | 0.055 |
| -2 | 0.297 | 10.402 | 0.063 |
| -1 | 0.097 | 10.711 | 0.119 |
| **+0** | **-0.182** | 7.800 | **0.404** |
| +1 | -0.056 | 7.474 | 0.364 |

**Critical caveat, not optional context:** every PASSIVE configuration fills
a small minority of the order -- at best 40.4% (offset=0), at worst 3.5%
(offset=-4). This is not a footnote; it directly undermines reading
"offset=0" as a straightforwardly "best" passive strategy. A policy that
fills 40% of an order and marks the unfilled 60% at the terminal mid-price
is not doing a good job at the actual task, even when the IS arithmetic
happens to look favorable -- this project has already diagnosed exactly this
artifact once before, for the pre-retrain L3 checkpoint's 0.2015 fill_ratio
in the milestone report ("its favorable IS number likely reflects avoided
price impact from incomplete trades, not real execution skill"). The same
logic applies here, more starkly: fill_ratio 0.048-0.119 for the deeply
passive offsets, up to a still-weak 0.40 at the best. The pattern in the
std column is corroborating, not incidental -- the low-fill configurations
carry roughly double TWAP's/B's variance (10.4-10.9 vs. TWAP's 5.0, vs. B's
0.8-6.5), consistent with a metric dominated by the terminal-price roll of
the dice on a large unfilled remainder, not by stable execution cost.

### B: REPLACE-ACTIVE(initial_offset, staleness_n, step) sweep

| config | IS_total_bps mean | std | fill_ratio |
|---|---|---|---|
| init=-5, N=20, step=1 | 1.251 | 2.869 | 1.000 |
| init=-5, N=20, step=2 | 1.362 | 2.584 | 1.000 |
| init=-5, N=100, step=1 | **0.700** | 4.289 | 1.000 |
| init=-5, N=100, step=2 | 0.788 | 4.292 | 1.000 |
| init=-5, N=300, step=1 | 1.582 | 6.501 | 1.000 |
| init=-5, N=300, step=2 | 0.992 | 4.993 | 1.000 |
| init=-3, N=20, step=1 | 1.281 | 2.681 | 1.000 |
| init=-3, N=20, step=2 | 1.252 | 2.308 | 1.000 |
| init=-3, N=100, step=1 | 0.764 | 4.235 | 1.000 |
| init=-3, N=100, step=2 | 1.056 | 3.724 | 1.000 |
| init=-3, N=300, step=1 | 0.915 | 5.232 | 1.000 |
| init=-3, N=300, step=2 | 0.842 | 4.297 | 1.000 |
| init=-1, N=20, step=1 | 1.231 | 2.254 | 1.000 |
| init=-1, N=20, step=2 | 1.138 | 0.831 | 1.000 |
| init=-1, N=100, step=1 | 1.014 | 3.452 | 1.000 |
| init=-1, N=100, step=2 | 1.205 | 2.745 | 1.000 |
| init=-1, N=300, step=1 | 0.836 | 4.104 | 1.000 |
| init=-1, N=300, step=2 | 0.885 | 4.380 | 1.000 |

Every B configuration fills the order completely (fill_ratio=1.000) -- as
expected, since the escalation eventually reaches a guaranteed-crossing
offset well inside the 3,000-tick horizon regardless of parameters. This
makes B directly, fairly comparable to TWAP (fill_ratio=0.9945) in a way no
A configuration is.

### C: TWAP baseline

IS_total_bps mean=1.182 (std 5.025), fill_ratio=0.9945 -- unchanged from
every prior measurement this session, confirming the eval harness is behaving
consistently.

### Paired comparisons (n=50, same seeds every arm)

| comparison | mean diff | paired t | Wilcoxon |
|---|---|---|---|
| best-B vs best-A | +0.882bps | t=0.805, p=0.425 | W=470, p=0.107 |
| best-A vs TWAP | -1.364bps | t=-1.682, p=0.099 | W=462, p=0.091 |
| best-B vs TWAP | -0.482bps | t=-0.919, p=0.363 | W=600, p=0.723 |

(Negative mean diff = the first-named arm has lower/better IS.)

**Read the middle row with the fill_ratio caveat above firmly in mind.**
"best-A vs TWAP" compares a 40%-filled policy against a 99%-filled one; its
borderline p-values (0.099 / 0.091) are exactly the kind of number that
would be easy to mis-read as "passive nearly beats TWAP." It almost
certainly does not reflect genuine execution quality -- it reflects the same
partial-fill/terminal-price-roll artifact discussed above. The clean,
apples-to-apples comparison is the bottom row, best-B vs TWAP (both ~100%
filled): not significant by either test, and not close (Wilcoxon p=0.723).

**Multiple-comparisons exposure, stated plainly per instruction:** 18 B
configurations were swept and the single best one selected post hoc for the
headline comparisons above. The naive p-values on that winner are
optimistic/exploratory, not from a pre-registered single test -- reading
best-b_vs_best-a's p=0.425 (or even best-B vs TWAP's p=0.363) as "the" p-value
for "does REPLACE help" overstates the rigor on offer. A Bonferroni-corrected
threshold for the B-sweep dimension is alpha = 0.05/18 = 0.00278; none of the
three headline comparisons come close to clearing even the *uncorrected*
0.05 bar, so the correction does not change the reading here -- it would only
matter if a result had looked marginally significant before correction, and
none did.

## Answering the actual question

**(b): REPLACE is not shown to be valuable in this environment on this
data.** No B configuration -- out of 18 spanning patient-to-impatient
staleness thresholds, passive-to-near-touch starting offsets, and two
escalation speeds -- beats a fair comparison point by a margin that survives
scrutiny. The single closest result (best-B vs TWAP, -0.48bps) is the
*optimistic, post hoc selected* number and still lands nowhere near
significance (p=0.36 / p=0.72). The apparent "PASSIVE nearly beats TWAP"
result is a measurement artifact of a 40%-fill policy, not a genuine
execution-quality finding, and does not change this reading -- if anything it
underscores that a policy needs to actually complete the order to be
evaluated meaningfully by this metric at all, and passive placement alone
mostly fails to do that within the horizon.

This is independently corroborated by Step 0's finding above: ordinary
continued training with the *unmodified* reward moved REPLACE usage by a
comparable amount, in a fraction of the steps, to what the direction-inversion
probe's reward change produced. Two separate pieces of evidence -- this
scripted-heuristic sweep, and the training-dynamics comparison -- point the
same way.

**Scope of this conclusion, stated precisely rather than left implicit:**
this tests one specific, if reasonably diverse, family of REPLACE strategies
-- linear offset escalation on a fixed staleness timer. It does not rule out
a fundamentally different repricing design (order-book-depth-aware
repricing, urgency-scaled step sizes, size-varying replaces) performing
differently. It also does not test REPLACE in combination with other order
types in ways an RL policy might discover that a hand-written heuristic
would not think to try. Within the family tested, though, REPLACE showed no
edge over either TWAP or a comparably-executing baseline, across a genuinely
wide sweep, which is meaningful negative evidence, not merely "we tried one
thing and it didn't work."

**Recommendation, not a decision made here:** the REPLACE-usage line of
inquiry (further reward engineering specifically aimed at increasing the
trained agent's near-0% REPLACE rate) does not have supporting evidence to
continue on. Effort would be better directed at the actual objective --
beating TWAP on IS -- which this probe also did not achieve with any hand-
written heuristic, PASSIVE or REPLACE-ACTIVE, at n=50. Whether that means
looking at fundamentally different execution strategies, revisiting whether
this environment's IS metric rewards behavior consistent with good execution
at all, or something else, is a direction question for whoever weighs it
next.

## Higher-n follow-up (n=500) -- what changed, what held

The comparisons above were run at n=50, which was never sized for statistical
power in the first place -- it was inherited from the milestone report's own
eval cadence. This section checks that directly and re-runs the single most
relevant comparison at adequate power.

### Power arithmetic, verified independently

At n=50, best-B vs TWAP: mean diff=-0.482bps, std_diff=3.71bps (paired
differences, ddof=1). Achieved power to detect this observed effect at n=50:
**14.7%** -- not "weak evidence," essentially no ability to distinguish the
observed effect from zero. Required n for 80% power to detect an effect of
this magnitude, given this std: **~465** (normal approximation,
alpha=0.05 two-sided), confirmed via the noncentral-t achieved-power
function directly, not just the closed-form approximation. Chosen run size:
**n=500** (~83% power for the originally-observed effect).

### Config selection: PASSIVE dropped, not substituted

The plan called for a "best fully-filling PASSIVE config" alongside best-B
and TWAP. Checking this before committing to n=500 revealed it does not
exist: extending the A sweep to offsets +2 through +5 (n=50 each, cheap
since crossing orders resolve fast) reproduced offset=+1's exact numbers
(IS=-0.056, fill=0.364) at every offset tested. This is not a coincidence --
once an offset crosses (offset >= +1 here, spread = 1 tick), `_place_limit()`
routes it through `walk_market_fill()` against whatever depth is visible on
that single tick; the price beyond the crossing threshold no longer matters,
only the size does, and the unfilled remainder is discarded (crossing prices
never create a resting order, so PASSIVE -- which only ever places once --
gets no second attempt at it). offset=0 (resting exactly at best_bid,
non-crossing) does slightly better, 40.4%, precisely because it keeps the
full 3,000-tick horizon to accumulate fills organically instead of spending
its one shot against a single tick's snapshot of depth. **40.4% is therefore
a structural ceiling for "place once, then only HOLD," not a point an
under-sized sweep missed.** No configuration in this policy family can be
made fill-comparable to TWAP/B, so no PASSIVE arm was run at n=500 --
substituting the 40%-fill config anyway would have repeated the exact
apples-to-oranges comparison this report already flagged as unreliable,
just at a higher n. `scripts/replace_value_probe_n500.py` runs only the two
configs this reasoning actually supports testing: best-B and TWAP.

### Results

| | n=50 (original) | n=500 (this follow-up) |
|---|---|---|
| best-B IS_total_bps | 0.700 | 1.103 |
| TWAP IS_total_bps | 1.182 | 0.889 |
| mean diff (B - TWAP) | -0.482bps | **+0.214bps** |
| paired t | t=-0.919, p=0.363 | t=1.643, **p=0.101** |
| Wilcoxon | W=600, p=0.723 | W=58394, p=0.191 |

**The sign flipped.** At n=50, best-B looked numerically better than TWAP --
this was the number that motivated running a properly-powered follow-up in
the first place. At n=500, best-B is numerically *worse* than TWAP. Neither
is statistically significant, but the reversal itself is the more important
finding: it is closer to a textbook illustration of post-hoc-selection bias
than to a threshold-crossing result. Best-B was the single best performer
selected out of 18 configurations screened at n=50; regression to the mean
at proper power is exactly what should be expected of a screening winner,
and that is what happened here, not a subtle effect that needed a larger
sample to resolve in the *predicted* direction.

**Significance vs. practical effect size, stated separately as required:**
p=0.101 is not significant at the conventional 0.05 threshold, though it is
closer than the n=50 result was. Read the effect size on its own terms
regardless of the p-value: +0.214bps is small in absolute terms -- roughly
4% of TWAP's own std (4.35bps) -- and this project's own established
significance framework (used throughout this whole L3 track) would not
treat an effect this size as economically meaningful even if it had cleared
significance. At n=500 (std_diff=2.91bps here, tighter than the n=50
estimate), the minimum effect detectable at 80% power is ~0.365bps -- the
observed +0.214bps sits below even that, meaning n=500 gives good power to
rule out anything *larger* than roughly a third of a basis point, and finds
nothing bigger than that. This is a well-powered null, not an underpowered
maybe.

**Comparison against v1's trained RL policy, stated plainly, not buried:**
v1's reported IS_total_bps is **1.245** (n=50, from the milestone report).
At their respective sample sizes, **both TWAP (0.889 at n=500) and even the
scripted, hand-tuned REPLACE-ACTIVE heuristic (1.103 at n=500) come in lower
(better) than v1's trained policy's own reported number.** This is a
descriptive comparison of point estimates, not a new formal paired test --
v1's 1.245 is an n=50 estimate with its own uncertainty (std=4.74 per the
milestone report) that was not itself re-run at n=500, and no per-episode
v1 data exists at n=500 to pair against these results directly. With that
caveat stated, not hidden: a simple fixed-schedule baseline (TWAP) and an
untrained, hand-written heuristic both nominally outperforming a policy that
went through 2,000,000 steps of RL training is a real finding about the RL
setup, not a footnote to the REPLACE question. It does not, on its own,
prove the trained policy is worse in a statistically rigorous sense (that
would need its own properly-powered, paired follow-up, not attempted here)
-- but it is not a result to let pass without comment either, and it raises
the same question the original milestone report's "near-parity with TWAP"
finding already raised: whether this environment's training setup is
producing genuine execution skill, or converging to something that merely
avoids doing obviously worse.

### What changed, what held

- **Held:** the answer is (b) -- REPLACE is not shown to be valuable in this
  environment on this data. This conclusion is now *more* strongly supported
  than at n=50, not merely repeated at higher confidence -- the specific
  number that made REPLACE look promising at n=50 did not survive proper
  power.
- **Changed:** the direction of the point estimate. At n=50, best-B nominally
  beat TWAP by 0.48bps; at n=500, it nominally loses by 0.21bps. Neither
  point estimate should be treated as "the" effect -- the honest summary is
  that proper power finds no distinguishable difference, in either direction,
  larger than a few tenths of a basis point.
- **New:** the explicit comparison against v1's trained policy, which did not
  appear in the original version of this report at all. Both TWAP and best-B
  nominally beat it. Flagged plainly above, not resolved here.

## Fixed since the original version of this report

`train_l3.py`'s hardcoded final-save path -- flagged below in its original
form when this report was first written -- has since been fixed in a
separate commit: the final save now refuses to overwrite an existing
`models/l3_executioner_v1.zip` / `l3_vecnormalize.pkl` unless
`--overwrite-canonical` is passed explicitly, redirecting to a run-tagged
path otherwise. While implementing that fix, the same hazard was found to
have already independently bitten the periodic `CheckpointCallback` saves
too (not just the final save) -- see `docs/TRACK_STATUS.md` for the scope of
what was silently lost there. Original flagged text kept below as the
historical record of what prompted the fix.

## Separately flagged, not part of this task (original text, now resolved -- see above)

`train_l3.py`'s final save writes to a hardcoded path
(`models/l3_executioner_v1.zip` / `l3_vecnormalize.pkl`) regardless of
whether a run is a full commitment or a bounded probe -- this caused the
prior probe's checkpoint-overwrite incident (see the prior report) and will
recur on any future run that is allowed to reach its own completion. Needs a
run-tagged output path or a refuse-to-overwrite guard before any further
training run completes on this box. Noted here per instruction; not fixed as
part of this task (touching `train_l3.py` was out of scope for this probe's
hard boundaries).

## Reproducibility

`scripts/replace_value_probe.py` is deterministic given the same checkpoint
state of the val split and the same seeds (no model/policy randomness
involved -- both heuristic policies are pure functions of environment state,
and `TWAPPolicy` is likewise deterministic). Re-running it reproduces this
report's numbers exactly. Raw per-episode results were saved to
`/tmp/replace_value_probe_results.json` (scratch, not committed -- easily
regenerated, and not needed to trust this report's aggregates, which were
computed directly by the same script run that produced this write-up).

`scripts/replace_value_probe_n500.py` is the adequate-power follow-up --
same determinism properties, reuses `run_config`/`paired_report` and the
policy classes from `replace_value_probe.py` directly rather than
duplicating them. Raw results at `/tmp/replace_value_probe_n500_results.json`
(scratch, same regenerability note as above).
