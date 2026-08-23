# Arm A budget extension: does more training push past parity into a genuine edge?

**Date:** 2026-08-22 to 2026-08-23
**Status:** Complete. Answer: no -- this run does not establish a genuine edge over
TWAP, and does not establish that more budget helps at all over Arm A's own result.

## Context and design

Arm A (the TWAP-baseline A/B test's control, `subtract_twap_baseline=False`) reached
parity with TWAP at n=500 after 1M steps of continued training from v1
(0.994 vs TWAP's 0.889, not significant, p=0.534/0.653 -- see
`docs/reports/l3_twap_baseline_reward.md`). This run tests whether more budget
pushes it past parity into a real edge: 2,000,000 additional steps, warm-started
(weights only) from Arm A's own final checkpoint
(`models/l3_executioner_v1_twap_ab_armA_control.zip`, sha256 `a5443e2a...`), same
reward config Arm A actually trained under (see the reward-config note below --
this matters), same `--n-envs 8`, run-tagged output
(`models/l3_executioner_v1_armA_control_ext2M.zip`), canonical checkpoint confirmed
untouched throughout.

**Reward config note, corrected before this run launched:** the instruction for this
run originally assumed Arm A trained under the *original* r_queue direction. It
didn't -- both Arm A and Arm B of the TWAP-baseline A/B test silently trained under
the *inverted*, empirically-unvalidated r_queue direction (inline, uncommitted code
left over from an earlier, unrelated probe and never reverted -- see commit
`4d81a96` for the full reconstruction, and the correction added to
`docs/reports/l3_twap_baseline_reward.md` and `TRACK_STATUS.md`). This run
deliberately continues with that same inverted direction, to keep the test isolated
to budget alone -- switching to the original direction here would have conflated
two changes (more budget + reward formula) into one run.

**Budget:** 2M steps was used as given. Reasoning for not overriding it: Arm A's own
1M-step run showed real, non-monotonic volatility (n=50 in-training eval swung
1.05->1.46->1.57->0.75 across its own four checkpoints) rather than a clean trend,
so a longer budget with more eval firings (8 vs Arm A's 4) was needed to have any
chance of distinguishing a real trend from continued noise -- a shorter budget would
not have resolved the ambiguity any better.

## Training trajectory (n=50, paired seeds 5,000,000-5,000,049, TWAP=1.1819)

| step (this run) | L3 IS_total_bps |
|---|---|
| 250,000 | 0.8557 |
| 500,000 | 0.6863 (best) |
| 750,000 | 0.9463 |
| 1,000,000 (halfway) | 0.9891 |
| 1,250,000 | 1.3809 (worst -- crosses worse than TWAP) |
| 1,500,000 | 1.1792 |
| 1,750,000 | 0.9627 |
| 2,000,000 (final) | 0.9458 |

**Shape: not a steady improvement, not a clean plateau -- ongoing volatility with no
visible convergence.** The best point (0.6863) came early, at only 500k steps into
this run; the trajectory then degraded for two consecutive firings, crossed worse
than TWAP at 1.25M, and partially recovered without ever bettering the 500k point
again. The halfway check-in (see below) flagged the post-500k downward trend but
judged it within the noise band already established by Arm A's own run rather than
a clear degradation signal, and continued to the full budget rather than stopping
early -- the subsequent 1.25M dip-then-recovery pattern validates that call
(a premature stop at the halfway point would have missed the eventual partial
recovery, though as the n=500 result below shows, that recovery did not translate
into a genuinely better policy either). There is no visible sign of convergence
toward a stable optimum anywhere in this trajectory -- it reads as a policy
continuing to wander within roughly the same performance band Arm A itself
occupied, not as one converging toward a new, better equilibrium. More budget
beyond 2M would need a strong reason to expect a different pattern to emerge, which
nothing in this trajectory provides.

**Note on the halfway check-in call:** at the 1M-step mark, IS was trending worse
for two consecutive readings (0.69 -> 0.95 -> 0.99) but every reading through
halfway, including the worst, still beat TWAP -- unlike Arm A's own run, which had
already crossed worse-than-TWAP twice by its own halfway point. That comparison
was the basis for continuing rather than stopping. In hindsight the trajectory did
go on to cross worse-than-TWAP shortly after (1.25M), so the "not yet a clear
signal" read was correct in the narrow sense (it wasn't clear yet) but the eventual
signal, once budget did reveal one, was not favorable either.

## n=500 evaluation

Same paired seeds (5,000,000-5,000,499) as every prior n=500 eval this session;
TWAP's numbers reused byte-for-byte, not recomputed.

| | IS_total_bps mean | std | fill_ratio | vs TWAP |
|---|---|---|---|---|
| TWAP | 0.889 | 4.353 | 0.994 | -- |
| Arm A (1M steps) | 0.994 | 3.570 | 0.919 | ties (p=0.534/0.653) |
| **This run (Arm A + 2M more)** | **1.237** | 2.039 | **0.9998** | diff +0.347bps, t p=0.0336, W p=0.0440 -- **significantly worse** |

Taken alone, this run looks like a step backward from Arm A: it is now
significantly worse than TWAP, where Arm A tied. **But this comparison, on its
own, cannot separate "more training hurt" from "this happens to be where a noisy
single-seed trajectory landed" -- exactly the same limitation the A/B test's
Arm-B-vs-TWAP-alone comparison had. The comparison that actually isolates the
effect of the additional budget is the direct paired test against Arm A itself.**

## The decisive test: this run vs Arm A, direct paired comparison

- **mean diff (this run - Arm A) = +0.243 bps** (this run nominally worse), std of
  paired differences = 3.217 bps
- **paired t-test: t=1.686, p=0.0924** -- NOT significant
- **Wilcoxon signed-rank: W=58748, p=0.2304** -- NOT significant, and
  considerably weaker than the t-test (the same pattern of test disagreement seen
  for v1-vs-TWAP earlier this session, again a sign of a fragile, not-robust
  effect)
- **Effect size:** Cohen's d_z = 0.076 (very small); the mean difference is 5.58%
  of TWAP's own std -- smaller than the Arm-B-vs-Arm-A effect size found in the
  TWAP-baseline test (~8%)
- **Win/loss:** this run better in 245/500 episodes (49.0%), Arm A better in
  255/500 (51.0%) -- as close to a coin flip as this session has produced
- **Median diff = 0.0002 bps** -- effectively exactly zero. The typical episode
  shows no difference at all between the two checkpoints.
- **Tail concentration is extreme:** the worst 10 of 500 episodes (2%) account for
  117.9 of the 121.4 total net-bps difference -- **97%** of the entire effect.
  Essentially all of the "this run is worse" signal comes from a handful of
  episodes; the other 490 are indistinguishable between the two checkpoints.

**Conclusion of the direct test: no statistically reliable evidence that the
additional 2M steps changed the policy's quality, in either direction.** The
significant-looking loss against TWAP in isolation does not survive the more
specific test against Arm A -- which is exactly why that direct test was the one
that mattered, per the task's own framing.

## What did change: fill ratio and variance, again not translating to a mean edge

- **fill_ratio: Arm A 0.919 -> this run 0.9998** -- essentially every order now
  fills completely, a substantial behavioral shift
- **outcome variance: Arm A std=3.570 -> this run std=2.039** (vs TWAP's own
  4.353) -- Levene's test confirms this is highly significant (p<0.0001)
- **episode length: nearly unchanged** (Arm A 1572.1 -> this run 1598.0 ticks)

This is the same pattern observed for Arm B in the TWAP-baseline test: more
training (here, more steps; there, the variance-reduction reward) pushed fill
ratio up and outcome variance down, without producing a mean improvement over
either TWAP or the shorter-trained control. Read together with the extreme
tail-concentration above, the most defensible interpretation is that additional
training mostly refines completion behavior (filling more reliably, more
consistently) rather than execution-price quality, and the handful of tail
episodes driving the (non-significant) net difference are not obviously a
structural improvement or regression -- just where this particular trajectory's
noise landed.

## Answer, plainly

**This run does not establish that the policy beats TWAP -- it does not, and by
its own isolated comparison is nominally, significantly worse.** More importantly,
**it does not establish that more training budget helps at all**: the test that
actually isolates the budget's effect (this run vs Arm A directly) found no
significant difference in either direction, with a very small effect size and a
result driven almost entirely by 2% of episodes. The honest summary is a null
result on the budget question specifically, layered under an apparent-but-
statistically-fragile decline against TWAP that does not hold up under the more
specific comparison.

**What this does NOT test, stated explicitly per instruction:** this is a single
extended trajectory of the same seed Arm A already used, not an independent
replication. It says nothing about whether Arm A's own parity-with-TWAP result
was itself seed luck -- that question remains exactly as open as it was before
this run, and would need a separate run from a different seed to address. Since
this run did not produce a genuine edge over TWAP, the specific follow-up of
"replicate the edge with a different seed" does not apply here -- but the more
general question of whether ANY of this lineage's results (v1, Arm A, this
extension) would replicate under a different seed remains untested and is a
reasonable next step if this direction is pursued further, not run here.

## Not done this round

No multi-seed replication (as above). No attempt to disentangle the r_queue
inversion's own contribution to any of these numbers -- that remains a separate,
still-untested question (see the correction in `l3_twap_baseline_reward.md`). No
further budget extension attempted -- the trajectory shape gives no positive
reason to expect one would resolve the ambiguity differently.
