# L2 Gamma Ablation: n=500 Checkpoint Evaluation

Follow-up to `docs/reports/l2v3_gamma_ablation_training_run_report.md`, which found that
`gamma=0.983` kept the SAC critic numerically stable across the entire 1,600,000-step budget —
unlike `l2v1` and `l2v2`, both trained at `gamma=0.995`, which diverged. This round asks the
question that report left open: does the stable critic translate into a better held-out policy,
or does it just converge to the same place more calmly? Test split untouched throughout.

## 1. Summary

**It converges to the same place.** `l2v3`'s final checkpoint (step 1,599,936, never diverged)
scores IS_total_bps=1.169 against TWAP-passthrough's 1.024 — nominally *worse*, not better, and
not statistically distinguishable from it (paired t-test p=0.280, Wilcoxon p=0.118; the
pre-registered bar requires both significant AND the same direction as an improvement, so this
fails outright, unlike some earlier checkpoints that at least cleared one test). The direct paired
comparison against `l2v2`'s pre-divergence checkpoint (same reward, same seeds, `gamma=0.995`)
shows no significant difference either (mean diff=-0.053bps, d_z=-0.024, t_p=0.592, w_p=0.912) —
`l2v2`'s pre-divergence checkpoint is nominally slightly ahead, but the gap is indistinguishable
from noise.

**Answering this round's central question directly**: a numerically stable critic (via lower
gamma) does not by itself produce a measurably better policy at n=500. The two mechanisms — a
critic that stays well-conditioned throughout training vs. one that has to be caught before it
diverges — appear to land at statistically the same held-out execution quality here. This doesn't
mean gamma=0.983's stability is worthless (a stable critic removes the checkpoint-selection
problem entirely — every checkpoint is usable, not just a pre-divergence one — which has its own
practical value), but it is not, on this evidence, a fix for the underlying gap to TWAP-passthrough.

Because the pre-registered bar was not cleared, **the full diagnostic battery was not run** —
per instruction, that step is conditional on clearing the bar, and running it on a checkpoint that
already fails the primary comparison would not change the verdict, only add more tests to a result
that already has a clear answer.

## 2. n=500 result

Same methodology as every prior round (`scripts/eval_l2_diagnostics.py --split val`, unmodified
apart from this session's own thread-capping fix — see Section 4): 500 paired episodes, seeds
5,000,000-5,000,499, val split (2025-07-16 to 2025-08-02, 18 days), against TWAP-passthrough
(frozen L3, unsteered) and Pure TWAP (base env, no L3/L2 at all).

| checkpoint | step | critic_loss at save | IS_bps | vs TWAP-passthrough (1.024) | vs Pure TWAP (0.889) | fill_ratio |
|---|---|---|---|---|---|---|
| l2v3 final | 1,599,936 | 0.059 (never diverged) | 1.169 | d_z=0.048, t_p=0.280, w_p=0.118 | d_z=0.072, t_p=0.109, w_p=0.197 | 0.916 |

For reference, prior round's table (unchanged, reproduced here for direct comparison):

| checkpoint | step | critic_loss at save | IS_bps | vs TWAP-passthrough (1.024) | vs Pure TWAP (0.889) | fill_ratio |
|---|---|---|---|---|---|---|
| l2v2 final | 1,999,992 | ~10,800 | 1.227 | d_z=0.060, t_p=0.179, **w_p=0.0045** | d_z=0.083, t_p=0.066, w_p=0.064 | 0.920 |
| l2v2 pre-divergence | 1,599,936 | ~1-2 | 1.117 | d_z=0.032, t_p=0.468, w_p=0.295 | d_z=0.061, t_p=0.175, w_p=0.218 | 0.924 |
| l2v1 mid-run | 499,980 | ~26 | 1.177 | d_z=0.049, t_p=0.270, w_p=0.110 | d_z=0.075, t_p=0.094, w_p=0.133 | 0.914 |

`l2v3` final sits between `l2v2` pre-divergence (1.117, still the best of everything tested) and
`l2v1` mid-run (1.177) — closer to the worse end, not the better one. No checkpoint across either
round clears the pre-registered bar.

## 3. Direct paired comparison: l2v3 final vs. l2v2 pre-divergence

The question this round exists to answer: same reward (`potential_is_shaping`), same 500 paired
seeds, only `gamma` differs (0.983 vs. 0.995) and which checkpoint was evaluated (final vs.
pre-divergence, since `l2v3` never needed to distinguish the two).

| comparison | mean diff (b-a) | d_z | t_p | w_p |
|---|---|---|---|---|
| l2v2_predivergence vs l2v3_final | -0.0526bps | -0.0240 | 0.5922 | 0.9117 |

Negative mean diff means `l2v2`'s pre-divergence checkpoint scores nominally lower (better) than
`l2v3`'s final checkpoint, by a twentieth of a basis point — an effect size (d_z=-0.024) an order
of magnitude below even the previously-flagged "practically zero" d_z=0.076 result from earlier
in this project. Neither test comes close to significance (p=0.59, p=0.91). **No detectable
difference between the two.**

## 4. Thread-capping fix: confirmed working in production, not just the short test

This round's eval ran through the fixed `scripts/eval_l2_diagnostics.py` (commit `4d5a544`,
previous round) rather than relying on external `OMP_NUM_THREADS=1` etc. env vars at launch time.
Confirmed via a fresh `ps` check ~15s into the run: `1594982 99.8% CPU` — essentially
single-threaded, versus the 1,353% CPU measured pre-fix. Total wall-clock: 2,458s (40.97 minutes,
`START_EPOCH=1788012138` to `END_EPOCH=1788014596`) for the full 500-episode, 3-arm run — in line
with (marginally faster than) the prior round's ~46-minute-per-checkpoint runs, which used the
external-env-var workaround. The in-script fix performs at least as well as the launch-time
workaround, with the added benefit that it can no longer be forgotten at launch.

## 5. Action distribution (l2v3 final, val split, n=500, 15,537 decisions)

| metric | mean | std | p1 | p10 | p25 | p50 | p75 | p90 | p99 | at lower bound | at upper bound |
|---|---|---|---|---|---|---|---|---|---|---|---|
| participation_mult (neutral=1.0, [0,2]) | 0.917 | 0.633 | 0.001 | 0.085 | 0.333 | 0.877 | 1.466 | 1.849 | 1.998 | 1.10% | 0.70% |
| urgency (neutral=0.5, [0,1]) | 0.260 | 0.302 | 0.000 | 0.012 | 0.043 | 0.127 | 0.356 | 0.850 | 1.000 | 3.26% | 1.68% |

Within-episode std (responds to state mid-episode): participation_mult=0.525, urgency=0.201.
Between-episode std (differs by episode/day): participation_mult=0.321, urgency=0.174. Both
non-trivial and comparable in scale to prior checkpoints' own action-distribution figures — this
is an actively steering policy, not a collapsed near-constant one. `urgency`'s mean (0.260) sits
well below its neutral value (0.5), i.e. the policy is systematically choosing lower urgency than
TWAP-passthrough's default — active steering that, per Section 2-3, does not translate into a
measurable execution-quality improvement.

## 6. What this round does not include, and why

Per the round's own instruction, the full diagnostic battery (train-vs-val relative comparison,
volatility strata) runs only if the checkpoint clears the pre-registered bar. It did not, so
those diagnostics were not run this round. This is consistent with the standing project norm of
not chasing a favorable secondary read once the primary comparison has already answered the
question — the same discipline applied when the prior round found no checkpoint cleared the bar
and moved straight to the honest verdict rather than mining for a subgroup that looked better.

## 7. Status

Test split untouched throughout. No further evaluation launched this round — reporting and
awaiting direction.
