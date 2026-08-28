# L2 Reward Redesign: Three-Checkpoint Evaluation and Diagnostic Battery

Follow-up to `docs/reports/l2v2_training_run_report.md`, which found that both `l2v1` (old
reward) and `l2v2` (new reward) ended with a diverged SAC critic, and identified clean
pre-divergence checkpoints for both runs. This round evaluates three candidates at n=500 against
the pre-registered bar, then runs the full diagnostic battery on the best performer. Test split
untouched throughout.

## 1. Summary

None of the three checkpoints beat TWAP-passthrough at the pre-registered bar (both paired
t-test AND Wilcoxon significant, same direction). The best performer by mean IS_total_bps is
`l2v2`'s pre-divergence checkpoint (step 1,599,936) at 1.117 bps, closest to the 1.024 baseline
and statistically indistinguishable from it (neither test significant). Every direct
checkpoint-vs-checkpoint comparison points the same direction — earlier, less-diverged
checkpoints score better than their own later, more-diverged counterparts — but none of those
gaps individually reaches significance at n=500. The volatility-stratified and cross-split
diagnostics on the best performer reproduce the same pattern found on the original checkpoint:
no edge that strengthens with volatility, and a real but smaller/less robust train-vs-val swing
than previously measured. A genuine, unrelated performance bug in the eval harness itself was
found and fixed mid-round (see Section 6).

## 2. Candidates and the pre-registered comparison

Same methodology as the original n=500 evaluation (`scripts/eval_l2_n500.py`/
`eval_l2_diagnostics.py`, unmodified): 500 paired episodes, seeds 5,000,000-5,000,499, val split
(2025-07-16 to 2025-08-02, 18 days), against TWAP-passthrough (frozen L3, unsteered) and Pure
TWAP (base env, no L3/L2 at all).

| checkpoint | step | critic_loss at save | IS_bps | vs TWAP-passthrough (1.024) | vs Pure TWAP (0.889) | fill_ratio |
|---|---|---|---|---|---|---|
| l2v2 final | 1,999,992 | ~10,800 | 1.227 | d_z=0.060, t_p=0.179, **w_p=0.0045** | d_z=0.083, t_p=0.066, w_p=0.064 | 0.920 |
| **l2v2 pre-divergence** | 1,599,936 | ~1-2 | **1.117** | d_z=0.032, t_p=0.468, w_p=0.295 | d_z=0.061, t_p=0.175, w_p=0.218 | 0.924 |
| l2v1 mid-run | 499,980 | ~26 | 1.177 | d_z=0.049, t_p=0.270, w_p=0.110 | d_z=0.075, t_p=0.094, w_p=0.133 | 0.914 |

Reading this plainly: l2v2 final is the only one with a significant test at all (Wilcoxon,
p=0.0045), but its t-test isn't significant, so it fails the "both agree" bar the same way the
project's very first n=500 result did — a real signal on one test alone is exactly the kind of
result this project pre-registered against trusting. The other two checkpoints don't clear
significance on either test — not evidence of beating baseline, but also not evidence of losing
to it at a level distinguishable from noise.

**l2v2's pre-divergence checkpoint is the best performer** and carries forward to Section 4-5.

## 3. Direct checkpoint-vs-checkpoint comparison (paired, same seeds)

Beyond comparing each checkpoint to baseline separately, the three share the identical seed
list, so they can be compared directly to each other:

| comparison | mean diff | d_z | t_p | w_p |
|---|---|---|---|---|
| l2v2 final − l2v2 pre-divergence | +0.110 (final worse) | 0.045 | 0.320 | 0.327 |
| l2v1 mid-run − l2v2 pre-divergence | +0.061 (mid-run worse) | 0.029 | 0.524 | 0.416 |
| l2v1 mid-run − l2v2 final | −0.050 (mid-run better) | −0.023 | 0.611 | 0.461 |

Every comparison points toward less-diverged checkpoints performing better, consistent with the
critic divergence degrading policy quality. **None reaches significance at n=500** — this is
directional, not established. Section 7 addresses what this does and doesn't support about
early stopping.

## 4. Cross-split relative comparison (train vs val, best performer)

Reruns the project's earlier Diagnostic-2-correction methodology
(`scripts/analyze_l2_relative_comparison.py`, new this round): compare each split's own
L2-minus-TWAP-passthrough difference distribution via Welch's t-test + Mann-Whitney U
(independent samples — the two splits' episode pools are disjoint).

| split | n | L2-minus-baseline mean | std |
|---|---|---|---|
| val | 500 | +0.093 bps (worse) | 2.866 |
| train | 500 | −0.341 bps (better) | 3.243 |

Swing = 0.434 bps. Welch's t: t=2.239, **p=0.025** (significant). Mann-Whitney: U=131,466,
**p=0.157** (not significant). Cohen's d=0.142 (small).

This is the same qualitative pattern the project found on the original checkpoint (val worse,
train better) but **weaker and less robust** here — the original swing (~0.46bps) cleared both
tests; this one clears only Welch's t. On train, L2 does significantly beat TWAP-passthrough by
the t-test alone (t_p=0.019, mean_diff=−0.341) but not Wilcoxon (w_p=0.259) — the same "one test
alone" pattern seen throughout this round, not a clean win anywhere.

## 5. Volatility-stratified evaluation (train days, best performer)

Same three buckets and methodology as the project's earlier bucketed evaluation
(`scripts/eval_l2_bucketed.py`, unmodified) — **train days only, confounded with memorization,
see that script's own docstring**: any edge found here cannot be distinguished from the policy
having memorized these specific days.

| bucket | n_days | IS_bps (L2) | vs TWAP-passthrough mean_diff | d_z | t_p | w_p |
|---|---|---|---|---|---|---|
| calm | 231 | 1.016 | +0.015 (worse) | 0.006 | 0.897 | 0.532 |
| moderate | 137 | 1.366 | −0.040 (better) | −0.011 | 0.814 | 0.370 |
| high | 37 | 0.940 | −0.029 (better) | −0.004 | 0.931 | 0.875 |

No strengthening with volatility — every effect size is negligible (|d_z| < 0.011 throughout),
matching almost exactly the original checkpoint's own high-volatility result (d_z=−0.011). This
closes out the "does L2 have hidden value in volatile conditions" question the same way it was
closed before: no, for this checkpoint either.

## 6. A real bug found and fixed mid-round

`scripts/eval_l2_n500.py`, `eval_l2_diagnostics.py`, and `eval_l2_bucketed.py` had **no
thread-capping** (`OMP_NUM_THREADS`/`MKL_NUM_THREADS`/etc.), unlike `train_l2.py`, which was
fixed for exactly this issue during the vectorization round (documented 7-9x slowdown from
thread oversubscription). The first attempt at checkpoint 1 of Section 2's table ran for 33
minutes at 1,353% CPU (out of 1,600% available on this 16-vCPU box) with zero output — killed
and relaunched with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1` set at launch. CPU dropped to ~100% (single core, as expected for a
single-env script) immediately, and the same checkpoint completed in ~46 minutes total. This is
a real, generalizable bug in three scripts, not specific to this round's checkpoints — only 33
minutes were lost (no output had been produced yet), but any future n=500 run against these
scripts without the env vars set will hit the same multi-x slowdown. Worth fixing in the scripts
themselves (adding the same env-var/`torch.set_num_threads(1)` pattern `train_l2.py` uses) as a
follow-up, not done this round since it wasn't asked for and the workaround was sufficient here.

## 7. Does the evidence support early stopping as the correct procedure?

**Directionally, yes, consistently — but not at a level any individual test establishes.**

Three separate comparisons in this round all point the same way: less-diverged checkpoints beat
more-diverged ones (Section 3, all three pairwise comparisons). Combined with
`docs/reports/l2v2_training_run_report.md`'s own finding — both `l2v1` and `l2v2` independently
developed the same terminal critic blowup, just on different timelines — this is a coherent
story: **training substantially past the point where the critic starts diverging does not help,
and the (weak, non-significant) evidence here is consistent with it hurting.**

What this round does NOT establish: no single checkpoint comparison reaches significance, so
"early stopping improves the policy" is not proven at conventional confidence from this data
alone. What it also does not establish: none of the three checkpoints — early, mid, or late —
beats TWAP-passthrough at the pre-registered bar either. Early stopping, if real, appears to
reduce how much the model *loses* by relative to baseline, not turn a losing policy into a
winning one. The honest reading is: stop training once the critic starts diverging (there is no
evidence that continuing past that point helps, and directional-if-unproven evidence it hurts),
but do not expect early stopping alone to fix the underlying finding that this reward redesign
has not yet produced a checkpoint that beats TWAP-passthrough.

## 8. Files and commits

- Scripts (this round): `scripts/analyze_l2_relative_comparison.py` (new, committed `1d4fdf6`).
- Results (uncommitted, per this project's `models/*.json` precedent — data files, not code):
  `models/l2_n500_{l2v2final,l2v2predivergence,l2v1midrun}_val.json` (+ episodes/actions CSVs),
  `models/l2_n500_l2v2predivergence_train.json` (+ CSVs), `models/l2_bucketed_{calm,moderate,
  high}_l2v2predivergence.json`.
- This report: `docs/reports/l2v2_checkpoint_evaluation_report.md`.
- Test split: untouched.
