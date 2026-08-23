# L3 frozen checkpoint: handoff for L1→L2→L3 integration

**Date:** 2026-08-23
**Status:** L3 research is closed. This document is written to stand alone --
read this before wiring the frozen checkpoint into anything, without needing
the rest of this project's session history.

## The frozen checkpoint

| | |
|---|---|
| Live working path | `models/l3_executioner_v1_twap_ab_armA_control.zip` |
| Live working VecNormalize | `models/l3_vecnormalize_twap_ab_armA_control.pkl` |
| **Permanent backup** | `models/l3_frozen_backup/l3_executioner_v1_frozen.zip` |
| **Permanent backup VecNormalize** | `models/l3_frozen_backup/l3_vecnormalize_frozen.pkl` |
| Checkpoint sha256 | `a5443e2a4c6c1d4427d4ce1cb83e65d622ea688d8953f5bf94b29e87fbcaa77d` |
| VecNormalize sha256 | `b459e17784c239be48069c47a7da6454610b4674a99e5d513d3ef0b616c182d8` |
| Backup committed | `3a4a283` (git), verified byte-identical to source by checksum after copy |

**This is "Arm A" from the TWAP-baseline-reward A/B test**: warm-started
(weights only) from v1's own step-2,000,000 checkpoint, then trained 1,000,000
further steps with `subtract_twap_baseline=False`. It is NOT v1 itself, NOT
the canonical stand-in checkpoint, and NOT the further-2M-step budget
extension (that one is a documented negative, see below).

**Checkpoint <-> VecNormalize pairing, confirmed not assumed:** both files
share an identical mtime down to the nanosecond
(`2026-08-21 23:09:09.236393768 +0800`), and the training log
(`logs/l3_train_twap_ab_armA_control.log`) has an explicit final-save line
naming both paths together as one save operation: *"Saved model to
models/l3_executioner_v1_twap_ab_armA_control, VecNormalize to
models/l3_vecnormalize_twap_ab_armA_control.pkl."* This is not two files that
happen to sit in the same directory -- they were written by the same call.

**Why this checkpoint and not another candidate:** every checkpoint this
session evaluated at proper n=500 is compared in the results table below.
Two checkpoints from the same lineage looked competitive at first glance and
were checked directly before designating anything (a cheap eval, not a new
training run):
- A 500,000-step in-training snapshot from the later budget-extension run
  scored 0.686 IS_total_bps at n=50 in training -- appealing on its face, but
  n=500 revealed 1.025, not significantly different from this checkpoint's
  own 0.994 (paired t p=0.82, Wilcoxon p=0.78) and not better. This is the
  THIRD time this session that an n=50 in-training reading understated the
  true (worse) n=500 result for a checkpoint later checked properly -- treat
  any n=50-only number in this project's history as provisional.
- The full 2,000,000-step budget extension (see "What was tried" below) is a
  documented, statistically clean negative -- not a candidate.

## Exact reward configuration this checkpoint trained under

Reproduce this exactly if fine-tuning or retraining from this checkpoint --
getting any of these wrong changes what the policy is actually optimizing.

| Parameter | Value | Notes |
|---|---|---|
| `zeta` | 0.06 | `RewardWeights()` default |
| `eta_replace` | 0.0 | `RewardWeights()` default -- inert (r_placement_stale contributes exactly 0 regardless of its own inputs) |
| `subtract_twap_baseline` | `False` | `RewardWeights()` default -- OFF |
| r_queue queue-weighted term direction | **INVERTED** (`EXPERIMENTAL 4` in `reward.py`'s module docstring) | **Empirically unvalidated. De facto active, not a deliberate choice for this checkpoint's own training run.** See "What was tried" below. |

**On the r_queue inversion specifically:** this is the single most important
fact in this table, because it is easy to get wrong by assumption. v1's own
original training used the ORIGINAL (non-inverted) r_queue direction. The
inversion was written into `reward.py` afterward, for an unrelated,
inconclusive probe (see below), left uncommitted, and never reverted --
every training run since (this checkpoint included) silently inherited it as
inline, unconditional code with no flag to turn it off. It is now committed
at HEAD (`4d81a96`) specifically so this can't happen silently again, but
committing it is not an endorsement -- it remains untested against the
original direction in any statistically adequate way (see "Known open
items"). **If you fine-tune from this checkpoint, you are fine-tuning under
the inverted direction by default (current `reward.py` HEAD) -- if you want
the original direction, you must deliberately revert `reward.py`'s
`step_reward()` r_queue block to `w.gamma * (queue_ahead_at_cancel /
queue_at_level)` (both occurrences) first.**

All other `RewardWeights()` fields (kappa, beta, gamma, lam, and the fee/slip
terms) are unmodified defaults -- this session's reward work only ever
touched zeta, eta_replace, the r_queue direction, and subtract_twap_baseline.

## Full n=500 results table

Every number below is from the same paired-seed methodology (seeds
5,000,000-5,000,499, the held-out `val` date-range split via
`load_split("val")`), so all rows are directly poolable.
TWAP's own per-episode numbers are the same 500-episode set reused
byte-for-byte across every comparison in this project from that point
onward.

| Checkpoint/policy | IS_total_bps mean (std) | fill_ratio | vs TWAP: mean diff | paired t | Wilcoxon |
|---|---|---|---|---|---|
| TWAP (fixed baseline) | 0.889 (4.353) | 0.994 | -- | -- | -- |
| **Arm A -- THE FROZEN CHECKPOINT** | **0.994 (3.570)** | **0.919** | +0.105bps | p=0.534 | p=0.653 |
| 500k-step snapshot (context only, not a candidate) | 1.025 (3.230) | 0.949 | +0.135bps | p=0.396 | p=0.358 |
| best-B heuristic (scripted, not RL) | 1.103 | 1.000 | +0.214bps | p=0.101 | p=0.191 |
| v1 (original 2M-step checkpoint) | 1.261 (4.242) | 0.892 | +0.372bps | **p=0.033** | p=0.115 |
| Arm B (TWAP-baseline reward, treatment) | 1.341 (2.405) | 0.990 | +0.452bps | **p=0.009** | **p=0.014** |
| Budget extension (Arm A + 2M more steps) | 1.237 (2.039) | 1.000 | +0.347bps | **p=0.034** | **p=0.044** |

(Lower IS_total_bps is better -- less implementation shortfall.
Bold p-values are conventionally significant at 0.05; none represent L3
beating TWAP -- every significant result in this table is L3 losing to it.)

## Honest performance statement

**This policy ties TWAP. It does not beat it.** At n=500, the frozen
checkpoint's mean diff against TWAP is +0.105bps in TWAP's favor, not
statistically distinguishable from zero (p=0.534/0.653). Read the whole
table plainly: nothing evaluated this session -- not this checkpoint, not
v1, not either A/B arm, not the budget extension, not even the hand-tuned
scripted heuristic -- beats TWAP at n=500. The frozen checkpoint is simply
the least-bad of everything tried, and "least-bad" here means
"statistically indistinguishable from a fixed, unlearned baseline," not
"good."

**The real engineering win is the fill-ratio arc, not reward shaping.**
Pre-retrain, this project's L3 checkpoint had a fill_ratio of 0.2015 (see
`docs/reports/l3_replace_value_probe.md`'s citation of the milestone
report) -- filling roughly a fifth of the order and marking the rest at
terminal price, an artifact that flatters IS_total_bps without reflecting
real execution. By this checkpoint, fill_ratio is 0.919; by the (separately
not-recommended) budget extension, it reached 0.9998 -- essentially complete
fills. This recovery tracks physics/matching-engine fixes made earlier in
this project's history, not any of the reward-shaping work in the table
above (zeta, eta_replace, the r_queue split/inversion, and
subtract_twap_baseline all left fill_ratio in the same 0.89-1.00 band
regardless of which one was active). Whoever picks this project back up
should not expect further reward tuning to move fill_ratio meaningfully --
that lever already moved, and it moved for a different reason.

## What was tried and didn't work (so it isn't retried blind)

1. **r_queue MARKET/REPLACE split** (`EXPERIMENTAL 3`, committed `7bbf709`):
   price MARKET-triggered cancels and CANCEL_AND_REPLACE-triggered cancels
   differently in `r_queue`, motivated by CANCEL_AND_REPLACE usage sitting
   at ~0.3% in v1. Committed and is part of this checkpoint's reward config
   (it's not optional/gated) -- but did not on its own move REPLACE usage
   materially (see the REPLACE finding below).
2. **r_queue direction inversion** (`EXPERIMENTAL 4`, committed `4d81a96`,
   empirically unvalidated): inverted which side of the queue-position ratio
   gets charged more, on the theory the original split's direction was
   backward. A bounded 500k-step probe against v1 found no significant
   difference either way (IS 1.245->1.829 at n=50, not significant vs TWAP
   for either checkpoint, p=0.90/0.53 vs p=0.27/0.15) and did NOT move
   REPLACE usage significantly (0.298%->0.336%, two-proportion z-test
   p=0.15). Left in the working tree uncommitted afterward and silently
   inherited by every subsequent run including this checkpoint's own
   training -- see "Known open items."
3. **TWAP-baseline (variance-reduction) reward**
   (`docs/reports/l3_twap_baseline_reward.md`, `EXPERIMENTAL 5`, gated
   behind `subtract_twap_baseline`, default OFF): subtracts a per-episode
   TWAP shadow-execution's IS from the terminal reward, explicitly a
   variance-reduction mechanism, not an objective change. Matched A/B test
   (this checkpoint as control vs. an identical run with the flag on):
   the treatment arm is significantly WORSE than both TWAP (p=0.009/0.014)
   and this checkpoint itself (p=0.010/0.022, direct paired test). The
   mechanism did reduce outcome variance as designed (Levene p<0.0001) --
   that reduction did not translate into better, or even equal, execution
   quality.
4. **Budget extension** (`docs/reports/l3_armA_budget_extension.md`): 2M
   more steps from this checkpoint, same reward config, to test whether
   more training alone helps. Null result: the direct paired test against
   this checkpoint is not significant (p=0.092/0.230, Cohen's d_z=0.076),
   97% of the nominal difference traced to 2% of episodes, no convergence
   visible across 8 in-training eval firings (volatile between 0.69 and
   1.38 bps throughout, best point at only 500k steps in and never
   bettered).

**Pattern across all four:** every reward-shaping and budget intervention
tried after this checkpoint's own training returned null or negative. This
is the basis for closing the research phase, not any single result alone.

## The REPLACE finding: near-zero usage is correct, not a bug

`docs/reports/l3_replace_value_probe.md` settled this with scripted
heuristics (no RL, no training): does CANCEL_AND_REPLACE actually improve
execution on this data at all? A properly-powered n=500 test of the best
scripted REPLACE-active heuristic found across an 18-config sweep came in
numerically WORSE than TWAP (+0.214bps, not significantly different,
p=0.101/0.191) -- the opposite sign from what an underpowered n=50 screen
had suggested. No PASSIVE-family configuration reaches fill comparable to
TWAP (40.4% at best, a structural ceiling from single-tick depth, not a
sweep gap). **Conclusion: the trained policy's near-0% CANCEL_AND_REPLACE
usage is correct behavior on this data, not a failure to learn a valuable
action.** Nobody should read low REPLACE usage in this or any downstream
checkpoint as something to fix.

## Known open items

- **Original-vs-inverted r_queue direction has never been cleanly
  compared.** The closest existing test (item 2 above) compared each
  direction against TWAP separately, at n=50 only, never the two directions
  against each other directly, and never at adequate power. Anyone who
  wants to resolve this needs a proper paired test, not a re-read of the
  existing probe.
- **Single training seed throughout this entire lineage.** v1, this
  checkpoint, Arm B, the budget extension, and the direction-inversion probe
  are all descended from one continuous single-seed trajectory. None of
  this project's "X ties/beats/loses to Y" findings have been replicated
  under an independent seed. Any of them, including this checkpoint's own
  tie with TWAP, could in principle be specific to this seed's trajectory --
  untested, not ruled out.
- **True v1 (`973b2883...`) is permanently unrecoverable.** Overwritten by
  an earlier probe run's own final save (a since-fixed save-path bug, see
  `src/train/train_l3.py`'s `resolve_final_save_paths()` and
  `--overwrite-canonical` guard). The step-2,000,000 stand-in
  (`27afa91e...`, itself backed up at `models/v1_near_backup_step2M/`) is
  numerically verified equivalent and is what this checkpoint and every
  other run this session actually descends from.

## Integration compatibility, verified by reading the real code

Checked directly against the current source, not assumed -- see
`docs/TRACK_STATUS.md`'s L3 section for the equivalent summary and any
newer status.

- **Observation space:** `LOBExecutionEnv.observation_space` is a 42-dim
  `Box` built from `_OBS_SPEC` (`src/envs/lob_execution_env.py`), matching
  L2's `FrozenL3Wrapper` index-mapping table exactly. `_build_obs()`'s
  `values` list order matches `_OBS_SPEC`'s index order one-for-one --
  checked line by line, no drift.
- **Action space:** `MultiDiscrete([4, 11, 5])`, matches.
- **VecNormalize:** `train_l3.py` constructs
  `VecNormalize(norm_obs=True, clip_obs=5.0, ...)` -- exactly what
  `wrappers.py` assumes and documents. No mismatch.
- **L2 path (obs idx 15/16):** `l2_target_slice_ratio_override`/`l2_urgency`
  are plain settable attributes on the env; `FrozenL3Wrapper` sets both each
  step and resets both to their neutral defaults (`None`, `0.5`) at the
  start of `reset()` (a real cross-episode leak L2's own session already
  found and fixed). `_build_obs()` reads them into idx 15/16 correctly.
  Works as documented.
- **L1 path (obs idx 17/18, and `l1_risk_score` into `step_reward`'s
  inventory term):** the env-side wiring is intact and correct --
  `l1_risk_score`/`l1_confidence` feed idx 17/18 via `_build_obs()`, and
  `l1_risk_score` is passed into `step_reward()` where it scales the
  inventory-holding cost (`r_inv`). **But `FrozenL3Wrapper` does not
  currently set either attribute** -- they sit at the base env's defaults
  (0.0, 0.0) whenever L3 is driven through L2's wrapper. This is not a bug
  or a mismatch (L1's real Ollama integration is still explicitly paused
  per L1's own status), but it means the L1 path is untested end-to-end
  through L2 and will need wiring, not just verification, before L1's
  signal actually reaches L3 in practice.

**No blockers found.** L2 can load this checkpoint via `train_l2.py`'s
existing `--l3-checkpoint`/`--l3-vecnormalize` flags with no code changes.

```
--l3-checkpoint models/l3_frozen_backup/l3_executioner_v1_frozen.zip
--l3-vecnormalize models/l3_frozen_backup/l3_vecnormalize_frozen.pkl
```
