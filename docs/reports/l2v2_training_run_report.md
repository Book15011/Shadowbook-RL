# L2 Reward Redesign: Real 2,000,000-Step Training Run Report

**run_name**: `l2v2_potentialis_20260827` (`--l2-reward-mode potential_is_shaping`)
**Comparison baseline**: `l2v1_20260825` (`--l2-reward-mode l3_passthrough`, the original run)
**Status**: training complete, not yet evaluated. n=500 eval and the diagnostic battery are a
separate, not-yet-started round. Test split untouched throughout.

## 1. Summary

The run completed mechanically clean — full 2,000,000 steps, no crash, no NaN, checkpoints and
eval firings landed on schedule the entire time. But it did not finish in a numerically healthy
state: starting around step 1.6M (80% through), the SAC critic diverged, with `critic_loss`
climbing roughly exponentially from a stable ~0.06 baseline to 10,800 by the final step.

The important new finding in this report: **`l2v1` (the old reward) shows the same terminal
critic blowup** (ending at `critic_loss`=7,230), via a different timeline — it started climbing
almost from the beginning of training, while `l2v2` stayed flat and stable for the first ~65% of
training and only diverged sharply near the end. This reframes the earlier live "ent_coef looks
much better under the new reward" read: that was real and substantial for most of training, but
by step 2M **both runs converge to a similarly degraded critic / elevated ent_coef state.** The
new reward delayed the failure mode, not eliminated it — and this raises a real question about
whether `l2v1`'s own already-reported n=500 negative result was partly shaped by the same
end-of-training degradation, not purely by a lack of learnable signal.

Two evaluation candidates now exist for `l2v2`: the final checkpoint (likely degraded) and a
clean pre-divergence checkpoint at step 1,599,936. Both should be evaluated in the next round,
and it may be worth re-examining an `l2v1` pre-divergence checkpoint too.

## 2. Launch and configuration

Launched 2026-08-27 01:17 HKT under `nohup`, completed 2026-08-28 06:31 HKT (105,219s / 29.23h
wall-clock — within 7 minutes of `l2v1`'s own 29.34h, confirming the new reward does not change
per-step cost materially).

Pre-launch verification (all done fresh immediately before launch, not from memory):
- L3 checkpoint sha256 `a5443e2a4c6c1d4427d4ce1cb83e65d622ea688d8953f5bf94b29e87fbcaa77d` and
  VecNormalize sha256 `b459e17784c239be48069c47a7da6454610b4674a99e5d513d3ef0b616c182d8`, both
  matching `docs/reports/l3_frozen_handoff.md` exactly.
- Fresh `free -h`/`nvidia-smi`/`ps`: 48GB RAM available, GPU 0MB/0% used, 207GB disk free, no
  competing training process.
- Canonical L2 checkpoint (`models/l2_strategist_v1.zip`, `l2_vecnormalize.pkl`) present and
  confirmed untouched pre- and post-run — the guard correctly diverted the final save to
  `models/l2_strategist_v1_l2v2_potentialis_20260827.zip` /
  `l2_vecnormalize_l2v2_potentialis_20260827.pkl` since `--overwrite-canonical` was not passed.

Config, identical to `l2v1_20260825` except the two flags below:
`--l3-checkpoint models/l3_frozen_backup/l3_executioner_v1_frozen.zip`,
`--l3-vecnormalize models/l3_frozen_backup/l3_vecnormalize_frozen.pkl`,
`--total-timesteps 2000000`, `--n-envs 6`, `--seed 42`, `--use-numeric-format`, `--eval`
(defaults: `--eval-freq 10000`, `--n-eval-episodes 10`, `--checkpoint-freq-timesteps 50000`).
Changed: `--run-name l2v2_potentialis_20260827`, `--l2-reward-mode potential_is_shaping`.

## 3. Mechanical health

- **Throughput**: steady ~19 dec/s the entire run (SB3-reported `fps`), matching `l2v1`'s own
  18.8-18.93 dec/s within noise.
- **Checkpoints**: all 40 landed on the expected ~44-minute/49,998-step cadence, no gaps, each
  with model + replay buffer (174,001,374-174,001,376 bytes, matching the known fixed footprint)
  + VecNormalize. Total checkpoint disk usage: ~6.5GB.
- **Errors**: 0 matches for `traceback|error|nan|killed|oom` across the entire log, checked at
  every check-in and again at completion.
- **Process health**: all 9 expected processes (main + resource_tracker + forkserver + 6 workers)
  confirmed alive at every check-in; clean exit at completion (no orphaned processes).
- **Memory**: point-in-time spot checks only, not a continuous trend (same honest gap flagged
  after the Task 3 shakedown) — main-process RSS measured at ~4.37GB (1h), ~4.40GB (2.8h),
  ~4.42GB (28.5h) — flat across the span checked. A full summed-across-workers RSS was only taken
  once, at the 1h mark: ≈25.26GB, in the same band as `l2v1`'s own confirmed-stable 25.8-25.9GB.
  No OOM occurred at any point.
- **VRAM**: 2,860MB (1h) to 4,192MB (2.8h) of 24,564MB total, low utilization (0-3%) throughout.

## 4. ent_coef: new reward vs. old, at matching steps

This was the specific diagnostic requested going into the run (a falling `ent_coef` would
indicate the critic finding real value differences between actions; a climbing one — what
`l2v1` showed throughout — indicates the opposite). Tracked live at four points:

| step | l2v2 (new reward) | l2v1 (old reward) | ratio |
|---|---|---|---|
| ~76k | 0.00279 | 0.0057 | ~2x lower |
| ~191k | 0.00118 | 0.0082 | ~7x lower |
| ~1.23M | 0.000579 | 0.0334 | ~58x lower, still falling |
| ~1.9M+ | 0.163 (reversed upward) | 0.135 | **roughly equal** |

The full picture, now that the run is complete: the divergence held and widened for the first
~65% of training, then reversed sharply and closed almost entirely by the end. **This should not
be read as "the new reward solved the ent_coef problem"** — it should be read as "the new reward
delayed it by roughly 1.3M steps." See Section 6 for why both runs seem to land in the same place.

## 5. Held-out eval trajectory during training

`ValISEvalCallback`, paired seeds 5000000-5000009, TWAP-passthrough baseline = 0.9976 IS_total_bps
(lower is better), 200 firings at 10,000-step intervals. **Not a clean improvement story.**
Block averages (each block = 10 consecutive firings):

| step range | mean IS_total_bps | note |
|---|---|---|
| 10k-100k | 2.62 | |
| 110k-300k | 2.78-2.85 | |
| 310k-500k | 3.00-3.17 | worsening |
| 510k-1.0M | 2.53-2.75 | |
| 1.01M-1.1M | **3.34** | worst block of the run |
| 1.11M-1.4M | 2.35-2.96 | |
| 1.31M-1.58M | 2.31-2.35 | modestly lower, more stable |
| 1.8M-1.99M (post-divergence) | ~2.4-3.4, still noisy | e.g. final firing at step 1,990,398: 3.41 |

Every block average is still 2-2.5x worse than the 0.9976 baseline — **no block ever got close to
beating TWAP-passthrough**. Several sharp, unexplained collapses occurred (steps ~670k, ~900k-970k)
where eval IS_total_bps briefly dropped to 0.40-0.59 (below baseline) then reverted to 3+ on the
very next firing 10,000 steps later — read as instability, not a discovered edge, since it never
held. No conclusion about final policy quality should be drawn from these training-time firings;
they exist to catch a broken run early, not to substitute for the n=500 round.

## 6. The critic divergence, in detail

Whole-run trajectory (`actor_loss`, `critic_loss`, `ent_coef`, sampled every ~100k steps),
**both runs side by side**:

| step | l2v2 actor/critic/ent_coef | l2v1 actor/critic/ent_coef |
|---|---|---|
| ~99k | 1.16 / 0.070 / 0.00245 | 2.97 / 0.231 / 0.00594 |
| ~297k | 0.79 / 0.057 / 0.00096 | 12.3 / 5.13 / 0.012 |
| ~495k | 0.85 / 0.074 / 0.00053 | 25.1 / 29.7 / 0.0162 |
| ~693k | 0.75 / 0.066 / 0.00044 | 25.7 / 33.6 / 0.0124 |
| ~891k | 0.68 / 0.066 / 0.00049 | 31.2 / 56.4 / 0.0145 |
| ~1.09M | 0.67 / 0.067 / 0.00056 | 60.6 / 183 / 0.0291 |
| ~1.29M | 0.72 / 0.059 / 0.00046 | 92.2 / 640 / 0.0392 |
| ~1.39M | 1.03 / 0.112 / 0.00069 | 84.5 / 457 / 0.038 |
| ~1.49M | 1.68 / 0.276 / 0.00109 | 79.7 / 546 / 0.0343 |
| ~1.58M | 4.79 / 2.24 / 0.00279 | 89.7 / 703 / 0.0367 |
| ~1.68M | 17.8 / 28.1 / 0.00886 | 111 / 983 / 0.0467 |
| ~1.78M | 46.9 / 196 / 0.0237 | 131 / 1,400 / 0.0624 |
| ~1.88M | 115 / 1,120 / 0.0559 | 232 / 3,310 / 0.084 |
| ~1.98M (final) | 352 / 11,100 / 0.146 | 302 / 7,230 / 0.135 |

Reading this side by side is the key result of this report. `l2v1`'s critic starts drifting
almost immediately — already above 5 by step 297k, above 100 by step 1.1M, and climbs at a
roughly steady rate the entire run. `l2v2`'s critic stays flat and small (0.05-0.11) for the
first ~1.4M steps — a genuinely different, more stable regime — then diverges sharply and
catches up to (and slightly exceeds) `l2v1`'s degradation by the end.

**Interpretation, held to what the data supports**: the new reward does appear to give SAC a
materially more stable critic for the majority of training — that difference is large (two
orders of magnitude in `critic_loss` through step 1.3M) and sustained, not noise. But whatever
is ultimately driving the terminal blowup — most plausibly Q-value overestimation compounding
over a long single training run, a known SAC/DDPG-family pathology — is not specific to the new
reward's shaping; `l2v1` hit an equivalent end state under the old reward too, just earlier and
more gradually. This looks like a general property of this SAC configuration (network size,
learning rate, target-update cadence, or lack of Q-value clipping) at the ~1.5-2M-step horizon,
not something either reward design fixes or causes outright.

**A retrospective question this raises, not yet answered**: `l2v1`'s canonical checkpoint
(`models/l2_strategist_v1.zip`) is the final save at step 1,999,974 — the same point where its
own `critic_loss` was already at 7,230. The earlier n=500 evaluation of that checkpoint (this
project's original negative result, IS_total_bps=1.233 vs TWAP-passthrough=1.024) was run against
a policy shaped by a critic that had been degrading for well over half the run. Whether that
degradation materially affected the evaluated policy, versus the policy having simply converged
to something IS-negative regardless, is not established here — flagging it as a live question
for whoever revisits that result, not a retraction of it.

## 7. Candidate checkpoints for the next round

| checkpoint | step | critic_loss at save | status |
|---|---|---|---|
| `models/l2_strategist_v1_l2v2_potentialis_20260827.zip` | 1,999,992 (final) | ~10,800 | likely degraded |
| `models/l2_checkpoints/l2_sac_l2v2_potentialis_20260827_1599936_steps.zip` | 1,599,936 | ~1-2 (clean) | pre-divergence |

Recommend evaluating both in the n=500 round rather than assuming the final save is the right
candidate. Given Section 6's finding, it may also be worth pulling an equivalent mid-run `l2v1`
checkpoint (e.g. around step 500k-1M, well before its own `critic_loss` climbed past a few
hundred) as a third comparison point, though that widens the next round's scope beyond what was
asked for here and should be a separate decision.

## 8. What this report does not establish

- No causal diagnosis of the critic divergence's root cause — flagged as a plausible-but-unproven
  SAC pathology, not investigated further this round (out of scope; the run was not intervened on
  since detection happened with under an hour of training remaining).
- No evaluation of either candidate checkpoint's actual execution quality — that is the explicit
  scope of the next, not-yet-started round (n=500 eval + diagnostic battery). Nothing in this
  report should be read as a performance verdict on the new reward.
- Memory stability is spot-checked, not continuously tracked, for the same reason flagged after
  the Task 3 shakedown.

## 9. Files and commits

- `docs/TRACK_STATUS.md` — L2 section updated with this run's completion and the divergence
  finding, commit `82e62d0`.
- This report: `docs/reports/l2v2_training_run_report.md`.
- Training log: `logs/l2_train_real_l2v2_potentialis_20260827.log` (105,219s, ~1.98MB).
- All committed locally, nothing pushed to `origin/master`.
