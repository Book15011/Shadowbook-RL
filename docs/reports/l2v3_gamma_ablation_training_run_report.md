# L2 Gamma Ablation: Real 1,600,000-Step Training Run Report

**run_name**: `l2v3_gamma0983_20260829` (`--gamma 0.983`, `--l2-reward-mode potential_is_shaping`)
**Comparison baselines**: `l2v2_potentialis_20260827` (same reward, `gamma=0.995`, 2,000,000
steps) and `l2v1_20260825` (old reward, `gamma=0.995`, 2,000,000 steps)
**Status**: training complete, not yet evaluated. n=500 evaluation is a separate, not-yet-started
round per direct instruction. Test split untouched throughout.

## 1. Summary

**The critic did not diverge.** Across the full 1,600,000-step budget, `critic_loss` stayed in a
tight 0.048-0.074 band the entire run — no upward trend at any point, including in the final
100k steps. This is a qualitatively different outcome from both prior runs: `l2v1` (`gamma=0.995`,
old reward) began drifting almost immediately and reached `critic_loss`=608 by a matching step
count; `l2v2` (`gamma=0.995`, same reward as this run) was *also* flat and stable through ~1.3M
steps, then diverged sharply starting almost exactly at the point this run stopped — `l2v2`'s
`critic_loss` went 0.11 (1.4M) → 0.29 (1.5M) → 2.78 (1.6M) → 26.4 (1.7M) → 335 (1.8M). `l2v3`, at
the same 1.6M-step mark, was still at 0.059.

This is consistent with the gamma hypothesis, via the corrected mechanism stated before the run
was launched (not the truncation-bootstrapping claim, which Task 2's pre-launch check found SB3
already handles correctly — see that check-in for the full reasoning). Section 6 covers what this
single comparison does and does not establish; the short version is that the effect is large and
sustained enough to be a real finding, but this is one run per gamma value, not a seeded ablation,
so "gamma=0.983 fixes the divergence" is a strong hypothesis with one clean confirming data point,
not a closed question.

Whether this translates into better held-out policy performance is untested here — Section 5's
training-time eval firings are noisy and not a substitute for the n=500 round, which was
explicitly deferred to a separate round.

## 2. Launch and configuration

Launched 2026-08-28 ~14:17 HKT under `nohup`, completed 2026-08-29 13:38:47 HKT (84,112s / 23.36h
wall-clock — shorter than `l2v1`/`l2v2`'s ~29h because the budget was 1.6M steps, not 2M; per-step
cost is unchanged, ~19 dec/s throughout, matching both prior runs within noise).

Pre-launch verification (all done fresh immediately before launch):
- L3 checkpoint sha256 `a5443e2a4c6c1d4427d4ce1cb83e65d622ea688d8953f5bf94b29e87fbcaa77d`, matching
  the frozen handoff record — re-verified again after this run's completion, unchanged.
- Fresh `free -h`/`nvidia-smi`/`ps` confirmed an idle box before launch.
- Canonical L2 checkpoint (`models/l2_strategist_v1.zip`, `l2_vecnormalize.pkl`) confirmed
  untouched pre- and post-run (unchanged mtime, Aug 26) — the overwrite guard correctly diverted
  the final save to `models/l2_strategist_v1_l2v3_gamma0983_20260829.zip` /
  `l2_vecnormalize_l2v3_gamma0983_20260829.pkl` since `--overwrite-canonical` was not passed.

Resolved config printed to the log at launch, confirming the single intended change:

```
checkpoint_freq_timesteps=50000
gamma=0.983                          <- only change vs. l2v2_potentialis_20260827
l2_reward_mode=potential_is_shaping  <- same as l2v2
n_envs=6
seed=42
total_timesteps=1600000              <- 1.6M budget vs. l2v2's 2M, per plan
use_numeric_format=True
device=cuda
```

`gamma=0.983` was newly exposed as a CLI flag for this run (commit `2a3eee3`, this round's Task 2)
— previously a hardcoded module constant (`L2_GAMMA=0.995`), threaded identically into both SAC's
own discounting and `VecNormalize`'s reward-return scaling, per the project's existing convention
that both must use the same value.

## 3. Mechanical health

- **Throughput**: steady ~19 dec/s the entire run, matching `l2v1`/`l2v2` within noise.
- **Checkpoints**: all 32 expected checkpoints (1,600,000 / 50,000) landed, each with model +
  replay buffer (174,001,374 bytes, matching the known fixed footprint) + VecNormalize — 64 `.pkl`
  files total (32 replay buffers + 32 VecNormalize), 32 model `.zip` files, no gaps.
- **Errors**: 0 matches for `traceback|error|nan|killed|oom` across the entire log.
- **Process**: clean exit — main training process and its subprocess confirmed no longer running
  immediately after the final "Saved model to..." log line, no orphaned workers.
- **Final save**: correctly diverted to run-tagged paths (`models/l2_strategist_v1_l2v3_gamma0983_20260829.zip`),
  canonical checkpoint (`models/l2_strategist_v1.zip`) and frozen L3 backup both confirmed
  byte/hash-identical to their pre-run state.

## 4. The critic trajectory — three-way comparison

`actor_loss` / `critic_loss` / `ent_coef`, sampled every ~100k steps, all three runs at matching
step counts. `l2v1` and `l2v2` both ran to 2M steps; `l2v3`'s budget was 1.6M, so rows beyond that
show `l2v1`/`l2v2` continuing while `l2v3` has no further data (its final value at 1,599,936 is
repeated for reference).

| step | l2v3 (gamma=0.983) | l2v2 (gamma=0.995, same reward) | l2v1 (gamma=0.995, old reward) |
|---|---|---|---|
| ~100k | 0.70 / 0.048 / 0.00224 | 1.11 / 0.131 / 0.00232 | 2.94 / 0.254 / 0.00591 |
| ~200k | 0.69 / 0.066 / 0.00113 | 0.60 / 0.037 / 0.00110 | 7.14 / 1.75 / 0.00907 |
| ~300k | 0.63 / 0.057 / 0.00076 | 0.82 / 0.055 / 0.00091 | 12.4 / 5.76 / 0.0120 |
| ~400k | 0.63 / 0.073 / 0.00059 | 0.91 / 0.081 / 0.00072 | 19.7 / 15.1 / 0.0141 |
| ~500k | 0.65 / 0.067 / 0.00056 | 0.85 / 0.075 / 0.00058 | 25.1 / 25.1 / 0.0158 |
| ~600k | 0.64 / 0.063 / 0.00049 | 0.69 / 0.069 / 0.00049 | 27.2 / 35.4 / 0.0160 |
| ~700k | 0.65 / 0.071 / 0.00052 | 0.75 / 0.064 / 0.00040 | 26.1 / 38.9 / 0.0134 |
| ~800k | 0.58 / 0.066 / 0.00048 | 0.71 / 0.068 / 0.00046 | 29.0 / 52.6 / 0.0134 |
| ~900k | 0.52 / 0.055 / 0.00043 | 0.71 / 0.085 / 0.00046 | 30.9 / 52.5 / 0.0141 |
| ~1.0M | 0.50 / 0.053 / 0.00045 | 0.75 / 0.067 / 0.00045 | 41.9 / 119 / 0.0195 |
| ~1.1M | 0.49 / 0.050 / 0.00048 | 0.68 / 0.066 / 0.00049 | 63.3 / 238 / 0.0259 |
| ~1.2M | 0.43 / 0.074 / 0.00053 | 0.67 / 0.067 / 0.00045 | 67.4 / 236 / 0.0323 |
| ~1.3M | 0.45 / 0.055 / 0.00056 | 0.75 / 0.075 / 0.00050 | 88.2 / 678 / 0.0375 |
| ~1.4M | 0.45 / 0.057 / 0.00060 | 0.97 / 0.111 / 0.00075 | 78.4 / 462 / 0.0415 |
| ~1.5M | 0.46 / 0.061 / 0.00048 | 1.62 / 0.290 / 0.00113 | 79.7 / 467 / 0.0366 |
| **1.6M (l2v3 final)** | **0.43 / 0.059 / 0.00050** | 5.43 / 2.78 / 0.00349 | 93.2 / 608 / 0.0447 |
| 1.7M | — (run stopped) | 22.0 / 26.4 / 0.0113 | 120 / 952 / 0.0449 |
| 1.8M | — | 63.6 / 335 / 0.0314 | 152 / 2,240 / 0.0650 |
| 1.9M | — | 144 / 1,670 / 0.0620 | 275 / 5,590 / 0.1020 |
| 2.0M (final) | — | 352 / 11,100 / 0.146 | 302 / 7,230 / 0.135 |

Reading this: `l2v3`'s `critic_loss` at 1.6M (0.059) is essentially unchanged from its own value
at 100k (0.048) — no trend at all across the entire run, actor_loss if anything trends slightly
*down* (0.70 → 0.43) rather than up. `l2v2`, running the identical reward under the old
`gamma=0.995`, is indistinguishable from `l2v3` through ~1.3M steps (both sit in the 0.05-0.09
band) and then breaks away sharply in exactly the window `l2v3` never got to run past —
`critic_loss` already 47x higher than `l2v3`'s at the 1.6M mark, and climbing.

## 5. Held-out eval trajectory during training

`ValISEvalCallback`, paired seeds 5000000-5000009, TWAP-passthrough baseline = 0.9976
IS_total_bps (lower is better), 159 firings at 10,000-step intervals.

- Overall: mean=2.494, median=2.780, std=1.051, range [0.083, 4.164] across all 159 firings.
- First 10 firings (early training): mean=2.198.
- Last 30 firings (last ~300k steps): mean=1.323 — pulled down by a cluster of low readings
  around steps 1,460k-1,500k (0.10-0.37), followed by a reversion back to the 2.5-3.5 range for
  the final ~90k steps (last firing at step 1,590,318: 3.399).
- Last 10 firings: mean=2.720.

Same read as `l2v2`'s report: noisy, no clean monotonic trend, every block average well above the
0.9976 baseline (worse IS, since lower is better here), with sporadic sharp dips that don't hold
on the next firing. **No conclusion about final policy quality should be drawn from this** — these
firings exist to catch a broken run early, not to substitute for the n=500 round, which uses 500
paired seeds rather than 10 and is unrun for this checkpoint.

## 6. What this does and doesn't establish

**Supports the gamma hypothesis, with the corrected mechanism.** Task 2's pre-launch check (this
round) found the user's originally stated mechanism — SAC bootstrapping value past a truncated
episode's end because it doesn't know the episode was cut short — does not apply here: SB3's
`ReplayBuffer` already reads `info["TimeLimit.truncated"]` and correctly skips zero-bootstrapping
at truncation (`dones * (1 - timeouts)` in the Bellman target), confirmed by reading the installed
library source directly. The gamma-as-lever case that was actually made before this run launched
rested on a different, more standard mechanism instead: near-flat discounting (`gamma=0.995`,
effective horizon 200 decisions against an episode that's at most ~60) means TD-bootstrap error
has far more opportunity to compound across a long single training run, independent of how
truncation itself is handled. This run's result is consistent with that corrected mechanism:
lowering gamma to match the episode's own horizon (0.983, effective horizon ~59) suppressed
critic-loss growth for the entire budget tested, at the exact reward configuration where the
higher-gamma run diverged.

**What this is not**: a seeded ablation. This is one run at `gamma=0.983` compared against one run
each at `gamma=0.995` (two different reward configurations, `l2v1` and `l2v2`). A single run could
in principle avoid divergence by chance — different random draws through the replay buffer,
different early trajectory. Two things weigh against "chance" as the explanation: (a) `l2v3`
tracks `l2v2` almost exactly for the first 1.3M steps (both flat in the same narrow band) before
`l2v2` breaks away — if this were pure noise, the two curves would not have been this close for
that long only to diverge in exactly the region `gamma=0.995` is expected to start compounding
error; and (b) the suppression held for the *entire* 1.6M-step budget with no late-run wobble, not
just a delay of a few hundred thousand steps the way `l2v2` delayed `l2v1`'s divergence. Neither
point is proof — a second seed at `gamma=0.983` would be the natural next check if this becomes
load-bearing for a future decision, but that wasn't asked for this round and isn't run here.

**Also unresolved**: whether a stable critic under `gamma=0.983` produces a *better* held-out
policy, or merely a numerically calmer one that converges to something equally IS-negative.
Section 5's noisy training-time firings don't answer this — only the deferred n=500 round can.

## 7. Files and provenance

- Log: `logs/l2_train_real_l2v3_gamma0983_20260829.log`
- Final save: `models/l2_strategist_v1_l2v3_gamma0983_20260829.zip` /
  `models/l2_vecnormalize_l2v3_gamma0983_20260829.pkl` (final step 1,599,936, `critic_loss`=0.059
  at save — no pre-divergence-vs-final distinction needed here, unlike `l2v1`/`l2v2`, since the
  final checkpoint never diverged)
- 32 intermediate checkpoints in `models/l2_checkpoints/l2_sac_l2v3_gamma0983_20260829_*_steps.{zip,pkl}`,
  every one numerically clean, any of which is a reasonable evaluation candidate — the earlier
  runs' pre-divergence-checkpoint question doesn't apply to this run.
- `--gamma` CLI flag: `src/train/train_l2.py` (commit `2a3eee3`)
- Thread-capping fix (this round's Task 1, applied before Task 3's launch, unrelated to the
  gamma result but required to have working eval tooling for the next round):
  `scripts/eval_l2_n500.py`, `eval_l2_diagnostics.py`, `eval_l2_bucketed.py` (commit `4d5a544`)

## 8. Status

Training and mechanical reporting for this round are complete. Per instruction, **stopping here**
— n=500 evaluation and the diagnostic battery on this checkpoint are a separate, not-yet-started
round. Test split untouched throughout.
