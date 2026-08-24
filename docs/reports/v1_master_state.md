# Project v1 state: master snapshot (2026-08-24, frozen before v2 env-optimization work)

**Purpose:** a single, self-contained "as of now" consolidation across all three tracks
(L1 Macro Analyst, L2 Strategist, L3 Executioner), written immediately before this round's
`env.reset()` optimization investigation touches anything. That investigation may change
`lob_execution_env.py`'s internals; this document freezes the record of what "v1" is
first, so any later before/after comparison has a fixed reference point. Nothing below is
new analysis -- it consolidates and cross-references the per-track reports already on
disk rather than duplicating them; follow the links for full detail and methodology.

## L3 -- Executioner: research closed, frozen checkpoint handed off

Full detail: `docs/reports/l3_frozen_handoff.md` (stands alone, read first if picking this
up cold), `docs/reports/phase3_l3_baseline_milestone.md`, `docs/reports/l3_twap_baseline_reward.md`,
`docs/reports/l3_armA_budget_extension.md`, `docs/reports/l3_replace_value_probe.md`.

**Frozen checkpoint:** `models/l3_frozen_backup/l3_executioner_v1_frozen.zip` (sha256
`a5443e2a4c6c1d4427d4ce1cb83e65d622ea688d8953f5bf94b29e87fbcaa77d`) /
`l3_vecnormalize_frozen.pkl` (sha256 `b459e17784c239be48069c47a7da6454610b4674a99e5d513d3ef0b616c182d8`).
This is "Arm A" -- warm-started from v1's step-2,000,000 checkpoint (true v1 final save is
permanently unrecoverable, see below), then trained 1,000,000 further steps with
`subtract_twap_baseline=False`, r_queue direction inverted (de facto inherited, not a
deliberate choice -- see the handoff doc's own explicit warning on this).

**Full n=500 results table** (paired-seed methodology, seeds 5,000,000-5,000,499, held-out
`val` split; lower IS_total_bps is better):

| Checkpoint/policy | IS_total_bps mean (std) | fill_ratio | vs TWAP: mean diff | paired t | Wilcoxon |
|---|---|---|---|---|---|
| TWAP (fixed baseline) | 0.889 (4.353) | 0.994 | -- | -- | -- |
| **Arm A -- THE FROZEN CHECKPOINT** | **0.994 (3.570)** | **0.919** | +0.105bps | p=0.534 | p=0.653 |
| 500k-step snapshot (context only) | 1.025 (3.230) | 0.949 | +0.135bps | p=0.396 | p=0.358 |
| best-B heuristic (scripted, not RL) | 1.103 | 1.000 | +0.214bps | p=0.101 | p=0.191 |
| v1 (original 2M-step checkpoint) | 1.261 (4.242) | 0.892 | +0.372bps | **p=0.033** | p=0.115 |
| Arm B (TWAP-baseline reward, treatment) | 1.341 (2.405) | 0.990 | +0.452bps | **p=0.009** | **p=0.014** |
| Budget extension (Arm A + 2M more steps) | 1.237 (2.039) | 1.000 | +0.347bps | **p=0.034** | **p=0.044** |

**Honest performance statement, stated plainly in the handoff doc and repeated here so it
isn't lost in a future skim:** the frozen checkpoint TIES TWAP (p=0.534/0.653); it does not
beat it. Nothing evaluated across this entire project -- not this checkpoint, not v1, not
either A/B arm, not the budget extension, not the hand-tuned scripted heuristic -- beats
TWAP at proper n. The frozen checkpoint is the least-bad of everything tried.

**The real engineering win is the fill-ratio arc, not reward shaping**: 0.2015 pre-retrain
-> 0.919 (this checkpoint) -> 0.9998 (budget extension, not recommended for other reasons).
Attributable to earlier physics/matching-engine fixes, not any reward-shaping intervention
in the table above -- all of zeta/eta_replace/r_queue-split/r_queue-inversion/
subtract_twap_baseline left fill_ratio in the same 0.89-1.00 band regardless of which was
active.

**Why research closed:** 8 in-training checkpoints across the budget-extension run show a
plateau, not a trend (best point at 500k steps in, never bettered across the remaining
1.5M). Three consecutive reward interventions since Arm A (r_queue inversion, TWAP-baseline
variance-reduction reward, the budget extension itself) all returned null or significantly
negative results -- see "What was tried and didn't work" below.

**Known open items, not resolved by this project:**
- Original-vs-inverted r_queue direction never cleanly compared head-to-head (only each
  vs. TWAP separately, at n=50, underpowered).
- Single training seed throughout the entire lineage -- no finding in this project's
  history (including the TWAP-tie itself) has been replicated under an independent seed.
- True v1 checkpoint (`973b2883...`) is permanently unrecoverable, overwritten by a since-
  fixed save-path bug; the step-2,000,000 stand-in (`27afa91e...`) is numerically verified
  equivalent and is what every later run in this lineage descends from.

## L1 -- Macro Analyst: complete

Full detail: `docs/reports/l1_real_llm_validation.md`, `docs/reports/l1_async_threading_validation.md`,
`docs/TRACK_STATUS.md`'s L1 section.

All three round tasks done: (1) `SYSTEM_PROMPT` states every field/range explicitly and
`maybe_refresh()` sends Ollama a real structured-output schema instead of a bare `"json"`
string -- fixes the real failure mode (invented field names); re-validated live, 10/10
schema-conformant, mean latency 4.505s -> 1.598s. (2) `AsyncL1Refresher` built --
background-thread wiring so L1 never blocks the tick loop, fail-closed by construction, 10
new tests. (3) Live-validated in the full stack: idx 17/18 change asynchronously as
designed, per-tick cost 3.81ms (stubbed) -> ~5.5-6.3ms (threaded, real) vs. 14.60ms
(synchronous, prior round) -- threading recovers ~60-70% of the gap, reported as a range,
not oversold as full parity.

**Real infra state, confirmed live (as of the correctness-harness report,
`docs/reports/l1l2l3_integration_correctness.md`, 2026-08-23, i.e. before the async
threading validation above landed):** the Ollama systemd service's model visibility and
proxy-env-var misrouting bugs are both fixed; real (unmocked) LLM calls succeed. 14B model
recommended over 32B, no remaining argument for 32B found.

**One unreproduced anomaly, disclosed not dropped:** first live run showed
`last_refresh_completed_tick` set correctly but its paired `on_l1_tick` callback never
fired -- not reproduced across 2 immediate reruns or 6 passing deterministic regression
tests. Flagged open, not blocking.

## L2 -- Strategist: built, wired, correct on inspection -- not yet trained

Full detail: `docs/reports/phase4_l2_reconciliation_and_plan.md` (CURRENT STATE section
at the top is self-contained), `docs/TRACK_STATUS.md`'s L2 section.

**What's built:** `src/envs/wrappers.py::FrozenL3Wrapper` (the full L2/L3 integration
layer -- action-space transform, observation downsampling, VecNormalize applied to the
frozen L3 policy's own inputs, explicit LSTM state/`episode_start` threading), and
`src/train/train_l2.py` (SAC wiring, CLI requires `--l3-checkpoint`/`--l3-vecnormalize`/
`--total-timesteps` explicitly -- no defaults, so a real run can't launch by omission).
21 fast unit tests + 1 gated integration smoke test, all passing. A 200-step smoke test
ran end-to-end with no shape/interface errors -- validated wiring, not policy quality.

**Checkpoint identity is resolved** (`docs/reports/l3_frozen_handoff.md`, verified
compatible by direct code-reading against `wrappers.py`'s own assumptions -- no mismatch
found in observation space, action space, or VecNormalize config). No code changes needed
on L2's side to use it.

**Why L2 has not trained: throughput, not correctness.** Single-env: `env.reset()` alone
is 51.0% of wall-clock (7 calls, 2083.8ms/call measured via monkeypatch profiling of the
real production path), `L3.predict()` 35.5% (5,735 calls, 1.769ms/call), `env.step()`
6.1%, `SAC.train()` 5.9% -- rate 4.194 decisions/sec, single env, GPU L3 inference.
Extrapolated: ~5.5 days for a real 2,000,000-step run at that rate -- not practical to run,
especially not blind.

**Parallelization was investigated as the fix, and it is a real but insufficient win.**
Controlled benchmark (fixed seed=42, fixed 10-day date pool, thread-capped, CoV 0.1-1.3%
across 12 runs -- trustworthy, unlike an earlier noisy attempt that disagreed >2x run to
run before seed/date_range were controlled): n_envs=1/2/4/8 -> 5.651/6.865/8.852/9.729
decisions/sec (1.00x/1.21x/1.57x/1.72x speedup, 100%/60.7%/39.2%/21.5% efficiency -- real,
reproducible, clearly sub-linear at every step). Real RAM/VRAM measured, not assumed:
26.2GB RSS / 3.6GB VRAM at n_envs=8 (genuinely less RAM than L3's own n_envs=8 PPO budget
of ~38.8GB -- confirmed different, not assumed to match).

**Extrapolated 2,000,000-step wall-clock at the best measured configuration (n_envs=8):
2.38 days.** Per the stated guidance (under ~1 day workable, 1-2 days marginal, beyond
that needs rethinking): **go/no-go = NO**, even at the best configuration -- and the
extrapolation's own caveats (this benchmark's narrow, repeated date_range likely benefits
from warm OS page cache that a real 405-day-diverse run wouldn't get; policy-behavior
drift as L2 actually learns is untested by a short benchmark) both lean toward this being
an optimistic reading, not pessimistic. **Parallelizing envs alone, at the scale tested,
does not make a full 2,000,000-step L2 training run practical.** This is the throughput
finding driving this round's `env.reset()` investigation.

**Integration correctness (data flow only, not a training/performance run):**
`docs/reports/l1l2l3_integration_correctness.md` confirms the full three-tier orchestrator
wires correctly end-to-end with correct cadences. L2's live-attribute path into L3
(`l2_target_slice_ratio_override`/`l2_urgency` -> obs idx 15/16) is confirmed working;
L1's path into L3 (obs idx 17/18) is correctly wired in the base env but **not currently
driven by `FrozenL3Wrapper`** -- a real gap (L1 signal doesn't reach L3 through L2 yet),
not a bug, and not blocking (matches L1's own real-Ollama integration being a separate,
not-yet-wired-into-L2 step).

## Everything tried that didn't work (consolidated across all tracks)

**L3 reward/training interventions**, all after the frozen checkpoint's own training,
all null or negative (detail in `l3_frozen_handoff.md`'s "What was tried" section):
1. r_queue MARKET/REPLACE split -- didn't move REPLACE usage materially.
2. r_queue direction inversion -- no significant difference either way, underpowered.
3. TWAP-baseline variance-reduction reward -- reduced variance as designed, but the
   treatment arm was significantly WORSE on execution quality (p=0.009/0.014).
4. Budget extension (+2M steps) -- null result, no convergence across 8 in-training evals,
   volatile throughout.
5. CANCEL_AND_REPLACE usage sweep (`l3_replace_value_probe.md`) -- best scripted
   REPLACE-active heuristic across an 18-config sweep came in numerically worse than TWAP;
   near-0% REPLACE usage in the trained policy is correct behavior, not a learning failure.

**L2 throughput engineering:**
6. Uncapped-thread parallelization -- catastrophic regression (n_envs=2: 0.14x, i.e. ~7x
   SLOWER; n_envs=4: 0.11x, ~9x slower), root cause thread oversubscription, confirmed via
   `ps aux` (~375% CPU per worker), not assumed.
7. Uncontrolled seed/date_range benchmarking -- two runs of the identical n_envs=2 config
   disagreed by >2x (3.38 vs 8.53 dec/sec) before this was diagnosed and fixed.
8. Parallelizing envs alone (the controlled, trustworthy result) -- real 1.72x speedup at
   n_envs=8, but insufficient to bring a 2M-step run under ~2 days; see above.

**L1 infra:**
9. Ollama model pulled under the wrong OS user context -- invisible to the actual running
   systemd service; not a code bug, an infra misconfiguration, fixed by relocating the
   blob store.
10. `http_proxy`/`https_proxy` env vars silently intercepting real Ollama calls,
    indistinguishable from a genuine outage under the existing exception handling --
    fixed with an explicit `proxies={}` override.

## What this document is not

Not a new analysis, not a v2 change, not a decision about what comes next -- it is the
fixed "before" snapshot this round's `env.reset()` optimization investigation is measured
against. See `docs/reports/phase4_l2_reconciliation_and_plan.md`'s "What's currently
blocking" section for the open question this snapshot feeds into.
