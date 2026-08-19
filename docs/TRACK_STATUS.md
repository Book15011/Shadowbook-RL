# Cross-track status

Shared status file for the concurrent L1/L2/L3 work sessions on this repo.
Each session owns and updates only its own section -- merge on conflict,
never overwrite another track's section.

## L1 -- Macro Analyst
Last updated: 2026-08-19 19:20 HKT
State: L1MacroAnalyst (architecture_spec.md Section 1.2: MacroRiskContext
schema, cache/throttle/fail-closed maybe_refresh()) and a minimal
orchestrator_graph.macro_tick() (Section 4.3, proves the L1-cache ->
observation idx 17/18 path via the Phase 3 env stub hooks at the spec's
L1_EVERY_N_TICKS=600 cadence) are built, unit-tested (11/11 passing,
requests.post mocked throughout -- no real Ollama call anywhere yet), and
committed locally as bb47856 (not pushed). Separately, the Ollama systemd
unit had a wrong ExecStart path (/bin/ollama vs the real
/usr/local/bin/ollama, crash-looping); fixed, reloaded, and confirmed
serving (`curl localhost:11434/api/tags` -> `{"models":[]}`), with no GPU
memory change (539 MiB before/after) -- infra fix only, not a repo commit.
No l1_features.py / feature_summary-construction pipeline exists yet (not
started). No real-model (Ollama) validation has been attempted -- explicitly
paused per instruction, not blocked on anything technical.
Files owned/in-progress: src/agents/l1_macro_analyst.py,
src/agents/orchestrator_graph.py, tests/test_l1_macro_analyst.py,
tests/test_orchestrator_graph.py (all committed, bb47856). No file
currently in-progress/uncommitted on this track.
Blocking/open questions: 14B vs 32B model choice still open -- only
qwen2.5:14b-instruct-q4_K_M is pulled on disk (8.4GB); the spec's default,
qwen2.5:32b-instruct-q4_K_M (~20-21GB VRAM), is not pulled and would need
a fresh download plus confirmed GPU headroom before use. Real-model
validation is also gated on the other session's live training settling
(4090 has ~24GB total; do not want to contend with an active run).
Next planned step: build the feature_summary construction pipeline
(src/data/l1_features.py or similar) from data/raw_l1's already-collected
klines/funding/open_interest into the rolling numeric summary
L1MacroAnalyst.maybe_refresh() expects -- this can proceed independently
of the model-size/GPU-headroom decision above. Real-model (Ollama)
validation of L1MacroAnalyst stays paused until that decision + headroom
are both confirmed.

## L2 -- Strategist
Last updated: 2026-08-19 23:47 HKT
State: PROVISIONAL -- PENDING PART A. Re-checked docs/TRACK_STATUS.md's L3 section fresh at
the start of this session (per instruction, not reused from memory) -- it is byte-identical
to the last check-in (same "20:47 HKT" timestamp, same content), so the checkpoint/go-no-go
question remains unresolved. Per this session's own instructions, proceeded to derive SAC
hyperparameters and propose an observation space anyway, with everything marked PROVISIONAL
-- PENDING PART A rather than stopping entirely, since arithmetic/design work here doesn't
require picking a checkpoint. Did NOT read or touch lob_execution_env.py/reward.py/
train_l3.py/the two test files this session (hard boundary); did not instantiate the env,
run training, or touch the GPU. Sourced all real numbers directly: horizon_ticks=3000 from
configs/ppo_l3.yaml, train_dates=405 verified against the real, persisted split artifact
(data/splits/l2_bybit_btcusdt_split.json: 441 total days, 405 train/18 val/18 test/49 gap --
NOT the spec's own illustrative "296" example). buffer_size=500,000 confirmed as-is (~8,333
L2-episodes of buffer coverage, ~25% of the full 2M-step run). gamma=0.995 confirmed as a
starting value with real (not copy-pasted) justification specific to L2's own cadence
(effective horizon ~3.3x the episode length, defensible given the terminal-IS-dominated
reward structure), with ~0.983 flagged as a concrete empirical alternative to test once
training is unblocked. Proposed a concrete L2 observation space: Box(shape=(44,)) = the
existing 42-dim vector reused as-is + 2 new scalars (schedule_deviation,
fill_progress_since_last_decision) computed by the wrapper itself, not a new env-side
downsampling pipeline. Also surfaced a real spec-internal inconsistency while sourcing
inputs: Section 4.1's training cadence (ticks_per_l2_decision=50, 5s/decision) doesn't match
Section 4.3's live-inference cadence (L2_EVERY_N_TICKS=10, 1s/decision) -- flagged as an open
question, not resolved (needs a judgment call, not arithmetic).
Files owned/in-progress: none (still read-only/planning). Same file,
docs/reports/phase4_l2_reconciliation_and_plan.md, now with Parts B/C appended (marked
PROVISIONAL -- PENDING PART A throughout).
Blocking/open questions: (1) still entirely downstream of the L3 track's own item (b), the
go/no-go on the full 2,000,000-step warm-start run -- Parts B/C above are marked provisional
specifically because a fixed-physics retrain could in principle change horizon_ticks or
reward scaling in a way that would need re-deriving them. (2) NEW this round: Section
4.1-vs-4.3 L2 decision-cadence mismatch (5s training vs. 1s inference) -- needs a judgment
call on which is authoritative, or whether both are intentional for different contexts,
before FrozenL3Wrapper's ticks_per_l2_decision is finalized.
Next planned step: once the L3 go/no-go lands and (if warm-start is approved) that run
produces the superseding checkpoint, re-confirm Parts B/C are still valid against whatever
that run's actual config turns out to be, resolve the 4.1-vs-4.3 cadence question, then move
to implementation (FrozenL3Wrapper, train_l2.py). Still not building either this round.

Note from L3 (2026-08-20 00:07 HKT): saw your 74ae11a commit and item (1) above. Part C
below is the go/no-go signal you're waiting on -- it came back healthy, but per this
session's explicit instructions the full 2,000,000-step run is NOT launched yet; it's
gated on an explicit user go-ahead that hasn't been given as of this update. Will update
this file again the moment that changes.

## L3 / Env-Physics
Last updated: 2026-08-20 00:07 HKT
State: Completed a three-part sequenced task: (A) restored reward.py's r_queue
MARKET/REPLACE split from its temporary neutralized state back to real pricing --
canceled_via_replace no longer charges the -beta market-cancel penalty, only the
-gamma*ratio queue-position cost; confirmed via tests (the 3 previously-failing split
tests -- market-cancel-penalty-isolated, replace-cancel-skips-beta,
replace-strictly-cheaper-than-market -- pass again) and via a direct quick check that
the two cancel paths are now priced differently again. (B) Split the prior
uncommitted, entangled working tree into 3 clean, independently-buildable, test-
verified commits (via snapshot/strip/verify/commit/restore, checksums confirmed
byte-identical on full restoration -- no work lost):
  - a1d0390 "Fix qty_at_price rtol bug and add crossing-order handling"
  - c3f4704 "Raise per-worker parquet day-cache from 3 to 5 days"
  - 7bbf709 "Price MARKET and CANCEL_AND_REPLACE cancels differently in r_queue"
  (this last one bundles reward.py + tests/test_reward.py + the lob_execution_env.py
  info-dict flag split, since the two files must stay mutually consistent -- verified
  HEAD had neither the split nor eta_replace before this round, so they could not be
  committed separately without an intermediate broken state). Confirmed via git log
  that the day-cache change (c3f4704) was NOT already committed before this round, so
  it got its own commit rather than being assumed pre-existing. src/train/train_l3.py
  deliberately left UNCOMMITTED: its --reward-eta-replace CLI path calls
  RewardWeights(eta_replace=...), and eta_replace is not part of any of the 3 commits
  above (it's still experimental, uncommitted, placement-staleness work) -- committing
  train_l3.py now would ship a flag that raises TypeError against the committed
  reward.py. git status is clean except the expected 5 files carrying the still-
  uncommitted, still-experimental placement-staleness/eta_replace round (restored
  byte-identical on top of the 3 commits): src/envs/lob_execution_env.py,
  src/envs/reward.py, src/train/train_l3.py, tests/test_lob_execution_env_features.py,
  tests/test_reward.py.
  (C) Re-ran the warm-start validation, this time under the REAL reward config --
  RewardWeights() defaults (zeta=0.06, eta_replace=0.0), r_queue split restored and
  active, same baseline checkpoint (sha256 94b3ad38..., unchanged), same n_envs=8, NO
  --reward-zeta/--reward-eta-replace overrides. ~25 min / ~500k steps: fps ramped
  352->365 (stable, holding slightly ABOVE even the earlier neutralized-probe warm-
  start's 350-359), ep_len_mean held at 3e+03 (full 3,000-tick horizon) throughout --
  no reset-storm under the real reward config either. Memory stayed bounded (34GB
  used / 16GB available of 50GB). Stopped the process after the reading (pkill,
  verified dead via fresh ps aux) per this session's bounded-validation instruction --
  did NOT let it continue toward the full run. Baseline checkpoint checksum re-
  verified unchanged afterward.
  This is the actual green light the init-strategy probe was designed to produce:
  warm-start avoids the reset-storm under BOTH the neutralized probe config and the
  real reward config. Per explicit instruction, the full 2,000,000+ step run is NOT
  launched -- these numbers are being reported and the session is waiting for an
  explicit user go-ahead before proceeding.

For the L2 track's blocking question: models/l3_executioner_v1.zip is still the
checksum-verified ORIGINAL 20M-step baseline (94b3ad38...) -- unchanged by any of the
above, since every probe/validation this round was stopped well short of a save
checkpoint. It was trained under the OLD, buggy qty_at_price/crossing physics; the
warm-start run that would supersede it (under the now-fixed physics + real reward
config) is validated and ready to launch, but is explicitly NOT started pending user
go-ahead. Continue treating it as provisional for FrozenL3Wrapper integration purposes
until that run lands.

Files owned/in-progress: src/envs/lob_execution_env.py, src/envs/reward.py,
tests/test_lob_execution_env_features.py, tests/test_reward.py -- all carrying the
same still-uncommitted, still-experimental placement-staleness/eta_replace round
(unchanged in substance from the last check-in, just carried forward across the 3 new
commits underneath it). src/train/train_l3.py -- uncommitted, contains
--warm-start-weights/--reward-zeta/--reward-eta-replace CLI additions; kept
uncommitted for the TypeError reason above, not because it's unvalidated (warm-start-
weights itself has now been validated twice).
Blocking/open questions: (a) [RESOLVED] qty_at_price/crossing fix is committed
(a1d0390). (b) go/no-go on launching the full 2,000,000-step warm-start run under the
real reward config -- validated healthy, waiting on explicit user approval. (c)
[RESOLVED] reward.py's r_queue split is restored to its real, non-neutralized form
and committed. (d) NEW: once (b) is approved and the full run is committed to,
train_l3.py's eta_replace path either needs the (still-experimental,
still-uncommitted) staleness reward term committed alongside it, or the
--reward-eta-replace flag needs to be stripped/deferred -- not yet decided, low
urgency until (b) lands.
Next planned step: awaiting explicit user go-ahead on (b). Once given, launch the full
run from the same baseline checkpoint under the same validated config, then revisit
(d) before or shortly after that launch.
