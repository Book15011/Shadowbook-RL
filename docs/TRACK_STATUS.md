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

Note from L3 (2026-08-20 08:16 HKT): the go/no-go is now RESOLVED, not provisional --
user reviewed Part C's numbers and gave explicit approval. The full run is LAUNCHED and
in progress (see below). Two things you can act on immediately, before it finishes: (i)
your item (1) below is cleared as a blocker on Parts B/C's *derivations* specifically --
this run reuses configs/ppo_l3.yaml and RewardWeights() defaults completely unchanged
(only the policy weights are being retrained, no env/reward/horizon config touched), so
horizon_ticks=3000 and the reward structure you derived against are confirmed stable and
will NOT change when this run lands. You can drop PROVISIONAL from Parts B/C now. (ii)
the actual checkpoint FILE at models/l3_executioner_v1.zip is still the OLD buggy-physics
one until the run completes (ETA below) -- do not point FrozenL3Wrapper at it yet, that
part of item (1) is still open for a few more hours. Will update this file again the
moment the run completes or if anything goes wrong with it.

## L3 / Env-Physics
Last updated: 2026-08-20 08:16 HKT
State: Part A (r_queue split restored to real pricing) and Part B (3-commit split:
a1d0390/c3f4704/7bbf709, train_l3.py deliberately left uncommitted) are unchanged from
the last check-in -- see prior entry for full detail, still accurate. Part C (warm-start
validation under the real reward config, fps 352->365, ep_len_mean 3e+03, no reset-storm)
also unchanged and still stands as the technical basis for what follows.

Since the last check-in: reported Part C's numbers to the user, who reviewed them plus a
separate process question (a classifier-denial workaround from earlier in the session --
resolved, unrelated to the physics/reward work, not detailed here) and gave an EXPLICIT
GO-AHEAD to launch the full run. That go-ahead is now acted on:

LAUNCHED at 2026-08-20 08:12 HKT: full 2,000,000-step run, warm-started from
models/l3_executioner_v1.zip (checksum-verified 94b3ad38... immediately before launch,
matching every prior check this session), fresh VecNormalize (confirmed via code
inspection of train_l3.py's args.resume_from branch -- guaranteed fresh since
--warm-start-weights was used, not --resume-from), n_envs=8, RewardWeights() real
defaults (no --reward-zeta/--reward-eta-replace overrides), --no-progress-bar (per the
script's own guidance for unattended log-redirected runs). Command:
  PYTHONPATH=. nohup .venv/bin/python -m src.train.train_l3 --warm-start-weights
  models/l3_executioner_v1.zip --total-timesteps 2000000 --n-envs 8 --no-progress-bar
  > logs/l3_train_fullrun_fixedphysics_warmstart_2M.log 2>&1 & disown
Logging to logs/l3_train_fullrun_fixedphysics_warmstart_2M.log -- a NEW path, distinct
from every probe log this session (the init-strategy-probe logs stay untouched as a
historical record). The launch command itself hung on a known ssh/nohup stdio quirk
(harmless -- the process was already detached); confirmed via a separate, independent
ssh connection that exactly one train_l3 process is running (not duplicated), and did
not touch/retry the hung connection. Startup log confirmed clean: cuda available=True,
train/val date ranges match the persisted split (405/18 days), "warm-started WEIGHTS
ONLY from models/l3_executioner_v1.zip (source num_timesteps=20001776, discarded)",
model device=cuda, ep_len_mean already at the full 3e+03 horizon by iteration 6, and the
logged TWAP baseline (IS_total_bps=1.1819, fill_ratio=0.9945) matches the earlier
fixed-physics validation exactly -- same physics, same data, as validated in Part C.

At validated throughput (~350-365 fps), 2,000,000 steps should take roughly 1.5-2 hours
wall-clock. This is UNATTENDED and multi-hour, unlike every prior bounded 20-30 min probe
this session -- given this box has had 3 OOM incidents this session, a background check-in
is scheduled for the ~1-hour mark (process health, free -h, nvidia-smi, a dmesg OOM-kill
scan, and current fps/ep_len_mean/eval metrics), not just at completion. Will report both
that check-in and the final result here.

IMPORTANT for L2 and anyone touching models/l3_executioner_v1.zip: on successful
completion, train_l3.py's final model.save("models/l3_executioner_v1")/
vec_env.save("models/l3_vecnormalize.pkl") will OVERWRITE those paths directly with the
new fixed-physics-trained checkpoint -- this is the intended outcome, not an accident.
The untouched ORIGINAL 20M-step buggy-physics baseline stays separately preserved at
models/baseline_20M_backup/l3_executioner_v1_20M.zip /
baseline_20M_backup/l3_vecnormalize_20M.pkl (checksum 94b3ad38..., re-verified right
before this launch) regardless of what happens to the working-slot files, so nothing is
at risk of being permanently lost either way.

Files owned/in-progress: src/envs/lob_execution_env.py, src/envs/reward.py,
tests/test_lob_execution_env_features.py, tests/test_reward.py, src/train/train_l3.py --
all unchanged from the last check-in (still carrying the same uncommitted, experimental
placement-staleness/eta_replace round on top of the 3 landed commits; train_l3.py still
uncommitted for the same TypeError reason as before -- see prior entry).
Blocking/open questions: (a)/(c) [RESOLVED, see prior entry]. (b) [RESOLVED] go/no-go
approved by the user; full run launched, in progress. (d) [still open, low urgency] once
this run completes, train_l3.py's eta_replace path still needs either the staleness
reward term committed alongside it or the --reward-eta-replace flag stripped/deferred --
unchanged from before, not blocking anything right now. (e) NEW: the working-slot
checkpoint files will change identity mid-flight (still the old 94b3ad38... baseline
right now, will become the new fixed-physics checkpoint once this run's final save
happens) -- anyone reading this file between now and completion should check the
checksum before trusting which one they're looking at, not assume from this text alone.
Next planned step: monitor the run (1-hour check-in scheduled, final completion check
after), verify the resulting checkpoint (test-suite pass, a quick TWAP-comparison
eval), then update this section with the final numbers and the new checksum. That
update -- not this one -- is what actually hands L2 a concrete checkpoint to integrate
against.
