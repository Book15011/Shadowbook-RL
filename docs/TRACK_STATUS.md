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

## L3 / Env-Physics
Last updated: 2026-08-19 20:47 HKT
State: Two real bugs found and fixed in src/envs/lob_execution_env.py, validated (49
lob_execution_env/reward/matching_engine tests pass, plus 2 new regression tests), but
NOT YET COMMITTED -- currently stacked, uncommitted, in the same 5 files alongside two
earlier, separately-uncertain rounds of work (the Part A/B/C staleness signal --
RewardWeights.eta_replace -- and the r_queue MARKET/REPLACE pricing split). (1)
qty_at_price()'s np.isclose call never overrode rtol, so at BTCUSDT's ~$120k price
scale the effective match tolerance was ~$1.2 (rtol*price) instead of the intended
$0.05 half-tick (atol) -- verified directly: it matched 100% of 4,400 synthetic
placements at every tested offset (-5..+5 ticks) regardless of stated distance from
the market, and the matched index was ALWAYS index 0 (the touch), never anything
offset-dependent. Fixed: rtol=0.0 added. (2) _place_limit() had no crossing-order
handling -- a price that crosses the opposing side (common: offset>=+1 ticks crossed
~100% of the time in the same sweep) fell through to the q_ahead lookup and became an
ordinary resting ghost order instead of trading immediately. Fixed: crossing prices
now route through walk_market_fill() against the opposing side's book, same as
ORDER_TYPE_MARKET, returning real fills instead of a resting QueueState. Full
mechanism writeup, the honest re-measurement of the ORIGINAL checkpoint under fixed
physics (fill_ratio 0.590 -> 0.2015, IS_total_bps 0.632 -> -0.1999, the
31/50-beats-TWAP result is NOT statistically significant at z~1.70/p~0.09), and the
init-strategy probe below are all in docs/reports/phase3_l3_baseline_milestone.md.

Init-strategy probe (does the existing 20M checkpoint's weights help warm-start a
fine-tune under the now-fixed physics, vs training from scratch): from-scratch is
ruled out as impractical -- a near-random initial policy exploits the (correct)
crossing fix constantly, terminating episodes in ~11-21 ticks instead of the
3,000-tick horizon, which turns every reset() into the dominant cost. Confirmed NOT a
day-cache sizing issue (_MAX_CACHED_DAYS raised 3->5 with real RAM-budget arithmetic
shown, RAM-safe, verified correct via direct cache-eviction/identity checks -- but
produced no throughput change: fps stayed at 8-10 and ep_len_mean stayed at 11-21
ticks both before and after). Root cause is the reset RATE itself (near-random
exploration x the crossing fix), not I/O; an untested, not-yet-investigated
alternative hypothesis worth checking later (not urgent) is per-reset cost in the
funding-rate lookup path. Warm-start (--warm-start-weights, loading ONLY the original
checkpoint's policy weights, fresh VecNormalize, step counter reset to 0) looks
healthy by contrast: fps ramped to a stable ~350-359 (above the pre-fix reference
run's own steady-state 247), ep_len_mean held at ~3,000 (full horizon) throughout a
~25-minute/499,712-step sample -- no reset-storm at all. Stopped there per
instruction; NOT yet committed to the full 2,000,000-step run pending a go/no-go
decision.

For the L2 track's blocking question above: models/l3_executioner_v1.zip right now IS
the checksum-verified ORIGINAL 20M-step baseline (sha256 94b3ad38...), restored after
every probe this session -- but it was trained entirely under the OLD, buggy
qty_at_price/crossing physics described above, and the init-strategy probe above
exists specifically to decide whether it gets superseded by a fixed-physics run. Do
not treat it as the final "frozen" checkpoint for FrozenL3Wrapper yet -- recommend
waiting for that decision before wiring integration against a specific checkpoint file.

Also for the L1 track: no training process is currently running on this box (verified
just now), so the GPU is free if that unblocks anything on your end -- but note a
go/no-go decision on this track's own full 2,000,000-step warm-start run is pending
and could start at any time once approved.

Files owned/in-progress (all UNCOMMITTED, three stacked rounds mixed in the same 5
files): src/envs/lob_execution_env.py, src/envs/reward.py, src/train/train_l3.py,
tests/test_lob_execution_env_features.py, tests/test_reward.py. IMPORTANT for
accuracy over what any docstring in these files claims: reward.py's
canceled_via_replace branch is CURRENTLY, TEMPORARILY reverted to charging the same
as canceled_via_market (the r_queue MARKET/REPLACE split from an earlier round is
neutralized in the working tree right now, for a clean init-strategy comparison) --
the split's code comments still describe it as active, but it is not, in the current
working tree, until explicitly restored.
Blocking/open questions: (a) commit the qty_at_price/crossing fix on its own,
separated from the still-on-hold staleness/r_queue-split code? It's validated and has
no known downside, but has not been committed pending this check-in. (b) go/no-go on
committing warm-start to the full 2,000,000-step run. (c) after that run (or
independently), restore reward.py's r_queue split back to its real (non-neutralized)
form.
Next planned step: awaiting explicit direction on (a)/(b)/(c) above before proceeding
further.
