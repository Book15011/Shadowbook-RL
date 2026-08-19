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
Last updated: 2026-08-19 19:26 HKT
State: reconciliation complete, integration plan proposed, awaiting review -- no code yet.
Confirmed (read-only, against the real src/envs/lob_execution_env.py and reward.py) that
architecture_spec.md Section 4.1's FrozenL3Wrapper/train_l2.py reference code does not match
the real env at all: tier=, l2_override=, seed= (constructor), l2_action_space,
l2_observation_space, apply_l2_action, step_l3, get_l3_obs, get_l2_obs, l2_info() -- all ten
confirmed absent by direct reading. The real env is single-tier (one step() call = one
100ms L3-shaped tick) and has exactly one obs space (42-dim, matches Section 3.1 index-for-
index) and one action space (MultiDiscrete([4,11,5]), matches Section 3.2's L3 spec). The one
piece of real L2-facing infra that DOES already work: l2_target_slice_ratio_override and
l2_urgency are plain public attributes, read fresh every tick -- mid-episode L2 target changes
without reconstructing the env already work today via direct attribute assignment, no new
method needed. Full writeup + proposed wrapper-only integration plan (recommends zero changes
to lob_execution_env.py) in docs/reports/phase4_l2_reconciliation_and_plan.md, including a
flagged correctness gap the Section 4.1 reference code misses entirely: the frozen L3
checkpoint was trained under VecNormalize(norm_obs=True, clip_obs=5.0) (confirmed from
train_l3.py + models/l3_vecnormalize.pkl on disk), so FrozenL3Wrapper must apply those saved
normalization stats before calling l3_model.predict(), or the frozen policy sees
out-of-distribution inputs.
Files owned/in-progress: none yet (read-only). New file added:
docs/reports/phase4_l2_reconciliation_and_plan.md (design doc only, no env/train code).
Blocking/open questions: awaiting sign-off on the integration plan. Separately: several
same-day models/l3_executioner_v1*.zip probe variants exist alongside
models/l3_executioner_v1.zip (the one Section 4.1's train_l2.py would load) -- need to confirm
with the L3 track which checkpoint is the intended "frozen" one before wiring it into
FrozenL3Wrapper.
Next planned step: build src/envs/wrappers.py (FrozenL3Wrapper) and src/train/train_l2.py per
the plan in docs/reports/phase4_l2_reconciliation_and_plan.md, pending approval.

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
