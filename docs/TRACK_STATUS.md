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
