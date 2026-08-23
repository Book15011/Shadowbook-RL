# Cross-track status

Shared status file for the concurrent L1/L2/L3 work sessions on this repo.
Each session owns and updates only its own section -- merge on conflict,
never overwrite another track's section.

## L1 -- Macro Analyst
Last updated: 2026-08-24 00:07 HKT
State: REAL (UNMOCKED) LLM PATH VALIDATED THIS ROUND -- full detail in
docs/reports/l1_real_llm_validation.md. Two real bugs found and fixed, one infra,
one code: (1) Ollama model-visibility (systemd service user ollama's own
~/.ollama/models was empty; the pulled 8.4GB qwen2.5:14b sat under
/home/ubuntu/.ollama instead, from before the ExecStart fix landed) -- fixed by
moving the blob store into the service's own default location (same filesystem,
instant mv, zero duplicate disk usage) and chown -- infra only, not a repo commit,
same precedent as the earlier ExecStart fix. (2) L1MacroAnalyst.maybe_refresh()
itself silently misrouted every real call through this host's http_proxy/
https_proxy env vars (ProxyError, indistinguishable from a real Ollama outage
under the existing except clause) -- fixed in code (proxies={} override),
committed 98e6cc1 with a regression test, 8/8 L1 tests still pass.

Both fixes verified with real calls, not service status alone: curl api/tags
lists the model; curl api/generate returns a real completion; VRAM measured
(15,033 MiB loaded+generating / 9,107 MiB idle-loaded / 24,564 MiB total).

Step 2 (5 real, unmocked maybe_refresh() calls, real build_l1_feature_summary()
input, 5 different real market dates spanning 2024-2026): 0/5 raw responses were
schema-conformant against MacroRiskContext -- the model returns syntactically
valid JSON (format:json only guarantees that much) but invents its own field
names every time, never the required timestamp_ms/regime/risk_score/confidence/
urgency_multiplier keys. Root-caused, not just observed: SYSTEM_PROMPT never
actually states the required schema/keys anywhere in the text sent to the model
-- this points at a PROMPT-DESIGN gap, not primarily a 14B-capability one.
Separately, and reported as the genuinely positive half of the same result: the
fail-closed mechanism was 5/5 reliable under this real, repeated failure --
correct neutral fallback every time, no crash, no garbage propagated. Real call
timing: 4.49-4.54s (mean 4.505s, tight variance), warm model, real prompt size.

Step 3 (14B vs 32B): recommend staying on 14B, NOT pulling 32B yet -- current
measured headroom is real (~9.3GB free with 14B loaded+generating, L2/L3 both
confirmed idle at measurement time) but the stronger reason is that 32B is
unlikely to fix Step 2's own finding (the schema is never shown to either model
size) and would commit ~20-21GB of shared 4090 VRAM on a weak bet. Recommended
next step instead: fix SYSTEM_PROMPT to actually state the required keys, retest
against the already-available 14B at zero additional cost, before spending
anything on a 32B pull. Not done this round -- a prompt/content judgment call,
flagged for direction rather than made unilaterally.

Step 4 (full 3-tier stack re-run, real L1 live, real val-split data, real frozen
L3 + L2 smoke checkpoints): runs correctly end to end -- 1249 ticks, L1 fires
exactly at 0/600/1200 (same cadence as stubbed), L2 fires 25x (same count as the
prior correctness run), truncates with finite IS_total_bps=5.53 and fill_ratio
in [0,1]. idx 17/18 did NOT visibly change at boundaries this run -- all 3 real
firings hit the identical neutral fallback, consistent with Step 2's 0/5 result,
reported plainly rather than reframed. Per-tick cost: 3.81ms (stubbed, prior
round) -> 14.60ms (real L1, this round), ~3.8x over this short episode --
matches a direct amortization calculation (3 calls x ~4.5s = ~13.5s added,
accounts for the gap almost exactly) rather than an unexplained regression.
Confirms the task's own suspicion directly: at 4.5s/call vs a ~2.29s wall-clock
budget between L1 firings (600 ticks at the stubbed 262 ticks/sec), a
SYNCHRONOUS real LLM call is a genuine new bottleneck. The fix is architectural,
already spec-sanctioned (Section 1.2: run L1 from a background thread, tick loop
reads self._cache only) and independent of model choice -- a bigger model would
likely make synchronous blocking WORSE, not better. Not built this round --
flagged as the concrete next step alongside the prompt fix above.

Not added to the committed pytest suite: the Step 4 live-L1 run, since it needs
a real running Ollama service -- would break the hermeticity every other test in
this file preserves by mocking requests.post. Run as a one-off validation
script; results captured in the report doc instead.

Files this round: src/agents/l1_macro_analyst.py, tests/test_l1_macro_analyst.py
(proxy fix + regression test, 98e6cc1). docs/reports/l1_real_llm_validation.md
(e7f8851). orchestrator_graph.py/test_orchestrator_graph.py unchanged this round
(Step 4 was a validation script, not new test code, per the hermeticity note
above). Ollama blob-store move: infra only, documented above and in the report,
not a repo commit.

Blocking/open questions: (a) NEW -- SYSTEM_PROMPT needs the actual required
schema/keys stated explicitly before real L1 output can ever be usable, at
either model size; this is now the primary blocker on real L1 signal quality,
ahead of the 14B/32B question. (b) NEW -- L1's synchronous call in
orchestrator_graph.run_episode() needs to move to a background thread/process
per the spec's own design, before a live L1 stops being a throughput bottleneck
in any real integration run. (c) 14B vs 32B: recommend 14B for now (Step 3), but
this is a recommendation, not a decision made here -- open for confirmation.

Next planned step: two independent options, neither gated on the other -- (a)
fix SYSTEM_PROMPT to state the required schema explicitly and re-run Step 2's
same 5-case robustness check against 14B; (b) build the background-thread/async
wiring for L1 in orchestrator_graph.py so a live LLM call never blocks the tick
loop. Recommend (a) first, since it is what actually determines whether real L1
signal is usable at all -- (b) matters only once (a) is producing real,
non-fallback values worth not blocking on.

PRIOR ENTRY BELOW, for context on the full orchestrator build:

## L1 -- Macro Analyst
Last updated: 2026-08-23 20:40 HKT
State: FULL L1->L2->L3 ORCHESTRATOR BUILT AND CORRECTNESS-VERIFIED this round --
full detail in docs/reports/l1l2l3_integration_correctness.md, committed 941aa0e
(orchestrator code+tests: 55e9b09). This is a correctness result, not a
performance/training one -- no training run, no long run.

Step 0 prerequisites, checked directly rather than trusted from this section's own
prior (partly stale) entries: (a) src/data/l1_features.py works -- verified live
against real data/raw_l1 at an in-range timestamp, real non-None numeric output,
feeds L1MacroAnalyst.maybe_refresh() cleanly. (b)/(c) real Ollama is BLOCKED by a
genuine infra bug, not just the open 14B/32B decision: the systemd service runs as
system user `ollama` (home /usr/share/ollama), whose own .ollama/models/{blobs,
manifests} are completely empty -- the 8.4GB qwen2.5:14b-instruct-q4_K_M pull
actually sits under /home/ubuntu/.ollama instead, invisible to the running service.
Confirmed live: a real generate call returns {"error":"model ... not found"}. Not
fixed this round (out of scope -- infra repair, not the correctness harness), but
now a precisely diagnosed, not just "still open," finding for whoever picks it up
(needs either OLLAMA_MODELS pointed at the right path, or a re-pull as the ollama
user). (d) L2's throughput measurement DID land (commit 20ce3b6, this section's own
prior entry was stale on this) -- ~4.15 L2-decisions/sec single-env, ~5.5 days
extrapolated for a real 2M-step run, verdict already "not practical," parallelize
across envs recommended, not yet built.

Step 1 (L1 wiring gap L3's handoff flagged): closed with NO change needed inside
wrappers.py -- FrozenL3Wrapper exposes the raw env via gym.Wrapper's own public
.env attribute, the same pattern the wrapper itself already uses internally for
idx 15/16, so macro_tick(l3_wrapper.env, ...) reaches l1_risk_score/l1_confidence
cleanly, unchanged function.

Steps 2-3 (orchestrator_graph.py extended to the full graph + correctness test):
executioner_node/env_step_node are folded into FrozenL3Wrapper.step() rather than
reimplemented at the orchestrator level (would duplicate L2's own already-tested
LSTM-threading/normalization logic with real divergence risk) -- treated as an
invariant proven by reading the source, then independently verified empirically.
New integration test loads the REAL frozen L3 checkpoint + L2's real smoke-test SAC
checkpoint, runs a short bounded synthetic episode (1250 ticks), and asserts: L1
fires exactly at ticks 0/600/1200 with 3 distinct mocked values, L2 fires in an
exact 50-tick arithmetic sequence, L3/env_step fires every tick (1250 real, gapless
env.step() calls, verified via a non-invasive monkeypatch recorder -- no edit to
lob_execution_env.py/wrappers.py), idx 15/16 and 17/18 each hold constant between
their own cadence boundaries and change only at them, the frozen checkpoint's
checksum matches the handoff doc's recorded value verified live, episode truncates
correctly with a sane implementation_shortfall. All assertions pass. Found and
documented (not silently worked around) a real, benign off-by-one: LOBExecutionEnv.
step()'s end-of-ticks-buffer clamp fires on exactly the horizon-truncating tick,
making info["ticks_elapsed"] under-report by 1 on that one final call only. Full
suite: 132 passed, same 4 pre-existing unrelated failures as every check this
session.

Step 4 (throughput + feasibility verdict): measured, real data, pure inference (no
training): 3.81ms/tick, 262 ticks/sec, 5.24 implied L2-decisions/sec, single env.
Combined with L2's own 4.15 dec/sec (real training, no L1) -- not apples-to-apples
(mine=inference-only+L1-stubbed, theirs=training+backprop+no-L1) but both land in
the same ~4-5/sec band, pointing at the SAME bottleneck L2 already found
(env.reset()/step() I/O, not gradient updates, not L1). Plain verdict: three-tier
training remains impractical at today's single-env throughput -- this confirms,
not changes, L2's own prior conclusion, now with L1 included. Full stack DOES run
end-to-end correctly (L1 stubbed at the LLM boundary only, everything else real) --
validated as a correctness harness, not a training-ready pipeline. Parallelizing
across envs (L2's own recommendation) remains the open path to making training
practical; L1's real-Ollama gap is a second, independent blocker on top of it for
anything beyond a stubbed-L1 correctness check.

Files this round: src/agents/orchestrator_graph.py, tests/test_orchestrator_graph.py
(both committed 55e9b09). docs/reports/l1l2l3_integration_correctness.md (941aa0e).
Did NOT touch src/agents/l1_macro_analyst.py, src/data/l1_features.py (unchanged,
already committed prior rounds), or any L2/L3-owned file.

PRIOR ENTRY BELOW, for context on l1_features.py's own build and CLAUDE.md:
State: L1MacroAnalyst + orchestrator_graph.macro_tick() (committed bb47856, as before)
and the Ollama systemd fix (infra only, not a repo commit, as before) are unchanged this
round. NEW this round: src/data/l1_features.py (build_l1_feature_summary()) is built,
tested, and committed (8d905e5) -- the point-in-time rolling numeric summary that turns
data/raw_l1's collected klines_1m/funding_rate/open_interest into the feature_summary
dict L1MacroAnalyst.maybe_refresh() consumes. Built only after inventorying the real
data on disk directly (not assumed from the spec): klines_1m and funding_rate are
gap-free with zero duplicate rows (full sweep, not sampled); open_interest has zero
missing days but 263 consecutive days (2020-09-01..2021-05-21) with every row exactly
duplicated (confirmed byte-identical, not assumed) -- deduplicated unconditionally on
load. Order-book imbalance is deliberately NOT included -- data/raw_l1 has no book-depth
data at all; real OBI belongs to the separate Bybit L2 pipeline that feeds L2/L3, and
faking one from L1-only data would misrepresent a signal this module cannot see. Every
derived field is explicitly None (not NaN) when its window lacks sufficient real
coverage, so the eventual Ollama prompt never receives invalid JSON -- verified directly
(json.dumps round-trip test, plus a real-data sanity run against actual data/raw_l1
producing sensible, valid output end to end). 15 hand-computed tests, all passing.
Also this round: authored CLAUDE.md (committed ca28939) -- repo-wide standing rules
distilled from real incidents already logged across all three tracks' sections here
(classifier-denial handling, canonical-checkpoint safety, this file's own merge
discipline, live-verification-over-cached-citations, comment/code drift, judgment-call
escalation, spec-vs-real-code precedence). Not L1-scoped work specifically, but
authored from this track this round, recorded here for provenance.
No real Ollama call has been made anywhere yet, unchanged -- still explicitly paused.
Files owned/in-progress: src/agents/l1_macro_analyst.py, src/agents/orchestrator_graph.py
(bb47856); src/data/l1_features.py, tests/test_l1_features.py (8d905e5);
tests/test_l1_macro_analyst.py, tests/test_orchestrator_graph.py (bb47856); CLAUDE.md
(ca28939, repo root). Nothing currently uncommitted on this track.
Blocking/open questions: 14B vs 32B model choice still open, unchanged -- only
qwen2.5:14b-instruct-q4_K_M is pulled (8.4GB); the spec's default 32B variant is not
pulled and needs both a fresh download and confirmed GPU headroom. Real-model validation
stays gated on both that decision and the other tracks' live GPU work settling.
Next planned step: two independent options, neither gated on the model-size decision --
(a) wire orchestrator_graph.macro_tick() to call build_l1_feature_summary() instead of
accepting a caller-supplied feature_summary dict directly (closes the last plumbing gap
between real data and the live orchestration path); (b) once GPU headroom and the 14B/32B
decision are both confirmed, run the first real (mocked-no-longer) Ollama call through
L1MacroAnalyst.maybe_refresh() using build_l1_feature_summary()'s real output as input.
## L2 -- Strategist
Last updated: 2026-08-21 22:42 HKT
State: Consolidation round -- no new features, no training. DONE: FrozenL3Wrapper
(src/envs/wrappers.py), train_l2.py, tests/test_wrappers.py (20/20 passing) are all built,
committed locally, and smoke-tested (200-step mechanics-only run, no shape/interface
errors between the wrapper and SAC). Design doc (docs/reports/
phase4_l2_reconciliation_and_plan.md) reorganized this round -- a "CURRENT STATE" section
now sits at the top and reads standalone, with the full decision trail (including reasoning
later corrected or superseded) preserved below it rather than interleaved.
Checkpoint-citation correction (thanks to L3's own proactive note on this section, folded
in here): the smoke test's earlier reported sha256 973b2883... was genuinely live-computed
and accurate at the time (20:49 HKT that day) -- it went stale ~2 hours later when a
separate L3-track probe run's own final save overwrote models/l3_executioner_v1.zip
(confirmed via file mtime and TRACK_STATUS's own incident writeup). Re-verified directly
this round: the canonical path now holds sha256 27afa91e... (L3's numerically-verified
step-2,000,000 stand-in). Corrected everywhere this track owns it (design doc, this
section); nothing in wrappers.py/train_l2.py/test_wrappers.py needed correction since none
of them ever hardcoded a checksum -- all checked, all compute/reference by path, not by
hash. The smoke test's own validity is unaffected: it tested wiring, not policy quality.
Two items flagged last round as deliberately deferred are now decided (not implemented):
(a) VecNormalize on L2's OWN observation/reward space (distinct from the frozen L3 stats
already applied to L3's inputs) -- recommend adding before a real run, does not block
further work; L2's 41-dim obs mixes window-averaged features (lower variance) with
instantaneous ones (full variance) plus a genuinely non-stationary new scalar
(schedule_deviation), arguably a stronger case for adaptive normalization than L3's own
already-homogeneous vector. (b) A held-out eval callback analogous to L3's
ValISEvalCallback -- recommend building one, and unlike (a) this DOES block a real run: the
wrapper's mechanism is structurally expensive per SAC step (a gradient update on every
single L2 decision, each costing up to 50 real L3-predict+env.step calls, single env, no
parallelism), so a real 2,000,000-step run could plausibly take multiple days of wall-clock
time with zero visibility into whether it's working without one. Full reasoning for both in
the design doc's CURRENT STATE section.
Files owned/in-progress: none uncommitted. src/envs/wrappers.py, src/train/train_l2.py,
tests/test_wrappers.py all committed (1603c61, 87d7ba7). docs/reports/
phase4_l2_reconciliation_and_plan.md reorganized and committed this round.
Blocking/open questions: L3's matched A/B training runs, which will determine the final
frozen L3 checkpoint -- this is now the ONLY thing blocking a real L2 training launch. When
the A/B result lands, only --l3-checkpoint/--l3-vecnormalize need to change on L2's side;
everything else (observation space, action-space transform, hyperparameters, wrapper
mechanics) is independent of which specific checkpoint wins. Separately, not blocking: (a)
and (b) above are recommended before a REAL full-budget run specifically, not before the
A/B result lands -- could be built in parallel with waiting, if useful.
Next planned step: awaiting the L3 A/B result. Candidate parallel work if useful before
then: build the VecNormalize wrapping and/or the simplified held-out eval callback decided
above (neither implemented yet, both are "decided, not built"). No training launch planned
until both the checkpoint question resolves and, ideally, (b) exists.

## L3 / Env-Physics
Last updated: 2026-08-23 18:15 HKT
State: L3 RESEARCH PHASE CLOSED. Frozen and handed off for L1->L2->L3
integration -- full detail in docs/reports/l3_frozen_handoff.md (new,
standalone, written for a reader with no session context), committed 5d0d243.

Frozen checkpoint: Arm A's final (the TWAP-baseline A/B test's control,
1M steps warm-started from v1's step-2M stand-in). Permanent backup at
models/l3_frozen_backup/ (l3_executioner_v1_frozen.zip sha256 a5443e2a...,
l3_vecnormalize_frozen.pkl sha256 b459e177...), committed 3a4a283, checksum
-verified identical to source after copy. Pairing confirmed via identical
mtime (nanosecond-exact) and the training log's own final-save line naming
both paths as one save operation, not assumed from directory listing. Before
designating, checked whether a better candidate existed rather than assuming:
a 500k-step in-training snapshot that scored 0.686 IS at n=50 was evaluated
fresh at n=500 (a cheap eval, not a training run) and came in at 1.025 --
not significantly different from Arm A's own 0.994 (p=0.82/0.78) and not
better. Arm A's final remains the best point estimate among everything this
project evaluated at proper n.

Reason for closing: 8 in-training checkpoints across the last (budget
-extension) run show a plateau, not a trend -- best point at 500k steps in,
never bettered across the remaining 1.5M. Three consecutive reward
interventions since Arm A (r_queue direction inversion, TWAP-baseline
variance-reduction reward, the budget extension itself) all returned null
or significantly negative results. No further L3 reward iteration or
training is planned unless something in L1/L2 integration specifically
forces it.

Performance, stated plainly per the handoff doc: the frozen checkpoint TIES
TWAP (p=0.534/0.653), it does not beat it -- and nothing evaluated this
session does. The real win is the fill-ratio arc (0.2015 pre-retrain ->
0.919 for this checkpoint -> 0.9998 for the not-recommended budget
extension), attributable to earlier physics/matching-engine fixes, not any
of this session's reward-shaping work.

Integration verification (code-reading, not new code -- L2 owns the wrapper
side, so any mismatch would be reported as a blocker rather than fixed here):
observation space (42-dim, index-exact against L2's mapping table), action
space (MultiDiscrete([4,11,5])), and VecNormalize params (norm_obs=True,
clip_obs=5.0) all confirmed matching by reading src/envs/lob_execution_env.py,
src/envs/wrappers.py, and src/train/train_l3.py directly. L2's live-attribute
path (l2_target_slice_ratio_override/l2_urgency -> obs idx 15/16) confirmed
working. L1's path (l1_risk_score/l1_confidence -> obs idx 17/18, and
l1_risk_score into step_reward's inventory term) is correctly wired in L3's
own code but NOT currently driven by L2's wrapper -- not a bug (matches L1's
still-paused real-Ollama status) but a real gap: L1 integration into L2 needs
wiring, not just verification, before L1's signal reaches L3 in practice.
NO BLOCKERS found for loading this checkpoint into L2 as-is.

L3 is available for integration. src/envs/lob_execution_env.py still carries
an uncommitted, functionally-inert (eta_replace=0.0 makes it contribute
exactly 0 regardless) staleness-round addition from earlier this session --
left as-is, not committed as part of this freeze since it does not affect
any reported result's reproducibility.

PRIOR ENTRY BELOW, for context on the budget-extension result and the
r_queue-inversion correction that precedes it:
State: Arm A budget-extension run COMPLETE -- null result, committed 9320268.
Full detail in docs/reports/l3_armA_budget_extension.md. Continued Arm A's own
checkpoint (not v1, not canonical) for 2M more steps, same (inverted, see
correction below) reward config Arm A itself used, to test whether more budget
pushes past Arm A's TWAP-parity into a genuine edge. It does not: n=500 gives
IS_total_bps mean=1.237 (vs Arm A's 0.994, TWAP's 0.889), nominally significantly
worse than TWAP alone (p=0.034/0.044) -- but the decisive test, the direct paired
comparison against Arm A itself, is NOT significant (t p=0.092, Wilcoxon p=0.230,
Cohen's d_z=0.076), and 97% of the nominal difference comes from just 10/500
episodes (median diff ~0.0002bps, essentially zero for the typical episode). No
reliable evidence more budget helps or hurts. Training trajectory (8 in-training
n=50 firings across the 2M steps) shows no convergence -- volatile throughout,
best point came early (500k steps in) and was never bettered, worst point (1.25M
steps) briefly crossed worse than TWAP before partially recovering. fill_ratio did
climb further (0.919->0.9998, essentially eliminating unfilled orders) and
outcome variance dropped further (std 3.570->2.039) -- same pattern as Arm B's
variance reduction in the TWAP-baseline test, again not translating to a mean
edge. Explicitly does NOT test whether Arm A's own result was seed luck -- this
is a longer trajectory of the same seed, not an independent replication;
reproducibility with a different seed remains untested and is the natural next
step if this direction is pursued further (not run this round).

PRIOR ENTRY BELOW, unchanged -- the r_queue inversion correction still stands and
applies to this run too (see the reward-config note in the new report for how this
run's launch handled it):
State: CORRECTION to the entry below, discovered while prepping a follow-up
run (reward.py inversion now committed 4d81a96, see that commit message for
the full reconstruction): both arms of the TWAP-baseline A/B test actually
trained with the r_queue queue-weighted term INVERTED (EXPERIMENTAL 4), not
in the original direction as implied below and in the report. This was
inline, unconditional, uncommitted code left over from the separate
direction-inversion probe (~22:00 HKT 2026-08-20) that was never reverted --
every run since silently inherited it, including both A/B arms (confirmed
via file mtimes: reward.py unchanged from 20:38 HKT 2026-08-21 through both
launches). v1's own original training DID use the original direction --
only the CONTINUED training (both arms, warm-started from v1) picked up the
inversion. The ARM B vs ARM A comparison itself is unaffected (both arms
shared it equally, so it's still a clean isolated test of
subtract_twap_baseline). What IS affected: "Arm A ties with TWAP, up from
v1's 1.261" is not "same reward + more training" as stated -- it's "more
training AND r_queue flipped," undisentangled. Full correction in
docs/reports/l3_twap_baseline_reward.md's "A/B test result" section.

PRIOR ENTRY BELOW, otherwise unchanged:
State: TWAP-baseline reward A/B test COMPLETE -- clean negative result,
committed 83508dc. Full detail in docs/reports/l3_twap_baseline_reward.md's
"A/B test result" section. Both arms trained 1M steps, warm-started from the
same canonical checkpoint (27afa91e..., re-verified before each run, and
confirmed untouched by either run afterward). n=500 eval (same paired seeds
5,000,000-5,000,499 as every prior n=500 eval this session; TWAP numbers
reused byte-for-byte, not recomputed):

Pooled ordering: TWAP (0.889) < Arm A/control (0.994) < best-B (1.103) < v1
(1.261) < Arm B/treatment (1.341). Arm A (plain continued training, no
reward change) ties with TWAP (p=0.534/0.653) and is now the
best-performing RL variant measured this session. Arm B (the TWAP-baseline
reward under test) is significantly WORSE than TWAP (p=0.0092/0.0140) --
and critically, the direct Arm-B-vs-Arm-A paired test (the one that actually
isolates the reward change from "more training helped") is also significant
in the same worse-for-B direction: mean diff +0.347bps, paired t p=0.0097,
Wilcoxon p=0.0224, Cohen's d_z=0.116 (small but real effect, ~8% of TWAP's
own std -- comparable in magnitude to the v1-vs-TWAP effect size found
earlier). Diagnostic: the mean difference is tail-driven (median diff ~0.02
bps, but the worst 10/500 episodes account for 69% of the net gap) -- most
episodes are near parity, a minority go badly for Arm B. The mechanism DID
reduce outcome variance as designed (Arm B std=2.405 vs Arm A std=3.570 vs
TWAP std=4.353, Levene p<0.0001) -- that reduction just didn't translate
into better, or even equal, mean execution quality. Arm B also converged to
a behaviorally different policy (fill_ratio 0.919->0.990, mean ep_len
1572->811 ticks), not simply a faster/cleaner version of Arm A's policy,
which the baseline-subtraction design's own premise (subtracting a
per-episode constant cannot change the optimal policy) did not anticipate.
Caveat carried into the report: single seed per arm, so this cannot fully
rule out single-run training variance as part of the explanation, though the
two-test agreement plus the coherent behavioral shift argue against pure
noise.

Not spun as a partial win around the variance reduction -- reported as what
it is, a negative result for the reward-change hypothesis as implemented and
tested here. No iteration on the reward formulation attempted this pass; no
multi-seed replication run (flagged as the natural next step if this
direction is worth another look, not decided here).

Old narrative below, from before this A/B test started, retained for context
on Task 1 (v1 re-evaluated at n=500) and Task 2's implementation (the reward
mechanism this A/B test above evaluates):

State: Two more tasks landed since the last check-in, both no-GPU/no-training. Pointer
update again -- see docs/reports/l3_replace_value_probe.md (Task 1) and
docs/reports/l3_twap_baseline_reward.md (Task 2, new file) for full detail.

TASK 1 (v1 re-evaluated at n=500, committed 332d1b9): checkpoint verified before running
anything -- models/l3_executioner_v1.zip is still the step-2,000,000 stand-in
(27afa91e...), NOT true v1 (973b2883..., unrecoverable), consistent with every check this
session. Result: v1's own headline numbers barely moved from n=50 (IS 1.245->1.2607, fill
0.918->0.892), but the gap vs TWAP grew ~6x (0.063bps->0.372bps) and now clears p<0.05 by
the paired t-test (p=0.0327) though NOT by Wilcoxon (p=0.115) -- stated plainly rather than
picking whichever test agrees. Three-way n=500 point-estimate ordering: TWAP (0.889) <
best-B (1.103) < v1 (1.261) -- v1 worst of the three, a more robust version of the
milestone report's own "near-parity, if anything slightly worse" framing, not a reversal
like best-B's sign flip was. Read as meaningfully suggestive, not fully unanimous, evidence
that the RL policy underperforms simple baselines here.

TASK 2 (TWAP-baseline variance-reduction reward, implemented + tested, NOT trained,
committed 1b442f9 -- separately from Task 1 and from the still-unconfirmed r_queue
direction-inversion, see Files owned/in-progress below): terminal reward becomes
-kappa*(IS_total_bps - twap_shadow_IS_total_bps), gated behind
RewardWeights.subtract_twap_baseline (default False, same opt-in convention as
zeta/eta_replace). Explicitly a baseline subtraction, not an objective change -- the
subtracted quantity never observes the real episode's policy, so it shifts reward scale,
not the argmax. Design decision (asked for, not picked silently): computed fresh in
reset(), NOT cached -- caching would not help training resets regardless of choice, since
training draws from ~349M possible windows with ~0% real repeat rate; caching only helps
periodic eval (same fixed seeds reused across firings), a separate, not-yet-built
optimization. Measured cost on REAL market data, not guessed: +48ms/reset (2.4% of
reset()'s own ~2027ms baseline, dominated by day-load I/O), ~0.1% of a full episode's
wall-clock budget at v1's measured throughput -- much cheaper than an initial
back-of-envelope worry (~2x) suggested; measuring instead of assuming caught that worry
being wrong. 4 new tests (tests/test_twap_baseline_reward.py) verify the subtraction
arithmetic, the flag's inertness, and that info["implementation_shortfall"] never changes
-- the key integration test (a policy matching TWAP exactly must get ~0 terminal
contribution) caught two real implementation bugs before either reached committed code
(missing discrete-SIZE_FRACTIONS rounding, and a decision/evolution ordering mismatch) --
concrete evidence the test design did its job, not just satisfied a checklist.

Old narrative retained below unchanged for context on the checkpoint-overwrite incident,
the direction-inversion probe, and the prior REPLACE-value follow-up:

State: Two follow-up tasks on top of the direction-inversion probe below (still the most
recent full narrative; this entry is a pointer + headline update, not a restatement --
see docs/reports/l3_replace_value_probe.md for the complete account of both).

TASK 1 (safety fix, committed c803249, separate from Task 2): train_l3.py's final-save
landmine flagged in the entry below is FIXED -- the final save now refuses to overwrite
an existing models/l3_executioner_v1.zip/l3_vecnormalize.pkl unless --overwrite-canonical
is passed explicitly, redirecting to a run-tagged path otherwise. IMPORTANT scope note
requested explicitly, recorded here per instruction: while building this fix, the SAME
hardcoded-path bug was found to ALSO affect CheckpointCallback's periodic saves
(name_prefix="l3_ppo" was shared across every run) -- and it had ALREADY silently
overwritten data before this fix landed, independently of the final-save incident already
documented below. Specifically: v1's own periodic checkpoints at steps 250,000 and
500,000 (from its 2,000,000-step run) were silently overwritten by the direction-inversion
probe's own periodic checkpoints at those same step numbers (confirmed via file
timestamps -- the surviving 250k/500k files are dated to the probe's run window, not
v1's). Consequence: v1's intermediate training trajectory below step 750,000 is no
longer recoverable from checkpoints (750k onward still exist and were unaffected). This
did not affect anything already reported -- v1's FINAL numbers came from the step-2M
checkpoint (recovered separately, see below) and the milestone report's own eval, neither
of which depended on the 250k/500k files -- but the early trajectory is gone, for the
record. Both this and the final-save fix are now namespaced by --run-name (auto-generated
timestamp if not given), so this cannot recur for either save path going forward. 4 new
tests in tests/test_train_l3.py cover the guard logic directly (tmp_path-based, no
GPU/training needed).

TASK 2 (adequate-power re-test of the direction-inversion probe's own follow-up question,
i.e. "is CANCEL_AND_REPLACE actually valuable" -- heuristic simulation only, no training,
no GPU): the original scripted-heuristic probe (docs/reports/l3_replace_value_probe.md)
concluded REPLACE shows no value at n=50, but n=50 had only 14.7% power to detect the
effect it actually observed (best-B nominally beat TWAP by 0.48bps). Re-ran best-B vs TWAP
ONLY (pre-registered, no re-sweep) at n=500 (~83% power for that effect size). RESULT: the
sign flipped -- best-B is now numerically WORSE than TWAP (+0.214bps, not -0.482bps),
still not significant (p=0.101), and the practical effect size is small either way
(~4% of TWAP's own std). This is a stronger, not weaker, confirmation of "REPLACE isn't
valuable here" -- the n=50 number that made REPLACE look promising was a selection
artifact from screening 18 configs, and did not survive proper power. Separately, and
stated plainly per instruction: at their respective sample sizes, BOTH TWAP (0.889 at
n=500) and best-B (1.103 at n=500) come in numerically better than v1's trained RL
policy's own reported IS_total_bps (1.245 at n=50) -- not a formal paired test against
v1 (no n=500 v1 data exists to pair against), but a real, plainly-stated finding about
the RL setup itself, not buried as a footnote. Also confirmed independently before running
anything: no PASSIVE-family config can reach comparable fill to TWAP/B (40.4% at offset=0
is a structural ceiling, not a sweep gap -- crossing offsets all collapse to the same
single-tick-depth-limited outcome regardless of how aggressive), so no PASSIVE arm was
re-tested at n=500; substituting it would have repeated a known apples-to-oranges
comparison at higher n rather than fixing anything.

Both tasks committed locally (Task 1: c803249; Task 2: pending commit alongside this
update), not pushed.

PRIOR ENTRY BELOW, unchanged, for full context on the checkpoint-overwrite incident and
the direction-inversion probe itself:

State: Ran a bounded probe testing whether inverting the r_queue REPLACE/MARKET
queue-cost direction increases CANCEL_AND_REPLACE usage (near-0% in v1 -- see prior
entries). Result: probe does NOT confirm the hypothesis. Also: an operational mistake
during this probe overwrote the v1 checkpoint in the working slot -- recovered, but
disclosing in full below since it affects what L2 (and anyone else) should trust right
now.

INCIDENT, disclosed in full: train_l3.py's final save always writes to the same
hardcoded path (models/l3_executioner_v1.zip / l3_vecnormalize.pkl) regardless of
whether the run is a full commitment or a bounded probe. Every prior probe this session
warm-started FROM a checkpoint without ever letting the run reach its own final save
(always stopped/killed first) -- this is the first one that was deliberately allowed to
run to completion, and its completion silently overwrote v1 (sha256 973b2883...) with
the probe's own output. This should have been anticipated and backed up beforehand,
the same way models/baseline_20M_backup/ protects the original 20M-step baseline -- it
was not, which is a real process gap, not a tooling failure. Caught immediately once
the probe's completion log showed a new checksum instead of 973b2883... (which is what
prompted this disclosure, not a routine status update).

RECOVERY: the true 973b2883... file is not byte-recoverable -- it was never backed up
and the working slot is the only place it lived. Best available fallback: v1's own
CheckpointCallback had already saved a periodic checkpoint at step 2,000,000 (10:07
HKT, 4 minutes before v1's actual final save at step 2,002,944) to
models/l3_checkpoints/l3_ppo_2000000_steps.zip -- essentially the same policy, short by
2,944 steps of further training. Restored this into the working slot
(models/l3_executioner_v1.zip, now sha256 27afa91e...). Verified, not assumed: ran the
exact same reproduction/eval script used for the milestone report against it and got
IS_total_bps=1.2450, fill_ratio=0.9180 -- bit-for-bit identical to the numbers
973b2883... itself printed live during training at its own step=2,000,000 eval firing
(this also resolves a discrepancy flagged in the milestone report, where a same-session
reproduction against the truly-final 973b2883... checkpoint gave slightly different
numbers, IS=1.265/fill=0.976 -- now explained: that gap was the extra 2,944 steps of
training between this checkpoint and the true final save, not a bug or nondeterminism).
The probe's own output was preserved under its own name before being overwritten
further (models/l3_executioner_v1_replace_direction_probe.zip /
l3_vecnormalize_replace_direction_probe.pkl) -- no data was lost, only the exact
973b2883... bytes are gone. A permanent, git-tracked backup of the restored near-v1
checkpoint now exists at models/v1_near_backup_step2M/ (committed as f848bba) so this
specific failure mode cannot recur for THIS checkpoint -- the same discipline should be
applied automatically before any future run that might reach its own final save,
without needing to be told again.

THE PROBE ITSELF, Steps 1-2 (src/envs/reward.py, tests/test_reward.py, both still
UNCOMMITTED pending the result below): inverted the queue-weighted term in both
canceled_via_market and canceled_via_replace branches of step_reward()'s r_queue block,
from gamma*(queue_ahead/queue_at_level) to gamma*(1 - queue_ahead/queue_at_level) --
charges more for discarding genuinely earned queue priority (q_ahead near 0 via real
trade-through volume), less for a fresh order or one behind a deep/stalled level.
Module docstring and RewardWeights.eta_replace's loophole derivation updated in place,
old reasoning kept verbatim and marked SUPERSEDED rather than deleted. Loophole
re-derivation (eta_replace stays 0.0/inert for this probe either way): the specific,
previously-flagged off-book exploit (price outside the visible top-20 book giving
EXACTLY zero replace-cost) is CLOSED -- inverted, in fact, to the maximum charge
-gamma, not the minimum. A weaker residual remains and is stated plainly, not glossed
over: a fresh, minimal-size order (size_frac has a 20% floor, never exactly 0) behind a
level with substantial real ambient depth can still approach near-zero cost -- but that
depends on actual market depth at the chosen tick, is not a free/guaranteed
policy-controlled knob the way off-book was, and only approaches zero asymptotically,
never hits it exactly. 17 reward tests pass (3 updated to the new numbers, 2 new tests
added making the near-front/close-to-gamma and fresh-deep-level/close-to-zero cases
explicit, 1 renamed to reflect the inverted off-book case -- see tests/test_reward.py).
Full env/matching-engine suite (30 tests) also re-verified passing; 4 unrelated,
pre-existing failures in test_bulk_backfill.py/test_l2_capture.py confirmed untouched by
this change (no reward.py import, network/order-book-resync issues unrelated to reward
shaping).

Step 3 (bounded probe): warm-started from the checksum-verified v1 (973b2883..., verified
immediately before launch -- this was correct; the overwrite happened at the END of the
run, not the start), same real reward config as v1 (RewardWeights() defaults, no CLI
overrides), n_envs=8, --total-timesteps 500000 (same order of magnitude as prior probes
this session). Logged to logs/l3_train_replace_direction_probe.log (new, distinct path).
Ran slower than v1's own run (~200-220 fps vs. v1's 350-365 -- not investigated further,
did not affect the probe's validity, just took longer wall-clock, ~42 min instead of the
~25-30 min originally estimated). Completed cleanly, no crash, no OOM (memory returned to
baseline afterward).

Step 4 (eval, reused the existing script unmodified in logic, only parameterized with a
--model-path/--vecnorm-path CLI flag to point at either checkpoint): ran the identical
50-paired-seed reproduction against BOTH the restored near-v1 checkpoint (clean baseline,
numbers above) and the probe's own checkpoint, for a same-methodology, same-script,
apples-to-apples comparison (deliberately not relying on the training log's own printed
numbers for either arm, given the discrepancy explained above):

  metric                near-v1 (baseline)   probe (500k more, inverted formula)
  CANCEL_AND_REPLACE %  0.298% (263/88141)    0.336% (312/92810)
  MARKET %              0.01%                 0.06%
  HOLD %                47.64%                46.70%
  LIMIT %               52.05%                52.91%
  IS_total_bps           1.245                 1.829
  fill_ratio              0.918                 0.942
  vs TWAP (paired t)     t=0.13 p=0.90         t=1.12 p=0.27
  vs TWAP (Wilcoxon)     W=572 p=0.53          W=487 p=0.15

Step 5, reported honestly per instruction -- NOT a confirmation: CANCEL_AND_REPLACE usage
moved from 0.298% to 0.336%, a change too small to trust. A two-proportion z-test on the
raw counts (263/88141 vs 312/92810) gives z=1.43, p=0.15 -- not significant even under a
naive test that ignores within-episode/within-policy autocorrelation, which would only
inflate the true variance further and make this LESS significant, not more. This bounded
probe does not show the direction-inversion unlocking meaningful REPLACE usage. IS_total_bps
also got numerically worse (1.245->1.829) while fill_ratio improved slightly (0.918->0.942)
-- neither reaches significance vs TWAP either, so this is not read as a real regression,
but it is also not a case for "this direction is obviously fine, just needs more budget."
Two honest, undecided readings, not a recommendation: (a) queue-cost direction was not
the actual binding constraint on REPLACE usage -- something else (a strongly converged
HOLD/LIMIT habit from the full 2M-step run, or genuinely low value of ever using REPLACE
given how well passive LIMIT already performs) may dominate regardless of price signal;
(b) 500k steps of continued fine-tuning from an already-deeply-converged policy may
simply be too short a window to reveal a shift even if the corrected incentive matters at
longer horizons -- v1's own split needed the full 2M-step run to show its effect, not a
short probe, so this would not be an unprecedented pattern.

Files owned/in-progress: src/envs/lob_execution_env.py (uncommitted, unchanged in
substance -- staleness/eta_replace round only, not touched by this probe per the hard
boundary), src/envs/reward.py + tests/test_reward.py (uncommitted, the direction
inversion above -- NOT committed pending a decision on the result), src/train/train_l3.py
(uncommitted, same TypeError reason as before). models/l3_executioner_v1.zip is
currently the restored near-v1 checkpoint (27afa91e...) -- NOT the probe's checkpoint,
which was rejected/not promoted given the result above and lives separately at
models/l3_executioner_v1_replace_direction_probe.zip.
Blocking/open questions: (g) NEW: does the checkpoint-overwrite incident above change
anything for L2's item (2) -- the answer should be no, since the restored checkpoint is
numerically verified equivalent to what was already being evaluated, but flagging since
it is a new fact, not assuming it is obviously irrelevant. (h) NEW: given the probe's
non-result, is further budget on this direction (longer probe, different formula
entirely, or dropping the REPLACE-usage question as not worth chasing further right
now) worth committing, or should focus shift elsewhere -- explicitly not this track's
call to make unilaterally, surfacing for direction. (f) from the prior entry (is
near-parity-with-TWAP good enough for L2 to build on) remains open and is unaffected by
this probe either way.
Next planned step: awaiting direction on (h) [this probe's own follow-up] and (f) [the
still-open handoff question from before]. reward.py/test_reward.py stay uncommitted
until one of those is resolved -- recommend not committing the direction inversion until
it is either confirmed by more data or deliberately abandoned, rather than landing an
unconfirmed change; that is a recommendation, not a decision made here.
