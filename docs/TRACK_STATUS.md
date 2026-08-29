# Cross-track status

Shared status file for the concurrent L1/L2/L3 work sessions on this repo.
Each session owns and updates only its own section -- merge on conflict,
never overwrite another track's section.

## L1 -- Macro Analyst
Last updated: 2026-08-25 09:15 HKT
NOTE: a duplicate session picked up the "fix Ollama + validate real L1 end to
end" prompt (sent to two sessions by mistake, per the user -- not a process
failure, nothing lost) and independently reconstructed most of this same
round's work (Ollama fix diagnosis, real-call testing incl. a clean 25/25
schema-conformant batch, VRAM/throughput measurements, a real-L1 integration
re-run) before checking this file and finding it already done, more
thoroughly, below. That session's batch ran AFTER the prompt fix (538d6ae)
had already landed, which is exactly why it never saw the 0/10 conformance
failure that is the actual finding -- caught only because it checked
TRACK_STATUS.md before writing anything up, rather than reporting a fresh
25/25 as if it were new information. Stood down without committing,
spot-checking, or re-reporting, per instruction. Recorded here so any
overlapping timestamps or artifact naming found later make sense. The
account below is unaffected and remains the authoritative one.

## L1 -- Macro Analyst
Last updated: 2026-08-24 08:29 HKT
State: ALL THREE ROUND TASKS DONE -- prompt fixed and re-validated, background
threading built and unit-tested, live threaded re-run confirms it works. Full
detail in docs/reports/l1_async_threading_validation.md (Tasks 2/3) and the
Task 1 commit message / docs/reports/l1_real_llm_validation.md (Task 1, prior
round's own follow-up finding this round closed out).

Task 1 (538d6ae): SYSTEM_PROMPT now states every field/range explicitly, and
maybe_refresh() now sends Ollama's structured-output format=<real
MacroRiskContext.model_json_schema()> instead of the bare string "json" --
confirmed live to fix the actual failure mode (invented field names), and
confirmed live to NOT enforce numeric ranges (an out-of-range value slipped
through even under schema-constrained decoding), so the prompt's explicit
ranges and pydantic's own validation both remain genuinely necessary, not
redundant. Re-validated on 10 real calls (5 dates x 2 reps, for a
determinism check too): 10/10 schema-conformant, all values in range, mean
latency dropped 4.505s -> 1.598s. 2 new regression tests.

Task 2 (586b881): AsyncL1Refresher built per architecture_spec.md Section
1.2's own design (background thread, hot path never waits). Staleness
policy is explicit -- skip a new refresh if one is already in flight, never
queue, bounded by L1MacroAnalyst's own timeout_s. Fail-closed survives
threading by construction (cache is set to exactly what maybe_refresh()
itself returns, no second freshness flag to desync). run_episode_async()
always joins in a finally block -- no thread survives past return, even on
an exception. 10 new tests, including a direct non-blocking timing test (20
tick calls total <0.1s while a mocked 0.3s call is in flight) and a
full-stack async integration test with real checkpoints. Full suite: 141
passed, same 4 pre-existing unrelated failures as always.

Task 3 (6d9290c, live validation script, not committed as a pytest test --
needs a live Ollama service, would break every other test's hermeticity in
that file): idx 17/18 confirmed to change ASYNCHRONOUSLY -- (0.0,0.0) ->
(0.0,0.75) at tick 250, 250 ticks after the tick-0 trigger, once the real
threaded call actually completed, not at the boundary tick itself (the
correct framing now that L1 is non-blocking, vs. the old "changes exactly
at tick 600" framing from the synchronous round). Per-tick cost: 3.81ms
(stubbed) -> ~5.5-6.3ms (threaded, this round, 3 runs) -> 14.60ms
(synchronous, prior round) -- threading recovers roughly 60-70% of the gap,
not 100%; real residual lock/thread-scheduling overhead remains and is
reported as a range, not oversold as full parity. One honest, unreproduced
anomaly disclosed in the report: the first live run showed
last_refresh_completed_tick set correctly but its paired on_l1_tick
callback never fired -- not reproduced across 2 immediate reruns or the 6
passing deterministic regression tests covering the same logic; flagged
open, not silently dropped.

Files this round: src/agents/l1_macro_analyst.py, tests/test_l1_macro_analyst.py
(Task 1, 538d6ae). src/agents/orchestrator_graph.py,
tests/test_orchestrator_graph.py (Task 2, 586b881).
docs/reports/l1_async_threading_validation.md (Task 3, 6d9290c). Nothing
uncommitted on this track.

Blocking/open questions: (a) RESOLVED this round -- SYSTEM_PROMPT schema gap
fixed, 10/10 conformant now. (b) RESOLVED this round -- background-thread
wiring built, tested, live-validated. (c) NEW, minor -- the unreproduced
on_l1_tick anomaly above; worth a second look if it recurs, not blocking.
(d) NEW -- residual ~1.5-2x per-tick overhead vs. stubbed baseline (lock/
scheduling cost); real but likely not worth chasing given the much larger
win already banked. 14B vs 32B: still recommend 14B (prior round's Step 3
reasoning), now on stronger footing since 14B's conformance problem is
resolved -- no remaining argument for 32B was found this round either.

Next planned step: no hard blocker remains on this track's own critical
path. Candidate next work, none urgent: (i) wire build_l1_feature_summary()
to use the env's actual current sim timestamp per L1 firing (this round's
validation scripts reused one fixed as_of_ms across all firings, a
simplification carried over from the prior round, not a production
behavior) so risk_score/confidence genuinely track evolving real market
conditions across an episode, not just a single frozen snapshot repeated.
(ii) investigate the residual threading overhead in (d) if it turns out to
matter for a real training-scale run. (iii) coordinate with L2/L3 on
whether/when a real L1-in-the-loop training run is wanted at all, now that
the plumbing is real end to end.

PRIOR ENTRY BELOW, for context on the Ollama/proxy infra fixes and Step 0-4
real-call validation:

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
Last updated: 2026-08-29 15:10 HKT
State: L2V3 n=500 EVALUATION COMPLETE. Full detail:
docs/reports/l2v3_checkpoint_evaluation_report.md. Test split untouched.

Result: a stable critic does NOT translate into a better policy here. l2v3
final (step 1,599,936, critic_loss=0.059, never diverged) scored
IS_bps=1.169 vs TWAP-passthrough (1.024) -- nominally worse, not
significant (t_p=0.280, w_p=0.118), fails the pre-registered bar outright
(wrong direction, not just non-significant). Direct paired comparison
against l2v2's pre-divergence checkpoint (same reward, same 500 seeds,
gamma the only difference: 0.983 vs 0.995): no detectable difference
(mean diff=-0.053bps, d_z=-0.024, t_p=0.592, w_p=0.912) -- l2v2
pre-divergence remains the best performer found so far (1.117), l2v3 final
statistically indistinguishable from it.

Answers this round's central question: a numerically well-conditioned
critic (via gamma=0.983) and a critic that has to be caught pre-divergence
(gamma=0.995) land at statistically the same held-out execution quality.
gamma=0.983's practical value is removing the checkpoint-selection problem
(every checkpoint usable, not just a pre-divergence one), not closing the
gap to TWAP-passthrough.

Full diagnostic battery (train-vs-val relative, volatility strata) NOT run
this round -- conditional on clearing the pre-registered bar, which this
checkpoint did not.

Thread-capping fix (prior round, commit 4d5a544) confirmed working in
production: 99.8% CPU during this run (vs 1,353% pre-fix), full run
completed in 40.97min, in line with (marginally faster than) the prior
round's ~46min/checkpoint runs that used the external-env-var workaround.

Action distribution: actively steering (within-episode std
participation_mult=0.525, urgency=0.201, both non-trivial), not collapsed
to a constant action. urgency mean (0.260) well below neutral (0.5) --
systematically less urgent than TWAP-passthrough's default, without a
measurable execution-quality payoff.

New script this round: scripts/compare_l2v3_vs_l2v2predivergence.py --
direct paired checkpoint-vs-checkpoint comparison from two already-computed
episodes CSVs (no new episodes run), reusable for any future two-checkpoint
same-seed comparison.

Next planned step: none launched -- reporting and awaiting direction, per
this round's own scope.

PRIOR ENTRY BELOW, for the gamma-ablation training run's own context:

## L2 -- Strategist
Last updated: 2026-08-29 13:50 HKT
State: GAMMA-ABLATION TRAINING RUN COMPLETE. Full detail:
docs/reports/l2v3_gamma_ablation_training_run_report.md. Test split untouched.
n=500 evaluation NOT yet run this round -- explicitly deferred per instruction.

Headline result: critic_loss stayed flat (0.048-0.074 band) across the entire
1,600,000-step budget under gamma=0.983 -- no divergence at any point. Same
reward (potential_is_shaping) under gamma=0.995 (l2v2) tracked l2v3 closely
through ~1.3M steps then diverged sharply starting almost exactly at the 1.6M
mark this run stopped at (critic_loss 47x higher than l2v3's at that same
step). l2v1 (old reward, gamma=0.995) diverged far earlier and more steadily
throughout. Consistent with the gamma-as-lever hypothesis, via a corrected
mechanism (TD-bootstrap-error compounding under near-flat discounting, not
the truncation-bootstrapping claim originally proposed -- SB3's ReplayBuffer
already handles truncation correctly, confirmed by reading its source before
this run launched). Not a seeded ablation (one run per gamma value) -- flagged
explicitly as a real but not fully closed finding.

Two other tasks completed this round, both prerequisite to the run:
(1) Thread-capping bug fixed in the three eval scripts (eval_l2_n500.py,
eval_l2_diagnostics.py, eval_l2_bucketed.py -- commit 4d5a544), matching the
pattern train_l2.py already had. Previously only worked around at launch time
via external env vars.
(2) --gamma exposed as a CLI flag on train_l2.py (commit 2a3eee3, default
unchanged at 0.995), replacing the previously hardcoded L2_GAMMA module
constant, so gamma is now actually ablatable and every run's own printed
resolved config records which value it used.

Mechanical health: 32/32 checkpoints landed on schedule with complete
model+replay_buffer+vecnormalize triplets, 0 errors in the full log, ~19
dec/s throughout (matching l2v1/l2v2), final save correctly diverted to
run-tagged paths (canonical checkpoint and frozen L3 backup both confirmed
untouched, byte/hash-identical pre- and post-run).

Next planned step: none launched -- reporting and awaiting direction, per
this round's own scope (n=500 evaluation is a separate round, not started).

PRIOR ENTRY BELOW, for the three-checkpoint evaluation round's own context:

## L2 -- Strategist
Last updated: 2026-08-29 10:15 HKT
State: THREE-CHECKPOINT n=500 EVALUATION + DIAGNOSTIC BATTERY COMPLETE.
Full detail: docs/reports/l2v2_checkpoint_evaluation_report.md (commit 7037805).
Test split untouched.

Evaluated l2v2 final (step 1,999,992), l2v2 pre-divergence (step 1,599,936),
and l2v1 mid-run (step 499,980, critic_loss~26) at n=500 against
TWAP-passthrough (1.024) and pure TWAP (0.889). None beat the pre-registered
bar (both t-test AND Wilcoxon significant). Results: l2v2 final=1.227
(Wilcoxon-only significant, p=0.0045, fails "both agree"), l2v2
pre-divergence=1.117 (best, neither test significant -- statistically
indistinguishable from baseline), l2v1 mid-run=1.177 (neither significant).

Direct paired checkpoint-vs-checkpoint comparisons (same seeds) all point
toward less-diverged checkpoints performing better (l2v2 final worse than
its own pre-divergence checkpoint by 0.110bps; l2v1 mid-run better than
l2v2 final by 0.050bps) -- consistent with the critic-divergence finding
in the prior entry below, but none individually reaches significance at
n=500 (p>=0.32 throughout).

Full diagnostic battery run on the best performer (l2v2 pre-divergence):
cross-split relative comparison reproduces the val-worse/train-better
pattern found on the original checkpoint but weaker (swing=0.434bps vs the
original ~0.46bps; only Welch's t significant here, p=0.025, not
Mann-Whitney, p=0.157 -- the original had both agree). Volatility-stratified
eval (train days, memorization-confounded) shows no strengthening with
volatility (all |d_z|<0.011), matching the original checkpoint's own
d_z=-0.011 high-volatility result closely.

Early-stopping question (explicitly asked): evidence is directionally
consistent across all three pairwise checkpoint comparisons (earlier/
less-diverged always better) but not statistically established at n=500.
Reading: stop once critic_loss starts diverging (no evidence continuing
past that point helps), but early stopping does not turn a losing
checkpoint into a winning one -- no checkpoint tested beats TWAP-passthrough.

Real bug found and fixed mid-round: scripts/eval_l2_n500.py,
eval_l2_diagnostics.py, eval_l2_bucketed.py have no thread-capping (unlike
train_l2.py, fixed for the same issue during vectorization) -- first
checkpoint attempt ran 33min at 1353% CPU with zero output, killed and
relaunched with OMP_NUM_THREADS=1/MKL_NUM_THREADS=1/etc, dropped to ~100%
CPU, completed normally (~46min/checkpoint thereafter). Not yet fixed in
the scripts themselves -- workaround applied at launch time only, a
follow-up fix is recommended but wasn't asked for this round.

New script this round: scripts/analyze_l2_relative_comparison.py (commit
1d4fdf6) -- reusable cross-split (Welch's t + Mann-Whitney on the
L2-minus-baseline difference distributions) comparison, reimplementing the
project's earlier ad-hoc Diagnostic-2-correction methodology.

Next planned step: none launched -- reporting and awaiting direction, per
this round's own scope (no instruction to proceed further was given).

PRIOR ENTRY BELOW, for the completed training run's own context:

Last updated: 2026-08-28 12:15 HKT
State: REAL 2,000,000-STEP RUN UNDER NEW REWARD COMPLETE
(run_name=l2v2_potentialis_20260827, --l2-reward-mode potential_is_shaping).
Ran 105,219s (~29.23h), total_timesteps=1,999,992 -- essentially identical
wall-clock to l2v1_20260825's 29.34h, confirming the new reward does not
change per-step cost materially. Mechanically clean start to finish: all 40
checkpoints landed on the expected ~44-min/49,998-step cadence with no gaps,
0 tracebacks/errors/NaN matches anywhere in the full log, canonical
checkpoint guard worked correctly (models/l2_strategist_v1.zip untouched --
final save went to models/l2_strategist_v1_l2v2_potentialis_20260827.zip /
l2_vecnormalize_l2v2_potentialis_20260827.pkl as expected without
--overwrite-canonical). Not evaluated yet -- n=500 eval and the full
diagnostic battery are a separate round; test split untouched throughout.

IMPORTANT CAVEAT ON THE FINAL CHECKPOINT -- late-run critic divergence,
found live during monitoring, NOT a mechanical crash: critic_loss was
normal (~1-2) through step ~1,600,000 (80% through training), then began
climbing roughly exponentially for the remaining ~400,000 steps, reaching
10,800 at the final logged point (step 1,999,992). actor_loss moved in
lockstep (4 -> 376). ent_coef, which had fallen to near-zero by ~1.2M steps
(the positive divergence from l2v1's own climbing ent_coef reported
earlier -- see below), reversed and climbed back to 0.163 by the end,
consistent with the critic no longer providing coherent value estimates for
the entropy auto-tuning to react to. No NaN ever appeared (0 matches, full
log checked), so this stopped short of an outright numerical crash, but the
FINAL checkpoint's policy should be treated as potentially degraded by this
-- do not assume later-in-training means better here. A clean checkpoint
from immediately before the divergence onset exists: models/l2_checkpoints/
l2_sac_l2v2_potentialis_20260827_1599936_steps.zip (critic_loss ~1-2 at
save time). Recommend evaluating BOTH the final save and this pre-
divergence checkpoint in the eventual n=500 round rather than assuming the
final one is the right candidate.

Root cause not diagnosed this round (out of scope -- reported live,
training was not intervened on since so little of the run remained once
detected). Plausible contributors, not established: Q-value overestimation
compounding late in training (a known SAC/DDPG-family failure mode) possibly
interacting with the new reward's own dynamics over long horizons in a way
the ~57min/65,000-step shakedown (Task 3 of the implementation round, see
PRIOR ENTRY BELOW) could not have caught -- that shakedown validated
mechanics and short-run reward scale, not million-step-horizon training
stability, and the divergence onset (~1.6M steps) is well past anything a
1-hour shakedown reaches.

ent_coef trend vs l2v1 at matching steps, tracked live during this round
(diagnostic requested going into this run, not itself a performance claim):
  step ~76k:    new 0.00279  vs l2v1 0.0057   (~2x lower)
  step ~191k:   new 0.00118  vs l2v1 0.0082   (~7x lower)
  step ~1.23M:  new 0.000579 vs l2v1 0.0334   (~58x lower, still falling)
  step ~1.9M+:  new reversed upward to 0.163, coinciding with the critic
                divergence above -- the earlier falling trend did NOT hold
                through the full run, so it should not be read as a clean
                "more learnable signal" signal on its own; it's confounded
                by the late instability.

Held-out eval (ValISEvalCallback, paired seeds 5000000..5000009 vs
TWAP-passthrough baseline 0.9976 IS_total_bps, fired every 10,000 steps,
~200 firings total): NOT a clean improvement story across the run. Mean
IS_total_bps oscillated 2.5-3.5 bps for most of training (worse than
baseline throughout), reached its single worst 10-firing block average
(3.34 bps) around steps 1.0M-1.1M, and only in the last ~500k steps before
the divergence settled into a modestly lower, still-noisy band (~2.1-2.5
bps) -- still 2-2.5x worse than baseline. A few sharp unexplained
collapses (steps ~670k, ~900k-970k) briefly dropped eval IS below baseline
(0.4-0.6 bps) then reverted to 3+ on the very next firing -- read as
instability, not a discovered edge. No conclusion about final policy
quality should be drawn from these training-time firings; that is what the
separate n=500 round is for.

Next planned step (separate round, not started): n=500 evaluation against
the pre-registered bar (beat TWAP-passthrough), run on BOTH the final
checkpoint and the pre-divergence ~1.6M checkpoint, followed by the same
diagnostic battery run on the original negative (post-mortem diagnostics,
volatility-stratified bucketing) if either checkpoint's n=500 result
warrants it. Test split stays untouched until then.

PRIOR ENTRY BELOW, for the reward-redesign implementation round's own context:

Last updated: 2026-08-27 00:58 HKT
State: REWARD REDESIGN IMPLEMENTED, RE-MEASURED, AND SHAKEDOWN-VERIFIED.
Real multi-day training run NOT launched -- stopping here for review per
explicit instruction. Test split still unspent.

TASK 1 (implement): src/envs/l2_reward.py (new) -- potential-based
mark-to-market IS shaping, Phi(t) = -kappa*compute_implementation_shortfall(
episode_fills_so_far, ..., terminal_mid_price=CURRENT mid).is_total_bps,
reward = Phi(t)-Phi(t-1). Reuses compute_implementation_shortfall() as-is
(not re-derived), so telescoping to the real terminal IS is exact by
construction, not approximate. Wired into FrozenL3Wrapper via new
l2_reward_mode param (default "l3_passthrough", unchanged existing
behavior; opt-in "potential_is_shaping") and train_l2.py's new
--l2-reward-mode CLI flag (same two choices). tests/test_l2_reward.py:
4 hand-computed pure-function tests, plus the hard gate --
test_potential_is_shaping_telescopes_exactly_on_real_episodes, 5 real
episodes, real frozen L3 checkpoint, summed shaped reward vs
-kappa*terminal_is within 1e-6 -- PASSED. Scale/variance sanity test (20
real episodes, both modes): new reward is tighter (mean|r|=0.4207) than old
(mean|r|=0.9046), not wildly different -- PASSED. Full suite: 50/50 passed
(112.71s), including all 43 pre-existing tests unchanged. Committed f728772.

TASK 2 (re-measure): scripts/analyze_l2_reward_components_v2.py, same
methodology as the original 100-real-val-episode / TWAP-passthrough-action
measurement. Headline confirmed: terminal-IS-derived signal is now 100% of
net reward and 100% of magnitude by construction (vs 6.9%/11.6% under
l3_passthrough). Internal composition of that signal (exec/opportunity/fees
sub-terms, per-window Phi-deltas, 3260 windows): by magnitude opp 80.4%,
exec 12.8%, fees 6.8%. fees deltas are one-signed (fee_bps_per_fill*
fill_ratio is monotonic non-decreasing over an episode) -- abs-total equals
signed-total exactly, a small steady drag, not a variance source. opp/exec
show heavy signed-vs-abs cancellation, confirming the shaping now supplies a
dense, market-linked per-window signal rather than a single terminal
payout -- the intended mechanism of potential-based shaping on an otherwise
sparse terminal reward. Committed ba5840f.

TASK 3 (shakedown, ~57 real minutes wall-clock, run_name=l2rewardshakedown1,
--n-envs 6, --total-timesteps 65000, --l2-reward-mode potential_is_shaping,
full production path -- real frozen L3 checkpoint sha256-verified against
the doc-recorded value, real numeric-format data, eval on, checkpoint at
default 50,000-step cadence):
- Throughput: ~18.97 dec/s steady-state (64,938 timesteps / 3,423s at the
  final logged point), matching the real l2v1_20260825 run's own 18.93 dec/s
  baseline within noise -- the new reward mode does not change per-step cost
  materially.
- 0 tracebacks/errors/NaNs anywhere in the full log.
- Checkpoint mechanics confirmed correct: fired at 49,998 steps, all three
  expected files present (model 3,459,861 bytes; replay buffer 174,001,374
  bytes, matching the known fixed ~174MB footprint; vecnormalize 3,997
  bytes).
- ValISEvalCallback fired all 6 expected times (steps 10002/20004/30006/
  40008/50010/60012), each completing cleanly against the paired-seed
  TWAP-passthrough baseline (0.9976 IS_total_bps). L2's own eval numbers
  this early (2.07-3.62 IS_total_bps) are NOT a performance signal -- 65k of
  what would be a multi-million-step budget, mechanics check only, no
  conclusion drawn from them.
- Canonical-checkpoint guard worked exactly as designed: models/
  l2_strategist_v1.zip / l2_vecnormalize.pkl (the real l2v1_20260825
  checkpoint) were correctly left untouched (--overwrite-canonical not
  given); final save went to models/l2_strategist_v1_l2rewardshakedown1.zip
  / l2_vecnormalize_l2rewardshakedown1.pkl instead, confirmed by direct
  listing.
- Memory: NOT fully re-verified this round -- only one spot-check was taken
  (main process RSS ~4.08GB at ~350s elapsed via a fresh ps snapshot), not a
  continuous trend the way the real 29h run's own explicit RSS tracking
  was. No crash/OOM occurred (clean run to completion is itself evidence
  against a fast leak), and the new reward path adds only a single float
  (self._l2_prev_phi) per env and reuses the existing
  compute_implementation_shortfall() call already made every step under the
  old mode -- no new persistent buffers, so no structural reason to expect
  a regression versus the old mode's already-confirmed-stable ~22.7GB at
  n_envs=6. Flagged honestly as a gap, not asserted as measured.

Files touched this round: src/envs/l2_reward.py (new), src/envs/wrappers.py
(pure insertion, +52/-0), src/train/train_l2.py (pure insertion, +20/-0),
tests/test_l2_reward.py (new), scripts/analyze_l2_reward_components_v2.py
(new). All committed locally, nothing pushed to origin/master.

STOPPING HERE FOR REVIEW per explicit instruction -- the real multi-day run
was not launched. If/when it is: TWAP-passthrough stays the valid eval
baseline (never touches L2's reward either way); this run's checkpoint
under the new reward is NOT training-reward-comparable to l2v1_20260825's;
run the identical diagnostic battery (n=500 eval, the three post-mortem
diagnostics, volatility-stratified bucketing) before drawing any conclusion,
no shortcuts because the reward changed. Test split remains untouched.

PRIOR ENTRY BELOW, for the design-only round's own context:

Last updated: 2026-08-27 12:30 HKT
State: L2 REWARD REDESIGN -- DESIGN ONLY, NOT IMPLEMENTED, NO TRAINING RUN.
Full detail: docs/reports/l2_reward_redesign_proposal.md (new, committed
661aea9, alongside scripts/analyze_l2_reward_components.py). Summary here,
not duplicated in full.

The test-split confirmation run from the prior entry was interrupted by
explicit instruction before producing any result -- models/l2_test_
confirmation.json does not exist, the log is empty, nothing was seen. The
pre-registration (commit 2de9fab) and the smoke-tested runner (commit
d578a3a) both still stand, untouched and ready to resume whenever that task
is picked back up; the test split remains unspent.

WHY this round: L2 has never had its own reward -- FrozenL3Wrapper sums
L3's per-tick step_reward() over each 50-tick window and hands that to SAC.
That reward was built for a tick-level executioner; L2 chooses
participation rate and urgency at 5s cadence and controls none of the
tick-level order-type/price/cancel decisions most of those components
price.

MEASURED (not asserted): instrumented step_reward()/compute_implementation_
shortfall() via a verified monkeypatch (0 mismatches across 154,192 real
ticks, 100 real val episodes, trained L2 checkpoint, deterministic) --
r_stale alone is 85.6% of L2's net accumulated reward and 75.4% of its
total signal magnitude. Terminal IS, the metric L2 is actually evaluated
on, is 6.9%/11.6%. r_stale outweighs terminal IS by ~12x in the mean.
r_placement_stale is exactly 0 (eta_replace=0 in production, structurally
correct, not a bug).

PROPOSED PRIMARY DESIGN: potential-based mark-to-market IS shaping --
Phi(t) = a running implementation-shortfall estimate using current
executed_frac and current mid_price, evaluated at each L2 decision;
reward = Phi(t) - Phi(t-1). Telescopes to the real terminal IS over the
episode while paying dense, per-decision credit along the way. Explicitly
argued against the TWAP-baseline (variance-reduction) reward's own
documented L3 failure (docs/reports/l3_twap_baseline_reward.md) rather than
re-proposing it: that change subtracted a SEPARATE reference trajectory
(a TWAP shadow) with its own independently-fixed exposure window, and the
real agent's own drift exposure shrinking on early completion created an
incentive to rush relative to that fixed comparison point -- the
mechanism the report itself observed behaviorally (episode length roughly
halved, fill_ratio up, occasional costly tail). L2 CAN plausibly trigger
the same effect (aggressive participation_rate_multiplier -> earlier
completion), so this is not proposed for L2 either. The primary design has
no separate reference trajectory at all -- Phi is a function of the real
agent's own state and the real current price only, sidestepping that
specific failure mode structurally, not by argument alone.
ALTERNATIVE (minimal-diff, not primary): drop r_queue/r_spread/r_stale/
r_placement_stale from L2's aggregation, keep only r_inv (weak but
plausible L2 attribution) + terminal IS with kappa raised. Simpler,
less implementation risk, doesn't solve sparse credit assignment as well.

IMPLEMENTATION PLAN (not started): new src/envs/l2_reward.py (pure
function, same style as reward.py), a gated l2_reward_mode parameter on
FrozenL3Wrapper defaulting to current behavior (same opt-in convention as
zeta/eta_replace/subtract_twap_baseline), a telescoping unit test (sum of
per-window shaped rewards must match compute_implementation_shortfall()'s
own terminal number) as the core correctness check, seed-equivalence check
on the untouched default path. L3's own training/reward.py/frozen
checkpoint: untouched. Comparability: a reward change means the current
checkpoint's training-time numbers stop being comparable to a redesigned
run's; TWAP-passthrough stays the valid comparison point either way since
it's reward-independent by construction.

HONEST EXPECTED VALUE: ~15-25% odds this flips the sign to beating
TWAP-passthrough with both tests agreeing. The credit-assignment problem is
real and measured, not hypothetical -- but frozen L3 itself (tick-level
control, matched reward, 20M steps) only ties TWAP, and the
volatility-stratified result was a remarkably consistent null across
regimes, both leaning toward "limited exploitable structure at this
cadence/instrument" over "reward noise was hiding something real." Worth
fixing regardless, since a clean signal is table stakes for trusting any
future result either way. Distinguishing test proposed: retrain under the
new reward, rerun the exact same diagnostic battery (val/train/volatility
strata) already built this round, before anything else -- clean training
dynamics + still-null result would point to "no structure"; a genuine,
replicating positive would confirm "reward was the binding constraint."

Not implemented, not trained, no checkpoint touched, test split untouched.
Reported for review per instruction -- awaiting decision on whether to
proceed to implementation.

PRIOR ENTRY BELOW, for context on the test-split pre-registration (still
standing, unresumed):

Last updated: 2026-08-27 11:00 HKT
State: PRE-REGISTRATION for the test-split confirmation run, committed BEFORE
any test-split evaluation is executed -- the whole point of a holdout is
declaring terms before the number is seen, so this commit's timestamp is the
record that these terms predate the result, not a post-hoc description of
them. No test-split code has run yet as of this commit.

CLAIM UNDER TEST: "L2 (trained) does not achieve lower IS_total_bps than
TWAP-passthrough on the held-out test split at n=500, with both paired tests
(t-test and Wilcoxon) agreeing at p<0.05."

INTERPRETATION OF EACH OUTCOME, FIXED NOW:
- L2 loses or ties (fails to beat passthrough with both tests agreeing
  p<0.05): confirms the existing body of evidence (val, unrestricted train,
  and all three volatility strata including the zero-val-overlap high
  bucket). The negative conclusion stands, now with a genuinely independent
  confirmation.
- L2 WINS (beats passthrough with both tests agreeing p<0.05): this
  contradicts every prior measurement in this project. The correct reading
  is "one anomalous result against a large, consistent body of contrary
  evidence," NOT "L2 works after all." A single 18-day window does not
  overturn val (500 episodes) + unrestricted train (500) + calm/moderate/
  high volatility strata (292/500/500) all showing the same null-to-negative
  pattern. Any writeup encountering a test-split win must carry this framing
  explicitly, not treat the holdout as having overturned the rest.

SCOPE NOTE, also fixed now: test sits at the ~21.5th percentile of train's
volatility distribution (models/l2_day_conditions_test.csv, computed last
round) -- calm, like val (23.7th percentile). Whatever this run finds, it
confirms or contradicts the CALM-regime finding specifically. The
high-volatility question remains genuinely unverified out-of-sample
regardless of this result -- no split, including test, samples train's
volatile tail. This scope limitation does not change based on which way the
test result comes out.

MECHANICS: same n=500 harness (scripts/eval_l2_n500.py's own functions,
imported), same three arms, same EVAL_SEED_BASE=5,000,000 paired-seed
convention, same methodology -- pointed at load_split("test") instead of
"val". This is the ONLY evaluation the test split will ever get from this
project: one run, no re-runs, no parameter adjustments after seeing the
number. The script will be mechanically smoke-tested on val (n=5, NOT test)
before the real run, per the same discipline every other real launch this
project has used -- testing the SCRIPT is not the same as spending the
HOLDOUT, and only the latter is restricted to once.

Next: build+verify the script, then the single real n=500 test run, then a
follow-up entry with the actual result -- kept as a separate commit so this
one stands unedited as the pre-registration record.
State: VOLATILITY-STRATIFIED EVALUATION COMPLETE -- the negative closes out.
No retraining, no checkpoint changes, test split's own evaluation still
untouched. Follow-up to the prior entry's regime-matching finding (train's
whole-pool L2-vs-passthrough advantage collapsed to near-zero when
restricted to val's own volatility range): does that advantage
correspondingly STRENGTHEN above val's range, where val has no equivalent
at all? Built scripts/eval_l2_bucketed.py (new, committed 27cc986) --
restricts LOBExecutionEnv's file pool to an explicit, non-contiguous date
bucket via a safe post-construction override of self._files (set before any
reset(), cannot perturb the RNG-driven file_idx draw order -- equivalent to
having constructed the env with that file set from the start). Reuses
eval_l2_n500.py's functions unchanged.

Three buckets (day counts from models/l2_day_conditions_train.csv's own
realized_vol_bps column, val's own max = 0.1882bps as the natural boundary):
  calm     (vol <= 0.1882, matches val's own range): 231 days
  moderate (0.1882 < vol <= 0.30, no val equivalent): 137 days
  high     (vol > 0.30, train's own top ~9%, no val equivalent): 37 days
n=500 for moderate and high (fresh runs, this round); calm reuses the prior
round's regime-matched result (n=292, post-hoc filtered from the original
whole-train run -- not re-run, same underlying data).

RESULT -- L2 vs TWAP-passthrough, per bucket:
  calm:     mean_diff=-0.0129bps  n=292  t-test p=0.937   Wilcoxon p=0.994
  moderate: mean_diff=+0.0362bps  n=500  t-test p=0.825   Wilcoxon p=0.221
  high:     mean_diff=-0.0762bps  n=500  t-test p=0.804   Wilcoxon p=0.923
                                          d_z=-0.0111 (negligible)
NO STRENGTHENING. All three buckets sit near zero with no consistent sign or
trend, and none clear significance in either test. The high-volatility
bucket -- specifically constructed to have NO counterpart anywhere in val,
520 fresh episodes across train's most volatile 9% of days -- shows
essentially nothing (effect size an order of magnitude below even a small
effect). Per the pre-stated criterion for this task: high-vol shows nothing,
so the negative conclusion gets substantially stronger, and this specific
line (does L2 have hidden value in volatile regimes it never gets credit
for on calm-skewed val) is closed out. Not checking for unseen high-vol
data elsewhere in the archive, per instruction (that check was conditional
on a real high-vol edge showing up; it didn't).

Also worth recording plainly, a point underweighted in the prior entry: the
ORIGINAL whole-train advantage (-0.2532bps) was never actually
double-test-significant even on its own terms -- Wilcoxon was p=0.3354 in
that very same run, only the t-test came close (p=0.0703, itself still
>0.05). The cross-split significance established in the prior entry (Task
1, p=0.014) was about the SWING between splits being real, not about
train's own within-split number being a robustly confirmed advantage to
begin with. That the volatility decomposition finds it nowhere in
particular is consistent with this having been a marginal, borderline
aggregate effect from the start rather than a real, localizable phenomenon
that a stratified look failed to find. Fully reconciles: nothing here
contradicts Task 1's finding that the CROSS-SPLIT swing is real -- it just
means the swing is better explained by regime composition (as the prior
entry's ~52% estimate already argued) than by L2 having a genuine,
findable edge anywhere in particular.

MANDATORY CAVEAT, stays attached to every number above: these are TRAINING
days. Even a strong result in the high bucket could not have been
distinguished from memorization of those 37 specific days rather than a
real, transferable volatility-conditioned skill -- this diagnostic was
built to be unable to tell those apart, by design (see the script's own
module docstring). Since the result came back null anyway, this caveat is
now moot for THIS finding specifically, but it applies to any future reader
who might otherwise be tempted to read a favorable train-side number as
held-out evidence.

TEST-SPLIT RECOMMENDATION (recommending only, not acting): stronger than
the prior entry's. L2 now shows no real edge over TWAP-passthrough anywhere
tested -- not on val, not on unrestricted train, not in any of three
volatility-stratified train subsets including one specifically chosen to
have no val counterpart. The diagnostic phase looks complete: every
plausible confound (absolute-vs-relative comparison, regime mismatch,
volatility-conditioned value) has been checked and none rescues a positive
read. Spending the test split now would be a clean, final, confound-free
confirmation -- recommended as the next and likely last step before writing
this up, not as another diagnostic round.

Files: scripts/eval_l2_bucketed.py (new, committed 27cc986). Output
artifacts left uncommitted, same precedent as prior entries:
models/l2_bucketed_{moderate,high}.json.

Reported, not acted on further -- test-split spend is a recommendation, not
a decision made here. Whoever owns next steps decides.

PRIOR ENTRY BELOW, for context on the regime-matching correction that
prompted this check:

Last updated: 2026-08-27 06:40 HKT
State: CORRECTION to the prior entry's Diagnostic 2 conclusion, plus a new
split-representativeness check that turned up a project-wide concern, not
just an L2 one. No retraining, no checkpoint changes, test split's own
per-episode evaluation still untouched (its raw market data was read for
descriptive stats only -- see below for why that's a different thing).

CORRECTION (Task 1): the prior entry's "no learnable signal, not overfitting"
verdict was drawn from the WRONG comparison -- L2's absolute number staying
flat between train (1.2384) and val (1.2330). That's the wrong test when the
baselines themselves moved (TWAP-passthrough 1.0237->1.4916, Pure TWAP
0.8893->1.2555 -- train days are harder overall). The RIGHT comparison is
L2 relative to each baseline within its own split:
  L2 - TWAP-passthrough:  val=+0.2094 (worse)   train=-0.2532 (better)
  L2 - Pure TWAP:          val=+0.3438 (worse)   train=-0.0171 (~tie)
Re-tested properly with a cross-split significance test (Welch's t + Mann-
Whitney U comparing the per-episode diff distributions from each split, not
just eyeballing two point estimates): the sign flip vs TWAP-passthrough IS
real (Welch t=-2.4656 p=0.0138, Mann-Whitney p=0.0137) -- a genuine ~0.46bps
swing, not noise. The sign flip vs Pure TWAP is NOT distinguishable from
noise (Welch p=0.459, Mann-Whitney p=0.548) -- Pure TWAP's variance is much
higher on train (std=10.24 vs 3.65 on val) and swamps it. TWAP-passthrough
is the cleaner comparison anyway (same L3 wrapper both arms; only L2's
steering differs), so this is the one that matters. Retracting "no learnable
signal" -- it does not hold up under the correct comparison.

TASK 2 -- is this real overfitting, or is val a harder regime (the
competing explanation)? Built scripts/analyze_split_representativeness.py
(new, committed 904d5bb) -- day-level realized_vol/mean_spread/|return|
computed directly from the raw price series (read_day(), no model
inference, no episode evaluation) for all 405 train days, 18 val days, and
(descriptive only) the 18 test days.
Result: val is NOT choppier/wider-spread than train -- it is significantly
CALMER on every metric (val's mean sits at train percentile 23.7 for
realized_vol, 27.4 for mean_spread, 32.8 for |return|; Mann-Whitney
p=0.0004/0.0019/0.0065). This is the OPPOSITE direction from what the naive
"(b) val is choppier" explanation needs -- so as literally stated, (b) is
rejected; the raw premise that val is a harder regime is false.
But a deeper check (restricting TRAIN episodes to only the 231/405 days
that fall within val's own volatility range, matching regimes rather than
splits) found something more specific: train's own -0.2532bps advantage
over passthrough is NOT uniform across train -- on volatility-matched-to-val
train days it shrinks to -0.0129bps (n=292, t-test p=0.937, not
significant), i.e. essentially the same regime-conditional advantage val
itself would need to show for "no overfitting" to hold. Comparing this
CALM-MATCHED train subset directly against val (both now covering the same
volatility range): residual swing = -0.2223bps (vs the original -0.4626bps
before matching) -- about 52% of the original swing is explained by
train's own aggregate being pulled by its higher-volatility days, which
val has no counterpart for at all (val's own volatility max, 0.1882, sits
below train's median). The remaining ~48% residual, while still pointing
the same direction (train even matched-regime nominally beats val), is
itself NOT statistically significant at this smaller matched sample (Welch
t=-1.079 p=0.281, Mann-Whitney p=0.095).
VERDICT: (c), both contribute, and roughly in proportion: ~52% regime
confound (NOT the specific "val is choppier" mechanism hypothesized --
rather "train's own high-volatility tail inflates its aggregate advantage
in a way val's narrower, calmer range never gets to demonstrate either
way"), ~48% residual that is directionally consistent with genuine
memorization/overfitting but not itself confirmed at the achievable n. This
is a real correction to a real correction: the swing is genuine (Task 1),
but it is NOT cleanly "textbook overfitting" once regime is controlled for
-- it is a mix, and the overfitting component specifically is suggestive,
not proven.

TASK 3 -- what this means beyond L2, stated plainly, not softened: val's
18 days are not just "not choppier than train" -- they are a NARROW slice
of train's overall regime diversity. Val's own realized_vol range
([0.058, 0.188]bps) sits entirely inside roughly train's bottom third
(train's own max is 0.984, over 5x higher); val never samples the more
volatile days that make up a real fraction of train's 405-day distribution.
EVERY conclusion in this project measured on val alone (L3's own "ties
TWAP" result, the REPLACE-has-no-value finding, the TWAP-baseline A/B) was
therefore only tested in this same narrow, calm-skewed slice of market
conditions -- none of them have been checked against the wider volatility
range that a real 405-day (or longer, in production) history actually
contains. This is a genuine, previously-unstated scope limitation, not a
claim that any of those results are wrong: it means "ties TWAP" etc. should
be read as "ties TWAP in calm-to-moderate conditions, unverified beyond
that," not as a general claim across all regimes. A final writeup should
carry this caveat explicitly rather than treat val-measured results as
settled across the board.
Test-split recommendation (recommending only, not acting): test's own
day-conditions percentiles (21.5 vol / 45.2 spread / 51.6 |return|, vs
val's 23.7/27.4/32.8) put it CLOSER to train's median than val on spread
and return, though similarly calm on volatility specifically. Spending test
now would be a genuine, useful independent confirmation point -- a
different calendar window reduces the "got an unlucky/lucky 18-day draw"
risk that a single held-out window always carries. It would NOT resolve the
deeper volatility-representativeness gap above, since test also skews calm
on that specific axis (percentile 21.5, close to val's own 23.7) -- no
existing split samples train's genuinely volatile tail at all. If someone
wants that question answered, it needs a deliberately volatility-stratified
check (e.g. evaluating against train's own high-volatility subset directly,
same technique used above), not spending test. Recommend: worth spending
test soon as one more confirmatory read given the project is at a real
decision point on L2, but go in knowing it answers "does this replicate on
different dates," not "does this hold up in volatile regimes" -- those are
different questions and test cannot resolve the second one either.

Files: scripts/analyze_split_representativeness.py (new, committed 904d5bb).
Output artifacts left uncommitted, same precedent as prior entries:
models/l2_day_conditions_{train,val,test}.csv. The Task 1 cross-split test
and Task 2's regime-matching residual test were one-off analyses (not
committed as reusable scripts) built directly on the already-saved
models/l2_diagnostics_{val,train}_episodes.csv -- numbers recorded here in
full since the scripts themselves weren't kept.

Reported, not acted on further -- test-split spend is a recommendation, not
a decision made here. Whoever owns next steps decides.

PRIOR ENTRY BELOW, for context on the post-mortem diagnostics themselves:

Last updated: 2026-08-26 10:05 HKT
State: POST-MORTEM DIAGNOSTICS COMPLETE on the n=500 negative (prior entry
below). Three diagnostics, all cheap (no retraining), scripts/eval_l2_diagnostics.py
(new, committed 968daa9) -- reuses eval_l2_n500.py's exact functions
(imported, not reimplemented), verified correct by reproducing that run's
exact val arm means (1.2330/1.0237/0.8893) before trusting anything else it
reports. Test split untouched throughout.

DIAGNOSTIC 1 -- collapsed or actively steering? ACTIVELY STEERING (b), not
collapsed (a). Logged all 14,727 real decisions across the val n=500 run:
participation_rate_multiplier mean=1.1123 std=0.6429 (neutral=1.0,
bounds=[0,2]), spanning p1=0.0045 to p99=1.9981 -- nearly the full range.
urgency mean=0.5008 std=0.3335 (neutral=0.5, bounds=[0,1]), p1=0.0000 to
p99=0.9992 -- same pattern. Saturation at either bound is rare (0.5-3.1% of
decisions). Within-episode std (0.48 participation / 0.26 urgency) confirms
it responds to state mid-episode, not just picks one action and holds it;
between-episode std (0.37 / 0.20) confirms different episodes get different
treatment too. The near-neutral MEANS are simply the average of a genuinely
wide, responsive distribution, not evidence of a constant action -- the
policy learned real, substantial, non-degenerate behavior. It just isn't
useful behavior (see Diagnostic 2/3, and the already-recorded n=500 result).

DIAGNOSTIC 2 -- overfitting or no learnable signal? NO LEARNABLE SIGNAL (b),
not overfitting (a). Ran the identical n=500 harness against TRAIN dates
(train_date_range=('2024-04-18','2025-07-15'), 405 real days -- confirmed
this is exactly what the real training run itself used, via
logs/l2_train_real_l2v1_20260825.log's own startup print, so this is the
real in-sample comparison, not an approximation).
  L2 (trained):      val=1.2330  train=1.2384  (gap: +0.0054bps, negligible)
  TWAP-passthrough:  val=1.0237  train=1.4916  (baseline itself is harder on train)
  Pure TWAP:         val=0.8893  train=1.2555  (baseline itself is harder on train)
L2's OWN absolute number is statistically indistinguishable between data it
trained on for 2,000,000 steps across 405 days and data it never saw at all
across 18 days -- if there were exploitable train-specific structure, 2M
steps of exposure should have produced at least some gap. There isn't one.
Note the baselines themselves ARE harder on train (market conditions differ
across the two chronological windows, expected) -- L2 doesn't track that
shift either direction, its absolute output level is essentially fixed
regardless of which regime it's actually in. On train, L2 nominally beats
TWAP-passthrough (mean_diff=-0.2532bps) but NOT significantly (t-test
p=0.0703, Wilcoxon p=0.3354 -- neither clears 0.05, and this is IN-SAMPLE
data); vs Pure TWAP the effect is ~zero (d_z=-0.0017). Even on the exact
data it trained on, L2's improvement over doing nothing special isn't
statistically real. Action distribution shape also barely differs between
splits (participation_mult mean=1.1123/std=0.6429 val vs mean=1.1738/
std=0.6464 train; urgency mean=0.5008/std=0.3335 val vs mean=0.5376/
std=0.3273 train) -- the policy behaves essentially the same way regardless
of whether it has seen the data before.

DIAGNOSTIC 3 -- broad-based or a few bad days? BROAD-BASED, not a few
outliers. Per-day breakdown of the val n=500 result (18 days, 16-36 episodes
each -- balanced, no single day's mean is built on a tiny unreliable
sample): L2 worse than TWAP-passthrough on 12/18 days (67%), worse than Pure
TWAP on 14/18 days (78%). Per-day L2-minus-passthrough diff ranges -0.667 to
+1.187bps (mean +0.188, std 0.571) -- a real spread on both sides, not one
or two catastrophic days dragging an otherwise-good aggregate. This is a
majority-of-days negative, not a regime-specific failure limited to a
handful of dates.
Extra (lightweight, n=18 so suggestive not conclusive): correlated the
per-day L2-vs-passthrough diff against day-level conditions computed
directly from the numeric day files (day return, realized volatility, mean
spread). |day_return_bps| r=0.469, realized_vol_bps r=0.533, mean_spread
r=0.493 -- all positive, moderate. Day DIRECTION (signed return) has ~zero
correlation (r=-0.024). Reading: L2 tends to do relatively worse (vs.
passthrough) on higher-volatility/wider-spread/bigger-swing days, and
relatively less-bad on calm, tight-spread days -- plausible but not proven
at n=18; a real pattern worth someone revisiting if this gets pursued
further, not a standalone conclusion.

OVERALL READ: "the approach doesn't work here," not "the training procedure
failed and a fix is worth trying" -- reasoning, not just the verdict. If this
were primarily a training-procedure failure (undertrained, bad
hyperparameters), the two most likely fingerprints would be a collapsed/
degenerate policy (ruled out by Diagnostic 1 -- it's genuinely, substantially
responsive) or a large train/val gap from latching onto train-specific
patterns even if the wrong ones (ruled out by Diagnostic 2 -- train and val
are statistically indistinguishable for L2's own number, and even IN-SAMPLE
performance doesn't clear significance against either baseline). What's left
is a policy that learned real, responsive, non-trivial behavior that simply
doesn't map onto better execution outcomes anywhere it's tested -- train,
val, most individual days. That reads as the observation/action-space
transform/reward combination not containing enough exploitable structure
for SAC to find a genuinely useful policy here, not a fixable training-run
defect. Secondary, unresolved consideration worth flagging honestly: actor/
critic losses grew substantially and never visibly plateaued across the full
2,000,000-step run, and ent_coef kept climbing throughout (0.0054 -> 0.132) --
consistent with a policy that hadn't fully settled by the end of training.
This doesn't overturn the diagnostics above (a not-fully-converged policy
would still show SOME train/val gap if there were real signal to
overfit toward, and still wouldn't need to be this actively responsive to
have just collapsed toward mediocrity) -- but it does mean "an even longer
run would clearly still fail" is not proven either, only "this specific run,
budget, and reward did not produce a useful policy."

Files: scripts/eval_l2_diagnostics.py (new, committed 968daa9). Output
artifacts left uncommitted, same precedent as models/l2_n500_eval_result.json:
models/l2_diagnostics_{val,train}.json, models/l2_diagnostics_{val,train}_
episodes.csv, models/l2_diagnostics_{val,train}_actions.csv,
models/l2_diagnostics_val_per_day{,_with_conditions}.csv.

Test split (data/splits/l2_bybit_btcusdt_split.json's test_dates) was not
touched by any of the above, per instruction. Flagging per instruction,
NOT acting on it: whoever owns next steps may judge that the remaining
untouched test split's time has come for a final confirmation run -- but
that is a deliberate, one-time spend or someone else's call, not something
to do as part of this round's own diagnostics.

Reported, not acted on further -- whoever owns next steps decides between
revisiting the approach (reward shaping, action-space transform, observation
set) versus abandoning this direction given the frozen L3 baseline it sits
on top of already only ties TWAP itself.

PRIOR ENTRY BELOW, for context on the n=500 evaluation result itself:

Last updated: 2026-08-26 08:14 HKT
State: REAL n=500 EVALUATION COMPLETE (scripts/eval_l2_n500.py, ~42 real
minutes, 3 arms x 500 paired episodes, val_date_range=('2025-07-16',
'2025-08-02'), seeds 5000000-5000499). L2 (trained, models/l2_strategist_v1.zip
+ l2_vecnormalize.pkl) does NOT beat the pre-registered bar.

Results (IS_total_bps mean +/- std, fill_ratio, lower IS is better):
- L2 (trained):        1.2330 +/- 3.4546, fill_ratio=0.9206
- TWAP-passthrough:     1.0237 +/- 3.5467, fill_ratio=0.9192
- Pure TWAP:            0.8893 +/- 4.3525, fill_ratio=0.9963
Pure TWAP's number here matches docs/reports/l3_frozen_handoff.md's own
recorded TWAP baseline (0.889) almost exactly -- a real sanity check that
this harness reproduces the known-good figure, not a coincidence to ignore.

Paired comparisons (L2 vs. each baseline, positive mean_diff = L2 WORSE):
- vs TWAP-passthrough (required bar): mean_diff=+0.2094bps, d_z=0.0748 (tiny),
  t-test p=0.0955 (not significant), Wilcoxon p=0.0068 (significant) -- the
  two tests DISAGREE, and the direction is unfavorable to L2 regardless.
- vs Pure TWAP (stretch goal): mean_diff=+0.3438bps, d_z=0.0942 (tiny),
  t-test p=0.0359 (significant), Wilcoxon p=0.0718 (not significant) -- same
  disagreement pattern, same small unfavorable effect size.
Per this project's own established caution about significance without
magnitude (the budget-extension result's d_z=0.076 precedent, referenced when
this harness was built): both effect sizes here are in that same negligible
range -- even where one test nominally clears p<0.05, the practical
difference is small AND in the wrong direction.

Harness's own verdict (models/l2_n500_eval_result.json): beats_twap_passthrough
= false, beats_pure_twap = false. Required bar (beat TWAP-passthrough, both
tests agreeing, p<0.05 each) is NOT met.

Worth noting for whoever picks this up next: the in-training ValISEvalCallback
signal (n=10/firing, noisy) sat much higher (mostly 2.5-4.0 over the run's
back half) than this real n=500 read (1.23) -- the small-n training-time
signal overstated how bad things looked, though the real, larger-sample
answer is still "did not beat baseline," just by a smaller and statistically
murkier margin than the noisy signal suggested.

scripts/replay_episode.py (built and verified during the training run, see
prior entry below) is available for a qualitative look at individual
episodes -- not run this round.

Reported plainly, no further action taken this round -- whoever owns next
steps decides whether to revisit reward shaping, training budget, or
architecture given this result.

PRIOR ENTRY BELOW, for context on the training run itself:

Last updated: 2026-08-26 06:36 HKT
State: REAL 2,000,000-STEP TRAINING RUN COMPLETE. run_name=l2v1_20260825, ran
~29.34h wall-clock (105,630s), total_timesteps=1,999,974. Throughput held
steady at 18.8-18.93 dec/s across the entire run (consistent at every check-in
from the 1h mark through completion -- never still settling, never
degrading). No mechanical failures across the full run: all 9 processes
(main + resource_tracker + forkserver + 6 workers) alive throughout, verified
directly via /proc at every check-in, not inferred from a pgrep pattern count
(one check-in's pgrep undercount was caught and corrected mid-run -- see this
session's own record if the detail matters). RSS held flat in a tight
25.8-25.9GB band the entire run, no leak. Checkpoints landed on the expected
~44-minute/49,998-step cadence with no gaps. Zero tracebacks/errors/NaNs
anywhere in the full log.

Final save confirmed at the canonical paths: models/l2_strategist_v1.zip
(3,459,901 bytes) and models/l2_vecnormalize.pkl (3,997 bytes), both written
2026-08-26 06:34 HKT -- the first real canonical L2 checkpoint to exist (the
pre-launch round's resume-test mishap created and then deleted these same
paths once; this is the genuine article).

Final eval-callback numbers (paired seeds 5000000..5000009, n=10/firing,
TWAP-passthrough baseline IS_total_bps=0.9976 throughout): last 12 firings
(steps 1.88M-1.99M) ranged 2.63-3.62, last logged firing at step=1,990,398
mean=2.6331. Reported as-logged only, per explicit instruction -- NOT
evaluated against the pre-registered bar (beat frozen-L3-alone at 0.994,
ideally beat TWAP at 0.889, both paired tests agreeing) here. That is
scripts/eval_l2_n500.py's job (see its own entry below), a separate,
not-yet-run round.

Both scripts/eval_l2_n500.py (n=500 evaluation harness) and
scripts/replay_episode.py (qualitative episode visualizer) were built and
verified on synthetic data during the run specifically so they'd be ready
the moment a real checkpoint existed -- see their own entries below for what
each does and how each was verified. Next step for whoever picks this up:
point eval_l2_n500.py at models/l2_strategist_v1.zip + models/l2_vecnormalize.pkl
and the frozen L3 checkpoint used for training
(models/l3_frozen_backup/l3_executioner_v1_frozen.zip + l3_vecnormalize_frozen.pkl),
run the real n=500 evaluation, and separately point replay_episode.py at a
handful of real episodes for a qualitative read alongside it. Not started
this round, per instruction -- reporting completion is where this round
stops.

GPU/CPU capacity is free again -- this run no longer needs exclusive use of
the box.

No git push (commit locally only, per standing instruction). No protected
files edited.

PRIOR ENTRY BELOW, for context on the episode replay visualizer and n=500
eval harness built alongside the run:

Last updated: 2026-08-25 03:05 HKT
State: episode replay visualizer built alongside the live training run
(l2v1_20260825, ~85,800 gradient updates at last check, healthy, no errors). Zero
interference: no GPU, no real data, tiny synthetic checkpoints only, verified via
ps/free -h and the training log before and after this task.

scripts/replay_episode.py: given one episode (checkpoint/seed as required CLI
args, no defaults), produces one coherent 3-panel PNG + plain-language summary
aimed at a trading reader, not a codebase reader -- price path with best bid/ask
band, arrival price, terminal mid, and child-order placements marked by outcome
(filled / replaced before filling / still open at episode end / market-crossing
fill); execution progress vs. the linear TWAP schedule with ahead/behind shading
(this IS what schedule_deviation means, shown rather than named); L2's own
participation-rate multiplier and urgency over the same time axis so steering is
visible against what else is happening; a text summary with the Perold IS
decomposition (execution/opportunity/fees) against a same-seed TWAP baseline run
on the base env for a same-window comparison. Labels are trader-plain throughout
("2 ticks behind touch" via marker position, not "offset=-2") -- no obs indices
anywhere in the output.

Captures tick-level detail via monkeypatch instrumentation on the real, unmodified
FrozenL3Wrapper/LOBExecutionEnv (same pattern as scripts/profile_l2_throughput.py)
-- inert by construction: the capture hook exists only inside this script's own
process and is never installed unless this script runs; wrappers.py/
lob_execution_env.py were not touched at all, so nothing about the live run's own
code path changed.

One real, non-obvious case found while reading _place_limit() and handled
explicitly: a crossing LIMIT/CANCEL_AND_REPLACE routes through
walk_market_fill() exactly like a real MARKET order, so its own placement tick
can carry a non-maker fill -- without explicit handling this looked in the chart
like an order silently left open forever. 5 new permanent tests lock this in
(tests/test_replay_episode.py), covering both sides' placement-price formula
against _place_limit()'s own source, not guessed.

Verified on synthetic data only (tiny untrained checkpoints, CPU, ~400-row
synthetic day): a direct CLI-level smoke run confirmed the full pipeline --
argparse, real file I/O for all four checkpoint/vecnormalize paths, both the L2
episode and the same-seed TWAP comparison episode, figure generation -- runs
end to end. Caught and fixed one real bug in the process: the summary text box
was overflowing the saved PNG's right edge at the original line lengths (fixed
by explicit shorter lines + more bottom margin, not by trusting matplotlib's own
wrapping). Actually inspected the rendered PNG (not just "it didn't crash") to
confirm the fix and that the chart reads sensibly.

One known, honestly-flagged limitation: reconstruct_child_orders() does not
handle a crossing placement that fills only PART of the order and rests the
remainder on the same tick -- not observed in the one episode verified, and
_place_limit()'s own source wasn't fully read far enough to confirm whether this
case is even possible; flagged rather than silently assumed away.

Files: scripts/replay_episode.py, tests/test_replay_episode.py (new, committed).
Uncommitted, harmless: same throwaway storage-format prototype scripts noted in
the eval-harness entry below, untouched this round.

Blocking/open questions: none for this task. Next planned step: once
l2v1_20260825 completes, point this at the real checkpoint alongside
scripts/eval_l2_n500.py -- replay a handful of real episodes (both
representative and any n=500 outlier the eval harness flags) for qualitative
read alongside the statistical one.

PRIOR ENTRY BELOW, for context on the n=500 eval harness this task ran alongside:

Last updated: 2026-08-25 02:10 HKT
State: n=500 evaluation harness built alongside the live training run
(l2v1_20260825, ~2.6% through 2,000,000 steps at last check, healthy -- fps~19,
n_updates climbing, no errors) -- built now specifically so evaluation can start the
moment training finishes rather than beginning a build after a ~24h wait. Zero
interference with the live run: no GPU, no real data, no benchmarks, tiny synthetic
fixtures only, verified via ps/free -h and the training log before and after every
step this round.

While confirming the live run's health, found and fixed a real risk: src/envs/lob_execution_env.py
(use_numeric_format support) and src/data/l2_numeric_format.py were STILL sitting
uncommitted in the working tree -- the live run has depended on this exact
on-disk code since 01:14 (HEAD had zero use_numeric_format support before this),
a fragile state for a 24h run to depend on. Committed both as-is (no edits, no
behavior change -- pure protective commit) alongside the numeric-format permanent
regression test.

scripts/eval_l2_n500.py: reuses L3's own established n=500 methodology rather than
designing fresh (per instruction) -- same EVAL_SEED_BASE=5,000,000 paired-seed
convention and load_split("val") population as scripts/replace_value_probe.py, same
paired t-test + Wilcoxon signed-rank reporting, extended with Cohen's d_z effect
size on every comparison (this project has been misled before by significance
without magnitude -- the budget-extension result's own d_z=0.076 is the concrete
precedent). Three arms, paired across the same seed list: (1) the trained L2 SAC
policy + its own paired VecNormalize steering frozen L3 through FrozenL3Wrapper,
(2) TWAP-passthrough (L2 outputs [1.0, 0.5] every decision, frozen L3 unsteered --
ported directly from train_l2.py's own ValISEvalCallback, the same baseline the
in-training callback already tracks), (3) pure TWAP (phase2a_sanity_suite.py's
TWAPPolicy, unmodified, on the base env) -- directly poolable with the existing
table's TWAP row (0.889), not just comparable in spirit. Pre-registered success bar
stated in the harness's own output, before any real result exists: L2 must beat
TWAP-passthrough with BOTH tests agreeing (p<0.05 each), ideally also beat TWAP
itself. --n defaults to 500, configurable. CLI takes explicit
--l2-checkpoint/--l2-vecnormalize/--l3-checkpoint/--l3-vecnormalize, no defaults,
same discipline as train_l2.py.

Verified mechanics two ways, both synthetic/tiny/CPU-only: (1) 4 pytest tests
(tests/test_eval_l2_n500.py) covering both wrapped-env arms' episode loop, the
paired-report statistics (Cohen's d_z checked against a hand computation), and
end-to-end arm-running + comparison; (2) a direct CLI-level smoke run (real
argparse, real SAC.load/VecNormalize.load/RecurrentPPO.load from actual saved
files, not just the internal functions) against tiny untrained checkpoints,
n=2 -- confirmed the full path runs end-to-end and the pre-registered-bar check
correctly reports NO/NO for an untrained policy (no reason it should beat
anything). Real n=500 evaluation explicitly NOT run this round, per instruction.

Files: scripts/eval_l2_n500.py, tests/test_eval_l2_n500.py (new, committed).
src/envs/lob_execution_env.py, src/data/l2_numeric_format.py,
tests/test_numeric_format_equivalence.py (protective commit of already-in-use,
already-verified content, no edits). Uncommitted, harmless: several throwaway
storage-format prototype scripts from the prior round (scripts/convert_one_day*.py,
scripts/test_zstd_raw.py, scripts/check_level_counts.py,
scripts/compare_formats_equivalence.py, scripts/convert_l2_to_numeric.py) --
superseded by the already-committed scripts/convert_l2_to_numeric_parallel.py,
left in place rather than deleted since cleanup wasn't requested and touching
files unnecessarily during a live run isn't worth it for zero-risk clutter.

Blocking/open questions: none for this task -- harness is ready and waiting.
Next planned step: once l2v1_20260825 completes (or is deliberately stopped),
point scripts/eval_l2_n500.py at the real checkpoint
(models/l2_strategist_v1.zip + models/l2_vecnormalize.pkl, or the run-tagged
variants if --run-name-based paths were used) and the real frozen L3
checkpoint, run at n=500, report the real numbers against the pre-registered bar.

PRIOR ENTRY BELOW, for context on the training launch itself:

Last updated: 2026-08-25 01:20 HKT
State: REAL 2,000,000-STEP TRAINING RUN LAUNCHED AND IN PROGRESS. run_name=l2v1_20260825,
launched ~2026-08-25 01:14 HKT under nohup (survives SSH disconnect), log at
logs/l2_train_real_l2v1_20260825.log. Do not launch a second L2 training run
concurrently -- GPU/CPU capacity is claimed by this one until it completes or is
deliberately stopped.

n_envs=6, NOT 4 or 8 -- a live-measured result, not the earlier round's own extrapolation:
with each n_envs's own correctly-scaled gradient_steps (the UTD-preserving fix from the
vectorization round), n_envs=6 measured ~23.3 dec/s vs n_envs=8's ~21 dec/s -- 6 is BOTH
faster AND lighter (RSS ~22.7GB vs ~28.8GB) than 8 for this real script, not a tradeoff.
Root cause: gradient_steps scales linearly with n_envs (more GPU gradient-update work
per training() call), while raw env-stepping throughput from added parallelism hits
diminishing returns (already visible in the original harness sweep's own efficiency
numbers, 59.5% at n_envs=4 down to 39.7% at n_envs=8) -- past 6 workers here, the added
gradient cost outweighs the extra parallelism. Checked this isn't a CPU-oversubscription
artifact: 16 vCPUs available, 6-8 thread-capped workers + main process doesn't approach
that ceiling directly Full n_envs sweep this round (n_envs: dec/s / RSS): 4: ~17 /
~18.5GB, 6: ~23.3 / ~22.7GB, 8: ~21 / ~28.8GB -- all measured with --no-eval on an
otherwise-idle box, matching what production will actually see now that the user has
committed to keeping the box dedicated for the run's duration.

Pre-launch checks, all done and reported before starting per instruction:
- models/l2_strategist_v1.zip / l2_vecnormalize.pkl confirmed absent (the earlier
  resume-test mishap that once created them -- see the previous entry below -- was
  already cleaned up; re-confirmed fresh immediately before this launch).
- Fresh free -h/nvidia-smi/df -h/ps: 47GB RAM available, GPU 0MB used, 214GB disk free,
  ps showed only baseline OS/vscode/streamlit processes -- genuinely idle box confirmed,
  not assumed, immediately before committing to a 24+ hour run.
- --l3-checkpoint sha256 (a5443e2a4c6c1d4427d4ce1cb83e65d622ea688d8953f5bf94b29e87fbcaa77d)
  verified fresh (not from memory) against docs/reports/l3_frozen_handoff.md's own
  recorded value -- exact match. Paired VecNormalize sha256 also checked and matches.
- Wall-clock estimate corrected before launch, not left ambiguous: the ~23.3 dec/s
  n_envs=6 measurement was --no-eval (training-only), so 2,000,000/23.3 ~= 23.8h training
  + 200 eval firings x ~37.5s ~= 2.1h eval overhead = ~25.9 HOURS TOTAL, honestly stated
  as inclusive of eval, not the training-only figure alone.

Small addition made just before this launch, committed separately (9eec0be): main() now
prints every resolved arg (sorted, after all defaults/auto-resolution) at startup, so
logs/l2_train_real_l2v1_20260825.log is a self-contained record of exactly what ran even
without the original launch command.

Launch config (also in the log itself, printed in full): --l3-checkpoint
models/l3_frozen_backup/l3_executioner_v1_frozen.zip, --l3-vecnormalize
models/l3_frozen_backup/l3_vecnormalize_frozen.pkl (the doc-recorded frozen checkpoint,
NOT L3's own currently-live/mid-flux working checkpoint -- see this file's own L3 section
for why those differ right now), --total-timesteps 2000000, --n-envs 6, --seed 42,
--run-name l2v1_20260825, --use-numeric-format (explicit, though already the default),
--eval (explicit, already the default), --checkpoint-freq-timesteps/--eval-freq/
--n-eval-episodes all left at their defaults (50000/10000/10 -- already checked for disk
headroom and eval-overhead budget in the prior entry below). VecNormalize
(norm_obs=True, norm_reward=True) is active throughout, per this round's own added
support. gradient_steps=6 confirmed correctly auto-derived (matches n_envs, per
_resolve_gradient_steps).

Monitoring plan per instruction: check-ins at ~1h, ~6h, then roughly every 6h until
completion, each covering process-alive/RSS/VRAM/dec-s/ep_len_mean/eval-IS-vs-TWAP-
passthrough. Hard stop-and-report triggers: RSS climbing steadily (not plateauing --
the buffer's own footprint is a fixed ~174MB regardless of how full it is, confirmed
directly from real checkpoint file sizes last round, so sustained growth would mean a
leak, not normal filling), a worker dying silently (the SubprocVecEnv/multiprocessing.Pool
hang failure mode from the numeric-conversion round's own diagnosis -- looks alive, isn't
making progress), eval IS diverging badly or NaN-ing, or checkpoints missing their
expected cadence. Explicitly NOT a stop condition: an unpromising eval trend on its own --
SAC is expected to look bad early, and per docs/reports/l3_frozen_handoff.md's own honest
performance statement, the frozen L3 checkpoint itself only ties TWAP (does not beat it),
so L2's own early numbers should be read against that same honest baseline, not an
inflated expectation. Final n=500 evaluation against the pre-registered bar (beat
frozen-L3-alone at 0.994, ideally beat TWAP at 0.889, both paired tests agreeing) is
explicitly a SEPARATE round per instruction -- not evaluated here, not evaluated at
completion either, only reported.


ITEM 1 (VecNormalize -- a real decision, not inherited from the wiring round's scope
boundary): sampled 40 real episodes (real frozen L3, real numeric-format data, random
actions) and found genuine evidence for adding it, not just theoretical concern: several
L2 obs dims have non-zero empirical means (time_remaining_norm 0.64, schedule_deviation
0.20, own_open_orders_norm 0.23, ticks_since_own_fill_norm 0.21 -- none settle near 0 the
way an already-centered feature would) and empirical std heterogeneous across dims whose
declared _OBS_SPEC clip ranges already differ 5x (0 for structurally-zero/L1-stub dims,
up to ~0.96 for some book_depth_norm_i). Matches L3's own in-project precedent of
normalizing despite already-range-bounded inputs. Added VecNormalize(norm_obs=True,
norm_reward=True, clip_obs=5.0, gamma=L2_GAMMA) around the production vec_env only --
make_l2_env's own single, non-vectorized construction (this test file's fast mechanics
tests) stays deliberately unnormalized, and its own canary test's comment now says why
that's still correct rather than stale.

Structural knock-on effects, all implemented: resolve_l2_final_save_paths is now
pair-returning (model, vecnormalize) like train_l3.py's own version (was single-path);
new --resume-vecnormalize, REQUIRED alongside --resume-from (unlike
--resume-replay-buffer, which stays optional -- VecNormalize's running stats are real
model state, not optional bookkeeping, matching train_l3.py's own asymmetry between its
required --resume-vecnormalize and this project's own optional replay-buffer resume);
CheckpointCallback gets save_vecnormalize=True on real runs.

Verified, not assumed, per instruction -- three separate checks: (a) new unit test
(test_l2_policy_action_applies_vecnormalize_when_present) builds a real VecNormalize-
wrapped vec env, confirms get_vec_normalize_env() resolves non-None, and confirms
normalize_obs() actually transforms the observation ValISEvalCallback feeds to predict()
(not silently a no-op -- the previously-dead branch in _l2_policy_action is dead no
longer). (b) A live kill-and-resume test with all three --resume-from/
--resume-replay-buffer/--resume-vecnormalize flags together: killed a real run after a
checkpoint landed, resumed, got 5 clean post-resume eval firings and zero errors across
the rest of a full run to its own final save (both model.zip and vecnormalize.pkl written
correctly) -- one process mistake made and caught during this: the resume test's own
completion poll used a wrong grep pattern (looked for step=800/1000/1200, but a resumed
run's eval callback resets its OWN _last_eval_step counter to 0, so it actually fired at
step=604 relative to the resumed num_timesteps=600 baseline -- never matched), so the
disposable test run finished naturally and its final save landed on the ACTUAL canonical
paths (models/l2_strategist_v1.zip, l2_vecnormalize.pkl, since neither existed yet).
Caught immediately by checking the log directly rather than trusting the stalled poll;
both files deleted right away -- canonical L2 checkpoint still does not exist, confirmed
after cleanup. (c) A separate ~13-minute confirmatory shakedown (run_name=vnshakedown1,
n_envs=4, production defaults, VecNormalize now active) showed RSS (17.6-18.6GB
throughout) and the first eval firing (step=10364) landing in the same range as the
original pre-VecNormalize shakedown's own numbers -- no material behavior change from
adding VecNormalize, confirmed rather than assumed negligible. All disposable checkpoint/
test artifacts from every test this round deleted afterward (own scratch output).

ITEM 2 (resume seeding inconsistency -- a real bug, not a doc gap): make_l2_subproc_env's
workers were constructed with args.seed BEFORE the --resume-from branch ran, so a
resumed run's workers always seeded at --seed's own value/42 default even when the SAC
model itself correctly reseeded at the ORIGINAL run's model.seed via
model.set_random_seed(model.seed) -- --seed's own help text claimed workers reused
model.seed on resume, but the code didn't actually do that. Fixed by reordering: on
--resume-from, the checkpoint now loads (env=None) to read model.seed BEFORE vec_env is
constructed, and that value threads into every worker's torch.manual_seed(seed+rank),
making the documented behavior actually true. The startup seed print also moved to after
this resolution and now states explicitly when a resume is using model.seed instead of
--seed's own value (verified directly in the resume test above: printed
"seed=42 (from resumed model.seed, not --seed=42 -- see --seed's own help)" correctly).

ITEM 3 (disk headroom): checked directly (df -h): 214GB available. Checkpointing at the
default cadence (--checkpoint-freq-timesteps 50,000) over the full 2,000,000-step run is
40 firings x (~174MB replay buffer + ~3.5MB model + ~3.6KB vecnormalize) ~= ~7GB
accumulated, not pruned as it goes -- comfortably inside headroom (~3.3% of available),
not remotely tight. No retention logic added (the review's own conditional -- "if it's
tight" -- didn't hold); --checkpoint-freq-timesteps's own CLI help now states this
arithmetic and the checked number directly rather than leaving it an unstated assumption.

ITEM 4 (wall-clock and eval budget, using the ALREADY-established REAL throughput, not
the harness's uncorrected number): at n_envs=4 with the correct gradient_steps=4 (~17
dec/s, isolated cleanly last round via a controlled --gradient-steps 1 vs 4 comparison --
NOT the harness's own uncorrected 23.847 dec/s, which was never run with this round's own
UTD-preserving fix), 2,000,000 steps ~= 32.7 hours ~= 1.36 days. n_envs=8 was not
independently re-measured with its own correctly-scaled gradient_steps=8 this round
either -- if the same proportional slowdown applies (17/23.847 ~= 71%), an EXTRAPOLATED,
not measured, ~24.5 hours ~= 1.02 days. --eval-freq's default (10,000) gives 200 firings
over the full run at ~35-40s each (measured, both the original and confirmatory
shakedowns) ~= ~2.1 hours total, ~6.4% of the 32.7h run -- reported as a real,
non-trivial cost, not assumed negligible from the single-env-era design comment's
outdated "under 2%" estimate.

Recommendation restated with these four items resolved: n_envs=4, total_timesteps=2,000,000,
expected wall-clock ~32.7 hours (not ~18h, not the harness's ~23.3h) -- awaiting the
user's own review and go-ahead before that launch, per instruction.


TASK 1 (vectorize train_l2.py): ported the pattern scripts/benchmark_controlled_numeric.py
measured (SubprocVecEnv, per-worker CPU-only frozen-L3 inference, mandatory thread-capping)
into the actual production script -- every n_envs number to date came from that throwaway
harness, never train_l2.py itself, which was still single-env before this round (confirmed
by reading it directly, not assumed). New make_l2_subproc_env() worker factory: each
SubprocVecEnv worker constructs its OWN LOBExecutionEnv + FrozenL3Wrapper + RecurrentPPO
instance inside its own _init() (post-fork), loads the frozen L3 checkpoint on CPU
regardless of --device (only the SAC policy itself trains on --device, typically cuda),
and sets OMP_NUM_THREADS/MKL_NUM_THREADS=1 (module-level, before any other import) +
torch.set_num_threads(1) per worker -- this project's own prior round measured an
un-thread-capped attempt at this same pattern at 7-9x SLOWER from oversubscription, so
this is treated as mandatory, not an optimization. use_numeric_format threaded through
make_l2_wrapped_env/ValISEvalCallback as a new trailing, defaulted kwarg (every existing
positional call site in tests/test_train_l2.py is unaffected); new --use-numeric-format
CLI flag defaults ON (the numeric-format archive, converted+equivalence-verified last
round, is now this project's production input), with --data-dir resolved from it
explicitly and printed at run start, per the round's own hard boundary that this choice
must never be silently inherited. --n-envs added, default 4 (rationale under Task 4
below). make_l2_env (single, non-vectorized, Monitor-wrapped) is UNCHANGED and still used
by tests/test_train_l2.py's existing fast mechanics tests -- the new vectorized path is
fully separate, so none of the pre-existing single-env test coverage needed to change.

TASK 2 (five correctness risks, all tested empirically this round, not just reasoned
about):
1. Seed reproducibility under SubprocVecEnv -- needed an EXTRA fix SB3 does not provide.
   SAC(seed=...) -> BaseAlgorithm.set_random_seed() seeds the MAIN process's own
   python/numpy/torch RNGs and calls env.seed(seed) on the VecEnv (SubprocVecEnv resolves
   this to per-worker env.reset(seed=seed+idx) on the next reset(), confirmed against the
   installed SB3 2.3.2 source -- same mechanism train_l3.py's own module docstring already
   documented). This does NOT reach a SubprocVecEnv worker's own separate process's torch
   RNG, which is exactly what the frozen L3's training-time predict(deterministic=False)
   samples from -- left unseeded, two runs with an identical --seed would still diverge
   through the frozen L3's own stochastic action choices. Fixed: torch.manual_seed(seed +
   rank) inside each worker's _init(), same seed+idx offset convention SB3 itself uses.
   Verified by running the real script twice with an identical --seed (--n-envs 4,
   --smoke-test, --eval-freq 100): grepped rollout/eval/loss metrics from both runs --
   byte-for-byte IDENTICAL output across the entire run, not approximately close.
2+3+4. Per-worker LSTM state isolation, the cross-episode leak fix (wrappers.py's
   FrozenL3Wrapper.reset(), which zeroes l2_target_slice_ratio_override/l2_urgency before
   calling env.reset()) holding under SubprocVecEnv's own auto-reset machinery, and exact
   matched-seed equivalence vs. a solo single-env run -- verified together via a dedicated
   scratch script (not committed): built a real n_envs=4 SubprocVecEnv seeded at
   BASE_SEED=999 using make_l2_subproc_env directly, stepped a fixed, pre-generated
   30-step action sequence, and captured each worker's full obs/reward/done trajectory.
   Compared each worker i's trajectory against a SOLO SubprocVecEnv(n_envs=1) seeded
   directly at BASE_SEED+i (matching SB3's own per-worker seed-offset convention) given
   the IDENTICAL action sequence for that worker slot. Stated criterion up front: exact
   byte-identical (np.array_equal) equivalence, not merely distributional -- achievable
   here since every component (env dynamics, RNG, frozen L3 sampling once seeded per
   point 1) is deterministic given a seed. Result: all 4/4 workers PASS, byte-identical
   obs/reward/done sequences, INCLUDING across an auto-reset episode boundary each of the
   4 workers hit naturally during the 30-step window (auto-reset correctly invoked
   FrozenL3Wrapper.reset(), not some inner unwrapped env, and no cross-worker state bled
   between the 4 separate OS processes). Any LSTM-isolation or leak-fix failure would have
   perturbed a worker's post-boundary trajectory away from its solo-run counterpart --
   none did.
5. SAC's train_freq=1/gradient_steps semantics under n_envs>1 -- confirmed against the
   installed SB3 2.3.2 source (off_policy_algorithm.py): train_freq=(1,"step")'s
   `num_collected_steps` counter increments once per env.step() CALL regardless of
   env.num_envs, so it always triggers exactly ONE training() call per env.step() call --
   meaning gradient_steps (fixed at 1, Section 4.1's literal single-env-only reference
   value, never re-derived for parallel workers) would silently cut the update-to-data
   ratio to 1/n_envs once n_envs>1, a real, silent change to SAC's sample efficiency
   nobody had deliberately chosen. Fixed: new _resolve_gradient_steps(n_envs, override)
   (pure function, unit-tested), defaults to n_envs, preserving the original
   1-gradient-step-per-transition ratio the reference value gave at n_envs=1. Also
   reapplied explicitly on --resume-from (model.gradient_steps = ... after SAC.load()) --
   caught in review before it shipped: SAC.load() restores gradient_steps from the
   ORIGINAL run's own pickled hyperparams, so resuming with a DIFFERENT --n-envs than the
   original run used would otherwise silently keep the stale, mismatched value.
   buffer_size=500,000 needed NO fix -- confirmed against SB3's ReplayBuffer source that
   this is a TOTAL transition cap, divided by n_envs internally
   (self.buffer_size = max(buffer_size // n_envs, 1)), so its real memory footprint is
   independent of n_envs by construction (see Task 4's own ~174MB measurement below).

TASK 3 (multi-day hardening):
- Save-path safety: new --run-name/--overwrite-canonical + resolve_l2_final_save_paths()
  (L2 analog of train_l3.py's own resolve_final_save_paths() guard and its
  docs/reports/l3_replace_value_probe.md incident -- single-path here since L2 has no
  VecNormalize to pair). A run's final save can no longer silently overwrite
  models/l2_strategist_v1.zip (still doesn't exist -- confirmed before AND after this
  round's testing). Periodic CheckpointCallback's name_prefix is now run-name-tagged too
  (l2_sac_<run-name>_*, was a fixed, collision-prone "l2_sac" before), matching
  train_l3.py's own precedent for why this matters (two real runs' intermediate
  checkpoints silently overwrote each other there once). --smoke-test saves are
  deliberately EXEMPT from both (already fixed, clearly-namespaced, no collision risk,
  genuinely low-stakes).
- New --checkpoint-freq-timesteps (default 50,000, n_envs-divided per SB3's own
  CheckpointCallback documentation), with save_replay_buffer=True on real (non-smoke)
  runs.
- --resume-from/--resume-replay-buffer: NOT just implemented -- actually tested by
  launching a real (non-smoke) run, waiting for a checkpoint to land (checkpoint_freq_
  timesteps=200 for this test), then SIGKILL-ing the process (not a graceful stop, a real
  crash simulation) once past a second checkpoint at 400 steps. Verified clean death (no
  leaked processes, memory/GPU fully released). Resumed from the 400-step checkpoint +
  its paired replay buffer: log confirmed "resumed replay buffer ... (99 transitions)"
  and "resumed from ...: loaded num_timesteps=400" -- both consistent with the pre-crash
  state (400 timesteps / 4 envs = 100 vec-steps, ~99 stored). Training continued correctly
  past the resume point (total_timesteps 400 -> 636+, n_updates climbing normally, losses
  evolving continuously not restarting), and the eval callback fired correctly at the
  resumed state (step=404, right after resume). Killed again (SIGKILL) before it could
  reach its own final save -- deliberately, since no canonical L2 checkpoint exists yet
  and this was a disposable test, not a real run. Test checkpoint files (killtest1,
  killtest1_resumed -- ~866MB total, mostly replay-buffer pickles) deleted afterward, this
  session's own scratch output.
- Survives SSH disconnect: every launch this round used nohup + a detached background
  shell (established session pattern), verified via a completely fresh, independent SSH
  connection after each launch per this session's own standing discipline -- the local
  Bash tool's own "launch" command hung on the known SSH/nohup stdio quirk on EVERY
  launch this round, never treated as a signal of remote state either way.
- ValISEvalCallback: confirmed wired in and active by default (--eval defaults True,
  unchanged), firing correctly under the vectorized path -- see Task 4's shakedown for
  real firing-cost measurements at production defaults (eval_freq=10,000,
  n_eval_episodes=10).

TASK 4 (shakedown, ~53 real minutes, --n-envs 4, real numeric-format data, full
production defaults -- eval on, checkpoint_freq_timesteps=50,000, run_name=shakedown1):
- RSS (summed VmRSS across the main process + all 4 workers): 18.34GB at launch -> 18.65GB
  after 50,736 timesteps / 6 eval firings / 1 checkpoint (~1.6% drift over the whole run)
  -- consistent with buffer_size being pre-allocated up front, not something that grows as
  it fills: every replay-buffer checkpoint .pkl this round (killtest1's, killtest1_
  resumed's, AND shakedown1's, at totally different num_timesteps) was exactly 174,002,069
  or 174,002,070 bytes, confirming the ~174MB footprint predicted from the SB3 buffer-math
  fix above is real, fixed, and independent of how full the buffer actually is. No leak.
- 4/4 workers stayed alive and stable the entire run, no crashes, no stalls.
- 6 eval firings (steps 10k/20k/30k/40k/50k+), each costing ~35-40s wall-clock (~6%
  overhead at this run's own throughput -- higher than the single-env design comment's
  "under 2%" estimate, since eval's fixed per-firing cost stayed the same while --n-envs
  raised how often it fires per unit wall-clock; still a small, acceptable overhead, not a
  problem, just a fact worth recording plainly rather than the older, no-longer-accurate
  single-env estimate).
- Checkpoint landed correctly at the production-default 50,000-step cadence with correct
  run-tagged naming and a valid paired replay-buffer file.
- All test artifacts from this shakedown deleted afterward (this session's own scratch
  output, same as the kill-resume test's).

Measured throughput vs. the harness, isolated cleanly: the shakedown's own steady-state
rate (~17 dec/s, delta-based, past the one-time eval-baseline startup cost) is materially
below the numeric-format harness's 31.808/23.847 dec/s (n_envs=8/4). Root-caused, not left
unexplained: a controlled --gradient-steps 1 vs --gradient-steps 4 comparison, both
--no-eval, both --n-envs 4 (apples-to-apples against the harness, which never had eval or
the gradient_steps fix): gradient_steps=1 (the harness's implicit, uncorrected value)
steady-states at ~24-25 dec/s, matching the harness's 23.847 dec/s closely; gradient_steps=4
(this round's correctness fix, Task 2 item 5) steady-states at ~17-18 dec/s, matching the
real shakedown almost exactly. The entire gap is the deliberate UTD-preserving fix itself
-- 4x more GPU gradient updates per collected batch than the harness ever measured, not a
bug, not subprocess or eval overhead, not an artifact of the real data vs. the harness's
own data pool.

This changes the 2,000,000-step run estimate materially: at n_envs=4's real, correctness-
fixed throughput (~17 dec/s), 2,000,000 / 17 ~= 32.7 hours ~= 1.36 days -- still better
than the original 1.84-day parquet baseline, but well above both the harness's own
uncorrected 0.73-day estimate AND the "~18 hours minimum" figure the round's own framing
used, since neither of those ever included this fix. Reported plainly, not smoothed over
with the faster, uncorrected number. n_envs=8 was not independently re-measured this round
(the round's own "then stop" boundary, see Task 5) -- the RSS-headroom evidence gathered
(production n_envs=4 needed ~18.6GB including the eval apparatus the harness never had,
comfortably under this box's 48GB available) suggests n_envs=8 remains viable on paper,
but its own gradient_steps=8 throughput was not directly measured, so no numeric claim is
made for it here.

TASK 5 (report): delivered directly to the user, not duplicated here in full -- covers
implementation state, all five correctness results, hardening verification (including the
actual kill-and-resume test), shakedown numbers, the isolated throughput-gap finding, and
a recommended n_envs=4 / total_timesteps=2,000,000 for the real run, pending the user's own
review before launch (explicitly not started this round, per instruction).

Files touched: src/train/train_l2.py (rewritten in place, vectorized + hardened),
tests/test_train_l2.py (extended: new tests for use_numeric_format threading,
_resolve_gradient_steps, resolve_l2_final_save_paths, and the new --n-envs/--seed/
--gradient-steps/--resume-replay-buffer-requires-resume-from CLI surface -- all 18 tests,
including every pre-existing one, still pass). Both committed as 242df76. Did NOT touch
src/envs/wrappers.py or tests/test_wrappers.py (read only, to understand FrozenL3Wrapper's
exact behavior before relying on it) -- no fix needed there this round, everything
required was achievable from train_l2.py's own side. src/envs/lob_execution_env.py
remains uncommitted with an unrelated in-progress diff (use_numeric_format support this
round's own work depends on, plus whatever L3's own current staleness/eta_replace round
has added) -- read to confirm the use_numeric_format constructor param's exact current
behavior, not modified or staged, consistent with this file being another track's
in-flight work.

No git push (commit locally only, per standing instruction). No protected files edited.


TASK 1 (diagnosis): the numeric-format conversion (scripts/convert_l2_to_numeric_parallel.py,
382/441 done) was HUNG, not slow -- 6 workers idle at 99.3% CPU, no new output in 80+ min.
py-spy dumped the parent and all 6 live workers: parent blocked in next() waiting for a
result, every worker idle in queue.get() waiting for a task that will never come. dmesg
confirmed root cause: the kernel OOM-killed 6 python workers within the first ~3.5 minutes
of the original N_WORKERS=8 run (6.1-7.8GB RSS each at time of death). multiprocessing.Pool
respawns killed workers automatically (why 6 workers still looked "running" -- they were
replacements) but never retries or surfaces an exception for a task whose worker was
SIGKILLed -- the result is silently lost forever, and next() blocks indefinitely once all
other dispatchable work is exhausted. Checked write_day() before considering any restart:
writes atomically (temp file + Path.replace()), so no corrupted output existed for the lost
file(s). The 59 still-unconverted files skewed toward the largest in the dataset (including
the top 2 by size) -- large files driving memory pressure, not one malformed input.

TASK 2 (equivalence gate on the 382 already-converted, run BEFORE converting further):
scripts/compare_formats_equivalence.py existed but had never been run -- no log, no
result, anywhere. Ran it: 10/10 seeds PASS on its own hardcoded pool. Caught before trusting
that alone: that pool is the pre-existing 10-day "benchmark pool" converted by a SEPARATE,
earlier process (mtimes 16:29-16:38, before the mass-conversion run even started at 17:12)
-- passing it does NOT touch anything the stalled parallel script actually produced. Built
an extended check targeting 8 individually-verified mass-converted files directly (the 3
largest, chronological first/last, 3 mid-range), date_range=(day,day) to pin each check to
one specific file: 8/8 PASS, 10 seeds each. 18 days x 10 seeds = 180 comparisons, all
byte-identical (np.array_equal, not np.allclose).

TASK 3 (finish conversion, only after 1-2 cleared): killed the hung process (clean SIGTERM,
cascaded correctly, 48GB memory freed). Fix: N_WORKERS 8 -> 4 (rationale recorded inline in
the script). Resumed (idempotent, skip-already-converted) -- completed all 441 files in 7.8
minutes, no further OOM, memory finished clean. Then re-ran equivalence across the FULL set
per instruction, with 100% coverage (not a sample) of the 59 files that were in flight
during the crash, since those are the most likely to be anomalous: 59/59 PASS, 10 seeds
each = 590 comparisons, 27.3 min, all byte-identical. Combined across all three equivalence
rounds this session: 77 distinct days x 10 seeds = 770 comparisons, 100% pass. Final output:
12.98GB (data/raw_l2_bybit_numeric/BTCUSDT/, 441 .npzst files).

TASK 4 (re-benchmark, numeric format, same controlled method as the parquet baseline --
fixed seed=42, same fixed 10-day pool, 3 trials/config, thread-capping,
scripts/benchmark_controlled_numeric.py, NOT wired into train_l2.py, same throwaway-
harness status as the original benchmark_controlled.py): n_envs=8 gives 31.808 dec/sec vs
the parquet baseline's 12.575 dec/sec -- a 2.53x speedup, very tight noise (CoV 0.1-0.9%
across the whole sweep). 2,000,000-step extrapolation: 0.73 days, down from 1.84 days.
This crosses the project's own stated go/no-go threshold from MARGINAL (1-2 days) into
WORKABLE (under ~1 day). Full sweep: n_envs=1 10.016/s, n_envs=2 15.491/s, n_envs=4
23.847/s, n_envs=8 31.808/s (efficiency 100%/77.3%/59.5%/39.7%, same diminishing-returns
shape as the parquet sweep, just uniformly faster).

Committed: scripts/convert_l2_to_numeric_parallel.py (with the N_WORKERS fix baked in --
was never committed by anyone before this, so its whole history starts here) and
scripts/benchmark_controlled_numeric.py, one commit (a2e51f4). Did NOT stage/commit any of
L2's other pre-existing untracked scratch files from this round (check_level_counts.py,
compare_formats_equivalence.py, convert_l2_to_numeric.py, convert_one_day*.py,
test_zstd_raw.py, src/data/l2_numeric_format.py, tests/test_numeric_format_equivalence.py)
-- those are L2's own work to commit when ready, not mine to sweep in.

IMPORTANT, explicitly noted per instruction and NOT done this round: train_l2.py is still
single-env, unchanged -- confirmed by reading it directly, not assumed. Every n_envs number
above (both parquet and numeric) comes from throwaway benchmark harnesses, never the
production training script. Building vectorized training (SubprocVecEnv + CPU inference
per worker + thread-capping, wired into train_l2.py itself) is next round's work, not
started here.

PRIOR ENTRY BELOW, L2's own last regular round (env.reset() I/O optimization, predates the
numeric-format work entirely):

Last updated: 2026-08-24 16:10 HKT
State: env.reset() I/O round (round 2 of 2 on this cost, last one before training per
instruction). Attacked _load_day's ~1400-1560ms/miss cost directly. Checked parquet file
layout first, not assumed: every day file is ONE row group (863,997 rows, 104.7MB) --
this rules out row-group pushdown entirely (no API-level way to decode a sub-range within
a single row group). Tested predicate/page-index pushdown directly: a ts-range filter for
a ~3,600-row window was SLOWER (0.84x) than a full read, not faster -- confirmed via
metadata that this pyarrow build doesn't support page indexes. Both angles are real,
measured dead ends, not unexplored.
Column pruning (symbol/update_id/seq, confirmed unused via grep) was the one real lever:
measured the split directly -- 5 needed numeric/ts columns decode in ~48ms, bids+asks
(JSON-string book levels) alone are ~97% of decode cost (~1500-1570ms). Pruning the unused
columns therefore only saved 0-4% across 5 real days (one day came in slightly slower) --
small, but free and zero-risk, so implemented anyway. Dtype downcasting: not applicable to
the actual bottleneck (bids/asks are strings, not a numeric column to downcast; the
columns that could be downcast are already the negligible ~48ms slice) -- not implemented,
no plausible upside to weigh against the equivalence risk.
Lazy TickView construction (flagged last round, evaluated not implemented this round):
with last round's vectorization already gutting _precompute_feature_series, the remaining
ceiling here is bounded at ~6% of reset() (the unused-horizon portion only -- the buffer
portion is needed unconditionally). Not worth the refactor risk in the round explicitly
framed as the last one before training.
Verified seed-equivalence (same 10-fixed-seed method as last round, reused exactly,
including the unseeded cache-hit path): byte-identical (np.array_equal) before/after the
column-pruning edit. Full suite: 158/162 passing, same 4 pre-existing unrelated failures.
Re-ran the controlled n_envs benchmark unmodified: n_envs=8 12.163 -> 12.575 dec/sec
(+3.4%). 2,000,000-step extrapolation: 1.90 -> 1.84 days. Verdict unchanged from last
round's bucket: MARGINAL (same band, not "workable," not "needs rethinking"). Attempted a
direct realistic-cache-rate measurement (full 405-day pool instead of estimating) and it
is reported but explicitly flagged unreliable: a single-trial reading came in FASTER than
the narrow pool, the wrong direction -- traced to a real confound (the much larger file
pool changes the RNG draw sequence entirely at the same seed, so episode-length varies
between pools for reasons unrelated to cache rate, the same class of confound this project
already learned to control for two rounds ago). The trustworthy source for the cache-rate
direction is the reset()-level profiling itself (n=40/scenario): real-pool reset() cost
(1573ms) vs. narrow-pool blended (921ms) confirms the expected direction cleanly, without
that confound.
Bottom line, stated plainly per instruction: parquet reads are irreducibly expensive for
this access pattern given the current JSON-string storage format for bids/asks -- a real
wall, not a gap closed by more engineering within this round's scope. Combined across both
optimization rounds: 9.729 -> 12.575 dec/s (+29.3%) at n_envs=8, 2.38 -> 1.84 days for
2,000,000 steps. Per instruction, proceeding to training regardless of the marginal
verdict -- this is the last optimization round.
Full detail: docs/reports/phase4_l2_reconciliation_and_plan.md's new "env.reset() I/O
round" section.
Files: src/envs/lob_execution_env.py (edited further, column pruning only -- same
in-scope basis as last round). scripts/measure_column_pruning.py,
scripts/measure_column_split.py, scripts/measure_predicate_pushdown.py,
scripts/benchmark_realistic_pool.py (new, throwaway investigation/measurement code).
Committed this round as one commit (edit + scripts + doc updates).
Files owned/in-progress: none uncommitted as of this update.
Blocking/open questions: throughput engineering on this approach is now closed, both
rounds' findings converge on the same MARGINAL verdict. Open for whoever owns training
launch: proceed at n_envs=8 (recommended, best measured), decide on decision (a)
(VecNormalize on L2's own obs/reward, still recommended not blocking) before launch.
Next planned step: per instruction, move to the real training launch -- awaiting explicit
go-ahead for that separate decision (a full training run remains outside this round's own
scope, which was measurement/optimization only).

PRIOR ENTRY BELOW, for context on the first reset() optimization round (vectorization)
this round built on:

Last updated: 2026-08-24 12:40 HKT
State: env.reset() optimization round -- Task 0 first (docs/reports/v1_master_state.md,
committed on its own, a full cross-track "as of now" snapshot frozen before any code
change). Then profiled reset() itself (Task 1): at real 405-day training scale, the
day-cache hit rate is only 2.4% (cache holds 5/405 days) -- essentially every reset()
during real training pays the full ~1.4s I/O miss cost; the prior benchmark's own 10-day
pool sees 48.8% hits, an artifact of that pool's narrowness, not representative. cProfile
pinpointed ~77% of _precompute_feature_series's own cost (34.8% of total reset()) as an
unvectorized python loop calling TickView.qty_at_price/np.isclose once per tick (~14,400
calls/reset).
Vectorized it (Task 2/4): _rolling_sum/_rolling_rms/_rolling_mean_std and the
touch-depletion loop now use numpy fancy-indexing instead of python range(n) loops -- same
arithmetic, no RNG/windowing touched. _precompute_feature_series dropped from ~498ms to
~19ms (~25x). Explicitly did NOT implement two candidates the task asked about: per-day
feature caching and larger _MAX_CACHED_DAYS both fail at real 405-day scale (a whole-day
precompute costs ~240x more per call than per-episode, and cache hit rate scales roughly
with cache_size/pool_size -- doubling the cache only lifts hit rate from ~2.4% to ~2.5% on
a 405-file pool) -- stated as real negative findings, not hedged.
Verified seed-equivalence (Task 3), not assumed: env.reset()/env.step() traces (obs at
every tick, rewards, terminal IS) for 10 fixed real seeds, run before and after the edit,
compared via exact np.array_equal (not np.allclose) -- byte-identical across all seeds and
both the initial seeded reset AND a subsequent unseeded reset (the day-cache-hit path).
Plus 17 new permanent hand-computed-fixture regression tests
(tests/test_reset_vectorization_equivalence.py) and the full existing suite (158/162
passing, same 4 pre-existing unrelated failures as always -- test_bulk_backfill.py
network-mock issues, test_l2_capture.py resync logic, confirmed unrelated).
Re-ran the controlled n_envs benchmark unmodified (Task 4): real, reproducible gain at
every n_envs value (+10.1% at n_envs=1 up to +25.0% at n_envs=8, CoV stayed 0.1-2.0%, not
noise). n_envs=8: 9.729 -> 12.163 dec/sec. Re-extrapolated 2,000,000-step wall-clock:
2.38 -> 1.90 days at the best configuration. Go/no-go moves from NO to MARGINAL against
the same stated guidance (under ~1 day workable, 1-2 days marginal, beyond that rethink) --
a real improvement, not a clean yes. The prior round's two representativeness caveats
still apply and now have a concrete number behind one of them (this benchmark's 10-day
pool's 48.8% cache hit rate vs. the real pool's 2.4%) -- both still lean toward the true
number being worse than 1.90 days, not better. Full detail:
docs/reports/phase4_l2_reconciliation_and_plan.md's new "env.reset() optimization round"
section.
Files: src/envs/lob_execution_env.py (edited, L3's file -- in-scope this round per
instruction, L3's research is closed; reward.py/train_l3.py/L1's files not touched --
short flag note left in L3's own section below for that track's awareness).
scripts/profile_reset.py, scripts/profile_reset_cprofile.py,
scripts/capture_reset_snapshot.py, scripts/compare_reset_snapshots.py (new, throwaway
measurement/verification code). tests/test_reset_vectorization_equivalence.py (new,
permanent). docs/reports/v1_master_state.md (new). Committed this round: Task 0
separately, then Tasks 1-4 (the env.py edit, new scripts/tests, doc updates) as one
follow-up commit.
Files owned/in-progress: none uncommitted as of this update.
Blocking/open questions: throughput is no longer a clean NO, but not a clean YES either --
1.90 days at n_envs=8 is inside the marginal band, with caveats leaning pessimistic. Open
for whoever owns the go/no-go: proceed at n_envs=8 accepting marginal economics, pursue
the flagged (not implemented) lazy-tick-construction lever for a further gain, a smaller
training budget, or reconsider given the checkpoint's own honest tie-not-beat result
against TWAP.
Next planned step: awaiting direction. No production parallelization code, no real
training launch, until one of the above is decided.

PRIOR ENTRY BELOW, for context on the controlled parallelization benchmark this round
built on:

Last updated: 2026-08-24 09:07 HKT
State: Controlled follow-up benchmark to last round's noisy parallelization result
(0.14x/0.11x uncapped, then two disagreeing thread-capped readings of the SAME n_envs=2
config, 3.38 vs 8.53 dec/sec -- root cause: unfixed seed + unfixed date_range). This round
fixed both: every one of 12 runs (n_envs=1/2/4/8 x 3 trials) sampled the SAME fixed 10-day
pool from the real train split (first 10 gap-free dates) with the SAME fixed seed (42).
Result: coefficient of variation dropped to 0.1-1.3% across every configuration (from >2x
swings) -- the fix worked, confirmed by measurement, not assumed.
Sweep result (thread-capped throughout, mandatory per last round): n_envs=1: 5.651 dec/sec
(NOTE -- already higher than the original 4.194 dec/sec GPU baseline from two rounds ago,
since this config also uses CPU L3 inference, this round's recommended design, not GPU --
that switch alone is a ~35% gain before any parallelism). n_envs=2: 6.865 (1.21x speedup,
60.7% efficiency). n_envs=4: 8.852 (1.57x, 39.2% efficiency) -- flagged explicitly per
instruction: reaches 91% of n_envs=8's raw throughput using half the RAM/VRAM, a real knee
in the curve worth knowing if resource contention with L1 becomes binding. n_envs=8: 9.729
(1.72x, 21.5% efficiency) -- real, reproducible speedup, but clearly sub-linear at every
step, not close to ideal scaling anywhere tested.
Real RAM/VRAM measured (not assumed) via /proc/<pid>/status + nvidia-smi per-process
accounting: n_envs=8 uses 26.2GB RSS (NOT the same as L3's own n_envs=8 PPO budget of
~38.8GB -- confirmed different, not assumed to match) and 3.6GB VRAM (fits comfortably
alongside L1's stated ~15GB peak). One anomaly flagged, not fixed: VRAM scales ~linearly
with n_envs (matching ~454MB/worker, this round's own earlier GPU-inference measurement)
even though these workers run L3 inference on CPU -- suggests CPU-inference workers are
still initializing an unneeded CUDA context, worth investigating in a full implementation
to reclaim that VRAM.
Extrapolation to a real 2,000,000-step run, from the controlled rates directly:
n_envs=8 -> 205,571s = 57.1h = 2.38 days (best measured). n_envs=4 -> 2.62 days. n_envs=2
-> 3.37 days. n_envs=1 -> 4.10 days. Go/no-go per the stated guidance (under ~1 day
workable, 1-2 days marginal, beyond that rethink): NO, plainly, even at the best
configuration -- 2.38 days is outside the marginal band. Extrapolation's own caveats
(benchmark's narrow, repeated 10-day date_range likely benefits from warm OS page cache a
real 405-day-diversity run wouldn't get; policy behavior drift as L2 actually learns is
untested by a short benchmark) both lean toward this being an OPTIMISTIC reading, not
pessimistic -- real wall-clock is more likely to be worse than 2.38 days than better.
Stated plainly per instruction, not stretched to clear the bar: parallelizing envs alone,
at the scale tested, does not make a full 2M-step L2 run practical.
Five correctness risks from last round still stand (seed reproducibility per-worker now
partially validated by this round's low CoV, but a dedicated different-workers-draw-
different-episodes test is still not built; per-worker LSTM isolation; the cross-episode-
leak fix holding inside SubprocVecEnv specifically; distributional equivalence via
matched-seed comparison; SAC train_freq/gradient_steps semantics under multi-env -- none
resolved by a throughput-only benchmark). Two new items added this round: unnecessary CUDA
context init in CPU workers (~454MB/worker wasted VRAM); this benchmark's controlled
date_range trading representativeness for low variance -- a full implementation should
re-validate against the real 405-day split before trusting these exact absolute numbers
for capacity planning (relative comparison across n_envs is trustworthy; absolute rate may
not transfer unchanged).
Separately this round: found and fixed a real .gitignore gap (models/*.zip and
models/*.pkl only matched files directly under models/, not nested subdirectories like
models/l2_checkpoints_smoke/ or the real models/l2_checkpoints/) -- discovered when
smoke-test binaries showed up staged by something/someone else on this shared box despite
looking gitignored. Also: a first commit attempt for that fix accidentally swept in four
unrelated files staged by another concurrent session in the window between this session's
status check and commit -- caught immediately via git show --stat HEAD, corrected with a
non-destructive git reset (no content lost), re-verified via git diff --cached before
re-committing. Noting this for the record since it's a real, live risk on this shared box,
not hypothetical -- verify staged content immediately before every commit, not just git
status shortly before.
Files: scripts/benchmark_controlled.py (new, throwaway measurement code, not wired into
train_l2.py) -- committed this round, along with the .gitignore fix and design doc
updates.
Files owned/in-progress: none uncommitted as of this update.
Blocking/open questions: throughput engineering via env-parallelization alone has now been
tested and answered -- not sufficient on its own to make a 2M-step run practical. Open
question for whoever owns the go/no-go, not resolved here: proceed with a smaller training
budget, pursue a fundamentally different approach to the reset()-dominated cost, or
reconsider given the checkpoint's own honest tie-not-beat result against TWAP.
Next planned step: awaiting direction. No production parallelization code, no real
training launch, until one of the above is decided.

## L3 / Env-Physics

**[Flag from the L2/shared-infra session, 2026-08-24 12:40 HKT, not an L3 entry --
L3's own status below is unmodified.]** This round's env.reset() optimization work
(see the L2 section above) edited `src/envs/lob_execution_env.py` -- explicitly
in-scope per this round's own instruction, since L3's research is closed. Vectorized
three rolling-window helpers and the touch-depletion loop inside
`_precompute_feature_series`; `reward.py`/`train_l3.py` were NOT touched. Verified
byte-identical behavior (10 fixed real seeds, before/after, exact `np.array_equal`)
plus 17 new permanent regression tests -- full detail in
`docs/reports/phase4_l2_reconciliation_and_plan.md`'s new "env.reset() optimization
round" section. Flagging here since this file already carried a separate,
pre-existing uncommitted change from L3's own prior session (the documented
functionally-inert staleness-round addition, eta_replace=0.0) -- that pre-existing
change is untouched by this edit and remains exactly as L3's own session left it.
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
