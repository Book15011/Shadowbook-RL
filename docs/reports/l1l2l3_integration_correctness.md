# L1->L2->L3 integration: correctness harness, not a performance run

**Date:** 2026-08-23
**Status:** Full three-tier orchestrator built, tested, and passing. This is
a CORRECTNESS result -- does data flow through all three tiers with correct
cadences -- not a performance or training result. No training was run.

## Step 0: prerequisites, checked directly rather than assumed

**(a) `src/data/l1_features.py` -- exists and works.** 15/15 tests pass.
Directly verified live against real `data/raw_l1` at an in-range timestamp
(2026-08-10): produces real, sensible numeric values for all 11 fields (e.g.
`return_1h_pct=0.0017`, `funding_rate_current=5.66e-05`,
`open_interest_level=107069.91`) -- not just `None` placeholders. Fed
directly through `L1MacroAnalyst.maybe_refresh()` without error. This
prerequisite is genuinely ready.

**(b) 14B vs 32B model decision -- NOT resolved, and worse than that.**
TRACK_STATUS's existing framing ("still open, 14B pulled, 32B not") is
outdated in a more serious way: the 14B model that WAS pulled is
**unusable by the actual running Ollama service.** The systemd service
(`/etc/systemd/system/ollama.service`) runs as system user `ollama`
(home `/usr/share/ollama`), whose own `~/.ollama/models/{blobs,manifests}`
are completely empty. The 8.4GB of real model data sits under
`/home/ubuntu/.ollama/models` instead -- pulled by a different (`ubuntu`)
user context, invisible to the service. This is a real infra bug, not just
an undecided model size -- confirmed live: `curl .../api/tags` (bypassing a
proxy env var that was silently intercepting the same request) returns
`{"models":[]}`.

**(c) Has a real Ollama call ever succeeded -- NO, checked directly.**
`curl .../api/generate` with the real model name returns
`{"error":"model 'qwen2.5:14b-instruct-q4_K_M' not found"}`, live-verified
this round. Combined with (b)'s empty service-visible manifests directory
(dated to the service's own start time, never populated since), no real
call has succeeded and none currently can, independent of the 14B/32B
choice.

**(d) L2's throughput measurement -- landed, TRACK_STATUS was stale on
this.** Commit `20ce3b6` (postdating the L2 section's own last TRACK_STATUS
update) measured ~4.15 L2-decisions/sec steady-state (single env, CPU,
extracted from smoke-test artifacts -- checkpoint mtimes, TensorBoard log,
console output, no new run). Extrapolated to a real 2,000,000-step SAC
run: ~5.5 days. ~47% of wall-clock is `env.reset()` overhead, not the
50-tick inner loop. Verdict already reached by L2's own track: **not
practical at this throughput**, single-env; recommends parallelizing
(not yet implemented).

**Decision, per instruction:** (a) is ready, (b)/(c) are not, and (b)/(c)'s
failure mode is now understood precisely rather than just "still pending."
Proceeding with L1 stubbed at the LLM boundary only -- the real, tested
feature pipeline is used nowhere in the integration test below (since real
Ollama can't be reached), but nothing about (a)'s readiness is wasted; it's
simply not exercised by this round's correctness harness.

## Step 1: the L1 wiring gap, closed with no L2-side change needed

`docs/reports/l3_frozen_handoff.md` flagged that `l1_risk_score`/
`l1_confidence` are correctly wired inside `LOBExecutionEnv` (idx 17/18,
plus `l1_risk_score` feeding `step_reward()`'s inventory term) but nothing
upstream writes them once `FrozenL3Wrapper` sits in the loop.

**The clean answer, confirmed by reading both files directly:**
`FrozenL3Wrapper` is a `gym.Wrapper` subclass, which exposes the raw env it
wraps via the standard, public `.env` attribute -- and the wrapper's own
`step()` already uses exactly this pattern internally for L2's idx 15/16
(`self.env.l2_target_slice_ratio_override = ...`,
`src/envs/wrappers.py`). So the orchestrator simply calls
`macro_tick(l3_wrapper.env, l1_agent, tick, feature_summary)` -- the SAME
`macro_tick()` function, completely unmodified, now pointed at the wrapped
env's own `.env` attribute instead of a bare env. **No change was needed
inside `wrappers.py`.** This was verified, not assumed: read
`FrozenL3Wrapper.__init__`/`step()`/`reset()` in full before concluding
this, including confirming `gym.Wrapper`'s own `.env` attribute is standard
gymnasium behavior, not something L2's wrapper had to add.

## Step 2: the orchestrator, and why it does NOT reimplement L3's inner loop

`src/agents/orchestrator_graph.py` now has, alongside the unchanged
`macro_tick()`: `L2_EVERY_N_TICKS = 50` (with an assertion that it divides
`L1_EVERY_N_TICKS` -- 600/50=12 exactly, which is what makes checking L1's
cadence only at L2-decision boundaries both correct and sufficient),
`strategist_tick()` (one L2 SAC prediction), and `run_episode()` (the full
graph).

**Reconciliation against the real interface, same kind of deviation L2's
own reconciliation needed against architecture_spec.md Section 4.1's
illustrative reference code:** the spec names `executioner_node` and
`env_step_node` as if they were separate per-tick orchestrator calls. They
are not implemented that way here. `FrozenL3Wrapper.step()` already
correctly implements both, for every one of its `ticks_per_l2_decision`
inner ticks -- it normalizes with the frozen checkpoint's own VecNormalize
stats, threads the L3 LSTM's `state=`/`episode_start=` explicitly (a real
bug class the wrapper's own module docstring documents fixing -- get this
wrong and the frozen policy silently degrades to stateless behavior,
nothing crashes), calls the frozen model's `predict()`, and steps the raw
env. Reimplementing that loop at the orchestrator level would duplicate
already-solved, already-tested logic (`tests/test_wrappers.py`, 20/20
passing) with real risk of silent divergence -- exactly the failure class
the wrapper's own docstring warns about -- and would require either editing
L2-owned code to stay in sync or drifting from it. So `run_episode()` calls
`wrapper.step(l2_action)` once per L2-decision boundary and treats
"executioner_node/env_step_node fire every tick" as a claim proven by
reading `wrapper.step()`'s source, then independently verified empirically
(Step 3) rather than trusted blind either way.

## Step 3: correctness assertions -- real checks, not a smoke test

New integration test,
`tests/test_orchestrator_graph.py::test_full_stack_integration_short_bounded_episode`.
Loads the REAL frozen L3 checkpoint and L2's real smoke-test SAC checkpoint
(`models/l2_strategist_smoke_test.zip` -- a real, correctly-shaped SAC
policy from L2's own 200-step mechanics smoke test; appropriate for a
correctness test since it just needs to produce syntactically valid
actions, not good ones). Synthetic constant-book data (same
`_write_synthetic_day` helper this file's existing tests already use, just
more rows: 3000, for a horizon_ticks=1250 episode -- chosen as an exact
multiple of `L2_EVERY_N_TICKS` and comfortably past 2x `L1_EVERY_N_TICKS`
so L1 fires 3 times). L1 mocked at the `requests.post` boundary (matching
this file's existing convention) with **three distinct** payload values per
call, not a fixed constant -- otherwise "changes at refreshes, holds
constant between them" couldn't be told apart from "never written at all."

Instrumentation: a non-invasive, pytest-`monkeypatch`-based recorder
wrapping the raw env's `step()` method at the INSTANCE level (auto-restored
after the test; `lob_execution_env.py` and `wrappers.py` stay byte-identical
on disk) captures idx-15/16/17/18-backing attribute values at every raw
tick before it executes.

**All assertions passed:**
- L1 fires exactly at ticks 0, 600, 1200 -- 3 firings, matching the 3
  distinct mocked values exactly (not just "changed").
- L2 fires in an exact arithmetic sequence (0, 50, 100, ..., 1200) -- 25
  firings, checked by construction (not assuming a fixed final tick, since
  early termination would change the count without changing the spacing).
- L3/`env_step_node` fires every tick: 1250 real `env.step()` calls,
  gapless, exactly covering ticks 0..1249 once each.
- idx 15/16 (`l2_target_slice_ratio_override`, `l2_urgency`) constant
  within every 50-tick block, and verified to actually DIFFER across
  blocks (not coincidentally constant the whole episode -- the underlying
  TWAP schedule baseline advances every block regardless of L2's own
  chosen multiplier).
- idx 17/18 (`l1_risk_score`, `l1_confidence`) constant between L1
  refreshes, change to the exact mocked value at ticks 0/600/1200 and
  nowhere else.
- Frozen checkpoint checksum verified live
  (`a5443e2a4c6c1d4427d4ce1cb83e65d622ea688d8953f5bf94b29e87fbcaa77d`),
  matches `docs/reports/l3_frozen_handoff.md`'s recorded value exactly.
- Episode truncates (horizon reached) with a finite `IS_total_bps` and
  `fill_ratio` in `[0, 1]`.

**A real, benign quirk found and documented, not silently worked around:**
`LOBExecutionEnv.step()` clamps `self._tick_idx` back to
`len(self._ticks) - 1` when it would run past the ticks buffer's end
(`src/envs/lob_execution_env.py` ~line 1067) -- which happens on EXACTLY
the tick that reaches `horizon_ticks`, since the ticks buffer is sized to
`episode_start + horizon_ticks` with no trailing margin. The clamp runs
after `truncated` is already correctly computed `True` but before
`_build_info()` runs, so `info["ticks_elapsed"]` under-reports by exactly 1
on that one final call. Not a bug -- truncation is still correct and every
tick still executed -- but a real semantic subtlety for anyone else
treating `ticks_elapsed` as a running tick counter across a
horizon-truncating boundary. The test asserts this exact relationship
explicitly (`len(tick_records) == result["tick"] + 1` on truncation) rather
than papering over it.

Full project suite: 132 passed, the same 4 pre-existing, unrelated failures
as every check this session (`test_bulk_backfill.py`/`test_l2_capture.py`,
network/resync issues, no `reward.py`/env import, confirmed untouched).

## Step 4: does the full stack run end-to-end, and is training feasible

**Yes, end-to-end, with L1 stubbed at the LLM boundary and everything else
real.** Real frozen L3 checkpoint, real L2 SAC checkpoint, real
`FrozenL3Wrapper`, real `LOBExecutionEnv` (real data for the timing
measurement below; synthetic constant-book data for the correctness test
above), real `build_l1_feature_summary()` pipeline available but not
exercised in the timing/correctness runs since Ollama can't be reached
(only the `requests.post` boundary is mocked/stubbed -- everything else in
L1's path, including the feature pipeline, is real code, real data,
verified working in Step 0).

**Measured wall-clock, real data (`data/val` split), pure inference (no
training, no gradient updates):**

| | |
|---|---|
| Setup (env + both model loads) | 1.046s (one-time) |
| Episode (1250 raw ticks, real data) | 4.768s |
| Per-tick | 3.81ms (262.15 ticks/sec) |
| Implied L2-decisions/sec (this stack, single env) | 5.24/sec |
| Extrapolated 3000-tick episode (L3's own standalone horizon) | 11.4s |

**Combined with L2's own measurement (commit `20ce3b6`, ~4.15
decisions/sec, single env) -- NOT a strictly apples-to-apples comparison,
stated plainly:** this round's number is pure inference (no SAC gradient
update) with L1 stubbed to near-zero cost; L2's own number is real SAC
training (inference + backprop) without L1 in the loop at all. The two
numbers landing in the same rough band (~4-5 decisions/sec, single env)
despite measuring different things is itself informative: it says the
per-decision cost is dominated by the SAME bottleneck L2's own measurement
already identified (env.reset()/step() I/O, ~47% reset()-dominated per
that measurement), not by gradient updates and not by L1. A real, full
3-tier training run would need at least L2's own inference+backprop cost
PLUS whatever a real (currently-broken) L1 LLM call adds -- even a fast
local call would very likely cost low-single-digit seconds per L1 firing;
amortized over the 12 L2-decisions between each L1 firing (600/50), that's
a real but not dominant addition on top of an already multi-day budget.

**Plain answer: three-tier training remains impractical at today's
single-env throughput, for the same reason L2's own track already
concluded -- this changes nothing about that verdict, it confirms it now
holds with L1 included too.** This orchestrator is validated as a
correctness harness (real checkpoints, real data flow, real cadence
guarantees, all independently verified) -- not as a training-ready
pipeline. L2's own recommendation (parallelize across multiple envs,
mirroring L3's own `n_envs=8` pattern) remains the clear, not-yet-implemented
path to making three-tier training practical; nothing in this round's
findings changes that recommendation, and L1's real-Ollama gap (Step 0
(b)/(c)) is now a second, independent blocker on top of it for anything
beyond a stubbed-L1 correctness check.

## Not done this round

No fix attempted for the Ollama service/model-directory mismatch (Step 0
(b)/(c)) -- reported as a real infra finding, not fixed, since this round's
scope was the correctness harness, not infra repair or the LLM path
itself. No multi-env parallelization built (L2's own recommendation, still
open). No real Ollama call attempted even after diagnosing the gap -- doing
so would require either fixing the systemd service's model path or
re-pulling under the correct user, neither of which was in scope this
round.
