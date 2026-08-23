# L1 real (unmocked) LLM validation

**Date:** 2026-08-24
**Status:** Ollama service model-visibility bug fixed. A second, independent bug found
and fixed in L1's own client code (environment proxy). Real end-to-end calls now
succeed at the transport level. Schema-conformance of the model's own JSON output is
poor (0/5 in this round's sample) -- reported as a real finding, not worked around.
Full 3-tier stack re-run with real L1 in the loop: runs correctly end to end; per-tick
cost roughly quadruples over the episode measured (3.81ms -> 14.60ms) due to 3 real
LLM calls, consistent with a straightforward amortization calculation.

## Step 1: Ollama model-visibility fix

**Root cause, confirmed directly (not assumed):** the systemd service runs as system
user `ollama` (uid 998, home `/usr/share/ollama`). Its own
`~/.ollama/models/{blobs,manifests}` were empty. The real 8.4GB
`qwen2.5:14b-instruct-q4_K_M` pull sat under `/home/ubuntu/.ollama/models` instead --
almost certainly because that pull happened while the service's systemd unit was still
crash-looping on the earlier `ExecStart=/bin/ollama` bug (fixed in an earlier round),
so `ollama pull` (run interactively as `ubuntu`) fell back to its own
embedded-server behavior using the invoking user's `$HOME`, rather than delegating to
a reachable running service.

**Fix chosen and why:** moved the blob store (`mv`, same filesystem --
`/dev/vda1` for both paths, confirmed via `df`, so this was an instant rename with zero
duplicate-disk-usage, not an 8.4GB copy) from `/home/ubuntu/.ollama/models` into the
service's own default location, `/usr/share/ollama/.ollama/models`, then
`chown -R ollama:ollama`. Considered and rejected: pointing `OLLAMA_MODELS` at the
existing `/home/ubuntu/.ollama/models` path instead -- checked directly and rejected,
not assumed: `/home/ubuntu` itself is `750 ubuntu:ubuntu`, and the `ollama` user
(groups: ollama, video, render) has no membership granting access, so the `ollama`
service could not even traverse into that directory regardless of an `OLLAMA_MODELS`
override, without ALSO loosening a real user's home-directory permissions -- a bigger,
less contained change than moving 8.4GB within the same filesystem.

**Why future pulls land in the right place automatically, not just this once:**
Ollama's CLI (`ollama pull`) delegates to a reachable local server's `/api/pull`
endpoint when one is running, rather than downloading into the invoking user's own
`$HOME` -- confirmed by this round's own root-cause finding above (the ONLY reason the
original pull landed under `/home/ubuntu` was that no server was reachable at pull
time). Now that the systemd service is healthy and serving from its own default
`OLLAMA_MODELS` location, any future `ollama pull` (by any user, since it just needs
`localhost:11434` reachable) will be executed BY the service process and land in ITS
`OLLAMA_MODELS` directory automatically -- no per-user environment variable to
remember or maintain going forward.

**Verified with real calls, not just service status:**
- `curl --noproxy "*" localhost:11434/api/tags` -> lists `qwen2.5:14b-instruct-q4_K_M`,
  8,988,124,069 bytes, correct digest.
- `curl --noproxy "*" localhost:11434/api/generate` with a real prompt -> real
  completion (`"response":"OK"`), `done_reason:"stop"`.
- Cold first call: 16.93s wall-clock (`load_duration` 16.78s of that -- model weights
  loading into VRAM for the first time). Warm calls: ~0.6-0.65s floor for a trivial
  prompt.
- VRAM: idle (no process) 1 MiB. 14B loaded + actively generating: 15,033 MiB. 14B
  loaded but idle between requests (Ollama's own keep-alive partial unload): 9,107 MiB.
  Total GPU: 24,564 MiB.

**This infra change is NOT a repo commit** (systemd unit content unchanged this round;
only file locations on disk moved) -- documented here and in TRACK_STATUS.md's L1
section, matching the earlier `ExecStart` fix's own precedent.

## Step 1b (found during Step 2, not anticipated going in): a second real bug, in code this time

Running `L1MacroAnalyst.maybe_refresh()` for real (unmocked) immediately failed
closed on every call, in well under a millisecond -- inconsistent with a real network
round-trip. Traced directly (not guessed): `requests.post()` in
`src/agents/l1_macro_analyst.py` honors this host's `http_proxy`/`https_proxy`
environment variables by default (both set, pointed at an external egress proxy
unrelated to Ollama), silently misrouting every `localhost:11434` call through it and
raising `ProxyError` -- caught by the existing `except requests.RequestException`
clause, indistinguishable from a real Ollama outage without inspecting the exception
message directly. **This is a real bug in L1's own file, not an infra quirk** -- it
would have silently defeated Step 1's own fix in any environment (like this one) where
those proxy variables are set, regardless of whether the model-visibility problem was
fixed.

**Fix:** added `proxies={"http": None, "https": None}` to the existing
`requests.post()` call (`src/agents/l1_macro_analyst.py`) -- `self.host` is always
local by construction (default `http://localhost:11434`), so bypassing any configured
proxy is unconditionally correct for this specific call, not a narrowing of behavior.
One-line regression test added
(`tests/test_l1_macro_analyst.py::test_maybe_refresh_bypasses_environment_proxy`,
mocked, asserts the `proxies=` kwarg is present on every call) so this cannot silently
regress. Full existing L1 test suite (8/8) still passes unchanged.

## Step 2: first real, unmocked `maybe_refresh()` calls -- robustness result

5 real calls (model `qwen2.5:14b-instruct-q4_K_M`, the one actually available --
spec's default 32B is not pulled, see Step 3), real `feature_summary` from
`build_l1_feature_summary()` at 5 different real, spread-out `as_of_ms` values
(2026-08, 2026-07, 2026-03, 2025-06, 2024-01) so this is genuinely testing different
real market conditions, not one lucky/unlucky draw repeated.

**Result: 0/5 raw responses were schema-conformant against `MacroRiskContext`.** The
model consistently returns syntactically valid JSON (Ollama's `format:"json"` mode
does guarantee that much) but with ITS OWN invented field structure --
`{"risk_level": "LOW", "as_of_ms": ..., "features": {...}}`,
`{"market_risk_score": 5, "risk_category": "Moderate", ...}`, etc. -- never the
required `timestamp_ms`/`regime`/`risk_score`/`confidence`/`urgency_multiplier`
top-level keys, across all 5 real calls. This is reported as the finding, not papered
over: `format:"json"` only constrains SYNTACTIC validity, not conformance to any
particular schema, and `SYSTEM_PROMPT` (architecture_spec.md Section 1.2, unchanged
this round) never actually states the required field names anywhere in the text sent
to the model -- it only prose-describes the input content and says output must match
"the required schema" without ever showing that schema. The model has no way to know
the exact keys it was never given. **This strongly suggests the 0% conformance rate is
a prompt-design gap, not primarily a 14B-capability gap** -- relevant directly to the
Step 3 model-size decision below.

**The fail-closed mechanism itself: 5/5 reliable under this real, repeated failure.**
Every one of the 5 real `maybe_refresh()` calls correctly returned the neutral
fallback context (`regime="neutral"`, `risk_score=0.0`, `confidence=0.0`,
`urgency_multiplier=1.0`) with no crash, no propagated garbage, and no hang -- exactly
the documented contract, now verified against a REAL, REPEATED failure mode rather
than a single simulated one. This is a genuinely positive result about the safety
design, reported alongside the negative schema-conformance result rather than let one
obscure the other.

**Timing (raw HTTP call, warm model, real feature_summary-sized prompt):** 4.49s,
4.46s, 4.52s, 4.51s, 4.54s across the 5 calls -- mean 4.505s, tight variance (+/-
0.03s). The `L1MacroAnalyst` class wrapper itself adds negligible overhead (its own
measured elapsed matched the raw call within ~0.1s in every case).

## Step 3: 14B vs 32B -- recommendation

**Recommend: stay on 14B for now. Do not pull 32B yet.**

VRAM headroom, measured, not estimated: 24,564 MiB total. 14B loaded and actively
generating measured 15,033 MiB, leaving ~9.3GB free right now (L2/L3 both idle at
measurement time -- checked `nvidia-smi --query-compute-apps` directly before and
after every load in this round, not assumed clear). The spec's own 32B estimate is
~20-21GB -- if accurate, that leaves only ~3.5GB headroom on a shared 4090 with L2
"starting vectorization work" and L3 having just run matched GPU A/B training, a much
tighter margin for concurrent multi-track GPU use than 14B's ~9.3GB free.

**The stronger reason, from this round's own evidence, not just VRAM:** Step 2 found
the observed 0% schema-conformance is very likely a PROMPT-DESIGN gap (the required
field names are never shown to the model at all, regardless of model size), not
primarily a 14B-capability gap. Spending a ~20GB download plus committing ~20-21GB of
shared VRAM to test whether a bigger model magically infers an unstated schema is a
weak bet against that root-cause read. The higher-value, zero-additional-cost next
step is fixing `SYSTEM_PROMPT` to actually state the required keys, then re-testing
against the already-available 14B before spending anything on 32B at all. Not done
this round (a prompt-design change is a judgment call, flagged for direction rather
than made unilaterally here -- see CLAUDE.md rule 7).

**Throughput implication, measured, confirming the task's own suspicion:** L1 fires
every 600 ticks (60s of simulated market time at the spec's 100ms tick interval,
architecture_spec.md Section 4.3). At the correctness harness's own measured 262
ticks/sec (L1 stubbed), 600 ticks take ~2.29s of REAL wall-clock time to compute. This
round's measured real 14B call latency is 4.49-4.54s (mean 4.505s, Step 2) --
**roughly double that 2.29s budget.** Run synchronously, as `orchestrator_graph.py`'s
current `run_episode()` does today, a real LLM call WOULD stall the tick loop and
become a new, measurable bottleneck (confirmed directly in Step 4 below, not just
predicted here). This is not fundamentally a model-size problem either -- a 32B model
would likely be SLOWER per call, not faster, making a synchronous-blocking bottleneck
worse, not better. The actual fix is architectural and already spec-sanctioned:
architecture_spec.md Section 1.2 explicitly says to call this "from a background
thread or a separate process, not inline in the tick loop", and
`maybe_refresh()`'s own docstring already documents this same non-blocking
contract -- `orchestrator_graph.run_episode()` just doesn't implement it yet (calls
`macro_tick()`, and therefore `maybe_refresh()`, synchronously inline). Building that
background-thread wiring would remove this bottleneck independent of the 14B/32B
choice; not done this round (a real implementation task in its own right, flagged for
direction rather than started unprompted).

## Step 4: full 3-tier stack re-run, real L1 live in the loop

Same real checkpoints and cadence structure as the prior correctness harness
(`docs/reports/l1l2l3_integration_correctness.md`): frozen L3 checkpoint
(`models/l3_frozen_backup/l3_executioner_v1_frozen.zip`), L2's real smoke-test SAC
checkpoint, `FrozenL3Wrapper`, `orchestrator_graph.run_episode()`. Two changes from
that prior run: (1) REAL market data (`data/raw_l2_bybit/BTCUSDT`, val split,
2025-07-16..2025-08-02, via `src.data.split.load_split("val")`) instead of synthetic
constant-book data, matching the prior round's own timing-measurement setup rather
than its correctness-assertion setup; (2) a REAL, unmocked `L1MacroAnalyst(model=
"qwen2.5:14b-instruct-q4_K_M")` fed by real `build_l1_feature_summary()` output,
instead of a mocked `requests.post`. Not added as a permanent pytest test (would make
the suite depend on a live external service, breaking the hermeticity every other test
in this file deliberately preserves by mocking `requests.post`) -- run as a one-off
validation script, results captured here.

**Runs correctly end to end, no crashes, sane output:** 1249 raw ticks, `truncated=
True` (horizon reached), `IS_total_bps=5.5347` (finite), `fill_ratio=0.7615` (in
[0,1]). L1 fired exactly 3 times at ticks 0/600/1200 (identical cadence to the stubbed
run). L2 fired 25 times, matching the prior round's own count exactly.

**idx 17/18 did NOT visibly change across boundaries this run** -- all 3 real L1
firings returned the identical neutral fallback (`risk_score=0.0, confidence=0.0`),
consistent with Step 2's 0/5 schema-conformance finding holding here too. Reported
plainly rather than reframed: with the real model/prompt as they stand today, a live
L1 in the loop is currently equivalent to the stubbed no-op baseline
(architecture_spec.md Section 4.4 step 4) from the L2/L3 observation's point of view --
not because the wiring is broken, but because no real call has yet produced
schema-valid output to write.

**Per-tick cost: 3.81ms (stubbed, prior round) -> 14.60ms (real L1, this round) --
episode-average, ~3.8x.** Setup (env + both model loads): 1.022s. Episode wall-clock:
18.236s / 1249 ticks = 14.60ms/tick (68.49 ticks/sec), down from 262.15 ticks/sec
stubbed. This matches a direct amortization calculation, not an unexplained
regression: 3 real L1 calls x ~4.5s (Step 2's measured mean) = ~13.5s of added
wall-clock, and 18.236s - 13.5s = 4.74s is within rounding of the prior 3.81ms/tick x
1249 ticks = 4.76s stubbed baseline for the same tick count. In a full-length training
episode (not this short 1250-tick harness), the same real per-call cost amortizes over
many more ticks between firings -- roughly `4.5s / 600 ticks` = 7.5ms/tick of steady-
state added overhead on top of the 3.81ms/tick baseline once L1 is genuinely
synchronous in the hot path, not the 3.8x multiplier seen in this short episode
specifically (which has an unusually high ratio of L1 firings to total ticks). Either
way: **synchronous L1 is a real, measurable throughput cost today**, and the
background-thread fix discussed in Step 3 is what removes it, not a faster model.

## Not done this round

No prompt-schema fix (Step 2/3's own recommended next step) -- a content/design
change, flagged for direction rather than made unilaterally. No background-thread /
async execution model for L1 (Step 3/4's own recommended throughput fix) -- a real
implementation task, same reasoning. No 32B pull -- recommended against for now, see
Step 3. `orchestrator_graph.py`/`tests/test_orchestrator_graph.py` unchanged this
round (Step 4 was a validation script, not new committed test code, per the
hermeticity reasoning above).
