"""Full L1 -> L2 -> L3 -> env orchestration (architecture_spec.md Section 4.3).

Extends the original macro_tick()-only wiring (still unchanged below) to the
full three-tier graph: macro_node (L1_EVERY_N_TICKS=600), strategist_node
(L2_EVERY_N_TICKS=50), executioner_node (every tick), env_step_node.

RECONCILIATION AGAINST THE REAL WRAPPER INTERFACE, same kind of deviation
L2's own reconciliation needed against Section 4.1's illustrative reference
code (see docs/reports/phase4_l2_reconciliation_and_plan.md): the spec's
graph names executioner_node and env_step_node as if they were separate,
per-tick orchestrator-level calls. They are not implemented that way here.
FrozenL3Wrapper.step() (src/envs/wrappers.py, L2-owned, NOT edited by this
module) already correctly implements both -- for every one of its
ticks_per_l2_decision inner ticks, it normalizes the observation with the
frozen checkpoint's own VecNormalize stats, threads the L3 LSTM's
state=/episode_start= explicitly (a real, non-obvious correctness
requirement -- see that file's own module docstring correction 4 for the
cross-episode leak bug this had to fix), calls the frozen L3 model's
predict(), and steps the raw env -- one call to wrapper.step(l2_action)
IS one full L2-decision window's worth of executioner_node + env_step_node
work, already tested (tests/test_wrappers.py, 20/20 passing at last count).
Reimplementing that loop here, against the same raw env, would duplicate
already-solved LSTM-threading/normalization logic with real risk of a
silent, non-crashing divergence (exactly the class of bug correction 4 in
wrappers.py's own docstring describes) -- and would require touching
L2-owned code to stay perfectly in sync, or drifting from it if not. So:
run_episode() below calls wrapper.step() once per L2-decision boundary and
treats "executioner_node/env_step_node fire every tick" as an invariant
PROVEN BY READING wrapper.step()'s source (see above) and independently
VERIFIED empirically by the correctness test (tests/test_orchestrator_graph.py),
which counts real env.step() calls via an external, non-invasive recorder
(no edit to wrappers.py or lob_execution_env.py) rather than trusted blind.

L1_EVERY_N_TICKS (600) is exactly 12x L2_EVERY_N_TICKS (50) -- asserted
below, not assumed -- so every L1 cadence boundary lands exactly on an
L2-decision boundary. This is what makes checking L1's cadence once per
L2-decision-boundary (rather than needing true per-raw-tick orchestrator
control) both correct and sufficient: no L1 boundary can ever fall in the
middle of an L2 window.

L1 wiring gap closed (flagged in docs/reports/l3_frozen_handoff.md): the
raw env's l1_risk_score/l1_confidence are plain public attributes -- the
SAME pattern FrozenL3Wrapper itself already uses for l2_target_slice_ratio_
override/l2_urgency (self.env.l2_target_slice_ratio_override = ...,
src/envs/wrappers.py). FrozenL3Wrapper exposes the raw env it wraps via its
own public `.env` attribute (inherited from gym.Wrapper), so the clean
answer is simply macro_tick(l3_wrapper.env, ...) -- no change needed inside
wrappers.py at all.
"""
from __future__ import annotations

import threading
from typing import Any, Callable

import numpy as np

from src.agents.l1_macro_analyst import L1MacroAnalyst, MacroRiskContext
from src.envs.wrappers import FrozenL3Wrapper

L1_EVERY_N_TICKS = 600  # ~60s at 100ms ticks (architecture_spec.md Section 4.3)
L2_EVERY_N_TICKS = 50  # matches FrozenL3Wrapper's own ticks_per_l2_decision default
assert L1_EVERY_N_TICKS % L2_EVERY_N_TICKS == 0, (
    "run_episode() below only checks L1's cadence at L2-decision boundaries -- this is "
    "only correct if every L1 boundary coincides with an L2 boundary, i.e. "
    "L2_EVERY_N_TICKS must divide L1_EVERY_N_TICKS."
)


def macro_tick(
    env, l1_agent: L1MacroAnalyst, tick: int, feature_summary: dict
) -> MacroRiskContext | None:
    """Called once per tick. Returns the MacroRiskContext just pushed into
    env.l1_risk_score/l1_confidence on a cadence tick (tick % L1_EVERY_N_TICKS
    == 0), or None on an off-cadence tick where env is left untouched."""
    if tick % L1_EVERY_N_TICKS != 0:
        return None
    context = l1_agent.maybe_refresh(feature_summary)
    env.l1_risk_score = context.risk_score
    env.l1_confidence = context.confidence
    return context


def strategist_tick(l2_model, l2_obs: np.ndarray, deterministic: bool = False):
    """strategist_node: one L2 decision from the current L2-visible observation.
    Called by run_episode() once per L2-decision boundary (every
    L2_EVERY_N_TICKS raw ticks) -- the cadence itself is enforced by the
    caller's loop structure (one call per wrapper.step()), not by this
    function, mirroring how macro_tick() enforces its own cadence internally
    via an explicit modulo check instead."""
    action, _ = l2_model.predict(l2_obs, deterministic=deterministic)
    return action


def run_episode(
    l3_wrapper: FrozenL3Wrapper,
    l2_model,
    l1_agent: L1MacroAnalyst,
    feature_summary_fn: Callable[[int], dict],
    seed: int | None = None,
    l2_deterministic: bool = False,
    on_l1_tick: Callable[[int, MacroRiskContext], None] | None = None,
    on_l2_tick: Callable[[int, np.ndarray], None] | None = None,
) -> dict[str, Any]:
    """The full three-tier graph for one episode: macro_node + strategist_node
    every L2-decision boundary, executioner_node/env_step_node folded into
    FrozenL3Wrapper.step() as described in this module's own docstring above.

    feature_summary_fn(tick) is called only on L1 cadence ticks -- pass e.g.
    a closure over build_l1_feature_summary(as_of_ms) for a real feature
    pipeline, or a fixed-dict lambda for a stubbed L1 (see this project's
    Step 0 findings on why the LLM call itself is currently stubbed, not the
    feature pipeline, which is real and tested).

    on_l1_tick/on_l2_tick are optional observation hooks (e.g. for a test
    harness to record cadence/values) -- neither is required for the graph
    itself to run correctly.

    Returns the final info dict plus tick/terminated/truncated, matching the
    raw env's own step()/reset() return shape closely enough for a caller to
    read implementation_shortfall off info["implementation_shortfall"].
    """
    l2_obs, info = l3_wrapper.reset(seed=seed)
    tick = 0
    terminated = truncated = False
    while not (terminated or truncated):
        if tick % L1_EVERY_N_TICKS == 0:
            context = macro_tick(l3_wrapper.env, l1_agent, tick, feature_summary_fn(tick))
            if context is not None and on_l1_tick is not None:
                on_l1_tick(tick, context)

        l2_action = strategist_tick(l2_model, l2_obs, deterministic=l2_deterministic)
        if on_l2_tick is not None:
            on_l2_tick(tick, l2_action)

        l2_obs, reward, terminated, truncated, info = l3_wrapper.step(l2_action)
        tick = info["ticks_elapsed"]  # ground truth from the raw env, robust to an
        # early-terminated/truncated window advancing fewer than L2_EVERY_N_TICKS ticks

    return {"tick": tick, "terminated": terminated, "truncated": truncated, "info": info}


class AsyncL1Refresher:
    """Wraps an L1MacroAnalyst so the tick loop never blocks on a real LLM
    call (architecture_spec.md Section 1.2: "Call this from a background
    thread or a separate process, not inline in the tick loop"). The tick
    loop only ever reads .cache -- whatever maybe_refresh() last produced,
    synchronously, inside a worker thread -- and never waits on a new call.

    Cache-staleness policy, explicit by design: if a NEW cadence-boundary
    refresh is due while a PREVIOUS background call is still in flight, the
    new request is SKIPPED (dropped), not queued and not run concurrently
    alongside it. Exactly one real LLM call is ever in flight at a time.
    Rejected alternative: queueing. A queue lets a slow patch of calls back
    up without bound, which is the wrong failure mode for a system whose own
    stated non-negotiable is "hot path never waits" -- an unbounded queue
    just moves the wait from the tick loop into memory. Skipping means the
    worst case is a bounded number of MISSED refreshes, not an unbounded
    backlog.

    Bounded, observable staleness: a single in-flight call is bounded by
    L1MacroAnalyst's own timeout_s (its underlying requests.post(timeout=...)
    -- a real network call that has not resolved by then raises
    requests.exceptions.Timeout, caught by maybe_refresh()'s own fail-closed
    except clause, and _in_flight clears). So worst-case staleness before a
    refresh can even be RETRIED is bounded by timeout_s, not indefinite; the
    next actual retry then waits for the next L1_EVERY_N_TICKS cadence
    boundary on top of that. last_refresh_started_tick/
    last_refresh_completed_tick/in_flight below make this directly
    observable rather than something a caller has to infer.

    Fail-closed survives threading by construction, not by a second
    mechanism: .cache is set to EXACTLY what agent.maybe_refresh() returns,
    and maybe_refresh() already fails closed internally (last good context,
    or neutral default) on any error. There is no separate "is this fresh"
    flag for the two to fall out of sync on -- a failed background call
    writes back the same value that was already there (or the neutral
    default on the very first call), never something worse.

    Thread lifecycle: at most one worker thread alive at a time (daemon=True
    as a backstop, not the primary cleanup mechanism -- see join()). Call
    join() once the episode's tick loop ends for a clean, bounded shutdown
    before starting a new episode/resetting -- see run_episode_async().
    """

    def __init__(self, agent: L1MacroAnalyst) -> None:
        self.agent = agent
        self._lock = threading.Lock()
        self._cache = agent._cache
        self._in_flight = False
        self._thread: threading.Thread | None = None
        self.last_refresh_started_tick: int | None = None
        self.last_refresh_completed_tick: int | None = None

    @property
    def cache(self) -> MacroRiskContext:
        with self._lock:
            return self._cache

    @property
    def in_flight(self) -> bool:
        with self._lock:
            return self._in_flight

    def maybe_refresh_async(self, tick: int, feature_summary_fn: Callable[[int], dict]) -> bool:
        """Non-blocking. Returns True if a new background call was started,
        False if one was already in flight and this request was skipped (see
        class docstring's staleness policy). feature_summary_fn(tick) is
        called INSIDE the worker thread, not here -- building a real
        feature_summary can itself do real file I/O
        (src/data/l1_features.py reads parquet), which must not block the
        tick loop either."""
        with self._lock:
            if self._in_flight:
                return False
            self._in_flight = True
            self.last_refresh_started_tick = tick

        def _worker() -> None:
            try:
                feature_summary = feature_summary_fn(tick)
                ctx = self.agent.maybe_refresh(feature_summary)
                with self._lock:
                    self._cache = ctx
                    self.last_refresh_completed_tick = tick
            finally:
                with self._lock:
                    self._in_flight = False

        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()
        return True

    def join(self, timeout: float | None = None) -> None:
        """Blocks until any in-flight background call finishes. Safe to call
        unconditionally (no-op if nothing is running) and safe to call with
        no timeout -- the thing it waits on is itself bounded by
        agent.timeout_s at the network layer (see class docstring)."""
        if self._thread is not None:
            self._thread.join(timeout=timeout)


def macro_tick_async(env, refresher: AsyncL1Refresher, tick: int, feature_summary_fn: Callable[[int], dict]) -> None:
    """Non-blocking counterpart to macro_tick(). At a cadence boundary, KICKS
    OFF a background refresh (if none is in flight) and, every tick
    regardless of cadence, writes the refresher's CURRENT cache -- whatever
    is freshest so far, possibly from several ticks ago, possibly still the
    neutral default -- into env.l1_risk_score/l1_confidence. Never waits."""
    if tick % L1_EVERY_N_TICKS == 0:
        refresher.maybe_refresh_async(tick, feature_summary_fn)
    ctx = refresher.cache
    env.l1_risk_score = ctx.risk_score
    env.l1_confidence = ctx.confidence


def run_episode_async(
    l3_wrapper: FrozenL3Wrapper,
    l2_model,
    refresher: AsyncL1Refresher,
    feature_summary_fn: Callable[[int], dict],
    seed: int | None = None,
    l2_deterministic: bool = False,
    on_l1_tick: Callable[[int, MacroRiskContext], None] | None = None,
    on_l2_tick: Callable[[int, np.ndarray], None] | None = None,
) -> dict[str, Any]:
    """Same three-tier graph as run_episode(), with macro_tick_async()/
    AsyncL1Refresher in place of macro_tick()/L1MacroAnalyst directly -- L1
    never blocks the tick loop here. A separate function rather than a
    shared code path with run_episode(): the two macro-tick calls need
    different inputs (a precomputed feature_summary dict vs. a
    feature_summary_fn callable the worker thread calls itself, see
    AsyncL1Refresher's own docstring for why), so unifying them would add a
    dispatch layer to a 15-line loop rather than remove real duplication --
    the same call this project already made elsewhere for a similar reason
    (src/data/download_manager.py vs. scripts/bulk_backfill.py's
    independent URL-building, "to keep CLI standalone" per that file's own
    comment).

    on_l1_tick fires exactly once per ACTUAL completion (not once per tick,
    and not at the tick that triggered the call -- at whichever tick the
    background thread happened to finish on, which may be several ticks
    later or, if skipped per the staleness policy, may not happen again
    until a later cadence boundary succeeds). Runs on the calling
    (main) thread, not the worker thread, since it is only invoked from
    inside this loop.

    ALWAYS calls refresher.join() before returning, including on an
    exception -- no thread is left running past this function's own
    return, satisfying "no leaked threads across episodes/resets" even if
    the episode itself errors out.
    """
    l2_obs, info = l3_wrapper.reset(seed=seed)
    tick = 0
    terminated = truncated = False
    last_reported_completion: int | None = None
    try:
        while not (terminated or truncated):
            macro_tick_async(l3_wrapper.env, refresher, tick, feature_summary_fn)
            completed = refresher.last_refresh_completed_tick
            if completed is not None and completed != last_reported_completion and on_l1_tick is not None:
                on_l1_tick(completed, refresher.cache)
                last_reported_completion = completed

            l2_action = strategist_tick(l2_model, l2_obs, deterministic=l2_deterministic)
            if on_l2_tick is not None:
                on_l2_tick(tick, l2_action)

            l2_obs, reward, terminated, truncated, info = l3_wrapper.step(l2_action)
            tick = info["ticks_elapsed"]
    finally:
        refresher.join()

    return {"tick": tick, "terminated": terminated, "truncated": truncated, "info": info}
