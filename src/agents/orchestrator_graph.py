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
