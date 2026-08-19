"""Minimal L1 -> env orchestration wiring (architecture_spec.md Section 4.3).

Proves only the L1-cache -> observation idx 17/18 path described in
Section 4.4 step 4/5: on every tick, macro_tick() is called; it only
actually refreshes L1 (itself already throttled to refresh_interval_s
inside maybe_refresh()) and pushes the result into the env's
l1_risk_score/l1_confidence stub hooks once every L1_EVERY_N_TICKS ticks,
matching the spec's macro_node cadence gate. Off-cadence ticks are a pure
no-op -- env.l1_risk_score/l1_confidence simply retain whatever was last
pushed, consistent with the "plain overridable attributes" contract
documented at the top of src/envs/lob_execution_env.py.

This is NOT yet the full LangGraph StateGraph from Section 4.3 (that also
needs L2/L3 model handles this task does not build) -- it exists to prove
the L1 side of the wiring in isolation before the rest of the graph is
assembled.
"""
from __future__ import annotations

from src.agents.l1_macro_analyst import L1MacroAnalyst, MacroRiskContext

L1_EVERY_N_TICKS = 600  # ~60s at 100ms ticks (architecture_spec.md Section 4.3)


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
