"""Tests for src/agents/orchestrator_graph.py -- proves the L1-cache ->
observation idx 17/18 path (architecture_spec.md Section 3.1 obs table,
Section 4.4 step 4/5), not the LLM call itself (see
tests/test_l1_macro_analyst.py for that). requests.post is mocked
throughout; no real Ollama call. Uses a tiny synthetic data day (same
technique as tests/test_lob_execution_env.py) so this is self-contained
and fast -- no dependency on the real Bybit archive.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import requests

from src.agents.l1_macro_analyst import L1MacroAnalyst
from src.agents.orchestrator_graph import L1_EVERY_N_TICKS, macro_tick
from src.envs.lob_execution_env import LOBExecutionEnv

VALID_PAYLOAD = {
    "timestamp_ms": 1_700_000_000_000,
    "regime": "risk_off",
    "risk_score": 0.6,
    "confidence": 0.8,
    "urgency_multiplier": 1.4,
    "rationale": "elevated funding + widening OBI",
}


class _FakeResponse:
    def __init__(self, body: dict):
        self._body = body

    def json(self):
        return self._body


def _fake_post_returning(response_field):
    def _fake_post(*args, **kwargs):
        return _FakeResponse({"response": response_field})
    return _fake_post


def _fake_post_raising(exc):
    def _fake_post(*args, **kwargs):
        raise exc
    return _fake_post


def _fake_post_counting(calls, response_field):
    def _fake_post(*args, **kwargs):
        calls.append(1)
        return _FakeResponse({"response": response_field})
    return _fake_post


def _write_synthetic_day(path, n_rows: int, base_price: float = 100.0, ts_start: int = 1_000_000) -> None:
    best_bid = base_price - 0.05
    best_ask = base_price + 0.05
    bids = json.dumps([[best_bid, 10.0], [best_bid - 0.1, 5.0]])
    asks = json.dumps([[best_ask, 10.0], [best_ask + 0.1, 5.0]])
    rows = [
        {
            "ts": ts_start + i,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid_price": base_price,
            "spread": best_ask - best_bid,
            "bids": bids,
            "asks": asks,
        }
        for i in range(n_rows)
    ]
    pd.DataFrame(rows).to_parquet(path, index=False)


def _make_env(tmp_path) -> LOBExecutionEnv:
    data_dir = tmp_path / "BTCUSDT"
    data_dir.mkdir()
    _write_synthetic_day(data_dir / "l2-BTCUSDT-2024-01-01.parquet", n_rows=20)
    env = LOBExecutionEnv(
        data_dir=data_dir, horizon_ticks=5, lookback_ticks=2,
        funding_rate_dir=tmp_path / "does_not_exist",
    )
    env.reset(seed=1)
    return env


def test_l1_every_n_ticks_matches_spec_cadence():
    assert L1_EVERY_N_TICKS == 600
    assert 0 % L1_EVERY_N_TICKS == 0  # tick 0 is always on-cadence


def test_macro_tick_on_cadence_pushes_nonneutral_context_into_obs_17_18(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.agents.l1_macro_analyst.requests.post",
        _fake_post_returning(json.dumps(VALID_PAYLOAD)),
    )
    env = _make_env(tmp_path)
    agent = L1MacroAnalyst(refresh_interval_s=45)

    context = macro_tick(env, agent, tick=0, feature_summary={"obi_1": 0.1})

    assert context is not None
    assert context.regime == "risk_off"
    assert env.l1_risk_score == pytest.approx(0.6)
    assert env.l1_confidence == pytest.approx(0.8)

    obs = env._build_obs()
    assert obs[17] == pytest.approx(0.6)   # l1_risk_score (architecture_spec.md Section 3.1)
    assert obs[18] == pytest.approx(0.8)   # l1_confidence


def test_macro_tick_off_cadence_is_a_noop(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "src.agents.l1_macro_analyst.requests.post",
        _fake_post_counting(calls, json.dumps(VALID_PAYLOAD)),
    )
    env = _make_env(tmp_path)
    agent = L1MacroAnalyst(refresh_interval_s=45)

    before_risk, before_conf = env.l1_risk_score, env.l1_confidence
    obs_before = env._build_obs()

    result = macro_tick(env, agent, tick=1, feature_summary={"obi_1": 0.1})

    assert result is None
    assert calls == [], "L1 should never even be asked on an off-cadence tick"
    assert env.l1_risk_score == before_risk
    assert env.l1_confidence == before_conf

    obs_after = env._build_obs()
    assert obs_after[17] == pytest.approx(obs_before[17])
    assert obs_after[18] == pytest.approx(obs_before[18])


def test_macro_tick_fail_closed_neutral_default_also_reaches_obs_17_18(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.agents.l1_macro_analyst.requests.post",
        _fake_post_raising(requests.exceptions.Timeout("simulated timeout")),
    )
    env = _make_env(tmp_path)
    agent = L1MacroAnalyst(refresh_interval_s=45)

    context = macro_tick(env, agent, tick=0, feature_summary={"obi_1": 0.1})

    assert context is not None
    assert context.regime == "neutral"
    assert env.l1_risk_score == 0.0
    assert env.l1_confidence == 0.0

    obs = env._build_obs()
    assert obs[17] == pytest.approx(0.0)
    assert obs[18] == pytest.approx(0.0)


# --- Full three-tier (L1->L2->L3) integration/correctness test ---
# Not a performance test -- see docs/reports/l3_frozen_handoff.md and
# docs/TRACK_STATUS.md's L1 section for that framing. Loads the REAL frozen
# L3 checkpoint and the REAL L2 smoke-test SAC checkpoint (both small,
# CPU/GPU-light), runs a short bounded synthetic episode, and asserts real
# cadence/data-flow invariants -- not a smoke test.
#
# L1 is STUBBED at the LLM boundary only (requests.post mocked, matching
# every other test in this file) -- the real, tested build_l1_feature_summary()
# pipeline is deliberately NOT used here, since Step 0 of this round found no
# real Ollama call can currently succeed (the pulled model sits under the
# wrong system user's home directory, invisible to the actual running
# service -- see TRACK_STATUS.md's L1 section). The mocked L1MacroAnalyst
# returns a DIFFERENT valid payload on each successive call (not just a fixed
# constant) so the "idx 17/18 changes exactly at L1 refreshes, holds constant
# between them" assertion is actually meaningful -- a mock that never changes
# could not distinguish "correctly held constant" from "never written at all".

FROZEN_L3_CHECKPOINT = "models/l3_frozen_backup/l3_executioner_v1_frozen.zip"
FROZEN_L3_VECNORM = "models/l3_frozen_backup/l3_vecnormalize_frozen.pkl"
FROZEN_L3_SHA256 = "a5443e2a4c6c1d4427d4ce1cb83e65d622ea688d8953f5bf94b29e87fbcaa77d"
L2_SMOKE_CHECKPOINT = "models/l2_strategist_smoke_test.zip"


def _fake_post_distinct_values(calls: list):
    """Returns a DIFFERENT valid MacroRiskContext payload on each call --
    calls.append records (risk_score, confidence) so the test can assert
    exactly which values env.l1_risk_score/l1_confidence should hold at
    each point, not just that something non-neutral happened."""
    def _fake_post(*args, **kwargs):
        i = len(calls)
        risk_score = round(-0.5 + 0.3 * i, 4)
        confidence = round(0.2 + 0.15 * i, 4)
        calls.append((risk_score, confidence))
        payload = {
            "timestamp_ms": 1_700_000_000_000 + i,
            "regime": "neutral",
            "risk_score": risk_score,
            "confidence": confidence,
            "urgency_multiplier": 1.0,
            "rationale": f"stubbed call #{i}",
        }
        return _FakeResponse({"response": json.dumps(payload)})
    return _fake_post


def test_full_stack_integration_short_bounded_episode(tmp_path, monkeypatch):
    import hashlib
    from pathlib import Path

    import torch
    from sb3_contrib import RecurrentPPO
    from stable_baselines3 import SAC

    from src.agents.orchestrator_graph import (
        L2_EVERY_N_TICKS, run_episode,
    )
    from src.envs.wrappers import FrozenL3Wrapper

    # --- Frozen checkpoint checksum, verified live, cross-checked against the
    # handoff doc's recorded value rather than trusted from memory. ---
    live_sha256 = hashlib.sha256(Path(FROZEN_L3_CHECKPOINT).read_bytes()).hexdigest()
    assert live_sha256 == FROZEN_L3_SHA256, (
        f"frozen L3 checkpoint checksum drifted from docs/reports/l3_frozen_handoff.md's "
        f"recorded value -- live={live_sha256} expected={FROZEN_L3_SHA256}"
    )

    # --- Synthetic multi-day-equivalent data: generous row count so the
    # random episode-start offset (LOBExecutionEnv.reset()) always has room
    # for the full horizon regardless of where it lands. Same _write_synthetic_day
    # helper/shape used by every other test in this file, just more rows. ---
    horizon_ticks = 1250  # > 2*L1_EVERY_N_TICKS so L1 fires 3x (ticks 0, 600, 1200);
    # also an exact multiple of L2_EVERY_N_TICKS (25*50) so a full-horizon truncation
    # lands exactly on an L2-decision boundary, not mid-block.
    lookback_ticks = 10
    data_dir = tmp_path / "BTCUSDT"
    data_dir.mkdir()
    _write_synthetic_day(data_dir / "l2-BTCUSDT-2024-01-01.parquet", n_rows=3000)

    env = LOBExecutionEnv(
        data_dir=data_dir, horizon_ticks=horizon_ticks, lookback_ticks=lookback_ticks,
        funding_rate_dir=tmp_path / "does_not_exist",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    l3_model = RecurrentPPO.load(FROZEN_L3_CHECKPOINT, device=device)
    l2_model = SAC.load(L2_SMOKE_CHECKPOINT, device=device)

    l3_wrapper = FrozenL3Wrapper(
        env, l3_model, FROZEN_L3_VECNORM,
        ticks_per_l2_decision=L2_EVERY_N_TICKS,
    )

    # --- Instrumentation: wraps the RAW env's step() to record, at every raw
    # tick BEFORE that tick executes, the exact idx-15/16/17/18-backing
    # attribute values and increment a call counter. This is a pure
    # instance-level monkeypatch (pytest's monkeypatch fixture, auto-restored)
    # -- it does not edit lob_execution_env.py or wrappers.py, both of which
    # stay byte-identical on disk. It is what lets "L3 fires every tick" and
    # the two hold-constant-between-cadence-boundaries claims be asserted
    # empirically rather than only argued from reading the source. ---
    tick_records: list[dict] = []
    real_env_step = env.step

    def _recording_step(action):
        tick_records.append({
            "tick_before": env._tick_idx - env._episode_start,
            "l2_target_slice_ratio_override": env.l2_target_slice_ratio_override,
            "l2_urgency": env.l2_urgency,
            "l1_risk_score": env.l1_risk_score,
            "l1_confidence": env.l1_confidence,
        })
        return real_env_step(action)

    monkeypatch.setattr(env, "step", _recording_step)

    l1_calls: list = []
    monkeypatch.setattr(
        "src.agents.l1_macro_analyst.requests.post",
        _fake_post_distinct_values(l1_calls),
    )
    l1_agent = L1MacroAnalyst(refresh_interval_s=0.0)  # tick-cadence is what's under
    # test here, not the agent's own wall-clock throttle -- see this file's own
    # test_macro_tick_* tests above for that throttle covered in isolation.

    l1_fires: list[tuple[int, float, float]] = []
    l2_fires: list[int] = []

    def _on_l1(tick, context):
        l1_fires.append((tick, context.risk_score, context.confidence))

    def _on_l2(tick, action):
        l2_fires.append(tick)

    result = run_episode(
        l3_wrapper, l2_model, l1_agent,
        feature_summary_fn=lambda tick: {"stub": True, "tick": tick},
        seed=1,
        on_l1_tick=_on_l1,
        on_l2_tick=_on_l2,
    )

    # --- Cadence assertions ---
    assert l1_fires == [
        (0, l1_calls[0][0], l1_calls[0][1]),
        (600, l1_calls[1][0], l1_calls[1][1]),
        (1200, l1_calls[2][0], l1_calls[2][1]),
    ], f"expected exactly 3 L1 firings at ticks 0/600/1200 with the 3 distinct mocked values, got {l1_fires}"
    assert len(l1_calls) == 3, "L1MacroAnalyst.maybe_refresh() (mocked requests.post) should have been called exactly 3 times, once per cadence tick"

    # L2 fires at an exact arithmetic sequence starting at 0, step L2_EVERY_N_TICKS --
    # checked by construction rather than assuming a specific final tick count, since
    # early termination (order completed before the horizon) would change how many
    # firings there are without changing their spacing.
    assert l2_fires == list(range(0, len(l2_fires) * L2_EVERY_N_TICKS, L2_EVERY_N_TICKS)), (
        f"L2 firings must form an exact arithmetic sequence (0, {L2_EVERY_N_TICKS}, "
        f"{2 * L2_EVERY_N_TICKS}, ...); got {l2_fires}"
    )
    assert result["tick"] - l2_fires[-1] <= L2_EVERY_N_TICKS, (
        f"final tick {result['tick']} is more than one L2 block past the last L2 firing "
        f"at {l2_fires[-1]} -- a block must have run without an L2 decision"
    )

    # L3/env_step_node fires every tick: tick_records' own internal consistency
    # (gapless 0..N-1, exactly once each) is checked independent of
    # info["ticks_elapsed"], which has a genuine, benign off-by-one on the specific
    # tick that triggers horizon truncation -- see below.
    recorded_ticks = [r["tick_before"] for r in tick_records]
    assert recorded_ticks == list(range(len(tick_records))), (
        f"raw env.step() calls should cover ticks 0..{len(tick_records) - 1} exactly once "
        f"in order (proving L3/executioner_node fires every tick, no gaps or repeats), "
        f"got {recorded_ticks[:10]}..."
    )
    # Cross-check tick_records' count against info["ticks_elapsed"] (result["tick"]),
    # accounting for a real quirk found by this test: LOBExecutionEnv.step() (line
    # ~1067) clamps self._tick_idx back to len(self._ticks) - 1 when it would run past
    # the ticks buffer's end -- which happens on exactly the tick that reaches
    # horizon_ticks, since self._ticks is sized to episode_start + horizon_ticks with
    # no trailing buffer past the horizon. The clamp is applied AFTER truncated is
    # already computed True but BEFORE _build_info() runs, so ticks_elapsed on that one
    # final call reports one less than the true count of ticks processed. Not a bug --
    # truncated is still correctly True and every tick still executed -- but a real
    # semantic subtlety for anyone else relying on info["ticks_elapsed"] as a tick
    # counter across a truncating boundary.
    if result["truncated"]:
        assert len(tick_records) == result["tick"] + 1, (
            f"on truncation, expected len(tick_records) ({len(tick_records)}) == "
            f"result['tick'] + 1 ({result['tick'] + 1}) due to the end-of-ticks-buffer "
            f"clamp in LOBExecutionEnv.step() -- if this no longer holds, the clamp's "
            f"behavior (or info['ticks_elapsed']'s semantics) may have changed"
        )
    else:
        assert len(tick_records) == result["tick"], (
            f"on termination (not truncation), expected len(tick_records) "
            f"({len(tick_records)}) == result['tick'] ({result['tick']}) exactly"
        )

    # idx 15/16 (l2_target_slice_ratio_override, l2_urgency): constant within each
    # L2_EVERY_N_TICKS block.
    for block_start in range(0, result["tick"], L2_EVERY_N_TICKS):
        block_records = [r for r in tick_records if block_start <= r["tick_before"] < block_start + L2_EVERY_N_TICKS]
        ratios = {r["l2_target_slice_ratio_override"] for r in block_records}
        urgencies = {r["l2_urgency"] for r in block_records}
        assert len(ratios) == 1, f"l2_target_slice_ratio_override changed mid-block at block starting tick {block_start}: {ratios}"
        assert len(urgencies) == 1, f"l2_urgency changed mid-block at block starting tick {block_start}: {urgencies}"
    # And it does change ACROSS blocks -- not a coincidentally-constant value throughout
    # the whole episode (the underlying TWAP schedule baseline strictly increases block
    # to block regardless of L2's own chosen participation multiplier, so this should
    # essentially always vary).
    per_block_ratio = [
        tick_records[block_start]["l2_target_slice_ratio_override"]
        for block_start in range(0, result["tick"], L2_EVERY_N_TICKS)
    ]
    assert len(set(per_block_ratio)) > 1, (
        f"l2_target_slice_ratio_override was identical across every L2 block "
        f"({per_block_ratio[0]!r}) -- expected it to vary with the advancing TWAP schedule"
    )

    # idx 17/18 (l1_risk_score, l1_confidence): constant between L1 refreshes, changes
    # exactly at ticks 0/600/1200 to the 3 distinct mocked values.
    expected_l1_value_at_tick = {0: l1_calls[0], 600: l1_calls[1], 1200: l1_calls[2]}
    current_expected = None
    for r in tick_records:
        if r["tick_before"] in expected_l1_value_at_tick:
            current_expected = expected_l1_value_at_tick[r["tick_before"]]
        assert current_expected is not None, "no L1 value should be expected before tick 0's own refresh runs"
        assert r["l1_risk_score"] == pytest.approx(current_expected[0]), (
            f"l1_risk_score at tick {r['tick_before']} was {r['l1_risk_score']}, "
            f"expected {current_expected[0]} (last L1 refresh's value)"
        )
        assert r["l1_confidence"] == pytest.approx(current_expected[1]), (
            f"l1_confidence at tick {r['tick_before']} was {r['l1_confidence']}, "
            f"expected {current_expected[1]} (last L1 refresh's value)"
        )

    # Episode termination and a sane implementation_shortfall.
    assert result["terminated"] or result["truncated"], "episode must end via one path or the other"
    is_result = result["info"]["implementation_shortfall"]
    assert np.isfinite(is_result.is_total_bps), f"IS_total_bps must be finite, got {is_result.is_total_bps}"
    assert 0.0 <= is_result.fill_ratio <= 1.0, f"fill_ratio out of [0,1]: {is_result.fill_ratio}"
