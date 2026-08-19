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
