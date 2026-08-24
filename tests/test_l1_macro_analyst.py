"""Tests for src/agents/l1_macro_analyst.py (architecture_spec.md Section
1.2). requests.post is always mocked -- no real network/model call
anywhere in this file; Ollama is never started or contacted.
"""
from __future__ import annotations

import json

import pytest
import requests

from src.agents.l1_macro_analyst import L1MacroAnalyst, MacroRiskContext

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


def test_maybe_refresh_valid_response_parses_correctly(monkeypatch):
    monkeypatch.setattr(
        "src.agents.l1_macro_analyst.requests.post",
        _fake_post_returning(json.dumps(VALID_PAYLOAD)),
    )
    agent = L1MacroAnalyst(refresh_interval_s=45)
    ctx = agent.maybe_refresh({"obi_1": 0.2})

    assert isinstance(ctx, MacroRiskContext)
    assert ctx.regime == "risk_off"
    assert ctx.risk_score == pytest.approx(0.6)
    assert ctx.confidence == pytest.approx(0.8)
    assert ctx.urgency_multiplier == pytest.approx(1.4)
    assert ctx.rationale == "elevated funding + widening OBI"
    assert agent._cache is ctx


def test_maybe_refresh_malformed_json_falls_back_to_neutral_default(monkeypatch):
    monkeypatch.setattr(
        "src.agents.l1_macro_analyst.requests.post",
        _fake_post_returning("not valid json{"),
    )
    agent = L1MacroAnalyst(refresh_interval_s=45)
    ctx = agent.maybe_refresh({"obi_1": 0.2})

    assert ctx.regime == "neutral"
    assert ctx.risk_score == 0.0
    assert ctx.confidence == 0.0
    assert ctx.urgency_multiplier == 1.0
    assert ctx.rationale == "fallback: no LLM signal yet"


def test_maybe_refresh_malformed_json_falls_back_to_last_good_cache(monkeypatch):
    # First a good call establishes a non-neutral cache, then a malformed one
    # after the throttle window elapses must keep that same cached context,
    # not silently reset to neutral -- the spec's "last good context (or
    # neutral default)" fallback is two distinct branches, not one.
    clock = [1000.0]
    monkeypatch.setattr("src.agents.l1_macro_analyst.time.time", lambda: clock[0])
    monkeypatch.setattr(
        "src.agents.l1_macro_analyst.requests.post",
        _fake_post_returning(json.dumps(VALID_PAYLOAD)),
    )
    agent = L1MacroAnalyst(refresh_interval_s=45)
    good_ctx = agent.maybe_refresh({"obi_1": 0.2})
    assert good_ctx.regime == "risk_off"

    clock[0] += 46  # past the throttle window
    monkeypatch.setattr(
        "src.agents.l1_macro_analyst.requests.post",
        _fake_post_returning("not valid json{"),
    )
    ctx = agent.maybe_refresh({"obi_1": 0.2})

    assert ctx is good_ctx
    assert ctx.regime == "risk_off"


def test_maybe_refresh_schema_violation_fails_closed(monkeypatch):
    bad_payload = {**VALID_PAYLOAD, "risk_score": 2.0}  # outside [-1, 1]
    monkeypatch.setattr(
        "src.agents.l1_macro_analyst.requests.post",
        _fake_post_returning(json.dumps(bad_payload)),
    )
    agent = L1MacroAnalyst(refresh_interval_s=45)
    ctx = agent.maybe_refresh({"obi_1": 0.2})

    assert ctx.regime == "neutral"
    assert ctx.risk_score == 0.0


def test_maybe_refresh_timeout_falls_back_cleanly(monkeypatch):
    monkeypatch.setattr(
        "src.agents.l1_macro_analyst.requests.post",
        _fake_post_raising(requests.exceptions.Timeout("simulated timeout")),
    )
    agent = L1MacroAnalyst(refresh_interval_s=45)
    ctx = agent.maybe_refresh({"obi_1": 0.2})

    assert ctx.regime == "neutral"
    assert ctx.risk_score == 0.0


def test_maybe_refresh_missing_response_key_fails_closed(monkeypatch):
    # resp.json()["response"] itself raises KeyError -- a distinct failure
    # mode from a malformed *value* in that field, also in the except tuple.
    monkeypatch.setattr(
        "src.agents.l1_macro_analyst.requests.post",
        lambda *a, **kw: _FakeResponse({}),
    )
    agent = L1MacroAnalyst(refresh_interval_s=45)
    ctx = agent.maybe_refresh({"obi_1": 0.2})

    assert ctx.regime == "neutral"


def test_maybe_refresh_throttles_within_interval_then_refreshes_after(monkeypatch):
    calls = []

    def _fake_post(*args, **kwargs):
        calls.append(1)
        return _FakeResponse({"response": json.dumps(VALID_PAYLOAD)})

    monkeypatch.setattr("src.agents.l1_macro_analyst.requests.post", _fake_post)
    clock = [1000.0]
    monkeypatch.setattr("src.agents.l1_macro_analyst.time.time", lambda: clock[0])

    agent = L1MacroAnalyst(refresh_interval_s=45)

    ctx1 = agent.maybe_refresh({"obi_1": 0.2})
    assert len(calls) == 1

    clock[0] += 10  # still inside the 45s throttle window
    ctx2 = agent.maybe_refresh({"obi_1": 0.2})
    assert len(calls) == 1, "should not re-call Ollama within refresh_interval_s"
    assert ctx2 is ctx1

    clock[0] += 40  # 50s since the first call -> past the throttle window
    ctx3 = agent.maybe_refresh({"obi_1": 0.2})
    assert len(calls) == 2, "should re-call once refresh_interval_s has elapsed"
    assert ctx3 is not ctx1


def test_maybe_refresh_bypasses_environment_proxy(monkeypatch):
    # Regression test for a real bug found running this against a real Ollama service:
    # requests.post() defaults to honoring HTTP(S)_PROXY env vars, which silently
    # misroutes every localhost call through an unrelated external proxy in an
    # environment where those are set (this host's ubuntu-user shell), producing a
    # ProxyError indistinguishable from a real Ollama outage. maybe_refresh() must pass
    # proxies={"http": None, "https": None} explicitly on every call.
    captured_kwargs = {}

    def _fake_post(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return _FakeResponse({"response": json.dumps(VALID_PAYLOAD)})

    monkeypatch.setattr("src.agents.l1_macro_analyst.requests.post", _fake_post)
    agent = L1MacroAnalyst(refresh_interval_s=45)
    agent.maybe_refresh({"obi_1": 0.2})

    assert captured_kwargs.get("proxies") == {"http": None, "https": None}


def test_maybe_refresh_sends_structured_output_schema_not_plain_json_string(monkeypatch):
    # Regression test for the fix in docs/reports/l1_real_llm_validation.md's own
    # follow-up: format="json" alone let the real model invent its own field names
    # (0/10 conformant before this fix -- see the report). format= must be the real
    # MacroRiskContext JSON schema object, not the bare string "json", and it must be
    # the SAME schema pydantic validates against afterward (single source of truth,
    # not a hand-copied duplicate that could drift).
    captured_kwargs = {}

    def _fake_post(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return _FakeResponse({"response": json.dumps(VALID_PAYLOAD)})

    monkeypatch.setattr("src.agents.l1_macro_analyst.requests.post", _fake_post)
    agent = L1MacroAnalyst(refresh_interval_s=45)
    agent.maybe_refresh({"obi_1": 0.2})

    sent_body = captured_kwargs.get("json", {})
    assert sent_body.get("format") == MacroRiskContext.model_json_schema()
    assert sent_body.get("format") != "json"


def test_maybe_refresh_parses_real_shaped_conformant_output(monkeypatch):
    # Real-shaped fixture: field values matching what live, unmocked calls actually
    # returned once structured output was enabled (docs/reports/l1_real_llm_validation.md's
    # follow-up), not a hand-picked easy case -- regime "neutral", small-magnitude
    # risk_score, sub-1.0 confidence, a full-sentence rationale.
    real_shaped_payload = {
        "timestamp_ms": 1786363200000,
        "regime": "neutral",
        "risk_score": -0.1,
        "confidence": 0.75,
        "urgency_multiplier": 1.0,
        "rationale": "Market shows slight bearish sentiment with moderate volatility.",
    }
    monkeypatch.setattr(
        "src.agents.l1_macro_analyst.requests.post",
        _fake_post_returning(json.dumps(real_shaped_payload)),
    )
    agent = L1MacroAnalyst(refresh_interval_s=45)
    ctx = agent.maybe_refresh({"return_1h_pct": 0.001})

    assert ctx.regime == "neutral"
    assert ctx.risk_score == pytest.approx(-0.1)
    assert ctx.confidence == pytest.approx(0.75)
    assert ctx.urgency_multiplier == pytest.approx(1.0)
    assert ctx.rationale == "Market shows slight bearish sentiment with moderate volatility."
    assert -1.0 <= ctx.risk_score <= 1.0
    assert 0.0 <= ctx.confidence <= 1.0
    assert 0.5 <= ctx.urgency_multiplier <= 2.0
