"""L1 Macro Analyst (architecture_spec.md Section 1.2).

Implements the spec's MacroRiskContext schema and L1MacroAnalyst class as
given -- same field constraints, same cache/throttle/fail-closed contract.
The LLM is never allowed to emit free text into the trading path: every
response is validated against MacroRiskContext before it can reach
env.l1_risk_score/l1_confidence (the Section 4.4 step 4 stub hooks already
landed in src/envs/lob_execution_env.py), and any malformed/out-of-range/
failed response fails closed to the last good cached context (or the
neutral default on first use), never raises into the caller's hot path.

maybe_refresh() is self-throttling and non-blocking: it returns the cached
context immediately if called again within refresh_interval_s of the last
attempt (successful or not), and only actually posts to Ollama once that
interval has elapsed. Call this from a background thread/process in the
real orchestrator, not inline on the tick loop -- see
src/agents/orchestrator_graph.py for the minimal wiring that reads
self._cache on a coarser tick cadence (Section 4.3's L1_EVERY_N_TICKS).

No real Ollama call happens anywhere in this module's own test suite
(tests/test_l1_macro_analyst.py) -- requests.post is mocked throughout.
"""
from __future__ import annotations

import json
import time

import requests
from pydantic import BaseModel, Field, ValidationError


class MacroRiskContext(BaseModel):
    timestamp_ms: int
    regime: str = Field(pattern="^(risk_on|risk_off|neutral|high_volatility)$")
    risk_score: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    urgency_multiplier: float = Field(ge=0.5, le=2.0)
    rationale: str = ""


SYSTEM_PROMPT = """You are a market-risk classifier for a BTCUSDT perpetual futures execution system.
You receive rolling numeric features (order-book imbalance, realized volatility, funding rate, recent
trade flow) and optional recent headline text. Output ONLY a JSON object matching the required schema.
Do not include markdown fences, commentary, or any text outside the JSON object."""


class L1MacroAnalyst:
    def __init__(
        self,
        model: str = "qwen2.5:32b-instruct-q4_K_M",
        host: str = "http://localhost:11434",
        refresh_interval_s: float = 45,
        timeout_s: float = 5.0,
    ) -> None:
        self.model, self.host = model, host
        self.refresh_interval_s = refresh_interval_s
        self.timeout_s = timeout_s
        self._cache = self._neutral_default()
        self._last_fetch = 0.0

    def _neutral_default(self) -> MacroRiskContext:
        return MacroRiskContext(
            timestamp_ms=int(time.time() * 1000), regime="neutral",
            risk_score=0.0, confidence=0.0, urgency_multiplier=1.0,
            rationale="fallback: no LLM signal yet",
        )

    def maybe_refresh(self, feature_summary: dict) -> MacroRiskContext:
        now = time.time()
        if now - self._last_fetch < self.refresh_interval_s:
            return self._cache  # non-blocking: hot path never waits on the LLM
        self._last_fetch = now
        try:
            resp = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "system": SYSTEM_PROMPT,
                    "prompt": json.dumps(feature_summary),
                    "format": "json",
                    "stream": False,
                    "options": {"temperature": 0.1, "num_ctx": 2048},
                },
                timeout=self.timeout_s,
            )
            payload = json.loads(resp.json()["response"])
            self._cache = MacroRiskContext(**payload)
        except (requests.RequestException, ValidationError, json.JSONDecodeError, KeyError):
            # fail closed: keep the last good context (or neutral default), never raise into the loop
            pass
        return self._cache
