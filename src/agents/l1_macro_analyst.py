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

self.host is assumed always-local (default http://localhost:11434) --
maybe_refresh() explicitly bypasses any environment-configured HTTP(S)
proxy for this reason (see the proxies= kwarg on the requests.post() call
below); a proxied local call is never correct here and was confirmed to
silently fail closed on every real call in an environment where
http_proxy/https_proxy are set, before this fix.

STRUCTURED OUTPUT (added after docs/reports/l1_real_llm_validation.md found
0/5 real calls schema-conformant with plain format="json"): the "format"
field sent to Ollama is now the real MacroRiskContext.model_json_schema()
object, not the bare string "json". Confirmed directly, not assumed, that
this Ollama install (0.32.8) supports passing a JSON Schema object here and
that it materially changes behavior -- a live test against the exact same
model/prompt content that previously invented its own field names (risk_
level, market_risk_score, etc., see the validation report) instead returned
every one of the six real field names, correctly typed, on the first try
once format= carried the real schema. This is grammar/structure-level
enforcement (required keys present, correct JSON types), confirmed BY THE
SAME TEST to NOT extend to numeric range constraints -- that same call
returned risk_score=2, outside the documented [-1,1] range, which the
schema's own minimum/maximum keywords did not prevent. So: structured
output is used because it demonstrably fixes the field-name problem (Step
2's actual failure mode), SYSTEM_PROMPT below now also states every valid
range in prose as a second, independent line of defense (a model is more
likely to respect a range it's told about in the instructions than one
implied only by a JSON Schema keyword the decoder may not enforce), and
MacroRiskContext's own pydantic Field(ge=/le=) validation remains the
authoritative gate either way -- confirmed non-redundant by this same test,
not kept out of caution alone.

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


# Sent as Ollama's format= parameter (structured output, not plain format="json") --
# single source of truth is MacroRiskContext itself, so the schema Ollama enforces can
# never drift from the schema pydantic validates against afterward.
_MACRO_RISK_CONTEXT_SCHEMA = MacroRiskContext.model_json_schema()


SYSTEM_PROMPT = """You are a market-risk classifier for a BTCUSDT perpetual futures execution system.
You receive rolling numeric features (returns, realized volatility, funding rate, open interest,
taker trade flow) as a JSON object. Output ONLY a JSON object with EXACTLY these six fields, no
others, no markdown fences, no commentary:

- timestamp_ms (integer): echo back the input's as_of_ms value unchanged.
- regime (string): exactly one of "risk_on", "risk_off", "neutral", "high_volatility".
- risk_score (number): between -1.0 and 1.0 inclusive. Negative favors patience/passivity,
  positive favors urgency. NEVER output a value outside [-1.0, 1.0].
- confidence (number): between 0.0 and 1.0 inclusive, how confident you are in this read.
  NEVER output a value outside [0.0, 1.0].
- urgency_multiplier (number): between 0.5 and 2.0 inclusive, a direct multiplier on execution
  pace (1.0 = no adjustment). NEVER output a value outside [0.5, 2.0].
- rationale (string): one brief sentence explaining the read.

Every numeric field must respect its stated range exactly -- this is a hard constraint, not a
suggestion."""


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
                    "format": _MACRO_RISK_CONTEXT_SCHEMA,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_ctx": 2048},
                },
                timeout=self.timeout_s,
                # self.host is always local (default http://localhost:11434) --
                # bypass any HTTP(S)_PROXY the environment sets for external egress.
                # Confirmed directly: this host's ubuntu-user shell exports
                # http_proxy/https_proxy pointed at an unrelated external proxy;
                # without this override, requests silently misroutes every call
                # through it and gets a ProxyError, indistinguishable from a real
                # Ollama outage under the except clause below without checking the
                # exception message -- a real bug, not a defensive no-op.
                proxies={"http": None, "https": None},
            )
            payload = json.loads(resp.json()["response"])
            self._cache = MacroRiskContext(**payload)
        except (requests.RequestException, ValidationError, json.JSONDecodeError, KeyError):
            # fail closed: keep the last good context (or neutral default), never raise into the loop
            pass
        return self._cache
