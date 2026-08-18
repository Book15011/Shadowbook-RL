# LOBExecutionEnv: Hierarchical Multi-Agent Optimal Execution System
### Master Implementation Plan & Architecture Specification

**Target:** BTCUSDT Perpetual Futures — tick-level Limit Order Book execution. **Cross-venue in
practice:** L1's aggregate context comes from Binance, L2/L3's order book comes from Bybit — see
§2.1.1 for why, added after the original single-venue plan hit a real infrastructure blocker.
**Scope:** Portfolio-grade system for quant execution / market-making roles (Jane Street, Citadel, Optiver-tier bar)
**Stack:** Python 3.11, Gymnasium, Stable-Baselines3 + sb3-contrib, LangGraph, Ollama (local LLM), RTX 4090

---

## How to read this document

This is a build spec, not a tutorial. Every section gives you the exact schema, formula, or code
skeleton needed to start writing tests against an interface — not a description of what an execution
system "generally" looks like. Code blocks are reference-grade scaffolding (correct interfaces, correct
math, realistic parameter choices) meant to be filled in and unit-tested, not copy-paste-and-ship.

A note on scope honesty, because it will come up in interviews: this is a **research/portfolio-grade**
simulator, not a venue-connected production system. The two places where you should expect the hardest
questions and should be ready to defend design tradeoffs are (1) the fidelity of the matching engine's
queue model (Section 2.4) and (2) whether the LLM macro layer earns its latency cost (Section 4.4,
ablation study). Build the project so you can answer both with numbers, not intuition.

---

## Section 1: System Architecture & Data Flow

### 1.1 Architecture Diagram

```
┌──────────────────────────────────┐      ┌──────────────────────────────────┐
│ data.binance.vision               │      │ Bybit historical L2               │
│ futures/um/daily/{trades,         │      │ quote-saver.bycsi.com             │
│ aggTrades,bookDepth}              │      │ -> primary L2 source (§2.1.1)     │
│ -> L1 aggregates only (§2.1.1)    │      └──────────────┬─────────────────────┘
└──────────────┬─────────────────────┘                    │
               │ raw .zip / .parquet                       │ raw .zip
               ▼                                           ▼
┌───────────────────────────────────────┐
│  Feature Pipeline (src/data/features.py)│
│  Rebuilds L2 book state, computes:       │
│  mid/micro price, OBI(1,5,10), spread,   │
│  cancel/add rate, queue estimates        │
└──────────────────┬──────────────────────┘
                    │ FeatureFrame (per-tick, Arrow/Parquet)
                    ▼
┌────────────────────────────────────────────┐        ┌─────────────────────────────┐
│  L1 — Macro Analyst (LLM, async, ~30-60s)    │◄──────►│  Ollama (localhost:11434)    │
│  Rolling window of OBI/vol/funding + news    │        │  deepseek-coder:33b or       │
│  text → structured risk JSON (§1.2)          │        │  qwen2.5:32b-instruct-q4     │
└──────────────────┬───────────────────────────┘        └─────────────────────────────┘
                    │ risk_context (JSON, cached, TTL-bound)
                    ▼
┌────────────────────────────────────────────┐
│  L2 — Strategist (SAC, decision every 1-10s) │
│  Modulates TWAP baseline: participation_rate,│
│  urgency ∈ [0,1] — see §3.2                  │
└──────────────────┬───────────────────────────┘
                    │ child_slice (target_qty, urgency)
                    ▼
┌────────────────────────────────────────────┐
│  L3 — Executioner (RecurrentPPO, per-tick)   │
│  order_type / price_offset_ticks /           │
│  size_fraction — see §3.2                    │
└──────────────────┬───────────────────────────┘
                    │ OrderIntent
                    ▼
┌────────────────────────────────────────────┐
│  LOBExecutionEnv-v0 (matching_engine.py)     │
│  Queue-position-aware fill simulation         │
└──────────────────┬───────────────────────────┘
                    │ fill_events + updated book state
                    ▼
┌────────────────────────────────────────────┐
│  Metric Tracker (src/metrics/)               │
│  IS, VWAP-Δ, mark-out, fill-rate, order life  │
└────────────────────────────────────────────┘
```

Orchestration of the three agent tiers is a **frequency-decoupled control loop**, not a single
`env.step()` call — this is the single most important architectural fact about the system and drives
almost every implementation decision in Sections 3–4. L1 runs on a ~30-60s cadence (LLM latency on a
4090 for a 14B-32B quantized model is 200ms-2s per call — too slow to sit on the tick hot path). L2 runs
on a 1-10s cadence (child-order slicing decisions). L3 runs on every LOB update (potentially sub-100ms
during high-message-rate periods). The orchestrator (§4.3) is responsible for stitching these three
clocks together and caching each tier's last output for the tiers below it to consume.

### 1.2 L1 Macro Analyst — JSON Schema

The LLM is never allowed to emit free text into the trading path. Constrain decoding with Ollama's
`format: "json"` parameter (or a JSON-grammar via `outlines`/`llama.cpp` grammars if you need stricter
guarantees), and validate every response against this schema with `pydantic` before it touches L2's
observation vector — malformed or out-of-range LLM output should fail closed to a neutral cached value,
never propagate.

```json
{
  "type": "object",
  "properties": {
    "timestamp_ms":       { "type": "integer" },
    "regime": {
      "type": "string",
      "enum": ["risk_on", "risk_off", "neutral", "high_volatility"]
    },
    "risk_score":         { "type": "number", "minimum": -1.0, "maximum": 1.0 },
    "confidence":         { "type": "number", "minimum": 0.0,  "maximum": 1.0 },
    "urgency_multiplier": { "type": "number", "minimum": 0.5,  "maximum": 2.0 },
    "rationale":          { "type": "string", "maxLength": 280 }
  },
  "required": ["timestamp_ms", "regime", "risk_score", "confidence", "urgency_multiplier"]
}
```

`risk_score` is a signed scalar (negative = favor patience/passivity, positive = favor urgency) that L2
folds directly into its observation vector and into the inventory-risk penalty coefficient (§3.3).
`urgency_multiplier` is a direct, bounded multiplier applied to L2's participation-rate target — bounding
it to `[0.5, 2.0]` is a deliberate safety rail so a degenerate LLM output can never force the schedule to
zero or to an unbounded liquidation rate.

```python
# src/agents/l1_macro_analyst.py
import time, json, requests
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
    def __init__(self, model="qwen2.5:32b-instruct-q4_K_M", host="http://localhost:11434",
                 refresh_interval_s=45, timeout_s=5.0):
        self.model, self.host = model, host
        self.refresh_interval_s = refresh_interval_s
        self.timeout_s = timeout_s
        self._cache = self._neutral_default()
        self._last_fetch = 0.0

    def _neutral_default(self) -> MacroRiskContext:
        return MacroRiskContext(timestamp_ms=int(time.time() * 1000), regime="neutral",
                                 risk_score=0.0, confidence=0.0, urgency_multiplier=1.0,
                                 rationale="fallback: no LLM signal yet")

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
```

Call this from a **background thread or a separate process**, not inline in the tick loop — the
orchestrator (§4.3) reads `self._cache` on every L2 decision, and `maybe_refresh` self-throttles so the
Ollama call only actually fires once per `refresh_interval_s`.

---

## Section 2: Data Pipeline & L2 Feature Engineering

### 2.1 Data source reality check (read this before writing the downloader)

`data.binance.vision` for `futures/um/BTCUSDT` gives you three relevant series — be precise about what
each one actually contains, because this determines what your simulator can and cannot claim to
reproduce:

| Dataset | Path | Granularity | What it actually is |
|---|---|---|---|
| `trades` | `futures/um/daily/trades/BTCUSDT/` | tick | Every executed trade: `id, price, qty, quoteQty, time, isBuyerMaker` |
| `aggTrades` | `futures/um/daily/aggTrades/BTCUSDT/` | tick (aggregated) | Same-price consecutive trades merged: `aggTradeId, price, qty, firstTradeId, lastTradeId, timestamp, isBuyerMaker` |
| `bookDepth` | `futures/um/daily/bookDepth/BTCUSDT/` | **~1000ms snapshots** | **Percentage-bucketed** depth: `timestamp, percentage, depth, notional` — depth aggregated into bands (e.g. 0-0.25%, 0.25-0.5% from mid), **not raw per-level L2**, and **not a diff stream** |

The important consequence: **`bookDepth` is not full-resolution L2 order book data.** It's a coarse,
already-aggregated snapshot. There is no historical, exchange-hosted archive of the raw incremental
depth-diff stream (`@depth`) for USD-M futures on `data.binance.vision`. If your reward function and
queue-position model (§2.4) genuinely need per-level, per-update L2 (which they do, given the spec), you
have three honest options, and your writeup/interview answer should name which one you picked and why:

1. **Capture your own going forward.** Run a WebSocket client against `wss://fstream.binance.com/stream?streams=btcusdt@depth@100ms` continuously, seeded by a REST snapshot (`GET /fapi/v1/depth?symbol=BTCUSDT&limit=1000`), and persist every diff to Parquet. This is the only way to get true incremental L2 for a period of your choosing — Binance's own [local order book guide](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/websocket-market-streams) specifies buffering diffs after the snapshot, discarding any event with `u < lastUpdateId`, and applying events in order while checking `pu == previous_event.u`. Budget 1-2 weeks of passive capture before you have anything to train on.
2. **Buy it.** Vendors (Tardis.dev, CryptoTick, Databento, Amberdata) sell historical L2 tick/diff archives for Binance futures at reasonable cost for a portfolio project — this is what most prop shops actually do rather than reconstructing from scratch, and citing it shows you know the industry-standard path.
3. **Degrade gracefully and say so.** Approximate L2 from `bookDepth`'s percentage bands + `aggTrades` order-flow, accept that queue-position modeling becomes an *estimate of an estimate*, and explicitly document the approximation error this introduces into the reward signal. This is a legitimate portfolio-scope tradeoff **as long as you state it**, not stumble into it.

The rest of this document originally assumed option (1) or (2) so that `src/data/features.py` operates
on a true per-level order book reconstruction. `src/data/l2_capture_daemon.py` below implements option
(1). **What actually happened in practice, superseding that assumption, is documented in §2.1.1 below —
read that before treating this section as the current data source.**

#### 2.1.1 UPDATE — what actually happened, and the current real data source

Option (1) was attempted first and hit a real infrastructure blocker, not a code bug: the dev
environment's network path routes through mainland China, which silently blocks Binance's *live
trading* API domains (`fapi.binance.com`, `fstream.binance.com`) via SNI-based filtering — confirmed
directly via `curl -v`/`nc`/IP geolocation, not assumed. `data.binance.vision` (the static archive)
stays reachable throughout; only the live REST/WebSocket endpoints are blocked. A same-country VPN
exit was also ruled out (Thailand independently blocks direct Binance.com access under local
regulation) before concluding this needed an infrastructure workaround, not a retry.

Rather than stand up a second always-on server just to keep pursuing live capture, a fourth option —
not in the original three above — was investigated and adopted: **a venue's own official historical
archive of already-captured L2**, as opposed to capturing it live yourself going forward. Bybit
publishes exactly this, for free, no account required, at `quote-saver.bycsi.com` — genuine
snapshot + incremental-delta + sequence-number order book data, 500 levels per side, confirmed via
real trial download to be genuinely ~100ms cadence (verified from actual consecutive-timestamp
diffs, not the vendor's marketing claim). This sidesteps the entire "historical L2 doesn't exist"
problem for BTCUSDT on Bybit's linear-perpetual venue, in a way no option above achieves for Binance
specifically.

**Current real data source, adopted as the project's actual design, not a hypothetical:**
- **L2 order book (feeds L2/L3's observation space and reward function): Bybit**, bulk-downloaded
  from its official historical archive, truncated from the published 500 levels down to the top 20
  (§3.1's `book_depth_norm` only needs 20), reusing this section's `L2Book`/`apply_diff` reconstruction
  logic rather than reimplementing it. Byte-budget-targeted (not a fixed day count) — walk backward
  from the most recent available day until a storage ceiling is hit, since 500-level 100ms data is
  genuinely large (~1.2GB/day uncompressed at full depth).
- **L1 aggregate context (feeds the Macro Analyst's `feature_summary`, §1.2): Binance**, via
  `data.binance.vision`'s `klines` (1-minute), `fundingRate` (monthly), and `metrics` (daily, contains
  open interest) archives — **not** the tick-level `trades`/`aggTrades` tables described below. L1 was
  originally scoped around those tick tables; in practice they're unnecessary and oversized for what
  L1 actually consumes (a handful of rolling summary statistics, not tick data) — full 5-year coverage
  of the three aggregate archives totals under 300MB, versus an estimated 70-180GB for 5 years of tick
  trades.
- `trades`/`aggTrades`/`bookDepth` (this section, below) remain accurate documentation of what those
  archives *are*, and stay useful for anyone who does want tick-level trade context — they're simply
  not what L1 or L2/L3 actually consume in the current design.
- `src/data/l2_capture_daemon.py` (below) is kept as a **secondary/optional** capability — e.g. live
  augmentation on top of the Bybit archive, or a fallback if network access to Binance's live API is
  restored later — not the primary L2 path.

**This is a deliberate cross-venue design** (L1's context from Binance, L2/L3's order book from
Bybit) and should be stated explicitly as such in any writeup — not silently blended as if both came
from the same exchange. If a fully single-venue system is ever required, the honest path is re-pulling
L1's aggregate context from Bybit's own equivalent endpoints instead, not pretending the venues match.

### 2.2 Data ingestion pipeline

```python
# src/data/download_manager.py
import hashlib, io, zipfile, requests, pandas as pd
from pathlib import Path
from datetime import date, timedelta

BASE_URL = "https://data.binance.vision/data/futures/um/daily"

COLUMNS = {
    "trades":    ["id", "price", "qty", "quote_qty", "time", "is_buyer_maker"],
    "aggTrades": ["agg_id", "price", "qty", "first_id", "last_id", "time", "is_buyer_maker"],
    "bookDepth": ["timestamp", "percentage", "depth", "notional"],
}

def _url(dataset: str, symbol: str, d: date) -> str:
    return f"{BASE_URL}/{dataset}/{symbol}/{symbol}-{dataset}-{d.isoformat()}.zip"

def download_and_verify(dataset: str, symbol: str, d: date, out_dir: Path) -> Path | None:
    url = _url(dataset, symbol, d)
    csum_resp = requests.get(url + ".CHECKSUM", timeout=10)
    if csum_resp.status_code != 200:
        return None  # day not published yet / doesn't exist
    expected_sha = csum_resp.text.split()[0]

    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    actual_sha = hashlib.sha256(resp.content).hexdigest()
    if actual_sha != expected_sha:
        raise ValueError(f"Checksum mismatch for {url}")

    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extractall(out_dir)
    return out_dir / f"{symbol}-{dataset}-{d.isoformat()}.csv"

def bulk_download(symbol: str, dataset: str, start: date, end: date, out_dir: Path) -> pd.DataFrame:
    frames = []
    d = start
    while d <= end:
        path = download_and_verify(dataset, symbol, d, out_dir)
        if path is not None:
            df = pd.read_csv(path, names=COLUMNS[dataset], header=0)
            df["date"] = d.isoformat()
            frames.append(df)
        d += timedelta(days=1)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS[dataset])
```

```python
# src/data/l2_capture_daemon.py — SECONDARY/OPTIONAL path; see §2.1.1 — the project's actual
# primary L2 source is Bybit's historical archive, reusing L2Book/apply_diff below rather than
# this file's REST/WebSocket orchestration, which requires live Binance API access this
# environment doesn't currently have. Kept for potential future live augmentation.
import asyncio, json, time, requests, websockets
import pyarrow as pa, pyarrow.parquet as pq
from collections import deque

class L2CaptureDaemon:
    """
    Seeds from a REST snapshot, then applies buffered @depth@100ms diffs per Binance's
    documented local-book procedure. Persists every applied diff event to Parquet in
    rolling 15-minute shards. Run this as a long-lived background process.
    """
    def __init__(self, symbol="BTCUSDT", out_dir="data/raw_l2", ws_url=None):
        self.symbol = symbol.upper()
        self.out_dir = out_dir
        self.ws_url = ws_url or f"wss://fstream.binance.com/stream?streams={symbol.lower()}@depth@100ms"
        self.buffer, self.book_bids, self.book_asks = deque(), {}, {}
        self.last_update_id = None

    def _rest_snapshot(self):
        r = requests.get("https://fapi.binance.com/fapi/v1/depth",
                          params={"symbol": self.symbol, "limit": 1000}, timeout=10).json()
        self.last_update_id = r["lastUpdateId"]
        self.book_bids = {float(p): float(q) for p, q in r["bids"]}
        self.book_asks = {float(p): float(q) for p, q in r["asks"]}

    def _apply(self, ev: dict) -> bool:
        # ev: {"U": first_update_id, "u": final_update_id, "pu": prev_final_update_id, "b": [...], "a": [...]}
        if ev["u"] < self.last_update_id:
            return False
        if ev["pu"] != self.last_update_id:
            raise RuntimeError("gap detected — resync required (re-fetch snapshot)")
        for p, q in ev["b"]:
            p, q = float(p), float(q)
            (self.book_bids.pop(p, None) if q == 0 else self.book_bids.__setitem__(p, q))
        for p, q in ev["a"]:
            p, q = float(p), float(q)
            (self.book_asks.pop(p, None) if q == 0 else self.book_asks.__setitem__(p, q))
        self.last_update_id = ev["u"]
        return True

    async def run(self):
        self._rest_snapshot()
        writer, rows, shard_start = None, [], time.time()
        async with websockets.connect(self.ws_url) as ws:
            async for raw in ws:
                ev = json.loads(raw)["data"]
                if ev["u"] <= self.last_update_id:
                    continue
                ok = self._apply(ev)
                if not ok:
                    continue
                rows.append({"ts_ms": ev["E"], "u": ev["u"],
                             "bids": json.dumps(ev["b"]), "asks": json.dumps(ev["a"])})
                if time.time() - shard_start > 900:  # 15-minute shards
                    pq.write_table(pa.Table.from_pylist(rows),
                                    f"{self.out_dir}/{self.symbol}_{int(shard_start)}.parquet")
                    rows, shard_start = [], time.time()
```

### 2.3 Order book feature engineering

All features are computed per tick (per applied L2 diff) from the reconstructed book, top-`N=20` levels
retained. Let $b_i, q^b_i$ denote the $i$-th bid price/qty (best-to-worst) and $a_i, q^a_i$ the ask side.

**Mid-price and spread**
$$
P_{mid} = \frac{b_1 + a_1}{2}, \qquad S = a_1 - b_1
$$

**Micro-price** (volume-weighted top-of-book — a materially better short-horizon fair-value estimate than
naive mid-price, since it leans toward the side with less resting size, i.e. the side likelier to move):
$$
P_{micro} = \frac{q^a_1 \cdot b_1 + q^b_1 \cdot a_1}{q^b_1 + q^a_1}
$$

**Order Book Imbalance at depth $k$**, for $k \in \{1, 5, 10\}$:
$$
OBI_k = \frac{\sum_{i=1}^{k} q^b_i - \sum_{i=1}^{k} q^a_i}{\sum_{i=1}^{k} q^b_i + \sum_{i=1}^{k} q^a_i} \in [-1, 1]
$$

**Depth invalidation rate / cancel volume** over a rolling window $[t-w, t]$ — the ratio of *canceled*
resting size to *added* resting size, computed separately per side from the stream of diff events (a
level going from $q>0 \to q=0$ without an intervening trade at that price is a cancel; an increase in $q$
at an existing or new level is an add):

$$
CAR_{side}(t, w) = \frac{\sum_{\text{cancels in } [t-w,t]} |\Delta q|}{\sum_{\text{adds in } [t-w,t]} |\Delta q| + \epsilon}
$$

A high $CAR$ on one side is a leading indicator of imminent price movement away from that side (resting
liquidity is being pulled, not consumed) — this is one of the stronger predictive OBI-adjacent features
in practice and worth logging separately from raw OBI.

```python
# src/data/features.py
import numpy as np

TICK_SIZE = 0.10  # BTCUSDT perp tick size — confirm against exchangeInfo at runtime, don't hardcode in prod

def mid_price(bids: np.ndarray, asks: np.ndarray) -> float:
    return (bids[0, 0] + asks[0, 0]) / 2.0

def micro_price(bids: np.ndarray, asks: np.ndarray) -> float:
    qb, qa = bids[0, 1], asks[0, 1]
    return (qa * bids[0, 0] + qb * asks[0, 0]) / (qb + qa + 1e-12)

def obi(bids: np.ndarray, asks: np.ndarray, k: int) -> float:
    bid_vol, ask_vol = bids[:k, 1].sum(), asks[:k, 1].sum()
    return float((bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-12))

class CancelAddTracker:
    """Stateful, call update() once per applied diff event per side."""
    def __init__(self, window_s: float = 5.0):
        self.window_s = window_s
        self.events = []  # (ts, side, kind, abs_delta_qty)

    def update(self, ts: float, side: str, delta_qty: float, was_trade: bool):
        kind = "cancel" if (delta_qty < 0 and not was_trade) else ("add" if delta_qty > 0 else None)
        if kind:
            self.events.append((ts, side, kind, abs(delta_qty)))
        cutoff = ts - self.window_s
        self.events = [e for e in self.events if e[0] >= cutoff]

    def ratio(self, side: str) -> float:
        cancels = sum(e[3] for e in self.events if e[1] == side and e[2] == "cancel")
        adds = sum(e[3] for e in self.events if e[1] == side and e[2] == "add")
        return cancels / (adds + 1e-9)
```

### 2.4 Queue position & wait-time estimation

Binance does not expose your order's true queue position — this must be modeled. The standard, defensible
approach (used in the Cont-Kukanov-Stoikov-style microstructure literature) is:

1. On order placement at price level $p$, record the **visible resting volume ahead of you**:
   $Q_{ahead}(0) = q_p^{\text{visible}}$ at the instant of placement (you are assumed to queue behind all
   currently-resting volume at that price — standard price-time priority assumption).
2. On every subsequent tick, decrement $Q_{ahead}$ by the **trade volume** executed at that price level
   (trades consume queue from the front) and **partially** by canceled volume, weighted by an assumed
   uniform-random cancel position:
$$
Q_{ahead}(t+1) = \max\left(0,\; Q_{ahead}(t) - V_{trade}(t) - V_{cancel}(t)\cdot \frac{Q_{ahead}(t)}{q_p(t)}\right)
$$
3. **Expected wait time** to fill uses the level's realized trade-through rate over a trailing window:
$$
\hat{T}_{fill} = \frac{Q_{ahead}(t)}{\bar{v}_{trade,p}(w)}, \qquad \bar{v}_{trade,p}(w) = \frac{1}{w}\sum_{\text{trades at } p \text{ in } [t-w,t]} \text{qty}
$$

This is a biased estimator (assumes uniform-random cancellation position, ignores order-type heterogeneity)
— acceptable for a training-signal feature, but you should validate it against realized fill outcomes in
your matching engine (Section 3) and report the estimator's calibration error in your writeup. That
calibration check is itself a good thing to show off: it demonstrates you understand your own model's
limitations rather than treating a heuristic as ground truth.

### 2.5 Train / Validation / Test Split (chronological — mandatory)

Splits must be **chronological, never randomly shuffled across dates** — train on the oldest
days, validate on the next-oldest block, test on the most recent block. Random shuffling
across dates leaks lookahead information (a policy trained partly on days *after* its
validation window is not being validated honestly) and silently invalidates every IS/mark-out
number reported downstream, including §4.4's ablation.

**Why held-out windows go on the recent end, not the old end:** the L2 backfill (§2.1.1) walks
*backward* from the most recent available day toward the past. The newest portion of the
dataset is therefore already final and will never be touched by future backfill runs — only
the older boundary keeps moving. Pinning val/test to the recent, stable end and letting train
absorb everything older means the split never needs to be recomputed as backfill continues;
train simply grows for free.

**Boundary heuristic** (exact day counts must be chosen against the real date coverage on
disk, not hardcoded blindly from this doc — check gap density in the proposed windows before
finalizing):

| Split | Window | Notes |
|---|---|---|
| test | most recent ~15-20 calendar days actually present | held out entirely; never touched until final Phase 5 backtest |
| val | next ~15-20 days back from test's boundary | used for Phase 3/4 model selection / early stopping |
| train | everything older | grows automatically as backfill adds older days |

If known coverage gaps (§2.1.1) fall inside the proposed val/test windows, either shift the
boundary to avoid concentrating gaps in the held-out set, or explicitly document the resulting
gap count in val/test as a known limitation — don't silently absorb it.

**Persisted artifact** — a list of dates per split, not just min/max boundaries, since gaps
mean a date range alone doesn't fully specify membership:

```json
{
  "generated_at": "2026-08-15T00:00:00Z",
  "source_day_count": 296,
  "train_dates": ["2024-01-01", "2024-01-02", "..."],
  "val_dates": ["2025-07-15", "..."],
  "test_dates": ["2025-08-01", "..."],
  "known_gap_dates": ["2024-08-12", "..."]
}
```

**Consumption pattern** — all training/val/test/ablation code reads from this single artifact,
never recomputes its own boundary:

```python
# src/data/split.py
def load_split(name: str) -> list[date]:
    """name in {'train', 'val', 'test'}. Reads the persisted split artifact;
    raises if the artifact doesn't exist yet (must be generated once, explicitly,
    not implicitly on first use)."""
```

`LOBExecutionEnv`'s existing `date_range` constructor param (added for cross-run seed
reproducibility, see the fix landed on master) may need widening from a `(start, end)` tuple to
accept an explicit date list, since a contiguous range can't represent a gapped split. This
artifact is also the mechanism that makes §4.4's requirement — "identical held-out backtest
windows" for the L1 on/off ablation — actually satisfiable: both runs load the same `test`
split from disk rather than each independently sampling and risking drift.

This artifact should not need regenerating as backfill continues, since backfill only appends
days *older* than train's boundary. If val or test's underlying files ever change after
generation, that's a signal something unexpected happened — flag it, don't silently
regenerate and move on.

---

## Section 3: Custom Gymnasium Environment — `LOBExecutionEnv-v0`

### 3.1 Observation space

A single flat `Box(shape=(42,), dtype=np.float32)` shared (with different active subsets) by L2 and L3 —
keeping one schema across both agents means the feature pipeline has one source of truth and simplifies
the LangGraph state object in §4.3. All continuous features are z-scored against a trailing rolling
window and clipped to `[-5, 5]` before being written into the vector; this keeps SB3's default
orthogonal-init MLP well-conditioned without a separate normalization layer, though `VecNormalize`
(§4.1) should still be used for the reward and to adapt to slow drift.

| Index | Feature | Range | Notes |
|---|---|---|---|
| 0 | `time_remaining_norm` | [0, 1] | fraction of parent-order horizon left |
| 1 | `inventory_remaining_norm` | [-1, 1] | signed by side; 1 = fully unexecuted |
| 2 | `spread_norm` | [0, 1] | spread / rolling p95 spread |
| 3 | `mid_return_1s_z` | [-5, 5] | z-scored 1s log-return |
| 4 | `mid_return_5s_z` | [-5, 5] | z-scored 5s log-return |
| 5 | `realized_vol_60s_z` | [-5, 5] | z-scored rolling realized vol |
| 6-8 | `OBI_1, OBI_5, OBI_10` | [-1, 1] | §2.3 |
| 9 | `micro_mid_dev_ticks` | [-5, 5] | (micro − mid) / tick_size, clipped |
| 10-11 | `cancel_add_ratio_{bid,ask}` | [0, ∞)→clip[0,5] | §2.3 |
| 12 | `trade_flow_imbalance_5s` | [-1, 1] | (taker-buy − taker-sell) / total, 5s window |
| 13 | `queue_position_ratio` | [-1, 1] | −1 = no resting order; else Q_ahead / (Q_ahead + own_qty) |
| 14 | `ticks_since_own_fill_norm` | [0, 1] | clipped/normalized |
| 15 | `l2_target_slice_ratio` | [0, 1] | from L2, passed down to L3 |
| 16 | `l2_urgency` | [0, 1] | from L2, passed down to L3 |
| 17 | `l1_risk_score` | [-1, 1] | from L1 cache, §1.2 |
| 18 | `l1_confidence` | [0, 1] | from L1 cache |
| 19-38 | `book_depth_norm[20]` | [-5, 5] | 10 bid + 10 ask level sizes, z-scored vs 20-level rolling mean |
| 39 | `funding_rate_z` | [-5, 5] | perp-specific |
| 40 | `taker_buy_sell_ratio_1m` | [-1, 1] | |
| 41 | `own_open_orders_norm` | [0, 1] | |

L2 (Strategist) consumes a **temporally downsampled** view of the same 42-dim vector (1s/10s aggregates
computed from the same rolling buffers — indices 15/16 are naturally excluded since L2 produces them)
rather than a separately engineered feature set, and additionally receives a **TWAP-schedule deviation**
scalar (executed-so-far vs. scheduled-so-far) that L3 does not need.

### 3.2 Action space — and an SB3 compatibility gotcha to design around up front

**Important implementation constraint:** Stable-Baselines3 supports `Dict`/`Tuple` **observation** spaces
(via `MultiInputPolicy`) but does **not** support `Dict` or `Tuple` **action** spaces for any of its
built-in algorithms. Design the action spaces as flat `MultiDiscrete` or `Box` from the start — retrofitting
this after you've built a `Dict` action space (which is the "natural" first instinct given the
order-type/offset/size structure of the problem) is a wasted afternoon.

**L3 — Executioner**, decision on every LOB update, `MultiDiscrete([4, 11, 5])`:

```python
import gymnasium as gym
l3_action_space = gym.spaces.MultiDiscrete([4, 11, 5])
# dim 0: order_type      -> {0: HOLD, 1: LIMIT, 2: MARKET, 3: CANCEL_AND_REPLACE}
# dim 1: price_offset_idx -> maps to ticks via  offset = idx - 5   (range -5..+5)
# dim 2: size_frac_idx    -> maps to fraction of L2's assigned slice: {0.2,0.4,0.6,0.8,1.0}[idx]
```

Discrete-but-partially-observable action selection under queue dynamics is exactly the case where
`RecurrentPPO` (sb3-contrib, `MlpLstmPolicy`) outperforms plain PPO — the agent needs memory of "I've
been resting at the top of the ask queue for 3 seconds and just got partially filled" that a Markovian
`MlpPolicy` cannot represent from a single-tick observation alone. Use `RecurrentPPO` for L3 (see §4.1).

**L2 — Strategist**, decision every 1-10s, continuous `Box`:

```python
l2_action_space = gym.spaces.Box(low=np.array([0.0, 0.0], dtype=np.float32),
                                  high=np.array([2.0, 1.0], dtype=np.float32))
# dim 0: participation_rate_multiplier -> scales the current TWAP-scheduled slice
#         (0 = defer/hide entirely, 1 = release exactly on-schedule, up to 2 = catch-up burst)
# dim 1: urgency -> passed into L3's observation (idx 16); 0 = maker-only bias, 1 = taker-biased
```

`SAC` is the natural fit here: continuous action, off-policy sample efficiency matters because L2's
effective episode length (decisions per parent order) is 1-2 orders of magnitude shorter than L3's, so
you have far fewer transitions per wall-clock hour of simulation to learn from.

### 3.3 Reward function

Total step reward is a weighted sum of four components, plus a terminal implementation-shortfall term.
All monetary terms are normalized by arrival mid-price and reported in basis points so the reward scale
is stable across BTC price regimes (an $80,000 BTC and a $40,000 BTC parent order should look identical
to the reward function after normalization).

**1. Realized slippage penalty** (paid on every fill, this step):
$$
r_{slip} = -\alpha \cdot \text{side} \cdot \frac{P_{fill} - P_{arrival}}{P_{arrival}} \cdot \frac{q_{fill}}{Q_{total}} \times 10^4
$$
where $\text{side}=+1$ for buy, $-1$ for sell, so a buy fill above arrival price (or a sell fill below it)
is always penalized regardless of direction.

**2. Inventory risk / holding penalty** (Almgren-Chriss-style quadratic holding cost, paid every step
regardless of fills, scaled by the L1 risk context):
$$
r_{inv} = -\lambda \cdot (1 + \max(0, \text{l1\_risk\_score})) \cdot \left(\frac{Q_{remaining}}{Q_{total}}\right)^2 \cdot \Delta t
$$
The $(1+\max(0,\text{risk\_score}))$ term is precisely how the L1 macro signal is wired into the reward,
not just the observation — a positive (risk-off) macro read makes carrying inventory more expensive,
pushing L2/L3 toward faster completion during turbulent regimes. This is the mechanism you'll want to be
able to explain and ablate cleanly in §4.4.

**3. Queue invalidation / unfilled order penalty** (paid when a resting order is canceled unfilled, or
expires at episode boundary):
$$
r_{queue} = -\beta \cdot \mathbb{1}[\text{canceled unfilled}] - \gamma \cdot \frac{Q_{ahead}}{q_p}\Big|_{\text{at cancel}}
$$
The second term specifically penalizes canceling *late* in a queue you'd already waited through most of
— it's cheap to cancel an order you just placed, expensive (in wasted queue-priority) to cancel one
you've been waiting on for a while, and the penalty should reflect that asymmetry.

**4. Spread capture bonus** (paid on maker fills only):
$$
r_{spread} = +\delta \cdot \text{side} \cdot \frac{P_{mid} - P_{fill}}{P_{mid}} \cdot \frac{q_{fill}}{Q_{total}} \times 10^4 \cdot \mathbb{1}[\text{maker fill}]
$$

**Terminal reward** (paid once, at episode end — end-of-episode implementation shortfall, see §5.1 for
the full IS decomposition; any residual unfilled quantity is marked at the terminal market price as a
forced liquidation, which is what makes leaving inventory unexecuted strictly costly and prevents the
agent from learning a degenerate "never trade" policy):
$$
r_{terminal} = -\kappa \cdot IS_{bps}
$$

```python
# src/envs/reward.py
from dataclasses import dataclass

@dataclass
class RewardWeights:
    alpha: float = 1.0     # slippage
    lam: float = 0.02      # inventory holding
    beta: float = 0.5      # unfilled cancel penalty (flat)
    gamma: float = 0.3     # unfilled cancel penalty (queue-position-weighted)
    delta: float = 0.8     # spread capture bonus
    kappa: float = 1.0     # terminal IS

def step_reward(w: RewardWeights, *, side: int, fills: list[dict], arrival_price: float,
                 mid_price: float, qty_remaining: float, qty_total: float, dt: float,
                 l1_risk_score: float, canceled_unfilled: bool,
                 queue_ahead_at_cancel: float | None, queue_at_level: float | None) -> float:
    r_slip = 0.0
    r_spread = 0.0
    for f in fills:
        r_slip += -w.alpha * side * (f["price"] - arrival_price) / arrival_price * (f["qty"] / qty_total) * 1e4
        if f.get("is_maker"):
            r_spread += w.delta * side * (mid_price - f["price"]) / mid_price * (f["qty"] / qty_total) * 1e4

    r_inv = -w.lam * (1 + max(0.0, l1_risk_score)) * (qty_remaining / qty_total) ** 2 * dt

    r_queue = 0.0
    if canceled_unfilled:
        r_queue -= w.beta
        if queue_ahead_at_cancel is not None and queue_at_level:
            r_queue -= w.gamma * (queue_ahead_at_cancel / queue_at_level)

    return r_slip + r_inv + r_queue + r_spread
```

### 3.4 Episode structure

One episode = one **parent order** (e.g., liquidate 50 BTC over a 30-minute horizon). `reset()` samples
a random historical window from the held-out training days (§2.5), a random parent order size (log-uniform over
a configured range to force the policy to generalize across order sizes relative to typical book depth —
report this ratio, e.g. "parent order = 3x median 30min traded volume," as a scenario-difficulty metric
in your eval suite), and a random side. `step()` advances one LOB tick, applies the current tier's action
if that tier is "on the clock" this tick (§4.3 frequency gating), runs the matching engine, and returns
`(obs, reward, terminated, truncated, info)` with `info` carrying the full fill/queue/metric breakdown
needed by the Metric Tracker (§5) — don't discard this into a scalar reward and reconstruct it later from
logs; that's a lossy round-trip you'll regret during the eval-writing phase.

---

## Section 4: Multi-Agent Training Pipeline

### 4.1 Stable-Baselines3 / sb3-contrib configuration

**L3 — Executioner (RecurrentPPO, sb3-contrib)**

```python
# src/train/train_l3.py
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecMonitor
from src.envs.lob_execution_env import LOBExecutionEnv

def make_env(rank: int, l2_stub_schedule):
    def _init():
        return LOBExecutionEnv(tier="l3", l2_override=l2_stub_schedule, seed=rank)
    return _init

if __name__ == "__main__":
    N_ENVS = 8
    vec_env = SubprocVecEnv([make_env(i, l2_stub_schedule="fixed_twap") for i in range(N_ENVS)])
    vec_env = VecMonitor(vec_env)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=5.0, gamma=0.995)

    model = RecurrentPPO(
        "MlpLstmPolicy", vec_env,
        n_steps=512,            # ticks per env per rollout; short horizon per parent slice
        batch_size=256,
        n_epochs=6,
        gamma=0.995,             # ~100ms ticks -> ~5s effective discount horizon of consequence
        gae_lambda=0.95,
        clip_range=0.15,
        ent_coef=0.005,          # small — action space is already fairly constrained/shaped
        learning_rate=3e-4,
        policy_kwargs=dict(lstm_hidden_size=128, n_lstm_layers=1, net_arch=dict(pi=[128], vf=[128])),
        tensorboard_log="logs/l3_ppo/",
        device="cuda",
        verbose=1,
    )
    model.learn(total_timesteps=20_000_000, progress_bar=True)
    model.save("models/l3_executioner_v1")
    vec_env.save("models/l3_vecnormalize.pkl")
```

**L2 — Strategist (SAC)** — trained with L3 **frozen** and wrapped inside the environment, so a single
L2 "step" internally rolls the frozen L3 policy forward for the number of ticks in that decision window
and returns the aggregated outcome. This is the standard practical pattern for training a two-timescale
hierarchy without the instability of joint from-scratch multi-agent RL, and it's the thing to say
explicitly in an interview if asked "why not train them jointly" — joint MARL here is non-stationary on
both sides at once and is a much harder, much slower path to a working baseline than most portfolio
timelines support.

```python
# src/envs/wrappers.py
class FrozenL3Wrapper(gym.Wrapper):
    """Wraps LOBExecutionEnv so that env.step(l2_action) internally executes N_ticks of the
    frozen L3 policy against that L2 decision, and returns the L2-cadence aggregate."""
    def __init__(self, env, l3_model, ticks_per_l2_decision: int = 50):
        super().__init__(env)
        self.l3_model, self.n_ticks = l3_model, ticks_per_l2_decision
        self.action_space = env.l2_action_space
        self.observation_space = env.l2_observation_space

    def step(self, l2_action):
        self.env.apply_l2_action(l2_action)  # sets target_slice_ratio / urgency for L3 to observe
        agg_reward, terminated, truncated, l3_obs = 0.0, False, False, self.env.get_l3_obs()
        for _ in range(self.n_ticks):
            l3_action, _ = self.l3_model.predict(l3_obs, deterministic=False)
            l3_obs, r, terminated, truncated, info = self.env.step_l3(l3_action)
            agg_reward += r
            if terminated or truncated:
                break
        return self.env.get_l2_obs(), agg_reward, terminated, truncated, self.env.l2_info()
```

```python
# src/train/train_l2.py
from stable_baselines3 import SAC
from sb3_contrib import RecurrentPPO

l3_frozen = RecurrentPPO.load("models/l3_executioner_v1")
env = FrozenL3Wrapper(LOBExecutionEnv(tier="l2"), l3_model=l3_frozen, ticks_per_l2_decision=50)

model = SAC(
    "MlpPolicy", env,
    buffer_size=500_000,
    batch_size=256,
    gamma=0.995,             # ~1-10s decisions -> longer effective horizon than L3
    tau=0.005,
    learning_rate=3e-4,
    train_freq=1,
    gradient_steps=1,
    tensorboard_log="logs/l2_sac/",
    device="cuda",
    verbose=1,
)
model.learn(total_timesteps=2_000_000, progress_bar=True)
model.save("models/l2_strategist_v1")
```

### 4.2 Why RecurrentPPO for L3 but SAC for L2

This is a design decision you should be able to defend in one sentence each: **L3 needs memory over a
partially observable queue state and benefits from PPO's stability under a fairly noisy, high-frequency
reward** (queue dynamics are Markovian only if you condition on your full order history, which the LSTM
approximates); **L2 needs sample efficiency because it gets far fewer decisions per unit of simulated
time**, and its action space is naturally continuous (a participation-rate multiplier), which is exactly
SAC's regime.

### 4.3 Orchestration with LangGraph

```python
# src/agents/orchestrator_graph.py
from typing import TypedDict
from langgraph.graph import StateGraph, END

class ExecState(TypedDict):
    tick: int
    market_obs: dict
    l1_context: dict
    l2_target: dict
    l3_action: dict
    done: bool

L1_EVERY_N_TICKS = 600   # ~60s at 100ms ticks
L2_EVERY_N_TICKS = 10    # ~1s at 100ms ticks

def macro_node(state: ExecState, l1_agent) -> ExecState:
    if state["tick"] % L1_EVERY_N_TICKS == 0:
        state["l1_context"] = l1_agent.maybe_refresh(state["market_obs"]).model_dump()
    return state

def strategist_node(state: ExecState, l2_model) -> ExecState:
    if state["tick"] % L2_EVERY_N_TICKS == 0:
        obs = build_l2_obs(state["market_obs"], state["l1_context"])
        action, _ = l2_model.predict(obs, deterministic=True)
        state["l2_target"] = {"participation_rate_mult": float(action[0]), "urgency": float(action[1])}
    return state

def executioner_node(state: ExecState, l3_model, lstm_state_holder) -> ExecState:
    obs = build_l3_obs(state["market_obs"], state["l2_target"], state["l1_context"])
    action, lstm_state_holder.state = l3_model.predict(
        obs, state=lstm_state_holder.state, episode_start=lstm_state_holder.episode_start,
        deterministic=True,
    )
    state["l3_action"] = decode_l3_action(action)
    lstm_state_holder.episode_start = False
    return state

def env_step_node(state: ExecState, env) -> ExecState:
    obs, reward, terminated, truncated, info = env.step(state["l3_action"])
    state["market_obs"] = obs
    state["done"] = terminated or truncated
    state["tick"] += 1
    return state

def build_graph(env, l1_agent, l2_model, l3_model, lstm_holder):
    g = StateGraph(ExecState)
    g.add_node("macro", lambda s: macro_node(s, l1_agent))
    g.add_node("strategist", lambda s: strategist_node(s, l2_model))
    g.add_node("executioner", lambda s: executioner_node(s, l3_model, lstm_holder))
    g.add_node("env_step", lambda s: env_step_node(s, env))
    g.set_entry_point("macro")
    g.add_edge("macro", "strategist")
    g.add_edge("strategist", "executioner")
    g.add_edge("executioner", "env_step")
    g.add_conditional_edges("env_step", lambda s: END if s["done"] else "macro")
    return g.compile()
```

For live inference/backtest replay this graph is invoked once per tick; the `% N_ticks` guards inside
`macro_node`/`strategist_node` are what implement the frequency decoupling described in §1.1 without
needing a real scheduler — this is adequate for a backtest loop but should become an actual async
scheduler (e.g. separate `asyncio` tasks per tier communicating via a shared cache object) the moment you
move this toward anything resembling live paper trading, since a real venue won't pause for your LLM call.

### 4.4 Modular build order (do not skip steps — each de-risks the next)

1. **Matching engine + metrics correctness, zero RL.** Implement `LOBExecutionEnv-v0` driven by a naive
   fixed-schedule TWAP (no learning anywhere). Validate that `Fill Rate`, `IS`, and `VWAP-Δ` (§5) compute
   sane numbers against a trivial baseline before any agent touches the loop. Bugs found here are cheap;
   bugs found after RL is in the loop look like "the policy is bad" and cost days to diagnose.
2. **Train L3 alone**, fed by the fixed TWAP schedule as a static stand-in for L2 (`l2_stub_schedule="fixed_twap"`
   above). This isolates whether the tick-level limit/market decision policy alone beats a naive
   same-price-level baseline, which is your first defensible portfolio result.
3. **Freeze L3, train L2** with `FrozenL3Wrapper` (§4.1). Compare against pure TWAP and against L3-alone
   with a fixed participation rate — this is where you show hierarchical decomposition actually buys
   something over flat TWAP.
4. **Integrate L1 as a no-op stub first** (`risk_score=0` always) purely to validate the plumbing — JSON
   schema, LangGraph wiring, observation indices 17-18 populated correctly — with zero model latency or
   nondeterminism in the loop. Only after that passes do you swap in the real Ollama calls.
5. **Ablation study, not a vibe check.** Run identical held-out backtest windows with L1 on vs. off (risk
   score forced to 0) and report the IS/mark-out delta with confidence intervals across enough windows to
   say something statistically defensible. This is the single most interview-relevant artifact in the
   whole project: a hiring manager at any of these firms will ask "did the LLM help, or is it decoration,"
   and "here's the ablation table" is a categorically stronger answer than "it seemed to help."

### 4.5 Post-L3 realism layer: stealth execution + calibrated market impact

Two additions sit strictly downstream of the core pipeline above — neither touches L3's action/observation
space, neither is required for Phases 3-5 to produce a complete, defensible result on their own. They exist
to answer a sharper version of a question any interviewer at these firms will ask: *if all of this trains
against a static historical tape, how do you know the agent actually achieves the objective — executing
without revealing the plan — rather than just looking good against a market that can't react to it?*

The honest answer has two parts, and this section is deliberately built to keep them separate rather than
conflate them:

- **Timing quality** — did the policy avoid trading into moments the market was about to move against it —
  is fully answerable from historical replay alone, no simulated reaction required. §5.3's mark-out formula
  already measures this against the real subsequent tape, independent of whether the agent's own trade
  caused anything.
- **Footprint avoidance** — would a live, adaptive counterparty learn to detect and exploit *this specific
  policy's* behavioral signature — cannot be validated from replay, full stop, at any level of fill-pricing
  realism. It requires an opponent that reacts to the agent over repeated interaction, which is what
  agent-based market simulation (ABIDES, JAX-LOB) exists to provide, and which this project has an explicit,
  documented reason to not adopt: ABIDES is CPU-bound (a separate line of GPU-accelerated research exists
  specifically because of this bottleneck, reporting up to 240x speedups over it) and ships no crypto
  calibration — its default agent populations are fit to equities, and calibrating a synthetic population to
  genuinely resemble BTC-USDT is its own multi-week research project (one benchmark reported ~155 CPU-core-
  hours to calibrate a single asset for a single day), disproportionate to this project's scope. This
  limitation is real and is explicitly out of scope here, not silently assumed away — the mark-out metric is
  the honest, achievable proxy for stealth at this project's scope, and §5.3 should be read with that caveat.

**A. Deterministic stealth wrapper (`src/agents/stealth_wrapper.py`)**

Sits between `executioner` and `env_step` in the LangGraph orchestrator (§4.3), intercepting L3's already-
decoded action and expanding a large clip into an iceberg-style schedule of smaller re-quoted peaks before
it reaches `matching_engine.py`. Stateless with respect to L3's weights — no retraining, no action-space
change, applies to any already-trained L3 checkpoint.

Randomization is anchored to real local context on three axes, not pure noise — a refresh pattern that looks
too random relative to actual market conditions is itself a kind of signature. All three anchors reuse
features the project already computes; no new data pipeline:

1. **Size** — jittered around the level's own trailing average trade size (§2.4's `v̄_trade,p(w)`, already
   computed for expected-wait-time).
2. **Depth cap** — additionally hard-capped as a fraction of *currently* visible resting size at the level
   (`book_depth_norm`, §3.1). Catches the case where a peak is reasonable on average but anomalously large
   on one particular thin tick.
3. **Pace** — inter-clip refresh timing jitters around the interval implied by L2's assigned pace (idx 15,
   `l2_target_slice_ratio`), rather than locking to a literal fixed interval. A mechanically regular refresh
   cadence is itself one of the standard iceberg-detection heuristics in the microstructure literature —
   syncing tightly to a TWAP clock would reintroduce the exact signature this layer exists to avoid.

```python
# src/agents/stealth_wrapper.py
from dataclasses import dataclass
import numpy as np

@dataclass
class StealthConfig:
    peak_frac_range: tuple[float, float] = (0.5, 1.5)     # jitter around the size anchor
    max_visible_depth_frac: float = 0.5                    # hard cap: never reveal >50% of what's resting
    pace_jitter_frac: float = 0.3                           # +/- jitter around the L2-implied refresh interval

class StealthWrapper:
    """Deterministic, stateless w.r.t. L3's weights. Expands a decoded L3 action into an
    iceberg clip schedule. Reference scaffolding -- fill in and unit test, same convention
    as matching_engine.py and reward.py."""

    def __init__(self, cfg: StealthConfig = StealthConfig()):
        self.cfg = cfg

    def expand(self, l3_action: dict, avg_trade_size_at_level: float,
               visible_depth_at_level: float, l2_pace_hint: float) -> list[dict]:
        if not l3_action.get("stealth_mode", False):
            return [l3_action]  # pass-through, identical to today's behavior

        target_qty = l3_action["size_frac"] * l3_action["l2_slice_qty"]
        clips, remaining = [], target_qty
        while remaining > 1e-9:
            size_anchor = np.random.uniform(*self.cfg.peak_frac_range) * avg_trade_size_at_level
            depth_cap = self.cfg.max_visible_depth_frac * visible_depth_at_level
            peak = min(remaining, size_anchor, depth_cap)
            clips.append({**l3_action, "size_frac": None, "qty": peak})
            remaining -= peak
            # each refreshed peak is a fresh placement event through §2.4's Q_ahead(0) reset --
            # correctly modeling that a refreshed iceberg peak loses time priority, at zero new
            # matching-engine code.
        return clips
```

`stealth_mode` is a boolean set externally (by the orchestrator, e.g. above a configurable notional
threshold) — not a dimension of L3's `MultiDiscrete([4,11,5])`. A learned version was considered and
explicitly rejected: extending L3's action space would change the policy network's output shape, which
`RecurrentPPO.load()` cannot warm-start through directly (it requires the saved model's action space to
match the environment's) — making this real model-surgery work, not a cheap continuation, for a benefit not
worth that cost at this project's scope.

**B. Calibrated market impact (Tier 1 realism, layered onto historical replay)**

Historical replay already prices real, *within-tick* cost via level-walking (§3.2/§3.4) — a large market
order pays real, increasing slippage as it consumes visible depth. What replay cannot capture is
*intertemporal* cost: the next tick's book is whatever actually happened historically, fully recovered,
regardless of how aggressively the agent traded the tick before. Two consequences, at two different
timescales, both worth naming explicitly since they affect different training phases:

- **Permanent impact** persists across the rest of the episode and is the primary signal L2's participation-
  rate decision needs — without it, there is limited reward-driven reason to prefer spreading a parent order
  over time rather than front-loading it.
- **Temporary impact** decays over a short half-life but still outlives a single tick — meaning it also bears
  on L3's own tick-level decisions (e.g. firing consecutive MARKET orders with no compounding cost), not only
  on L2's pacing.

$$\Delta_{perm} = \eta \cdot \text{side} \cdot \frac{\text{child\_qty}}{\text{typical\_volume}} \quad
  \text{(persistent shift to the replayed mid-price path for the rest of the episode)}$$

$$\Delta_{temp} = \lambda \cdot \text{side} \cdot \sqrt{\text{participation\_rate}} \quad
  \text{(decays back toward the historical path with a short half-life)}$$

**Calibration is mandatory before this is trusted, not optional:** $\eta$ and $\lambda$ must be fit via a
short historical regression (realized short-horizon return vs. signed order flow/volume) against the
project's own trade/order-flow data, not hand-picked illustrative constants. Uncalibrated constants would
make this cosmetic rather than genuinely load-bearing — inconsistent with how every other quantitative
choice in this document is expected to be justified.

**Depth replenishment** — when the agent consumes book depth, replenish it stochastically at a rate
calibrated from §2.3's `CancelAddTracker` cancel/add features, already computed. This reuses existing
feature-pipeline code and is the piece that makes counterparty behavior look reactive rather than static,
without needing any agent population.

**Sequencing: between Phase 3 and Phase 4, not folded into either.** Phase 3 (§6.2) trains L3 alone against
a *frozen* TWAP schedule — L2 isn't learning pacing yet, so Tier 1's primary payoff isn't accessible during
Phase 3, and Phase 3 should run first, unmodified, exactly as scoped: a clean, isolated read on whether
`RecurrentPPO` learns a sensible tick-level policy at all, before a second, never-yet-validated piece of new
logic (impact model + fresh calibration) enters the same environment. This project's own history argues for
that staging discipline directly — the free-market-order-fill bug, the `ref_depth` buffer-widening bug, and
the seed/window drift bug were each caught *because* something was isolated and tested before the next layer
went on top; introducing RL training and a new, uncalibrated environment change simultaneously would make a
bad Phase 3 result ambiguous between "policy problem" and "environment problem." The real cost of this
staging is a second, warm-started fine-tune run (`RecurrentPPO.load(...)` → `.learn()`) after Tier 1 lands —
genuine extra GPU time, though bounded, since it continues from a working checkpoint rather than a fresh
init. Tier 1 must be in place, calibrated, and validated *before* Phase 4's L2 training begins, since L2's
entire objective is the pacing decision Tier 1 makes meaningful.

**Considered and explicitly out of scope, with reasons on record:**

- **Reactive rule-based counterparty agents (closed-form Avellaneda-Stoikov/Cartea-Jaimungal quoting bots,
  e.g. via `mbt_gym`).** Would give real, if limited, counterparty reaction — closed-form agents avoid the
  expensive calibration-search cost that makes full ABIDES-style populations impractical here. Rejected
  anyway: proving robustness against a known, simple, closed-form adversary is not a meaningfully stronger
  claim than proving it against static replay, and isn't worth the setup cost at this project's scope.
- **A self-trained RL counterparty**, as a higher-fidelity substitute for the above. Investigated directly —
  no viable pretrained, drop-in open-source option exists for crypto LOB market-making specifically
  (available projects either ship no portable weights, requiring training from scratch anyway, or solve a
  different problem entirely, such as directional price prediction rather than LOB quoting). Training one
  from scratch is a real option but a **larger** lift than the rule-based agents above, not a lighter
  substitute for them, and would require the same freeze-and-alternate pattern already used for L2/L3
  (`FrozenL3Wrapper`) — train the counterparty against static data first, freeze it, then train the execution
  agent against the frozen counterparty — since co-training both simultaneously reintroduces the exact
  non-stationarity problem the hierarchical L1/L2/L3 decomposition (§4.2) was specifically designed to avoid.
  Noted here as a real future stretch option, structured correctly if ever pursued, not planned for this
  project's scope.
- **Full agent-based simulation (ABIDES, JAX-LOB).** See the opening of this section — CPU-bound, no crypto
  calibration, and calibrating one is its own multi-week research project. The footprint-avoidance question
  this would answer is left as a documented, disclosed limitation (see above), not silently assumed solved.

---

## Section 5: Quant Metrics & Evaluation Suite

### 5.1 Implementation Shortfall (Perold decomposition)

Let $P_0$ = arrival mid-price at decision time, side $\in \{+1,-1\}$ (buy/sell), $Q$ = total parent order
quantity, $\{(p_i, q_i)\}$ the realized fills, and $P_T$ the mid-price at the horizon end (used to mark
any unfilled residual as a forced exit / opportunity cost).

$$
P_{avg} = \frac{\sum_i p_i q_i}{\sum_i q_i}, \qquad \text{fill\_ratio} = \frac{\sum_i q_i}{Q}
$$

$$
IS_{exec,bps} = \text{side} \cdot \frac{P_{avg} - P_0}{P_0} \times 10^4 \qquad \text{(execution component)}
$$

$$
IS_{opp,bps} = (1 - \text{fill\_ratio}) \cdot \text{side} \cdot \frac{P_T - P_0}{P_0} \times 10^4 \qquad \text{(opportunity cost of the unfilled residual)}
$$

$$
IS_{total,bps} = \text{fill\_ratio} \cdot IS_{exec,bps} + IS_{opp,bps} + \text{fees}_{bps}
$$

### 5.2 VWAP / TWAP performance delta

Compare your realized average fill price against the *market's* VWAP over the same execution window
(computed from the trade tape, not your own fills):

$$
\Delta_{VWAP,bps} = \text{side} \cdot \frac{P_{avg} - VWAP_{market}}{VWAP_{market}} \times 10^4, \qquad
VWAP_{market} = \frac{\sum_j p_j^{trade} v_j^{trade}}{\sum_j v_j^{trade}}
$$

$\Delta_{TWAP,bps}$ is identical with $VWAP_{market}$ replaced by the simple time-average mid-price over
the window — report both, since VWAP-beat and TWAP-beat can diverge meaningfully in a trending market.

### 5.3 Post-trade mark-out (adverse selection)

For each individual fill at time $t_f$, price $p_f$, side $s$:
$$
MO_h = s \cdot \frac{P_{mid}(t_f + h) - p_f}{p_f} \times 10^4, \qquad h \in \{1s, 5s, 60s\}
$$
Average $MO_h$ across all fills, reported per horizon. **Negative** mark-out means the market moved
against you immediately after your fill — you got picked off, most often on maker fills that only get hit
because informed flow just arrived. A systematically negative $MO_{1s}$ on your maker fills specifically
(vs. neutral-to-positive on taker fills) is the clearest quantitative signature of adverse selection and
worth breaking out by `is_maker` in your report, not just in aggregate.

### 5.4 Fill rate & order lifetime

`fill_rate = filled_qty / total_qty` per parent order, plus an order-lifetime distribution: for every
child limit order, `lifetime = t_filled_or_canceled - t_placed`. Report the empirical distribution (not
just the mean — this is heavily right-skewed) and, if you want a genuinely portfolio-differentiating
addition, fit a Kaplan-Meier survival curve to "time to fill" censored by cancellation, which is the
standard tool for this exact question in the market-microstructure literature.

```python
# src/metrics/tracker.py
from dataclasses import dataclass, field

@dataclass
class MetricTracker:
    arrival_price: float
    side: int
    total_qty: float
    fills: list = field(default_factory=list)          # {price, qty, ts, is_maker}
    child_orders: list = field(default_factory=list)   # {placed_ts, resolved_ts, status}
    market_mid_series: list = field(default_factory=list)  # (ts, mid) for mark-out lookups

    def record_fill(self, price, qty, ts, is_maker):
        self.fills.append(dict(price=price, qty=qty, ts=ts, is_maker=is_maker))

    def implementation_shortfall_bps(self, terminal_mid: float) -> dict:
        filled_qty = sum(f["qty"] for f in self.fills)
        fill_ratio = filled_qty / self.total_qty
        if filled_qty > 0:
            p_avg = sum(f["price"] * f["qty"] for f in self.fills) / filled_qty
            is_exec = self.side * (p_avg - self.arrival_price) / self.arrival_price * 1e4
        else:
            is_exec = 0.0
        is_opp = (1 - fill_ratio) * self.side * (terminal_mid - self.arrival_price) / self.arrival_price * 1e4
        return dict(is_exec_bps=is_exec, is_opp_bps=is_opp,
                     is_total_bps=fill_ratio * is_exec + is_opp, fill_ratio=fill_ratio)

    def markout_bps(self, horizons_s=(1, 5, 60)) -> dict:
        out = {h: [] for h in horizons_s}
        for f in self.fills:
            for h in horizons_s:
                mid_h = self._mid_at(f["ts"] + h)
                if mid_h is not None:
                    out[h].append(self.side * (mid_h - f["price"]) / f["price"] * 1e4)
        return {h: (sum(v) / len(v) if v else None) for h, v in out.items()}

    def _mid_at(self, ts: float):
        # nearest-neighbor lookup into market_mid_series; replace with interpolation if needed
        candidates = [m for t, m in self.market_mid_series if t >= ts]
        return candidates[0] if candidates else None
```

---

## Section 6: Phased Developer Roadmap

### 6.1 Recommended directory structure

```
lob-execution-hma/
├── configs/
│   ├── data.yaml
│   ├── env.yaml
│   ├── ppo_l3.yaml
│   ├── sac_l2.yaml
│   └── ollama_l1.yaml
├── src/
│   ├── data/
│   │   ├── download_manager.py
│   │   ├── l2_capture_daemon.py
│   │   ├── features.py
│   │   └── dataset.py
│   ├── envs/
│   │   ├── lob_execution_env.py
│   │   ├── matching_engine.py
│   │   ├── reward.py
│   │   └── wrappers.py
│   ├── agents/
│   │   ├── l1_macro_analyst.py
│   │   ├── l2_strategist.py
│   │   ├── l3_executioner.py
│   │   └── orchestrator_graph.py
│   ├── metrics/
│   │   ├── tracker.py
│   │   ├── implementation_shortfall.py
│   │   └── markout.py
│   └── train/
│       ├── train_l3.py
│       ├── train_l2.py
│       └── evaluate.py
├── notebooks/        # exploratory only — nothing here should be load-bearing for training/eval
├── tests/
│   ├── test_matching_engine.py
│   ├── test_features.py
│   └── test_reward.py
├── models/
├── logs/
└── README.md
```

### 6.2 Phase timeline

| Phase | Focus | Key deliverable | Est. duration |
|---|---|---|---|
| **1 — Data Engine** | `download_manager.py` (Binance, L1 aggregates), Bybit L2 historical backfill (primary L2 source, §2.1.1), checksum-verified ingestion, feature pipeline (§2) | Reproducible, tested feature dataset for a fixed date range; unit tests on OBI/micro-price against hand-computed fixtures | 1-1.5 weeks — **in practice took longer**, most of it spent diagnosing a live-capture network blocker before pivoting to Bybit's archive; budget for infrastructure surprises, not just code time |
| **2 — Gym Env** | `matching_engine.py`, `LOBExecutionEnv-v0`, reward function, fixed-TWAP baseline agent | Env passes a "does nothing dumb" sanity suite: a no-op policy loses exactly the opportunity-cost IS, a perfect-foresight oracle policy (cheat: peek at future prices) achieves near-zero IS | 1.5-2 weeks |
| **3 — L3 Baseline** | `RecurrentPPO` training against fixed TWAP schedule, hyperparameter sweep, tensorboard eval curves | L3 beats naive same-level limit-order baseline on IS across held-out days, with a clean training-curve writeup | 2-3 weeks |
| **3.5 — Market Realism Layer** | Calibrated Tier 1 impact model (η/λ via historical regression, §4.5) + deterministic stealth wrapper, built and unit-tested; warm-started fine-tune of Phase 3's L3 checkpoint in the impact-aware environment | Impact model calibrated against real order-flow data (not illustrative constants); stealth wrapper passes fixture tests; fine-tuned L3 checkpoint ready for Phase 4 | 3-5 days — reuses existing feature pipeline (§2.3, §2.4), no new data collection |
| **4 — L2/L1 Integration** | `FrozenL3Wrapper`, `SAC` training for L2 (now in the Phase 3.5 impact-aware environment), LangGraph orchestrator, Ollama L1 plumbing + ablation (§4.4) | Full 3-tier system beats L3-alone; L1-on-vs-off ablation table with confidence intervals | 2-3 weeks |
| **5 — Backtest & Benchmarking** | Held-out multi-day backtest, VWAP/TWAP/IS/mark-out report, comparison table vs. industry-standard baselines (TWAP, VWAP-tracking, POV) | Final report/notebook + reproducible eval script; this is the artifact you actually walk an interviewer through | 1-2 weeks |

Total: roughly 8-12 weeks at a portfolio-project pace, front-loaded by however long you run the L2 capture
daemon (§2.1) before Phase 1 has real data to work against — start that capture running on day one, in
parallel with everything else, since it's the one dependency that can't be compressed by working harder.

### 6.3 What to have ready to defend in an interview

- **Why hierarchical decomposition over end-to-end MARL** (§4.1/4.4) — non-stationarity and sample
  efficiency, stated in one sentence with the ablation numbers to back it.
- **Why implementation shortfall as the primary objective rather than VWAP-tracking** — IS captures
  opportunity cost of *not* trading, which a pure VWAP-tracking objective structurally ignores.
- **The data-fidelity caveat (§2.1)** — which of the three options you chose, and what it costs your
  queue-position model's accuracy.
- **The L1 ablation result** — this is the one number every interviewer at these firms will ask for
  first, because "we added an LLM" is a claim, and the ablation table is the evidence.
