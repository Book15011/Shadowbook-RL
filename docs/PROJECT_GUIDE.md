# lob-execution-hma — Project Guide (Current Status & Usage)

## Where you actually are right now

Phase 0 (bootstrap) and Phase 1 (data engine) are done. **Only `src/data/` has real logic.**
`src/envs/`, `src/agents/`, `src/metrics/`, `src/train/` are empty directories waiting on
Phase 2+. Nothing "trades" or "decides" anything yet, in any sense — L1/L2/L3 exist only as
a model choice + hyperparameter config, not as running code.

| Level | Role (per spec) | Algorithm | Decision frequency | Status |
|---|---|---|---|---|
| L1 | Macro Analyst — risk regime | Ollama LLM (qwen2.5:14b) | ~30-60s | Config + smoke test only |
| L2 | **Strategist** — TWAP schedule modulation | SAC | ~1-10s | Config only |
| L3 | **Executioner** — limit/market tick decisions | (Recurrent)PPO | every tick | Config only |

Double-check `configs/sac_l2.yaml` and `configs/ppo_l3.yaml` against this table — see the
note at the top of this conversation about a possible L2/L3 label swap.

## Directory structure with status

```
lob-execution-hma/
├── configs/                     [stub — placeholder keys only, Phase 0]
│   ├── data.yaml                 symbol/venue/paths/LOB depth
│   ├── env.yaml                  episode length, fees, latency, inventory limit
│   ├── ollama_l1.yaml             Ollama host/model/temperature/timeout
│   ├── sac_l2.yaml                SAC hyperparams (placeholder scale — see caveat below)
│   └── ppo_l3.yaml                PPO hyperparams (placeholder scale — see caveat below)
├── src/
│   ├── data/                     ✅ IMPLEMENTED (Phase 1)
│   │   ├── download_manager.py
│   │   ├── features.py
│   │   └── l2_capture_daemon.py
│   ├── envs/                     🔲 empty — Phase 2
│   ├── agents/                   🔲 empty — Phase 3/4
│   ├── metrics/                  🔲 empty — Phase 5
│   └── train/                    🔲 empty — Phase 3/4
├── tests/                        ✅ covers src/data/ only right now
│   ├── test_download_manager.py
│   ├── test_features.py
│   └── test_l2_capture.py
├── scripts/
│   ├── smoke_test_l1.py          ✅ validates the Ollama JSON schema call
│   └── run_l2_capture.py         ✅ CLI entrypoint for the capture daemon
├── models/                       empty — will hold SB3 .zip checkpoints later
├── logs/                         empty — tensorboard logs land here later
└── README.md
```

## Important caveat: the config numbers are placeholders, not real targets

`sac_l2.yaml` and `ppo_l3.yaml` currently show 1000 timesteps because Phase 0 explicitly
scoped config files to "placeholder keys only, no values that require real data yet." That's
enough to smoke-test that a training loop doesn't crash — it is **not** a real training
budget. When Phase 3/4 actually train these agents, PPO for L3 needs on the order of 20M
timesteps and SAC for L2 needs on the order of 2M (see Section 4.1 of the architecture spec).
Don't be surprised when those numbers jump by 3-4 orders of magnitude later — that's expected,
not scope creep.

## Setup & commands

**Install:**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .          # or: pip install -r requirements.txt
```

**Run all tests:**
```bash
pytest -v
```

**Run one test file:**
```bash
pytest tests/test_features.py -v
pytest tests/test_download_manager.py -v
pytest tests/test_l2_capture.py -v
```

**Start Ollama** (must be running before the L1 smoke test — separate terminal or background):
```bash
ollama serve
# if installed via the install script it may already run as a service:
systemctl status ollama
# or just check directly:
curl -s localhost:11434
```

**Confirm the model is present:**
```bash
ollama list
ollama pull qwen2.5:14b-instruct-q4_K_M   # only if not already pulled
```

**Run the L1 smoke test:**
```bash
python scripts/smoke_test_l1.py
```

**Start the live L2 capture daemon** (leave this running — the longer it runs before Phase 3,
the more real historical L2 you'll have to train against):
```bash
python scripts/run_l2_capture.py --symbol BTCUSDT --out-dir data/raw_l2 &

# to survive closing the terminal:
nohup python scripts/run_l2_capture.py --symbol BTCUSDT --out-dir data/raw_l2 \
  > logs/l2_capture.log 2>&1 &
```

## How each implemented script actually works

### `src/data/download_manager.py`
Pulls historical daily files from `data.binance.vision`, SHA-256 verifies them against the
published `.CHECKSUM`, and loads them into a DataFrame.

```python
from datetime import date
from pathlib import Path
from src.data.download_manager import bulk_download, download_and_verify

df = bulk_download(
    symbol="BTCUSDT",
    dataset="aggTrades",              # or "trades", "bookDepth"
    start=date(2026, 6, 1),
    end=date(2026, 6, 3),
    out_dir=Path("data/raw/aggTrades"),
)
print(df.shape, df.columns.tolist())

# single-day building block underneath bulk_download() — use directly when debugging
# one specific date (e.g. a checksum failure) rather than a whole range:
path = download_and_verify("aggTrades", "BTCUSDT", date(2026, 6, 1), Path("data/raw/aggTrades"))
```
Any day that 404s (not yet published on Binance's side, or a genuine gap) is skipped
silently rather than raised. If you need to know what actually landed, check
`df["date"].unique()` against the range you requested.

### `src/data/features.py`
Pure functions on an in-memory order book snapshot — no I/O. This is the math layer the Gym
env (Phase 2) will call on every tick, which is exactly why it's tested in isolation first.

```python
import numpy as np
from src.data.features import mid_price, micro_price, obi, CancelAddTracker

# [price, qty] rows, best-to-worst
bids = np.array([[65000.0, 1.2], [64999.5, 0.8], [64999.0, 2.1]])
asks = np.array([[65000.5, 0.9], [65001.0, 1.5], [65001.5, 1.0]])

print(mid_price(bids, asks))      # 65000.25
print(micro_price(bids, asks))    # leans toward the thinner side of the book
print(obi(bids, asks, k=1))       # top-of-book imbalance, in [-1, 1]
print(obi(bids, asks, k=3))       # deeper imbalance

tracker = CancelAddTracker(window_s=5.0)
tracker.update(ts=1000.0, side="bid", delta_qty=-0.5, was_trade=False)  # a cancel
tracker.update(ts=1001.0, side="bid", delta_qty=+0.3, was_trade=False)  # an add
print(tracker.ratio("bid"))
```
These take raw numpy arrays rather than a live book object on purpose — it decouples the math
from any stateful book implementation, which is what makes it possible to unit-test against
hand-built fixtures (exactly what `tests/test_features.py` does) with no running system
underneath.

### `src/data/l2_capture_daemon.py`
The one stateful piece in this layer — maintains a live reconstructed order book from a REST
snapshot plus streaming diffs, and persists it to Parquet.

```python
import asyncio
from src.data.l2_capture_daemon import L2CaptureDaemon

daemon = L2CaptureDaemon(symbol="BTCUSDT", out_dir="data/raw_l2")
asyncio.run(daemon.run())   # blocks forever — this is what run_l2_capture.py wraps
```
For debugging without hitting the live WebSocket, drive the lower-level methods directly:
```python
daemon._rest_snapshot()                                     # seeds book_bids / book_asks
daemon._apply({"u": 123, "pu": 122, "b": [["65000.0","1.5"]], "a": []})  # apply one diff
print(daemon.book_bids, daemon.book_asks)
```
If `_apply()` raises `RuntimeError("gap detected")`, a diff was missed (dropped message,
network hiccup) and the book must be re-seeded from a fresh REST snapshot. This is expected
to happen occasionally on any long-running capture — it's exactly why `run()` should be
supervised (systemd, a retry wrapper, a cron healthcheck) rather than left as a bare
background process that silently dies on the first gap.

## What's genuinely next

Phase 2 builds `src/envs/matching_engine.py` and `LOBExecutionEnv-v0` — the first point where
these data-layer functions get called inside a real loop instead of at a REPL. Nothing in
`src/agents/` or `src/train/` can start until the env exists and passes its own sanity checks
(Section 6.2, Phase 2 row of the master spec: a no-op policy should lose exactly the
opportunity-cost IS, and a perfect-foresight oracle should achieve near-zero IS).
