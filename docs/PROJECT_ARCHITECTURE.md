# lob-execution-hma: Project Architecture & Developer Onboarding Guide

*Last updated: 2026-09-04. Supersedes `project_architecture_draft1.md` (2026-08-15), which is*
*kept only as a historical snapshot — see its own note at the top of that file.*

## Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [Directory Structure](#2-directory-structure-current-verified-against-the-real-repo-at-head)
3. File-by-File Breakdown
   - [4.1 `src/envs/` — The Simulator](#41-srcenvs--the-simulator-the-foundation-everything-else-stands-on)
   - [4.2 `src/data/` — Data Pipeline Library Code](#42-srcdata--data-pipeline-library-code)
   - [4.3 `src/analysis/` — Market-Impact Calibration](#43-srcanalysis--market-impact-calibration)
   - [4.4 `src/agents/` — L1 Macro Analyst + Orchestration](#44-srcagents--l1-macro-analyst--orchestration)
   - [4.5 `src/train/` — Training Entry Points](#45-srctrain--training-entry-points)
   - [4.6 `src/metrics/` — Empty Stub](#46-srcmetrics--empty-stub)
   - [4.7 `configs/` — Which YAMLs Are Real](#47-configs--which-yamls-are-real)
   - [4.8 `scripts/` — Data Acquisition](#48-scripts--data-acquisition-reusable-infrastructure)
   - [4.9 `scripts/` — Numeric-Format Conversion](#49-scripts--numeric-format-conversion-reusable--the-production-pipeline-with-its-full-prototyping-lineage-preserved)
   - [4.10 `scripts/` — Throughput & Profiling Investigations](#410-scripts--throughput--profiling-investigations-throwaway--historical-record-of-real-findings-not-maintained-tooling)
   - [4.11 `scripts/` — Evaluation & Statistical-Analysis Harnesses](#411-scripts--evaluation--statistical-analysis-harnesses)
   - [4.12 `scripts/replay_episode.py` + `scripts/analyze_predictability.py`](#412-scriptsreplay_episodepy--scriptsanalyze_predictabilitypy)
   - [4.13 `tests/`](#413-tests--the-test-suite)
4. [Connections & Dependencies](#5-connections--dependencies)
5. [Project History & Current Status](#6-project-history--current-status)
6. [Getting Started](#7-getting-started)
7. [Known Issues & Technical Debt](#8-known-issues--technical-debt)
8. [Where to Look Next](#9-where-to-look-next)

---

**Status: project complete.** This document describes the system as it stands at the end of
active development. For the *findings* (what worked, what didn't, and why), read
[`docs/reports/PROJECT_FINAL_REPORT.md`](reports/PROJECT_FINAL_REPORT.md) first — it is the
authoritative summary of results. This document is different in purpose: it is a **map of the
code**, for a developer who needs to find their way around, understand how a file connects to the
rest of the system, or extend something. Read this to learn *how the system is built*; read the
final report to learn *what was learned by building it*.

`docs/architecture_spec.md` is the *original design document*, written before implementation
began. It is still worth reading for the reasoning behind early design choices, but it describes
intent, not the as-built system — several things changed during implementation (documented inline
in this guide wherever the real code diverges from the spec). Do not edit `architecture_spec.md`;
it is preserved as a historical record of the original plan.

`docs/TRACK_STATUS.md` is the full chronological working log — every round of work, in the order
it happened, for all three tracks. This guide and the final report both distill it; TRACK_STATUS
is the raw material if you need the blow-by-blow.

---

## 1. The Big Picture

### 1.1 What this project is

A **hierarchical multi-agent reinforcement-learning system for optimal trade execution** on a
cryptocurrency limit order book (BTCUSDT perpetual futures). The task: given an order to buy or
sell some quantity of BTC within a fixed time horizon, execute it to minimize **Implementation
Shortfall** (the gap between the price you got and the price the market was at when you decided to
trade) — while competing against a simple, well-understood baseline: **TWAP** (time-weighted
average price — slice the order evenly across the horizon and execute passively).

The system is not one model. It is **three decision tiers, each running on its own clock**,
stacked so that a slower, more strategic layer sets targets for a faster, more tactical layer
underneath it:

```
L1 — Macro Analyst     (a local LLM, reads market regime,  every 600 ticks  = ~60s equivalent)
        │                emits a structured risk assessment
        ▼
L2 — Strategist         (SAC, continuous control,           every 50 ticks  = ~60 decisions/episode)
        │                paces execution against the TWAP schedule
        ▼
L3 — Executioner        (RecurrentPPO, LSTM policy,         every tick)
        │                places/cancels/replaces individual child orders
        ▼
LOBExecutionEnv         (Gymnasium env: queue-position-aware fill simulation
                          against REAL, replayed historical order-book data)
```

This is a **frequency-decoupled control loop**, not one shared `env.step()` call for all three
tiers — that single fact drives most of the implementation. L1 is far too slow (an LLM call takes
hundreds of milliseconds to a few seconds) to sit on the tick-level hot path, so it runs on its own
much slower cadence and its output is *cached* for the tiers below it to read. L2 similarly runs
far less often than L3. The orchestrator (`src/agents/orchestrator_graph.py`) is the piece
responsible for stitching these three independent clocks together correctly.

### 1.2 What each tier actually decides

| Tier | Framework | Cadence | Action | Where it's decided |
|---|---|---|---|---|
| **L1** | Local LLM via Ollama | every 600 ticks (`L1_EVERY_N_TICKS`) | A structured JSON assessment: `regime`, `risk_score` ∈ [-1,1], `confidence` ∈ [0,1], `urgency_multiplier` ∈ [0.5, 2.0] | `src/agents/l1_macro_analyst.py` |
| **L2** | SAC (`stable_baselines3`) | every 50 ticks (`L2_EVERY_N_TICKS`, ≈ 60 decisions over a 3,000-tick episode) | `[participation_rate_multiplier, urgency]` — a continuous 2-vector. `participation_rate_multiplier` scales the env's own linear-TWAP baseline (0 = defer entirely, 1 = exactly on-schedule, 2 = max catch-up burst); `urgency` is passed straight through to L3's own observation | `src/train/train_l2.py`, wired through `src/envs/wrappers.py::FrozenL3Wrapper` |
| **L3** | RecurrentPPO (`sb3_contrib`) | every tick | `MultiDiscrete([4, 11, 5])`: order type (HOLD / LIMIT / MARKET / CANCEL_AND_REPLACE) × price offset (-5..+5 ticks from the touch) × size fraction (of the current slice target, one of {0.2, 0.4, 0.6, 0.8, 1.0}) | `src/train/train_l3.py` |

Underneath all three: `src/envs/lob_execution_env.py`'s `LOBExecutionEnv` — a Gymnasium
environment that replays real, historical order-book snapshots and runs a queue-position-aware
fill simulation (`src/envs/matching_engine.py`) against them, computing Implementation Shortfall
from the resulting real fills.

### 1.3 A critical, load-bearing fact about the simulator

**`LOBExecutionEnv` has no adversarial participants and no market impact from the agent's own
orders.** Every "other participant" in the simulation is replayed historical data — a fixed
record of what really happened in the market, independent of what the agent does. The agent's own
market orders consume against a fixed historical book snapshot with no mechanism to feed forward
and change what happens on a later tick; the queue-position model's trade-volume input is derived
from real historical book-depth change, not from anything the agent's own resting order does. This
is a deliberate, necessary simplification for training against real historical data (there is no
way to simulate how the market *would have* reacted differently to the agent's presence) — but it
means **nothing evaluated in this project demonstrates real-market alpha**. A policy can be shown
to execute better than TWAP *in this simulator*, or to have a less predictable placement pattern
than TWAP *in this simulator* — never that either property would hold, or pay off, against real
counterparties who can see and react to the agent's own footprint. This caveat applies to every
result in every report in `docs/reports/`, and is worth internalizing before reading any of them.

### 1.4 Current status, in one paragraph

All three tiers are built and (individually) working. L3 was trained extensively (an original
20,000,000-step baseline, several retraining/reward-variant rounds afterward) and its best
checkpoint **ties** TWAP on execution quality — doesn't beat it, doesn't lose to it, at real
statistical power. L2 was trained three times (two reward functions, two discount factors) on top
of the frozen L3 and **never** improved on L3 alone, in any configuration — a result reached only
after methodically eliminating six separate candidate explanations (wrong reward objective, a
diverging SAC critic, insufficient training, a calm-vs-volatile regime mismatch, policy collapse,
overfitting) one at a time, each with real evidence, documented in
`docs/reports/PROJECT_FINAL_REPORT.md` Section 5. L1 was validated as a working component (real
LLM calls, correct JSON schema, correct cadence) but its live signal was **never** actually wired
into a training run — its contribution to outcomes is unmeasured, and Section 7 of the final
report explains precisely why closing that gap is a substantially bigger undertaking than it
looks (an out-of-distribution confound: L3 was trained with L1's obs dimensions held at a stub
value that also scales a reward term, so testing live L1 without retraining L3 would confound "L1
helps" with "L3 sees inputs it never trained on"). A late follow-up round found that although L2
steering doesn't improve execution *quality*, the frozen L3 policy's order-placement *pattern* is
measurably less predictable than TWAP's — a property, explicitly not shown to be a real-market
advantage, since (per §1.3) this simulator cannot test that
(`docs/reports/l3_execution_predictability_report.md`).

---

## 2. Directory Structure (current, verified against the real repo at HEAD)

```text
lob-execution-hma/
├── CLAUDE.md                      # project-specific agent instructions
├── README.md                      # short pointer to this doc + the final report
├── pyproject.toml                 # dependencies (see §2.1)
├── uv.lock
├── configs/                       # YAML configs — see §4.7 for which are real vs. placeholder
│   ├── data.yaml, env.yaml, ollama_l1.yaml    # still unpopulated placeholders
│   ├── ppo_l3.yaml                            # REAL — consumed by src/train/train_l3.py
│   └── sac_l2.yaml                            # STILL a placeholder — train_l2.py takes its
│                                                 config via CLI flags instead, not this file
├── data/
│   ├── README.md                  # regeneration instructions for all data pipelines
│   ├── splits/l2_bybit_btcusdt_split.json     # the frozen chronological train/val/test split
│   ├── raw/                       # Binance historical (trades/aggTrades/bookDepth)
│   ├── raw_l1/                    # Binance L1 aggregates (klines/funding/open-interest)
│   ├── raw_l2/                    # live Binance L2 capture output — EMPTY, network-blocked path
│   ├── raw_l2_bybit/BTCUSDT/      # PRIMARY L2 source, original parquet format, 34GB/441 days
│   └── raw_l2_bybit_numeric/BTCUSDT/   # same data, converted to the faster numeric/zstd format
│                                        # (src/data/l2_numeric_format.py) — this is what real
│                                        # training runs actually read (--use-numeric-format)
├── docs/
│   ├── architecture_spec.md       # ORIGINAL design doc — do not edit, historical record
│   ├── TRACK_STATUS.md            # full chronological working log, all three tracks
│   └── reports/                   # 19 point-in-time findings reports + this guide's siblings
│       └── figures/               # PNG figures referenced by reports
├── models/                        # trained checkpoints (large binaries mostly gitignored;
│                                    # 3 small backup pairs ARE tracked — see §2.2)
├── project_architecture_draft1.md # SUPERSEDED by this document — an earlier, now very stale
│                                    # snapshot from 2026-08-15, kept only as historical record
├── scripts/                       # ~45 CLI entry points — see §4.8-4.11 for the full breakdown;
│                                    # roughly: data acquisition, throughput/profiling
│                                    # investigations, eval/analysis harnesses, replay/visualization
├── src/
│   ├── agents/                    # L1 macro analyst + 3-tier orchestrator (§4.4)
│   ├── analysis/                  # market-impact calibration (§4.3)
│   ├── data/                      # data pipeline library code (§4.2)
│   ├── envs/                      # the Gymnasium environment + matching engine + rewards (§4.1)
│   ├── metrics/                   # still an empty stub — see §4.6
│   └── train/                     # train_l2.py / train_l3.py entry points (§4.5)
└── tests/                         # 24 files — see §4.12
```

### 2.1 Tech stack (`pyproject.toml`)

- **Language/runtime**: Python ≥3.11.
- **RL**: `gymnasium==0.29.1`, `stable-baselines3==2.3.2` (SAC, for L2), `sb3-contrib==2.3.0`
  (RecurrentPPO, for L3).
- **Agents/orchestration**: `langgraph==0.2.34`.
- **Data**: `pandas==2.2.2`, `pyarrow==17.0.0`.
- **ML tooling**: `scikit-learn==1.9.0` — added late in the project specifically for the execution-
  predictability follow-up's classifier (`scripts/analyze_predictability.py`); not used anywhere
  else.
- **Networking**: `requests==2.32.3`, `websockets==12.0`.
- **Validation**: `pydantic>=2` (used for L1's structured LLM output schema).
- **Training infra**: `tensorboard==2.21.0`, `pyyaml>=6.0`, `tqdm>=4.66`, `rich>=13`.
- **Testing**: `pytest==8.3.2`.
- Not declared in `pyproject.toml` but present/used in the real `.venv`: `matplotlib` (used by
  `scripts/replay_episode.py`), `scipy` (used throughout the eval/analysis scripts for
  `scipy.stats`). This is a real, longstanding gap between the declared and actual dependency set
  — worth fixing if this project is picked up again, not fixed as part of this document.

### 2.2 What's tracked in git vs. gitignored

`.gitignore` excludes `.venv/`, `logs/`, `models/*.pkl` and `models/*.zip` (at any depth), and
`data/raw*` — so the multi-gigabyte data archives and the large training checkpoints/replay
buffers are never committed. **Three small checkpoint-backup pairs ARE deliberately tracked**
despite the general pattern (added with an explicit override, because they are the project's
canonical reference points, not routine training output): `models/baseline_20M_backup/` (the
original L3 baseline), `models/l3_frozen_backup/` (the checkpoint every L2 round and the
predictability analysis actually use — see §1.4), and `models/v1_near_backup_step2M/` (a recovery
checkpoint from an incident described in the final report's methodological-lessons section). Each
pair is a `.zip` (policy weights) + `.pkl` (`VecNormalize` statistics) — both are needed together;
loading one without its matching other will not reproduce the checkpoint's real behavior.

## 3. File-by-File Breakdown

The rest of this document walks every real file in the project, grouped by directory in
roughly dependency order: the simulator first (§4.1, since every other layer depends on it),
then the data pipeline and analysis code that feeds it (§4.2-4.3), then the agents and training
entry points that consume it (§4.4-4.7), then the ~45 CLI scripts built around all of the above
(§4.8-4.12), then the test suite that guards all of it (§4.13). Each subsection states a file's
purpose, its key classes/functions with enough implementation detail to actually extend the
code (not just what it's called), and — wherever this project's own history makes it relevant —
the real bug, measurement, or design correction that produced the code as it stands today,
with exact numbers rather than vague characterizations. Section numbers below intentionally
keep their original "4.x" form throughout, matching every in-repo cross-reference this guide
makes to itself.

## 4.1 `src/envs/` — The Simulator (the foundation everything else stands on)

This package is the single most important directory in the project. Every tier (L1, L2, L3),
every training script, and every evaluation script ultimately runs episodes through
`LOBExecutionEnv`. Read this section before any other file-by-file section.

### `src/envs/matching_engine.py` (139 lines)

**Purpose**: The core fill-simulation primitives — queue-position tracking for resting limit
orders, and level-by-level walking for market orders. Pure functions and one frozen dataclass;
no state, no I/O, no knowledge of the RL environment around it.

- **`QueueState`** (frozen dataclass): `q_ahead` (visible resting volume ahead of your order at
  its price level), `own_qty_remaining`, `filled_qty`. `.is_resolved` is `own_qty_remaining <= 0`.
- **`queue_position_ratio(state)`**: `q_ahead / (q_ahead + own_qty)`, or `-1.0` if there's no
  resting order. This is observation index 13.
- **`update_queue(state, *, v_trade, v_cancel, q_p_before)`**: the one-tick physics update.
  Cancels deplete `q_ahead` proportionally (`v_cancel * q_ahead/q_p_before`) **without ever
  producing a fill** (canceled volume never traded through anyone); trade volume depletes
  whatever's left of `q_ahead` first, and only the *leftover* trade volume reaches your own
  order. This sequential cancel-then-trade decomposition is proven algebraically identical to
  the architecture spec's single combined formula
  `max(0, Q_ahead - V_trade - V_cancel*Q_ahead/q_p)` — see `tests/test_matching_engine.py`'s
  hand-worked fixtures for the proof-by-example.
- **`walk_market_fill(qty, prices, sizes)`**: consumes quantity starting at `prices[0]` (the
  touch) and walking outward level-by-level until either `qty` is exhausted or the visible book
  runs out. **Never invents a fill at a synthetic price** — any unconsumed remainder comes back
  as `qty_unfilled` for the caller to carry forward or fold into the terminal opportunity-cost
  IS component. This is the function responsible for the "no market impact" property: it reads
  a fixed snapshot of `prices`/`sizes` and has no mechanism to feed back into what a *later*
  tick's book looks like.
- **`expected_wait_time(q_ahead, avg_trade_rate)`**: `q_ahead / avg_trade_rate`, returns
  `math.inf` when the rate is zero (correct answer, not an error). **Confirmed via grep: never
  called anywhere in production code** — only exercised by its own unit tests
  (`test_expected_wait_time_hand_computed`, `test_expected_wait_time_zero_rate_is_infinite`).
  `LOBExecutionEnv._PLACEMENT_STALENESS_WINDOW_TICKS` explicitly documents that it was *not*
  derived from this function because the env doesn't currently compute a live per-price trade
  rate.

### `src/envs/reward.py` (400 lines)

**Purpose**: L3's per-tick reward function (`step_reward`) and the Implementation Shortfall
decomposition (`compute_implementation_shortfall`) used both as the terminal reward bump and as
the actual evaluation metric everywhere in the project.

**`RewardWeights`** (dataclass) — six components, three of them original-spec, three added
during the project with their own extensive in-code derivation history:

| Field | Default | Role |
|---|---|---|
| `alpha` | 1.0 | slippage weight |
| `lam` | 0.02 | inventory-holding penalty weight |
| `beta` | 0.5 | flat unfilled-cancel penalty (MARKET only) |
| `gamma` | 0.3 | queue-position-weighted cancel penalty |
| `delta` | 0.8 | spread-capture (maker) bonus |
| `kappa` | 1.0 | terminal IS weight |
| `zeta` | 0.06 | **experimental**: staleness penalty on `ticks_since_own_fill_norm` while resting |
| `eta_replace` | 0.0 (inert) | **experimental**: placement-anchored staleness penalty, feeds only `r_placement_stale` |
| `subtract_twap_baseline` | `False` (inert) | **experimental**: variance-reduction reward shaping (see below) |

**`step_reward(...)`** computes, per tick: `r_slip` (signed slippage vs. arrival price, scaled
by fill fraction), `r_spread` (maker bonus, only on `is_maker` fills), `r_inv`
(`-lam * (1 + max(0, l1_risk_score)) * (qty_remaining/qty_total)^2 * dt` — this is the exact
formula that makes L1's `risk_score` load-bearing in the *reward*, not just the observation,
which is central to the L1 out-of-distribution-confound argument in the final report),
`r_queue` (the cancel penalty — see below), `r_stale`, `r_placement_stale`. Returns their sum.

**The `r_queue` history is the single richest piece of in-repo documentation in the whole
project** — five numbered "EXPERIMENTAL ADDITION" blocks in the module docstring, each
correcting the previous one, driven entirely by measured checkpoint behavior:

1. *(baseline, Section 3.3)* MARKET and CANCEL_AND_REPLACE both charged
   `-beta - gamma*queue_ratio` identically.
2. *r_stale added* after the 20M-step baseline was found to use CANCEL_AND_REPLACE/MARKET **0%
   of the time** — an idle unfilled order costs almost nothing per tick under `r_inv` alone
   (break-even ≈ 400 idle ticks), so nothing ever pushed the policy to act.
3. *r_placement_stale added* after confirming `r_stale` is **structurally incapable** of ever
   rewarding CANCEL_AND_REPLACE at any coefficient — it depends only on
   `_last_fill_tick_idx`, which a replace never touches, so "one stale order for 1000 ticks"
   and "replaced 5 times in the last 10 ticks" look identical to it.
4. *`r_queue` split* — MARKET keeps the full `-beta - gamma*ratio` charge; CANCEL_AND_REPLACE
   drops the flat `-beta` and keeps only the queue-weighted part, since only MARKET guarantees
   a fill (MARKET strictly dominated REPLACE under identical pricing — confirmed as the
   structural cause of 0% REPLACE usage at *every* coefficient tried).
5. *Direction of the queue-weighted term inverted* — the original split charged **less** for
   discarding a fresh order (which hasn't earned any queue priority) and **more** for one that
   had already waited its way to the front (`queue_ratio = q_ahead/(q_ahead+own_qty)` is
   *largest* for a fresh order behind a deep book, not smallest) — backward from any sensible
   "wasted patience" cost model, and specifically it made a price outside the visible top-20
   book (`q_ahead=0` exactly) cost **nothing** to spam-replace. The correction charges
   `gamma * (1 - ratio)` instead, closing that exact free-replace exploit (off-book now costs
   the *maximum* `-gamma`) while leaving a weaker, market-depth-dependent residual explicitly
   flagged as not fully closed (own resting size has a floor of 20% of `qty_remaining` at
   placement, from `SIZE_FRACTIONS`, so the ratio can approach but never hit exactly zero via
   that residual path).

`RewardWeights.eta_replace`'s docstring re-derives the spam-replace loophole bound after *each*
of these changes — it is worth reading in full in the source if extending this reward.

**EXPERIMENTAL 5 — `subtract_twap_baseline`**: variance reduction, explicitly **not** an
objective change. `LOBExecutionEnv._compute_twap_shadow_terminal_is()` (see below) computes what
a TWAP execution would have scored on the *same window*, once per episode, entirely from local
state with zero interaction with the real episode. Because that quantity doesn't depend on the
policy's actions, subtracting it from the terminal reward is a per-episode constant shift — it
cannot change which policy is optimal, only reduce how much of the reward's variance is
"unlearnable market drift" the critic could never have predicted anyway. **Does not affect
`info["implementation_shortfall"]`** — only the scalar reward used for training changes.

**`compute_implementation_shortfall(...)`** — the Perold decomposition:
`is_total_bps = fill_ratio * is_exec_bps + is_opp_bps + fees_bps`. The `fill_ratio == 0` edge
case is special-cased explicitly (`exec_contribution = 0.0` by construction) rather than
computed as `0 * NaN` — this is the single most-tested edge case in the whole reward system
(`phase2a_sanity_suite.py`'s `NoOpPolicy` check, `test_reward.py`'s
`test_implementation_shortfall_no_fills_is_exactly_opportunity_cost`).

### `src/envs/l2_reward.py` (113 lines)

**Purpose**: An alternative, opt-in L2 reward — potential-based mark-to-market IS shaping,
added in the L2-reward-redesign round after `analyze_l2_reward_components.py` measured that the
*old* aggregation (FrozenL3Wrapper just summing L3's raw `step_reward()` over each window) gave
L2 a signal that was **85.6% `r_stale`** — a component L2 doesn't control at all — while
terminal IS, the metric L2 is actually scored on, was only **6.9%**.

**`l2_potential(...)`** computes `Phi(t) = -kappa * compute_implementation_shortfall(fills=
episode_fills_so_far, ..., terminal_mid_price=CURRENT mid_price).is_total_bps` — "what would my
reward-scale IS be if I marked the unfilled remainder to the current price right now."

**`l2_window_reward(...)`** returns `Phi(t) - Phi(t-1)` (Ng/Harada/Russell 1999 potential-based
shaping, `gamma=1` since SAC's own discounting applies separately). This **telescopes exactly**
(not approximately) to the real terminal reward over a full episode, because `Phi(T)` is
computed from the exact same `compute_implementation_shortfall()` call `LOBExecutionEnv.step()`
itself makes at the real terminal tick, and `Phi(0) = 0.0` exactly (`fill_ratio=0` at reset, and
`arrival_price` **is** the reset tick's own `mid_price` by construction, so the opportunity term
at `t=0` is a literal same-value subtraction) — proven both algebraically in the module
docstring and empirically by `tests/test_l2_reward.py::test_potential_is_shaping_telescopes_
exactly_on_real_episodes`, which runs on real data with the real frozen L3 checkpoint (gated,
not synthetic).

The docstring also explains **why this doesn't inherit the failure mode of an earlier, different
experiment** — `subtract_twap_baseline` (reward.py, above) subtracts a *separately-timed*
reference trajectory, which gives an agent that finishes faster a real (if unintended) incentive
to rush, since its own drift exposure shrinks relative to a fixed comparison window. `Phi(t)`
has no second reference trajectory at all — it is a pure function of the real agent's own
accumulated fills evaluated only at the real agent's own actual decision points — so this
specific failure mode cannot arise by construction.

### `src/envs/wrappers.py` (349 lines)

**Purpose**: `FrozenL3Wrapper` — **this file IS the entire L2/L3 integration layer**, not a
thin adapter. It wraps a single-tier `LOBExecutionEnv` so that one call to `wrapper.step(l2_
action)` internally rolls the frozen L3 policy forward for `ticks_per_l2_decision` (default 50)
raw ticks and returns the L2-cadence aggregate observation/reward.

**Constructor**: `(env, l3_model, l3_vecnormalize_path, ticks_per_l2_decision=50,
l2_include_prev_action=False, l3_deterministic=False, l2_reward_mode="l3_passthrough")`. Loads
the frozen L3 checkpoint's *own* saved `VecNormalize` stats via a `DummyVecEnv([lambda: env])`
wrapper trick (confirmed against SB3 2.3.2 source: `DummyVecEnv.__init__` only reads
`observation_space`/`action_space`/`metadata`, never resets or steps — safe to build around the
same live env instance).

**Observation downsampling** (`_downsample_window`, module-level function): takes the raw
`(n_ticks, 42)` window of L3 observations from one L2 decision window and reduces it to a
40-feature L2 observation. Two groups: `_MEAN_OLD_IDX` (idx 2, 6-9, and all 20 book-depth
indices 19-38) are averaged over the window; everything else takes the *last* tick's value.
Indices 15/16 (`l2_target_slice_ratio`, `l2_urgency`) are excluded entirely — L2 *produces*
these, it doesn't consume them. Plus one engineered feature, `schedule_deviation`
(executed-so-far minus TWAP-scheduled-so-far, `[-1,1]`), giving `L2_BASE_OBS_DIM = 41`; with
`l2_include_prev_action=True` (default off), two more raw action scalars are appended for
`L2_FULL_OBS_DIM = 43`.

**Action-space transform**: `participation_rate_multiplier` (dim 0, range `[0,2]`) is **not** a
direct map onto the target — it *scales* the env's own default linear-TWAP baseline:
`override = clip(twap_baseline * participation_mult, 0, 1)`. `urgency` (dim 1, `[0,1]`) **is** a
direct 1:1 pass-through to `env.l2_urgency`. This asymmetry is deliberate (a direct `Box(0,1)`
map for participation would discard the schedule's own built-in monotonic structure) and is
exactly what `tests/test_wrappers.py::test_apply_l2_action_participation_multiplier_scales_
twap_not_1to1` pins down.

**Four corrections made during implementation, each with its own regression test** — worth
knowing before touching this file again:

1. `schedule_deviation`'s baseline is computed by **directly recomputing** the linear-TWAP
   formula from `info["ticks_elapsed"]`/`horizon_ticks`, *not* by calling
   `env._compute_l2_target_slice_ratio()` — that hook silently returns whatever override is
   currently set (which the wrapper itself keeps permanently non-`None` after the first ever
   step), so calling it here would have collapsed the deviation signal to ~0 after the very
   first window.
2. `l2_include_prev_action` defaults `False`, not `True` as originally planned — the
   recurrent-policy precedent that reasoning leaned on doesn't transfer to SAC's plain
   `MlpPolicy`.
3. **`l3_deterministic` parameter added** (default `False`) — the frozen L3 policy's inner
   `predict()` calls were hardcoded `deterministic=False` unconditionally, with no way for a
   caller to get reproducible frozen-L3 behavior. Caught by `tests/test_train_l2.py`'s eval
   determinism test: identical seed + identical L2 action still produced a *different* per-tick
   reward trajectory (same terminal fill_ratio/IS, but the intermediate path differed), because
   L3's own actions were still sampled stochastically underneath regardless of the caller's
   intent. Eval call sites now construct with `l3_deterministic=True`.
4. **`reset()` now explicitly zeroes `l2_target_slice_ratio_override`/`l2_urgency`** at the
   start, before calling `env.reset()` — `LOBExecutionEnv.reset()` never touches either
   attribute, so on a reused instance (every episode after the first, in both training and
   eval) they would otherwise silently leak the *previous* episode's last values into the new
   episode's very first observation (raw obs idx 15/16). Real cross-episode data leak during
   training, and it also broke eval reproducibility (episode N's first observation would depend
   on whatever L2 action ended episode N-1). Caught by the same determinism test — fixing
   correction 3 alone wasn't enough.

**`l2_reward_mode`** (added later, same opt-in convention): `"l3_passthrough"` (default) is the
original raw-sum behavior; `"potential_is_shaping"` swaps in `l2_window_reward()` from
`l2_reward.py` entirely (replaces, doesn't add to, `agg_reward`).

### `src/envs/lob_execution_env.py` (1,239 lines — the largest file in the project)

**Purpose**: `LOBExecutionEnv`, the Gymnasium environment. Replays real historical Bybit L2
snapshots tick-by-tick, running the matching-engine primitives above against them.

**Action space**: `MultiDiscrete([4, 11, 5])` — order type (`HOLD=0, LIMIT=1, MARKET=2,
CANCEL_AND_REPLACE=3`) × price offset (`price_offset_idx - 5`, so `-5..+5` ticks from touch) ×
size fraction (index into `SIZE_FRACTIONS = (0.2, 0.4, 0.6, 0.8, 1.0)` of `qty_remaining`).

**Observation space**: the full 42-dim vector in `_OBS_SPEC` (a tuple of `(index, name, clip
range)` — `observation_space` bounds are built directly from it, so it is the single source of
truth). Notable indices: 13 = `queue_position_ratio`, 14 = `ticks_since_own_fill_norm`, 15/16 =
`l2_target_slice_ratio`/`l2_urgency` (L2's own outputs, fed back in), **17/18 =
`l1_risk_score`/`l1_confidence`** (the exact indices the L1 out-of-distribution-confound
argument in the final report turns on), 19-38 = 20 book-depth levels (each z-scored against its
*own* trailing 60s rolling mean/std — a corrected reading from an earlier, ambiguous
cross-sectional interpretation), 39 = `funding_rate_z`, 40 = `taker_buy_sell_ratio_1m`, 41 =
`own_open_orders_norm`. Indices 10/11 (`cancel_add_ratio_bid/ask`) are **hardcoded 0.0** —
genuinely blocked, since the Bybit snapshot archive cannot distinguish a cancel from a trade at
the same price (this is the same root cause behind `v_cancel=0` in `_estimate_trade_volume`
below, and behind `CancelAddTracker.ratio()`'s degenerate always-0.5 behavior in
`src/data/features.py`).

**`TickView`** (dataclass): one snapshot's worth of book state (`best_bid/ask`, `mid_price`,
`spread`, and the full `bid_prices/sizes`/`ask_prices/sizes` arrays). `.qty_at_price(price,
side)` looks up the resting size at a price via `np.isclose(..., atol=TICK_SIZE/2, rtol=0.0)` —
**the `rtol=0.0` is a real bug fix**: `np.isclose`'s default `rtol=1e-5` at BTCUSDT's ~$100k+
price scale is ~$1-1.5 wide, 20-30x wider than the intended half-tick ($0.05) match window,
which was measured to make the match rate **100% at every offset tested** (including offsets
that cross the opposing side entirely) with **~89% of matches ambiguous** (multiple array
entries satisfying the loose tolerance, with the first — not the nearest — silently selected).

**`__init__`**: key parameters — `horizon_ticks=3000`, `lookback_ticks=10`,
`tick_interval_s=0.1`, `min_size_mult=0.5`/`max_size_mult=8.0` (order-size sampling range
relative to typical top-of-book depth), `reward_weights`, `date_range` (pins the exact file set
for reproducibility — without it, a fixed seed's file-index draw silently drifts as backfill
adds more files over time), `l1_risk_score`/`l1_confidence`/`l2_urgency`/
`l2_target_slice_ratio_override` (plain public attributes — the L1/L2 "stub hooks," the same
pattern the orchestrator later reads/writes directly), `use_numeric_format` (opt-in switch to
the `.npzst` archive, see `src/data/l2_numeric_format.py`).

**Day caching**: `_MAX_CACHED_DAYS = 5` (measured ~828MB/day in memory per cached day; at
`n_envs=8` this is ≈34.4GB cache + ≈4.4GB overhead ≈ 38.8GB, leaving headroom on the box's 50GB —
this number replaced an earlier comment that had claimed ~85MB/day, an order-of-magnitude
error caught during the throughput-investigation round).

**`reset()`**: draws a random file index and a random window start (`self.np_random`, so the
draw sequence — and therefore reproducibility — is sensitive to call order); builds a lookback
buffer (up to 600 ticks / 60s, for the widest rolling-window features) *without* perturbing the
RNG draw (the buffer size is decided *after* `start` is already drawn, purely a post-hoc slicing
choice); computes local per-episode normalization stats (`_spread_p95`, `_ret1s_mean/std`) over
exactly the *original* Phase 2a window (`legacy_ticks`) — **not** the larger buffered range, a
distinction that mattered in practice: `ref_depth` (which sizes `qty_total`, and therefore feeds
the reward/IS directly, unlike the z-score stats which only feed observations) silently drifted
when this scoping was gotten wrong, caught via a controlled seed-matched before/after comparison
(`qty_total`: 23.9086 vs. 22.7718 for one test seed). Order size is drawn log-uniformly between
`min_size_mult`/`max_size_mult` × the window's median top-of-book depth.

**`_compute_twap_shadow_terminal_is()`**: simulates a full TWAP execution (n_slices=10) over the
*same* window via purely local state — never touches `self._resting`/`qty_remaining`/`_tick_idx`
— reusing the exact matching-engine primitives `step()` itself uses. The TWAP-specific slicing
logic is **duplicated**, not imported, from `scripts/phase2a_sanity_suite.py`'s `TWAPPolicy`
deliberately (that script is documented as throwaway, not something core env code should import
from) — but `tests/test_lob_execution_env_features.py`'s (via
`test_twap_baseline_reward.py::test_subtract_twap_baseline_matches_real_twap_policy_exactly`)
integration test runs the *real* `TWAPPolicy` through the *real* env on matching seeds and
asserts bit-identical output, which is what actually guards against the duplication drifting —
not code review alone. A real, subtle bug was caught and fixed here during development: the
size-fraction decision must snap to the nearest discrete `SIZE_FRACTIONS` value exactly like the
real system does (rather than using the continuous fraction), or the shadow silently drifts by
a small but real amount (~0.02bps, measured).

**`step(action)`**: (1) evolves any existing resting order against market activity since the
last tick via `update_queue()`; (2) decodes the action and applies it — `MARKET` and
`CANCEL_AND_REPLACE` both tear down a resting order identically but are tracked via **two
separate flags** (`canceled_via_market`/`canceled_via_replace`) purely so `step_reward()` can
price them differently (see the `r_queue` history above); (3) computes `step_reward()`; (4)
checks termination (`qty_remaining <= 1e-12`) and truncation (horizon reached); (5) on either,
computes and adds the terminal IS reward bump (optionally TWAP-baseline-adjusted for the reward
only — `info["implementation_shortfall"]` always reports the real, unadjusted number).

**`_place_limit()`**: **the crossing-order fix** — a price that crosses the opposing side (e.g.
a buy priced at or above the current ask) is **routed through `walk_market_fill()`** against the
opposing side's depth, exactly like an explicit `MARKET` action, rather than falling through to
become an ordinary resting "ghost" order. Before this fix (combined with the `qty_at_price`
tolerance bug above), this produced fictitious fills for **~45% of placements** in an offset
sweep — this is the second of the two documented fill-simulation bugs referenced throughout
`docs/reports/PROJECT_FINAL_REPORT.md`.

**`_estimate_trade_volume(prev_idx, curr_idx, price, side)`**: approximates `v_trade` as the
**full observed decrease** in resting quantity at a price between two ticks (`v_cancel=0`
always, in this adapter) — this Bybit snapshot archive has no separate trade tape, so trade and
cancel volume are fundamentally indistinguishable from this data source alone. **This is the
other half of the no-market-impact/no-adversary property**: `v_trade` is derived purely from
real historical book-depth change, independent of anything the agent itself does.

**Vectorization** (`_rolling_return`, `_rolling_rms`, `_rolling_sum`, `_rolling_mean_std`,
`_vec_qty_at_price`): all five were rewritten from Python `range(n)` loops to vectorized numpy
via cumulative-sum tricks during the throughput-investigation round, each verified
byte-identical against the original scalar version by `tests/test_reset_vectorization_
equivalence.py` (which reproduces the *original* loop implementations verbatim as reference
fixtures, since the old code no longer exists in source once replaced). The touch-depletion
computation (feeding idx 12/40) alone was ~77% of `_precompute_feature_series()`'s cost at
n≈3600 ticks/reset before vectorization, per cProfile.

## 4.2 `src/data/` — Data Pipeline Library Code

### `src/data/__init__.py` (15 lines)

Re-exports `bulk_download`/`download_and_verify` (from `download_manager.py`),
`CancelAddTracker`/`mid_price`/`micro_price`/`obi` (from `features.py`), `L2Book`/`apply_diff`/
`reconstruct_book` (from `l2_capture_daemon.py`). **Confirmed dead**: a repo-wide grep for
`from src.data import` returns zero matches — every real consumer imports directly from the
submodule instead (e.g. `lob_execution_env.py` does `from src.data.features import obi,
zscore`). This re-export list is a maintained but functionally unused piece of the package.

### `src/data/download_manager.py` (77 lines)

Generic Binance-futures daily-archive downloader: `_build_url`/`_build_checksum_url`,
`download_and_verify()` (fetch zip + `.CHECKSUM`, verify SHA-256, extract, write parquet),
`bulk_download()` (loops a date range, swallows per-day exceptions so one bad day doesn't kill
the whole range). This is a smaller, simpler, **earlier** version of the same pattern
`scripts/bulk_backfill.py` reimplements independently with production features (manifest
tracking, retries, concurrency, signal handling) — the two do not share code, "to keep the CLI
standalone" (the same design choice `orchestrator_graph.py` cites for its own sync/async split).

### `src/data/features.py` (84 lines)

Small numeric primitives: `mid_price()`, `micro_price()` (size-weighted), `obi()` (order-book
imbalance over the top-k levels), `zscore()` (clipped, with a `std<=0` → `0.0` neutral-value
guard — this exact function is imported directly into `lob_execution_env.py` and used
throughout `_build_obs()`). `CancelAddTracker` — tracks a rolling window of `(cancel, add)`
event pairs; **`ratio()` is confirmed structurally degenerate**: `cancel_total = len(events)`
and `add_total = len(events)` are set to the *same* value regardless of the actual
cancel/add counts passed to `observe()`, so `ratio()` always returns exactly `0.5` once the
window has any history, or `0.0` when empty. **Confirmed unused in production**: `mid_price()`
and `micro_price()` (as function calls) appear nowhere outside their own `tests/test_features.py`
— `lob_execution_env.py` computes an inlined micro-price calculation itself rather than calling
this module's version.

### `src/data/l1_features.py` (284 lines)

**Purpose**: `build_l1_feature_summary(as_of_ms, raw_l1_dir)` — the missing link between the
collected `data/raw_l1/` archive (klines/funding/open-interest) and what
`L1MacroAnalyst.maybe_refresh()` actually consumes as its `feature_summary` dict argument.
Deliberately omits an order-book-imbalance field: `data/raw_l1` has no book-depth data at all
(that lives in the separate Bybit L2 archive feeding L2/L3), and the module docstring is
explicit that inventing an OBI proxy from L1-only data would misrepresent a genuine signal as
one this pipeline cannot actually see.

Point-in-time discipline: every window is filtered strictly to `timestamp <= as_of_ms`,
mirroring `split.py`'s chronological-only rule for the same underlying reason (a signal
partly built from future information is not honestly evaluated). Every derived field is
explicitly `None` (→ JSON `null`) rather than a silently-propagated NaN when its window has
insufficient real coverage (`_MIN_WINDOW_COVERAGE_FRAC = 0.8`) — `json.dumps` would otherwise
emit a literal (invalid) `NaN` token into the payload `L1MacroAnalyst` trusts as input.

Real, verified data-inventory notes baked into the docstring: `open_interest` has a **confirmed
exact-duplicate-row artifact** across every day from 2020-09-01 through 2021-05-21 (263
consecutive days, 100% of that span, byte-identical across every non-timestamp column) — an
early collection-run artifact, unconditionally deduplicated on `create_time` by `_load_oi_window`
regardless of date (a correct no-op on the clean files too). **Confirmed via grep: never
actually imported by `src/agents/orchestrator_graph.py`** — the orchestrator's `run_episode()`
takes a `feature_summary_fn` callable parameter and only *suggests* wiring this function in via
docstring, but no production call site does so yet; every test and real invocation currently
passes a stub/lambda instead. This is the concrete gap behind "L1's live signal was never
wired into training."

**`build_l1_feature_summary()` output fields**: `return_1h_pct`/`return_24h_pct`,
`realized_vol_1h`/`realized_vol_24h` (RMS of log returns), `taker_flow_imbalance_1h`,
`funding_rate_current`/`funding_rate_z`, `open_interest_level`/`open_interest_change_24h_pct`,
`top_trader_long_short_ratio`, `taker_long_short_vol_ratio`.

### `src/data/l2_capture_daemon.py` (204 lines)

**Purpose**: live Binance L2 order-book capture over websocket — `L2Book` (dataclass:
`bids`/`asks` dicts, `last_u`/`last_pu` sequence trackers), `apply_diff()` (merges one delta
message into a book; a `size <= 0` entry removes the price level), `reconstruct_book()`
(replays a list of diffs from scratch), `L2CaptureConfig`/`L2CaptureDaemon` (the actual
websocket client: seeds from a REST snapshot, applies streamed diffs, detects sequence gaps via
`pu != last_u` and re-seeds via a fresh snapshot on gap, shards output to parquet every
`shard_rows`/`flush_interval_s`). **Important, confirmed by direct reading**: `apply_diff()`
itself has **no built-in sequence-gap detection** — it unconditionally merges whatever it's
given. Gap detection is the *daemon's* (`_apply_event()`'s) responsibility, one layer up. This
distinction is exactly what `scripts/collect_l2_bybit.py`'s own module comment flags (an earlier
draft of that script wrongly assumed gap-checking lived inside `apply_diff` itself), and is also
the reason `tests/test_l2_capture.py`'s one gap-resync test fails today (see §4.13 — the test
calls `apply_diff()` directly and expects it to self-heal a skipped update, which by design it
does not; that responsibility sits in a caller neither this test nor `l2_capture_daemon.py`'s
own daemon path exercises for this exact case).

**Live production status**: `data/raw_l2/` (this module's own real output directory) is
confirmed **empty** — the live Binance capture path is network-blocked from this host
(`fapi.binance.com`/websocket unreachable), which is exactly why `scripts/collect_l2_bybit.py`
(a separate, historical-archive-based script) became the project's real L2 data source instead.

### `src/data/l2_numeric_format.py` (82 lines)

**Purpose**: single source of truth for the converted numeric day-file format — shared between
the conversion scripts and `LOBExecutionEnv`'s own fast-load path, "so write and read can never
silently drift apart" (module docstring).

**Format**: every real L2 day file has *exactly* 20 bid + 20 ask levels per row (verified
exhaustively, not sampled, across 31 real days by `scripts/check_level_counts.py`) — stored as
flat float64/int64 binary arrays, zstd-compressed at `ZSTD_LEVEL=9`. File layout: 4-byte
big-endian header length, then a JSON header (dtype/shape per array), then the zstd-compressed
concatenated array bytes in the header's own declared order (`ARRAY_ORDER`). `float32` was
tested and **rejected**: 87% of real price/size values don't round-trip exactly through it
(measured on 1.6M real values) — every array stays `float64`, which is what makes
byte-for-byte seed-equivalence with the original format achievable at all.

`write_day(arrays, out_path)` writes atomically (via a `.tmp` file + `Path.replace()`, so no
reader ever sees a partial file). `read_day(path)` reverses it via `np.frombuffer` views into
the decompressed blob (zero-copy slices, not fresh allocations).

### `src/data/split.py` (151 lines)

**Purpose**: the chronological train/val/test split — **the single most widely-imported symbol
in `src/data`** (confirmed via grep: 45 references project-wide to `load_split`/`split.py`).

`SPLIT_SIZE = 18` real day-files per split (val, test); everything older is train. **Critical
interpretation flag, explicit in the module docstring**: "most recent ~15-20 days" is read as a
fixed *real-file count*, not a calendar-day window — a naive most-recent-18-*calendar*-day
window was checked directly against the real archive during Phase 3 preflight and found to
contain only 7 real files (11 of 18 days missing), an unacceptable shrink for a set meant to be
held out entirely. Picking by real-file count guarantees the intended day count at the cost of a
wider, gappier calendar span (test's 18 real files span 34 calendar days) — every known gap date
is recorded in the persisted artifact (`known_gap_dates`), not silently absorbed.

`generate_split()` computes fresh from whatever's on disk now (pure function, no I/O to the
artifact); `write_split()` persists to `data/splits/l2_bybit_btcusdt_split.json` — a small,
**deliberately git-tracked** artifact (explicitly not under `data/raw*`, which is gitignored) —
and **refuses (raises)** if a pre-existing artifact's `val_dates`/`test_dates` would change on
regeneration, since backfill is only supposed to add *older* days; any change there is treated
as a real anomaly to investigate, never silently overwritten. `load_split(name)` reads the
persisted artifact; raises if it doesn't exist yet rather than implicitly generating it.

Real, current split (confirmed live in `eval_l2_diagnostics.py`'s own startup log and multiple
eval scripts): **train = 405 days (2024-04-18 .. 2025-07-15), val = 18 days (2025-07-16 ..
2025-08-02)**, chronologically disjoint, no leakage.

---

## 4.3 `src/analysis/` — Market-Impact Calibration

### `src/analysis/calibrate_impact.py` (425 lines) + `calibrate_impact_results.md`

**Purpose**: fits permanent (`eta`) and temporary (`lambda`) market-impact coefficients from the
project's own historical L2 order-flow data, per the architecture spec's Section 4.5 (Tier 1,
"lands between Phase 3 and Phase 4"). **Confirmed completely standalone**: reads the L2 archive
and the split artifact directly; imports nothing from `src/envs/` and is imported by nothing in
`src/train/`, `src/agents/`, or any active eval script — this is a real, finished piece of
analysis work that was never wired into the live environment (matching the module's own
docstring: "Produces calibrated numbers and methodology for review — NOT wired into the
environment yet").

**Methodology**: permanent impact via the canonical Cont-Kukanov-Stoikov (2014) per-event
order-flow-imbalance formula (touch-level price/size only), bucketed at 5s and regressed
(through-origin OLS) against bucket mid-price return. Temporary impact regresses `|deviation
from a 60s rolling reference|` against `sqrt(participation_rate)`. Decay half-life is fit via an
**excess-over-control event-study design**: naively tracking `|deviation|` after a
high-participation burst showed *monotonic growth* (R²≈0.99) — not a real finding, just ordinary
random-walk diffusion swamping any real impact signal — so the actual method tracks a same-day
non-burst control group identically and uses the **excess** of burst over control at each lag,
which should shrink toward zero as decay happens (module docstring, `DayResult` dataclass
comment, documents this false-start explicitly as a worked example of the trap). 80/20
calibration/holdout split by fixed-seed random day shuffle, both regressions reported on both
sets so overfitting is visible rather than assumed away. Runs with the same `free -h`/`nvidia-smi`
GPU-headroom guardrail discipline (`_memory_guardrail()`, checked against a specific live-run
PID) used throughout this project's heavy scripts.

---

## 4.4 `src/agents/` — L1 Macro Analyst + Orchestration

### `src/agents/l1_macro_analyst.py` (149 lines)

**Purpose**: `L1MacroAnalyst` — wraps a local Ollama LLM call behind a strict pydantic schema,
with a self-throttling, fail-closed, non-blocking contract.

**`MacroRiskContext`** (pydantic `BaseModel`): `timestamp_ms`, `regime` (regex-constrained to
one of `risk_on|risk_off|neutral|high_volatility`), `risk_score` (`[-1,1]`), `confidence`
(`[0,1]`), `urgency_multiplier` (`[0.5,2.0]`), `rationale`. The LLM is **never** allowed to emit
free text into the trading path — every response is validated against this schema before it can
reach `env.l1_risk_score`/`l1_confidence`.

**`maybe_refresh(feature_summary)`**: self-throttling (returns the cached context immediately if
called again within `refresh_interval_s`, default 45s, of the last attempt — successful or not)
and **fails closed** to the last good cached context (or a neutral default on first use) on
*any* error — `requests.RequestException`, `ValidationError`, `json.JSONDecodeError`, `KeyError`
— never raises into the caller's hot path. Explicitly bypasses any environment-configured HTTP
proxy (`proxies={"http": None, "https": None}`) — this host's shell exports `http_proxy`/
`https_proxy` pointed at an unrelated external proxy, and a proxied call to `localhost:11434`
silently fails closed with a `ProxyError` indistinguishable from a real Ollama outage without
this fix.

**Structured output**: the `format=` field sent to Ollama is `MacroRiskContext.model_json_schema()`
itself (a real JSON Schema object), not the bare string `"json"` — added after a validation round
found **0 of 5 real calls schema-conformant** under plain `format="json"` (the model invented its
own field names like `risk_level`/`market_risk_score`). Confirmed live that this Ollama install
(0.32.8) enforces required-keys-and-types via the schema, but **not** numeric range constraints —
a live test with the schema still returned `risk_score=2` (outside `[-1,1]`), so `SYSTEM_PROMPT`
also states every valid range in prose as a second line of defense, and `MacroRiskContext`'s own
pydantic `Field(ge=/le=)` validation remains the authoritative gate regardless — confirmed
non-redundant, not kept "just in case."

Default `model="qwen2.5:32b-instruct-q4_K_M"`. **Note**: `scripts/smoke_test_l1.py`'s one real,
standalone validation call instead used `qwen2.5:14b-instruct-q4_K_M` — the model actually
exercised in that real-call smoke test differs from the class's coded default.

### `src/agents/orchestrator_graph.py` (312 lines)

**Purpose**: stitches L1's ~60s cadence, L2's ~5s cadence, and L3/env's per-tick cadence into one
coherent three-tier control loop.

`L1_EVERY_N_TICKS = 600`, `L2_EVERY_N_TICKS = 50` — a module-level `assert
L1_EVERY_N_TICKS % L2_EVERY_N_TICKS == 0` guarantees every L1 boundary lands exactly on an L2
boundary, which is what makes checking L1's cadence only at L2-decision boundaries (rather than
needing true per-raw-tick orchestrator control) both correct and sufficient.

**Reconciliation note, explicit in the docstring**: the architecture spec's reference code names
`executioner_node`/`env_step_node` as if they were separate per-tick orchestrator-level calls.
They are not implemented that way — `FrozenL3Wrapper.step()` (L2-owned, not touched by this
file) already correctly implements both inside its own inner loop, so `run_episode()` below
calls `wrapper.step(l2_action)` once per L2-decision boundary and treats "L3/env fire every
tick" as an invariant *proven by reading the wrapper's source* and *independently verified* by
`tests/test_orchestrator_graph.py`'s own external, non-invasive tick-counting instrumentation —
not trusted blind.

**`macro_tick(env, l1_agent, tick, feature_summary)`**: synchronous — on a cadence tick
(`tick % 600 == 0`), calls `l1_agent.maybe_refresh()` and writes the result directly into
`env.l1_risk_score`/`l1_confidence` (plain public attributes, the same pattern
`FrozenL3Wrapper` already uses for `l2_target_slice_ratio_override`). Off-cadence: no-op,
returns `None`.

**`run_episode(l3_wrapper, l2_model, l1_agent, feature_summary_fn, ...)`**: the synchronous
full-stack loop — macro tick check, then `strategist_tick()` (one `l2_model.predict()` call),
then `wrapper.step(l2_action)`. `feature_summary_fn(tick)` is only ever called on L1 cadence
ticks; real wiring would pass a closure over `build_l1_feature_summary()`, but **no current
call site does this** — see the `l1_features.py` note above.

**`AsyncL1Refresher`**: wraps `L1MacroAnalyst` so the tick loop never blocks on a real LLM call.
Explicit staleness policy: if a new cadence-boundary refresh is due while a previous background
call is still in flight, the new request is **skipped**, not queued — an unbounded queue would
just move the "hot path never waits" violation from the tick loop into memory; skipping bounds
worst-case staleness to `L1MacroAnalyst.timeout_s` instead. At most one worker thread alive at a
time; `join()` guarantees clean shutdown at episode end (`daemon=True` is a backstop, not the
primary cleanup mechanism). `macro_tick_async()`/`run_episode_async()` are the async
counterparts of `macro_tick()`/`run_episode()`.

---

## 4.5 `src/train/` — Training Entry Points

### `src/train/train_l3.py` (494 lines)

**Purpose**: `RecurrentPPO` training for L3, against a plain `LOBExecutionEnv` with L2 stubbed
as fixed-linear-TWAP (the env's own *default* `_compute_l2_target_slice_ratio()` behavior — no
extra wiring needed, since it's already exactly a TWAP-schedule fraction unless overridden).

**`make_env(date_range, horizon_ticks, lookback_ticks, reward_weights)`**: `SubprocVecEnv`
worker factory.

**`resolve_final_save_paths(run_name, overwrite_canonical, models_dir)`** — the canonical-
checkpoint overwrite guard. Checks both `l3_executioner_v1.zip` **and** `l3_vecnormalize.pkl`
with **OR** (not AND): if *either* exists, *both* outputs redirect to run-tagged names together,
so a run can never leave a mismatched model/VecNormalize pair behind. Added after a real
incident: a bounded probe run's final save silently overwrote a verified checkpoint (documented
in `docs/reports/l3_replace_value_probe.md`) — and separately, that same probe's periodic
`CheckpointCallback` files (before `--run-name` namespacing existed) silently collided with an
unrelated earlier run's identically-named intermediate checkpoints too. Pure path-decision logic
with no I/O beyond an existence check, unit-tested without any GPU/data/config dependency
(`tests/test_train_l3.py`, 4 tests).

**`ValISEvalCallback`**: periodic held-out evaluation against `load_split("val")` **only**, never
test. Baseline = `TWAPPolicy` from `phase2a_sanity_suite.py`. **Paired design**: both arms
(and every firing across the whole training run) reuse the exact same fixed seed list
(`EVAL_SEED_BASE = 5_000_000`, `n_eval_episodes` consecutive integers from there) — the TWAP arm
is computed once at training start and cached as a constant reference line, since a fixed,
non-learning policy has no reason to be re-evaluated every firing.

**CLI** (`build_parser()`): `--config` (default `configs/ppo_l3.yaml`), `--total-timesteps`/
`--n-envs`/`--eval-freq`/`--n-eval-episodes` (all optional config overrides), `--resume-from`/
`--resume-vecnormalize` (paired, required together), `--reward-zeta`/`--reward-eta-replace`
(per-run `RewardWeights` overrides without editing the shared default), `--subtract-twap-
baseline`, `--warm-start-weights` (loads *only* policy weights, fresh `VecNormalize`, step
counter resets to 0 — distinct from and mutually exclusive with `--resume-from`), `--run-name`,
`--overwrite-canonical`, `--progress-bar`/`--no-progress-bar` (explicitly recommended off for
unattended `nohup` runs — tqdm's carriage-return redraws don't collapse in a plain log file the
way they do on a real terminal, so they repeat every refresh for the run's full duration).

### `src/train/train_l2.py` (969 lines — the largest, most heavily-annotated file in the project)

**Purpose**: `SAC` training for L2, driving a frozen L3 checkpoint through `FrozenL3Wrapper`.

**Thread-capping is set at module import time**, before any other import (`OMP_NUM_THREADS`/
`MKL_NUM_THREADS` env vars) — this is the file where the "an earlier un-thread-capped attempt at
this pattern measured 7-9x SLOWER" finding (see §4.10) was first turned into a hard, permanent
rule rather than a one-off benchmark observation.

**`L2_GAMMA = 0.995`** — default SAC discount, independently re-derived for L2's own ~60-decision
episode cadence (effective horizon `1/(1-0.995) = 200` decisions, ≈3.3x episode length);
`gamma=0.983` (effective horizon ≈59, matching episode length almost exactly) is named in the
CLI help as the concrete, flagged-but-then-actually-ablated alternative — this is the value the
project's diagnostic history refers to as "l2v3" vs. the original-gamma "l2v2."

**`make_l2_wrapped_env()`** / **`make_l2_subproc_env(rank, ...)`**: the per-worker factory for
real (`SubprocVecEnv`) training. Each worker, inside its own forked process: caps its own torch
thread count (`torch.set_num_threads(1)`, called post-fork since env-var-only capping is read at
each worker's *own* torch import time, which works, but the explicit call defends against any
code path that imports torch before the env var is read), loads its **own separate** frozen L3
checkpoint on CPU (confirmed correct-by-construction: `FrozenL3Wrapper._l3_lstm_state` is a
plain instance attribeute of an object built fresh inside each worker's own process — no shared
memory across `SubprocVecEnv` workers to accidentally alias), and seeds
`torch.manual_seed(seed + rank)` explicitly — a real gap SB3's own seeding chain doesn't reach:
`BaseAlgorithm.set_random_seed()` seeds the *env's* RNG per worker via `SubprocVecEnv`, but never
reaches a worker's separate-process torch RNG, which is exactly what the frozen L3's stochastic
`predict(deterministic=False)` (the training-time default) samples from — left unseeded, two
identically-`--seed`ed runs would still diverge through L3's own action choices.

**`_resolve_gradient_steps(n_envs, override)`**: pure function (unit-tested in isolation).
Confirmed against the installed SB3 2.3.2 source that `train_freq=(1, "step")` triggers exactly
one `training()` call per `env.step()` call **regardless of `n_envs`**, while `num_timesteps`
itself increments by `n_envs` each call — so leaving `gradient_steps=1` (the single-env
reference value) unchanged at `n_envs>1` would silently cut the update-to-data ratio to
`1/n_envs`, a real, undocumented change to SAC's sample efficiency. Default (`override=None`)
resolves to `n_envs`, preserving a 1-gradient-step-per-transition ratio regardless of worker
count.

**`resolve_l2_final_save_paths()`**: L2's analog of L3's `resolve_final_save_paths` — same
OR-not-AND pairing guarantee, same rationale.

**`ValISEvalCallback`**: L2's own held-out eval, modeled directly on L3's version but with
`eval_freq`/`n_eval_episodes` defaults sized against L2's *own* measured throughput (≈4.15
decisions/sec single-env, roughly half of which is `env.reset()` overhead — a materially
different cost profile than L3's tick-level training, since L2 episodes routinely end well short
of the full 60-decision horizon, paying `reset()` disproportionately more often per unit of
training). Baseline here is "TWAP-passthrough" — L2 always emitting `[1.0, 0.5]` (on-schedule,
neutral urgency), i.e. the frozen L3 policy completely unsteered.

**Full CLI flag table** (`build_parser()`, 26 flags total):

| Flag | Default | Purpose |
|---|---|---|
| `--l3-checkpoint` / `--l3-vecnormalize` | *required* | frozen L3 pair (no default — deliberately, "which checkpoint is the frozen one" is an L3-track decision, not this script's) |
| `--total-timesteps` | *required* | SAC budget (no default — forces an explicit choice) |
| `--n-envs` | 4 | parallel `SubprocVecEnv` workers (not the higher-throughput 8 — see §4.10's OOM history) |
| `--gradient-steps` | `None`→resolves to `--n-envs` | SAC update-to-data ratio |
| `--seed` | 42 | threaded through SAC + each worker's env RNG + each worker's own torch RNG |
| `--gamma` | 0.995 (`L2_GAMMA`) | SAC discount + VecNormalize reward-scaling gamma (must match) |
| `--ticks-per-l2-decision` | 50 | L2 cadence in raw ticks |
| `--horizon-ticks` | 3000 | matches `configs/ppo_l3.yaml`'s production value |
| `--lookback-ticks` | 10 | |
| `--l2-include-prev-action` | `False` | ablation toggle, see wrappers.py |
| `--use-numeric-format` | `True` | reads the `.npzst` archive by default (equivalence-verified 770/770 fixed-seed comparisons) |
| `--data-dir` | derived from the flag above | override point |
| `--l2-reward-mode` | `l3_passthrough` | or `potential_is_shaping` — see `l2_reward.py` |
| `--eval` / `--no-eval` | `True` | |
| `--eval-freq` | 10,000 | |
| `--n-eval-episodes` | 10 | not sized for statistical power — sized to be cheap |
| `--checkpoint-freq-timesteps` | 50,000 | |
| `--run-name` | UTC timestamp | |
| `--overwrite-canonical` | `False` | |
| `--resume-from` / `--resume-vecnormalize` | `None` | paired, required together |
| `--resume-replay-buffer` | `None` | optional even with `--resume-from` |
| `--smoke-test` | `False` | caps `--total-timesteps` at `_SMOKE_TEST_MAX_TIMESTEPS=10,000`, fixed non-colliding save paths |
| `--device` | `cuda` | |
| `--progress-bar` / `--no-progress-bar` | `True` | |

**A real resume-ordering bug caught in review** (documented in the `--seed` flag's own help
text): on `--resume-from`, the SAC checkpoint must be loaded *before* `SubprocVecEnv` workers are
constructed, specifically to read `model.seed` back out and thread the *original* run's seed
into each worker's `torch.manual_seed(seed+rank)` — an earlier draft built the workers first
(using `--seed`'s own value/default), so a resumed run's workers silently kept seeding at the
wrong value even though the SAC model itself correctly reseeded at `model.seed`.

## 4.6 `src/metrics/` — Empty Stub

Contains only an empty `__init__.py` (0 bytes). No other files. Confirmed genuinely unused —
nothing in `src/` or `scripts/` imports from `src.metrics`. This package exists as a placeholder
in the original architecture spec's directory layout and was never populated.

## 4.7 `configs/` — Which YAMLs Are Real

| File | Status |
|---|---|
| `configs/ppo_l3.yaml` (8,740 bytes) | **Real** — the only config file actually loaded by any code (`train_l3.py`'s `yaml.safe_load`). Every field is either a direct architecture-spec value or an explicitly labeled "JUDGMENT CALL" with its own reasoning inline (e.g. `vf_coef`/`max_grad_norm` fall back to sb3-contrib's own defaults since the spec doesn't specify them). |
| `configs/data.yaml`, `configs/env.yaml`, `configs/ollama_l1.yaml` | Unpopulated placeholders — every key present, every value empty/`None`. Confirmed via grep: **never referenced by any `.py` file** in the project. |
| `configs/sac_l2.yaml` | Same placeholder shape. **Confirmed unused**: `train_l2.py` never loads any YAML config at all — every hyperparameter is a CLI flag default or a derived value; the file's existence has no effect on L2 training. |

## 4.8 `scripts/` — Data Acquisition (REUSABLE infrastructure)

These are the actual pipelines that built `data/`. All follow the same shape: a thread-safe/
file-based JSONL `Manifest` for resumability, checksum verification where the upstream source
provides one, exponential-backoff retry, `SIGINT`/`SIGTERM` handling for a clean stop mid-run.

### `scripts/bulk_backfill.py` (Binance trades/aggTrades/bookDepth)

Overnight bulk historical backfill from `data.binance.vision`. `Manifest` keyed by
`(date, dataset)`; `ThreadPoolExecutor` concurrency (default 4). Default date range
2021-08-10 → 2026-08-10. **This is also the file whose real `_checksum_url()`/`_zip_url()`
URL shape (`.../{symbol}-{dataset}-{day}.zip[.CHECKSUM]`, flat, not a dated subdirectory)
diverges from what `tests/test_bulk_backfill.py`'s mocks assume — see §4.13's test-suite
notes for the exact, currently-failing consequence.**

### `scripts/clean_manifest_bug.py` (one-off, run once)

A 20-line one-shot cleanup script: drops manifest entries for a known URL-path bug
(`bookDepth` marked `"missing"` incorrectly) and for days where both `trades` and `aggTrades`
are missing (a whole-day anomaly, since BTCUSDT trades every day). Not parameterized, not a
reusable tool — a historical record of a specific data-cleanup action.

### `scripts/collect_l2_bybit.py` (34GB/441 days — the PRIMARY L2 data source)

**Byte-budget-targeted** backfill of Bybit's public daily order-book archives
(`quote-saver.bycsi.com/orderbook/linear/{symbol}/{date}_{symbol}_ob500.data.zip`). Walks
backward day-by-day from the most recent confirmed-available day (auto-discovered via
exponential-step-then-binary-search HEAD probing — no hardcoded date, since archive currency
drifts) until either the byte budget (default 35GB) or a day cap is hit. Reconstructs book
state per day using `L2Book`/`apply_diff` from `src/data/l2_capture_daemon.py` (reused as-is,
not reimplemented), with an important, explicitly-documented divergence: **this script performs
its own sequence-gap detection one layer above `apply_diff`** (checking Bybit's per-message
incrementing `u` for jumps `>1` and reseeding via an empty-book snapshot when found), since
`apply_diff()` itself has no such check built in — confirmed by directly reading
`l2_capture_daemon.py`'s real source, correcting an earlier draft's wrong assumption. Bybit
publishes no checksum for these archives, so resumability instead verifies against this script's
*own* output parquet's SHA-256. The raw zip is deleted immediately after each day is processed
(never accumulated on disk).

### `scripts/collect_l1_data.py` (klines/funding/open-interest)

Pulls all three from `data.binance.vision` **archive files** (zip+CHECKSUM), not from any
`fapi.binance.com`/`api.binance.com` REST endpoint — those domains are confirmed network-blocked
from this host (DNS resolves to a non-routing address; direct connection resets). Per-dataset
availability differs and is handled explicitly: klines tries monthly archives first, falls back
to daily files for any month where the monthly 404s or the requested range only partially
covers a month; funding rate is monthly-only (no daily fallback exists at all, so the
in-progress current month is simply unavailable until Binance publishes it — `--funding-end`
defaults to the last *fully completed* month, not "today"); open interest is daily-only, and its
archive happens to bundle top-trader and taker long/short ratios in the same file "for free."
Flags a >500MB total-size anomaly explicitly (`_print_summary`'s FLAG check) since L1 aggregate
data is expected to be small.

### `scripts/run_l2_capture.py` (thin CLI wrapper)

Wires `src/data/l2_capture_daemon.py`'s `L2CaptureDaemon` to argparse + `asyncio` + signal
handling. This is the **live** Binance-websocket path, whose real output directory
(`data/raw_l2/`) is confirmed empty (network-blocked) — `collect_l2_bybit.py` above is what
actually populated the project's L2 archive.

### `scripts/smoke_test_l1.py` (one-shot manual validation, not automated)

A standalone, non-pytest script: makes one real Ollama call (model
`qwen2.5:14b-instruct-q4_K_M`, plain `format="json"`, not the structured-schema approach
`L1MacroAnalyst` later adopted) and validates the response against a locally-redefined
`L1Signal` pydantic model (structurally identical to but a separate class from
`MacroRiskContext`). Prints `PASS`/`FAIL`. This is the one place in the whole project where a
real, successful LLM call against the actual running Ollama service is exercised outside a
mocked test.

---

## 4.9 `scripts/` — Numeric-Format Conversion (REUSABLE — the production pipeline, with its full prototyping lineage preserved)

The final format (`src/data/l2_numeric_format.py`) was reached through four tried formats, each
script below a real, sequential step in that investigation — kept in the repo as the record of
*why* the final format was chosen, not just *what* it is.

1. **`convert_one_day.py`** — first prototype: raw, **uncompressed** `.npz` (numpy's own zip
   container). Fails loudly (does not silently pad) if any row lacks exactly 20 bid/ask levels.
   Result: uncompressed came in **5.61x larger** than the original parquet (≈587.5MB/day) — would
   need ≈259GB for the full train+val+test set against only ≈227GB free at the time.
2. **`convert_one_day_compressed.py`** — same shape, `np.savez_compressed` instead. Tests whether
   compression solves the disk problem without eating the load-time win numpy's own zip-based
   compression gave up too much load speed to be the final answer.
3. **`convert_one_day_parquet_numeric.py`** — third format tried: parquet with *real numeric*
   list columns (`pa.FixedSizeListArray`) instead of JSON strings, sweeping `snappy`/`zstd`/
   `none`/`gzip` codecs, timing both write and 5-trial read.
4. **`test_zstd_raw.py`** — fourth format: raw concatenated float64 bytes, zstd-compressed at
   levels `1, 3, 6, 9, 15`, testing directly (not assumed) that zstd decompression speed is
   roughly level-independent even though compression time isn't — this became the winning
   design, formalized as `src/data/l2_numeric_format.py` (`ZSTD_LEVEL=9`).

**`scripts/check_level_counts.py`** — the exhaustiveness check the fixed-20-levels design
assumption rests on: a fast vectorized bracket-counting heuristic (`str.count("[") - 1` per row,
cross-validated against real `json.loads` on row 0 of each file) applied to all rows of 10
benchmark-pool days plus a ≈20-day spread sample across the full train range — confirmed
"ALL EXACTLY 20" for every file checked.

**`scripts/convert_l2_to_numeric.py`** — single-file/`--all` prototype using the finalized
`write_day()` format (imports `src.data.l2_numeric_format`, not a fourth reimplementation).

**`scripts/convert_l2_to_numeric_parallel.py`** — **the production conversion**, actually run
against the full 441-file archive. `multiprocessing.Pool`, `N_WORKERS = 4` — explicitly *reduced
from 8* after the original 8-worker run **OOM-killed 6 workers within ~3.5 minutes** (dmesg
confirmed, each holding 6.1-7.8GB RSS at time of death; per-worker memory scales with file size
via the row-by-row `json.loads` parsing approach). A real, documented gotcha: `multiprocessing.
Pool` silently drops the result for any task whose worker is SIGKILLed — no retry, no exception —
so the original run hung forever waiting on a lost result once all other dispatchable work was
exhausted. Idempotent (skips a day whose output already exists), and `write_day()`'s atomic-write
guarantee means re-running after a crash carries no partial/corrupt-output risk.

**Equivalence gates** (permanent regression coverage, not throwaway): `scripts/compare_formats_
equivalence.py` runs the real 10-fixed-seed, byte-exact (`np.array_equal`, never `np.allclose`)
`env.reset()`/`step()` comparison between the original-format and numeric-format paths on real
converted data — reported result: **770/770 fixed-seed comparisons byte-identical** across all
441 converted days. `scripts/capture_reset_snapshot.py` / `scripts/compare_reset_snapshots.py`
are the earlier, more general before/after-a-code-edit version of the same snapshot-diff
technique (used originally to prove the reset-vectorization edit — §4.10 — changed nothing).
`tests/test_numeric_format_equivalence.py` is the permanent, synthetic-fixture pytest coverage
of the same round-trip property (see §4.13).

## 4.10 `scripts/` — Throughput & Profiling Investigations (THROWAWAY — historical record of real findings, not maintained tooling)

None of these are wired into CI or re-run routinely; each answered one specific performance
question and is kept as the evidence trail behind a real engineering decision.

### The parallelization-oversubscription discovery (the project's single most consequential throughput finding)

- **`benchmark_parallel_l2.py`** — first attempt at `SubprocVecEnv`-parallelized L2 training
  (each worker with its own frozen-L3 CPU copy, motivated by a separate finding that CPU
  `predict()` for this tiny model is *faster* than GPU: 0.94ms vs. 1.72ms/call, and sidesteps
  VRAM contention with L1's Ollama usage). **Result: 0.14x/0.11x "speedup" at n_envs=2/4 — i.e.
  7.1x and 9.1x SLOWER than single-env**, the opposite of the design's prediction.
- **`benchmark_parallel_l2_v2.py`** — diagnostic follow-up. Live `ps aux` during the first run
  showed **~375% CPU per worker process** — textbook thread oversubscription (each worker's
  torch/BLAS backend defaulting to multi-threaded ops, N processes × multiple threads each
  competing for only 16 physical cores). Capping each worker to one thread
  (`torch.set_num_threads(1)` + `OMP_NUM_THREADS`/`MKL_NUM_THREADS=1`) is the fix tested here,
  and this exact env-var pattern became **mandatory, not optional**, throughout every subsequent
  heavy script in the project (`train_l2.py`, `eval_l2_n500.py` and its whole family,
  `calibrate_impact.py`, every `benchmark_*`/`profile_*` script below).
- **`benchmark_controlled.py`** — the rigorous, trustworthy version: fixed seed, fixed 10-day
  date pool (both variance sources the first two runs left uncontrolled), 3 trials per
  `n_envs ∈ {1,2,4,8}`, real RSS (`/proc/<pid>/status`) and VRAM (`nvidia-smi`) measurement, not
  estimated. Reports mean/stdev/CoV%, speedup, and per-`n_envs` parallel efficiency.
- **`benchmark_controlled_numeric.py`** — identical methodology, numeric-format data instead of
  parquet/JSON — the number this project actually planned real training around. Extrapolates
  the measured `n_envs=8` rate to a full 2,000,000-step run duration, explicitly comparing
  against a stated parquet-format baseline of **1.84 days**.
- **`benchmark_realistic_pool.py`** — a cheap, bounded companion using the **real 405-day** train
  split instead of the narrow 10-day pool, to see how a realistic (much lower) day-cache hit
  rate actually shifts throughput — only `n_envs ∈ {1,8}`, 1 trial each, explicitly *not* a
  replacement for `benchmark_controlled.py`'s trustworthy 3-trial numbers, just a directional
  check of which way and roughly how far cache pressure moves things.

### `env.reset()` cost investigation

- **`profile_reset.py`** — coarse, monkeypatch-based breakdown of `reset()` into `_load_day`
  (tagged hit vs. miss), `_build_ticks`, `_precompute_feature_series`, and "other," across two
  scenarios: the narrow 10-day benchmark pool (cache holds 5/10) and the real 405-day train
  split (cache holds 5/405, near-always-miss) — quantifying exactly how much of `reset()`'s cost
  is genuinely data-load-bound at production scale vs. an artifact of the narrow benchmark pool.
- **`profile_reset_cprofile.py`** — finer-grained cProfile pass specifically to see which
  sub-computation inside `_precompute_feature_series` dominates.
- **`profile_l2_throughput.py`** — end-to-end decomposition of full L2 training wall-clock into
  `l3_predict`/`env_step`/`env_reset`/`sac_train`, via monkeypatch instrumentation of the real
  production code path (`make_l2_wrapped_env`, the real frozen checkpoint — SHA-256-verified
  against the recorded handoff-doc value before running, not trusted from a filename alone).

### Parquet I/O investigation (the source of `LOBExecutionEnv._NEEDED_DAY_COLUMNS`)

- **`measure_column_pruning.py`** — full-column vs. 7-needed-column `pd.read_parquet`, across 5
  real days, each variant timed after an identical page-cache warm — isolates decode cost from
  disk I/O variance. **Result: only ~0-4% real gain** — a much smaller win than pruning might
  suggest on paper.
- **`measure_column_split.py`** — the finding that explains why: splits the 7 needed columns
  into "5 small numeric/ts columns" vs. "bids+asks (JSON strings)" and times each separately (plus
  pyarrow `use_threads=True/False`). **Confirmed: bids/asks JSON-string decode is ~97% of the
  real cost** — pruning the other, cheap columns barely moves the total, which is exactly why
  column pruning was kept anyway (free, zero behavior risk) but never treated as *the* fix — the
  numeric-format conversion (§4.9), which eliminates JSON parsing entirely, was.
- **`measure_predicate_pushdown.py`** — tests empirically (not assumed from format knowledge
  alone) whether pyarrow can skip decoding rows outside a `ts`-range filter within a single row
  group, via a real column-index metadata check plus a timed full-vs-filtered read comparison.

### Reset-vectorization correctness gates

- **`capture_reset_snapshot.py`** / **`compare_reset_snapshots.py`** — the general-purpose
  before/after-a-code-edit snapshot-diff pair: fixed seeds, a fixed observation-independent
  action every tick (so any divergence can only come from `reset()`/`step()`'s own computation,
  never from a policy reacting to a slightly-different observation), byte-exact comparison via
  `np.array_equal`. Used to prove the `_rolling_*`/`_vec_qty_at_price` vectorization (§4.1) was
  behavior-preserving before trusting it in production.

**Thread-capping coverage note**: most of these scripts apply the `OMP_NUM_THREADS`/
`MKL_NUM_THREADS=1` pattern; the earliest ones in this list (`benchmark_parallel_l2.py`, before
the oversubscription finding existed) and the pure-numpy `measure_*`/`convert_one_day*` scripts
(single-threaded by nature, no torch/BLAS parallelism to cap) do not need it and don't apply it.

## 4.11 `scripts/` — Evaluation & Statistical-Analysis Harnesses

These are the scripts that produced every headline number in `docs/reports/`. Import graph is
rooted at `phase2a_sanity_suite.py`; `eval_l2_n500.py` is the second root most others build on.

```
phase2a_sanity_suite.py  (TWAPPolicy, NoOpPolicy, OraclePolicy, run_episode, paired helpers)
        │
        ├── eval_l2_n500.py  (paired_report w/ Cohen's d_z, run_arm, run_wrapped_episode,
        │   │                 make_l2_policy_action_fn, EVAL_SEED_BASE/HORIZON_TICKS consts)
        │   ├── eval_l2_diagnostics.py
        │   ├── eval_l2_bucketed.py
        │   ├── eval_l2_test_confirmation.py
        │   ├── analyze_l2_reward_components.py
        │   └── analyze_l2_reward_components_v2.py  (imports src.envs.l2_reward directly instead)
        │
        └── replace_value_probe.py  (its OWN separate paired_report, no d_z)
                └── replace_value_probe_n500.py  (imports paired_report FROM replace_value_probe.py)
```

**`compare_l2v3_vs_l2v2predivergence.py`** does not import from either `paired_report` — it
reimplements the same paired t-test + Wilcoxon + Cohen's d_z formulas as its own local function.
Its own docstring says it "reuses `paired_report`'s exact **methodology**," which is accurate —
the formulas match `eval_l2_n500.py`'s version exactly — but the *code* is a third independent
copy, not an import. **Net effect: this codebase has three separate implementations of
essentially the same paired-comparison statistical helper** (`eval_l2_n500.py`'s, with d_z;
`replace_value_probe.py`'s, without d_z; and this file's own, with d_z) — worth knowing before
adding a fourth eval script that needs one.

### `scripts/phase2a_sanity_suite.py` (12,241 bytes — the foundational eval script)

Documented in its own header as "a Phase 2a throwaway evaluation script — NOT a permanent
module," yet it is the single most-imported file in this whole layer. **`TWAPPolicy`**: posts a
passive LIMIT at the touch (`offset_idx=5` → offset 0, a literal, unconditional constant on
*every* placement — confirmed directly from source, this is the fact that makes TWAP's price
offset a hard zero-entropy quantity, later load-bearing for the predictability-analysis
encoding choice in §4.12), forces MARKET completion only at slice-end if the slice's target
isn't yet met. **`NoOpPolicy`**: never trades — exists purely to prove the IS decomposition's
zero-fill edge case end-to-end. **`OraclePolicy`**: cheats by pre-scanning the entire episode
window at `reset()` to find the single best mid-price tick, then executes everything there in
one MARKET order — a near-optimal-execution sanity ceiling, not a real baseline.
**`run_episode(env, policy, seed, horizon_ticks)`**: the shared episode-running loop nearly
every other eval script reuses. `main()`'s own sanity suite (Checks A/B/C) is itself a real,
runnable correctness gate: no-op fill_ratio must be exactly 0 with `is_exec_bps=None`; oracle
must beat TWAP; TWAP must fill 100% of every episode by construction.

### `scripts/eval_l2_n500.py` (the shared "library" for every later eval script)

Three-arm design, all paired on the same `EVAL_SEED_BASE=5,000,000` seed list: **Arm 1** = the
trained L2 SAC policy steering frozen L3 via `FrozenL3Wrapper`. **Arm 2** = "TWAP-passthrough" —
L2 always emits `[1.0, 0.5]`, i.e. frozen L3 completely unsteered (answers "does learned steering
beat no steering"). **Arm 3** = pure `TWAPPolicy` on the *base* `LOBExecutionEnv`, no L3/L2 at
all — directly poolable with the project's original TWAP baseline number (0.889bps) since it's
the identical policy/methodology. **Pre-registered success bar, stated in the module docstring
before any real result existed**: Arm 1 must beat Arm 2 with **both** a paired t-test **and**
Wilcoxon signed-rank agreeing (`p<0.05`, same direction) — this project had already been misled
once by a single significant test with a practically-irrelevant effect size (Cohen's d_z=0.076),
and three separate times by underpowered n=50 reads that reversed at proper power, so this bar
exists specifically to prevent a repeat.

**`paired_report(name_a, r_a, name_b, r_b)`**: paired t-test (`scipy.stats.ttest_rel`) + Wilcoxon
(`wilcoxon`) + Cohen's d_z (`mean_diff/std_diff`) — this is the canonical version every later
script either imports or re-derives.

### `scripts/eval_l2_diagnostics.py` — three diagnostics reusing `eval_l2_n500.py`'s machinery

1. **Action distribution** (Arm 1 only): logs every real `participation_mult`/`urgency` L2
   emits, computing within-episode std (does it respond to state mid-episode?) vs. between-
   episode std of per-episode means (does it differ by day at all?) — this is the mechanism
   that ruled out policy-collapse as an explanation for L2's negative result.
2. **Train-vs-val gap**: `--split {val,train}` re-evaluates against either real, frozen date
   pool via `load_split()` — confirmed the real training run used
   `train_date_range=('2024-04-18','2025-07-15')` (405 days) / `val_date_range=('2025-07-16',
   '2025-08-02')` (18 days), chronologically disjoint (pulled directly from that run's own
   startup log).
3. **Per-day breakdown**: captures each episode's picked calendar day via a safe, inert
   monkeypatch on `LOBExecutionEnv._load_day`/`_load_day_numeric` — lets the aggregate n=500
   result be split by day to check whether a negative finding is broad-based or driven by a
   handful of regime-specific days.

Its own docstring states an exact, checkable reproduction target: re-running at `--split val`
with the same checkpoints/seeds/n=500 must reproduce `eval_l2_n500.py`'s already-recorded arm
means (**L2=1.2330, TWAP-passthrough=1.0237, Pure TWAP=0.8893**) exactly, since every component
is deterministic given a seed — any mismatch would mean this script introduced a bug, not a
real behavior change.

### `scripts/eval_l2_bucketed.py` — volatility-stratified evaluation ON TRAIN DAYS

Follow-up to the finding that L2's aggregate advantage over passthrough collapsed from
`-0.253` to `-0.013` bps when restricted to val's own (calm) volatility range: does it instead
*strengthen* above that range, where val has no equivalent days at all? **Explicit, prominent
caveat in the module docstring**: any edge found here **cannot be distinguished from
memorization**, since these are training days the policy was directly optimized against for
2,000,000 steps — this script can only say whether an edge exists on these *particular seen*
days, never whether it would transfer to unseen volatile conditions. Restricts the env's file
pool via a safe post-construction override of `LOBExecutionEnv._files` (set before any `reset()`
is called, so it cannot perturb the RNG draw order — equivalent to having constructed the env
with that file set from the start). Bucket boundaries (`calm`/`moderate`/`high`) are defined
relative to `VAL_MAX_VOL_BPS = 0.1882`, read from a pre-computed day-conditions CSV.

### `scripts/eval_l2_test_confirmation.py` — THE ONLY EVALUATION THE TEST SPLIT EVER GETS

One run, no re-runs, no parameter adjustments after seeing the number — the module docstring is
explicit that the pre-registered claim (committed in `docs/TRACK_STATUS.md` at a specific,
named commit) predates this script even being built. Adds an action-type distribution
(HOLD/LIMIT/MARKET/CANCEL_AND_REPLACE — L3's own discrete component) via the same safe
monkeypatch-on-`step()` pattern, and reports episodes-per-day explicitly (`n / n_days`) as a
standing reminder that 500 episodes over 18 real days are not 500 independent samples.

### `scripts/analyze_l2_reward_components.py` / `_v2.py` — the credit-assignment measurements

`_v1` (old `l3_passthrough` mode): monkeypatches `lob_execution_env`'s own **local name
bindings** for `step_reward`/`compute_implementation_shortfall` — explicitly *not*
`src.envs.reward`'s names, since `from X import Y` binds a local name at import time, so
patching `X.Y` would never intercept `lob_execution_env.py`'s own call site. Recomputes every
component independently from the same captured arguments using reward.py's own formulas, and
**asserts the recomputed sum matches the real return on every single call** — any transcription
mistake would surface immediately as an assertion failure, not a silently wrong headline number.
This is the script that produced the finding driving the whole L2-reward-redesign round: under
the old aggregation, **`r_stale` was 85.6% of L2's net reward and 75.4% of its signal
magnitude**, while terminal-IS-derived signal (the metric L2 is actually scored on) was only
**6.9%/11.6%**.

`_v2` (new `potential_is_shaping` mode): companion measurement, patching `src.envs.l2_reward`'s
own local binding instead. Decomposes the new signal's *internal* composition
(`exec_contribution`/`is_opp_bps`/`fees_bps`, as per-window deltas) rather than the old six
components, since those no longer exist under this mode. Confirms the redesign's own claim
numerically: **100% of L2's reward is now terminal-IS-derived** (vs. 6.9%/11.6% before).

### `scripts/analyze_l2_relative_comparison.py` — cross-split significance, done correctly

Answers "does L2's edge over baseline differ between train and val" using each split's own
**L2-minus-baseline difference**, compared across splits with **Welch's t-test + Mann-Whitney U**
(*independent*-samples tests, since the two difference distributions come from disjoint episode
pools — not a paired test) plus Cohen's d (pooled-std formula, the independent-samples version,
distinct from `eval_l2_n500.py`'s paired d_z). This is the corrected version of an earlier
mistake in the project's own history: comparing each split's *absolute* IS_total_bps score
directly (rather than each split's own edge over its own baseline) hid a real overfitting signal
because the reference point itself had shifted between splits.

### `scripts/analyze_split_representativeness.py` — is val a regime artifact of train?

Pure market-data descriptive statistics (`day_return_bps`/`realized_vol_bps`/`mean_spread`,
computed directly from each day's `mid_price`/`spread` series via `read_day()`) — **no model
inference at all**, which is explicitly why this script is allowed to touch all three splits
including test: reading a raw price series to compute its own volatility doesn't draw any
model/policy conclusion from held-out data, unlike actually running an eval harness against it.
Reports where val's mean sits within train's percentile distribution, per-day percentiles, and a
Mann-Whitney U test for distributional difference. This is the script whose output
(`models/l2_day_conditions_{train,val,test}.csv`) directly feeds `eval_l2_bucketed.py`'s bucket
boundaries.

### `scripts/compare_l2v3_vs_l2v2predivergence.py` — direct checkpoint-vs-checkpoint

Pure post-hoc analysis of two *already-completed* n=500 CSV outputs (no new episodes run):
isolates whether L2v3's stable critic (gamma=0.983) produced a genuinely *better* policy or just
a numerically calmer one converging to the same place as L2v2's pre-divergence checkpoint
(gamma=0.995, step 1,599,936) — same reward, gamma the only real difference, matched by seed.

### `scripts/replace_value_probe.py` / `scripts/replace_value_probe_n500.py` — does CANCEL_AND_REPLACE ever help?

Hand-written heuristic policies, **no RL, no training, no GPU, no model loading** — settles
whether higher CANCEL_AND_REPLACE usage would even *help*, a question four rounds of reward
engineering had treated as "fix the near-0% usage" without ever testing. `PassivePolicy(offset)`:
one LIMIT at a fixed offset, full size, then HOLD until filled. `ReplaceActivePolicy(initial_
offset, staleness_n, step)`: escalates the offset by `step` (capped at guaranteed-crossing) every
time an order has rested unfilled for `>= staleness_n` ticks. The first script sweeps 7 PASSIVE
offsets × 18 REPLACE-ACTIVE configs (3 initial offsets × 3 staleness thresholds × 2 step sizes)
at n=50, applying a **Bonferroni correction** (the only file in the project using one:
`alpha = 0.05 / 18`) to the post-hoc-selected best-B-vs-best-A comparison, since sweeping 18
configs and picking the best is exploratory, not a single pre-registered test. `_n500.py` is the
adequate-power follow-up: at n=50 the original best-B-vs-TWAP comparison had only ~14.7% power
to detect its own observed effect (std_diff=3.71); n=500 gives ~83% power for that same effect
size. Deliberately re-tests **only** the single pre-selected config (`init=-5, N=100, step=1`)
against TWAP — no re-sweep, keeping this a clean confirmatory test rather than a second
multiple-comparisons search. Also confirms a structural finding along the way: no PASSIVE
configuration reaches TWAP/REPLACE-like fill rates at all (40.4% at offset=0 is a ceiling, not a
sweep gap), so a fill-fair PASSIVE-vs-REPLACE comparison cannot be constructed and isn't
attempted.

## 4.12 `scripts/replay_episode.py` + `scripts/analyze_predictability.py`

Both self-authored during this project's later rounds; full function-level detail (capture
mechanism, `reconstruct_child_orders()`, the fragmentation-aggregation fix, the encoding-fairness
reasoning behind the predictability classifier) is preserved verbatim in this document's own
§4.12 subsection below, drawn from direct authorship rather than a fresh read.

### scripts/replay_episode.py (714 lines)

**Purpose**: Makes one episode of execution legible to a non-code reader — a trading/finance
audience, not a codebase reader. Runs the real `FrozenL3Wrapper`/`LOBExecutionEnv` completely
unmodified and captures tick-level detail via monkeypatching, then renders it as a set of
matplotlib figures: the price path with every child order placement marked, execution progress
against the on-schedule TWAP pace, and (when an L2 policy is driving) L2's own steering decisions
over time, each labeled with its exact chosen value. Two episodes always run at the *same seed* for
direct comparison — the real policy under test, and `phase2a_sanity_suite.TWAPPolicy` on the base
env — so the two Implementation Shortfall numbers are directly comparable, not just similar in
spirit.

**Capture mechanism** (`EpisodeCapture` dataclass + `install_tick_capture()` /
`install_capture()`): monkeypatches `LOBExecutionEnv.step()` (and, when wrapped, also
`FrozenL3Wrapper.step()`) so the *original* method still runs and returns its real result
unchanged — the patch only records additional data as a side channel. This capture is installed
only inside this script's own process and is inert for every other caller (training, evaluation,
anything else). `install_tick_capture(base_env, capture=None)` is the standalone half — installable
on *any* `LOBExecutionEnv`, not just one sitting inside a wrapper — factored out specifically so
`scripts/analyze_predictability.py` could reuse it on a bare base env for the pure-TWAP arm, where
there is no `FrozenL3Wrapper` to speak of. `install_capture(wrapped_env)` calls
`install_tick_capture()` internally and then layers a second patch on `wrapped_env.step()` to also
record L2's own decisions (`participation_mult`, `urgency`, the resulting
`l2_target_slice_ratio_override`). **Important gotcha for any future caller**: `install_capture()`/
`install_tick_capture()` must be called exactly ONCE per env instance — calling it again nests a
second instrumented layer around the first (double-recording, leaked closures) instead of
replacing it. To run many episodes on one persistent env (day-cache reuse), build the env and
install capture once, then call `capture.tick_records.clear()` (and `.l2_decision_records.clear()`
if applicable) before each episode's `.reset()` — this exact pattern is what
`analyze_predictability.py` does.

**`reconstruct_child_orders(tick_records, side)`** — the other half of this file's real value,
reused by `analyze_predictability.py` unmodified. Walks the tick-level capture and regroups it
into discrete child-order lifetimes (placement → outcome), mirroring `LOBExecutionEnv.step()`'s own
resting-order state machine exactly (evolve the existing resting order against market activity →
apply an explicit cancel → only then decide whether this tick's action places something new).
Returns a list of `ChildOrder` objects: `kind` ("resting" or "market"), `placement_tick`,
`placement_price`, `offset_from_touch`, `outcome` ("filled" / "replaced" / "open_at_episode_end"),
`fill_ticks`/`fill_qtys`/`fill_prices`, and `placed_size` (the quantity requested AT placement, not
just what ended up filled — for resting orders this is the env's own
`info["resting_own_remaining"]` captured on the placement tick; for market-kind orders it's
`sum(fill_qtys)`, since those fill immediately and in full against available depth by
construction). The function's own docstring documents a real bug it replaced: an earlier version
created a new `ChildOrder` for *every* recorded LIMIT/CANCEL_AND_REPLACE tick regardless of whether
the env's own resting slot was already occupied, fabricating phantom placements and mislabeling
still-resting orders "replaced" every time L3 (which was never trained to prefer HOLD once resting)
kept emitting LIMIT while already resting.

**Known limitation, discovered and worked around downstream, not here**: this function creates one
`ChildOrder` per *fill event*, by design — useful for this script's own visualization (each fill
deserves its own marker on the price chart) but means a single market order that walks several
thin book levels shows up as several same-tick "orders." `analyze_predictability.py`'s
`aggregate_placement_events()` exists specifically to undo this for placement-*pattern* analysis;
this file itself does not need to, since its own use (drawing markers) is exactly the case where
per-fill granularity is correct.

**Figure-building** (`build_figure()` for a combined 3-panel overview, `build_separate_figures()`
for larger standalone versions of each panel — added specifically so a reader isn't stuck
interpreting a cramped combined figure): three panels — price path + child order placements
(`_draw_price_panel`, full comma-formatted prices via an adaptive-precision `_price_formatter()`,
never scientific/offset notation; decision-boundary vertical lines labeled D1/D2/... via
`_mark_decisions()`), execution progress vs. on-schedule TWAP pace (`_draw_execution_panel`), and
L2's own steering values (`_draw_steering_panel`, each decision's exact chosen participation-rate
multiplier and urgency labeled directly on the chart, with legend-corner-collision avoidance for
the common case of an early decision sitting at the 2.0 ceiling). A shared summary text box
(`_summary_lines()`/`_add_summary_box()`) reports fill_ratio, the full IS decomposition, and which
arm "beat" the other on that single episode — explicitly labeled "not a significance test."

**CLI**: `--l2-checkpoint`/`--l2-vecnormalize` (omit both, and pass `--frozen-l3-only` instead, to
drive the episode with the constant TWAP-passthrough action — frozen L3, completely unsteered),
`--l3-checkpoint`/`--l3-vecnormalize` (required), `--seed` (required, no default — "pick one
deliberately"), `--use-numeric-format`, `--output` (a path *stem* — the script always writes 4
files: `{stem}.png` the combined overview, plus `{stem}_price.png`/`_execution.png`/`_steering.png`
standalone versions).

---

### scripts/analyze_predictability.py (547 lines)

**Purpose**: A late follow-up analysis (not part of the original architecture spec) asking
whether frozen L3's order-placement pattern is measurably less predictable than pure TWAP's —
motivated by TWAP's well-known real-world weakness of being trivially predictable/exploitable by
adversarial flow. Its own module docstring states the scope limit prominently and repeatedly: this
measures a *property* of the two policies via descriptive statistics and a classifier, not a
payoff, because `LOBExecutionEnv` has no adversarial participants and no market impact from the
agent's own orders (see §1.3 of this guide) — a predictability gap found here is not evidence it
would pay off against real counterparties.

**Two tasks, one script**:

*Task 1 — descriptive regularity metrics.* Reuses `reconstruct_child_orders()` and
`install_tick_capture()` from `scripts/replay_episode.py` (imported, not reimplemented) to
reconstruct discrete placements for both arms across the same 500 paired val episodes used
throughout this project's other evaluations. Adds `aggregate_placement_events()` — merges
same-tick "market"-kind fragments (see the note under `reconstruct_child_orders()` above) into one
real placement event before computing any metric, since a matching-engine artifact of thin book
depth is not the same thing as a separate policy decision. Computes, for both arms: inter-placement
tick gaps, placed order sizes, and price offsets from the touch (offsets scoped to `kind=="resting"`
orders only — a true MARKET action's offset isn't a chosen/meaningful value). Reports both *pooled*
distribution statistics (mixes within-episode rhythm with between-episode variation) and
*per-episode* coefficient of variation, averaged across episodes (the more precise "how metronomic
is each episode's own rhythm" measure) — `_dist_stats()`, `_per_episode_cov()`. A decision gate in
`main()` compares the two arms' per-episode gap-timing CoV; if they look similar (within 15%
relative), Task 2 is skipped (with `--force-task2` available to override) — in the one real run
this was launched for, they were clearly different (L3 CoV≈2.27 vs. TWAP CoV≈0.79) and Task 2 ran.

*Task 2 — direct predictability test.* A deliberately shallow `RandomForestClassifier`
(`n_estimators=50, max_depth=8`) predicts the next tick's `order_type` from a small, symmetric
feature set built identically for both arms (`build_features()`): 1- and 5-tick mid-price returns,
spread level and 5-tick change, own quantity-remaining fraction, own elapsed-time fraction, ticks
since the policy's own last fill/placement (capped/normalized), and the current tick's own
`order_type` (an autocorrelation feature). Every feature uses only information available up to and
including the current tick — no lookahead. Train/test split is by *episode*, not by tick (ticks
within one episode are highly autocorrelated, so a tick-level split would leak): the first 400
episodes are train, the last 100 are test, both arms, same seeds. **The one place fairness required
a genuinely deliberate, asymmetric choice, stated explicitly in the module docstring**: `order_type`
alone (not the full 3-part action including price offset) is the *primary* target, specifically
because TWAP's own price offset is a hardcoded constant (confirmed directly from
`TWAPPolicy.act()`'s source — `offset_idx=5` on every single placement, no exceptions) — folding it
into the primary target would make TWAP's measured "predictability" partly an artifact of a
zero-entropy label component baked in by construction, not a genuine finding about flow
predictability. Offset and size are still reported as separate, secondary classifiers (conditional
on a placement of the relevant kind actually happening at *t+1*), specifically so that mechanical,
by-construction predictability stays visible and separate from real timing/flow signal rather than
being blended into one number (`task2_predictability()`, `train_and_eval()`,
`_majority_baseline_acc()`).

**Real result from the one full run** (`docs/reports/l3_execution_predictability_report.md` has
the complete numbers): TWAP's `order_type` sits at 99.65% test accuracy — 0.18 points above its
own already-near-ceiling majority baseline, i.e. genuinely at ceiling. L3 sits at 78.31% — a real
20.3 points above its own (much lower, 58.0%) majority baseline, so not pure noise, but leaving a
substantial 21.3-point gap to TWAP's ceiling that never closes. The secondary classifiers sharpen
this: TWAP's offset is perfectly predictable (100.00%/100.00%, the mechanical artifact described
above), while L3's is barely above baseline (+1.3 points) — close to unpredictable from this
feature set.

**CLI**: `--l3-checkpoint`/`--l3-vecnormalize` (required), `--n` (default 500), `--test-episodes`
(default 100 — the last this-many, by seed order, become the classifier's test split),
`--classifier-seed`, `--force-task2`, `--output-dir` (required — writes
`predictability_result.json` with the full numeric output of both tasks).

**A real bug found and fixed while smoke-testing this script before the full run** (worth knowing
if extending it): building the wrapped env and installing capture *inside* the per-episode loop
(rather than once, before it, reused via `.reset()`) would have both discarded the day-cache reuse
every other n=500 script in this project relies on for throughput, and — more seriously — re-
patched an already-patched `step()` on each iteration, silently nesting instrumented layers. Fixed
by building each env and installing capture exactly once (`make_l3_capture_env()`,
`make_twap_capture_env()`), then clearing and re-reading the capture's records per episode inside
`run_l3_episode()`/`run_twap_episode()`.

**New dependency**: `scikit-learn==1.9.0`, installed and added to `pyproject.toml` specifically for
this script's classifier — not used anywhere else in the codebase.

## 4.13 `tests/` — The Test Suite

**Ground truth** (re-run directly for this document, not quoted from memory):
`PYTHONPATH=. .venv/bin/python -m pytest tests/ -q` → **195 collected, 191 passed, 4 failed,
163.38s (2:43)**. All 22 files, all real. Below, each file's scope, notable regression-test
provenance, and whether it needs real data/GPU/network.

### The 4 pre-existing, unrelated-to-current-work failures (confirmed exact, re-run fresh)

**3 in `tests/test_bulk_backfill.py`** (`test_manifest_skips_already_ok_entry`,
`test_retries_on_transient_5xx_then_succeeds`, `test_404_on_checksum_marks_missing_and_not_
retried`) — all three fail identically: the test file's `responses`-library mocks register URLs
shaped like `.../BTCUSDT/2024-01-15/.CHECKSUM` (a dated subdirectory), but
`scripts/bulk_backfill.py`'s real `_checksum_url()`/`_zip_url()` build a **flat** shape,
`.../BTCUSDT/BTCUSDT-trades-2024-01-15.zip.CHECKSUM` (dataset and date embedded in the
filename, no subdirectory) — confirmed directly by reading both the test fixtures and the real
URL-building functions side by side. The mock and the real code have simply diverged; the tests
were not updated when the URL scheme changed (or vice versa).

**1 in `tests/test_l2_capture.py`** (`test_reconstruct_book_with_resync_on_gap`) — exact
assertion failure: `assert 101.0 == 101.5`. The test calls `apply_diff()` directly (not through
`L2CaptureDaemon._apply_event()`) with a sequence that skips update `u=3`, and expects the ask
side to have picked up the new level from the `u=4` diff. But `apply_diff()` — confirmed by
direct reading — has **no gap-detection logic of its own at all**; it unconditionally merges
whatever bid/ask entries it's given, and does not clear a stale level that a skipped update
would have removed. Gap detection is a caller's responsibility (see `L2CaptureDaemon.
_apply_event()`'s `pu != last_u` check, or `collect_l2_bybit.py`'s own one-layer-up gap check) —
this test's premise (that `apply_diff` itself self-heals a gap) doesn't match how the function
is actually designed. A genuine, longstanding test/implementation mismatch, not a regression
from recent work.

### Files read in full (highest architectural value)

- **`test_matching_engine.py`** (16 tests) — every expected value hand-computed in the test's
  own comments, not derived by running the code and asserting whatever it produced. Includes
  `test_sequential_decomposition_matches_spec_combined_formula`, which directly checks
  `update_queue`'s cancel-then-trade decomposition against the architecture spec's single
  combined formula across 5 hand-picked tuples including a gross-over-depletion case.
- **`test_reward.py`** (17 tests) — hand-computed fixtures for every `RewardWeights` term,
  including a full walk through the EXPERIMENTAL-4 direction inversion:
  `test_step_reward_replace_off_book_now_costs_full_gamma` is a direct regression test named
  after (and superseding) an *older* test that asserted the opposite (`..._costs_nothing`) —
  the test file itself documents the sign flip, not just the reward code.
- **`test_wrappers.py`** (22 tests) — the largest test file relative to its source file's
  complexity. Two real, named regression tests: `test_l3_deterministic_flag_controls_inner_
  predict_determinism` and `test_reset_clears_l2_target_slice_ratio_override_and_urgency`,
  both citing the exact bug they guard against in their own docstrings (see §4.1's wrapper
  corrections 3/4). One gated integration test
  (`test_integration_smoke_real_checkpoint`, skipped unless both the real checkpoint file
  exists **and** a live GPU-headroom check passes).
- **`test_orchestrator_graph.py`** (11 tests) — the two full-stack integration tests
  (`test_full_stack_integration_short_bounded_episode`, `test_full_stack_async_integration_
  idx_17_18_change_and_threads_clean_up`) are the richest in the suite: they load the *real*
  frozen L3 checkpoint (SHA-256-verified live against the recorded handoff-doc value) and the
  real L2 smoke-test SAC checkpoint, run a genuine bounded episode, and assert cadence
  invariants empirically via an external, non-invasive tick-recording monkeypatch — not just
  "it doesn't crash." One documented, benign quirk found and explained rather than silently
  worked around: `info["ticks_elapsed"]` under-counts by exactly one on the specific tick that
  triggers horizon-truncation, because `LOBExecutionEnv.step()`'s end-of-buffer clamp runs
  after `truncated` is computed `True` but before `_build_info()` reads `_tick_idx`.

### Remaining 18 files (scope, by source module — docstrings and test names extracted, not read line-by-line)

| Test file | Tests | Covers | Data/gating |
|---|---|---|---|
| `test_sanity.py` | 1 | trivial harness-alive check | none |
| `test_features.py` | 2 | `src/data/features.py` — hand-computed `mid_price`/`micro_price`/`obi`, `CancelAddTracker` window rollover | synthetic |
| `test_download_manager.py` | 1 | `src/data/download_manager.py` — `bulk_download` on an empty range | synthetic |
| `test_l1_features.py` | 15 | `src/data/l1_features.py` — hand-computed fixtures matching the real `data/raw_l1` column schemas (not the live archive itself); covers monthly/daily klines fallback, OI dedup, all-`None` degenerate case | synthetic, self-contained |
| `test_l1_macro_analyst.py` | 10 | `src/agents/l1_macro_analyst.py` — throttle timing, fail-closed on every error type, proxy bypass, structured-schema payload shape | `requests.post` always mocked, no real network |
| `test_l2_capture.py` | 1 | `src/data/l2_capture_daemon.py` — the one failing gap-resync test above | synthetic |
| `test_split.py` | 9 | `src/data/split.py` — both the **real** `data/raw_l2_bybit/BTCUSDT/` directory (validates the persisted artifact against actual disk state) and synthetic fixtures for boundary-logic edge cases (gaps, insufficient files, artifact-mismatch refusal) | real dir + synthetic |
| `test_lob_execution_env.py` | 4 | `LOBExecutionEnv.reset()`'s file-selection/`date_range` reproducibility specifically | synthetic parquet days |
| `test_lob_execution_env_features.py` | 14 | Phase 2b observation features (idx 4-5, 10-19, 39-41) hand-computed, plus `test_full_vector_completeness`, `test_qty_at_price_no_false_match_beyond_half_tick` (regression for the `rtol` bug), `test_crossing_limit_placement_fills_immediately_not_a_resting_order` (regression for the crossing-order bug) | synthetic, deliberately shaped so rolling windows are fully populated regardless of the random episode start |
| `test_reset_vectorization_equivalence.py` | 5 | the 4 vectorized rolling-window helpers, each checked against a **verbatim-reproduced original Python-loop reference** (the old code no longer exists in source once replaced) | synthetic |
| `test_numeric_format_equivalence.py` | 4 | `write_day`/`read_day` round-trip byte-identity, missing-array rejection, full env comparison original-vs-numeric-format on a tiny synthetic day built in both formats, glob-pattern correctness | synthetic |
| `test_twap_baseline_reward.py` | 4 | `RewardWeights.subtract_twap_baseline` + `_compute_twap_shadow_terminal_is()` — verifies exact algebraic relationships between paired runs rather than hand-simulating the matching engine a second time; `test_subtract_twap_baseline_matches_real_twap_policy_exactly` is the integration test guarding the shadow-computation duplication against drift from the real `TWAPPolicy` | synthetic + real `TWAPPolicy` import |
| `test_l2_reward.py` | 7 | `src/envs/l2_reward.py` — hand-computed `l2_potential`/`l2_window_reward` fixtures, plus `test_potential_is_shaping_telescopes_exactly_on_real_episodes` (the hard gate the whole redesign's correctness claim rests on) and a reward-scale sanity check vs. `l3_passthrough` | mixed: pure-function tests synthetic; the telescoping test needs the **real** frozen L3 checkpoint + real numeric data, `@pytest.mark.skipif`-gated on both being present |
| `test_replay_episode.py` | 7 | `scripts/replay_episode.py::reconstruct_child_orders()` — the one piece of real logic in that script (everything else is plotting); hand-built tick-record fixtures covering all 4 outcome types the price panel distinguishes | synthetic, no env/model |
| `test_eval_l2_n500.py` | 4 | `scripts/eval_l2_n500.py`'s own machinery — synthetic data, tiny/untrained models, CPU-only, n=3; proves the harness runs end-to-end, explicitly does **not** assert anything about untrained-model IS values (there's no reason a random policy should beat or lose to anything meaningfully) | synthetic, deliberately not run against real checkpoints (a live 24h training run must not be disturbed by real-data I/O) |
| `test_train_l2.py` | 21 | `src/train/train_l2.py` — `ValISEvalCallback`, `_resolve_gradient_steps`, `resolve_l2_final_save_paths`, and a full CLI-defaults/flag-interaction table (`test_cli_*`) | synthetic, tiny untrained `RecurrentPPO` stand-in, no GPU, no real checkpoint |
| `test_train_l3.py` | 4 | `resolve_final_save_paths()` only, deliberately **not** `main()` itself (needs real config/data/GPU to construct) | none — pure path-decision logic |

### Coverage gaps (files in `src/`/`scripts/` with no corresponding test file)

`src/analysis/calibrate_impact.py` (standalone, self-verifying via its own holdout split, no
pytest coverage), `src/agents/orchestrator_graph.py`'s `strategist_tick()` specifically (covered
only indirectly through the full-stack integration tests, no isolated unit test), and — as
expected for one-off investigation tooling, not a gap worth closing — every `scripts/`
throughput/profiling/benchmark script in §4.10, plus the eval/analysis harnesses in §4.11 beyond
`eval_l2_n500.py`'s own mechanics test.

## 5. Connections & Dependencies

### 5.1 Data flow, top to bottom

```
data/raw_l2_bybit/BTCUSDT/*.parquet  (original archive, 34GB/441 days, JSON-string book levels)
        │  read by
        ▼
src/data/l2_numeric_format.py::write_day()  ◄── scripts/convert_l2_to_numeric_parallel.py
        │  produces
        ▼
data/raw_l2_bybit_numeric/BTCUSDT/*.npzst  (production training input, use_numeric_format=True)
        │  read by
        ▼
src/envs/lob_execution_env.py::LOBExecutionEnv._load_day_numeric() / read_day()
        │  builds
        ▼
TickView objects  →  _precompute_feature_series()  →  _build_obs() (42-dim vector, _OBS_SPEC)
        │
        ├─ obs idx 17/18 ◄── env.l1_risk_score / l1_confidence ◄── src/agents/orchestrator_graph.py::macro_tick()
        │                                                          ◄── src/agents/l1_macro_analyst.py::L1MacroAnalyst
        │                                                          ◄── src/data/l1_features.py::build_l1_feature_summary()
        │                                                              (data/raw_l1/*, NOT wired into any real training run)
        │
        ├─ obs idx 15/16 ◄── env.l2_target_slice_ratio_override / l2_urgency
        │                    ◄── src/envs/wrappers.py::FrozenL3Wrapper.step() (real L2 training/eval)
        │                    ◄── default linear-TWAP formula (L3 solo training/eval — L2 stubbed)
        │
        ▼
L3 policy (RecurrentPPO) . predict(normalized_obs, lstm_state, episode_start)
        │  action: MultiDiscrete([4, 11, 5])
        ▼
LOBExecutionEnv.step(action)
        │  ├── src/envs/matching_engine.py (update_queue, walk_market_fill)
        │  └── src/envs/reward.py::step_reward() / compute_implementation_shortfall()
        ▼
(obs, reward, terminated, truncated, info)   info["implementation_shortfall"] = the real metric
        │
        │  L2 aggregates ticks_per_l2_decision of the above into ONE L2-cadence step:
        ▼
src/envs/wrappers.py::FrozenL3Wrapper  (downsample_window → L2 obs; l3_passthrough OR
        │                               src/envs/l2_reward.py::l2_window_reward → L2 reward)
        ▼
L2 policy (SAC) . predict(...)   action: Box([participation_rate_mult, urgency])
```

### 5.2 Who trains/evaluates what

```
src/train/train_l3.py  ──uses──►  LOBExecutionEnv (L2 stubbed as fixed-TWAP)
                                   configs/ppo_l3.yaml (the one real config file)
                                   scripts/phase2a_sanity_suite.py (TWAPPolicy, run_episode — eval baseline)
                        ──produces──► models/l3_executioner_v1.zip + l3_vecnormalize.pkl
                                      (or models/l3_frozen_backup/*_frozen.* once a checkpoint is frozen)

src/train/train_l2.py  ──uses──►  src/envs/wrappers.py::FrozenL3Wrapper (loads the FROZEN L3 checkpoint)
                                   src/data/split.py::load_split (train/val date ranges)
                        ──produces──► models/l2_strategist_v1.zip + l2_vecnormalize.pkl

scripts/eval_l2_n500.py  ──imports──►  phase2a_sanity_suite.py + train_l2.py::make_l2_wrapped_env
        │
        ├── scripts/eval_l2_diagnostics.py
        ├── scripts/eval_l2_bucketed.py
        ├── scripts/eval_l2_test_confirmation.py    (test split — one-shot only)
        ├── scripts/analyze_l2_reward_components.py
        └── scripts/analyze_l2_reward_components_v2.py  (imports src.envs.l2_reward directly)

scripts/replace_value_probe.py  ──imports──►  phase2a_sanity_suite.py (no wrapper/L2/checkpoint at all)
        └── scripts/replace_value_probe_n500.py

scripts/replay_episode.py  ──imports──►  phase2a_sanity_suite.py::TWAPPolicy
        └── scripts/analyze_predictability.py  (imports install_tick_capture, reconstruct_child_orders)
```

### 5.3 Reverse-dependency notes (what would break if you changed X)

- **`src/envs/lob_execution_env.py`** — the one file nothing in the project can avoid depending
  on, directly or transitively. Any observation-space or action-space change here invalidates
  every existing checkpoint's `VecNormalize` stats and the LSTM/MLP weight shapes.
- **`src/envs/reward.py`'s `RewardWeights`** — read by `LOBExecutionEnv.__init__` (default),
  every `train_l3.py`/eval-script invocation that doesn't override it, and duplicated (not
  imported) inside `_compute_twap_shadow_terminal_is()`'s local re-simulation and inside
  `analyze_l2_reward_components.py`'s recompute-and-assert capture.
- **`src/envs/wrappers.py`'s `FrozenL3Wrapper`** — the sole integration point between L2 and
  L3/env; `train_l2.py`, every `eval_l2_*.py` script, `analyze_l2_reward_components*.py`, and
  `benchmark_controlled*.py`/`benchmark_parallel_l2*.py` all construct one.
- **`src/data/split.py`'s `load_split`** — 45 project-wide references; changing `SPLIT_SIZE` or
  the boundary heuristic would silently move the val/test population under every already-
  recorded eval number in `docs/reports/`.
- **`scripts/phase2a_sanity_suite.py`'s `TWAPPolicy`/`run_episode`** — imported by `train_l3.py`,
  every eval script in §4.11, `replace_value_probe*.py`, and duplicated logic (not import) inside
  `LOBExecutionEnv._compute_twap_shadow_terminal_is()`. A behavior change here would silently
  desync from that duplicated copy unless `test_subtract_twap_baseline_matches_real_twap_
  policy_exactly` is re-run.
- **`scripts/eval_l2_n500.py`** — the shared library for 5 other scripts (§4.11's import graph);
  its `EVAL_SEED_BASE`/`HORIZON_TICKS`/`_TWAP_PASSTHROUGH_ACTION` constants and `paired_report`
  formula are load-bearing for every downstream comparison staying poolable with the original
  n=500 table.

### 5.4 Confirmed dead / unused code (verified by grep, not assumed)

- `src/data/__init__.py`'s entire re-export list (`bulk_download`, `download_and_verify`,
  `CancelAddTracker`, `mid_price`, `micro_price`, `obi`, `L2Book`, `apply_diff`,
  `reconstruct_book`) — zero `from src.data import ...` references anywhere in the project.
- `src/data/features.py`'s `mid_price()`/`micro_price()` as function calls — used only by their
  own unit test; `lob_execution_env.py` computes an inline equivalent instead.
- `src/data/features.py`'s `CancelAddTracker.ratio()` — structurally always returns `0.5` (or
  `0.0` empty) regardless of real cancel/add activity, since both counters are set to
  `len(events)`.
- `src/envs/matching_engine.py`'s `expected_wait_time()` — never called outside its own test.
- `src/data/l1_features.py`'s `build_l1_feature_summary()` — real, tested, and correct, but never
  actually imported by `src/agents/orchestrator_graph.py`'s production path; every current
  invocation (tests and real runs alike) passes a stub `feature_summary_fn` instead.
- `src/analysis/calibrate_impact.py` — a complete, self-verifying, standalone module with zero
  consumers anywhere else in the project.
- `configs/data.yaml`, `configs/env.yaml`, `configs/ollama_l1.yaml`, `configs/sac_l2.yaml` —
  unpopulated placeholders, confirmed never loaded by any `.py` file.
- `src/metrics/` — an empty package, never populated, never imported.

## 6. Project History & Current Status


*Full detail, exact numbers, and complete methodology for everything summarized here:*
*[`docs/reports/PROJECT_FINAL_REPORT.md`](reports/PROJECT_FINAL_REPORT.md). This section exists*
*so a new developer has the shape of the story before diving into code — not as a replacement for*
*the full report.*

### The headline result

**None of L1, L2, or L3 — alone or combined, in any configuration tested — beats a plain TWAP
baseline on execution quality at real statistical power.** The frozen L3 executioner ties TWAP
(p=0.534/0.653 — statistically indistinguishable). L2 steering on top of that frozen L3 never
improves on it: five checkpoints tested, across two reward functions and two SAC discount factors,
every one lands at "ties baseline" or "loses to baseline" at n=500 with a pre-registered
significance bar (both a paired t-test AND a Wilcoxon signed-rank test must agree, p<0.05, same
direction — a single significant test with a negligible effect size was mistaken for a real result
early in the project and this bar exists specifically so that doesn't happen again). L1's
contribution was never measured at all (see §1.4 of this guide).

This is a genuine, well-supported negative result, not a failed search for a positive one. Getting
there required real engineering wins (below) and a disciplined diagnostic process that eliminated
six separate candidate explanations for L2's negative result one at a time — see the final report's
Section 5 for the full narrative; the short version:

1. **Wrong reward objective** — fixed via a redesigned, potential-based reward
   (`src/envs/l2_reward.py`) after measuring that 85.6% of L2's original training signal came from
   a component it doesn't control, while the metric it's evaluated on was only 6.9%. Result
   improved from significantly-worse-than-baseline to statistically-tied — real progress, still not
   a win.
2. **A diverging SAC critic** — both original-gamma (0.995) training runs ended with a badly
   diverged critic. Fixed by lowering gamma to 0.983 (matching the effective discount horizon to
   the actual ~60-decision episode length) — critic stayed stable for a full 1.6M-step run. Did not
   change the outcome versus a pre-divergence checkpoint under the old gamma.
3. **Insufficient training** — checked across checkpoints from step ~500k through ~2M, three
   reward/gamma configurations; no step count ever beat baseline.
4. **Regime mismatch** — checked via volatility-stratified evaluation; no edge that strengthens
   with volatility.
5. **Policy collapse** — checked via action-distribution diagnostics; L2 actively steers (real
   within-episode variance), doesn't collapse to a constant action.
6. **Overfitting** — a real train/val gap exists but traces to the two splits' own baselines having
   shifted, not memorization — confirmed by a volatility-stratified check that isolates
   memorization from regime shift and finds no edge on the specific days trained on.

### Real engineering wins (the durable contributions, independent of the RL result)

- **Two genuine fill-simulation bugs**, found and fixed in `src/envs/lob_execution_env.py`
  (`qty_at_price`'s missing `rtol` override at BTC's price scale, and crossing orders never being
  routed to a market-style fill). The fix's *immediate* effect on the existing checkpoint was a
  fill-ratio *drop* (revealing the bug had been inflating apparent fills, not deflating them) —
  recovery to a workable fill rate came from retraining under the corrected physics, a genuinely
  more nuanced story than "fixed a bug, number went up."
- **A ~2 orders-of-magnitude throughput improvement**, without which L2 could never have trained
  at all: `env.reset()` profiling found the bottleneck was an unvectorized per-row Python loop
  (fixed, ~25x); a second I/O round found JSON-string order-book decoding was ~97% of remaining
  decode cost; converting the data archive to a numeric zstd format
  (`src/data/l2_numeric_format.py`) eliminated that entirely (2.53x on top of the first fix). Real
  training runs this project launched sustained ~19 decisions/sec and completed 2,000,000 steps in
  under 30 hours — inside the range this work was chasing.

### Negative results, each with real evidence (not just "didn't work")

CANCEL_AND_REPLACE has no exploitable value on this data (a well-powered n=500 heuristic probe, not
a hunch). A queue-position reward-term split didn't move behavior. A variance-reduction reward
variant was *significantly worse* than the matched control (both tests agreeing). Extending
training by 2,000,000 more steps was a plateau, not further convergence (Cohen's d_z=0.076 vs. the
checkpoint it extended — "practically zero"). Full detail and exact numbers: final report Section
4.

### Methodological lessons worth knowing before trusting any number in this codebase

- An n=50 evaluation oversold a result **three separate times** in this project's history, each
  overturned once re-measured at proper (n=500) statistical power — including one case where the
  effect's *sign flipped*. Treat any n=50-only reading anywhere in `docs/TRACK_STATUS.md`'s history
  as provisional.
- A post-hoc-selected "best of 18" swept configuration regressed to the mean (and flipped sign)
  once re-tested at proper power — exactly what should be expected of a screening winner, not a
  surprise.
- An L2 train-vs-val overfitting signal was invisible in L2's own *absolute* score on each split
  and only appeared once the comparison was made *relative* to each split's own baseline (which had
  itself shifted) — comparing a raw score across conditions without accounting for how the
  reference point moved is a reusable trap.
- Uncommitted code has, at least once, silently governed a real training run (an inverted reward
  term, left in the working tree during a probe, silently inherited by every subsequent run until
  discovered and disclosed) — and a hardcoded final-save path once silently overwrote a canonical
  checkpoint before an explicit `--overwrite-canonical` guard was added in response (this guard is
  now load-bearing throughout `train_l2.py`/`train_l3.py` — see §4.5).

### The execution-predictability follow-up (most recent work)

A late, separate round asked a narrower question: TWAP is trivially predictable in the real world
(fixed schedule, uniform slices) — is frozen L3's own placement pattern measurably *less*
predictable? Answer: yes, clearly, on both a descriptive-statistics basis and a direct classifier
test (`scripts/analyze_predictability.py`, full detail in
`docs/reports/l3_execution_predictability_report.md` and §"scripts/analyze_predictability.py"
above). **This explicitly measures a property, not a market advantage** — see §1.3 of this guide
for exactly why the simulator cannot show the second thing.

### What a next attempt should do differently

The strongest untested lever, per the final report's own conclusion: **regime coverage**. The
chronological train/val/test split put almost all of the archive's volatile days in train, leaving
val and test both calm (val's own realized-volatility range sits inside roughly train's bottom
third). Every result in this project — the well-diagnosed L2 negative included — was measured in
the regime where a sophisticated execution policy has the least room to add value over a naive
schedule in the first place. A deliberately regime-stratified split, or an archive extended
backward far enough to capture genuine high-volatility/crash conditions, would directly test the
question this project surfaced but could not answer.

## 7. Getting Started

All commands assume the repo root as the working directory and `.venv/` already set up
(`.venv/bin/pip install -e .` / the dependencies in `pyproject.toml`).

### 7.1 Run the test suite

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/ -q
```
Expect **195 collected, 191 passed, 4 failed** (the 4 pre-existing failures documented in
§4.13 — none are caused by, or block, ongoing work). Full run takes ≈2:43.

### 7.2 Regenerate the train/val/test split (only needed after backfill adds older days)

```bash
PYTHONPATH=. .venv/bin/python -m src.data.split
```
Refuses to overwrite `data/splits/l2_bybit_btcusdt_split.json` if doing so would change the
already-persisted `val_dates`/`test_dates`.

### 7.3 Train L3 (RecurrentPPO executioner) from scratch

```bash
PYTHONPATH=. .venv/bin/python -m src.train.train_l3 \
    --config configs/ppo_l3.yaml \
    --run-name my_run
```
Real target is 20,000,000 timesteps (per `configs/ppo_l3.yaml`); pass `--total-timesteps 2000`
for a short mechanics smoke test. Add `--overwrite-canonical` only when this run is deliberately
meant to replace `models/l3_executioner_v1.zip` — otherwise it's saved to a run-tagged path
automatically.

### 7.4 Train L2 (SAC strategist) against a frozen L3 checkpoint

```bash
PYTHONPATH=. .venv/bin/python -m src.train.train_l2 \
    --l3-checkpoint models/l3_frozen_backup/l3_executioner_v1_frozen.zip \
    --l3-vecnormalize models/l3_frozen_backup/l3_vecnormalize_frozen.pkl \
    --total-timesteps 2000000 \
    --n-envs 4 \
    --run-name my_l2_run
```
For a fast mechanics-only check first: add `--smoke-test --total-timesteps 200 --no-eval`.
`--l3-checkpoint`/`--l3-vecnormalize` have no default on purpose — always name the frozen
checkpoint explicitly.

### 7.5 Evaluate a trained L2 checkpoint (the canonical n=500 comparison)

```bash
PYTHONPATH=. .venv/bin/python scripts/eval_l2_n500.py \
    --l2-checkpoint models/l2_strategist_v1.zip \
    --l2-vecnormalize models/l2_vecnormalize.pkl \
    --l3-checkpoint models/l3_frozen_backup/l3_executioner_v1_frozen.zip \
    --l3-vecnormalize models/l3_frozen_backup/l3_vecnormalize_frozen.pkl \
    --n 500 --use-numeric-format \
    --output-json models/l2_n500_my_run.json
```
Reports three paired arms (trained L2, TWAP-passthrough, pure TWAP) with paired t-test +
Wilcoxon + Cohen's d_z. **Never point any `eval_*`/`analyze_*` script at `load_split("test")`**
except `scripts/eval_l2_test_confirmation.py`, and only once.

### 7.6 Visualize one episode

```bash
PYTHONPATH=. .venv/bin/python scripts/replay_episode.py \
    --l3-checkpoint models/l3_frozen_backup/l3_executioner_v1_frozen.zip \
    --l3-vecnormalize models/l3_frozen_backup/l3_vecnormalize_frozen.pkl \
    --l2-checkpoint models/l2_strategist_v1.zip \
    --l2-vecnormalize models/l2_vecnormalize.pkl \
    --seed 5000042 \
    --use-numeric-format \
    --output /tmp/episode_5000042
```
Omit `--l2-checkpoint`/`--l2-vecnormalize` and pass `--frozen-l3-only` instead to see frozen L3
completely unsteered. Writes 4 PNGs: `{stem}.png` (combined overview) plus separate
`_price.png`/`_execution.png`/`_steering.png` detail figures.

### 7.7 Everything is thread-capped for a reason

Any script that constructs more than one `LOBExecutionEnv`/model process at once (training,
n=500 eval, `benchmark_*`, `calibrate_impact.py`) sets
`OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=NUMEXPR_NUM_THREADS=1` **before importing
torch**, plus `torch.set_num_threads(1)` inside each worker process. This is not optional
tuning — an un-capped attempt at this exact pattern measured **7-9x slower** than single-process
from CPU thread oversubscription (§4.10). If writing a new multi-process script, copy this
pattern from `src/train/train_l2.py`'s or `scripts/eval_l2_n500.py`'s top-of-file env-var block
verbatim, placed before every other import.

### 7.8 Before launching anything heavy

Every real launch in this project's history checked `free -h` and `nvidia-smi` immediately
before starting, and verified via a fresh, independent process listing that exactly the intended
job was running after `nohup`-backgrounding it (a known SSH/nohup stdio quirk can make a launch
command appear to hang locally while the remote job is actually running fine — never treat that
appearance alone as a failure signal, and never treat it alone as a success signal either;
check the process list).

## 8. Known Issues & Technical Debt

- **`tests/test_bulk_backfill.py`** (3 failures) and **`tests/test_l2_capture.py`** (1 failure)
  — pre-existing, documented in full in §4.13. Fixing the former means updating the test mocks'
  URL shape to match `bulk_backfill.py`'s real, flat `_checksum_url()`/`_zip_url()` output (or
  vice versa, if the flat shape itself is wrong against Binance's real archive — worth checking
  live before assuming the test is at fault). Fixing the latter means deciding whether
  `apply_diff()` should genuinely own gap-detection (a real design change) or whether the test's
  premise should be corrected to exercise `L2CaptureDaemon._apply_event()` instead, which already
  has the right logic.
- **Dependency list drift**: `pyproject.toml` doesn't declare `matplotlib` (used by
  `scripts/replay_episode.py`) or `scipy` (used throughout the eval/analysis layer), both
  present and load-bearing in the real `.venv`. `scikit-learn==1.9.0` *is* declared (added
  specifically for `scripts/analyze_predictability.py`'s classifier) — worth adding the other
  two alongside it for a fully reproducible environment.
- **Three near-duplicate `paired_report` implementations** (`eval_l2_n500.py`'s with Cohen's
  d_z, `replace_value_probe.py`'s without it, `compare_l2v3_vs_l2v2predivergence.py`'s own
  reimplementation) — functionally consistent today, but a future edit to the statistical
  methodology (e.g. switching effect-size formulas) would need to be applied in three places by
  hand, with no shared import forcing them to stay in sync.
- **`src/data/l1_features.py::build_l1_feature_summary()` is real and tested but not wired
  into any production call site** — `src/agents/orchestrator_graph.py`'s `run_episode()`/
  `run_episode_async()` both take a `feature_summary_fn` parameter that every current caller
  fills with a stub. This is the concrete, fixable half of "L1's live signal was never used in
  training" — the harder half (retraining L3 with live L1 in the loop to avoid the
  out-of-distribution confound described in §6) is the part that's expensive, not this wiring
  gap itself.
- **`src/analysis/calibrate_impact.py`'s calibrated `eta`/`lambda` were never fed back into the
  environment** — real, holdout-validated work with zero downstream consumers. If Tier-1 market
  impact is ever wired into `LOBExecutionEnv` (per the original architecture spec's Section 4.5
  sequencing), this is the calibration to start from.
- **`configs/data.yaml`/`env.yaml`/`ollama_l1.yaml`/`sac_l2.yaml`** are placeholder skeletons
  that could mislead a new developer into thinking L2/L1 are YAML-configured — they are not;
  every real parameter for both is a CLI flag with an in-code default. Either populate these
  files to match reality or remove them to avoid the false signal.
- **`src/metrics/` is an empty package** with no clear purpose beyond directory-structure
  scaffolding from the original spec. Safe to leave, safe to remove, if this project resumes.

## 9. Where to Look Next

- For **results and findings** (what worked, what didn't, exact numbers): start at
  [`docs/reports/PROJECT_FINAL_REPORT.md`](reports/PROJECT_FINAL_REPORT.md).
- For **the original design intent** behind a specific formula or index (e.g. "why is
  `book_depth_norm` z-scored per-level over time rather than cross-sectionally"): check
  `docs/architecture_spec.md` first, then the real code's own module docstring, which usually
  documents exactly where and why the implementation diverged.
- For **the full chronological history** of every round of work, in the order it happened,
  including intermediate results that were later superseded: `docs/TRACK_STATUS.md`.
- For a **worked example of this project's own investigative standard** — measure, don't guess;
  write down the pre-registered bar before seeing the result; report a null result as plainly as
  a positive one — `docs/reports/l3_execution_predictability_report.md` and its two source
  scripts (`scripts/analyze_predictability.py`, extending `scripts/replay_episode.py`) are the
  most recent and most self-contained example, worth reading end to end as a template for any
  future analysis in this codebase.
- The **strongest untested lever** for a future attempt, per the final report's own conclusion,
  is **regime coverage** — the chronological split put nearly all of the archive's volatile days
  in train, leaving val (and test) calm relative to train's own distribution
  (`scripts/analyze_split_representativeness.py` is the tool that measures this precisely). Every
  result in this project, the well-diagnosed L2 negative included, was measured in the regime
  where a sophisticated execution policy has the least room to add value over a naive schedule
  in the first place.
