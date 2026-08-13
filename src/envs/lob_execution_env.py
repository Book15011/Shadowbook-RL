"""LOBExecutionEnv-v0 (architecture_spec.md Section 3) -- Phase 2a foundation.

Scope for this phase (explicitly deferred per the task, not an oversight):
  - Reduced 9-dim observation subset only (see _build_obs docstring for the
    exact index mapping and the flagged "mid/micro price" interpretation).
  - L3 (Executioner) tier only. No L1/L2 orchestration, no LangGraph, no
    frequency gating -- every tick is L3's turn. l1_risk_score is a fixed
    0.0 placeholder (neutral) until L1 exists.
  - Single symbol/venue (BTCUSDT, Bybit L2 archive collected in Part 2).

Flagged data-source limitation (see src/envs/matching_engine.py's docstring
for the core-engine side of this): the Bybit L2 archive has book-state
snapshots only, no separate trade tape. This env's per-tick queue update
approximates v_trade as the full observed decrease in resting qty at a
price level between ticks (v_cancel=0 in this adapter) -- it cannot
distinguish "consumed by a trade" from "canceled" using this data source.
The core matching_engine.py formula itself is unaffected; only this
integration path inherits the approximation.

Other flagged interpretations (task instructions: "flag rather than
silently pick"):
  - "mid/micro price" in the reduced obs subset has no exact spec index.
    Mapped to mid_return_1s_z (full-vector idx 3) and micro_mid_dev_ticks
    (idx 9) -- there is no raw-price feature in the full 42-dim vector by
    design (raw price wouldn't generalize/normalize), making these the
    closest legitimate proxies.
  - inventory_remaining_norm = side * (qty_remaining / qty_total). The
    spec's "1 = fully unexecuted" reads naturally for a buy order; the
    side-signed fraction is the only internally consistent reading across
    both sides sharing one feature index.
  - spread_norm's "rolling p95 spread" and the z-score stats for
    mid_return_1s_z/micro_mid_dev_ticks are computed from the EPISODE'S OWN
    sampled window at reset() (a local/per-episode normalizer), not a
    persistent cross-episode online estimator -- appropriately deferred
    alongside the other explicitly-deferred features for this phase.
  - action size_frac_idx scales qty_remaining directly (fraction of what's
    left), not "L2's assigned slice" -- L2 doesn't exist yet in Phase 2a.
  - LIMIT when already resting is a no-op (same as HOLD); CANCEL_AND_REPLACE
    is the only action that tears down and replaces an existing resting
    order, giving it a distinct, unambiguous meaning.
  - MARKET orders walk the visible top-20 book level-by-level (best-to-
    worst, matching TickView's stored order -- see matching_engine.py's
    walk_market_fill()), producing one fill per level touched at that
    level's own price. If the requested quantity exceeds all visible
    retained levels, only the visible depth fills; the unconsumed
    remainder stays in qty_remaining and is picked up on a later tick, or
    falls through to the terminal opportunity-cost IS component (Section
    5.1) if the episode ends first -- no synthetic price levels are
    invented beyond what is actually observed.
  - A resting LIMIT order whose computed price falls outside the visible
    top-20 book is seeded with Q_ahead=0 (we cannot observe deeper levels;
    flagged as likely optimistic for very passive/deep prices).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd

from src.data.features import obi
from src.envs.matching_engine import QueueState, queue_position_ratio, update_queue, walk_market_fill
from src.envs.reward import RewardWeights, compute_implementation_shortfall, step_reward

TICK_SIZE = 0.1  # BTCUSDT perpetual tick size, matches observed real data

ORDER_TYPE_HOLD = 0
ORDER_TYPE_LIMIT = 1
ORDER_TYPE_MARKET = 2
ORDER_TYPE_CANCEL_REPLACE = 3
SIZE_FRACTIONS = (0.2, 0.4, 0.6, 0.8, 1.0)

# Reduced 9-dim observation: (full-vector idx, name, clip range)
_OBS_SPEC = (
    (0, "time_remaining_norm", (0.0, 1.0)),
    (1, "inventory_remaining_norm", (-1.0, 1.0)),
    (2, "spread_norm", (0.0, 1.0)),
    (3, "mid_return_1s_z", (-5.0, 5.0)),
    (9, "micro_mid_dev_ticks", (-5.0, 5.0)),
    (6, "OBI_1", (-1.0, 1.0)),
    (7, "OBI_5", (-1.0, 1.0)),
    (8, "OBI_10", (-1.0, 1.0)),
    (13, "queue_position_ratio", (-1.0, 1.0)),
)
OBS_FEATURE_NAMES = tuple(name for _, name, _ in _OBS_SPEC)


@dataclass
class TickView:
    ts: int
    best_bid: float
    best_ask: float
    mid_price: float
    spread: float
    bid_prices: np.ndarray
    bid_sizes: np.ndarray
    ask_prices: np.ndarray
    ask_sizes: np.ndarray

    def qty_at_price(self, price: float, side: str) -> float:
        prices = self.bid_prices if side == "bid" else self.ask_prices
        sizes = self.bid_sizes if side == "bid" else self.ask_sizes
        matches = np.isclose(prices, price, atol=TICK_SIZE / 2)
        if not matches.any():
            return 0.0
        return float(sizes[matches][0])


def _parse_levels(raw_json: str) -> tuple[np.ndarray, np.ndarray]:
    levels = json.loads(raw_json)
    if not levels:
        return np.array([]), np.array([])
    arr = np.asarray(levels, dtype=float)
    return arr[:, 0], arr[:, 1]


class LOBExecutionEnv(gym.Env):
    """Single-agent (L3-only) limit order book execution environment,
    replaying real historical Bybit L2 archive data. See module docstring
    for the Phase 2a scope and every flagged design assumption."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        data_dir: str | Path = "data/raw_l2_bybit/BTCUSDT",
        horizon_ticks: int = 3000,
        lookback_ticks: int = 10,
        tick_interval_s: float = 0.1,
        min_size_mult: float = 0.5,
        max_size_mult: float = 8.0,
        reward_weights: RewardWeights | None = None,
        fee_bps_per_fill: float = 1.0,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.horizon_ticks = horizon_ticks
        self.lookback_ticks = lookback_ticks
        self.tick_interval_s = tick_interval_s
        self.min_size_mult = min_size_mult
        self.max_size_mult = max_size_mult
        self.reward_weights = reward_weights or RewardWeights()
        self.fee_bps_per_fill = fee_bps_per_fill

        self.observation_space = gym.spaces.Box(
            low=np.array([lo for _, _, (lo, _) in _OBS_SPEC], dtype=np.float32),
            high=np.array([hi for _, _, (_, hi) in _OBS_SPEC], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.MultiDiscrete([4, 11, 5])

        self._files = sorted(self.data_dir.glob("*.parquet"))
        if not self._files:
            raise FileNotFoundError(f"No parquet files found in {self.data_dir}")
        self._day_cache: dict[Path, pd.DataFrame] = {}

        # episode state, set in reset()
        self._ticks: list[TickView] = []
        self._episode_start: int = 0
        self._tick_idx: int = 0
        self.side: int = 1
        self.qty_total: float = 0.0
        self.qty_remaining: float = 0.0
        self.arrival_price: float = 0.0
        self._resting: QueueState | None = None
        self._resting_price: float | None = None
        self._resting_side: str | None = None  # "bid" or "ask" -- which side of the book we rest on
        self._spread_p95: float = 1.0
        self._ret1s_mean: float = 0.0
        self._ret1s_std: float = 1.0
        self._episode_fills: list[dict] = []  # full history for the episode, for eval reporting

    _MAX_CACHED_DAYS = 3  # ~85MB/day; a small cache trims repeat reads across 50+ episodes without holding all 115+ files resident

    def _load_day(self, path: Path) -> pd.DataFrame:
        if path not in self._day_cache:
            if len(self._day_cache) >= self._MAX_CACHED_DAYS:
                self._day_cache.pop(next(iter(self._day_cache)))
            self._day_cache[path] = pd.read_parquet(path)
        return self._day_cache[path]

    def _build_ticks(self, day_df: pd.DataFrame, start: int, end: int) -> list[TickView]:
        sl = day_df.iloc[start:end]
        ticks = []
        for row in sl.itertuples(index=False):
            bid_p, bid_s = _parse_levels(row.bids)
            ask_p, ask_s = _parse_levels(row.asks)
            ticks.append(
                TickView(
                    ts=int(row.ts), best_bid=float(row.best_bid), best_ask=float(row.best_ask),
                    mid_price=float(row.mid_price), spread=float(row.spread),
                    bid_prices=bid_p, bid_sizes=bid_s, ask_prices=ask_p, ask_sizes=ask_s,
                )
            )
        return ticks

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        file_idx = int(self.np_random.integers(0, len(self._files)))
        day_path = self._files[file_idx]
        day_df = self._load_day(day_path)
        n_rows = len(day_df)

        needed = self.lookback_ticks + self.horizon_ticks
        if n_rows <= needed:
            # degenerate case: day shorter than one full episode window (shouldn't
            # happen with ~864k rows/day vs a few-thousand-tick horizon, but a real
            # day could have gaps -- fall back to using the whole day, horizon
            # clipped to what's available).
            start = self.lookback_ticks
            end = n_rows
        else:
            start = int(self.np_random.integers(self.lookback_ticks, n_rows - self.horizon_ticks))
            end = start + self.horizon_ticks

        self._ticks = self._build_ticks(day_df, start - self.lookback_ticks, end)
        self._episode_start = self.lookback_ticks  # index into self._ticks where the real episode begins
        self._tick_idx = self._episode_start

        # Local (per-episode-window) normalization stats -- see module docstring.
        mids = np.array([t.mid_price for t in self._ticks], dtype=float)
        spreads = np.array([t.spread for t in self._ticks], dtype=float)
        self._spread_p95 = float(np.percentile(spreads, 95)) if len(spreads) else 1.0
        if self._spread_p95 <= 0:
            self._spread_p95 = TICK_SIZE
        rets_1s = (mids[self.lookback_ticks:] - mids[: -self.lookback_ticks]) / mids[: -self.lookback_ticks]
        self._ret1s_mean = float(np.mean(rets_1s)) if len(rets_1s) else 0.0
        self._ret1s_std = float(np.std(rets_1s)) if len(rets_1s) > 1 else 1.0
        if self._ret1s_std <= 0:
            self._ret1s_std = 1e-9

        # Order-size bound relative to typical top-of-book depth in this window
        # (task instruction: bound so episodes aren't degenerate by construction).
        top_depths = np.array(
            [t.bid_sizes[0] + t.ask_sizes[0] for t in self._ticks if len(t.bid_sizes) and len(t.ask_sizes)],
            dtype=float,
        )
        ref_depth = float(np.median(top_depths)) if len(top_depths) else 1.0
        if ref_depth <= 0:
            ref_depth = 1.0
        log_mult = self.np_random.uniform(math.log(self.min_size_mult), math.log(self.max_size_mult))
        self.qty_total = ref_depth * math.exp(log_mult)
        self.qty_remaining = self.qty_total
        self._scenario_depth_ratio = self.qty_total / ref_depth  # reported as a scenario-difficulty metric, per spec 3.4

        self.side = int(self.np_random.choice([-1, 1]))
        self.arrival_price = self._ticks[self._episode_start].mid_price

        self._resting = None
        self._resting_price = None
        self._resting_side = None
        self._episode_fills = []
        self._terminated_early = False

        obs = self._build_obs()
        info = self._build_info(step_fills=[], canceled_unfilled=False)
        return obs, info

    def _current_tick(self) -> TickView:
        return self._ticks[self._tick_idx]

    def _build_obs(self) -> np.ndarray:
        tick = self._current_tick()
        ticks_elapsed = self._tick_idx - self._episode_start
        time_remaining_norm = max(0.0, min(1.0, 1.0 - ticks_elapsed / self.horizon_ticks))

        inventory_remaining_norm = self.side * (self.qty_remaining / self.qty_total) if self.qty_total > 0 else 0.0
        inventory_remaining_norm = float(np.clip(inventory_remaining_norm, -1.0, 1.0))

        spread_norm = float(np.clip(tick.spread / self._spread_p95, 0.0, 1.0))

        prior = self._ticks[self._tick_idx - self.lookback_ticks]
        ret_1s = (tick.mid_price - prior.mid_price) / prior.mid_price if prior.mid_price > 0 else 0.0
        mid_return_1s_z = float(np.clip((ret_1s - self._ret1s_mean) / self._ret1s_std, -5.0, 5.0))

        if len(tick.bid_sizes) and len(tick.ask_sizes) and (tick.bid_sizes[0] + tick.ask_sizes[0]) > 0:
            micro = (
                tick.bid_prices[0] * tick.ask_sizes[0] + tick.ask_prices[0] * tick.bid_sizes[0]
            ) / (tick.bid_sizes[0] + tick.ask_sizes[0])
        else:
            micro = tick.mid_price
        micro_mid_dev_ticks = float(np.clip((micro - tick.mid_price) / TICK_SIZE, -5.0, 5.0))

        obi_1 = obi(tick.bid_prices, tick.ask_prices, tick.bid_sizes, tick.ask_sizes, k=1) if len(tick.bid_prices) else 0.0
        obi_5 = obi(tick.bid_prices, tick.ask_prices, tick.bid_sizes, tick.ask_sizes, k=5) if len(tick.bid_prices) >= 5 else obi_1
        obi_10 = obi(tick.bid_prices, tick.ask_prices, tick.bid_sizes, tick.ask_sizes, k=10) if len(tick.bid_prices) >= 10 else obi_5

        qpr = queue_position_ratio(self._resting) if self._resting is not None else -1.0

        return np.array(
            [time_remaining_norm, inventory_remaining_norm, spread_norm, mid_return_1s_z,
             micro_mid_dev_ticks, obi_1, obi_5, obi_10, qpr],
            dtype=np.float32,
        )

    def _build_info(self, *, step_fills: list[dict], canceled_unfilled: bool) -> dict[str, Any]:
        tick = self._current_tick()
        return {
            "side": self.side,
            "qty_total": self.qty_total,
            "qty_remaining": self.qty_remaining,
            "arrival_price": self.arrival_price,
            "mid_price": tick.mid_price,
            "scenario_depth_ratio": getattr(self, "_scenario_depth_ratio", None),
            "fills_this_step": list(step_fills),
            "canceled_unfilled": canceled_unfilled,
            "resting_q_ahead": self._resting.q_ahead if self._resting is not None else None,
            "resting_own_remaining": self._resting.own_qty_remaining if self._resting is not None else None,
            "tick_idx": self._tick_idx,
            "ticks_elapsed": self._tick_idx - self._episode_start,
        }

    def _estimate_trade_volume(self, prev_idx: int, curr_idx: int, price: float, side: str) -> tuple[float, float]:
        """Approximates v_trade as the full observed decrease in resting qty
        at `price` between two ticks (v_cancel=0 in this adapter) -- see
        module docstring for why this data source can't separate the two."""
        prev_qty = self._ticks[prev_idx].qty_at_price(price, side)
        curr_qty = self._ticks[curr_idx].qty_at_price(price, side)
        v_trade = max(0.0, prev_qty - curr_qty)
        return v_trade, prev_qty

    def _place_limit(self, tick: TickView, offset: int, size_frac: float) -> None:
        if self.side == 1:
            price = round(tick.best_bid + offset * TICK_SIZE, 1)
            side = "bid"
        else:
            price = round(tick.best_ask - offset * TICK_SIZE, 1)
            side = "ask"
        # Q_ahead(0): visible resting volume at this price at placement (Section 2.4
        # point 1); 0.0 if the price falls outside the visible top-20 book -- flagged
        # in the module docstring as likely optimistic for deep/passive prices.
        q_ahead = tick.qty_at_price(price, side)
        size = min(size_frac * self.qty_remaining, self.qty_remaining)
        if size <= 0:
            return
        self._resting = QueueState(q_ahead=q_ahead, own_qty_remaining=size)
        self._resting_price = price
        self._resting_side = side

    def step(self, action):
        tick_before = self._current_tick()
        step_fills: list[dict] = []
        canceled_unfilled = False
        queue_ahead_at_cancel: float | None = None
        queue_at_level: float | None = None

        # 1. Evolve any existing resting order against market activity since last tick.
        if self._resting is not None and not self._resting.is_resolved:
            v_trade, q_p_before = self._estimate_trade_volume(
                self._tick_idx - 1, self._tick_idx, self._resting_price, self._resting_side
            )
            prev_filled = self._resting.filled_qty
            self._resting = update_queue(self._resting, v_trade=v_trade, v_cancel=0.0, q_p_before=q_p_before)
            newly_filled = self._resting.filled_qty - prev_filled
            if newly_filled > 0:
                step_fills.append({"price": self._resting_price, "qty": newly_filled, "is_maker": True})
                self.qty_remaining = max(0.0, self.qty_remaining - newly_filled)
            if self._resting.is_resolved:
                self._resting = None
                self._resting_price = None
                self._resting_side = None

        # 2. Decode and apply the action.
        order_type, price_offset_idx, size_frac_idx = int(action[0]), int(action[1]), int(action[2])
        offset = price_offset_idx - 5
        size_frac = SIZE_FRACTIONS[size_frac_idx]

        if order_type in (ORDER_TYPE_CANCEL_REPLACE, ORDER_TYPE_MARKET) and self._resting is not None:
            canceled_unfilled = True
            queue_ahead_at_cancel = self._resting.q_ahead
            queue_at_level = self._resting.q_ahead + self._resting.own_qty_remaining
            self._resting = None
            self._resting_price = None
            self._resting_side = None

        if order_type == ORDER_TYPE_MARKET:
            mkt_qty = min(size_frac * self.qty_remaining, self.qty_remaining)
            if mkt_qty > 0:
                book_prices, book_sizes = (
                    (tick_before.ask_prices, tick_before.ask_sizes) if self.side == 1
                    else (tick_before.bid_prices, tick_before.bid_sizes)
                )
                level_fills, qty_unfilled = walk_market_fill(mkt_qty, book_prices, book_sizes)
                for level_price, level_qty in level_fills:
                    step_fills.append({"price": level_price, "qty": level_qty, "is_maker": False})
                filled_qty = mkt_qty - qty_unfilled
                self.qty_remaining = max(0.0, self.qty_remaining - filled_qty)
        elif order_type in (ORDER_TYPE_LIMIT, ORDER_TYPE_CANCEL_REPLACE):
            if self._resting is None and self.qty_remaining > 0:
                self._place_limit(tick_before, offset, size_frac)
        # HOLD (order_type == 0): nothing further.

        r = step_reward(
            self.reward_weights, side=self.side, fills=step_fills,
            arrival_price=self.arrival_price, mid_price=tick_before.mid_price,
            qty_remaining=self.qty_remaining, qty_total=self.qty_total,
            dt=self.tick_interval_s, l1_risk_score=0.0,
            canceled_unfilled=canceled_unfilled,
            queue_ahead_at_cancel=queue_ahead_at_cancel, queue_at_level=queue_at_level,
        )
        self._episode_fills.extend(step_fills)

        terminated = self.qty_remaining <= 1e-12
        self._tick_idx += 1
        truncated = (self._tick_idx - self._episode_start) >= self.horizon_ticks or self._tick_idx >= len(self._ticks)
        if self._tick_idx >= len(self._ticks):
            self._tick_idx = len(self._ticks) - 1

        terminal_is = None
        if terminated or truncated:
            terminal_tick = self._ticks[min(self._tick_idx, len(self._ticks) - 1)]
            terminal_is = compute_implementation_shortfall(
                side=self.side, fills=self._episode_fills, qty_total=self.qty_total,
                arrival_price=self.arrival_price, terminal_mid_price=terminal_tick.mid_price,
                fee_bps_per_fill=self.fee_bps_per_fill,
            )
            r += -self.reward_weights.kappa * terminal_is.is_total_bps

        obs = self._build_obs()
        info = self._build_info(step_fills=step_fills, canceled_unfilled=canceled_unfilled)
        if terminal_is is not None:
            info["implementation_shortfall"] = terminal_is

        return obs, r, terminated, truncated, info
