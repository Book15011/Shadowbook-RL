"""LOBExecutionEnv-v0 (architecture_spec.md Section 3) -- Phase 2a/2b.

Phase 2a scope: L3-only, single symbol/venue (BTCUSDT, Bybit L2 archive).
Bybit L2 archive has book-state snapshots only, no separate trade tape --
this env approximates v_trade as the full observed decrease in resting qty
at a price level between ticks (v_cancel=0 in this adapter); it cannot
distinguish trade from cancel using this data source. See
matching_engine.py module docstring for the core-engine side of this.

Phase 2a flagged interpretations:
  - inventory_remaining_norm = side * (qty_remaining / qty_total).
  - spread_norm rolling p95 spread is a STATIC per-episode value computed
    once at reset() over the whole lookback+horizon window -- not a
    genuinely trailing/rolling window within the episode. Contrast idx 5
    below, which is a real fixed-size (60s) trailing window.
  - action size_frac_idx scales qty_remaining directly, not an L2 slice.
  - LIMIT while resting is a no-op; CANCEL_AND_REPLACE tears down and
    replaces the resting order.
  - MARKET orders walk the visible top-20 book level-by-level (see
    matching_engine.walk_market_fill()); unfilled remainder stays in
    qty_remaining, picked up later or folded into terminal IS.
  - A resting LIMIT order priced outside the visible top-20 book seeds
    Q_ahead=0 (optimistic for deep/passive prices, flagged).

Phase 2b scope: full 42-dim observation vector (Section 3.1), superseding
Phase 2a reduced 9-dim subset. Every index is in _OBS_SPEC below. New
flagged interpretations (full rationale in the Phase 2b completion report,
kept brief here to keep this docstring maintainable):
  - idx 12 / 40 (trade flow, taker ratio): derived from L2 diffs directly
    (touch-level depletion inferred as taker buy/sell pressure) -- NO new
    trade-tape data needed. CAVEAT (added on closer consistency check with
    idx 10-11 below): _estimate_trade_volume(), the source of this signal,
    is an ASSUMPTION (all touch-level depletion = trade), not a real
    trade-print field -- the Bybit archive has no execution/trade-id field
    alongside the book deltas, confirmed by direct inspection of the
    collector script. So a real cancel on either side is silently
    mislabeled as taker flow here, same underlying ambiguity CAR faces.
    The reason this is buildable while CAR is not: idx 12/40 only need
    WHICH SIDE depleted (always known), not the cancel-vs-trade split CAR
    needs as its numerator (which is definitionally 0 under the
    v_cancel=0 assumption below) -- so idx 12/40 produce a real, noisy
    signal, CAR produces a trivial, exactly-zero one. Report this plainly
    as noisier than the spec implies, not clean. See
    _precompute_feature_series().
  - idx 10-11 (cancel_add_ratio): genuinely blocked -- CAR needs real
    cancel volume, and this snapshot archive cannot separate a cancel from
    a trade at the same price. Consistent with the existing v_cancel=0
    adapter assumption, hardcoded 0.0 for both sides.
  - idx 15-18 (L2/L1 stub hooks): plain overridable attributes, not real
    agent calls (architecture_spec.md Section 4.4 step 4/5). l1_risk_score
    is now the single source of truth also passed to step_reward().
  - idx 19-38 (book_depth_norm): each of the 20 levels z-scored against
    ITS OWN trailing rolling mean/std over TIME (Section 3.1's blanket
    "trailing rolling window" rule, matching every other z-scored
    feature in the vector -- corrected from an earlier cross-sectional
    (across-levels) reading, which was a judgment call against ambiguous
    table phrasing rather than a confirmed-correct one). Window reuses
    ticks_60s, the same window as realized_vol_60s_z/
    taker_buy_sell_ratio_1m -- the spec gives no explicit window length
    for this feature, so this is a documented, consistent default rather
    than an arbitrary new one.
  - idx 39 (funding_rate_z): joined by timestamp against data/raw_l1/
    funding_rate/ (both L2 ts and funding calc_time are epoch-ms,
    confirmed by direct inspection). Z-scored vs trailing ~30d history.
  - idx 9 TICK_SIZE: exchangeInfo runtime check is spec-requested but
    fapi.binance.com is network-blocked (Section 2.1.1) -- 0.10 hardcoded,
    flagged explicitly.
  - idx 41 (own_open_orders_norm): own_qty_remaining / qty_total.
  - idx 4, 14: not explicitly itemized in the Phase 2b task step list, but
    built anyway since the full 42-dim vector requires every index.
  - Lookback buffer increased to up to 600 ticks (60s) for idx 5/12/40,
    WITHOUT changing the RNG draw order (buffer size decided after start
    is already drawn) -- preserves exact Phase 2a seed reproducibility.
"""
from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd

from src.data.features import obi, zscore
from src.envs.matching_engine import QueueState, queue_position_ratio, update_queue, walk_market_fill
from src.data.l2_numeric_format import read_day as _read_numeric_day
from src.envs.reward import RewardWeights, compute_implementation_shortfall, step_reward

_log = logging.getLogger(__name__)

TICK_SIZE = 0.1  # BTCUSDT perpetual tick size. Spec asks to confirm against exchangeInfo
# at runtime rather than hardcode; exchangeInfo lives at fapi.binance.com, which is
# network-blocked here (architecture_spec.md Section 2.1.1). Runtime verification is
# genuinely not possible in this environment -- flagged deviation, not a silent shortcut.

ORDER_TYPE_HOLD = 0
ORDER_TYPE_LIMIT = 1
ORDER_TYPE_MARKET = 2
ORDER_TYPE_CANCEL_REPLACE = 3
SIZE_FRACTIONS = (0.2, 0.4, 0.6, 0.8, 1.0)

# Full 42-dim observation vector (architecture_spec.md Section 3.1): (index, name, clip
# range). observation_space bounds are built directly from this; _build_obs() returns
# values in this exact order. Full rationale for each new index is in the module
# docstring above and the Phase 2b completion report.
_OBS_SPEC = (
    (0, "time_remaining_norm", (0.0, 1.0)),
    (1, "inventory_remaining_norm", (-1.0, 1.0)),
    (2, "spread_norm", (0.0, 1.0)),
    (3, "mid_return_1s_z", (-5.0, 5.0)),
    (4, "mid_return_5s_z", (-5.0, 5.0)),
    (5, "realized_vol_60s_z", (-5.0, 5.0)),
    (6, "OBI_1", (-1.0, 1.0)),
    (7, "OBI_5", (-1.0, 1.0)),
    (8, "OBI_10", (-1.0, 1.0)),
    (9, "micro_mid_dev_ticks", (-5.0, 5.0)),
    (10, "cancel_add_ratio_bid", (0.0, 5.0)),
    (11, "cancel_add_ratio_ask", (0.0, 5.0)),
    (12, "trade_flow_imbalance_5s", (-1.0, 1.0)),
    (13, "queue_position_ratio", (-1.0, 1.0)),
    (14, "ticks_since_own_fill_norm", (0.0, 1.0)),
    (15, "l2_target_slice_ratio", (0.0, 1.0)),
    (16, "l2_urgency", (0.0, 1.0)),
    (17, "l1_risk_score", (-1.0, 1.0)),
    (18, "l1_confidence", (0.0, 1.0)),
    *[(19 + i, f"book_depth_norm_{i}", (-5.0, 5.0)) for i in range(20)],
    (39, "funding_rate_z", (-5.0, 5.0)),
    (40, "taker_buy_sell_ratio_1m", (-1.0, 1.0)),
    (41, "own_open_orders_norm", (0.0, 1.0)),
)
OBS_FEATURE_NAMES = tuple(name for _, name, _ in _OBS_SPEC)
assert len(_OBS_SPEC) == 42


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
        # rtol=0.0: np.isclose's default rtol=1e-05 was never overridden here, and at
        # BTCUSDT's price scale (~$100k+) rtol*price alone is ~$1-1.5 -- 20-30x wider
        # than the intended/documented atol=TICK_SIZE/2=$0.05 half-tick match window.
        # Verified directly (see docs/reports/phase3_l3_baseline_milestone.md): this
        # made the match rate 100% at every offset tested, including offsets that cross
        # the opposing side entirely, and ~89% of all matches were ambiguous (multiple
        # array entries satisfied the loose tolerance, with the first in array order --
        # not the nearest -- silently selected). Pinning rtol=0.0 makes the match
        # governed purely by atol, as documented.
        matches = np.isclose(prices, price, atol=TICK_SIZE / 2, rtol=0.0)
        if not matches.any():
            return 0.0
        return float(sizes[matches][0])


_DATE_RE = re.compile(r'(\d{4}-\d{2}-\d{2})')


def _extract_date(path: Path) -> str:
    """Pulls the YYYY-MM-DD date out of an l2-{symbol}-{date}.parquet filename
    (see scripts/collect_l2_bybit.py day_str = day.isoformat()). Raises if a
    file in data_dir does not follow that convention -- a silently-skipped
    file would be a worse failure mode than a loud one when date_range
    filtering is what is supposed to make window selection reproducible."""
    m = _DATE_RE.search(path.stem)
    if m is None:
        raise ValueError(f"Could not parse a YYYY-MM-DD date out of filename: {path.name}")
    return m.group(1)


def _parse_levels(raw_json: str) -> tuple[np.ndarray, np.ndarray]:
    levels = json.loads(raw_json)
    if not levels:
        return np.array([]), np.array([])
    arr = np.asarray(levels, dtype=float)
    return arr[:, 0], arr[:, 1]


def _rolling_return(mids: np.ndarray, window_ticks: int) -> np.ndarray:
    """out[i] = (mids[i]-mids[i-window_ticks])/mids[i-window_ticks], 0.0 where
    there is not enough history yet (i < window_ticks) or the denominator is
    non-positive."""
    n = len(mids)
    out = np.zeros(n, dtype=float)
    if n > window_ticks and window_ticks > 0:
        prior = mids[:-window_ticks]
        curr = mids[window_ticks:]
        valid = prior > 0
        out[window_ticks:][valid] = (curr[valid] - prior[valid]) / prior[valid]
    return out


def _rolling_rms(values: np.ndarray, window_ticks: int) -> np.ndarray:
    """out[i] = sqrt(mean(values[max(0,i-window_ticks+1):i+1]**2)) -- a
    trailing-window RMS, window shrinking gracefully near the start of the
    array rather than requiring the full window to be available.

    Vectorized (2026-08-24, was a python range(n) loop) -- w=min(window_ticks,i+1)
    is never 0 for i>=0/window_ticks>=1 (both guaranteed by every caller), so the
    scalar version's `if w > 0 else 0.0` branch was dead code, not a case this
    vectorization needs to reproduce. Same csum/subtract/divide/sqrt arithmetic
    per element, just computed via fancy indexing instead of a python loop --
    verified byte-identical against the original, not assumed (see
    tests/test_reset_vectorization_equivalence.py)."""
    n = len(values)
    sq = values.astype(float) ** 2
    csum = np.concatenate([[0.0], np.cumsum(sq)])
    idx = np.arange(1, n + 1)
    w = np.minimum(window_ticks, idx)
    left = idx - w
    return np.sqrt((csum[idx] - csum[left]) / w)


def _rolling_sum(values: np.ndarray, window_ticks: int) -> np.ndarray:
    """out[i] = sum(values[max(0,i-window_ticks+1):i+1]) -- trailing-window
    sum, same shrinking-near-the-start behavior as _rolling_rms.

    Vectorized (2026-08-24, was a python range(n) loop) -- same csum/subtract
    arithmetic per element via fancy indexing instead of a python loop,
    verified byte-identical (see tests/test_reset_vectorization_equivalence.py)."""
    n = len(values)
    csum = np.concatenate([[0.0], np.cumsum(values.astype(float))])
    idx = np.arange(1, n + 1)
    left = np.maximum(0, idx - window_ticks)
    return csum[idx] - csum[left]


def _rolling_mean_std(values: np.ndarray, window_ticks: int) -> tuple[np.ndarray, np.ndarray]:
    """out[i] = (trailing mean, trailing std) of values[max(0,i-window_ticks+1):i+1] --
    unlike _rolling_rms (which assumes a near-zero-mean series like returns), this is for
    a raw, non-centered series (e.g. a book level's resting size) where the mean itself
    is a meaningful, non-zero quantity to track.

    Vectorized (2026-08-24, was a python range(n) loop) -- same csum/subtract/divide
    arithmetic per element via fancy indexing instead of a python loop, verified
    byte-identical (see tests/test_reset_vectorization_equivalence.py)."""
    n = len(values)
    v = values.astype(float)
    csum = np.concatenate([[0.0], np.cumsum(v)])
    csum_sq = np.concatenate([[0.0], np.cumsum(v * v)])
    idx = np.arange(1, n + 1)
    w = np.minimum(window_ticks, idx)
    left = idx - w
    s = csum[idx] - csum[left]
    sq = csum_sq[idx] - csum_sq[left]
    mean_out = s / w
    variance = np.maximum(0.0, sq / w - mean_out * mean_out)
    return mean_out, np.sqrt(variance)


def _vec_qty_at_price(price_matrix: np.ndarray, size_matrix: np.ndarray, query_prices: np.ndarray) -> np.ndarray:
    """Vectorized, row-wise form of TickView.qty_at_price: for each row i, the size
    at the first column (in that row's own original array order -- price_matrix's
    column order must match the tick's own bid_prices/ask_prices order exactly,
    zero-padded past each tick's real level count) within atol=TICK_SIZE/2
    (rtol=0.0) of query_prices[i], or 0.0 if no column matches. Identical semantics
    to the scalar `sizes[matches][0]` / `0.0` in TickView.qty_at_price -- argmax on
    a boolean array returns the first True, matching `[0]` on the boolean-masked
    array exactly. Zero-padding is safe here specifically because BTCUSDT trades at
    ~$100k+, so a padded 0.0 price can never fall within the $0.05 atol of a real
    query price and can never be mistaken for a real match -- verified byte-
    identical against the scalar version, not assumed (see
    tests/test_reset_vectorization_equivalence.py)."""
    matches = np.isclose(price_matrix, query_prices[:, None], atol=TICK_SIZE / 2, rtol=0.0)
    found = matches.any(axis=1)
    first_idx = matches.argmax(axis=1)
    rows = np.arange(len(query_prices))
    return np.where(found, size_matrix[rows, first_idx], 0.0)


class LOBExecutionEnv(gym.Env):
    """Single-agent (L3-only) limit order book execution environment,
    replaying real historical Bybit L2 archive data. See module docstring
    for the Phase 2a/2b scope and every flagged design assumption."""

    metadata = {"render_modes": []}

    # Trailing history needed by the widest-window feature (realized_vol_60s_z /
    # taker_buy_sell_ratio_1m, both 60s). Buffer fetched at reset() is up to this many
    # ticks before the episode start -- see module docstring Lookback buffer note for
    # why this does not change the RNG draw sequence / Phase 2a reproducibility.
    _MAX_LOOKBACK_S = 60.0
    _FUNDING_LOOKBACK_PERIODS = 90  # roughly 30 days at the standard 8h funding cadence
    # Part A (docs/reports/phase3_l3_baseline_milestone.md): normalization window for
    # _ticks_since_placement_norm(), the new placement-anchored staleness signal.
    # Deliberately NOT horizon_ticks (3000) -- that timescale is already covered by the
    # existing ticks_since_own_fill_norm (obs idx 14). This signal needs to flag a
    # SPECIFIC resting placement going stale much sooner, so a CANCEL_AND_REPLACE has a
    # chance to become reward-positive before the episode just runs out. 300 ticks (30s)
    # is chosen from this project's own measured fast-fill timescale, not an arbitrary
    # guess: successful fills clustered at a 125-195 tick median across every checkpoint
    # tested in the coefficient sweep (docs/reports/phase3_l3_baseline_milestone.md) --
    # 300 ticks gives roughly 1.5-2.4x that typical fast-fill time as headroom before
    # flagging a placement as unusually stale, relative to how fast fills happen when
    # the policy is working. Not derived from Section 2.4's expected_wait_time directly
    # (q_ahead / avg_trade_rate) -- that needs a live per-price trade-rate estimate this
    # environment does not currently compute (see the earlier rollout investigation,
    # which flagged this same gap).
    _PLACEMENT_STALENESS_WINDOW_TICKS = 300.0

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
        date_range: tuple[str, str] | None = None,
        funding_rate_dir: str | Path = "data/raw_l1/funding_rate",
        l1_risk_score: float = 0.0,
        l1_confidence: float = 0.0,
        l2_urgency: float = 0.5,
        l2_target_slice_ratio_override: float | None = None,
        use_numeric_format: bool = False,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.date_range = date_range
        self.horizon_ticks = horizon_ticks
        self.lookback_ticks = lookback_ticks
        self.tick_interval_s = tick_interval_s
        self.min_size_mult = min_size_mult
        self.max_size_mult = max_size_mult
        self.reward_weights = reward_weights or RewardWeights()
        self.fee_bps_per_fill = fee_bps_per_fill

        # L1/L2 stub/override hooks (architecture_spec.md Section 4.4 step 4/5: L1 is a
        # no-op stub until real Ollama calls exist; L2 does not exist until Phase 3/4).
        # Plain settable attributes, not calls to real agent code -- see module docstring.
        self.l1_risk_score = l1_risk_score
        self.l1_confidence = l1_confidence
        self.l2_urgency = l2_urgency
        self.l2_target_slice_ratio_override = l2_target_slice_ratio_override

        self._max_lookback_ticks = max(self.lookback_ticks, round(self._MAX_LOOKBACK_S / self.tick_interval_s))

        self.observation_space = gym.spaces.Box(
            low=np.array([lo for _, _, (lo, _) in _OBS_SPEC], dtype=np.float32),
            high=np.array([hi for _, _, (_, hi) in _OBS_SPEC], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.MultiDiscrete([4, 11, 5])

        # Reproducibility (architecture_spec.md Section 4.4 needs identical held-out
        # windows across on/off ablation runs): globbing data_dir fresh in every
        # instantiation means the file list -- and therefore what a fixed seed integer
        # index resolves to -- silently drifts as the L2 backfill job (which walks
        # backward, prepending older days) adds files between runs. Passing an explicit
        # date_range pins the exact file set independent of whatever else has landed on
        # disk since; without it, behavior is unchanged (whatever is currently present),
        # which remains fine for exploratory/dev use.
        # use_numeric_format (2026-08-24, storage-format round): opt-in, same pattern as
        # zeta/eta_replace/subtract_twap_baseline -- defaults to the original JSON/parquet
        # path, no behavior change for any existing caller. When True, data_dir is expected
        # to hold the converted *.npzst files (src/data/l2_numeric_format.py), NOT the
        # original *.parquet files -- the two formats are read from separate directory
        # trees by design (the original archive is read-only, never converted in place).
        self.use_numeric_format = use_numeric_format
        glob_pattern = "*.npzst" if use_numeric_format else "*.parquet"
        all_files = sorted(self.data_dir.glob(glob_pattern))
        if date_range is not None:
            start_date, end_date = date_range
            self._files = [p for p in all_files if start_date <= _extract_date(p) <= end_date]
        else:
            self._files = all_files
            _log.warning(
                "date_range not set; cross-run window reproducibility is not guaranteed "
                "(see architecture_spec.md Section 4.4)."
            )
        if not self._files:
            raise FileNotFoundError(
                f"No {glob_pattern} files found in {self.data_dir}"
                + (f" for date_range={date_range}" if date_range is not None else "")
            )
        self._day_cache: dict[Path, pd.DataFrame] = {}
        self._day_cache_numeric: dict[Path, dict[str, np.ndarray]] = {}

        self._funding_df = self._load_funding_history(Path(funding_rate_dir))

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
        self._resting_side: str | None = None  # bid or ask -- which side of the book we rest on
        self._spread_p95: float = 1.0
        self._ret1s_mean: float = 0.0
        self._ret1s_std: float = 1.0
        self._ret5s_mean: float = 0.0
        self._ret5s_std: float = 1.0
        self._rv60s_mean: float = 0.0
        self._rv60s_std: float = 1.0
        self._funding_rate_z: float = 0.0
        self._last_fill_tick_idx: int | None = None
        self._resting_placed_tick_idx: int | None = None
        self._episode_fills: list[dict] = []  # full history for the episode, for eval reporting

    # ~828MB/day measured in memory (pd.read_parquet + memory_usage(deep=True) on
    # real train/val days -- NOT the ~85MB the old comment here claimed, an order-
    # of-magnitude error). At n_envs=8 parallel workers, each with its own cache:
    # 8 * 5 * 0.86GB =~ 34.4GB cache + ~4.4GB non-cache overhead =~ 38.8GB total,
    # leaving ~11GB headroom out of this box's 50GB (measured baseline: a single
    # n_envs=8 job at the old default of 3 used ~25GB). See
    # docs/reports/phase3_l3_baseline_milestone.md, "Physics fix + init-strategy
    # probe" for the full arithmetic and why this was raised from 3.
    _MAX_CACHED_DAYS = 5

    # Columns _build_ticks/_precompute_feature_series/reset() actually touch --
    # confirmed via grep, not assumed (symbol/update_id/seq are never referenced
    # anywhere in this file). Column pruning at read time (2026-08-24, this
    # round's I/O investigation): measured ~0-4% real gain across 5 real days,
    # not the large win it might look like on paper -- cProfile + a direct
    # column-split measurement showed bids/asks (JSON-string levels) alone are
    # ~97% of _load_day's real decode cost, so pruning the OTHER (cheap, ~48ms
    # total) columns barely moves the total. Kept anyway: free, zero behavior
    # risk (n_rows is unaffected -- row count, not column selection -- so the
    # RNG draw sequence for file_idx/start is untouched), verified
    # byte-identical (see tests/test_reset_vectorization_equivalence.py).
    _NEEDED_DAY_COLUMNS = ["ts", "best_bid", "best_ask", "mid_price", "spread", "bids", "asks"]

    def _load_day(self, path: Path) -> pd.DataFrame:
        if path not in self._day_cache:
            if len(self._day_cache) >= self._MAX_CACHED_DAYS:
                self._day_cache.pop(next(iter(self._day_cache)))
            self._day_cache[path] = pd.read_parquet(path, columns=self._NEEDED_DAY_COLUMNS)
        return self._day_cache[path]

    def _load_day_numeric(self, path: Path) -> dict[str, np.ndarray]:
        """Numeric-format counterpart to _load_day -- same cache dict shape/eviction
        policy (FIFO, _MAX_CACHED_DAYS), a separate dict since the cached value type
        differs (dict[str, np.ndarray] vs pd.DataFrame). len(day_data["ts"]) is the
        numeric-format equivalent of len(day_df) -- both give the day's real row
        count, which reset() uses BEFORE the window is known (see reset()'s own
        comment on this) -- verified equal to the original for every converted file
        (see tests/test_reset_vectorization_equivalence.py and the conversion
        script's own per-file row-count check)."""
        if path not in self._day_cache_numeric:
            if len(self._day_cache_numeric) >= self._MAX_CACHED_DAYS:
                self._day_cache_numeric.pop(next(iter(self._day_cache_numeric)))
            self._day_cache_numeric[path] = _read_numeric_day(path)
        return self._day_cache_numeric[path]

    def _build_ticks_numeric(self, day_data: dict[str, np.ndarray], start: int, end: int) -> list[TickView]:
        """Numeric-format counterpart to _build_ticks -- constructs the same TickView
        list from pre-parsed arrays instead of per-row JSON parsing. bid_prices[i]
        etc. are row-slices (views, read-only since the source came from
        np.frombuffer) of the already-loaded (n, 20) arrays -- values are identical
        to what _parse_levels(row.bids) would produce from the original JSON at the
        same row (verified directly, not assumed -- see the conversion round's
        equivalence tests), just without paying the JSON-decode cost per tick."""
        ts = day_data["ts"][start:end]
        best_bid = day_data["best_bid"][start:end]
        best_ask = day_data["best_ask"][start:end]
        mid_price = day_data["mid_price"][start:end]
        spread = day_data["spread"][start:end]
        bid_prices = day_data["bid_prices"][start:end]
        bid_sizes = day_data["bid_sizes"][start:end]
        ask_prices = day_data["ask_prices"][start:end]
        ask_sizes = day_data["ask_sizes"][start:end]
        ticks = []
        for i in range(end - start):
            ticks.append(
                TickView(
                    ts=int(ts[i]), best_bid=float(best_bid[i]), best_ask=float(best_ask[i]),
                    mid_price=float(mid_price[i]), spread=float(spread[i]),
                    bid_prices=bid_prices[i], bid_sizes=bid_sizes[i],
                    ask_prices=ask_prices[i], ask_sizes=ask_sizes[i],
                )
            )
        return ticks

    def _load_funding_history(self, funding_dir: Path) -> pd.DataFrame:
        """Loaded once at construction (not per-episode -- it is a small,
        symbol-wide archive independent of which L2 day/window an episode
        samples). Missing directory/files degrade gracefully to an empty
        frame (funding_rate_z then stays at its 0.0 neutral default) rather
        than raising -- funding context is a genuinely optional enrichment,
        not something the env core loop depends on."""
        if not funding_dir.exists():
            return pd.DataFrame(columns=["calc_time", "last_funding_rate"])
        files = sorted(funding_dir.glob("*.parquet"))
        if not files:
            return pd.DataFrame(columns=["calc_time", "last_funding_rate"])
        df = pd.concat(
            [pd.read_parquet(f, columns=["calc_time", "last_funding_rate"]) for f in files],
            ignore_index=True,
        )
        return df.sort_values("calc_time").reset_index(drop=True)

    def _compute_funding_rate_z(self, episode_ts: int) -> float:
        """episode_ts and calc_time are both epoch-milliseconds (confirmed
        by direct inspection of both datasets, see module docstring) -- a
        plain numeric searchsorted is a valid as-of backward join, no unit
        conversion needed."""
        df = self._funding_df
        if len(df) == 0:
            return 0.0
        calc_times = df["calc_time"].to_numpy()
        idx = int(np.searchsorted(calc_times, episode_ts, side="right")) - 1
        if idx < 0:
            return 0.0  # episode predates all known funding history
        window_start = max(0, idx - self._FUNDING_LOOKBACK_PERIODS + 1)
        window = df["last_funding_rate"].to_numpy()[window_start : idx + 1]
        current = float(window[-1])
        return zscore(current, float(np.mean(window)), float(np.std(window)))

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

    def _precompute_feature_series(self) -> None:
        """Once per reset(), vectorized over the full self._ticks range (buffer +
        episode): every rolling-window feature series (idx 4, 5, 12, 40), so
        _build_obs() is an O(1) lookup per tick rather than recomputing a window
        from scratch every step. See module docstring for the window-size
        rationale and the touch-depletion trade-sign inference technique."""
        n = len(self._ticks)
        mids = np.array([t.mid_price for t in self._ticks], dtype=float)

        ticks_5s = max(1, round(5.0 / self.tick_interval_s))
        ticks_60s = max(1, round(60.0 / self.tick_interval_s))

        tick_rets = _rolling_return(mids, 1)  # per-tick (1-step) returns, building block for realized vol
        self._ret5s_series = _rolling_return(mids, ticks_5s)
        self._rv60s_series = _rolling_rms(tick_rets, ticks_60s)

        self._ret5s_mean = float(np.mean(self._ret5s_series)) if n else 0.0
        self._ret5s_std = float(np.std(self._ret5s_series)) if n > 1 else 1.0
        self._rv60s_mean = float(np.mean(self._rv60s_series)) if n else 0.0
        self._rv60s_std = float(np.std(self._rv60s_series)) if n > 1 else 1.0

        # Single pass over self._ticks builds every per-tick array/matrix the rest of
        # this method needs: capped-at-10 size matrices (book_depth_norm, idx 19-38,
        # unchanged from before) and, new (2026-08-24), FULL (uncapped) price/size
        # matrices + best_bid/best_ask arrays for the vectorized touch-depletion
        # computation below, which must search each tick's ENTIRE book (not just the
        # top 10 levels) to match TickView.qty_at_price's own unpadded search exactly.
        # Moved earlier in this method (was after the depletion loop) so the
        # capped-at-10 matrices can be built in the same pass -- book_depth_mean/std
        # below has no data dependency on signed/absvol, so this reordering does not
        # change any computed value, only which order two independent blocks run in.
        max_bid_levels = max((len(t.bid_prices) for t in self._ticks), default=0)
        max_ask_levels = max((len(t.ask_prices) for t in self._ticks), default=0)
        bid_size_matrix = np.zeros((n, 10), dtype=float)
        ask_size_matrix = np.zeros((n, 10), dtype=float)
        bid_price_full = np.zeros((n, max_bid_levels), dtype=float)
        bid_size_full = np.zeros((n, max_bid_levels), dtype=float)
        ask_price_full = np.zeros((n, max_ask_levels), dtype=float)
        ask_size_full = np.zeros((n, max_ask_levels), dtype=float)
        best_bid_arr = np.empty(n, dtype=float)
        best_ask_arr = np.empty(n, dtype=float)
        for i, t in enumerate(self._ticks):
            k = min(10, len(t.bid_sizes))
            bid_size_matrix[i, :k] = t.bid_sizes[:k]
            k = min(10, len(t.ask_sizes))
            ask_size_matrix[i, :k] = t.ask_sizes[:k]
            k = len(t.bid_prices)
            bid_price_full[i, :k] = t.bid_prices
            bid_size_full[i, :k] = t.bid_sizes
            k = len(t.ask_prices)
            ask_price_full[i, :k] = t.ask_prices
            ask_size_full[i, :k] = t.ask_sizes
            best_bid_arr[i] = t.best_bid
            best_ask_arr[i] = t.best_ask

        # Touch-depletion (idx 12/40 flow series): vectorized (2026-08-24, was a
        # python range(1, n) loop calling TickView.qty_at_price/np.isclose once per
        # tick -- cProfile showed this loop was ~77% of this method's own cost at
        # n~3600 ticks/reset). Same qty_at_price semantics via _vec_qty_at_price
        # above, same max(0, ...)/subtract/add arithmetic, verified byte-identical
        # against the original scalar loop (see tests/test_reset_vectorization_equivalence.py).
        signed = np.zeros(n, dtype=float)
        absvol = np.zeros(n, dtype=float)
        if n > 1:
            prev_best_bid = best_bid_arr[:-1]
            prev_best_ask = best_ask_arr[:-1]
            prev_bid_at_prev = _vec_qty_at_price(bid_price_full[:-1], bid_size_full[:-1], prev_best_bid)
            curr_bid_at_prev = _vec_qty_at_price(bid_price_full[1:], bid_size_full[1:], prev_best_bid)
            prev_ask_at_prev = _vec_qty_at_price(ask_price_full[:-1], ask_size_full[:-1], prev_best_ask)
            curr_ask_at_prev = _vec_qty_at_price(ask_price_full[1:], ask_size_full[1:], prev_best_ask)
            bid_dep = np.maximum(0.0, prev_bid_at_prev - curr_bid_at_prev)
            ask_dep = np.maximum(0.0, prev_ask_at_prev - curr_ask_at_prev)
            signed[1:] = ask_dep - bid_dep  # ask depletion -> taker BUY pressure; bid depletion -> taker SELL
            absvol[1:] = ask_dep + bid_dep

        signed_5s = _rolling_sum(signed, ticks_5s)
        abs_5s = _rolling_sum(absvol, ticks_5s)
        self._flow5s_series = np.clip(np.divide(signed_5s, abs_5s + 1e-9), -1.0, 1.0)

        signed_60s = _rolling_sum(signed, ticks_60s)
        abs_60s = _rolling_sum(absvol, ticks_60s)
        self._flow60s_series = np.clip(np.divide(signed_60s, abs_60s + 1e-9), -1.0, 1.0)

        # idx 19-38 (book_depth_norm): each of the 20 levels z-scored against ITS OWN
        # trailing rolling mean/std over TIME (Section 3.1's blanket "trailing rolling
        # window" rule, not a cross-sectional level-axis reading -- see module docstring).
        # Window reuses ticks_60s (no explicit window length is given in the spec for this
        # feature; 60s matches the already-established realized_vol_60s_z/
        # taker_buy_sell_ratio_1m window, a consistent, documented choice rather than an
        # arbitrary new one). bid_size_matrix/ask_size_matrix built above, in the
        # single combined pass over self._ticks.
        self._book_depth_mean = np.zeros((n, 20), dtype=float)
        self._book_depth_std = np.zeros((n, 20), dtype=float)
        for level in range(10):
            m, s = _rolling_mean_std(bid_size_matrix[:, level], ticks_60s)
            self._book_depth_mean[:, level] = m
            self._book_depth_std[:, level] = s
        for level in range(10):
            m, s = _rolling_mean_std(ask_size_matrix[:, level], ticks_60s)
            self._book_depth_mean[:, 10 + level] = m
            self._book_depth_std[:, 10 + level] = s

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        file_idx = int(self.np_random.integers(0, len(self._files)))
        day_path = self._files[file_idx]
        if self.use_numeric_format:
            day_data = self._load_day_numeric(day_path)
            n_rows = len(day_data["ts"])
        else:
            day_df = self._load_day(day_path)
            n_rows = len(day_df)

        # needed/start/end use self.lookback_ticks (NOT self._max_lookback_ticks) --
        # deliberately unchanged from Phase 2a so the RNG draw sequence, and therefore
        # which file/window/side/size a given seed resolves to, stays byte-identical.
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

        # Buffer for feature lookback (idx 4/5/12/40's 5s/60s windows): as much of
        # _max_lookback_ticks as is actually available before `start`. This is a pure
        # post-hoc slicing decision made AFTER start is drawn above -- it does not
        # perturb the RNG state, so it cannot change which window a given seed selects.
        buffer_ticks = min(self._max_lookback_ticks, start)
        if self.use_numeric_format:
            self._ticks = self._build_ticks_numeric(day_data, start - buffer_ticks, end)
        else:
            self._ticks = self._build_ticks(day_df, start - buffer_ticks, end)
        self._episode_start = buffer_ticks  # index into self._ticks where the real episode begins
        self._tick_idx = self._episode_start

        # legacy_ticks is EXACTLY the Phase 2a window (lookback_ticks + horizon_ticks
        # ticks, i.e. day_df.iloc[start-lookback_ticks:end]) -- a strict sub-range of the
        # now-larger self._ticks (which additionally carries up to _max_lookback_ticks of
        # buffer for idx 4/5/12/40's rolling windows below). Every Phase 2a-era statistic
        # (spread_p95, ret1s stats, ref_depth/qty_total sizing) stays scoped to
        # legacy_ticks so it is byte-identical to before. Getting this wrong is exactly
        # how ref_depth silently drifted during development: it is used to size
        # qty_total, which DOES feed reward/IS (unlike idx 2/3's z-score stats, which
        # only feed obs) -- caught via a controlled before/after comparison at identical
        # seed+date_range (qty_total: 23.9086 vs 22.7718 for one test seed, before this
        # fix scoped ref_depth back down).
        legacy_ticks = self._ticks[buffer_ticks - self.lookback_ticks :]

        # Local (per-episode-window) normalization stats -- see module docstring.
        mids = np.array([t.mid_price for t in legacy_ticks], dtype=float)
        spreads = np.array([t.spread for t in legacy_ticks], dtype=float)
        self._spread_p95 = float(np.percentile(spreads, 95)) if len(spreads) else 1.0
        if self._spread_p95 <= 0:
            self._spread_p95 = TICK_SIZE
        rets_1s = (mids[self.lookback_ticks:] - mids[: -self.lookback_ticks]) / mids[: -self.lookback_ticks]
        self._ret1s_mean = float(np.mean(rets_1s)) if len(rets_1s) else 0.0
        self._ret1s_std = float(np.std(rets_1s)) if len(rets_1s) > 1 else 1.0
        if self._ret1s_std <= 0:
            self._ret1s_std = 1e-9

        self._precompute_feature_series()
        episode_ts = self._ticks[self._episode_start].ts
        self._funding_rate_z = self._compute_funding_rate_z(episode_ts)

        # Order-size bound relative to typical top-of-book depth in this window
        # (task instruction: bound so episodes aren't degenerate by construction).
        top_depths = np.array(
            [t.bid_sizes[0] + t.ask_sizes[0] for t in legacy_ticks if len(t.bid_sizes) and len(t.ask_sizes)],
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
        self._last_fill_tick_idx = None
        self._resting_placed_tick_idx = None
        self._terminated_early = False

        # EXPERIMENTAL 5 (reward.py module docstring): computed once here,
        # BEFORE the real episode's own step() calls begin, over the
        # identical window this reset() just drew -- never touches
        # self._resting/self.qty_remaining/self._tick_idx, so it cannot
        # interfere with (or be interfered with by) the real episode that
        # follows. None (not computed) when the flag is off, so this costs
        # nothing unless deliberately enabled.
        self._twap_shadow_terminal_is_bps: float | None = None
        if self.reward_weights.subtract_twap_baseline:
            self._twap_shadow_terminal_is_bps = self._compute_twap_shadow_terminal_is()

        obs = self._build_obs()
        info = self._build_info(step_fills=[])
        return obs, info

    def _compute_twap_shadow_terminal_is(self) -> float:
        """Simulates a TWAP execution (n_slices=10) over the SAME window
        (self._ticks/self._episode_start/self.side/self.qty_total/
        self.arrival_price) the real episode just drew, entirely via LOCAL
        state -- never reads or writes self._resting/self.qty_remaining/
        self._tick_idx. Reuses the exact matching-engine primitives step()
        itself uses (walk_market_fill, update_queue, TickView.qty_at_price,
        compute_implementation_shortfall, self._estimate_trade_volume) --
        only the TWAP-specific slicing/routing decision is duplicated here
        rather than imported, since scripts/phase2a_sanity_suite.py (where
        the canonical TWAPPolicy lives) is documented as a throwaway
        evaluation script, not something core env code should import from.
        tests/test_lob_execution_env_features.py's
        test_twap_shadow_matches_real_twap_policy_exactly runs the REAL
        TWAPPolicy through the REAL env on the same seeds and asserts this
        method's output matches info["implementation_shortfall"] exactly --
        that integration test is what actually guards against drift between
        this duplicated decision logic and the canonical policy, not just
        code review.

        See RewardWeights.subtract_twap_baseline and the reward.py module
        docstring's EXPERIMENTAL 5 section for why this exists (variance
        reduction, not an objective change) and docs/reports/ for the
        measured per-reset cost of calling this."""
        n_slices = 10
        side = self.side
        qty_total = self.qty_total
        qty_remaining = qty_total
        resting: QueueState | None = None
        resting_price: float | None = None
        resting_side: str | None = None
        episode_fills: list[dict] = []
        current_slice = -1
        qty_remaining_at_slice_start = qty_total

        tick_idx = self._episode_start
        for _ in range(self.horizon_ticks + 1):
            tick = self._ticks[tick_idx]

            # Snapshot BEFORE this tick's resting-order evolution runs --
            # this is exactly what TWAPPolicy.act() sees in the real
            # run_episode() loop, since act() is always called BEFORE
            # step()'s own evolution for the tick about to be processed
            # (env.qty_remaining/env._resting reflect the END of the
            # PREVIOUS step() call, not this tick's evolution yet). The
            # decision below (slice accounting, is_last_tick_of_slice,
            # size_frac, and whether a new order may be placed) must use
            # THESE snapshots, not the live post-evolution values, or the
            # shadow silently sees a one-tick-later reality than the real
            # policy decided against -- this was caught by
            # test_subtract_twap_baseline_matches_real_twap_policy_exactly
            # diverging by a small but real amount before this fix.
            qty_remaining_decision = qty_remaining
            resting_active_decision = resting is not None

            if resting is not None and not resting.is_resolved:
                v_trade, q_p_before = self._estimate_trade_volume(
                    tick_idx - 1, tick_idx, resting_price, resting_side
                )
                prev_filled = resting.filled_qty
                resting = update_queue(resting, v_trade=v_trade, v_cancel=0.0, q_p_before=q_p_before)
                newly_filled = resting.filled_qty - prev_filled
                if newly_filled > 0:
                    episode_fills.append({"price": resting_price, "qty": newly_filled, "is_maker": True})
                    qty_remaining = max(0.0, qty_remaining - newly_filled)
                if resting.is_resolved:
                    resting = None
                    resting_price = None
                    resting_side = None

            ticks_elapsed = tick_idx - self._episode_start
            slice_ticks = self.horizon_ticks / n_slices
            slice_idx = min(n_slices - 1, int(ticks_elapsed // slice_ticks))
            slice_end_tick = (slice_idx + 1) * slice_ticks
            if slice_idx != current_slice:
                current_slice = slice_idx
                qty_remaining_at_slice_start = qty_remaining_decision
            slice_target = qty_total / n_slices
            filled_this_slice = qty_remaining_at_slice_start - qty_remaining_decision
            slice_unfilled = max(0.0, slice_target - filled_this_slice)

            if slice_unfilled > 1e-9 and qty_remaining_decision > 1e-9:
                is_last_tick_of_slice = (ticks_elapsed + 1) >= slice_end_tick
                # TWAPPolicy.act() computes this same continuous fraction, but the
                # action space only has 5 discrete SIZE_FRACTIONS -- the real system
                # snaps to the nearest one (scripts/phase2a_sanity_suite.py's
                # _closest_size_frac_idx) before step() ever sizes an order. Skipping
                # this rounding was the actual source of the mismatch
                # test_subtract_twap_baseline_matches_real_twap_policy_exactly caught
                # (an 0.02bps drift from using the un-rounded continuous fraction).
                frac_of_remaining = min(1.0, slice_unfilled / qty_remaining_decision)
                size_frac = min(SIZE_FRACTIONS, key=lambda f: abs(f - frac_of_remaining))

                if is_last_tick_of_slice:
                    # MARKET, forcing this slice's completion -- tears down any
                    # resting order first, exactly as step()'s own MARKET path does.
                    resting = None
                    resting_price = None
                    resting_side = None
                    # Actual requested quantity uses the LIVE (post-evolution)
                    # qty_remaining, matching step()'s own self.qty_remaining
                    # usage at the point it applies the action -- only the
                    # DECISION (size_frac, above) uses the pre-evolution snapshot.
                    mkt_qty = min(size_frac * qty_remaining, qty_remaining)
                    if mkt_qty > 0:
                        book_prices, book_sizes = (
                            (tick.ask_prices, tick.ask_sizes) if side == 1
                            else (tick.bid_prices, tick.bid_sizes)
                        )
                        level_fills, qty_unfilled = walk_market_fill(mkt_qty, book_prices, book_sizes)
                        for level_price, level_qty in level_fills:
                            episode_fills.append({"price": level_price, "qty": level_qty, "is_maker": False})
                        filled_qty = mkt_qty - qty_unfilled
                        qty_remaining = max(0.0, qty_remaining - filled_qty)
                elif not resting_active_decision:
                    # LIMIT at the touch (offset 0), same as TWAPPolicy.act().
                    # Gated on the PRE-evolution resting snapshot, not the live
                    # value -- if evolution just resolved a fill THIS tick,
                    # TWAPPolicy.act() had already decided HOLD before it knew
                    # that, and the real system honors that stale decision.
                    if side == 1:
                        price = round(tick.best_bid, 1)
                        place_side = "bid"
                        crossed = price >= tick.best_ask
                    else:
                        price = round(tick.best_ask, 1)
                        place_side = "ask"
                        crossed = price <= tick.best_bid
                    size = min(size_frac * qty_remaining, qty_remaining)
                    if size > 0:
                        if crossed:
                            book_prices, book_sizes = (
                                (tick.ask_prices, tick.ask_sizes) if side == 1
                                else (tick.bid_prices, tick.bid_sizes)
                            )
                            level_fills, qty_unfilled = walk_market_fill(size, book_prices, book_sizes)
                            for level_price, level_qty in level_fills:
                                episode_fills.append({"price": level_price, "qty": level_qty, "is_maker": False})
                            filled_qty = size - qty_unfilled
                            qty_remaining = max(0.0, qty_remaining - filled_qty)
                        else:
                            q_ahead = tick.qty_at_price(price, place_side)
                            resting = QueueState(q_ahead=q_ahead, own_qty_remaining=size)
                            resting_price = price
                            resting_side = place_side
                # else: decision snapshot already had a resting order -> HOLD,
                # nothing to do this tick (matches the real one-tick lag).

            if qty_remaining <= 1e-12:
                break
            tick_idx += 1
            if (tick_idx - self._episode_start) >= self.horizon_ticks or tick_idx >= len(self._ticks):
                break

        terminal_tick_idx = min(tick_idx, len(self._ticks) - 1)
        terminal_is = compute_implementation_shortfall(
            side=side, fills=episode_fills, qty_total=qty_total,
            arrival_price=self.arrival_price, terminal_mid_price=self._ticks[terminal_tick_idx].mid_price,
            fee_bps_per_fill=self.fee_bps_per_fill,
        )
        return terminal_is.is_total_bps

    def _current_tick(self) -> TickView:
        return self._ticks[self._tick_idx]

    def _ticks_since_own_fill_norm(self) -> float:
        """Obs idx 14. Factored out (Part B, docs/reports/
        phase3_l3_baseline_milestone.md) so step()'s new staleness reward
        term and _build_obs() share one formula rather than risk drifting
        apart -- value and meaning are unchanged from before."""
        if self._last_fill_tick_idx is None:
            return 1.0
        return float(np.clip((self._tick_idx - self._last_fill_tick_idx) / self.horizon_ticks, 0.0, 1.0))

    def _ticks_since_placement_norm(self) -> float:
        """Part A (docs/reports/phase3_l3_baseline_milestone.md): reward-internal
        only, NOT part of the 42-dim observation vector -- feeds only the new
        eta_replace reward term, unlike _ticks_since_own_fill_norm() which feeds
        both obs idx 14 and r_stale. Normalized against
        _PLACEMENT_STALENESS_WINDOW_TICKS (see its own docstring for why), not
        horizon_ticks. self._resting_placed_tick_idx is stamped on every new
        placement in _place_limit(), including the fresh order from a
        CANCEL_AND_REPLACE -- unlike self._last_fill_tick_idx, it is never
        explicitly cleared when self._resting resolves, matching that field's own
        init-only pattern exactly; this is safe because the reward term this feeds
        is itself gated on `resting`, so a stale leftover value between one
        order resolving and the next placement can never actually be read."""
        if self._resting_placed_tick_idx is None:
            return 1.0
        ticks_since_placement = self._tick_idx - self._resting_placed_tick_idx
        return float(np.clip(ticks_since_placement / self._PLACEMENT_STALENESS_WINDOW_TICKS, 0.0, 1.0))

    def _compute_l2_target_slice_ratio(self) -> float:
        """Default: what a fixed-TWAP schedule would have executed by now, as a
        fraction of the full parent order (linear in elapsed time -- only the
        minimal scheduling arithmetic, not scripts/phase2a_sanity_suite.py full
        N-slice TWAPPolicy). Overridable via l2_target_slice_ratio_override for
        when a real L2 agent exists to supply its own target."""
        if self.l2_target_slice_ratio_override is not None:
            return float(np.clip(self.l2_target_slice_ratio_override, 0.0, 1.0))
        if self.horizon_ticks <= 0:
            return 0.0
        ticks_elapsed = self._tick_idx - self._episode_start
        return float(np.clip(ticks_elapsed / self.horizon_ticks, 0.0, 1.0))

    def _build_obs(self) -> np.ndarray:
        tick = self._current_tick()
        pos = self._tick_idx  # index into self._ticks / the precomputed feature series
        ticks_elapsed = self._tick_idx - self._episode_start
        time_remaining_norm = max(0.0, min(1.0, 1.0 - ticks_elapsed / self.horizon_ticks))

        inventory_remaining_norm = self.side * (self.qty_remaining / self.qty_total) if self.qty_total > 0 else 0.0
        inventory_remaining_norm = float(np.clip(inventory_remaining_norm, -1.0, 1.0))

        spread_norm = float(np.clip(tick.spread / self._spread_p95, 0.0, 1.0))

        prior = self._ticks[self._tick_idx - self.lookback_ticks]
        ret_1s = (tick.mid_price - prior.mid_price) / prior.mid_price if prior.mid_price > 0 else 0.0
        mid_return_1s_z = float(np.clip((ret_1s - self._ret1s_mean) / self._ret1s_std, -5.0, 5.0))

        mid_return_5s_z = zscore(self._ret5s_series[pos], self._ret5s_mean, self._ret5s_std)
        realized_vol_60s_z = zscore(self._rv60s_series[pos], self._rv60s_mean, self._rv60s_std)

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

        # idx 10-11: genuinely blocked, not a fresh stub -- see module docstring.
        cancel_add_ratio_bid = 0.0
        cancel_add_ratio_ask = 0.0

        trade_flow_imbalance_5s = float(self._flow5s_series[pos])

        qpr = queue_position_ratio(self._resting) if self._resting is not None else -1.0

        ticks_since_own_fill_norm = self._ticks_since_own_fill_norm()

        l2_target_slice_ratio = self._compute_l2_target_slice_ratio()
        l2_urgency = float(np.clip(self.l2_urgency, 0.0, 1.0))
        l1_risk_score = float(np.clip(self.l1_risk_score, -1.0, 1.0))
        l1_confidence = float(np.clip(self.l1_confidence, 0.0, 1.0))

        # idx 19-38: each of the 20 bid+ask levels z-scored against ITS OWN trailing
        # rolling mean/std over time (Section 3.1's blanket "trailing rolling window"
        # rule -- see _precompute_feature_series() and module docstring; NOT a
        # cross-sectional level-axis reading). Missing levels (book shallower than 10 on
        # a side) pad with 0.0.
        bid_sizes_10 = np.zeros(10, dtype=float)
        bid_sizes_10[: min(10, len(tick.bid_sizes))] = tick.bid_sizes[:10]
        ask_sizes_10 = np.zeros(10, dtype=float)
        ask_sizes_10[: min(10, len(tick.ask_sizes))] = tick.ask_sizes[:10]
        sizes_20 = np.concatenate([bid_sizes_10, ask_sizes_10])
        book_depth_norm = [
            zscore(float(sizes_20[k]), self._book_depth_mean[pos, k], self._book_depth_std[pos, k])
            for k in range(20)
        ]

        taker_buy_sell_ratio_1m = float(self._flow60s_series[pos])

        own_open_orders_norm = 0.0
        if self._resting is not None and self.qty_total > 0:
            own_open_orders_norm = float(np.clip(self._resting.own_qty_remaining / self.qty_total, 0.0, 1.0))

        values = [
            time_remaining_norm, inventory_remaining_norm, spread_norm, mid_return_1s_z,
            mid_return_5s_z, realized_vol_60s_z, obi_1, obi_5, obi_10, micro_mid_dev_ticks,
            cancel_add_ratio_bid, cancel_add_ratio_ask, trade_flow_imbalance_5s, qpr,
            ticks_since_own_fill_norm, l2_target_slice_ratio, l2_urgency, l1_risk_score,
            l1_confidence, *book_depth_norm, self._funding_rate_z, taker_buy_sell_ratio_1m,
            own_open_orders_norm,
        ]
        return np.array(values, dtype=np.float32)

    def _build_info(
        self,
        *,
        step_fills: list[dict],
        canceled_via_market: bool = False,
        canceled_via_replace: bool = False,
    ) -> dict[str, Any]:
        tick = self._current_tick()
        return {
            "side": self.side,
            "qty_total": self.qty_total,
            "qty_remaining": self.qty_remaining,
            "arrival_price": self.arrival_price,
            "mid_price": tick.mid_price,
            "scenario_depth_ratio": getattr(self, "_scenario_depth_ratio", None),
            "fills_this_step": list(step_fills),
            # Kept as the OR of the two so existing consumers of this info key
            # keep working unchanged; the split is exposed alongside it.
            "canceled_unfilled": canceled_via_market or canceled_via_replace,
            "canceled_via_market": canceled_via_market,
            "canceled_via_replace": canceled_via_replace,
            "resting_q_ahead": self._resting.q_ahead if self._resting is not None else None,
            "resting_own_remaining": self._resting.own_qty_remaining if self._resting is not None else None,
            "tick_idx": self._tick_idx,
            "ticks_elapsed": self._tick_idx - self._episode_start,
        }

    def _estimate_trade_volume(self, prev_idx: int, curr_idx: int, price: float, side: str) -> tuple[float, float]:
        """Approximates v_trade as the full observed decrease in resting qty
        at `price` between two ticks (v_cancel=0 in this adapter) -- see
        module docstring for why this data source cannot separate the two."""
        prev_qty = self._ticks[prev_idx].qty_at_price(price, side)
        curr_qty = self._ticks[curr_idx].qty_at_price(price, side)
        v_trade = max(0.0, prev_qty - curr_qty)
        return v_trade, prev_qty

    def _place_limit(self, tick: TickView, offset: int, size_frac: float) -> list[dict]:
        if self.side == 1:
            price = round(tick.best_bid + offset * TICK_SIZE, 1)
            side = "bid"
            crossed = price >= tick.best_ask
        else:
            price = round(tick.best_ask - offset * TICK_SIZE, 1)
            side = "ask"
            crossed = price <= tick.best_bid

        size = min(size_frac * self.qty_remaining, self.qty_remaining)
        if size <= 0:
            return []

        if crossed:
            # A price that crosses the opposing side is marketable, not restable: no
            # real exchange lets a bid rest above the current ask (or a sell rest below
            # the current bid) -- it trades immediately against the opposing side
            # instead. Route through walk_market_fill() against the OPPOSING side's
            # visible depth, exactly like ORDER_TYPE_MARKET does in step() -- same
            # mechanism, just reached via a crossing LIMIT/CANCEL_REPLACE price rather
            # than an explicit MARKET action. Before this fix, a crossing price fell
            # through to the q_ahead lookup below and became an ordinary resting ghost
            # order -- see docs/reports/phase3_l3_baseline_milestone.md for how that,
            # combined with the qty_at_price tolerance bug above, produced fictitious
            # fills for ~45% of placements in the offset sweep.
            book_prices, book_sizes = (
                (tick.ask_prices, tick.ask_sizes) if self.side == 1
                else (tick.bid_prices, tick.bid_sizes)
            )
            level_fills, qty_unfilled = walk_market_fill(size, book_prices, book_sizes)
            fills = [{"price": p, "qty": q, "is_maker": False} for p, q in level_fills]
            filled_qty = size - qty_unfilled
            self.qty_remaining = max(0.0, self.qty_remaining - filled_qty)
            return fills

        # Q_ahead(0): visible resting volume at this price at placement (Section 2.4
        # point 1); 0.0 if the price falls outside the visible top-20 book -- flagged
        # in the module docstring as likely optimistic for deep/passive prices. (Prices
        # that cross the opposing side never reach here -- handled above instead.)
        q_ahead = tick.qty_at_price(price, side)
        self._resting = QueueState(q_ahead=q_ahead, own_qty_remaining=size)
        self._resting_price = price
        self._resting_side = side
        self._resting_placed_tick_idx = self._tick_idx
        return []

    def step(self, action):
        tick_before = self._current_tick()
        step_fills: list[dict] = []
        # Two flags instead of one canceled_unfilled: MARKET and
        # CANCEL_AND_REPLACE tear a resting order down identically, but are
        # priced differently by step_reward()'s r_queue -- see reward.py.
        canceled_via_market = False
        canceled_via_replace = False
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

        # Teardown is identical for both actions (resting state cleared, queue
        # state captured for the reward). They are split into two handlers
        # purely so the reward can charge them differently: MARKET pays the
        # full -beta - gamma*queue_ratio, CANCEL_AND_REPLACE pays only the
        # queue-weighted part. See step_reward()'s r_queue block.
        if order_type == ORDER_TYPE_MARKET and self._resting is not None:
            canceled_via_market = True
            queue_ahead_at_cancel = self._resting.q_ahead
            queue_at_level = self._resting.q_ahead + self._resting.own_qty_remaining
            self._resting = None
            self._resting_price = None
            self._resting_side = None
        elif order_type == ORDER_TYPE_CANCEL_REPLACE and self._resting is not None:
            canceled_via_replace = True
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
                # Normally [] (a resting order was created); non-empty only when the
                # requested price crossed the opposing side and _place_limit() routed
                # it through walk_market_fill() instead -- see _place_limit().
                step_fills.extend(self._place_limit(tick_before, offset, size_frac))
        # HOLD (order_type == 0): nothing further.

        if step_fills:
            self._last_fill_tick_idx = self._tick_idx

        r = step_reward(
            self.reward_weights, side=self.side, fills=step_fills,
            arrival_price=self.arrival_price, mid_price=tick_before.mid_price,
            qty_remaining=self.qty_remaining, qty_total=self.qty_total,
            dt=self.tick_interval_s, l1_risk_score=self.l1_risk_score,
            canceled_via_market=canceled_via_market,
            canceled_via_replace=canceled_via_replace,
            queue_ahead_at_cancel=queue_ahead_at_cancel, queue_at_level=queue_at_level,
            resting=self._resting is not None,
            ticks_since_own_fill_norm=self._ticks_since_own_fill_norm(),
            ticks_since_placement_norm=self._ticks_since_placement_norm(),
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
            # EXPERIMENTAL 5 (reward.py module docstring): baseline-subtracted
            # for the REWARD only when enabled -- info["implementation_shortfall"]
            # below still reports terminal_is UNCHANGED, the real, un-adjusted
            # execution outcome. Only this scalar used for r is ever adjusted.
            terminal_is_for_reward = terminal_is.is_total_bps
            if self.reward_weights.subtract_twap_baseline and self._twap_shadow_terminal_is_bps is not None:
                terminal_is_for_reward -= self._twap_shadow_terminal_is_bps
            r += -self.reward_weights.kappa * terminal_is_for_reward

        obs = self._build_obs()
        info = self._build_info(
            step_fills=step_fills,
            canceled_via_market=canceled_via_market,
            canceled_via_replace=canceled_via_replace,
        )
        if terminal_is is not None:
            info["implementation_shortfall"] = terminal_is

        return obs, r, terminated, truncated, info
