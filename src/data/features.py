from __future__ import annotations

from collections import deque
from typing import Deque

import numpy as np


def mid_price(bids: np.ndarray | list[float], asks: np.ndarray | list[float]) -> float:
    bids = np.asarray(bids, dtype=float)
    asks = np.asarray(asks, dtype=float)
    return float((bids[0] + asks[0]) / 2.0)


def micro_price(
    bids: np.ndarray | list[float],
    asks: np.ndarray | list[float],
    bid_sizes: np.ndarray | list[float] | None = None,
    ask_sizes: np.ndarray | list[float] | None = None,
) -> float:
    bids = np.asarray(bids, dtype=float)
    asks = np.asarray(asks, dtype=float)
    bid_sizes = np.asarray(bid_sizes if bid_sizes is not None else np.ones(len(bids)), dtype=float)
    ask_sizes = np.asarray(ask_sizes if ask_sizes is not None else np.ones(len(asks)), dtype=float)
    numerator = bids[0] * ask_sizes[0] + asks[0] * bid_sizes[0]
    denominator = bid_sizes[0] + ask_sizes[0]
    return float(numerator / denominator)


def obi(
    bids: np.ndarray | list[float],
    asks: np.ndarray | list[float],
    bid_sizes: np.ndarray | list[float] | None = None,
    ask_sizes: np.ndarray | list[float] | None = None,
    k: int = 1,
) -> float:
    bids = np.asarray(bids, dtype=float)
    asks = np.asarray(asks, dtype=float)
    bid_sizes = np.asarray(bid_sizes if bid_sizes is not None else np.ones(len(bids)), dtype=float)
    ask_sizes = np.asarray(ask_sizes if ask_sizes is not None else np.ones(len(asks)), dtype=float)
    bid_volume = float(np.sum(bid_sizes[:k]))
    ask_volume = float(np.sum(ask_sizes[:k]))
    if bid_volume + ask_volume == 0:
        return 0.0
    return float((bid_volume - ask_volume) / (bid_volume + ask_volume))


class CancelAddTracker:
    def __init__(self, window: int = 10) -> None:
        self.window = window
        self._bid_events: Deque[tuple[float, float]] = deque(maxlen=window)
        self._ask_events: Deque[tuple[float, float]] = deque(maxlen=window)

    def observe(self, side: str, cancel: float, add: float) -> None:
        event = (float(cancel), float(add))
        if side == "bid":
            self._bid_events.append(event)
        elif side == "ask":
            self._ask_events.append(event)
        else:
            raise ValueError(side)

    def ratio(self, side: str) -> float:
        events = self._bid_events if side == "bid" else self._ask_events
        if not events:
            return 0.0
        cancel_total = len(events)
        add_total = len(events)
        return float(cancel_total / (cancel_total + add_total))

    def window_ratio(self, side: str) -> float:
        return self.ratio(side)
