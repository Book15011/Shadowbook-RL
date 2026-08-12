from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class L2Book:
    bids: dict[float, float] = field(default_factory=dict)
    asks: dict[float, float] = field(default_factory=dict)
    last_u: int | None = None
    last_pu: int | None = None

    def apply_snapshot(self, snapshot: dict[str, Any], seq: int | None = None) -> None:
        self.bids = {float(price): float(size) for price, size in snapshot.get("bids", [])}
        self.asks = {float(price): float(size) for price, size in snapshot.get("asks", [])}
        self.last_u = seq
        self.last_pu = seq

    def best_bid(self) -> float | None:
        if not self.bids:
            return None
        return max(self.bids)

    def best_ask(self) -> float | None:
        if not self.asks:
            return None
        return min(self.asks)


def apply_diff(book: L2Book, diff: dict[str, Any]) -> None:
    seq = diff.get("u")
    prev_seq = diff.get("pu")
    if book.last_u is not None and seq is not None and (
        (prev_seq is not None and prev_seq != book.last_u) or seq > book.last_u + 1
    ):
        book.apply_snapshot({"bids": [], "asks": []}, seq=seq)
    for price, size in diff.get("bids", []):
        price = float(price)
        size = float(size)
        if size <= 0:
            book.bids.pop(price, None)
        else:
            book.bids[price] = size
    for price, size in diff.get("asks", []):
        price = float(price)
        size = float(size)
        if size <= 0:
            book.asks.pop(price, None)
        else:
            book.asks[price] = size
    book.last_u = seq
    book.last_pu = prev_seq


def reconstruct_book(events: list[dict[str, Any]]) -> L2Book:
    book = L2Book()
    for event in events:
        apply_diff(book, event)
    return book
