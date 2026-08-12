from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import websockets


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


@dataclass
class L2CaptureConfig:
    symbol: str = "BTCUSDT"
    out_dir: Path = Path("data/raw_l2")
    top_levels: int = 10
    shard_rows: int = 2000
    flush_interval_s: float = 10.0
    rest_timeout_s: int = 15
    stream: str = "depth@100ms" 


class L2CaptureDaemon:
    def __init__(self, config: L2CaptureConfig) -> None:
        self.config = config
        self.book = L2Book()
        self.rows: list[dict[str, Any]] = []
        self.last_flush = datetime.now(timezone.utc)
        self.shard_seq = 0
        self.log = logging.getLogger("l2_capture")
        (self.config.out_dir / self.config.symbol).mkdir(parents=True, exist_ok=True)

    @property
    def _ws_url(self) -> str:
        return f"wss://fstream.binance.com/ws/{self.config.symbol.lower()}@{self.config.stream}"

    @property
    def _rest_url(self) -> str:
        return "https://fapi.binance.com/fapi/v1/depth"

    def _seed_snapshot(self) -> bool:
        try:
            resp = requests.get(
                self._rest_url,
                params={"symbol": self.config.symbol, "limit": 1000},
                timeout=self.config.rest_timeout_s,
            )
            resp.raise_for_status()
            data = resp.json()
            self.book.apply_snapshot(
                {"bids": data.get("bids", []), "asks": data.get("asks", [])},
                seq=int(data["lastUpdateId"]),
            )
            self.log.info("Seeded snapshot at lastUpdateId=%s", self.book.last_u)
            return True
        except Exception as exc:
            self.log.warning("Snapshot seed failed; fallback to stream-only mode: %s", exc)
            return False

    def _top_levels(self, side: str) -> list[list[float]]:
        levels = self.book.bids if side == "bid" else self.book.asks
        ordered = sorted(levels.items(), key=lambda kv: kv[0], reverse=(side == "bid"))
        return [[price, qty] for price, qty in ordered[: self.config.top_levels]]

    def _append_row(self, event: dict[str, Any]) -> None:
        best_bid = self.book.best_bid()
        best_ask = self.book.best_ask()
        if best_bid is None or best_ask is None:
            return
        self.rows.append(
            {
                "symbol": self.config.symbol,
                "event_time": int(event.get("E", 0)),
                "update_id": int(event.get("u", 0)),
                "prev_update_id": int(event.get("pu", 0)),
                "best_bid": float(best_bid),
                "best_ask": float(best_ask),
                "mid_price": float((best_bid + best_ask) / 2.0),
                "spread": float(best_ask - best_bid),
                "bids": json.dumps(self._top_levels("bid")),
                "asks": json.dumps(self._top_levels("ask")),
            }
        )

    def _should_flush(self) -> bool:
        if len(self.rows) >= self.config.shard_rows:
            return True
        elapsed = (datetime.now(timezone.utc) - self.last_flush).total_seconds()
        return bool(self.rows and elapsed >= self.config.flush_interval_s)

    def _flush(self) -> None:
        if not self.rows:
            return
        now = datetime.now(timezone.utc)
        day_dir = self.config.out_dir / self.config.symbol / now.strftime("%Y-%m-%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        path = day_dir / f"l2-{self.config.symbol}-{now.strftime('%Y%m%dT%H%M%S')}-{self.shard_seq:06d}.parquet"
        self.shard_seq += 1
        pd.DataFrame(self.rows).to_parquet(path, index=False)
        self.log.info("Flushed %d rows -> %s", len(self.rows), path)
        self.rows.clear()
        self.last_flush = now

    def _apply_event(self, event: dict[str, Any]) -> None:
        u = int(event.get("u", 0))
        pu = int(event.get("pu", 0))

        if self.book.last_u is None:
            apply_diff(self.book, {"u": u, "pu": pu, "bids": event.get("b", []), "asks": event.get("a", [])})
            self._append_row(event)
            return

        if u <= self.book.last_u:
            return

        if pu != self.book.last_u:
            self.log.warning(
                "Sequence gap detected: expected pu=%s got pu=%s (u=%s)",
                self.book.last_u,
                pu,
                u,
            )
            if not self._seed_snapshot():
                self.book.apply_snapshot({"bids": [], "asks": []}, seq=pu)

        apply_diff(self.book, {"u": u, "pu": pu, "bids": event.get("b", []), "asks": event.get("a", [])})
        self._append_row(event)

    async def run_forever(self) -> None:
        self._seed_snapshot()
        while True:
            try:
                async with websockets.connect(self._ws_url, ping_interval=20, ping_timeout=20) as ws:
                    self.log.info("Connected websocket %s", self._ws_url)
                    async for msg in ws:
                        event = json.loads(msg)
                        if event.get("e") != "depthUpdate":
                            continue
                        self._apply_event(event)
                        if self._should_flush():
                            self._flush()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.log.exception("Websocket loop error: %s", exc)
                await asyncio.sleep(1.0)
            finally:
                self._flush()
