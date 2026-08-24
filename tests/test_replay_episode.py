"""Regression test for scripts/replay_episode.py's reconstruct_child_orders() --
the one piece of real logic in that script (everything else is plotting). Hand-built
tick_records fixtures (no env/model needed), covering the four outcomes the price
panel distinguishes: a resting order that later gets a real maker fill, one crossed
immediately on its own placement tick (routed through walk_market_fill, same as a
market order -- see _place_limit()'s own source), one replaced before filling, and
one still open at episode end. Also checks placement-price computation matches
_place_limit()'s own formula (tick.best_bid/ask +/- offset*TICK_SIZE) for both sides.
"""
from __future__ import annotations

from scripts.replay_episode import reconstruct_child_orders
from src.envs.lob_execution_env import ORDER_TYPE_CANCEL_REPLACE, ORDER_TYPE_HOLD, ORDER_TYPE_LIMIT, ORDER_TYPE_MARKET


def _tick(tick_idx, order_type, offset=0, best_bid=100.0, best_ask=100.2, fills=None):
    return {
        "tick_idx": tick_idx, "ts": tick_idx, "mid_price": (best_bid + best_ask) / 2,
        "best_bid": best_bid, "best_ask": best_ask, "order_type": order_type, "offset": offset,
        "size_frac": 1.0, "fills": fills or [], "canceled_via_market": False,
        "canceled_via_replace": False, "qty_remaining": 0.0,
    }


def test_resting_order_later_maker_fill():
    records = [
        _tick(0, ORDER_TYPE_LIMIT, offset=-2),
        _tick(1, ORDER_TYPE_HOLD),
        _tick(2, ORDER_TYPE_HOLD, fills=[{"price": 99.8, "qty": 3.0, "is_maker": True}]),
    ]
    orders = reconstruct_child_orders(records, side=1)
    assert len(orders) == 1
    o = orders[0]
    assert o.kind == "resting"
    assert o.outcome == "filled"
    assert o.placement_price == 99.8  # best_bid(100.0) + offset(-2)*TICK_SIZE(0.1)
    assert o.fill_ticks == [2] and o.fill_qtys == [3.0]


def test_crossed_placement_fills_immediately_not_left_open():
    # offset=+5 crosses -- _place_limit() routes this through walk_market_fill(),
    # same mechanism as ORDER_TYPE_MARKET, so the fill on the SAME tick is is_maker=False.
    records = [_tick(0, ORDER_TYPE_LIMIT, offset=5, fills=[{"price": 100.2, "qty": 4.0, "is_maker": False}])]
    orders = reconstruct_child_orders(records, side=1)
    assert len(orders) == 1
    o = orders[0]
    assert o.kind == "market"  # relabeled, not left as a "still resting" order
    assert o.outcome == "filled"
    assert o.fill_qtys == [4.0]


def test_replaced_before_filling():
    records = [
        _tick(0, ORDER_TYPE_LIMIT, offset=-2),
        _tick(5, ORDER_TYPE_CANCEL_REPLACE, offset=-1),  # replaces the first, still open at end
    ]
    orders = reconstruct_child_orders(records, side=1)
    assert len(orders) == 2
    assert orders[0].outcome == "replaced"
    assert orders[1].outcome == "open_at_episode_end"


def test_pure_market_order_recorded_separately():
    records = [_tick(0, ORDER_TYPE_MARKET, fills=[{"price": 100.2, "qty": 2.0, "is_maker": False}])]
    orders = reconstruct_child_orders(records, side=1)
    assert len(orders) == 1
    assert orders[0].kind == "market"
    assert orders[0].placement_price == 100.2


def test_placement_price_matches_place_limit_formula_sell_side():
    # side=-1 (sell): price = best_ask - offset*TICK_SIZE (see _place_limit()).
    records = [_tick(0, ORDER_TYPE_LIMIT, offset=3, best_bid=100.0, best_ask=100.2)]
    orders = reconstruct_child_orders(records, side=-1)
    assert orders[0].placement_price == 99.9  # 100.2 - 3*0.1


def test_partial_crossing_fill_marked_filled_not_left_open():
    # _place_limit()'s `crossed` branch returns unconditionally after
    # walk_market_fill() -- confirmed from source, not assumed -- so a crossing
    # placement that only PARTIALLY fills (visible book ran out of depth) still
    # never rests a remainder. Two non-maker fills on the placement's own tick
    # (walk_market_fill() can span multiple price levels) -- both must be
    # attributed, and the order must read "filled", not "still open".
    records = [_tick(0, ORDER_TYPE_LIMIT, offset=5, fills=[
        {"price": 100.2, "qty": 2.0, "is_maker": False},
        {"price": 100.3, "qty": 1.0, "is_maker": False},
    ])]
    orders = reconstruct_child_orders(records, side=1)
    assert len(orders) == 1
    o = orders[0]
    assert o.kind == "market"
    assert o.outcome == "filled"
    assert o.fill_qtys == [2.0, 1.0]
    assert o.fill_prices == [100.2, 100.3]
