"""Hand-computed fixtures for matching_engine.py -- every expected value in
this file is worked out by hand in the comments, not derived by running the
code and asserting whatever it happened to produce."""
import math

import pytest

from src.envs.matching_engine import (
    QueueState,
    expected_wait_time,
    queue_position_ratio,
    update_queue,
    walk_market_fill,
)


def test_normal_depletion_no_fill():
    # Q_ahead=100, v_trade=30, v_cancel=20, q_p=150 (full level size incl. us).
    # cancel_effective = 20 * 100/150 = 13.3333...
    # q_after_cancel = 100 - 13.3333... = 86.6666...
    # consumed_from_queue = min(86.6666, 30) = 30 (trade doesn't clear the queue)
    # new_q_ahead = 86.6666... - 30 = 56.6666...
    # leftover_trade = 30 - 30 = 0 -> no fill
    state = QueueState(q_ahead=100.0, own_qty_remaining=10.0)
    new_state = update_queue(state, v_trade=30.0, v_cancel=20.0, q_p_before=150.0)
    assert new_state.q_ahead == pytest.approx(56.666666666666664)
    assert new_state.filled_qty == 0.0
    assert new_state.own_qty_remaining == 10.0


def test_cancel_only_proportional_weighting():
    # Q_ahead=50, q_p=200, v_trade=0, v_cancel=80.
    # cancel_effective = 80 * 50/200 = 20
    # new_q_ahead = max(0, 50 - 0 - 20) = 30, no fill (no trade volume at all)
    state = QueueState(q_ahead=50.0, own_qty_remaining=5.0)
    new_state = update_queue(state, v_trade=0.0, v_cancel=80.0, q_p_before=200.0)
    assert new_state.q_ahead == pytest.approx(30.0)
    assert new_state.filled_qty == 0.0


def test_queue_exhaustion_full_fill_in_one_tick():
    # Q_ahead=20, own_qty=10, v_trade=35, v_cancel=0.
    # q_after_cancel = 20, consumed_from_queue = min(20,35) = 20, new_q_ahead = 0
    # leftover_trade = 35 - 20 = 15 -> fill_qty = min(15, 10) = 10 (fully filled)
    state = QueueState(q_ahead=20.0, own_qty_remaining=10.0)
    new_state = update_queue(state, v_trade=35.0, v_cancel=0.0, q_p_before=100.0)
    assert new_state.q_ahead == 0.0
    assert new_state.filled_qty == pytest.approx(10.0)
    assert new_state.own_qty_remaining == 0.0
    assert new_state.is_resolved


def test_queue_exhaustion_partial_fill():
    # Q_ahead=20, own_qty=30, v_trade=25, v_cancel=0.
    # consumed_from_queue = min(20,25) = 20, new_q_ahead = 0
    # leftover_trade = 25 - 20 = 5 -> fill_qty = min(5, 30) = 5 (still resting)
    state = QueueState(q_ahead=20.0, own_qty_remaining=30.0)
    new_state = update_queue(state, v_trade=25.0, v_cancel=0.0, q_p_before=100.0)
    assert new_state.q_ahead == 0.0
    assert new_state.filled_qty == pytest.approx(5.0)
    assert new_state.own_qty_remaining == pytest.approx(25.0)
    assert not new_state.is_resolved


def test_front_of_queue_immediate_fill():
    # Already at front (q_ahead=0), own_qty=10, v_trade=15, v_cancel=0.
    # consumed_from_queue = min(0,15) = 0, leftover_trade = 15
    # fill_qty = min(15, 10) = 10 (fully filled, 5 units of trade "wasted" past us)
    state = QueueState(q_ahead=0.0, own_qty_remaining=10.0)
    new_state = update_queue(state, v_trade=15.0, v_cancel=0.0, q_p_before=10.0)
    assert new_state.q_ahead == 0.0
    assert new_state.filled_qty == pytest.approx(10.0)
    assert new_state.is_resolved


def test_already_fully_filled_is_a_noop():
    state = QueueState(q_ahead=0.0, own_qty_remaining=0.0, filled_qty=10.0)
    new_state = update_queue(state, v_trade=100.0, v_cancel=100.0, q_p_before=50.0)
    assert new_state == state


def test_zero_q_p_before_does_not_divide_by_zero():
    # q_p_before=0 is a degenerate/empty-level tick; cancel_effective must be 0,
    # not raise ZeroDivisionError.
    state = QueueState(q_ahead=5.0, own_qty_remaining=5.0)
    new_state = update_queue(state, v_trade=0.0, v_cancel=10.0, q_p_before=0.0)
    assert new_state.q_ahead == 5.0
    assert new_state.filled_qty == 0.0


def test_sequential_decomposition_matches_spec_combined_formula():
    # Direct algebraic check that the cancel-then-trade sequential
    # decomposition used by update_queue produces the SAME q_ahead(t+1) as
    # the spec's literal single formula
    #   max(0, Q_ahead(t) - V_trade(t) - V_cancel(t)*Q_ahead(t)/q_p(t))
    # across a range of hand-picked (q_ahead, v_trade, v_cancel, q_p) tuples,
    # including an over-depletion case.
    cases = [
        (100.0, 30.0, 20.0, 150.0),
        (50.0, 0.0, 80.0, 200.0),
        (20.0, 35.0, 0.0, 100.0),
        (10.0, 3.0, 4.0, 40.0),   # under-depletion, both terms partial
        (10.0, 50.0, 50.0, 40.0),  # gross over-depletion
    ]
    for q_ahead, v_trade, v_cancel, q_p in cases:
        expected = max(0.0, q_ahead - v_trade - v_cancel * (q_ahead / q_p))
        state = QueueState(q_ahead=q_ahead, own_qty_remaining=1000.0)
        got = update_queue(state, v_trade=v_trade, v_cancel=v_cancel, q_p_before=q_p)
        assert got.q_ahead == pytest.approx(expected), (q_ahead, v_trade, v_cancel, q_p)


def test_expected_wait_time_hand_computed():
    # Q_ahead=100, avg_trade_rate=25 -> T_hat = 4.0
    assert expected_wait_time(100.0, 25.0) == pytest.approx(4.0)


def test_expected_wait_time_zero_rate_is_infinite():
    assert expected_wait_time(100.0, 0.0) == math.inf
    assert expected_wait_time(100.0, -5.0) == math.inf


def test_queue_position_ratio_hand_computed():
    # Q_ahead=30, own_qty=10 -> 30 / (30+10) = 0.75
    state = QueueState(q_ahead=30.0, own_qty_remaining=10.0)
    assert queue_position_ratio(state) == pytest.approx(0.75)


def test_queue_position_ratio_no_resting_order_is_negative_one():
    state = QueueState(q_ahead=0.0, own_qty_remaining=0.0)
    assert queue_position_ratio(state) == -1.0


def test_walk_market_fill_multi_level_partial():
    # Book (best-to-worst): (100.0, 5.0), (100.1, 3.0), (100.2, 10.0), (100.3, 4.0).
    # qty=10.5 fully clears level 1 (5.0) and level 2 (3.0), leaving 2.5 which
    # partially clears level 3 (10.0 available there); level 4 is never touched.
    # Blended avg price = (100.0*5.0 + 100.1*3.0 + 100.2*2.5) / 10.5
    #                    = (500.0 + 300.3 + 250.5) / 10.5 = 1050.8 / 10.5 = 100.07619047619048
    prices = [100.0, 100.1, 100.2, 100.3]
    sizes = [5.0, 3.0, 10.0, 4.0]
    fills, qty_unfilled = walk_market_fill(10.5, prices, sizes)
    assert fills == [
        pytest.approx((100.0, 5.0)),
        pytest.approx((100.1, 3.0)),
        pytest.approx((100.2, 2.5)),
    ]
    assert qty_unfilled == pytest.approx(0.0)
    total_qty = sum(q for _, q in fills)
    blended_avg = sum(p * q for p, q in fills) / total_qty
    assert total_qty == pytest.approx(10.5)
    assert blended_avg == pytest.approx(100.07619047619048)


def test_walk_market_fill_exceeds_all_visible_levels():
    # Only 3.0 total visible (2.0 + 1.0) against a 5.0 request -- the visible
    # depth fills exactly, and the un-invented remainder (2.0) comes back as
    # qty_unfilled rather than being filled at a synthetic price.
    prices = [100.0, 100.1]
    sizes = [2.0, 1.0]
    fills, qty_unfilled = walk_market_fill(5.0, prices, sizes)
    assert fills == [pytest.approx((100.0, 2.0)), pytest.approx((100.1, 1.0))]
    assert qty_unfilled == pytest.approx(2.0)


def test_walk_market_fill_small_order_touches_only_touch_level():
    # A request smaller than level 1's size should behave exactly like the
    # old single-price fill: one fill entry, level 2 untouched.
    prices = [100.0, 100.1]
    sizes = [5.0, 3.0]
    fills, qty_unfilled = walk_market_fill(2.0, prices, sizes)
    assert fills == [pytest.approx((100.0, 2.0))]
    assert qty_unfilled == pytest.approx(0.0)


def test_walk_market_fill_empty_book_returns_all_unfilled():
    fills, qty_unfilled = walk_market_fill(5.0, [], [])
    assert fills == []
    assert qty_unfilled == pytest.approx(5.0)
