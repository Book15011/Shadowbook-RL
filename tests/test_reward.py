"""Hand-computed fixtures for reward.py."""
import pytest

from src.envs.reward import RewardWeights, compute_implementation_shortfall, step_reward


def test_step_reward_slip_and_spread_maker_fill():
    # side=1, arrival=100.0, fill=100.5 qty=1.0/10.0, mid=100.0, is_maker=True, dt=0
    # r_slip = -1.0*1*(100.5-100.0)/100.0*(1/10)*1e4 = -5.0
    # r_spread = 0.8*1*(100.0-100.5)/100.0*(1/10)*1e4 = -4.0
    # r_inv = 0 (dt=0), r_queue = 0 (no cancel) -> total = -9.0
    w = RewardWeights()
    r = step_reward(
        w, side=1, fills=[{"price": 100.5, "qty": 1.0, "is_maker": True}],
        arrival_price=100.0, mid_price=100.0, qty_remaining=9.0, qty_total=10.0,
        dt=0.0, l1_risk_score=0.0, canceled_unfilled=False,
        queue_ahead_at_cancel=None, queue_at_level=None,
    )
    assert r == pytest.approx(-9.0)


def test_step_reward_inventory_holding_isolated():
    # no fills, qty_remaining=6, qty_total=10, dt=2.0, l1_risk=0.5
    # r_inv = -0.02*(1+0.5)*(0.6)^2*2.0 = -0.0216
    w = RewardWeights()
    r = step_reward(
        w, side=1, fills=[], arrival_price=100.0, mid_price=100.0,
        qty_remaining=6.0, qty_total=10.0, dt=2.0, l1_risk_score=0.5,
        canceled_unfilled=False, queue_ahead_at_cancel=None, queue_at_level=None,
    )
    assert r == pytest.approx(-0.0216)


def test_step_reward_cancel_penalty_isolated():
    # canceled_unfilled, queue_ahead=30, queue_at_level=100
    # r_queue = -0.5 - 0.3*(30/100) = -0.59
    w = RewardWeights()
    r = step_reward(
        w, side=1, fills=[], arrival_price=100.0, mid_price=100.0,
        qty_remaining=10.0, qty_total=10.0, dt=0.0, l1_risk_score=0.0,
        canceled_unfilled=True, queue_ahead_at_cancel=30.0, queue_at_level=100.0,
    )
    assert r == pytest.approx(-0.59)


def test_step_reward_cancel_with_zero_queue_at_level_skips_gamma_term():
    # queue_at_level=0 is falsy -> gamma term skipped (matches spec reference code's
    # `if queue_ahead_at_cancel is not None and queue_at_level:` truthiness check,
    # which also avoids a division by zero here).
    w = RewardWeights()
    r = step_reward(
        w, side=1, fills=[], arrival_price=100.0, mid_price=100.0,
        qty_remaining=10.0, qty_total=10.0, dt=0.0, l1_risk_score=0.0,
        canceled_unfilled=True, queue_ahead_at_cancel=5.0, queue_at_level=0.0,
    )
    assert r == pytest.approx(-0.5)


def test_step_reward_staleness_isolated():
    # resting=True, ticks_since_own_fill_norm=0.5, default zeta=0.06
    # r_stale = -0.06*0.5 = -0.03; no fills/cancel/dt -> everything else is 0
    w = RewardWeights()
    r = step_reward(
        w, side=1, fills=[], arrival_price=100.0, mid_price=100.0,
        qty_remaining=10.0, qty_total=10.0, dt=0.0, l1_risk_score=0.0,
        canceled_unfilled=False, queue_ahead_at_cancel=None, queue_at_level=None,
        resting=True, ticks_since_own_fill_norm=0.5,
    )
    assert r == pytest.approx(-0.03)


def test_step_reward_staleness_zero_when_not_resting():
    # resting=False -> r_stale must be exactly 0 regardless of
    # ticks_since_own_fill_norm (it is only meaningful while an order is
    # actually resting and unfilled).
    w = RewardWeights()
    r = step_reward(
        w, side=1, fills=[], arrival_price=100.0, mid_price=100.0,
        qty_remaining=10.0, qty_total=10.0, dt=0.0, l1_risk_score=0.0,
        canceled_unfilled=False, queue_ahead_at_cancel=None, queue_at_level=None,
        resting=False, ticks_since_own_fill_norm=1.0,
    )
    assert r == pytest.approx(0.0)


def test_implementation_shortfall_with_fills_hand_computed():
    # side=1, fills=[(101,3),(102,2)], qty_total=10, arrival=100, terminal_mid=103, fee=2bps
    # p_avg = (101*3+102*2)/5 = 101.4, fill_ratio=0.5
    # is_exec_bps = (101.4-100)/100*1e4 = 140.0 -> exec_contribution = 0.5*140 = 70.0
    # is_opp_bps = 0.5*(103-100)/100*1e4 = 150.0
    # fees_bps = 2.0*0.5 = 1.0
    # is_total_bps = 70 + 150 + 1 = 221.0
    result = compute_implementation_shortfall(
        side=1, fills=[{"price": 101.0, "qty": 3.0}, {"price": 102.0, "qty": 2.0}],
        qty_total=10.0, arrival_price=100.0, terminal_mid_price=103.0, fee_bps_per_fill=2.0,
    )
    assert result.fill_ratio == pytest.approx(0.5)
    assert result.p_avg == pytest.approx(101.4)
    assert result.is_exec_bps == pytest.approx(140.0)
    assert result.is_opp_bps == pytest.approx(150.0)
    assert result.fees_bps == pytest.approx(1.0)
    assert result.is_total_bps == pytest.approx(221.0)


def test_implementation_shortfall_no_fills_is_exactly_opportunity_cost():
    # side=1, zero fills, qty_total=10, arrival=100, terminal_mid=105, fee=2bps (irrelevant, no fills)
    # fill_ratio=0 -> is_exec_bps must be None (undefined), exec_contribution must be
    # exactly 0.0 (not 0*NaN), fees_bps must be exactly 0.0, and
    # is_total_bps must equal is_opp_bps EXACTLY.
    # is_opp_bps = 1*1*(105-100)/100*1e4 = 500.0
    result = compute_implementation_shortfall(
        side=1, fills=[], qty_total=10.0, arrival_price=100.0,
        terminal_mid_price=105.0, fee_bps_per_fill=2.0,
    )
    assert result.fill_ratio == 0.0
    assert result.p_avg is None
    assert result.is_exec_bps is None
    assert result.fees_bps == 0.0
    assert result.is_opp_bps == pytest.approx(500.0)
    assert result.is_total_bps == pytest.approx(500.0)
    assert result.is_total_bps == pytest.approx(result.is_opp_bps)


def test_implementation_shortfall_no_fills_sell_side():
    # side=-1, zero fills, arrival=100, terminal_mid=95 (price dropped -- favorable
    # for an unfilled SELL not having sold yet... wait, side=-1 order never sold and
    # price dropped, meaning the residual is marked at a WORSE price than arrival for
    # a seller -> should be a real cost (positive IS).
    # is_opp_bps = 1*(-1)*(95-100)/100*1e4 = 500.0
    result = compute_implementation_shortfall(
        side=-1, fills=[], qty_total=5.0, arrival_price=100.0, terminal_mid_price=95.0,
    )
    assert result.is_total_bps == pytest.approx(500.0)
    assert result.is_total_bps == pytest.approx(result.is_opp_bps)
