"""Tests for src/envs/l2_reward.py and FrozenL3Wrapper's l2_reward_mode="potential_is_shaping"
path (docs/reports/l2_reward_redesign_proposal.md). Same fixture discipline as
tests/test_reward.py (hand-computed expected values) for the pure-function tests; the
telescoping test is the hard gate this design's whole correctness claim rests on, and per
instruction runs on REAL market data with the REAL frozen L3 checkpoint, not a synthetic
fixture -- gated (skipped, not failed) if either isn't present on this box, same pattern
tests/test_wrappers.py already uses for its own real-checkpoint integration smoke test.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pytest
from sb3_contrib import RecurrentPPO

from src.envs.l2_reward import l2_potential, l2_window_reward
from src.envs.wrappers import FrozenL3Wrapper
from src.train.train_l2 import make_l2_wrapped_env

# --------------------------------------------------------------------------------------
# Pure-function tests: hand-computed, no env/model needed.
# --------------------------------------------------------------------------------------

def test_l2_potential_is_zero_with_no_fills_and_mid_equal_to_arrival():
    # fill_ratio=0, opportunity term's (mid-arrival) is a literal same-value subtraction.
    phi = l2_potential(
        side=1, episode_fills=[], qty_total=10.0, arrival_price=100.0,
        current_mid_price=100.0, fee_bps_per_fill=1.0, kappa=1.0,
    )
    assert phi == pytest.approx(0.0, abs=1e-12)


def test_l2_potential_matches_hand_computed_implementation_shortfall():
    # side=1 (buy), one fill at 101 for qty 4 out of qty_total=10, arrival=100,
    # current (mark) mid=102, fee_bps_per_fill=1.0.
    # fill_ratio=0.4, p_avg=101, is_exec_bps = 1*(101-100)/100*1e4 = 100
    # exec_contribution = 0.4*100 = 40
    # is_opp_bps = 0.6*1*(102-100)/100*1e4 = 0.6*200 = 120
    # fees_bps = 1.0*0.4 = 0.4
    # is_total_bps = 40+120+0.4 = 160.4 -> Phi = -kappa*160.4 = -160.4 (kappa=1.0)
    phi = l2_potential(
        side=1, episode_fills=[{"price": 101.0, "qty": 4.0}], qty_total=10.0,
        arrival_price=100.0, current_mid_price=102.0, fee_bps_per_fill=1.0, kappa=1.0,
    )
    assert phi == pytest.approx(-160.4, rel=1e-9)


def test_l2_potential_scales_with_kappa():
    phi_k1 = l2_potential(
        side=1, episode_fills=[{"price": 101.0, "qty": 4.0}], qty_total=10.0,
        arrival_price=100.0, current_mid_price=102.0, fee_bps_per_fill=1.0, kappa=1.0,
    )
    phi_k2 = l2_potential(
        side=1, episode_fills=[{"price": 101.0, "qty": 4.0}], qty_total=10.0,
        arrival_price=100.0, current_mid_price=102.0, fee_bps_per_fill=1.0, kappa=2.0,
    )
    assert phi_k2 == pytest.approx(2.0 * phi_k1, rel=1e-9)


def test_l2_window_reward_is_exactly_phi_delta():
    reward, new_phi = l2_window_reward(
        prev_phi=-50.0, side=1, episode_fills=[{"price": 101.0, "qty": 4.0}],
        qty_total=10.0, arrival_price=100.0, current_mid_price=102.0,
        fee_bps_per_fill=1.0, kappa=1.0,
    )
    assert new_phi == pytest.approx(-160.4, rel=1e-9)
    assert reward == pytest.approx(-160.4 - (-50.0), rel=1e-9)


def test_l2_window_reward_zero_delta_when_nothing_changes():
    # Same fills, same mid -> Phi(t) == Phi(t-1) -> reward == 0 exactly.
    prev_phi = l2_potential(
        side=-1, episode_fills=[{"price": 99.0, "qty": 2.0}], qty_total=5.0,
        arrival_price=100.0, current_mid_price=99.5, fee_bps_per_fill=0.5, kappa=1.0,
    )
    reward, new_phi = l2_window_reward(
        prev_phi=prev_phi, side=-1, episode_fills=[{"price": 99.0, "qty": 2.0}],
        qty_total=5.0, arrival_price=100.0, current_mid_price=99.5,
        fee_bps_per_fill=0.5, kappa=1.0,
    )
    assert reward == pytest.approx(0.0, abs=1e-12)
    assert new_phi == pytest.approx(prev_phi, rel=1e-12)


# --------------------------------------------------------------------------------------
# Telescoping test -- THE hard gate. Real market data, real frozen L3 checkpoint. Gated
# (skipped) if either isn't present on this box, not failed -- same pattern as
# tests/test_wrappers.py's own real-checkpoint integration smoke test.
# --------------------------------------------------------------------------------------

def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def _real_l3_paths():
    root = _repo_root()
    return (
        root / "models" / "l3_frozen_backup" / "l3_executioner_v1_frozen.zip",
        root / "models" / "l3_frozen_backup" / "l3_vecnormalize_frozen.pkl",
    )


def _real_numeric_data_present() -> bool:
    return (_repo_root() / "data" / "raw_l2_bybit_numeric" / "BTCUSDT").is_dir()


_l3_ckpt, _l3_vecnorm = _real_l3_paths()


@pytest.mark.skipif(not _l3_ckpt.exists(), reason="real frozen L3 checkpoint not present on this box")
@pytest.mark.skipif(not _real_numeric_data_present(), reason="real numeric-format data not present on this box")
def test_potential_is_shaping_telescopes_exactly_on_real_episodes():
    from src.data.split import load_split

    val_dates = load_split("val")
    val_date_range = (val_dates[0].isoformat(), val_dates[-1].isoformat())
    l3_model = RecurrentPPO.load(str(_l3_ckpt), device="cpu")

    kappa = 1.0  # RewardWeights() default, confirmed unmodified by make_l2_wrapped_env
    neutral_action = np.array([1.0, 0.5], dtype=np.float32)  # TWAP-passthrough; no SAC model needed

    # Multiple episodes, multiple seeds, to catch anything a single lucky/unlucky
    # episode wouldn't -- partial fills, zero-fill episodes, mid-episode truncation.
    for seed in (5_000_000, 5_000_001, 5_000_002, 5_000_010, 5_000_050):
        wrapper = make_l2_wrapped_env(
            val_date_range, horizon_ticks=3000, lookback_ticks=10, l3_model=l3_model,
            l3_vecnormalize_path=str(_l3_vecnorm), ticks_per_l2_decision=50,
            l2_include_prev_action=False, data_dir="data/raw_l2_bybit_numeric/BTCUSDT",
            l3_deterministic=True, use_numeric_format=True,
            l2_reward_mode="potential_is_shaping",
        )

        obs, info = wrapper.reset(seed=seed)
        total_shaped_reward = 0.0
        max_decisions = 3000 // 50 + 1
        info_final = info
        for _ in range(max_decisions):
            obs, r, term, trunc, info = wrapper.step(neutral_action)
            assert np.isfinite(r), f"seed={seed}: non-finite shaped reward {r}"
            total_shaped_reward += r
            info_final = info
            if term or trunc:
                break

        terminal_is = info_final["implementation_shortfall"].is_total_bps
        expected_total = -kappa * terminal_is

        assert total_shaped_reward == pytest.approx(expected_total, abs=1e-6), (
            f"seed={seed}: telescoping FAILED -- summed shaped rewards "
            f"({total_shaped_reward}) != -kappa*terminal_is ({expected_total}). "
            "This is the hard gate: if this fails, the shaping is not actually "
            "potential-based and the optimal-policy-invariance guarantee does not hold."
        )


@pytest.mark.skipif(not _l3_ckpt.exists(), reason="real frozen L3 checkpoint not present on this box")
@pytest.mark.skipif(not _real_numeric_data_present(), reason="real numeric-format data not present on this box")
def test_potential_is_shaping_scale_is_sane_relative_to_l3_passthrough():
    """Not a pass/fail correctness gate -- a printed sanity check (per-window reward
    mean/std/min/max) comparing the new shaped reward's scale against the OLD
    l3_passthrough aggregation on the SAME episodes, so a human can see the two are in
    the same rough order of magnitude rather than trusting an assertion alone. Run with
    -s to see the printed comparison; the only hard assertion is finiteness and that the
    new scale isn't absurdly (>100x) larger, which would risk destabilizing VecNormalize's
    running reward-variance estimate."""
    from src.data.split import load_split

    val_dates = load_split("val")
    val_date_range = (val_dates[0].isoformat(), val_dates[-1].isoformat())
    l3_model = RecurrentPPO.load(str(_l3_ckpt), device="cpu")
    neutral_action = np.array([1.0, 0.5], dtype=np.float32)
    seeds = list(range(5_000_000, 5_000_020))

    def _run(mode: str) -> list[float]:
        wrapped_env = make_l2_wrapped_env(
            val_date_range, horizon_ticks=3000, lookback_ticks=10, l3_model=l3_model,
            l3_vecnormalize_path=str(_l3_vecnorm), ticks_per_l2_decision=50,
            l2_include_prev_action=False, data_dir="data/raw_l2_bybit_numeric/BTCUSDT",
            l3_deterministic=True, use_numeric_format=True, l2_reward_mode=mode,
        )
        rewards = []
        for seed in seeds:
            wrapped_env.reset(seed=seed)
            for _ in range(3000 // 50 + 1):
                _, r, term, trunc, _ = wrapped_env.step(neutral_action)
                rewards.append(r)
                if term or trunc:
                    break
        return rewards

    old_rewards = np.array(_run("l3_passthrough"))
    new_rewards = np.array(_run("potential_is_shaping"))

    assert np.isfinite(old_rewards).all()
    assert np.isfinite(new_rewards).all()

    old_scale = np.abs(old_rewards).mean()
    new_scale = np.abs(new_rewards).mean()
    print(f"\nl3_passthrough : mean={old_rewards.mean():.4f} std={old_rewards.std():.4f} "
          f"min={old_rewards.min():.4f} max={old_rewards.max():.4f} mean|r|={old_scale:.4f}")
    print(f"potential_shaping: mean={new_rewards.mean():.4f} std={new_rewards.std():.4f} "
          f"min={new_rewards.min():.4f} max={new_rewards.max():.4f} mean|r|={new_scale:.4f}")

    assert new_scale < 100 * max(old_scale, 1e-9), (
        f"New reward scale ({new_scale:.4f}) is more than 100x the old scale "
        f"({old_scale:.4f}) -- likely to destabilize VecNormalize's running variance."
    )
