"""L2-specific reward: potential-based mark-to-market implementation-shortfall
shaping (docs/reports/l2_reward_redesign_proposal.md, Task 2's approved primary
design). This is NOT L3's reward -- src/envs/reward.py's own step_reward() is
completely untouched by this file and stays L3's own per-tick reward exactly as
before; this module exists because L2 chooses participation rate and urgency at
a ~5s cadence and controls none of the tick-level order-type/price/cancel
decisions L3's own per-tick reward components price. Measured directly, not
hypothesized (scripts/analyze_l2_reward_components.py, 100 real val episodes):
under the OLD aggregation (FrozenL3Wrapper simply summing L3's raw step_reward()
over each window), r_stale alone was 85.6% of L2's net reward and 75.4% of its
signal magnitude -- a component L2 does not control -- while terminal IS, the
metric L2 is actually evaluated on, was 6.9%/11.6%.

THE POTENTIAL FUNCTION: Phi(t) = -kappa * compute_implementation_shortfall(
fills=episode_fills_so_far, ..., terminal_mid_price=CURRENT mid_price).is_total_bps
-- "what would my reward-scale IS be if I marked the unfilled remainder to the
current price right now." Reward per L2 decision window = Phi(t) - Phi(t-1)
(potential-based shaping, Ng/Harada/Russell 1999: F(s,a,s')=gamma*Phi(s')-Phi(s);
gamma=1 here since L2's own SAC discounting is applied by the algorithm, not this
shaping term, and the two decision points are consecutive with no other reward
in between under this mode).

WHY THIS TELESCOPES EXACTLY, NOT APPROXIMATELY: compute_implementation_shortfall()
is reused here as-is -- not re-derived, not approximated -- so Phi(t) at the REAL
terminal tick is computed from the exact same inputs LOBExecutionEnv.step() itself
uses for its own terminal reward bump (self._episode_fills, self.arrival_price,
self.fee_bps_per_fill, and a terminal_mid_price that is the SAME tick's mid_price
by construction -- see FrozenL3Wrapper.step()'s own comment on this). Phi(T) is
therefore bit-identical to -kappa*terminal_is_total_bps, not merely close.
Phi(0) = 0.0 exactly: at episode start fill_ratio=0 (no fills yet) and
arrival_price IS DEFINED as the episode-start tick's own mid_price
(LOBExecutionEnv.reset(): `self.arrival_price = self._ticks[self._episode_start].mid_price`),
so the opportunity term's (mid_price(0) - arrival_price) is a literal same-value
float subtraction, exactly 0.0. Consequently sum_t(Phi(t)-Phi(t-1)) over a full
episode telescopes to EXACTLY Phi(T) - Phi(0) = -kappa*terminal_is_total_bps + 0.0
-- see tests/test_l2_reward.py's telescoping test, checked on real episodes at a
tight tolerance, not just algebraically argued here.

WHY THIS DOES NOT INHERIT THE TWAP-BASELINE REWARD'S FAILURE MODE
(docs/reports/l3_twap_baseline_reward.md): that change subtracted a SEPARATE
reference trajectory (a TWAP shadow run, computed once per episode over its own
independently-fixed window) from the real agent's terminal IS. The real agent's
own IS is only ever exposed to market drift up to ITS OWN termination tick, while
the shadow's exposure window does not move with it -- an agent that learns to
finish faster reduces its own drift exposure relative to a comparison point that
stays fixed, which is a real incentive to rush, independent of genuine execution
skill, and is consistent with what that report measured (Arm B's episodes roughly
halved in length, fill_ratio up, an occasional costly tail). Phi(t) above has NO
separate reference trajectory at all -- it is a pure function of the REAL agent's
own accumulated state (its own fills, its own qty_remaining) and the REAL current
market price, evaluated only at the real agent's own actual decision points,
whatever those turn out to be. There is no second, independently-timed window for
timing choices to become mismatched against, so this specific failure mode cannot
arise here by construction, not merely by argument.
"""
from __future__ import annotations

from src.envs.reward import compute_implementation_shortfall


def l2_potential(
    *,
    side: int,
    episode_fills: list[dict],
    qty_total: float,
    arrival_price: float,
    current_mid_price: float,
    fee_bps_per_fill: float,
    kappa: float,
) -> float:
    """Phi(t): mark-to-market running reward-scale estimate (higher = better,
    matching LOBExecutionEnv.step()'s own `r += -kappa * terminal_is_for_reward`
    sign convention for the real terminal bump). `episode_fills` MUST be the full
    cumulative fill list since episode start (LOBExecutionEnv's own
    `self._episode_fills`), not a per-window slice -- passing only the current
    window's fills would silently break both fill_ratio and the telescoping
    property described in this module's own docstring."""
    is_result = compute_implementation_shortfall(
        side=side,
        fills=episode_fills,
        qty_total=qty_total,
        arrival_price=arrival_price,
        terminal_mid_price=current_mid_price,
        fee_bps_per_fill=fee_bps_per_fill,
    )
    return -kappa * is_result.is_total_bps


def l2_window_reward(
    *,
    prev_phi: float,
    side: int,
    episode_fills: list[dict],
    qty_total: float,
    arrival_price: float,
    current_mid_price: float,
    fee_bps_per_fill: float,
    kappa: float,
) -> tuple[float, float]:
    """F(t) = Phi(t) - Phi(t-1). Returns (reward, new_phi) -- callers must thread
    new_phi back in as the next call's prev_phi (FrozenL3Wrapper stores this as
    self._l2_prev_phi, reset to 0.0 exactly at the start of every episode --
    see this module's own docstring for why 0.0 is exact, not a placeholder)."""
    phi_t = l2_potential(
        side=side,
        episode_fills=episode_fills,
        qty_total=qty_total,
        arrival_price=arrival_price,
        current_mid_price=current_mid_price,
        fee_bps_per_fill=fee_bps_per_fill,
        kappa=kappa,
    )
    return phi_t - prev_phi, phi_t
