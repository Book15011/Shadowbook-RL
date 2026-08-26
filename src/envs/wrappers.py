"""FrozenL3Wrapper (architecture_spec.md Section 4.1), reconciled against the real
LOBExecutionEnv API -- see docs/reports/phase4_l2_reconciliation_and_plan.md (Part B,
FINAL SPEC section) for the full design rationale this implements: the observation-space
index-mapping table (2a/2e), the TWAP-schedule-deviation scalar (2b), the previous-action
toggle (2c), and the participation-rate-multiplier action-space transform.

Four corrections discovered during implementation, not in the design doc as originally
written (see that doc's own implementation-note addendum for the first two):

1. schedule_deviation's TWAP baseline is NOT read via
   `env._compute_l2_target_slice_ratio()` as the design doc originally described --
   that hook silently returns whatever override is currently set (not the default
   linear-TWAP formula) on every call after the wrapper's first ever step(), since the
   wrapper keeps `l2_target_slice_ratio_override` permanently non-None from then on.
   Calling it post-window would collapse schedule_deviation to ~0 (executed-so-far minus
   itself) after the very first window -- a real, silent bug, not a style choice. Fixed
   here by recomputing the same public-info-sourced formula directly
   (`info["ticks_elapsed"] / horizon_ticks`, both already present on every step()/reset()
   info dict) instead of relying on the private, override-shadowed hook.
2. `l2_include_prev_action` defaults to False, not True as the design doc originally
   recommended -- the "recurrent policy" precedent that reasoning leaned on does not
   transfer cleanly to SAC's plain MlpPolicy, and the closer in-repo precedent
   deliberately excludes action history and flags its own findings as untested for
   off-policy methods. Kept as a toggle (a legitimate ablation), not presented as
   precedent-backed.
3. The frozen L3 policy's own inner predict() calls were hardcoded `deterministic=False`
   unconditionally, with no way for a caller to get deterministic frozen-L3 behavior --
   caught by tests/test_train_l2.py's own eval-callback determinism test (identical seed
   + identical L2 action produced the same terminal fill_ratio/IS but a DIFFERENT
   per-tick reward trajectory, because L3's own actions were still being sampled
   stochastically underneath regardless of what "deterministic" meant to the outer
   caller). This directly undermined the whole point of a paired, reproducible eval
   comparison -- an eval callback calling `predict(..., deterministic=True)` for its OWN
   L2 action still got a non-reproducible frozen L3 policy underneath it. Fixed by adding
   an `l3_deterministic` constructor parameter (default `False`, preserving prior
   training-time behavior) that `FrozenL3Wrapper.step()` now actually passes through to
   `self.l3_model.predict()`; eval call sites should construct with
   `l3_deterministic=True`.
4. `l2_target_slice_ratio_override`/`l2_urgency` were left holding whatever the PREVIOUS
   episode's last step() set them to across a reused instance (real for every episode
   after the first, in both training and eval) -- LOBExecutionEnv.reset() never touches
   either attribute, so env.reset()'s own first _build_obs() call (which is exactly what
   gets fed to the frozen L3 policy) would silently read stale, prior-episode state
   instead of a fresh default. Also caught by the same determinism test as correction 3
   (fixing correction 3 alone was not enough -- fill_ratio/IS matched but total_reward
   still didn't, because the L3 action sequence itself still differed run to run). Fixed
   by resetting both to their neutral defaults (None, 0.5) at the START of reset(),
   before calling env.reset() -- see reset()'s own comment for why this matters for both
   training (a real cross-episode leak) and eval (undermines paired-seed
   reproducibility specifically).

L2_REWARD_MODE (2026-08-27, docs/reports/l2_reward_redesign_proposal.md): a fifth,
opt-in change, same convention as correction 3's l3_deterministic -- defaults to prior
behavior, nothing changes unless deliberately selected. `l2_reward_mode="l3_passthrough"`
(default) is exactly the old behavior: `agg_reward` is the raw sum of L3's own
per-tick step_reward() across the window. `l2_reward_mode="potential_is_shaping"`
replaces that sum with src.envs.l2_reward's potential-based mark-to-market IS shaping --
see that module's own docstring for why this exists (measured: r_stale alone was 85.6%
of L2's reward under l3_passthrough, a component L2 does not control) and for the exact
telescoping guarantee (Phi(t)-Phi(t-1), summing to EXACTLY -kappa*terminal_is_total_bps
over a full episode, not approximately).
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.envs.l2_reward import l2_window_reward
from src.envs.lob_execution_env import LOBExecutionEnv, _OBS_SPEC

_L2_REWARD_MODES = frozenset(("l3_passthrough", "potential_is_shaping"))

# --- Index-mapping, per docs/reports/phase4_l2_reconciliation_and_plan.md FINAL SPEC 2a/2e. ---
# Old _OBS_SPEC indices 15/16 (l2_target_slice_ratio, l2_urgency) are excluded -- L2
# produces them, doesn't consume them (architecture_spec.md Section 3.1).
_EXCLUDED_OLD_IDX = frozenset((15, 16))
# Group 3 (idx 2, 6, 7, 8, 9) + Group 5 (idx 19-38, book depth): mean over the
# ticks_per_l2_decision window. Every other non-excluded index (Groups 1/2/4/6 in the
# design doc: already-windowed z-scores, point-in-time state, structurally-zero
# cancel-ratio, slow external context) is behaviorally identical here -- take the
# freshest (last) value in the window -- even though the design doc justifies them as
# four conceptually distinct groups.
_MEAN_OLD_IDX = frozenset((2, 6, 7, 8, 9)) | frozenset(range(19, 39))
_L2_BASE_OLD_IDX_ORDER = tuple(i for i in range(42) if i not in _EXCLUDED_OLD_IDX)
assert len(_L2_BASE_OLD_IDX_ORDER) == 40

L2_BASE_OBS_DIM = 41  # 40 downsampled features (2a) + schedule_deviation (2b)
L2_FULL_OBS_DIM = 43  # + prev_participation_rate_mult, prev_urgency (2c, toggle)

# architecture_spec.md Section 3.2's L2 action space exactly: dim 0 participation_rate_
# multiplier (0 = defer/hide entirely, 1 = exactly on-schedule, up to 2 = catch-up burst),
# dim 1 urgency (passed straight through to L3's observation idx 16).
L2_ACTION_LOW = np.array([0.0, 0.0], dtype=np.float32)
L2_ACTION_HIGH = np.array([2.0, 1.0], dtype=np.float32)

_INITIAL_PREV_ACTION = np.array([1.0, 0.5], dtype=np.float32)  # on-schedule, neutral urgency
# -- matches the base env's own l2_urgency default (0.5); used only as the placeholder
# "previous action" for the very first decision of an episode, before any real L2 action
# has been taken.

_OBS_SPEC_RANGE = {i: (lo, hi) for i, _, (lo, hi) in _OBS_SPEC}


def _twap_schedule_baseline(info: dict[str, Any], horizon_ticks: int) -> float:
    """The env's own default linear-TWAP fraction (ticks_elapsed / horizon_ticks),
    sourced from the env's public info dict (info["ticks_elapsed"], present on every
    step()/reset() return) and the public horizon_ticks constructor attribute.

    Deliberately NOT env._compute_l2_target_slice_ratio() -- see module docstring point 1.
    """
    if horizon_ticks <= 0:
        return 0.0
    return float(np.clip(info["ticks_elapsed"] / horizon_ticks, 0.0, 1.0))


def _schedule_deviation(info: dict[str, Any], horizon_ticks: int) -> float:
    """Executed-so-far minus TWAP-scheduled-so-far (design doc 2b). Positive = ahead of
    schedule, negative = behind. Both qty_total/qty_remaining are on every info dict."""
    qty_total, qty_remaining = info["qty_total"], info["qty_remaining"]
    executed_so_far = (qty_total - qty_remaining) / qty_total if qty_total > 0 else 0.0
    twap_baseline = _twap_schedule_baseline(info, horizon_ticks)
    return float(np.clip(executed_so_far - twap_baseline, -1.0, 1.0))


def _downsample_window(
    window_obs: np.ndarray,
    info: dict[str, Any],
    horizon_ticks: int,
    prev_action: np.ndarray | None,
) -> np.ndarray:
    """Builds the L2-cadence observation from one window of raw 42-dim L3 observations
    (shape (n_ticks, 42), one row per tick actually stepped this window -- may be shorter
    than ticks_per_l2_decision if the episode terminated mid-window), per design doc
    Steps 2a-2e. `info` is the info dict from the LAST env.step()/reset() call in the
    window (defines "at decision time")."""
    last = window_obs[-1]
    mean = window_obs.mean(axis=0)
    features = [float(mean[i] if i in _MEAN_OLD_IDX else last[i]) for i in _L2_BASE_OLD_IDX_ORDER]
    features.append(_schedule_deviation(info, horizon_ticks))
    if prev_action is not None:
        features.append(float(np.clip(prev_action[0], L2_ACTION_LOW[0], L2_ACTION_HIGH[0])))
        features.append(float(np.clip(prev_action[1], L2_ACTION_LOW[1], L2_ACTION_HIGH[1])))
    return np.array(features, dtype=np.float32)


def _l2_observation_space(include_prev_action: bool) -> gym.spaces.Box:
    low = [_OBS_SPEC_RANGE[i][0] for i in _L2_BASE_OLD_IDX_ORDER] + [-1.0]
    high = [_OBS_SPEC_RANGE[i][1] for i in _L2_BASE_OLD_IDX_ORDER] + [1.0]
    if include_prev_action:
        low += [float(L2_ACTION_LOW[0]), float(L2_ACTION_LOW[1])]
        high += [float(L2_ACTION_HIGH[0]), float(L2_ACTION_HIGH[1])]
    return gym.spaces.Box(low=np.array(low, dtype=np.float32), high=np.array(high, dtype=np.float32), dtype=np.float32)


class FrozenL3Wrapper(gym.Wrapper):
    """Wraps a single-tier LOBExecutionEnv so that env.step(l2_action) internally rolls
    a frozen L3 RecurrentPPO policy forward for ticks_per_l2_decision ticks and returns
    the L2-cadence aggregate -- architecture_spec.md Section 4.1's FrozenL3Wrapper,
    rebuilt against the real env API (see docs/reports/phase4_l2_reconciliation_and_plan.md
    Part A: the reference version's tier=/l2_action_space/apply_l2_action/step_l3/
    get_l3_obs/get_l2_obs/l2_info() do not exist on the real, single-tier class -- this
    wrapper IS the entire L2/L3 integration layer, not a thin adapter over env-native
    support for it).
    """

    def __init__(
        self,
        env: LOBExecutionEnv,
        l3_model: RecurrentPPO,
        l3_vecnormalize_path: str,
        ticks_per_l2_decision: int = 50,
        l2_include_prev_action: bool = False,
        l3_deterministic: bool = False,
        l2_reward_mode: str = "l3_passthrough",
    ) -> None:
        super().__init__(env)
        self.l3_model = l3_model
        self.n_ticks = ticks_per_l2_decision
        self.l2_include_prev_action = l2_include_prev_action
        if l2_reward_mode not in _L2_REWARD_MODES:
            raise ValueError(f"l2_reward_mode must be one of {sorted(_L2_REWARD_MODES)}, got {l2_reward_mode!r}")
        self.l2_reward_mode = l2_reward_mode
        # Phi(t-1) for potential_is_shaping -- see l2_reward.py's module docstring for why
        # 0.0 is the EXACT correct initialization (not an approximation): fill_ratio=0 and
        # arrival_price IS the episode-start tick's own mid_price by construction
        # (LOBExecutionEnv.reset()), so Phi(0)'s opportunity term is a literal same-value
        # subtraction, exactly 0.0. Re-zeroed in reset(), see there.
        self._l2_prev_phi: float = 0.0
        # Controls the FROZEN L3 policy's own inner predict() calls -- separate from
        # whatever determinism the caller (e.g. SAC training vs. an eval callback) uses
        # for its OWN action selection. False (stochastic sampling) by default, matching
        # L3's own training-time convention and this wrapper's original behavior.
        # Real bug this parameter fixes, caught by tests/test_train_l2.py's own
        # determinism test (identical seed + identical L2 action still produced a
        # different per-tick reward trajectory, though the same terminal fill_ratio/IS,
        # because the frozen L3 policy was ALWAYS sampled stochastically inside step()
        # regardless of what "deterministic" meant to the caller): an eval callback
        # driving L2 with deterministic=True still got a non-reproducible frozen L3
        # policy underneath it, undermining the whole point of a paired, repeatable eval
        # comparison. Eval call sites should pass l3_deterministic=True.
        self.l3_deterministic = l3_deterministic

        # Load the frozen checkpoint's saved observation-normalization stats -- the
        # checkpoint was trained under VecNormalize(norm_obs=True, clip_obs=5.0)
        # (confirmed from train_l3.py), so raw env observations must be normalized with
        # these SAME stats before predict(), or the frozen policy sees out-of-
        # distribution inputs (design doc Part B.3). DummyVecEnv.__init__ only reads
        # env.observation_space/action_space/metadata -- confirmed against the installed
        # SB3 2.3.2 source -- it does not reset or step the env, so wrapping the same
        # live `env` instance here has no side effects on it. VecNormalize.load() only
        # unpickles and wires up the venv reference (also confirmed against source) --
        # normalize_obs() is a pure function of the loaded stats and never mutates them.
        dummy_venv = DummyVecEnv([lambda: env])
        self._l3_obs_normalizer = VecNormalize.load(l3_vecnormalize_path, dummy_venv)
        self._l3_obs_normalizer.training = False  # inference only, never update running stats
        self._l3_obs_normalizer.norm_reward = False  # irrelevant: we aggregate raw per-tick reward

        self.action_space = gym.spaces.Box(low=L2_ACTION_LOW, high=L2_ACTION_HIGH, dtype=np.float32)
        self.observation_space = _l2_observation_space(l2_include_prev_action)

        self._l3_obs: np.ndarray | None = None
        self._last_info: dict[str, Any] | None = None
        self._l3_lstm_state: tuple[np.ndarray, ...] | None = None
        self._l3_episode_start: bool = True

    def reset(self, **kwargs):
        # Reset the env's own l2_target_slice_ratio_override/l2_urgency BEFORE calling
        # env.reset() -- LOBExecutionEnv.reset() does not touch either attribute (only
        # __init__ and the wrapper's own step() ever set them), so on a reused instance
        # (every real episode after the first, in both training and eval) they would
        # otherwise still hold whatever the PREVIOUS episode's last step() left them at.
        # Since env.reset()'s own _build_obs() call reads them immediately (idx 15/16 of
        # the raw obs, which IS what gets fed to the frozen L3 policy), that leftover
        # state would leak into the very first tick of the new episode -- silently, not
        # an error, but breaking two real things: (1) each episode no longer starts from
        # the neutral state a fresh instance would give, a real cross-episode data leak
        # during training; (2) ValISEvalCallback's reused eval env would make episode N's
        # first observation depend on whatever L2 action ended episode N-1, which differs
        # firing-to-firing as L2 trains -- undermining the paired-seed reproducibility the
        # whole eval design depends on. Caught by tests/test_train_l2.py's determinism
        # test (identical seed + identical action still gave a different total_reward,
        # despite identical fill_ratio/IS -- the L3 action sequence itself differed
        # because its first observation differed).
        self.env.l2_target_slice_ratio_override = None
        self.env.l2_urgency = 0.5
        # Phi(t-1) reset to 0.0 for potential_is_shaping -- see __init__'s own comment
        # and l2_reward.py's module docstring for why this is exact, not approximate.
        # Cross-episode leak risk here is the same class of bug corrections 3/4 above
        # already fixed for l2_target_slice_ratio_override/l2_urgency -- reset unconditionally
        # regardless of which l2_reward_mode is active (cheap, and keeps this state never
        # silently stale if the mode is switched on a reused instance).
        self._l2_prev_phi = 0.0
        obs, info = self.env.reset(**kwargs)
        self._l3_obs = obs
        self._last_info = info
        # New episode -> L3's LSTM must start from a blank hidden state on its very next
        # predict() call (Step 3b). Threaded explicitly rather than left to whatever
        # predict()'s own defaults would do -- see step() for why getting this wrong is
        # silent, not an error.
        self._l3_lstm_state = None
        self._l3_episode_start = True
        l2_obs = _downsample_window(
            obs[np.newaxis, :], info, self.env.horizon_ticks,
            _INITIAL_PREV_ACTION if self.l2_include_prev_action else None,
        )
        return l2_obs, info

    def step(self, l2_action):
        l2_action = np.asarray(l2_action, dtype=np.float32)
        participation_mult = float(l2_action[0])
        urgency = float(l2_action[1])

        # Action-space transform (design doc Step 1): participation_rate_multiplier
        # scales the env's own default linear-TWAP baseline -- NOT a 1:1 Box(0,1) map onto
        # l2_target_slice_ratio_override. Matches architecture_spec.md Section 3.2's own
        # semantics (0 = defer, 1 = on-schedule, up to 2 = catch-up burst) and the same
        # multiplier-relative-to-a-baseline pattern L1's real urgency_multiplier already
        # uses (src/agents/l1_macro_analyst.py: Field(ge=0.5, le=2.0), "a direct, bounded
        # multiplier applied to L2's participation-rate target" per architecture_spec.md).
        # A direct Box(0,1) map would make SAC output an absolute target fraction from
        # scratch, discarding the schedule's own built-in monotonic structure for free.
        # urgency, by contrast, genuinely is a 1:1 map -- its action-space range ([0,1])
        # already matches env.l2_urgency's own range exactly, no transform needed.
        twap_baseline = _twap_schedule_baseline(self._last_info, self.env.horizon_ticks)
        self.env.l2_target_slice_ratio_override = float(np.clip(twap_baseline * participation_mult, 0.0, 1.0))
        self.env.l2_urgency = float(np.clip(urgency, 0.0, 1.0))

        window_obs: list[np.ndarray] = []
        agg_reward = 0.0
        terminated = truncated = False
        info = self._last_info
        l3_obs = self._l3_obs
        for _ in range(self.n_ticks):
            # Step 3a: normalize with the frozen checkpoint's own saved stats.
            norm_obs = self._l3_obs_normalizer.normalize_obs(l3_obs[np.newaxis, :])[0]
            # Step 3b: thread state=/episode_start= explicitly across every inner-loop
            # tick, INCLUDING across L2 window boundaries within the same episode (only
            # reset() sets episode_start back to True) -- get this wrong (e.g. the
            # Section 4.1 reference code's bare `predict(l3_obs, deterministic=False)`,
            # no state=/episode_start= at all) and RecurrentPPO silently falls back to
            # stateless behavior: it still returns a syntactically valid action every
            # call, so nothing errors, but the frozen policy loses the queue/order
            # memory its LSTM was trained to use. See test_wrappers.py for a test that
            # actually catches this rather than a read-through.
            l3_action, self._l3_lstm_state = self.l3_model.predict(
                norm_obs,
                state=self._l3_lstm_state,
                episode_start=np.array([self._l3_episode_start]),
                deterministic=self.l3_deterministic,
            )
            self._l3_episode_start = False
            l3_obs, r, terminated, truncated, info = self.env.step(l3_action)
            window_obs.append(l3_obs)
            agg_reward += r
            if terminated or truncated:
                break

        self._l3_obs = l3_obs
        self._last_info = info

        if self.l2_reward_mode == "potential_is_shaping":
            # Replaces agg_reward entirely -- does not add to it. self.env._episode_fills
            # is the SAME cumulative list LOBExecutionEnv.step() itself passes to
            # compute_implementation_shortfall() at the real terminal tick, and info["mid_price"]
            # (built from self.env._current_tick(), i.e. self.env._ticks[self.env._tick_idx]
            # AFTER that tick's own increment) is the SAME tick LOBExecutionEnv.step() uses for
            # terminal_tick.mid_price when this window happens to be the terminal one -- see
            # l2_reward.py's module docstring for why this makes Phi(T) bit-identical to the
            # real terminal IS, not merely close, and the telescoping sum therefore exact.
            agg_reward, self._l2_prev_phi = l2_window_reward(
                prev_phi=self._l2_prev_phi,
                side=self.env.side,
                episode_fills=self.env._episode_fills,
                qty_total=self.env.qty_total,
                arrival_price=self.env.arrival_price,
                current_mid_price=info["mid_price"],
                fee_bps_per_fill=self.env.fee_bps_per_fill,
                kappa=self.env.reward_weights.kappa,
            )

        l2_obs = _downsample_window(
            np.asarray(window_obs, dtype=np.float32), info, self.env.horizon_ticks,
            l2_action if self.l2_include_prev_action else None,
        )
        return l2_obs, agg_reward, terminated, truncated, info
