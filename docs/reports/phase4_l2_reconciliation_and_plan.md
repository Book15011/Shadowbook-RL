# Phase 4 prep, L2 (Strategist) track: reconciliation + integration plan

Read-only reconciliation of architecture_spec.md Section 4.1's FrozenL3Wrapper/train_l2.py
reference code against the real, current `LOBExecutionEnv` (`src/envs/lob_execution_env.py`,
916 lines as of this writing) and `RewardWeights`/`step_reward` (`src/envs/reward.py`). No code
written yet -- this is the design doc called for before FrozenL3Wrapper/train_l2.py are built.

Precedent: `train_l3.py`'s own module docstring already did this exact reconciliation for L3's
own constructor call (`tier=`, `l2_override=`, `seed=` on the constructor -- none real). This
doc is the same exercise for the L2/L3 boundary Section 4.1 describes.

---

## Part A -- Reconciliation

### A.1 What the real LOBExecutionEnv actually exposes

**Constructor** (`__init__`, lob_execution_env.py:284-300):

```python
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
) -> None:
```

No `tier=`, `l2_override=`, or `seed=` param. Confirmed identical to what train_l3.py's own
docstring already established -- this doc doesn't re-litigate that, just confirms it still
holds.

**Per-tick mutation hooks that already exist:** `l1_risk_score`, `l1_confidence`, `l2_urgency`,
and `l2_target_slice_ratio_override` are plain public instance attributes, not properties or
setter methods. They are read fresh on every `_build_obs()` call
(`_compute_l2_target_slice_ratio()`, line 630, reads `self.l2_target_slice_ratio_override`
live; `_build_obs()`, line 684, reads `self.l2_urgency` live). This means **changing L2's
target mid-episode without reconstructing the env already works today**, via plain attribute
assignment (`env.l2_urgency = 0.8`) immediately before the next `env.step()` call -- no new
method is required for this specific need. This is the one piece of real L2-facing
infrastructure that already exists and is correct.

There is no method that changes anything else mid-episode (no partial-reset, no
"re-seed inventory" hook, nothing tier-related) -- the class is single-tier throughout.

**Real observation space** (`_OBS_SPEC`, lines 110-134): `gym.spaces.Box(shape=(42,),
dtype=float32)`, matches architecture_spec.md Section 3.1's table index-for-index, including
idx 15 (`l2_target_slice_ratio`) and idx 16 (`l2_urgency`) as ordinary entries in the *same*
vector L3 consumes. There is exactly one observation space on the class. No
`l2_observation_space` attribute, property, or anything resembling Section 3.1's described
"L2 consumes a temporally downsampled view (1s/10s aggregates) ... plus a TWAP-schedule
deviation scalar" exists anywhere in `_build_obs()` or `_precompute_feature_series()`. That
downsampled/aggregated L2-specific feature pipeline is design-doc prose only -- not built, not
stubbed, not partially present.

**Real action space** (line 327): `gym.spaces.MultiDiscrete([4, 11, 5])`. There is exactly one
action space on the class, matching Section 3.2's L3 action space exactly (order_type,
price_offset_idx, size_frac_idx). No `l2_action_space` attribute or property exists. Section
3.2's L2 `Box([0,0],[2,1])` (participation_rate_multiplier, urgency) is nowhere in the real
code.

**Tier gating:** none. `step()` (line 802) unconditionally decodes an L3-shaped
`MultiDiscrete([4,11,5])` action and advances exactly one 100ms tick. There is no branch
anywhere in the class keyed on which "tier" is acting.

### A.2 Section 4.1 reference code vs. real class -- confirmed gaps

From the reference `train_l3.py`/`train_l2.py` snippets:

| Reference symbol | Exists on real `LOBExecutionEnv`? |
|---|---|
| `tier=` (constructor kwarg) | No |
| `l2_override=` (constructor kwarg) | No |
| `seed=` (constructor kwarg) | No (seed lives on `reset(seed=...)`, standard gym) |
| `env.l2_action_space` | No |
| `env.l2_observation_space` | No |
| `env.apply_l2_action(l2_action)` | No |
| `env.get_l3_obs()` | No |
| `env.step_l3(l3_action)` | No -- only `env.step()` exists, already L3-shaped/tick-granular |
| `env.get_l2_obs()` | No |
| `env.l2_info()` | No |

All ten confirmed absent by direct reading, not assumed. Nine of these are exactly the
`tier=`/`l2_action_space`/`apply_l2_action`/`step_l3`/`get_l3_obs`/`get_l2_obs`/`l2_info()` list
flagged going in, plus two more found in the same pass: `l2_override=` (constructor kwarg,
same category as `tier=`) and `l2_observation_space` (attribute, same category as
`l2_action_space`).

Net effect: the reference `FrozenL3Wrapper`/`train_l2.py` code will not run against the real
class at all -- every non-trivial line in the reference `FrozenL3Wrapper.__init__`/`step()`
touches a symbol that doesn't exist. This isn't a naming mismatch fixable by a thin shim; the
tier-orchestration logic the reference code assumes lives *inside* the env has to be built
somewhere, and the real env doesn't have it.

---

## Part B -- Proposed integration plan

### B.1 Mechanism: wrapper-only, zero changes to `lob_execution_env.py`

Recommend building all L2/L3-boundary logic in a new `src/envs/wrappers.py::FrozenL3Wrapper`
(a `gym.Wrapper`), and making **no changes to `lob_execution_env.py`**. Rationale:

- The one piece of state L2 needs to mutate live (`l2_target_slice_ratio_override`,
  `l2_urgency`) is already public and already read fresh every tick -- nothing to add.
- Everything else the reference code assumes (tier gating, dual action/obs spaces, L2-cadence
  aggregation, LSTM state carry-forward) is orchestration logic that belongs in a wrapper
  around a single-tier env, not inside the tier-agnostic simulator itself. Keeping it out of
  `lob_execution_env.py` also avoids touching a file live in another session.
- Order/queue/inventory state (`self._resting`, `self.qty_remaining`, `self._last_fill_tick_idx`,
  etc.) is preserved across the L2 decision boundary *for free* -- the wrapper drives the same
  live env instance's `step()` in a tight inner loop, no reset/reconstruction between L2
  decisions, so this state already carries over without any special handling.

If a future need arises for e.g. input validation on the override or a richer setter, that's a
small, additive, optional method on the env -- not a prerequisite for unblocking L2 training.

### B.2 FrozenL3Wrapper -- real role vs. reference version

The reference version is a thin adapter: it assumes the env already does tier gating,
aggregation, and dual obs/action spaces, and just calls through to them. The real version has
to **be** the entire L2/L3 integration layer, since the env provides none of that. Concretely:

```python
class FrozenL3Wrapper(gym.Wrapper):
    def __init__(self, env: LOBExecutionEnv, l3_model, vecnormalize_path: str,
                 ticks_per_l2_decision: int = 50):
        super().__init__(env)
        self.l3_model = l3_model
        self.n_ticks = ticks_per_l2_decision
        # See B.3 -- l3_model was trained under VecNormalize(norm_obs=True, clip_obs=5.0);
        # raw env obs must be normalized with the SAME saved stats before predict(), or the
        # frozen policy sees out-of-distribution inputs.
        self._obs_rms = load_vecnormalize_obs_rms(vecnormalize_path)
        self._lstm_state = None       # RecurrentPPO hidden state, threaded across steps
        self._l3_episode_start = True  # RecurrentPPO episode_start flag

        self.action_space = gym.spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([2.0, 1.0], dtype=np.float32),
        )
        # No separate L2 obs space exists in real code (see A.1) -- reuse the identical
        # 42-dim Box L3 uses rather than inventing a downsampled feature pipeline that isn't
        # built anywhere (see B.4). Revisit only if L2 training shows this is a real
        # bottleneck, not preemptively.
        self.observation_space = env.observation_space

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._lstm_state = None
        self._l3_episode_start = True
        return obs, info

    def step(self, l2_action):
        participation_mult, urgency = float(l2_action[0]), float(l2_action[1])

        # env's own _compute_l2_target_slice_ratio() already IS the fixed-linear-TWAP
        # baseline (ticks_elapsed/horizon_ticks) whenever the override is None -- read that
        # baseline BEFORE overriding it, then scale by the multiplier. (Uses a private method;
        # flagged below as a minor fragility, not blocking.)
        twap_baseline = self.env._compute_l2_target_slice_ratio()
        self.env.l2_target_slice_ratio_override = float(np.clip(twap_baseline * participation_mult, 0.0, 1.0))
        self.env.l2_urgency = float(np.clip(urgency, 0.0, 1.0))

        agg_reward = 0.0
        terminated = truncated = False
        l3_obs = self.env._build_obs()  # current obs, no public getter exists -- see B.4
        info = {}
        for _ in range(self.n_ticks):
            norm_obs = self._obs_rms.normalize(l3_obs)  # see B.3
            l3_action, self._lstm_state = self.l3_model.predict(
                norm_obs, state=self._lstm_state,
                episode_start=np.array([self._l3_episode_start]),
                deterministic=False,
            )
            self._l3_episode_start = False
            l3_obs, r, terminated, truncated, info = self.env.step(l3_action)
            agg_reward += r
            if terminated or truncated:
                break

        return l3_obs, agg_reward, terminated, truncated, info
```

This is illustrative, not final code to commit -- written here to make the concrete gaps below
legible, per the "design doc, not code" instruction.

### B.3 Critical, non-obvious gap the reference code misses entirely: VecNormalize

Confirmed directly from `train_l3.py`:

```python
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=5.0, gamma=ppo_cfg["gamma"])
...
print("Saved model to models/l3_executioner_v1, VecNormalize to models/l3_vecnormalize.pkl")
```

The frozen L3 policy (`models/l3_executioner_v1.zip`, confirmed present on disk, alongside
`models/l3_vecnormalize.pkl`, same timestamp) was trained on **normalized** observations
(`norm_obs=True, clip_obs=5.0`). The Section 4.1 reference `FrozenL3Wrapper.step()` calls
`self.l3_model.predict(l3_obs, ...)` directly on raw env observations, with no mention of
loading or applying the saved normalization stats. Feeding raw (unnormalized) observations to
a policy trained under `VecNormalize` is a real, silent correctness bug, not a style issue --
the policy would see out-of-distribution inputs on every call and its behavior during L2
training would not reflect the actual trained L3 policy. This needs `l3_vecnormalize.pkl`
loaded and its `obs_rms` applied to every observation before `predict()` (reward normalization
doesn't matter here since the wrapper aggregates raw per-tick `r` from `self.env.step()`, which
is unaffected by L3's own VecNormalize reward scaling).

Flagging this now since it's the kind of gap that would otherwise surface only as "L2 training
mysteriously doesn't work" much later.

### B.4 Other minor gaps/fragilities in the illustrative code above

- `self.env._compute_l2_target_slice_ratio()` and `self.env._build_obs()` are both
  underscore-prefixed (private) methods on the real env, called from outside the class. Works
  today, but is coupling the wrapper to private implementation details rather than a public
  API. Acceptable for an initial version; worth a one-line public accessor later if this
  friction shows up more than once (not a blocker now).
- The reference code's `self.l3_model.predict(l3_obs, deterministic=False)` (no `state=`/
  `episode_start=`) silently drops LSTM recurrence for a `RecurrentPPO` policy -- this is a
  correctness gap in the reference code itself, not just an API-naming mismatch. The plan above
  threads `state=`/`episode_start=` explicitly and resets both in `reset()`.

### B.5 `ticks_per_l2_decision=50` given the real `tick_interval_s`

Confirmed `tick_interval_s` is never overridden anywhere in the codebase (grepped
`configs/`, `src/`) -- it stays at the constructor default of 0.1s in actual practice.
`50 ticks x 0.1s/tick = 5.0s` per L2 decision, squarely inside Section 3.2's own "L2 decision
every 1-10s" framing. `horizon_ticks=3000` (confirmed the real production value from
`configs/ppo_l3.yaml`'s `env:` block) x 0.1s = 300s (5 min) episodes -> 60 L2 decisions per
parent-order episode. **This number is not aspirational** -- unlike most of Section 4.1, it
checks out cleanly against the real tick rate without adjustment.

---

## Part D -- Other §4.1/§4.2 items: aspirational vs. confirmed

| Item | Status |
|---|---|
| `l2_observation_space` (Section 3.1's "downsampled 1s/10s aggregate + TWAP-deviation scalar") | **Aspirational.** No such feature pipeline exists anywhere in `_build_obs()`/`_precompute_feature_series()`. Plan above sidesteps this by reusing the 42-dim vector as-is; building the described downsampled pipeline would be real, separate feature-engineering work. |
| `l2_action_space = Box([0,2],[0,1])` | Not on the env (expected, per A.1) -- but the *shape* itself is a reasonable, buildable target; recommend defining it only on the wrapper, per B.2. |
| `models/l3_executioner_v1.zip` (frozen L3 checkpoint train_l2.py would load) | **Confirmed present on disk**, with a matching `l3_vecnormalize.pkl` from the same run -- one of the few Section 4.1 assumptions that holds as-is (module-docstring caveat in `reward.py`: this is described as "the finished 20M-step baseline" but multiple same-day probe variants also exist in `models/`; confirm with the L1/L3 session which exact checkpoint is the intended "frozen" one before wiring it in, since `l3_executioner_v1.zip`'s mtime is very recent and may be a probe run rather than the full 20M-step baseline). |
| `gamma=0.995` reused for L2's SAC config | **Unconfirmed / worth rederiving, not just copy-pasting.** Section 4.1 justifies L3's own `gamma=0.995` explicitly as "~100ms ticks -> ~5s effective discount horizon." L2's decision cadence is 5s/decision (B.5), so the same `gamma=0.995` gives an effective discount horizon of roughly `1/(1-0.995) = 200 decisions x 5s = 1000s (~16.7 min)` -- well past a single 300s episode. Not necessarily wrong (SAC bootstraps across truncation normally), but it's a different regime than the reasoning Section 4.1 gives for reusing the identical number, so it deserves its own derivation rather than being inherited unchanged. |
| `buffer_size=500_000` | **Unconfirmed, arbitrary.** No measurement or derivation behind it in the spec (contrast `configs/ppo_l3.yaml`'s `eval_freq_timesteps`, which the L3 session derived from measured wall-clock throughput). At 60 decisions/episode this covers ~8,333 parent-order-episodes of L2 transitions -- plausible, not validated. |
| `total_timesteps=2_000_000` for L2 | **Unconfirmed.** No L2 training has run yet, so there's nothing yet to check this against (L3's 20M number at least has smoke-test throughput data behind it). Flag as a starting point to revisit once real L2 wall-clock/throughput numbers exist. |
| `train_freq=1, gradient_steps=1` | SAC library defaults, unremarkable -- not flagged. |

---

## Summary / what's next

No code written. Recommend: (1) confirm which `models/l3_executioner_v1*.zip` is the intended
frozen L3 checkpoint with whoever owns the L3 track before wiring it into `FrozenL3Wrapper`,
since several same-day probe variants exist alongside it; (2) build `src/envs/wrappers.py`
(`FrozenL3Wrapper`, per B.2, including the VecNormalize fix in B.3) and `src/train/train_l2.py`
next, once this plan is reviewed; (3) `lob_execution_env.py` itself needs no changes for any of
this.

---

## Follow-up (2026-08-19 22:07 HKT): checkpoint question is BLOCKED, not resolved

Checked against `docs/TRACK_STATUS.md`'s L3/Env-Physics section before proceeding further,
per instruction. Resolution of the "which checkpoint is frozen L3" open question from Part D
above:

**Not resolved -- genuinely blocked, not just ambiguous.** Two real bugs were found and fixed
in `lob_execution_env.py` since this doc's original Part D table was written
(`qty_at_price()`'s missing `rtol=0.0`, and `_place_limit()`'s missing crossing-order
handling -- both confirmed and fixed, per the L3 track's own writeup). `models/
l3_executioner_v1.zip` is confirmed (sha256-verified) to be the original 20M-step baseline --
but it was trained entirely under the OLD, buggy physics. An init-strategy probe (does that
checkpoint's weights warm-start a fine-tune under the now-fixed physics, vs. training from
scratch) is in progress: from-scratch is already ruled out as impractical (near-random policy
exploits the correct crossing fix, episodes terminate in 11-21 ticks instead of 3,000), and
warm-start looks healthy (fps ~350-359, full 3,000-tick episodes, ~25min/499,712-step sample)
-- but the L3 track explicitly stopped short of committing to the full 2,000,000-step run,
pending a go/no-go decision, and says outright: **"Do not treat \[l3_executioner_v1.zip\] as
the final 'frozen' checkpoint for FrozenL3Wrapper yet -- recommend waiting for that decision
before wiring integration against a specific checkpoint file."**

Per this session's own task instructions: when this probe is not yet resolved, do not pick a
checkpoint and do not proceed to hyperparameter derivation (Part B) or observation-space design
(Part C) -- stop and report the block instead. Both of those would still need redoing (SAC's
effective horizon depends on which policy generates L2's training transitions; L2's
observation-space proposal is independent of the checkpoint choice, but sequencing all of this
behind one clean go/no-go avoids building against a moving target twice, the exact repeated
pattern this project has already paid for). No further plan content added this round --
Parts B/C remain open until the L3 go/no-go lands.

**Concrete unblock condition:** L2 wrapper work (this doc's Parts B/C, and eventually
`FrozenL3Wrapper`/`train_l2.py` themselves) can resume once the L3 track's go/no-go on the
full 2,000,000-step warm-start run is decided AND (if warm-start is approved) that run
completes and produces the checkpoint that supersedes `l3_executioner_v1.zip`. If warm-start
is rejected in favor of some other resolution, the frozen-checkpoint choice needs to be
re-confirmed against whatever that resolution produces -- this doc should not assume warm-start
is the outcome.
