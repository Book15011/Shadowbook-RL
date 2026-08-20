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

---

## Part A status check (2026-08-19, this session) -- still unresolved, everything below is PROVISIONAL -- PENDING PART A

Read `docs/TRACK_STATUS.md`'s L3/Env-Physics section fresh at the start of this session
(fresh read, not reused from a prior turn). It is byte-identical to the version read in the
previous follow-up ("Last updated: 2026-08-19 20:47 HKT", same content) -- the go/no-go on
committing warm-start to the full 2,000,000-step run is still pending, and the section still
says explicitly not to treat `models/l3_executioner_v1.zip` as final for `FrozenL3Wrapper`
yet. Per this session's own instructions: this is the "still unresolved" case, so Parts B/C
below proceed but are marked **PROVISIONAL -- PENDING PART A** throughout. No checkpoint path
is picked or assumed anywhere below. None of `lob_execution_env.py` / `reward.py` /
`train_l3.py` / the two test files were read or touched this session (per this session's own
hard boundary) -- all facts below are sourced from `architecture_spec.md`,
`configs/ppo_l3.yaml`, `data/splits/l2_bybit_btcusdt_split.json`, and this doc's own prior,
already-committed findings (which were themselves sourced from the live env file in an earlier
session, before it became live/uncommitted work).

## Part B (PROVISIONAL -- PENDING PART A): deriving L2's real SAC hyperparameters

### B.1 Ground truth inputs, sourced directly (not assumed)

- `horizon_ticks = 3000` -- confirmed from `configs/ppo_l3.yaml`'s `env:` block (the real
  production value used by L3 training), not the illustrative "30-minute horizon" example in
  architecture_spec.md Section 3.4 (which, at the real `tick_interval_s`, would actually be
  18,000 ticks -- the spec's own example number doesn't match the real configured horizon;
  flagged, not resolved here, out of scope for this track).
- `tick_interval_s = 0.1` (100ms/tick) -- this session did not re-read the live env file to
  re-verify this (out of bounds this round), but it was already established and committed in
  this doc's original Part A (grepped `configs/` and `src/` for any override; found none) and
  is independently corroborated by architecture_spec.md Section 4.3's own
  `L1_EVERY_N_TICKS = 600 # ~60s at 100ms ticks` / `L2_EVERY_N_TICKS = 10 # ~1s at 100ms ticks`
  comments, which assume the same 100ms tick rate.
- `ticks_per_l2_decision = 50` -- Section 4.1's literal reference value.
- => L2 decisions per episode = 3000 / 50 = **60** (matches this task's stated figure).
- => L2 decision cadence = 50 x 0.1s = **5.0s/decision**, inside Section 3.2's own "L2 decides
  every 1-10s" framing.
- => episode wall-clock length = 3000 x 0.1s = **300s (5 minutes)**.
- `train_dates = 405` -- **verified against the real, persisted split artifact**,
  `data/splits/l2_bybit_btcusdt_split.json` (not the spec's own illustrative
  `"source_day_count": 296` example in Section 2.5, which is placeholder text, not real data).
  Real artifact: `source_day_count: 441` total, **`train_dates: 405`**, `val_dates: 18`,
  `test_dates: 18`, `known_gap_dates: 49`, train window 2024-04-18 to 2025-07-15. The task's
  stated "405 train days" figure checks out exactly against the real artifact.

**Flagging one real spec-internal inconsistency surfaced while sourcing the above:** Section
4.1's training-time cadence (`ticks_per_l2_decision=50` -> 5s/decision) and Section 4.3's
live-inference-loop cadence (`L2_EVERY_N_TICKS=10` -> 1s/decision) are different numbers for
what is presented as the same tier's decision frequency. This may be intentional (a coarser
cadence during training to save frozen-L3-rollout compute per SAC gradient step, vs. a finer
one at deployment), but it is not stated as intentional anywhere in the spec, and if L2 is
*trained* at a 5s cadence but *deployed* at a 1s cadence, that is a genuine train/inference
distribution mismatch (the policy would be executed on a decision cadence 5x finer than
anything it was trained against). This doc uses the Section 4.1 training-time value
(`ticks_per_l2_decision=50`) throughout, per this task's explicit instruction -- flagging the
4.3 discrepancy as a separate open question, not resolving it here (it needs a judgment call,
not arithmetic, and isn't blocking the derivations below).

### B.2 `buffer_size=500_000` -- confirmed as reasonable, with arithmetic

Each L2 episode (one parent order = one env episode) yields at most 60 L2-cadence transitions
(fewer only if the full parent order fills and the episode terminates before the 3000-tick
horizon).

- `buffer_size / transitions_per_episode = 500,000 / 60 ≈ 8,333` episodes' worth of L2
  transitions the buffer can hold at capacity.
- `total_timesteps / transitions_per_episode = 2,000,000 / 60 ≈ 33,333` total episodes over
  the entire training run (Section 4.1's stated `total_timesteps=2_000_000` for L2's SAC).
- So the buffer holds roughly `8,333 / 33,333 ≈ 25%` of the run's total transition volume at
  any point once full -- i.e., a sliding window over the most recent quarter of training.
  This is a normal, unremarkable fraction for an off-policy replay buffer (SAC buffers
  commonly span a meaningful fraction, not a tiny sliver, of `total_timesteps` in runs of this
  size) -- not oversized relative to available RAM-scale concerns, not so small it forces
  near-fully-online behavior.
- Diversity check against the 405 real train days: with `reset()` drawing a random file +
  random start tick + random side + random size per episode (existing, already-documented env
  behavior), and each train day holding roughly 864,000 ticks against a ~3,010-tick episode
  window, there is enormous room for many non-overlapping windows per day -- day count is not
  the binding constraint on episode diversity, transition budget is. Across the full run,
  405 days get sampled roughly `33,333 / 405 ≈ 82` times each on average; within any one
  8,333-episode buffer snapshot, that is still on the order of 20 samples/day of coverage --
  plenty of within-buffer diversity, not degenerate.

**Confirmed as-is: `buffer_size=500_000` holds up under L2's real cadence.** No adjustment
proposed.

### B.3 `gamma=0.995` -- re-derived on L2's own terms, confirmed as-is with an explicit caveat

Standard geometric-series "effective/characteristic horizon" for a discount factor:
`1 / (1 - gamma)` steps, in whatever the step unit is.

- At L2's own decision cadence: `1 / (1 - 0.995) = 200` **decisions**.
- `200 decisions x 5.0s/decision = 1000s ≈ 16.7 minutes`.
- Episode length is `300s (5 minutes) = 60 decisions`.
- So `gamma=0.995`'s effective horizon (200 decisions) is **~3.3x longer than the entire
  episode** (60 decisions) -- concretely, the discount weight on the very LAST decision of an
  episode, relative to the first, is `0.995^59 ≈ 0.744`: barely discounted at all across the
  whole parent-order execution.

This is a materially different regime than how Section 4.1 justifies the *same* number for
L3 ("~100ms ticks -> ~5s effective discount horizon" -- note this comment does not itself
match `1/(1-0.995)=200 ticks x 0.1s = 20s`, not 5s, under the standard formula; L3's own
gamma justification comment appears to already be a loose approximation, not exact arithmetic
-- out of scope to fix here, but worth naming so the L2 number isn't held to a standard the L3
number doesn't actually meet either). Copy-pasting `0.995` into L2's config is not, on its own,
a derivation.

**Is near-flat discounting across the whole episode actually wrong for L2, though?**
Re-derived on L2's own terms: no, not obviously. This task's reward structure (Section 3.3) is
dominated by one large terminal implementation-shortfall term (`r_terminal = -kappa * IS_bps`,
paid once at episode end) plus smaller per-step shaping terms. For a bounded, single-parent-
order episode where the thing that actually matters is the *whole order's* execution quality,
near-flat discounting is arguably the *correct* choice, not an artifact to fix -- it stops SAC
from myopically overweighting early-episode participation-rate decisions relative to the
terminal outcome that reflects the entire order. This is also less costly for SAC specifically
than it would be for an on-policy, rollout-based method (PPO/GAE), since SAC learns Q-values
via off-policy TD bootstrapping through the replay buffer rather than needing long on-policy
rollouts to estimate returns -- a generous effective horizon doesn't inflate rollout variance
the way it would for L3's own PPO.

**Recommendation: confirm `gamma=0.995` as the starting value, with the above as its real
justification (not L3's copy-pasted rationale) -- but flag it as an explicit ablation
candidate, not a fully closed question.** A lower alternative worth testing once real L2
training is unblocked: `gamma ≈ 0.983` (`1/(1-gamma) ≈ 60` decisions, matching effective
horizon to episode length exactly) would give more meaningful within-episode credit
assignment (last-decision weight `0.983^59 ≈ 0.37` instead of `0.74`) at the cost of weighting
the terminal IS term less heavily relative to early per-step shaping terms. This is ultimately
an empirical question this session cannot settle -- no training may run under this task's own
hard boundaries, and it's downstream of the Part A checkpoint block regardless. Recorded here
as the concrete alternative to try, not left as a bare "flagged, unresolved."

## Part C (PROVISIONAL -- PENDING PART A): L2's real observation space

### C.1 What's actually in the existing 42-dim vector vs. what Section 3.1 claims L2 gets

Section 3.1 states L2 "consumes a temporally downsampled view of the same 42-dim vector
(1s/10s aggregates ...) ... and additionally receives a TWAP-schedule deviation scalar." Per
this doc's original Part A, **no such downsampled/aggregated pipeline exists anywhere in the
real code** -- there is exactly one `_build_obs()`, producing one 42-dim snapshot per 100ms
tick, with no L2-cadence aggregation step. Building genuine temporal downsampling (e.g.
mean/std of each of the 42 dims over the trailing 50-tick window) is real, non-trivial
feature-engineering work -- a new rolling-buffer mechanism, ~50x more computation per L2
decision than a single snapshot -- and is out of scope for this task (no implementation code
this round, and this task's own hard boundary prohibits touching the live env file that would
need to change).

Re-examining the 42-dim vector's *existing* contents, though, several dims are already
rolling-window features whose window length happens to land near L2's own cadence, purely by
coincidence of the numbers involved, not by design for L2:
- idx 4 `mid_return_5s_z` -- a trailing 5s return z-score. At `ticks_per_l2_decision=50`
  (5.0s/decision), this is *almost exactly* "price drift since the last L2 decision" already.
- idx 12 `trade_flow_imbalance_5s` -- trailing 5s taker buy/sell imbalance. Same coincidence:
  already close to "net order flow since the last L2 decision" at the current cadence.
- idx 5 `realized_vol_60s_z`, idx 40 `taker_buy_sell_ratio_1m` -- wider windows (60s), genuinely
  different from and complementary to the per-decision window, not a substitute for it.

This coincidence is fragile, not a real design guarantee: if `ticks_per_l2_decision` is ever
changed (e.g. reconciled toward Section 4.3's `L2_EVERY_N_TICKS=10` / 1s cadence, per the B.1
flag above), idx 4/12's fixed 5s windows would stop lining up with the decision cadence at all.
Noting this explicitly rather than silently relying on a coincidence that could silently break
under a config change nobody thinks to re-check.

### C.2 What is genuinely missing from the 42-dim vector for L2's purposes

- **TWAP-schedule deviation** -- Section 3.1 explicitly calls for this, and it does not exist
  in any of the 42 dims. It is cheaply derivable from state the env already exposes, though,
  without new rolling-window infrastructure: `(qty_total - qty_remaining) / qty_total` (actual
  executed-so-far fraction) minus the existing TWAP baseline formula (already implemented as
  `_compute_l2_target_slice_ratio()`'s default path: `ticks_elapsed / horizon_ticks`) gives
  signed deviation of real execution from the fixed-TWAP schedule -- positive = ahead of
  schedule, negative = behind.
- **Fill progress since L2's own last decision** -- feedback on whether L3 actually executed
  close to what L2 last asked for. Nothing in the 42-dim vector isolates "since my last
  decision" specifically (idx 1 `inventory_remaining_norm` and idx 41 `own_open_orders_norm`
  are current-state snapshots, not deltas over the just-elapsed window). This is computable by
  the wrapper itself (it already straddles the L2-decision boundary and can diff
  `qty_remaining` before/after its own inner tick loop) without any env change.
- Time-remaining-in-episode is **not** missing -- idx 0 `time_remaining_norm` already covers
  it identically for both tiers; not duplicating it.

### C.3 Concrete proposed observation space

**`Box(shape=(44,), dtype=np.float32)`** -- the existing 42-dim vector, snapshotted at the
tick where each L2 decision is made, plus 2 new L2-specific scalars appended at the end:

| Index | Feature | Range | Source |
|---|---|---|---|
| 0-41 | identical to `_OBS_SPEC` (Section 3.1 table) | as existing | reused as-is, no new computation |
| 42 | `schedule_deviation` | `[-1, 1]` | `clip((qty_total - qty_remaining)/qty_total - twap_baseline, -1, 1)`; `twap_baseline` from the env's existing default-path formula |
| 43 | `fill_progress_since_last_decision` | `[0, 2]` | `clip(qty_filled_last_window / max(assigned_slice_last_window, eps), 0, 2)`, tracked by the wrapper itself across its own inner-loop boundary; capped at 2 to represent "over-filled relative to what was asked" (e.g. a MARKET order fired more aggressively than the assigned slice implied) rather than clipping that signal away at 1 |

Note on idx 15/16 within the reused block (`l2_target_slice_ratio`, `l2_urgency`): at L2
decision time these reflect the *result of L2's own previous decision* (the wrapper set them
before the just-finished inner tick loop), not something new being computed for L2 -- this is
the standard "observe your own last action" RL pattern, not a bug, but worth stating
explicitly rather than leaving ambiguous which tier "owns" those two values in this vector.

**Why this shape over a genuine downsampled/aggregated pipeline:** it is the cheapest option
that still closes Section 3.1's specific, named gap (the TWAP-deviation scalar) and adds the
one other signal (fill-progress feedback) that a wrapper sitting at the L2/L3 boundary can
supply for free from state it already has to touch. A true temporally-downsampled feature set
(rolling mean/std of all 42 dims over each 50-tick window) is left as a documented future
enhancement, to be built only if this simpler version proves empirically insufficient once L2
training can actually run -- not built preemptively against an unvalidated need.

## Summary of this round

Part A: still blocked, unchanged since the last check-in -- reported, not resolved, no
checkpoint picked. Parts B/C: buffer_size=500,000 confirmed as-is with real arithmetic;
gamma=0.995 confirmed as a starting value with real (not copy-pasted) justification, plus a
named empirical alternative (~0.983) to test once training is unblocked; L2 observation space
concretely proposed as `Box(shape=(44,))` (42 existing dims + 2 new L2-specific scalars). All
of Parts B/C marked **PROVISIONAL -- PENDING PART A** -- ready for review, but not to be
treated as final until the L3 checkpoint question resolves, since the checkpoint choice could
still, in principle, come with its own guidance that affects these numbers (e.g. if a fixed-
physics retrain changes `horizon_ticks` or reward scaling). No `FrozenL3Wrapper`/`train_l2.py`
code written.

---

## FINAL SPEC (2026-08-20, this session): L2 observation space + SAC hyperparameters

Supersedes the "PROVISIONAL -- PENDING PART A" Parts B/C above for the specific items
resolved below (cadence, observation space). Concrete enough to implement `FrozenL3Wrapper`
directly from this section without re-deriving anything -- implementation itself is still
not done this round, per instruction.

### Step 1 -- Cadence conflict: confirmed resolved in the spec, confirmed clean elsewhere

Re-read architecture_spec.md Section 4.1 and Section 4.3 fresh this session (not reused from
a prior read) via `git diff docs/architecture_spec.md` against the last commit, to see the
actual change rather than infer it:

```diff
 L1_EVERY_N_TICKS = 600   # ~60s at 100ms ticks
-L2_EVERY_N_TICKS = 10    # ~1s at 100ms ticks
+L2_EVERY_N_TICKS = 50    #  I just change here
```

`L2_EVERY_N_TICKS` is now `50`, matching Section 4.1's `ticks_per_l2_decision=50` exactly --
the conflict this doc flagged previously is gone. **Flagging the fix's own confidence
level, not just its outcome:** this is a bare one-line value change with the comment
`# I just change here` -- unlike its `L1_EVERY_N_TICKS` sibling (which recomputes and states
"~60s at 100ms ticks"), there is no updated timing comment, no stated rationale for why 50 is
now the right number for the *inference*-loop context specifically (Section 4.3 describes a
live/backtest orchestration loop, a different context from Section 4.1's *training*-time
wrapper). Treating this as **resolved with moderate, not high, confidence** -- matches how
this task's own instructions characterized it going in.

**Grepped the real repo** (not just the spec doc) for anything else that hardcodes or
assumes the old 10-tick/1s L2 cadence -- `configs/`, `src/`, `tests/`, `scripts/`, plus a
broad `L2_EVERY_N_TICKS|ticks_per_l2_decision|l2.{cadence,frequency}` sweep across all
`.py`/`.md`/`.yaml`/`.json` files. **Nothing else in the codebase references either
constant at all.** The only other L2-related config artifact, `configs/sac_l2.yaml`, is an
empty placeholder (`# Placeholder SAC (L2) config keys only`, every key unfilled) -- no
cadence value, no hyperparameter value, nothing to conflict with. No test file, script, or
config hardcodes a 10-tick/1s L2 assumption anywhere. **Stating this explicitly, per
instruction: nothing else turned up.** `configs/sac_l2.yaml` is the natural landing spot for
this section's finalized hyperparameters once someone is ready to fill it in -- not done this
round (no code/config writing this round).

### Step 2 -- Concrete L2 observation-space spec

#### 2a. Per-feature aggregation rule for all 40 non-excluded indices (0-14, 17-41)

Grouped by feature type/mechanism, not decided index-by-index blind:

**Group 1 -- already rolling-window features computed by the base env (idx 3, 4, 5, 12, 40):
use the LAST (freshest) value at decision time, no further transformation.**
These are already trailing-window quantities (`_precompute_feature_series()`, confirmed
unchanged in the current working tree). Re-averaging an already-smoothed series over the L2
window would double-smooth it and destroy the "how fresh is this read" property that makes a
short-window z-score useful at all.
- **idx 4 `mid_return_5s_z`** and **idx 12 `trade_flow_imbalance_5s`**: both 5s windows.
  With the cadence now settled at 50 ticks x 0.1s = **exactly 5.0s**, these two are not just
  "close to" but **exactly aligned** with the L2 decision window -- their last value at
  decision time already *is* "return / net flow since my last decision" to a very good
  approximation. No transformation needed, confirmed the intended reading.
- **idx 3 `mid_return_1s_z`** (1s window, shorter than the 5s decision cadence) and
  **idx 5 `realized_vol_60s_z`**, **idx 40 `taker_buy_sell_ratio_1m`** (60s windows, longer):
  kept as complementary multi-timescale reads (fast + slow, bracketing the ~5s decision-
  scale read from Group above) -- also last-value, no new aggregation.

**Group 2 -- point-in-time state/stock variables (idx 0, 1, 13, 14, 41): instantaneous
snapshot at decision time.**
`time_remaining_norm`, `inventory_remaining_norm`, `queue_position_ratio`,
`ticks_since_own_fill_norm`, `own_open_orders_norm`. These describe "what state am I in
right now" (time/inventory left, current resting-order queue position, recency since last
fill, current resting-order size) -- state, not a flow. A mean/max/sum over the window
doesn't correspond to any real quantity for a stock variable; the decision-relevant value is
always "current state as of decision time," which is operationally identical to "last value
in the window."

**Group 3 -- instantaneous microstructure reads with no built-in persistence (idx 2, 6, 7, 8,
9): aggregate via MEAN over the 50-tick window.**
`spread_norm`, `OBI_1`, `OBI_5`, `OBI_10`, `micro_mid_dev_ticks` -- each computed fresh every
tick directly from the current top-of-book snapshot, with no trailing window in their own
formula (confirmed against the current `_build_obs()`: OBI/micro-price/spread-ratio are all
single-tick computations). A single instantaneous tick's reading is noisy and can flip
sign/swing tick-to-tick; MEAN over the window turns this into "was there sustained
imbalance/spread-widening/micro-price pressure over the period I'm evaluating," which is
what L2 actually needs at its coarser cadence. Ranges are unchanged by meaning (mean of
values already in `[-1,1]`/`[0,1]`/`[-5,5]` stays in the same range).

**Group 4 -- structurally zero, no design decision needed (idx 10, 11): pass through as-is.**
`cancel_add_ratio_bid`, `cancel_add_ratio_ask` -- both hardcoded `0.0` in the real env
(confirmed in this doc's original Part A read, unchanged since -- genuinely blocked by the
data source, not a stub awaiting a real signal). Mean/max/last of a constant 0.0 is still
0.0; noting this explicitly rather than silently omitting the two indices.

**Group 5 -- depth-level snapshots (idx 19-38, 20 dims): aggregate via MEAN over the window,
per level.**
`book_depth_norm_0..19`. Individual order placements/cancellations at a given book level
cause sharp, transient jumps tick-to-tick; a single instantaneous depth reading at decision
time is a weak proxy for "how deep was this level, typically, during the period I'm
evaluating." MEAN per level gives a materially more stable read, same rationale as Group 3.
(MIN per level, for a "worst-case thinness" read, is a documented alternative -- not adopted
here since MEAN is the more standard default and this is unvalidated either way until real
L2 training exists to compare against.)

**Group 6 -- slow-moving external context, changes far slower than the L2 window (idx 17,
18, 39): instantaneous/last-value; aggregation is a mathematical no-op here, not just an
approximation.**
`l1_risk_score`, `l1_confidence` are mutated externally on the L1 track's own
`L1_EVERY_N_TICKS=600` (~60s) cadence -- 12x slower than the 50-tick L2 window, so within any
single L2 decision window these are constant in the large majority of cases. `funding_rate_z`
is stronger still: confirmed (current code) it is computed **exactly once, at `reset()`**,
not per-tick at all -- it is the *same* value for the entire episode, so "aggregating" it over
any window, L2-sized or otherwise, cannot change its value. Last-value is not an
approximation for idx 39; it is exact by construction.

Total: 5 + 5 + 5 + 2 + 20 + 3 = **40 indices**, every one of 0-14/17-41 accounted for exactly
once.

#### 2b. TWAP-schedule-deviation scalar -- confirmed computable from existing state, no new instrumentation

Re-read the current, live working tree of `lob_execution_env.py` (post-physics-fix; this
session did not modify it) to confirm this against real code rather than an earlier read from
memory. `_compute_l2_target_slice_ratio()` (still the right hook, unchanged) and the
attributes it depends on (`self.qty_total`, `self.qty_remaining`, `self._tick_idx`,
`self._episode_start`, `self.horizon_ticks`) are all present, unchanged, at the same names:

```python
schedule_deviation = (qty_total - qty_remaining) / qty_total - twap_baseline
# twap_baseline = env._compute_l2_target_slice_ratio(), read BEFORE the wrapper
# overrides l2_target_slice_ratio_override for the upcoming decision window (so it
# reads the env's own default linear-TWAP path: ticks_elapsed / horizon_ticks).
# Positive = ahead of schedule, negative = behind. Clip to [-1, 1].
```

**No new env instrumentation needed.** Everything on the right-hand side is already tracked
by the live env; this is arithmetic the wrapper performs on values it already has to read
anyway (it already reads/overrides `l2_target_slice_ratio_override` per this doc's original
Part B design).

#### 2c. Previous-action-as-input -- explicit, separate toggle (not silently baked in)

**Could not locate the referenced precedent.** Searched the repo (`find ... -iname '*.pdf'`)
and the wider box for "the recurrent-policy paper in this project's own PDFs" -- there are no
PDFs anywhere inside `lob-execution-hma` at all; the only PDFs on the box belong to unrelated
projects/coursework. **Flagging this as something I could not find rather than guessing which
paper was meant or fabricating a citation** -- if there's a specific paper you have in mind,
point me to it and I'll fold its guidance in; the recommendation below is my own reasoning,
not sourced from that paper.

**Why this is a genuinely separate question from idx 15/16's exclusion:** idx 15/16
(`l2_target_slice_ratio`, `l2_urgency`) are *excluded* from this proposal per Section 3.1's
own instruction ("naturally excluded since L2 produces them") -- but even if they weren't,
they would not reliably let L2 recover its own previous *raw* action. idx 16 (`urgency`) is
copied through unchanged, but idx 15 is `twap_baseline * participation_rate_multiplier`
(clipped to `[0,1]`) -- recovering the raw multiplier from idx 15 requires dividing by
`twap_baseline`, which is near-zero at the start of every episode (the exact regime where
knowing your own last action matters for producing a smooth next one). So "does L2 see its
own last action" and "are idx 15/16 in the vector" are not the same question, and conflating
them would silently under- or over-specify the design -- treating them as separate, as
instructed.

**Recommendation: include it, as an explicit, named, off-by-default-able parameter --
default ON.** `FrozenL3Wrapper(..., l2_include_prev_action: bool = True)`, adding 2 raw
scalars (`prev_participation_rate_mult` in `[0, 2]`, `prev_urgency` in `[0, 1]`) -- direct,
unlossy copies of L2's own last action, not the derived idx-15-style proxy.
Reasoning: Section 4.1 specifies plain `SAC("MlpPolicy", ...)` for L2 -- a feedforward,
non-recurrent policy, unlike L3's `MlpLstmPolicy`. L3's LSTM can in principle learn to
remember its own recent actions internally across a sequence without being told explicitly;
a memoryless `MlpPolicy` genuinely cannot -- each `predict()` call sees only the current
observation, with zero information about what it just committed to unless that information
is explicitly present in the vector. This is the standard justification for previous-action
conditioning in continuous-control RL generally (reducing action chatter/oscillation between
consecutive decisions, since the policy can see and smooth relative to its own last output)
-- not sourced from a specific paper here, flagged above. The toggle exists so this can be
turned off cheaply as an ablation if it doesn't help empirically, without touching the rest
of the observation-space design.

#### 2d. Final dimension count -- stated explicitly

- **Base (Section 3.1's literal requirement only, no previous-action):**
  40 (Step 2a) + 1 (schedule_deviation, 2b) = **`Box(shape=(41,), dtype=np.float32)`**.
- **Recommended (with `l2_include_prev_action=True`, 2c):**
  41 + 2 = **`Box(shape=(43,), dtype=np.float32)`**.

One item from this doc's own prior round is deliberately **not** carried into this final
spec: the previously-proposed `fill_progress_since_last_decision` scalar. This round's task
scoped Step 2 tightly to Section 3.1's literal requirement (downsampled view + TWAP-deviation
scalar) plus the explicitly-requested previous-action toggle -- adding a third, previously-
invented scalar back in without being asked would be exactly the kind of silent inclusion
this round's instructions are pushing against. Recorded here for traceability, not carried
forward: still a plausible future enhancement, not part of the final spec.

#### 2e. Concrete index-mapping table (old idx -> new position, transform)

| New pos | Old idx | Feature | Transform | Range |
|---|---|---|---|---|
| 0 | 0 | `time_remaining_norm` | instantaneous | [0,1] |
| 1 | 1 | `inventory_remaining_norm` | instantaneous | [-1,1] |
| 2 | 2 | `spread_norm` | mean/50-tick window | [0,1] |
| 3 | 3 | `mid_return_1s_z` | last value | [-5,5] |
| 4 | 4 | `mid_return_5s_z` | last value (= since-last-decision, exact match) | [-5,5] |
| 5 | 5 | `realized_vol_60s_z` | last value | [-5,5] |
| 6 | 6 | `OBI_1` | mean/window | [-1,1] |
| 7 | 7 | `OBI_5` | mean/window | [-1,1] |
| 8 | 8 | `OBI_10` | mean/window | [-1,1] |
| 9 | 9 | `micro_mid_dev_ticks` | mean/window | [-5,5] |
| 10 | 10 | `cancel_add_ratio_bid` | pass-through (always 0.0) | [0,5] |
| 11 | 11 | `cancel_add_ratio_ask` | pass-through (always 0.0) | [0,5] |
| 12 | 12 | `trade_flow_imbalance_5s` | last value (= since-last-decision, exact match) | [-1,1] |
| 13 | 13 | `queue_position_ratio` | instantaneous | [-1,1] |
| 14 | 14 | `ticks_since_own_fill_norm` | instantaneous | [0,1] |
| 15 | 17 | `l1_risk_score` | instantaneous (~constant within window) | [-1,1] |
| 16 | 18 | `l1_confidence` | instantaneous (~constant within window) | [0,1] |
| 17-36 | 19-38 | `book_depth_norm_0..19` | mean/window, per level | [-5,5] |
| 37 | 39 | `funding_rate_z` | instantaneous (exactly constant per episode) | [-5,5] |
| 38 | 40 | `taker_buy_sell_ratio_1m` | last value | [-1,1] |
| 39 | 41 | `own_open_orders_norm` | instantaneous | [0,1] |
| 40 | -- | `schedule_deviation` (new, 2b) | computed | [-1,1] |
| 41 | -- | `prev_participation_rate_mult` (new, 2c, toggle) | raw copy of L2's last action | [0,2] |
| 42 | -- | `prev_urgency` (new, 2c, toggle) | raw copy of L2's last action | [0,1] |

### Step 3 -- SAC hyperparameters re-confirmed against the now-settled cadence

The prior round's `buffer_size=500,000` and `gamma=0.995` derivations already assumed
`ticks_per_l2_decision=50` throughout (Section 4.1's literal training-time value, used
consistently even while Section 4.3's *inference*-time constant conflicted with it) -- the
arithmetic does not change now that Section 4.3 has been patched to match. What changes is
the caveat: previously PROVISIONAL for two independent reasons (the cadence conflict, and the
separate L3-checkpoint question); the cadence reason is now resolved (Step 1, moderate
confidence). **Dropping PROVISIONAL on `buffer_size` and `gamma` specifically** -- both stand
as this doc's earlier Part B stated them:
- `buffer_size=500,000` -- confirmed: ~8,333 L2-episode-equivalents of buffer coverage, ~25%
  of the full 2,000,000-step run's transition volume. No change.
- `gamma=0.995` -- confirmed as a starting value with L2-specific justification (effective
  horizon ~3.3x the 60-decision episode length, defensible given the terminal-IS-dominated
  reward), `~0.983` still flagged as the concrete empirical alternative to test once training
  runs. No change.

**Observation-dimensionality effect on network architecture, per instruction:** Section 4.1's
SAC reference (`SAC("MlpPolicy", env, buffer_size=..., ...)`) does not specify a `net_arch`,
meaning SB3's SAC default applies (`[256, 256]` for both actor and critic). The dimensionality
change here (41 -> 43 with the recommended toggle) only changes the first layer's input width
by 2 columns (~5%) -- not large enough on its own to justify widening or deepening the default
net_arch. No `net_arch` change proposed; noting this explicitly rather than leaving it
implicit, per instruction, precisely because the answer is "no architecture change needed,"
not because the question doesn't apply.

**Context noted, not required by this round's task, worth stating for anyone reading this
doc next:** `docs/TRACK_STATUS.md`'s L3 section (checked fresh while re-reading the env code
for Step 2b) now shows the full 2,000,000-step warm-start run **completed** --
`models/l3_executioner_v1.zip` (sha256 `973b2883...`) is the new checkpoint, superseding the
old buggy-physics baseline, with checkpoint identity itself marked RESOLVED there. That
closes this doc's *other* prior blocker (which checkpoint) independently of this round's work.
A new, different open question has taken its place on that track (item (f): whether a
near-parity-with-TWAP validation result is "good enough" to build L2 on top of, vs. training
further) -- a judgment call for whoever owns that decision, not something this design doc
resolves. **This design doc's observation-space and hyperparameter spec above is now final
on its own terms**, but actually starting `FrozenL3Wrapper`/`train_l2.py` implementation
still depends on that separate, still-open call.

## Summary of this round

Step 1: cadence conflict confirmed resolved in the spec (moderate confidence -- bare one-line
patch, no restated rationale), confirmed nothing else in the real repo hardcodes the old
value. Step 2: concrete L2 observation space specified -- `Box(shape=(41,))` base or
`Box(shape=(43,))` recommended (toggle default ON), full index-mapping table, TWAP-deviation
scalar confirmed computable from existing env state with zero new instrumentation, previous-
action inclusion presented as an explicit toggle with a recommendation (paper precedent not
found in-repo, flagged rather than guessed). Step 3: `buffer_size`/`gamma` re-confirmed
unchanged, PROVISIONAL dropped for the cadence-related reason; dimensionality's effect on
network architecture named explicitly (none needed). No `FrozenL3Wrapper`/`train_l2.py` code
written this round.

---

## Correction (2026-08-20, implementation session): `l2_include_prev_action` default flipped to OFF

Per direct correction before implementation began: the "recurrent policy" precedent cited
in Step 2c above (used to justify defaulting the previous-action toggle ON) does not
transfer as cleanly as that section implied -- it applies to genuinely recurrent
architectures, not SAC's plain `MlpPolicy`. The closer analogy in this project's own papers
deliberately excludes action history in favor of pure state-history, and that same paper
flags its own findings as untested for off-policy/SAC-style methods specifically.

**Corrected default: `l2_include_prev_action: bool = False`.** The toggle itself stays --
still a legitimate thing to ablate empirically once L2 training exists to compare against
-- but Step 2c's recommendation and its "SAC's MlpPolicy has no recurrence, unlike L3's
LSTM" reasoning should be read as **an open empirical question, not precedent-backed
guidance**. This session did not independently re-derive a replacement justification for
either default; recording the correction and its stated reason here rather than
constructing a new rationale to fill the gap.

`FrozenL3Wrapper` in `src/envs/wrappers.py` implements the corrected default directly.
