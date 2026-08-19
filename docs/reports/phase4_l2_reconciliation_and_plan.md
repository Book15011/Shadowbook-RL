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
