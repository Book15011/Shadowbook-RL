# L2 reward redesign: proposal (design only, not implemented)

**Date:** 2026-08-27. **Status:** design round, no training run, no checkpoint changes, test split untouched.

## Summary

L2 has never had its own reward function. `FrozenL3Wrapper.step()` sums L3's per-tick
`step_reward()` over each 50-tick window and hands that sum to SAC as L2's reward. That reward
was designed for a tick-level executioner choosing order type and price offset; L2 chooses
participation rate and urgency at a ~5s cadence and does not make any of the tick-level choices
most of the components are actually pricing.

Task 1 measured this directly rather than asserting it: **85.6% of L2's net accumulated reward,
and 75.4% of its total signal magnitude, comes from `r_stale` alone** — a component L2 does not
control. Terminal IS, the metric L2 is actually evaluated on, is 6.9% of net reward and 11.6% of
magnitude. This is a real, quantified credit-assignment failure, not a hypothesis.

Task 2 proposes a primary redesign (potential-based mark-to-market IS shaping) plus one
conservative alternative, and explicitly engages with why the project's own TWAP-baseline
variance-reduction reward failed on L3 and whether that failure mode transfers to L2 (partially —
argued below, not asserted).

Task 4's honest read: **this is worth doing, but I'd put modest odds (roughly 15-25%) on it
flipping the sign to a genuine win.** The credit-assignment problem is real; whether it is the
*dominant* cause of the negative, versus "no exploitable structure at this cadence," is not
resolved by Task 1 alone. A distinguishing test is proposed at the end.

---

## Task 1 — Diagnosing the credit-assignment problem concretely

### How the wrapper currently aggregates reward

`FrozenL3Wrapper.step()` (`src/envs/wrappers.py`) runs L3's frozen policy for `n_ticks` (50)
inner ticks per L2 decision. Each inner tick calls `self.env.step(l3_action)`
(`LOBExecutionEnv.step()`), which computes `r = step_reward(...)` — the sum of six components —
and, on the tick the episode terminates, adds `-kappa * IS_total_bps` once. The wrapper does
`agg_reward += r` across the window and returns `agg_reward` as L2's reward for that decision.
Nothing about this aggregation is L2-cadence-aware; it is exactly L3's own tick-level reward,
summed.

The six components (`src/envs/reward.py`):

| Component | Fires | What it prices | Who controls it |
|---|---|---|---|
| `r_slip` | on fill ticks | fill price vs. arrival price | L3's price-offset choice |
| `r_spread` | on maker-fill ticks | spread capture rebate | L3's order-type/price choice |
| `r_inv` | every tick | quadratic inventory-holding cost | mediated by L3, but responsive to L2's pacing |
| `r_queue` | on cancel/replace/market ticks only | discarded queue position | L3's cancel/replace timing |
| `r_stale` | every tick while resting unfilled | accumulated staleness | L3's resting/replace timing |
| `r_placement_stale` | every tick while resting (eta_replace=0 in production, so inert) | — | — |
| terminal IS | once, episode end | full-episode execution quality | the metric L2 should optimize |

L2 chooses `participation_rate_multiplier` (scales the schedule target L3 tries to hit) and
`urgency` (an L3 observation input). It does not choose order type, price offset, or cancel
timing at any tick — those remain 100% L3's own decisions, only *nudged* by L2's steering.

### Measurement

Instrumented `src.envs.lob_execution_env`'s own `step_reward`/`compute_implementation_shortfall`
name bindings (monkeypatched at the point they're imported into that module — `from X import Y`
binds locally, so patching `X.Y` would not intercept the call site; the local binding must be
patched instead). Each patched function calls the original unchanged and returns its result
unmodified; components are independently recomputed from the captured call arguments using the
exact formulas in `reward.py`, and the recomputed sum is asserted against the original scalar
return on every call. **0 mismatches across 154,192 ticks** — the breakdown below is not a
transcription artifact.

Real trained L2 checkpoint (`models/l2_strategist_v1.zip` + VecNormalize), deterministic,
100 real val episodes (paired seeds 5,000,000–5,000,099), frozen L3, numeric-format data:

```
SIGNED TOTAL (L2's actual accumulated reward)              MAGNITUDE (sum of |component|)
  r_stale             -1634.97   85.59%  (-16.35/ep)          r_stale            75.37%
  terminal_is_reward   -131.85    6.90%   (-1.32/ep)          terminal_is_reward 11.56%
  r_inv                 -89.62    4.69%   (-0.90/ep)          r_slip              6.88%
  r_queue                -42.68    2.23%   (-0.43/ep)          r_inv               4.13%
  r_slip                 -12.71    0.67%   (-0.13/ep)          r_queue             1.97%
  r_spread                  1.57   -0.08%                       r_spread            0.10%
  r_placement_stale         0.00    0.00%  (inert, eta_replace=0)  r_placement_stale  0.00%
```

### Answer

L2's learning signal is dominated by a component its actions cannot directly influence, and it
is dominated by a wide margin — not "somewhat more than terminal IS," but **12x the magnitude of
the one component that reflects what L2 is actually graded on.** `r_stale` alone outweighs
everything else combined. This is a scale problem as much as a controllability problem:
`zeta=0.06` was calibrated (per `reward.py`'s own derivation comment) so that ~200 idle ticks
costs about as much as one `CANCEL_AND_REPLACE` — a sensible calibration for L3's own tick-level
choice between "replace now" and "wait." Summed across an entire ~3,000-tick episode inside a
window that also sums 50 of these at a time, that same calibration compounds into a per-episode
total that swamps everything else in the aggregate SAC actually trains against. `r_queue` (2.2%)
is a distant second controllable-adjacent term; `r_spread`/`r_slip` are negligible in the mean
and modest in magnitude. Terminal IS — the only component that is unambiguously about the thing
L2 should be optimizing — is a rounding error next to `r_stale` in the current aggregation.

---

## Task 2 — Design

### What "L2 controls" actually means here

None of the six existing components map cleanly onto L2's own decision. The closest is `r_inv`:
it is a direct function of `qty_remaining`, which L2's participation-rate steering most directly
shapes (a higher multiplier should shrink `qty_remaining` faster, other things equal) — but its
per-tick timing is still mediated by whether L3 actually executes on that steered target.
`r_queue`/`r_spread`/`r_stale`/`r_placement_stale` are tick-level consequences of L3's own
order-management choices with no clean L2 attribution at all, and `r_stale` specifically is now
measured to dominate the signal by an order of magnitude while being the least L2-attributable
term of the six. **None of the six should be kept as-is.** `r_inv`'s *shape* (a pacing-sensitive
cost) is worth preserving in spirit; its literal per-tick sum, mediated by L3's cadence, is not.

### Primary design: potential-based mark-to-market IS shaping

Terminal IS is what L2 should optimize, but a single signal across ~60 decisions is sparse.
Rather than borrow L3's tick-level components to fill that gap (which is how the current,
broken aggregation happened in the first place), decompose the *same terminal quantity* into
dense per-decision increments using potential-based reward shaping (Ng, Harada & Russell 1999):
add `F(t) = Φ(t) - Φ(t-1)` at each L2 decision, where `Φ` is a running, mark-to-market estimate
of implementation shortfall using only information available at that decision boundary:

```
Φ(t) = -kappa * [ executed_frac(t) * realized_avg_slip_bps(t)
                  + (1 - executed_frac(t)) * side * (mid_price(t) - arrival_price) / arrival_price * 1e4 ]
```

This is exactly `compute_implementation_shortfall()`'s own formula (`fill_ratio * IS_exec_bps +
(1-fill_ratio) * IS_opp_bps`, fees folded in separately per-window), evaluated at the *current*
tick using the *current* mid-price as the mark for whatever hasn't filled yet, instead of waiting
for the terminal tick. `Φ(0) ≈ 0` (nothing executed, mid ≈ arrival by definition), and
`Φ(T) ≈` the real terminal IS at the real terminal tick, whatever tick that turns out to be — so
`Σ F(t)` over the episode telescopes to (very nearly) the same terminal IS that
`info["implementation_shortfall"]` already reports, while paying out dense, per-window credit
along the way instead of one lump sum. Fee contribution and any true realized-fill slippage
within the window are exact (computed from `info["fills_this_step"]`, accumulated across the
window's inner ticks the same way `window_obs` already is); only the *unfilled remainder's* mark
is an estimate, and it is an unbiased one at each point in time it's taken.

**Why this doesn't inherit the TWAP-baseline reward's specific failure mode.** That change
subtracted a *separate reference policy's* terminal outcome (a TWAP shadow, computed once in
`reset()` over a fixed window) from the real agent's terminal IS. The project's own report on it
(`docs/reports/l3_twap_baseline_reward.md`) found the trained policy converged toward
faster, more aggressive completion (mean episode length roughly halved, fill_ratio up,
early-termination rate up) with an occasional costly tail — framed there as "a plausible, not
proven" behavioral story. My reading of the mechanism underneath that story: the real agent's own
IS is only ever exposed to market drift up to *its own* termination tick, while the shadow is
exposed to drift over its own, independently-fixed window — an agent that learns to finish faster
reduces its own drift exposure relative to the (unmoving) comparison point, which is a real
incentive to rush, separate from genuine execution skill. **This does plausibly apply to L2 too**,
not just L3: L2's participation multiplier can push L3 toward faster completion (a multiplier near
2.0 raises the schedule target L3 chases, which can pull qty_remaining to zero early) through the
same causal chain, just one step more indirect than L3's own direct order-type choice. That is
exactly why I am not proposing TWAP-baseline subtraction here. The potential-based design above
has no separate reference trajectory at all — `Φ(t)` is a function of the *real* agent's own
accumulated state and the *real* current price, evaluated only at the real agent's own actual
decision points, whatever those turn out to be. There is no second, independently-timed window
for the agent's own timing choices to become mismatched against. This is a structurally different
kind of change from the one that failed, not a re-labeling of it — stated so it doesn't get
mistaken for a second attempt at the same idea.

**Reward scale and variance.** IS variance is dominated by market drift the critic cannot predict
— that finding motivated the (failed) TWAP-baseline attempt and still holds; this design doesn't
solve it, and isn't trying to. What it does instead: removes the ~85%-of-signal `r_stale` noise
floor entirely, so whatever drift-driven variance remains in the terminal-IS-derived signal is at
least the *dominant* source of variance in what SAC sees, not one of several comparably-sized
noise sources stacked on top of it. That's a meaningfully different problem than what the
TWAP-baseline change was aimed at (further *reducing* an already-dominant, clean signal's
variance) — this is about first making the signal dominant at all.

**Schedule adherence.** L2's neutral action (`multiplier=1.0`) already *is* on-schedule TWAP
pacing — participation_rate_multiplier is defined relative to the TWAP baseline, not as an
absolute target. Rewarding adherence to that baseline directly would train L2 toward always
outputting the neutral action, which collapses to TWAP-passthrough by construction — not a useful
signal, and not obviously "optimal" pacing so much as a convenient default with no evidence it's
actually the best achievable schedule. I am not adding a schedule-deviation reward term for this
reason: the terminal-IS-derived shaping above already implicitly rewards good pacing (inventory
left unfilled when the market has moved unfavorably shows up directly in `Φ`'s opportunity term)
without hard-coding TWAP as the target to imitate.

### Alternative: minimal-diff (drop, don't replace)

If the primary design's implementation risk (a new, more involved reward function, a telescoping
property that needs verifying) is more than wanted for a first probe: keep `step_reward()`
untouched, but have the wrapper aggregate only `r_inv` (retained, as the one component with a
plausible, if mediated, L2 attribution) plus the terminal IS term (with `kappa` raised, since it
currently has to compete against `r_stale`'s scale and wouldn't need to once `r_stale`/`r_queue`/
`r_spread`/`r_placement_stale` are dropped from L2's own sum). This is a much smaller code change
— filtering, not adding — and removes the measured 85%+11%+2%+0% ≈ 98%+ of magnitude that isn't
attributable to L2 at all, without introducing a new shaping mechanism to validate. It does not
solve sparse credit assignment as elegantly (still one terminal bump per episode for the IS
component specifically); `r_inv`'s dense per-tick presence provides *some* denser signal, but one
step further removed from the actual objective than the primary design's telescoping shaping is.
Worth keeping as the fallback if the primary design's added complexity turns out not to be
worth it, not as a first choice.

---

## Task 3 — Implementation plan and expected cost

**New file:** `src/envs/l2_reward.py`, mirroring `reward.py`'s own style — a pure function,
e.g. `l2_window_reward(prev_state, window_fills, window_end_info, weights) -> float`, taking
enough of the pre/post-window state (`qty_remaining` before and after, `arrival_price`, `side`,
`mid_price` at the window's start and end tick, the window's own fills from
`info["fills_this_step"]` accumulated across the inner loop, `fee_bps_per_fill`) to compute
`Φ(t) - Φ(t-1)` plus the window's realized fee contribution. Pure and unit-testable the same way
`step_reward()` already is, independent of the env/wrapper machinery.

**Changed:** `FrozenL3Wrapper.step()` (`src/envs/wrappers.py`) — currently `agg_reward += r`
inside the inner-tick loop; needs to also accumulate `info["fills_this_step"]` across the window
(the same pattern `window_obs` already uses) and, after the loop, call `l2_window_reward(...)`
instead of returning the summed raw `r`. Gate behind a constructor parameter (e.g.
`l2_reward_mode: Literal["l3_passthrough", "potential_is_shaping"] = "l3_passthrough"`) so the
existing behavior stays the default and every current test/script that constructs
`FrozenL3Wrapper` is unaffected unless it opts in — same convention this project has used for
every other reward-affecting toggle (`zeta`, `eta_replace`, `subtract_twap_baseline`).

**Untouched:** `src/envs/reward.py`, `src/envs/lob_execution_env.py`'s own reward wiring, L3's
training entirely (`train_l3.py`, `l3_frozen_backup/*` — no risk to the frozen checkpoint this
whole line of work depends on), and L2's observation space, action space, and network
architecture. This is a reward-aggregation change only.

**Tests:**
- Unit tests for `l2_window_reward()` against hand-computed fixtures (`tests/test_l2_reward.py`,
  same fixture discipline as `tests/test_reward.py`).
- A **telescoping test**: sum `l2_window_reward()` across every window of a real (or fixture)
  episode and assert it matches `compute_implementation_shortfall()`'s own terminal
  `is_total_bps` to a tight tolerance — this is the core correctness property of the whole design
  and should be checked directly, not just argued in prose.
- Seed-equivalence test: with `l2_reward_mode="l3_passthrough"` (the default), confirm
  byte-identical behavior to before this change — this only ever touches the *value* returned as
  reward, never action selection, fills, or physics, so nothing about L3's own RNG draw sequence
  or the env's matching-engine behavior should be able to change either way.
- A mechanics smoke test of `train_l2.py` accepting the new flag and completing a short run
  without crashing (no real training commitment, same status as every other round's shakedown).

**Comparability, flagged explicitly per instruction:** a reward change means
`models/l2_strategist_v1.zip`'s own training-time numbers (loss curves, `ep_rew_mean`, the
in-training `ValISEvalCallback` values) are not comparable to whatever a reward-redesigned run
produces — different objective scale, different signal composition, nothing about "did training
get better/worse" can be read by comparing raw reward magnitudes across the two. The only
comparison that stays valid is against **TWAP-passthrough**, which is reward-independent by
construction (it never touches L2's reward at all, only the terminal `IS_total_bps`/`fill_ratio`
the eval harness measures) — every existing n=500/diagnostic/bucketed result stays a valid
comparison point for a reward-redesigned checkpoint's own eval numbers, even though the training
process that produced it isn't comparable to the current checkpoint's own training process.

---

## Task 4 — Honest expected value

**Odds this moves the number (beats TWAP-passthrough with both tests agreeing, p<0.05): roughly
15-25%.** Reasoning, not a bare number:

**For "wrong reward, worth fixing":** Task 1's measurement is not a hunch — 85.6% of the training
signal is a component L2 cannot influence, outweighing the only component that reflects L2's
actual objective by 12x in the mean and by a similar margin in magnitude. A value function trying
to predict returns dominated by an uncontrollable term has a genuinely harder learning problem,
independent of whether useful structure exists underneath it. The observed training dynamics are
at least consistent with this: `ent_coef` climbed continuously across the full 2,000,000-step run
(0.0054 -> 0.132) without visibly plateauing, and actor/critic losses grew substantially rather
than settling — both patterns fit a critic that never found a clean, confident signal to converge
around, which a dominant noise term would produce regardless of whether real structure exists.

**Against ("no exploitable structure" more likely):** three points, not one. First, frozen L3
alone — the tick-level executioner, with a reward built specifically for tick-level decisions and
20,000,000 training steps behind it — only ties TWAP (0.994 vs. 0.889). If the tier with the most
granular control and the best-matched reward can't beat a fixed schedule, it's not obviously more
likely that a coarser layer sitting on top of it, choosing from only two scalar knobs every 5s,
would find exploitable edge the finer-grained tier couldn't. Second, the volatility-stratified
result (calm/moderate/high, all near-zero, no trend) is a remarkably *consistent* null across
genuinely different market regimes — if bad reward shaping were merely obscuring real structure,
I'd have some expectation of an inconsistent, noisy pattern across regimes rather than the same
flat null everywhere; consistency is weak evidence for "nothing there" over "signal buried
uniformly everywhere," though it doesn't rule the latter out. Third, even in-sample (train,
2,000,000 steps of direct exposure), L2 never reached significance against TWAP-passthrough on
both tests together (Wilcoxon p=0.335 in the original full-train run) — the policy that had every
advantage this project could currently give it (maximum data exposure, memorization included)
still didn't produce a robust edge, under the very reward this proposal argues is broken. That
last point cuts both ways honestly: it's consistent with "reward too noisy to learn from even
with unlimited exposure," but it's at least as consistent with "nothing to learn."

**Which is more likely:** I lean toward "no exploitable structure" as the larger piece, with
"wrong reward" as a real but partial contributor — hence odds closer to 1-in-5 than 1-in-2 that a
redesign alone flips the sign. The reward problem is real and worth fixing regardless of whether
it turns out to be decisive, since a cleaner signal is table stakes for trusting any future result
either way; I would not skip this fix even at these odds.

**What would distinguish them:** train under the redesigned reward (a real run, next round, not
this one), then run the *exact same diagnostic battery* already built this round — val n=500,
unrestricted train n=500, and the three volatility strata — before touching anything else. Two
readable outcomes: (1) training dynamics visibly clean up (ent_coef converges, losses stabilize)
**and** the diagnostic battery still comes back null/negative -> strong evidence for "no
exploitable structure," the reward was real but not the binding constraint. (2) A genuine,
robust positive appears (ideally replicating across at least val and unrestricted train, not just
one) -> confirms the reward was the dominant issue. A murkier result (cleaner training, still
negative-but-smaller effect) would say the reward was a real, partial contributor without being
the whole story — a plausible third outcome, not just the two clean ones above.
