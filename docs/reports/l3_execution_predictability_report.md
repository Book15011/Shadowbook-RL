# Execution Predictability: Frozen L3 vs. Pure TWAP

**Question asked**: TWAP's known real-world weakness is that it is trivially predictable — fixed
schedule, uniform slices — which makes it detectable by adversarial flow in a real market. Is the
trained L3 policy measurably less predictable, and by how much?

**What this measures, and what it explicitly does not** — read before the numbers below, and
before drawing any conclusion from them: this compares a *property* of the two policies
(regularity of their placement pattern) on the same 500 paired val episodes already used
throughout this project's evaluations. It does **not**, and structurally **cannot**, show that
lower predictability is *beneficial*. `LOBExecutionEnv` has no adversarial participants and no
market impact from the agent's own orders — confirmed directly from `src/envs/matching_engine.py`
and its call sites in `src/envs/lob_execution_env.py`, not assumed: `walk_market_fill()` consumes
against a fixed, historically-recorded book snapshot passed in as `prices`/`sizes`, with no
mechanism to feed forward and alter any later tick's book state; the queue-position model's
`v_trade` input comes from `_estimate_trade_volume()`, which reads real historical book-depth
change between two actual recorded ticks, independent of whether the agent has an order resting
there at all (`v_cancel=0` throughout, by explicit design). Every "other participant" in this
simulator is just replayed historical data — never a responsive counterparty, never aware of or
reacting to the agent's own footprint. A predictability gap found here is real evidence about a
property of the two policies. It is not evidence that the gap pays off in this environment, and it
is not treated as such anywhere below.

Test split untouched — val only, matching every other n=500 round in this project (seeds
5,000,000–5,000,499, same paired-seed convention). No retraining, no checkpoint changes.

## 1. Task 1 — descriptive regularity metrics

Methodology: both arms run through the same harness used throughout this project — frozen L3 =
the constant TWAP-passthrough L2 action ([1.0, 0.5]) through `FrozenL3Wrapper` (the same "frozen
L3, unsteered" arm as every prior n=500 evaluation's Arm 2, not a bare unwrapped L3 — there is no
supported way to run L3 without some wrapper, since its own observation space includes the
L2-related dims it was trained with); pure TWAP = `TWAPPolicy(n_slices=10)` on the base env, same
as every prior evaluation's Arm 3. Tick-level actions were captured for the full episode (new
`install_tick_capture()`, factored out of `scripts/replay_episode.py`'s existing, already-verified
`install_capture()` so both share one instrumentation path) and reconstructed into discrete child
orders via that same script's `reconstruct_child_orders()`, imported not reimplemented.

**A real wrinkle, found and corrected before trusting any number**: `reconstruct_child_orders()`
creates one `ChildOrder` per *fill event*, by design, for its own replay-visualization use case
(each fill deserves its own marker on a price chart). A single forced-completion market order that
walks several thin book levels therefore shows up as many same-tick fragments — confirmed
directly: one sampled episode's single slice-end completion produced 13 separate same-tick
fragments as small as 0.001–0.13 units each. Left unaggregated, this inflates order counts and
distorts size/gap distributions asymmetrically between arms, since TWAP's forced slice-end
completions hit this far more often than L3's near-0% `MARKET` usage (confirmed in the raw counts
below). Fixed with a new `aggregate_placement_events()` step that merges same-tick market
fragments into one placement event before computing any metric — a real placement/decision, not a
book-depth artifact, is the correct unit of analysis here.

### Results (pooled across all episodes, and per-episode CoV — see note below on why both are reported)

| metric | arm | n placements/episode | pooled mean | pooled std | pooled CoV | p10 / p50 / p90 | per-episode mean CoV |
|---|---|---|---|---|---|---|---|
| **inter-placement gap (ticks)** | L3 | 22.88 | 55.15 | 240.66 | 4.36 | 1 / 1 / 68 | **2.27** (n=462 episodes) |
| | TWAP | 12.89 | 189.72 | 155.55 | 1 / 299 / 300 | 0.82 | **0.79** (n=500 episodes) |
| **placed order size (units)** | L3 | — | 2.52 | 7.51 | 2.98 | 0.000 / 0.001 / 7.71 | 1.57 (n=488) |
| | TWAP | — | 5.18 | 5.31 | 1.02 | 0.821 / 3.417 / 11.818 | 0.46 (n=500) |
| **price offset from touch (ticks)**, resting placements only | L3 | — | 0.23 | 2.63 | 11.43* | -3 / 0 / 4 | -0.54* (n=279) |
| | TWAP | — | 0.000 | 0.000 | n/a | 0 / 0 / 0 | n/a |

\* CoV (std/mean) is not a well-behaved statistic for offset specifically — it's a *signed*
quantity centered near zero, so a near-zero mean (L3's pooled mean offset is 0.23 ticks, close to
the center of its own [-5,+5] range) makes CoV numerically huge or sign-flipping depending on
which side of zero a given episode's mean happens to land, not a meaningful measure of dispersion.
**Std is the honest number for this row** — L3's offsets spread with std≈2.6 ticks around a
near-zero center; TWAP's is a literal, exact zero, every single placement, every episode (n=4,441
resting placements, std=0.0 to the last decimal printed). This is TWAP's design, not an emergent
property: `TWAPPolicy.act()` hardcodes `offset_idx=5` (offset=0) on every call, confirmed directly
from `scripts/phase2a_sanity_suite.py`.

**Two CoV numbers, two different questions.** Pooled CoV mixes within-episode timing regularity
with between-episode variation (different order sizes, different arrival conditions change the
*pace* an episode needs, inflating pooled spread for reasons that have nothing to do with rhythm).
Per-episode CoV — computed within each episode, then averaged across episodes — is the more
precise answer to "how metronomic is each policy's own rhythm," and it's the one used for the
Task 1→2 decision below.

**Gap-timing CoV: L3 = 2.27, TWAP = 0.79 → clearly DIFFERENT** (L3's own placement rhythm is
~2.9x less regular than TWAP's, by this measure). Combined with the offset finding — an exact,
literal constant for TWAP vs. a real, non-trivial spread for L3 — Task 1 does not show similar
regularity between the two policies. **Task 2 is warranted and was run.**

## 2. Task 2 — direct predictability test

**Model**: `RandomForestClassifier(n_estimators=50, max_depth=8)` — deliberately shallow; the
point is comparative predictability between two policies under an identical model, not the best
achievable predictor for either one alone.

**Train/test split, by episode** (not by tick — ticks within one episode are highly
autocorrelated, so splitting by tick would leak information across the boundary): the first 400
episodes (seeds 5,000,000–5,000,399) are train, the last 100 (seeds 5,000,400–5,000,499) are test,
for both arms, using the same split for a fair comparison.

**Feature set — identical construction for both arms**, computed at tick *t* using only
information available up to and including *t* (no lookahead): `mid_ret_1`/`mid_ret_5` (1- and
5-tick mid-price returns), `spread` and its 5-tick change, `qty_remaining_frac` (own inventory
state), `ticks_elapsed_frac` (own elapsed-time state), `ticks_since_fill_norm` and
`ticks_since_placement_norm` (own recent order-flow state, capped at 200 ticks), and
`cur_order_type` (the action just taken at *t*, an autocorrelation feature). Every feature is
computed from the same `tick_records` schema for both arms — this symmetry is what makes the
comparison fair; see the encoding discussion below for the one place fairness required a
deliberate, non-symmetric choice instead.

**Labels — and why order_type, not the full action, is the primary target.** The label is the
order_type issued at *t+1* (HOLD/LIMIT/MARKET/CANCEL_AND_REPLACE) — the coarsest, most
behaviorally meaningful "what happens next" question, and the primary/headline metric for exactly
one reason: TWAP's own price offset is a **hardcoded constant** (confirmed above, `offset_idx=5`
on every placement, no exceptions). Folding offset into the primary target would make TWAP's
measured "predictability" partly an artifact of a zero-entropy label component baked in by
construction — not a finding about genuine flow predictability, and exactly the kind of encoding
choice that could manufacture the result. Offset and size are still reported as **separate**,
secondary classifiers (conditional on a placement of the relevant kind actually happening at
*t+1*) specifically so the mechanical, by-construction component (TWAP's offset) is visible and
kept apart from the genuine timing signal (order_type), rather than hidden inside one blended
number.

### Results

| target | arm | n train / n test | majority baseline (test) | train acc | **test acc** | gap over baseline |
|---|---|---|---|---|---|---|
| **order_type** (primary) | L3 | 631,592 / 153,109 | 0.5802 | 0.7835 | **0.7831** | +0.2029 |
| | TWAP | 918,461 / 235,007 | 0.9947 | 0.9965 | **0.9965** | +0.0018 |
| offset, given LIMIT/REPLACE (secondary) | L3 | — | 0.2018 | — | 0.2152 | +0.0134 |
| | TWAP | — | **1.0000** | — | **1.0000** | +0.0000 |
| size, given any placement (secondary) | L3 | — | 0.2371 | — | 0.2782 | +0.0412 |
| | TWAP | — | 0.6664 | — | 0.9355 | +0.2691 |

**Headline gap (L3 test_acc − TWAP test_acc, order_type): -0.2134.** TWAP sits at 99.65% — 0.18
points above its own already-near-ceiling majority baseline, i.e. genuinely at ceiling, nothing
meaningfully left to predict beyond "mostly HOLD, on a fixed schedule." L3 sits at 78.31%, a real
20.3 points above its own (much lower, 58.0%) majority baseline — L3's order-flow is *not* pure
noise, a classifier does meaningfully better than guessing the majority class — but L3 leaves a
substantial 21.3-point gap to TWAP's ceiling that never closes.

**The secondary classifiers sharpen this rather than complicate it.** Offset is where the contrast
is starkest: TWAP's is perfectly predictable (100.00%/100.00%) for the mechanical, by-construction
reason stated above — while L3's own offset choice, conditional on placing a LIMIT/REPLACE order,
is barely more predictable than guessing the majority bucket (+1.3 points over baseline; 21.5% test
accuracy against a 5-way discrete choice-ish space) — genuinely close to unpredictable from this
feature set. Size runs the other direction in magnitude but the same direction in conclusion: TWAP's
placed size is highly predictable beyond baseline (+26.9 points — it follows a smooth,
deterministic function of elapsed time and remaining inventory, both in the feature set), while
L3's is only modestly so (+4.1 points). On every one of the three dimensions tested — timing,
offset, and size — TWAP is at or extremely near its own ceiling, and L3 is not.

## 3. What this does, and does not, establish

**Does establish**: on the same 500 real val episodes, using an identical, symmetric, explicitly-
justified feature set and an identical shallow classifier, frozen L3's placement pattern is
measurably and consistently less predictable than pure TWAP's — not a marginal or borderline
effect. The gap is large on the primary timing metric (21.3 points) and larger still on the two
secondary metrics once TWAP's mechanical constant-offset behavior is factored in honestly rather
than blended into one number.

**Does not establish, and this report makes no such claim**: that this predictability gap would
translate into any execution-quality benefit in a real market with actual adversarial participants
who could detect and exploit TWAP's regularity. This simulator has none — no market impact from
the agent's own orders, no counterparty that observes or reacts to either policy's pattern, exactly
as confirmed against the code at the top of this report. A property was measured, cleanly. A payoff
was not, and could not have been, in this environment.

## 4. Limitations

- **Tick-level, lag-based features, not a sequence model.** The classifier sees a small set of
  hand-built recency features, not the full recent history as a sequence (no LSTM/transformer
  over raw ticks). A more sophisticated sequence model might close part of the gap for either arm
  — this measures predictability under one reasonable, simple, explicitly-specified featurization,
  not the ceiling of what any possible model could extract.
- **`RandomForestClassifier` at fixed, shallow hyperparameters** (50 trees, depth 8) — deliberately
  not tuned per-arm, so a tuning pass could in principle move either arm's number, though not
  obviously in a direction that would close a 21-point gap.
- **One checkpoint, one seed, no replication** — same caveat as every other single-checkpoint
  result in this project; not confirmed to hold for a different training seed.
- **CoV is not a well-behaved statistic for the offset metric** (see Section 1's note) — std is
  reported alongside it for exactly this reason, and is the number to trust there.
- Val split only, by design (test split stays unspent, per every prior round in this project).

## Files

- `scripts/analyze_predictability.py` (new) — Task 1 + Task 2, both arms, full pipeline.
- `scripts/replay_episode.py` — extended (not reimplemented): `install_tick_capture()` factored
  out for reuse on a bare env, `ChildOrder.placed_size` added, `install_capture()` now calls the
  factored function internally (behavior-preserving — its own 7 tests pass unchanged).
- `pyproject.toml` — added `scikit-learn==1.9.0` (installed, was not previously a project
  dependency).
- `models/predictability_n500/predictability_result.json` — full numeric output, both tasks.
