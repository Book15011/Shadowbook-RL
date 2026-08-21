# Cross-track status

Shared status file for the concurrent L1/L2/L3 work sessions on this repo.
Each session owns and updates only its own section -- merge on conflict,
never overwrite another track's section.

## L1 -- Macro Analyst
Last updated: 2026-08-19 19:20 HKT
State: L1MacroAnalyst (architecture_spec.md Section 1.2: MacroRiskContext
schema, cache/throttle/fail-closed maybe_refresh()) and a minimal
orchestrator_graph.macro_tick() (Section 4.3, proves the L1-cache ->
observation idx 17/18 path via the Phase 3 env stub hooks at the spec's
L1_EVERY_N_TICKS=600 cadence) are built, unit-tested (11/11 passing,
requests.post mocked throughout -- no real Ollama call anywhere yet), and
committed locally as bb47856 (not pushed). Separately, the Ollama systemd
unit had a wrong ExecStart path (/bin/ollama vs the real
/usr/local/bin/ollama, crash-looping); fixed, reloaded, and confirmed
serving (`curl localhost:11434/api/tags` -> `{"models":[]}`), with no GPU
memory change (539 MiB before/after) -- infra fix only, not a repo commit.
No l1_features.py / feature_summary-construction pipeline exists yet (not
started). No real-model (Ollama) validation has been attempted -- explicitly
paused per instruction, not blocked on anything technical.
Files owned/in-progress: src/agents/l1_macro_analyst.py,
src/agents/orchestrator_graph.py, tests/test_l1_macro_analyst.py,
tests/test_orchestrator_graph.py (all committed, bb47856). No file
currently in-progress/uncommitted on this track.
Blocking/open questions: 14B vs 32B model choice still open -- only
qwen2.5:14b-instruct-q4_K_M is pulled on disk (8.4GB); the spec's default,
qwen2.5:32b-instruct-q4_K_M (~20-21GB VRAM), is not pulled and would need
a fresh download plus confirmed GPU headroom before use. Real-model
validation is also gated on the other session's live training settling
(4090 has ~24GB total; do not want to contend with an active run).
Next planned step: build the feature_summary construction pipeline
(src/data/l1_features.py or similar) from data/raw_l1's already-collected
klines/funding/open_interest into the rolling numeric summary
L1MacroAnalyst.maybe_refresh() expects -- this can proceed independently
of the model-size/GPU-headroom decision above. Real-model (Ollama)
validation of L1MacroAnalyst stays paused until that decision + headroom
are both confirmed.

## L2 -- Strategist
Last updated: 2026-08-21 22:42 HKT
State: Consolidation round -- no new features, no training. DONE: FrozenL3Wrapper
(src/envs/wrappers.py), train_l2.py, tests/test_wrappers.py (20/20 passing) are all built,
committed locally, and smoke-tested (200-step mechanics-only run, no shape/interface
errors between the wrapper and SAC). Design doc (docs/reports/
phase4_l2_reconciliation_and_plan.md) reorganized this round -- a "CURRENT STATE" section
now sits at the top and reads standalone, with the full decision trail (including reasoning
later corrected or superseded) preserved below it rather than interleaved.
Checkpoint-citation correction (thanks to L3's own proactive note on this section, folded
in here): the smoke test's earlier reported sha256 973b2883... was genuinely live-computed
and accurate at the time (20:49 HKT that day) -- it went stale ~2 hours later when a
separate L3-track probe run's own final save overwrote models/l3_executioner_v1.zip
(confirmed via file mtime and TRACK_STATUS's own incident writeup). Re-verified directly
this round: the canonical path now holds sha256 27afa91e... (L3's numerically-verified
step-2,000,000 stand-in). Corrected everywhere this track owns it (design doc, this
section); nothing in wrappers.py/train_l2.py/test_wrappers.py needed correction since none
of them ever hardcoded a checksum -- all checked, all compute/reference by path, not by
hash. The smoke test's own validity is unaffected: it tested wiring, not policy quality.
Two items flagged last round as deliberately deferred are now decided (not implemented):
(a) VecNormalize on L2's OWN observation/reward space (distinct from the frozen L3 stats
already applied to L3's inputs) -- recommend adding before a real run, does not block
further work; L2's 41-dim obs mixes window-averaged features (lower variance) with
instantaneous ones (full variance) plus a genuinely non-stationary new scalar
(schedule_deviation), arguably a stronger case for adaptive normalization than L3's own
already-homogeneous vector. (b) A held-out eval callback analogous to L3's
ValISEvalCallback -- recommend building one, and unlike (a) this DOES block a real run: the
wrapper's mechanism is structurally expensive per SAC step (a gradient update on every
single L2 decision, each costing up to 50 real L3-predict+env.step calls, single env, no
parallelism), so a real 2,000,000-step run could plausibly take multiple days of wall-clock
time with zero visibility into whether it's working without one. Full reasoning for both in
the design doc's CURRENT STATE section.
Files owned/in-progress: none uncommitted. src/envs/wrappers.py, src/train/train_l2.py,
tests/test_wrappers.py all committed (1603c61, 87d7ba7). docs/reports/
phase4_l2_reconciliation_and_plan.md reorganized and committed this round.
Blocking/open questions: L3's matched A/B training runs, which will determine the final
frozen L3 checkpoint -- this is now the ONLY thing blocking a real L2 training launch. When
the A/B result lands, only --l3-checkpoint/--l3-vecnormalize need to change on L2's side;
everything else (observation space, action-space transform, hyperparameters, wrapper
mechanics) is independent of which specific checkpoint wins. Separately, not blocking: (a)
and (b) above are recommended before a REAL full-budget run specifically, not before the
A/B result lands -- could be built in parallel with waiting, if useful.
Next planned step: awaiting the L3 A/B result. Candidate parallel work if useful before
then: build the VecNormalize wrapping and/or the simplified held-out eval callback decided
above (neither implemented yet, both are "decided, not built"). No training launch planned
until both the checkpoint question resolves and, ideally, (b) exists.

## L3 / Env-Physics
Last updated: 2026-08-21 20:40 HKT
State: Two more tasks landed since the last check-in, both no-GPU/no-training. Pointer
update again -- see docs/reports/l3_replace_value_probe.md (Task 1) and
docs/reports/l3_twap_baseline_reward.md (Task 2, new file) for full detail.

TASK 1 (v1 re-evaluated at n=500, committed 332d1b9): checkpoint verified before running
anything -- models/l3_executioner_v1.zip is still the step-2,000,000 stand-in
(27afa91e...), NOT true v1 (973b2883..., unrecoverable), consistent with every check this
session. Result: v1's own headline numbers barely moved from n=50 (IS 1.245->1.2607, fill
0.918->0.892), but the gap vs TWAP grew ~6x (0.063bps->0.372bps) and now clears p<0.05 by
the paired t-test (p=0.0327) though NOT by Wilcoxon (p=0.115) -- stated plainly rather than
picking whichever test agrees. Three-way n=500 point-estimate ordering: TWAP (0.889) <
best-B (1.103) < v1 (1.261) -- v1 worst of the three, a more robust version of the
milestone report's own "near-parity, if anything slightly worse" framing, not a reversal
like best-B's sign flip was. Read as meaningfully suggestive, not fully unanimous, evidence
that the RL policy underperforms simple baselines here.

TASK 2 (TWAP-baseline variance-reduction reward, implemented + tested, NOT trained,
committed 1b442f9 -- separately from Task 1 and from the still-unconfirmed r_queue
direction-inversion, see Files owned/in-progress below): terminal reward becomes
-kappa*(IS_total_bps - twap_shadow_IS_total_bps), gated behind
RewardWeights.subtract_twap_baseline (default False, same opt-in convention as
zeta/eta_replace). Explicitly a baseline subtraction, not an objective change -- the
subtracted quantity never observes the real episode's policy, so it shifts reward scale,
not the argmax. Design decision (asked for, not picked silently): computed fresh in
reset(), NOT cached -- caching would not help training resets regardless of choice, since
training draws from ~349M possible windows with ~0% real repeat rate; caching only helps
periodic eval (same fixed seeds reused across firings), a separate, not-yet-built
optimization. Measured cost on REAL market data, not guessed: +48ms/reset (2.4% of
reset()'s own ~2027ms baseline, dominated by day-load I/O), ~0.1% of a full episode's
wall-clock budget at v1's measured throughput -- much cheaper than an initial
back-of-envelope worry (~2x) suggested; measuring instead of assuming caught that worry
being wrong. 4 new tests (tests/test_twap_baseline_reward.py) verify the subtraction
arithmetic, the flag's inertness, and that info["implementation_shortfall"] never changes
-- the key integration test (a policy matching TWAP exactly must get ~0 terminal
contribution) caught two real implementation bugs before either reached committed code
(missing discrete-SIZE_FRACTIONS rounding, and a decision/evolution ordering mismatch) --
concrete evidence the test design did its job, not just satisfied a checklist.

Old narrative retained below unchanged for context on the checkpoint-overwrite incident,
the direction-inversion probe, and the prior REPLACE-value follow-up:

State: Two follow-up tasks on top of the direction-inversion probe below (still the most
recent full narrative; this entry is a pointer + headline update, not a restatement --
see docs/reports/l3_replace_value_probe.md for the complete account of both).

TASK 1 (safety fix, committed c803249, separate from Task 2): train_l3.py's final-save
landmine flagged in the entry below is FIXED -- the final save now refuses to overwrite
an existing models/l3_executioner_v1.zip/l3_vecnormalize.pkl unless --overwrite-canonical
is passed explicitly, redirecting to a run-tagged path otherwise. IMPORTANT scope note
requested explicitly, recorded here per instruction: while building this fix, the SAME
hardcoded-path bug was found to ALSO affect CheckpointCallback's periodic saves
(name_prefix="l3_ppo" was shared across every run) -- and it had ALREADY silently
overwritten data before this fix landed, independently of the final-save incident already
documented below. Specifically: v1's own periodic checkpoints at steps 250,000 and
500,000 (from its 2,000,000-step run) were silently overwritten by the direction-inversion
probe's own periodic checkpoints at those same step numbers (confirmed via file
timestamps -- the surviving 250k/500k files are dated to the probe's run window, not
v1's). Consequence: v1's intermediate training trajectory below step 750,000 is no
longer recoverable from checkpoints (750k onward still exist and were unaffected). This
did not affect anything already reported -- v1's FINAL numbers came from the step-2M
checkpoint (recovered separately, see below) and the milestone report's own eval, neither
of which depended on the 250k/500k files -- but the early trajectory is gone, for the
record. Both this and the final-save fix are now namespaced by --run-name (auto-generated
timestamp if not given), so this cannot recur for either save path going forward. 4 new
tests in tests/test_train_l3.py cover the guard logic directly (tmp_path-based, no
GPU/training needed).

TASK 2 (adequate-power re-test of the direction-inversion probe's own follow-up question,
i.e. "is CANCEL_AND_REPLACE actually valuable" -- heuristic simulation only, no training,
no GPU): the original scripted-heuristic probe (docs/reports/l3_replace_value_probe.md)
concluded REPLACE shows no value at n=50, but n=50 had only 14.7% power to detect the
effect it actually observed (best-B nominally beat TWAP by 0.48bps). Re-ran best-B vs TWAP
ONLY (pre-registered, no re-sweep) at n=500 (~83% power for that effect size). RESULT: the
sign flipped -- best-B is now numerically WORSE than TWAP (+0.214bps, not -0.482bps),
still not significant (p=0.101), and the practical effect size is small either way
(~4% of TWAP's own std). This is a stronger, not weaker, confirmation of "REPLACE isn't
valuable here" -- the n=50 number that made REPLACE look promising was a selection
artifact from screening 18 configs, and did not survive proper power. Separately, and
stated plainly per instruction: at their respective sample sizes, BOTH TWAP (0.889 at
n=500) and best-B (1.103 at n=500) come in numerically better than v1's trained RL
policy's own reported IS_total_bps (1.245 at n=50) -- not a formal paired test against
v1 (no n=500 v1 data exists to pair against), but a real, plainly-stated finding about
the RL setup itself, not buried as a footnote. Also confirmed independently before running
anything: no PASSIVE-family config can reach comparable fill to TWAP/B (40.4% at offset=0
is a structural ceiling, not a sweep gap -- crossing offsets all collapse to the same
single-tick-depth-limited outcome regardless of how aggressive), so no PASSIVE arm was
re-tested at n=500; substituting it would have repeated a known apples-to-oranges
comparison at higher n rather than fixing anything.

Both tasks committed locally (Task 1: c803249; Task 2: pending commit alongside this
update), not pushed.

PRIOR ENTRY BELOW, unchanged, for full context on the checkpoint-overwrite incident and
the direction-inversion probe itself:

State: Ran a bounded probe testing whether inverting the r_queue REPLACE/MARKET
queue-cost direction increases CANCEL_AND_REPLACE usage (near-0% in v1 -- see prior
entries). Result: probe does NOT confirm the hypothesis. Also: an operational mistake
during this probe overwrote the v1 checkpoint in the working slot -- recovered, but
disclosing in full below since it affects what L2 (and anyone else) should trust right
now.

INCIDENT, disclosed in full: train_l3.py's final save always writes to the same
hardcoded path (models/l3_executioner_v1.zip / l3_vecnormalize.pkl) regardless of
whether the run is a full commitment or a bounded probe. Every prior probe this session
warm-started FROM a checkpoint without ever letting the run reach its own final save
(always stopped/killed first) -- this is the first one that was deliberately allowed to
run to completion, and its completion silently overwrote v1 (sha256 973b2883...) with
the probe's own output. This should have been anticipated and backed up beforehand,
the same way models/baseline_20M_backup/ protects the original 20M-step baseline -- it
was not, which is a real process gap, not a tooling failure. Caught immediately once
the probe's completion log showed a new checksum instead of 973b2883... (which is what
prompted this disclosure, not a routine status update).

RECOVERY: the true 973b2883... file is not byte-recoverable -- it was never backed up
and the working slot is the only place it lived. Best available fallback: v1's own
CheckpointCallback had already saved a periodic checkpoint at step 2,000,000 (10:07
HKT, 4 minutes before v1's actual final save at step 2,002,944) to
models/l3_checkpoints/l3_ppo_2000000_steps.zip -- essentially the same policy, short by
2,944 steps of further training. Restored this into the working slot
(models/l3_executioner_v1.zip, now sha256 27afa91e...). Verified, not assumed: ran the
exact same reproduction/eval script used for the milestone report against it and got
IS_total_bps=1.2450, fill_ratio=0.9180 -- bit-for-bit identical to the numbers
973b2883... itself printed live during training at its own step=2,000,000 eval firing
(this also resolves a discrepancy flagged in the milestone report, where a same-session
reproduction against the truly-final 973b2883... checkpoint gave slightly different
numbers, IS=1.265/fill=0.976 -- now explained: that gap was the extra 2,944 steps of
training between this checkpoint and the true final save, not a bug or nondeterminism).
The probe's own output was preserved under its own name before being overwritten
further (models/l3_executioner_v1_replace_direction_probe.zip /
l3_vecnormalize_replace_direction_probe.pkl) -- no data was lost, only the exact
973b2883... bytes are gone. A permanent, git-tracked backup of the restored near-v1
checkpoint now exists at models/v1_near_backup_step2M/ (committed as f848bba) so this
specific failure mode cannot recur for THIS checkpoint -- the same discipline should be
applied automatically before any future run that might reach its own final save,
without needing to be told again.

THE PROBE ITSELF, Steps 1-2 (src/envs/reward.py, tests/test_reward.py, both still
UNCOMMITTED pending the result below): inverted the queue-weighted term in both
canceled_via_market and canceled_via_replace branches of step_reward()'s r_queue block,
from gamma*(queue_ahead/queue_at_level) to gamma*(1 - queue_ahead/queue_at_level) --
charges more for discarding genuinely earned queue priority (q_ahead near 0 via real
trade-through volume), less for a fresh order or one behind a deep/stalled level.
Module docstring and RewardWeights.eta_replace's loophole derivation updated in place,
old reasoning kept verbatim and marked SUPERSEDED rather than deleted. Loophole
re-derivation (eta_replace stays 0.0/inert for this probe either way): the specific,
previously-flagged off-book exploit (price outside the visible top-20 book giving
EXACTLY zero replace-cost) is CLOSED -- inverted, in fact, to the maximum charge
-gamma, not the minimum. A weaker residual remains and is stated plainly, not glossed
over: a fresh, minimal-size order (size_frac has a 20% floor, never exactly 0) behind a
level with substantial real ambient depth can still approach near-zero cost -- but that
depends on actual market depth at the chosen tick, is not a free/guaranteed
policy-controlled knob the way off-book was, and only approaches zero asymptotically,
never hits it exactly. 17 reward tests pass (3 updated to the new numbers, 2 new tests
added making the near-front/close-to-gamma and fresh-deep-level/close-to-zero cases
explicit, 1 renamed to reflect the inverted off-book case -- see tests/test_reward.py).
Full env/matching-engine suite (30 tests) also re-verified passing; 4 unrelated,
pre-existing failures in test_bulk_backfill.py/test_l2_capture.py confirmed untouched by
this change (no reward.py import, network/order-book-resync issues unrelated to reward
shaping).

Step 3 (bounded probe): warm-started from the checksum-verified v1 (973b2883..., verified
immediately before launch -- this was correct; the overwrite happened at the END of the
run, not the start), same real reward config as v1 (RewardWeights() defaults, no CLI
overrides), n_envs=8, --total-timesteps 500000 (same order of magnitude as prior probes
this session). Logged to logs/l3_train_replace_direction_probe.log (new, distinct path).
Ran slower than v1's own run (~200-220 fps vs. v1's 350-365 -- not investigated further,
did not affect the probe's validity, just took longer wall-clock, ~42 min instead of the
~25-30 min originally estimated). Completed cleanly, no crash, no OOM (memory returned to
baseline afterward).

Step 4 (eval, reused the existing script unmodified in logic, only parameterized with a
--model-path/--vecnorm-path CLI flag to point at either checkpoint): ran the identical
50-paired-seed reproduction against BOTH the restored near-v1 checkpoint (clean baseline,
numbers above) and the probe's own checkpoint, for a same-methodology, same-script,
apples-to-apples comparison (deliberately not relying on the training log's own printed
numbers for either arm, given the discrepancy explained above):

  metric                near-v1 (baseline)   probe (500k more, inverted formula)
  CANCEL_AND_REPLACE %  0.298% (263/88141)    0.336% (312/92810)
  MARKET %              0.01%                 0.06%
  HOLD %                47.64%                46.70%
  LIMIT %               52.05%                52.91%
  IS_total_bps           1.245                 1.829
  fill_ratio              0.918                 0.942
  vs TWAP (paired t)     t=0.13 p=0.90         t=1.12 p=0.27
  vs TWAP (Wilcoxon)     W=572 p=0.53          W=487 p=0.15

Step 5, reported honestly per instruction -- NOT a confirmation: CANCEL_AND_REPLACE usage
moved from 0.298% to 0.336%, a change too small to trust. A two-proportion z-test on the
raw counts (263/88141 vs 312/92810) gives z=1.43, p=0.15 -- not significant even under a
naive test that ignores within-episode/within-policy autocorrelation, which would only
inflate the true variance further and make this LESS significant, not more. This bounded
probe does not show the direction-inversion unlocking meaningful REPLACE usage. IS_total_bps
also got numerically worse (1.245->1.829) while fill_ratio improved slightly (0.918->0.942)
-- neither reaches significance vs TWAP either, so this is not read as a real regression,
but it is also not a case for "this direction is obviously fine, just needs more budget."
Two honest, undecided readings, not a recommendation: (a) queue-cost direction was not
the actual binding constraint on REPLACE usage -- something else (a strongly converged
HOLD/LIMIT habit from the full 2M-step run, or genuinely low value of ever using REPLACE
given how well passive LIMIT already performs) may dominate regardless of price signal;
(b) 500k steps of continued fine-tuning from an already-deeply-converged policy may
simply be too short a window to reveal a shift even if the corrected incentive matters at
longer horizons -- v1's own split needed the full 2M-step run to show its effect, not a
short probe, so this would not be an unprecedented pattern.

Files owned/in-progress: src/envs/lob_execution_env.py (uncommitted, unchanged in
substance -- staleness/eta_replace round only, not touched by this probe per the hard
boundary), src/envs/reward.py + tests/test_reward.py (uncommitted, the direction
inversion above -- NOT committed pending a decision on the result), src/train/train_l3.py
(uncommitted, same TypeError reason as before). models/l3_executioner_v1.zip is
currently the restored near-v1 checkpoint (27afa91e...) -- NOT the probe's checkpoint,
which was rejected/not promoted given the result above and lives separately at
models/l3_executioner_v1_replace_direction_probe.zip.
Blocking/open questions: (g) NEW: does the checkpoint-overwrite incident above change
anything for L2's item (2) -- the answer should be no, since the restored checkpoint is
numerically verified equivalent to what was already being evaluated, but flagging since
it is a new fact, not assuming it is obviously irrelevant. (h) NEW: given the probe's
non-result, is further budget on this direction (longer probe, different formula
entirely, or dropping the REPLACE-usage question as not worth chasing further right
now) worth committing, or should focus shift elsewhere -- explicitly not this track's
call to make unilaterally, surfacing for direction. (f) from the prior entry (is
near-parity-with-TWAP good enough for L2 to build on) remains open and is unaffected by
this probe either way.
Next planned step: awaiting direction on (h) [this probe's own follow-up] and (f) [the
still-open handoff question from before]. reward.py/test_reward.py stay uncommitted
until one of those is resolved -- recommend not committing the direction inversion until
it is either confirmed by more data or deliberately abandoned, rather than landing an
unconfirmed change; that is a recommendation, not a decision made here.
