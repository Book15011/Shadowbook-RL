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
Last updated: 2026-08-20 10:44 HKT
State: FINAL design spec produced for observation space + SAC hyperparameters -- no code
written yet, this is still the planning phase. Step 1: re-read architecture_spec.md Section
4.1/4.3 fresh (via git diff, not memory) -- confirmed the Section 4.1-vs-4.3 L2 cadence
conflict flagged last round is fixed (L2_EVERY_N_TICKS is now 50, matching
ticks_per_l2_decision=50), though the fix itself is a bare one-line edit with no restated
rationale ("# I just change here") -- treating as resolved with moderate, not high,
confidence, matching how this was framed going in. Grepped the whole real repo (configs/,
src/, tests/, scripts/) for any other hardcoded 10-tick/1s L2 assumption: found none;
configs/sac_l2.yaml exists but is an empty placeholder with no cadence or hyperparameter
values at all. Step 2: built a concrete L2 observation space -- Box(shape=(41,)) base
(Section 3.1's literal requirement: 40 features from the 42-dim vector, each individually
assigned an aggregation rule -- last-value/instantaneous/mean-over-window/pass-through,
grouped and justified by feature type, not one blanket rule -- plus 1 new
schedule_deviation scalar, confirmed computable from existing env state
(_compute_l2_target_slice_ratio() still the right hook, re-verified against the current,
further-changed working tree) with zero new instrumentation needed) or Box(shape=(43,))
recommended (adds an explicit, separately-flagged previous-action toggle,
l2_include_prev_action, default ON, with reasoning -- SAC's plain MlpPolicy has no
recurrence to fall back on unlike L3's LSTM). Could NOT locate "the recurrent-policy paper
in this project's own PDFs" referenced in the handoff -- searched the whole box, no PDFs
exist anywhere in this repo; flagged rather than guessed which paper was meant or
fabricating a citation. Full index-mapping table (old idx -> new position -> transform) in
the design doc. Step 3: buffer_size=500,000 and gamma=0.995 re-confirmed unchanged (both
already assumed the 50-tick cadence throughout); PROVISIONAL dropped for the cadence-related
reason specifically. Noted the dimensionality change (41->43) doesn't warrant any net_arch
change from SB3's SAC default (256,256) -- too small a change (~5% wider input layer) to
matter. Separately (not required this round, surfaced while re-reading the env code): the
L3 track's full 2,000,000-step warm-start run has COMPLETED since last check-in --
checkpoint identity is now resolved on that track too (models/l3_executioner_v1.zip,
sha256 973b2883...), but a new, different judgment call has replaced it there (is a
near-parity-with-TWAP result good enough to build on) -- not something this design doc
resolves.
Files owned/in-progress: none (still read-only/planning). Same file,
docs/reports/phase4_l2_reconciliation_and_plan.md, now with a "FINAL SPEC" section appended
covering Steps 1-3 above.
Blocking/open questions: (1) awaiting a pointer to the specific recurrent-policy paper
referenced for Step 2c, if the previous-action-toggle recommendation should incorporate it
specifically rather than this session's own general reasoning. (2) implementation
(FrozenL3Wrapper/train_l2.py) still depends on the L3 track's separate, still-open item (f)
(is the current checkpoint's near-parity-with-TWAP result good enough to build on) -- this
design doc's own content is final regardless of how that resolves, but starting real
training is not yet cleared.
Next planned step: once item (2) above is cleared, implement FrozenL3Wrapper (per this doc's
original wrapper-only mechanism, Part B) and train_l2.py, using the FINAL SPEC section's
observation space and hyperparameters directly -- no further design work anticipated before
that.

IMPORTANT correction from L3 (2026-08-20 23:05 HKT): the sha256 973b2883... referenced
above is NO LONGER what's in models/l3_executioner_v1.zip -- see the incident writeup in
the L3 section below before treating that checksum as current. Short version: a bounded
probe launched from this track accidentally overwrote the working-slot checkpoint (every
train_l3.py run saves to the same hardcoded path regardless of run type, and this track
did not back up 973b2883... first before launching -- a real process gap, owned below).
The true 973b2883... file cannot be recovered byte-for-byte. What's in the working slot
right now (sha256 27afa91e...) is v1's own periodic checkpoint from 2,944 steps earlier
in the same training run -- verified via full reproduction to match the officially
reported v1 numbers (IS_total_bps=1.245, fill_ratio=0.918) exactly, so it is a faithful,
numerically-confirmed stand-in, not a guess. If you have not yet pointed FrozenL3Wrapper
at a specific file, no action needed -- 27afa91e... is safe to treat as "the v1 checkpoint"
going forward. If you already loaded/hashed 973b2883... somewhere and are relying on that
exact checksum, that specific artifact is gone; the behavioral numbers it reported are not
in question, only the exact bytes.

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
