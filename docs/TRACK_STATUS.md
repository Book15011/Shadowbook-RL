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

## L3 / Env-Physics
Last updated: 2026-08-20 10:14 HKT
State: Parts A/B/C (r_queue split restored, 3-commit split, warm-start validation) are
unchanged from prior check-ins -- see earlier entries for full detail. This entry covers
the full run itself: LAUNCHED, MONITORED, and now COMPLETED.

LAUNCHED at 08:12 HKT, warm-started from models/l3_executioner_v1.zip (checksum-verified
94b3ad38... immediately before launch), fresh VecNormalize, n_envs=8, RewardWeights()
real defaults, --total-timesteps 2000000, logging to
logs/l3_train_fullrun_fixedphysics_warmstart_2M.log (a new path, distinct from every
probe log this session). Startup confirmed clean (cuda=True, 405/18 train/val days,
warm-start confirmed, ep_len_mean at full horizon by iteration 6, TWAP baseline matching
Part C exactly). The launch command's own ssh connection hung on a known stdio quirk
(harmless, process already detached via nohup+disown) -- confirmed the real process via
a separate connection instead of touching the hung one; that connection later closed on
its own (exit 0) once its stdio backlog cleared, well after the real process was already
independently verified.

1-HOUR CHECK-IN (09:15 HKT): single process, no duplicates. Memory 35GB/50GB used, 14GB
available -- no OOM risk (dmesg scanned, no OOM-kill signatures; box has had 3 OOM
incidents earlier this session, so this was checked deliberately, not assumed). Progress
1,179,648/2,000,000 steps (~59%), fps steady 306-311, ep_len_mean still 2.99e+03 (full
horizon) -- healthy, no reset-storm.

COMPLETED at 10:11 HKT (~1h59m wall-clock, time_elapsed=7130s in-script). Process exited
on its own after its own final save -- not killed. Memory returned to baseline (1.8GB
used / 48GB available) confirming no leak. Final total_timesteps=2,002,944 (slightly over
2M -- PPO's fixed rollout-chunk size, expected, not an error).

New checkpoint: models/l3_executioner_v1.zip (sha256 973b2883339568595188034c22be2fb3d
0136abd0b325fb5e08d108735c6e739, 2,610,702 bytes -- sane vs. the old baseline's 2,609,950)
and models/l3_vecnormalize.pkl (sha256 839ea093ed69169fc8444f9dc42e8c3cd90869ed38fc92c3
56bc7f789ae14856). Verified, not assumed: zip integrity checked directly
(zipfile.testzip(), all 6 expected SB3 entries present, none corrupt). models/*.zip is
gitignored (confirmed via git check-ignore) -- correctly not committed; the ORIGINAL
20M-step buggy-physics baseline remains separately preserved AND git-tracked at
models/baseline_20M_backup/ (sha256 94b3ad38..., untouched, safe beyond local disk too).

FINAL VALIDATION (ValISEvalCallback, step=2,000,000, paired seeds 5000000..5000049,
n=50 episodes, same fixed-physics data as every eval this session):
  L3 IS_total_bps mean=1.245 (std 4.74) vs TWAP baseline=1.182 (std ~5.03 elsewhere in
  this session's TWAP measurements) -> val_l3_beats_twap_bps=-0.0631 -- L3 is marginally
  WORSE than TWAP, but well within noise at this n (this session's own earlier
  significance analysis put unpaired SE at ~0.7-1.3bps even with pairing's noise-
  cancellation benefit) -- read this as a statistical DEAD HEAT with TWAP, not a loss.
  fill_ratio mean=0.918 (TWAP=0.994) -- NOT a strong result on its own, but a large,
  substantive recovery from the OLD checkpoint's fixed-physics fill_ratio of 0.2015
  (measured in Phase 3, evaluating the never-retrained-under-fixed-physics checkpoint).
  Read together: the Phase 3 figure of "L3 beats TWAP by 1.38bps" used a policy that
  barely executed (0.2015 fill_ratio) -- its favorable IS number likely reflects avoided
  price impact from incomplete trades, not real execution skill, making it a less
  trustworthy comparison than this one. This retrained checkpoint actually executes
  (0.918 fill_ratio) and, once it does, lands at parity with TWAP rather than clearly
  beating it. That is a more honest number, even though it is a more modest headline.

OBSERVATION (not investigated further this round): ep_len_mean declined from ~2.99e+03
at the 1-hour/59% mark to ~1.74-1.83e+03 by the final iterations, while fps stayed
healthy (280-290) and approx_kl/value_loss/explained_variance all looked like normal
late-training convergence, not divergence. Plausible read: the policy learned to
complete orders well before the 3,000-tick horizon in many episodes (a sensible
execution-agent behavior, and consistent with the fill_ratio recovery above) rather than
regressing toward the from-scratch reset-storm pathology (which showed fps 8-10, not
280-290+, as its actual signature). Flagged for whoever looks at this checkpoint next,
not confirmed via separate analysis.

Files owned/in-progress: src/envs/lob_execution_env.py, src/envs/reward.py,
tests/test_lob_execution_env_features.py, tests/test_reward.py, src/train/train_l3.py --
unchanged from prior check-ins (same uncommitted, experimental placement-staleness/
eta_replace round on top of the 3 landed commits; train_l3.py still uncommitted for the
same TypeError reason as before).
Blocking/open questions: (a)/(b)/(c) [RESOLVED, see prior entries]. (d) [still open, low
urgency] train_l3.py's eta_replace path still needs either the staleness reward term
committed alongside it or the --reward-eta-replace flag stripped/deferred. (e)
[RESOLVED] checkpoint identity is settled -- models/l3_executioner_v1.zip is now
973b2883... going forward, not 94b3ad38.... (f) NEW: whether a near-parity-with-TWAP
result is "good enough" to build L2 on top of, vs. training further (toward the full
20M target) or iterating on the reward shape first, is an open judgment call this track
is surfacing, not resolving.
Next planned step: awaiting direction on (f) above. This track's own immediate work
(items (d), and the still-uncommitted staleness/eta_replace round) can proceed
independently of that decision whenever picked back up.
