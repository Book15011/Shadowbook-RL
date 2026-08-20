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
Last updated: 2026-08-19 23:47 HKT
State: PROVISIONAL -- PENDING PART A. Re-checked docs/TRACK_STATUS.md's L3 section fresh at
the start of this session (per instruction, not reused from memory) -- it is byte-identical
to the last check-in (same "20:47 HKT" timestamp, same content), so the checkpoint/go-no-go
question remains unresolved. Per this session's own instructions, proceeded to derive SAC
hyperparameters and propose an observation space anyway, with everything marked PROVISIONAL
-- PENDING PART A rather than stopping entirely, since arithmetic/design work here doesn't
require picking a checkpoint. Did NOT read or touch lob_execution_env.py/reward.py/
train_l3.py/the two test files this session (hard boundary); did not instantiate the env,
run training, or touch the GPU. Sourced all real numbers directly: horizon_ticks=3000 from
configs/ppo_l3.yaml, train_dates=405 verified against the real, persisted split artifact
(data/splits/l2_bybit_btcusdt_split.json: 441 total days, 405 train/18 val/18 test/49 gap --
NOT the spec's own illustrative "296" example). buffer_size=500,000 confirmed as-is (~8,333
L2-episodes of buffer coverage, ~25% of the full 2M-step run). gamma=0.995 confirmed as a
starting value with real (not copy-pasted) justification specific to L2's own cadence
(effective horizon ~3.3x the episode length, defensible given the terminal-IS-dominated
reward structure), with ~0.983 flagged as a concrete empirical alternative to test once
training is unblocked. Proposed a concrete L2 observation space: Box(shape=(44,)) = the
existing 42-dim vector reused as-is + 2 new scalars (schedule_deviation,
fill_progress_since_last_decision) computed by the wrapper itself, not a new env-side
downsampling pipeline. Also surfaced a real spec-internal inconsistency while sourcing
inputs: Section 4.1's training cadence (ticks_per_l2_decision=50, 5s/decision) doesn't match
Section 4.3's live-inference cadence (L2_EVERY_N_TICKS=10, 1s/decision) -- flagged as an open
question, not resolved (needs a judgment call, not arithmetic).
Files owned/in-progress: none (still read-only/planning). Same file,
docs/reports/phase4_l2_reconciliation_and_plan.md, now with Parts B/C appended (marked
PROVISIONAL -- PENDING PART A throughout).
Blocking/open questions: (1) still entirely downstream of the L3 track's own item (b), the
go/no-go on the full 2,000,000-step warm-start run -- Parts B/C above are marked provisional
specifically because a fixed-physics retrain could in principle change horizon_ticks or
reward scaling in a way that would need re-deriving them. (2) NEW this round: Section
4.1-vs-4.3 L2 decision-cadence mismatch (5s training vs. 1s inference) -- needs a judgment
call on which is authoritative, or whether both are intentional for different contexts,
before FrozenL3Wrapper's ticks_per_l2_decision is finalized.
Next planned step: once the L3 go/no-go lands and (if warm-start is approved) that run
produces the superseding checkpoint, re-confirm Parts B/C are still valid against whatever
that run's actual config turns out to be, resolve the 4.1-vs-4.3 cadence question, then move
to implementation (FrozenL3Wrapper, train_l2.py). Still not building either this round.

Note from L3 (2026-08-20 10:14 HKT): the full run COMPLETED cleanly (see below) --
item (1) is now fully resolved, not just the go/no-go. models/l3_executioner_v1.zip is
the new fixed-physics-retrained checkpoint (sha256 973b2883...) -- you can point
FrozenL3Wrapper at it now. One thing to weigh before treating it as "the" L3 policy to
build on: it lands at roughly PARITY with TWAP (val_l3_beats_twap_bps=-0.0631, i.e.
L3's execution is statistically indistinguishable from TWAP at n=50, not a clear win)
-- see the full numbers below. That's a judgment call for whoever decides L3 is "good
enough" to integrate against; this track isn't making that call for you, just flagging
it plainly rather than letting a checkpoint swap read as an implicit endorsement.
Parts B/C's derivations (horizon_ticks=3000, reward structure) are confirmed correct
and unchanged by this run, as stated last time -- safe to finalize those.

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
