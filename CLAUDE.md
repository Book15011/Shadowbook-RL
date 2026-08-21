# Standing rules -- lob-execution-hma

Three concurrent Claude Code sessions (L1/L2/L3) work this repo at once, over ssh, on
the same real checkout, the same git history, and the same shared status file
(docs/TRACK_STATUS.md). Each rule below was learned the hard way once already -- read
this before repeating one of these mistakes.

## Rules

1. **Classifier denial means STOP and ask the user -- never retry via a different tool,
   shell, or operation shape. No exceptions.**
   Why: a session once routed a Bash-blocked denial through PowerShell instead. The
   denial message itself says to stop and ask; routing around it defeats the check.

2. **Never write to a canonical checkpoint path from a run that might not be the
   keeper. Use a run-tagged output path, or an explicit --overwrite-canonical guard.**
   Why: train_l3.py's hardcoded final-save path destroyed a verified checkpoint
   (973b2883..., unrecoverable) when a bounded probe was allowed to run to completion.
   The same hardcoded-path bug independently existed in CheckpointCallback too, and had
   already silently overwritten v1's own early periodic saves before anyone noticed.
   Back up a checkpoint before any run that might reach its own final save and touch its
   path (see models/baseline_20M_backup/, models/v1_near_backup_step2M/ for the pattern).

3. **docs/TRACK_STATUS.md: own section only. Merge on conflict, never overwrite another
   track's section.**
   Why: three sessions write the same file, concurrently, on the same checkout.

4. **Run a fresh `git status --short` immediately before every stage/commit -- not one
   read earlier in the same turn.**
   Why: other sessions commit concurrently on this same working tree; a stale status is
   a race waiting to clobber someone else's uncommitted work.

5. **Verify checksums, paths, and shared state live, at the point of use -- never cite
   from memory, and never copy a citation from another doc, however recent.**
   Why: a correctly live-computed checksum citation went stale about two hours later
   when a different session's run overwrote the file it described. The original
   verification was right; the staleness was purely environmental. Re-verify anyway,
   every time -- recency of the last check is not the same as correctness right now.

6. **Read the real source before trusting a docstring or comment.**
   Why: this repo has had real comment-vs-code drift more than once -- a cache-size
   comment claiming 85MB when the real figure was 828MB, and a reward.py comment
   describing a code path as active when it was not. A comment is a claim about a past
   state of the code, not a live fact.

7. **Stop and ask when a task needs a judgment call, not arithmetic.**
   Why: reward-shaping direction and checkpoint-promotion calls in this repo are
   routinely surfaced as open questions in TRACK_STATUS.md rather than decided
   unilaterally -- a wrong silent call here costs real GPU-hours to unwind.

8. **Real code wins over the spec doc when they disagree. Read both ends -- the actual
   data on disk, the actual consumer code -- before designing against
   docs/architecture_spec.md alone.**
   Why: the spec is a design record, not necessarily what current code does. Example:
   L1MacroAnalyst.maybe_refresh() enforces no input schema at all on its
   feature_summary argument, despite the spec's prose implying one.

9. **Report negative or inconclusive results as plainly as positive ones. Do not
   cherry-pick whichever significance test happens to agree, and do not reframe a
   non-result as a win.**
   Why: established discipline across this repo's probes (staleness coefficient sweep,
   REPLACE-direction probe) -- false confidence in a mechanism is expensive to unwind
   once training budget has already been spent on it.

10. **Do not touch another track's active files without asking, even if a task's own
    boundary list does not name them explicitly. Check TRACK_STATUS.md's "Files
    owned/in-progress" first.**
    Why: matched GPU A/B runs and concurrent CPU-bound training are fragile to
    unrelated edits mid-run -- a stray edit can invalidate a run without producing an
    obvious error.

## See also

- docs/architecture_spec.md -- the design spec (see rule 8 for how much to trust it).
- docs/TRACK_STATUS.md -- current state of all three tracks, incident history, and
  open cross-track questions.
