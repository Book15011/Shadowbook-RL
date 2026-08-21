"""Tests for train_l3.py's final-save path-safety guard. Covers only
resolve_final_save_paths() -- a pure path-decision function with no I/O
beyond an existence check -- deliberately, not main() itself, which requires
real config/data files, a SubprocVecEnv, and a GPU to construct. See that
function's docstring in src/train/train_l3.py for the guard's rationale: a
bounded probe run's final save silently overwrote a verified checkpoint here
once, because the save path was unconditionally hardcoded."""
from src.train.train_l3 import resolve_final_save_paths


def test_resolve_final_save_paths_fresh_dir_uses_canonical(tmp_path):
    # Neither canonical file exists yet -- first-ever save, no guard needed.
    model_stem, vecnorm_path = resolve_final_save_paths(
        run_name="20260101_000000", overwrite_canonical=False, models_dir=tmp_path,
    )
    assert model_stem == str(tmp_path / "l3_executioner_v1")
    assert vecnorm_path == str(tmp_path / "l3_vecnormalize.pkl")


def test_resolve_final_save_paths_existing_canonical_redirects(tmp_path):
    # The exact incident this guard exists to prevent: canonical model.zip
    # already exists, overwrite not explicitly authorized -> redirect.
    (tmp_path / "l3_executioner_v1.zip").write_bytes(b"existing checkpoint")
    model_stem, vecnorm_path = resolve_final_save_paths(
        run_name="probe_20260101", overwrite_canonical=False, models_dir=tmp_path,
    )
    assert model_stem == str(tmp_path / "l3_executioner_v1_probe_20260101")
    assert vecnorm_path == str(tmp_path / "l3_vecnormalize_probe_20260101.pkl")
    # And nothing was actually touched -- resolve_final_save_paths only decides,
    # it does not write.
    assert (tmp_path / "l3_executioner_v1.zip").read_bytes() == b"existing checkpoint"


def test_resolve_final_save_paths_existing_vecnorm_only_still_redirects(tmp_path):
    # Only the VecNormalize half exists (e.g. an interrupted prior save) --
    # OR, not AND: still redirects, so a run can never leave a mismatched
    # model/VecNormalize pair behind by only overwriting the missing half.
    (tmp_path / "l3_vecnormalize.pkl").write_bytes(b"existing vecnormalize")
    model_stem, vecnorm_path = resolve_final_save_paths(
        run_name="probe_20260101", overwrite_canonical=False, models_dir=tmp_path,
    )
    assert model_stem == str(tmp_path / "l3_executioner_v1_probe_20260101")
    assert vecnorm_path == str(tmp_path / "l3_vecnormalize_probe_20260101.pkl")


def test_resolve_final_save_paths_overwrite_canonical_flag_forces_canonical(tmp_path):
    # Explicit opt-in: this run IS deliberately meant to supersede the
    # current canonical checkpoint.
    (tmp_path / "l3_executioner_v1.zip").write_bytes(b"existing checkpoint")
    (tmp_path / "l3_vecnormalize.pkl").write_bytes(b"existing vecnormalize")
    model_stem, vecnorm_path = resolve_final_save_paths(
        run_name="20260101_000000", overwrite_canonical=True, models_dir=tmp_path,
    )
    assert model_stem == str(tmp_path / "l3_executioner_v1")
    assert vecnorm_path == str(tmp_path / "l3_vecnormalize.pkl")
