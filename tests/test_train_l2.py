"""Fixture-based unit tests for src/train/train_l2.py -- ValISEvalCallback and the env
construction helpers. Same style as tests/test_wrappers.py (small synthetic parquet day, a
tiny freshly-initialized RecurrentPPO as the frozen-L3 stand-in, no GPU, no real
checkpoint) -- self-contained rather than importing test_wrappers.py's helpers, matching
this project's own convention of each test file owning its fixtures
(tests/test_lob_execution_env.py / tests/test_lob_execution_env_features.py both do this
independently too).
"""
from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
import pytest
from sb3_contrib import RecurrentPPO
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.data.l2_numeric_format import write_day
from src.envs.lob_execution_env import LOBExecutionEnv
from src.train.train_l2 import (
    ValISEvalCallback,
    _resolve_gradient_steps,
    make_l2_env,
    make_l2_wrapped_env,
    resolve_l2_final_save_paths,
)


# --------------------------------------------------------------------------------------
# Helpers (mirrors tests/test_wrappers.py's pattern, kept local/self-contained)
# --------------------------------------------------------------------------------------

def _write_synthetic_day(path, n_rows: int, base_price: float, ts_start: int) -> None:
    best_bid = base_price - 0.05
    best_ask = base_price + 0.05
    bids = json.dumps([[best_bid, 10.0], [best_bid - 0.1, 5.0]])
    asks = json.dumps([[best_ask, 10.0], [best_ask + 0.1, 5.0]])
    rows = [
        {
            "ts": ts_start + i,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid_price": base_price,
            "spread": best_ask - best_bid,
            "bids": bids,
            "asks": asks,
        }
        for i in range(n_rows)
    ]
    pd.DataFrame(rows).to_parquet(path, index=False)


def _write_synthetic_numeric_day(path, n_rows: int, base_price: float, ts_start: int) -> None:
    # Same values as _write_synthetic_day above, arrays instead of per-row JSON --
    # 2 levels/side (not 20) to exactly mirror that fixture's own level count. The
    # 20-level shape is a real-data integrity check the CONVERSION script enforces
    # (scripts/convert_l2_to_numeric_parallel.py), not something write_day/read_day or
    # LOBExecutionEnv's numeric read path themselves require -- both treat bid_prices/
    # bid_sizes/ask_prices/ask_sizes generically by whatever shape is given.
    best_bid = base_price - 0.05
    best_ask = base_price + 0.05
    write_day(
        {
            "ts": np.arange(ts_start, ts_start + n_rows, dtype=np.int64),
            "best_bid": np.full(n_rows, best_bid, dtype=np.float64),
            "best_ask": np.full(n_rows, best_ask, dtype=np.float64),
            "mid_price": np.full(n_rows, base_price, dtype=np.float64),
            "spread": np.full(n_rows, best_ask - best_bid, dtype=np.float64),
            "bid_prices": np.tile([best_bid, best_bid - 0.1], (n_rows, 1)),
            "bid_sizes": np.tile([10.0, 5.0], (n_rows, 1)),
            "ask_prices": np.tile([best_ask, best_ask + 0.1], (n_rows, 1)),
            "ask_sizes": np.tile([10.0, 5.0], (n_rows, 1)),
        },
        path,
    )


def _build_env(tmp_path, horizon_ticks: int = 20, lookback_ticks: int = 2) -> LOBExecutionEnv:
    data_dir = tmp_path / "BTCUSDT"
    data_dir.mkdir(exist_ok=True)
    _write_synthetic_day(data_dir / "l2-BTCUSDT-2024-01-01.parquet", n_rows=200, base_price=100.0, ts_start=1_000_000)
    return LOBExecutionEnv(
        data_dir=data_dir, horizon_ticks=horizon_ticks, lookback_ticks=lookback_ticks,
        tick_interval_s=1.0, date_range=("2024-01-01", "2024-01-01"),
    )


def _build_tiny_recurrent_ppo(env: LOBExecutionEnv) -> RecurrentPPO:
    venv = DummyVecEnv([lambda: env])
    return RecurrentPPO(
        "MlpLstmPolicy", venv,
        policy_kwargs=dict(lstm_hidden_size=8, n_lstm_layers=1, net_arch=dict(pi=[8], vf=[8])),
        n_steps=8, batch_size=8, device="cpu", seed=0,
    )


def _write_fake_vecnormalize(path, env: LOBExecutionEnv) -> str:
    venv = DummyVecEnv([lambda: env])
    vn = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=5.0)
    out = path / "fake_vecnormalize.pkl"
    with open(out, "wb") as f:
        pickle.dump(vn, f)
    return str(out)


def _date_range(tmp_path) -> tuple[str, str]:
    # _build_env's own synthetic day, reused as both "train" and "val" -- irrelevant for
    # pure mechanics testing, matches how a real invocation of train_l2.py would reuse
    # the same l3_model object for both, just possibly different LOBExecutionEnv
    # instances/date ranges.
    return ("2024-01-01", "2024-01-01")


def _data_dir(tmp_path) -> str:
    # make_l2_wrapped_env/ValISEvalCallback default to the real data archive path --
    # _build_env already wrote the synthetic day under tmp_path/"BTCUSDT" (matching
    # LOBExecutionEnv's own default dirname convention), so point them there instead.
    return str(tmp_path / "BTCUSDT")


# --------------------------------------------------------------------------------------
# make_l2_wrapped_env / make_l2_env
# --------------------------------------------------------------------------------------

def test_make_l2_env_is_monitor_wrapped(tmp_path):
    env = _build_env(tmp_path)
    model = _build_tiny_recurrent_ppo(env)
    vecnorm_path = _write_fake_vecnormalize(tmp_path, env)
    wrapped = make_l2_env(_date_range(tmp_path), 20, 2, model, vecnorm_path, 4, False, data_dir=_data_dir(tmp_path))
    assert isinstance(wrapped, Monitor)


def test_make_l2_wrapped_env_is_not_monitor_wrapped(tmp_path):
    env = _build_env(tmp_path)
    model = _build_tiny_recurrent_ppo(env)
    vecnorm_path = _write_fake_vecnormalize(tmp_path, env)
    wrapped = make_l2_wrapped_env(_date_range(tmp_path), 20, 2, model, vecnorm_path, 4, False, data_dir=_data_dir(tmp_path))
    assert not isinstance(wrapped, Monitor)
    obs, info = wrapped.reset(seed=0)
    assert obs.shape == (41,)


def test_make_l2_wrapped_env_use_numeric_format_reads_npzst(tmp_path):
    # use_numeric_format is a new, trailing, defaulted kwarg this round (vectorization --
    # see train_l2.py's module docstring) -- confirms it actually threads through to
    # LOBExecutionEnv rather than being silently ignored, and that the numeric-format
    # read path produces the same obs shape as the parquet path above.
    data_dir = tmp_path / "BTCUSDT_numeric"
    data_dir.mkdir(exist_ok=True)
    _write_synthetic_numeric_day(
        data_dir / "l2-BTCUSDT-2024-01-01.npzst", n_rows=200, base_price=100.0, ts_start=1_000_000,
    )
    # The frozen-L3 stand-in and its VecNormalize are built against a PARQUET env
    # (_build_env) purely as a source of a matching observation_space -- L3's own obs
    # space is format-independent (it reads whatever LOBExecutionEnv hands it), so this
    # does not need to be the numeric env itself.
    l3_env = _build_env(tmp_path)
    model = _build_tiny_recurrent_ppo(l3_env)
    vecnorm_path = _write_fake_vecnormalize(tmp_path, l3_env)

    wrapped = make_l2_wrapped_env(
        ("2024-01-01", "2024-01-01"), 20, 2, model, vecnorm_path, 4, False,
        data_dir=str(data_dir), use_numeric_format=True,
    )
    assert wrapped.env.use_numeric_format is True
    obs, info = wrapped.reset(seed=0)
    assert obs.shape == (41,)


# --------------------------------------------------------------------------------------
# ValISEvalCallback -- _run_episode determinism, the TWAP-passthrough baseline, and full
# SAC.learn() integration (fires, logs, doesn't crash).
# --------------------------------------------------------------------------------------

def _build_eval_callback(tmp_path, eval_freq: int = 2, n_eval_episodes: int = 2) -> ValISEvalCallback:
    env = _build_env(tmp_path)
    l3_model = _build_tiny_recurrent_ppo(env)
    vecnorm_path = _write_fake_vecnormalize(tmp_path, env)
    return ValISEvalCallback(
        val_date_range=_date_range(tmp_path),
        horizon_ticks=20,
        lookback_ticks=2,
        ticks_per_l2_decision=4,
        l3_model=l3_model,
        l3_vecnormalize_path=vecnorm_path,
        l2_include_prev_action=False,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        verbose=0,
        data_dir=_data_dir(tmp_path),
    )


def test_run_episode_is_deterministic_for_a_fixed_action_and_seed(tmp_path):
    # Same seed + same fixed action_fn (TWAP-passthrough) -> identical is_result, since
    # the underlying env's reset() is seeded and the matching-engine mechanics are
    # deterministic given a fixed action sequence and replayed market data. A real
    # reproducibility assertion, not just "did it run."
    cb = _build_eval_callback(tmp_path)
    action_fn = lambda obs: cb._TWAP_PASSTHROUGH_ACTION  # noqa: E731
    r1 = cb._run_episode(seed=42, action_fn=action_fn)
    r2 = cb._run_episode(seed=42, action_fn=action_fn)
    assert r1["is_result"].fill_ratio == pytest.approx(r2["is_result"].fill_ratio)
    assert r1["is_result"].is_total_bps == pytest.approx(r2["is_result"].is_total_bps)
    assert r1["total_reward"] == pytest.approx(r2["total_reward"])


def test_run_episode_different_seeds_can_differ(tmp_path):
    # Sanity check the determinism test above isn't trivially true because every seed
    # produces the same outcome regardless (e.g. a broken seed passthrough) -- side/size
    # are drawn from np_random per reset(seed=...), so different seeds should generally
    # produce different qty_total/side at minimum.
    cb = _build_eval_callback(tmp_path)
    action_fn = lambda obs: cb._TWAP_PASSTHROUGH_ACTION  # noqa: E731
    outcomes = {cb._run_episode(seed=s, action_fn=action_fn)["is_result"].fill_ratio for s in range(5)}
    assert len(outcomes) > 1


def test_twap_passthrough_baseline_populated_with_correct_shape(tmp_path):
    cb = _build_eval_callback(tmp_path, n_eval_episodes=3)
    assert cb._twap_passthrough_is_bps is None
    cb._on_training_start()
    assert cb._twap_passthrough_is_bps is not None
    assert cb._twap_passthrough_is_bps.shape == (3,)
    assert cb._twap_passthrough_fill.shape == (3,)
    assert np.isfinite(cb._twap_passthrough_is_bps).all()
    # 1e-9 tolerance: fill_ratio = filled_qty / qty_total can land a hair over 1.0 from
    # ordinary float summation error on a fully-filled episode (e.g. 1.0000000000000002),
    # not a real correctness bug -- a strict <=1.0 would be testing float precision, not
    # the thing this test cares about.
    assert ((cb._twap_passthrough_fill >= -1e-9) & (cb._twap_passthrough_fill <= 1.0 + 1e-9)).all()


def test_eval_seeds_are_fixed_and_paired(tmp_path):
    cb = _build_eval_callback(tmp_path, n_eval_episodes=5)
    assert cb._eval_seeds == [cb.EVAL_SEED_BASE + i for i in range(5)]


def test_eval_callback_wired_into_sac_learn_fires_and_logs(tmp_path):
    # Full integration: a tiny SAC model actually training against make_l2_env(), with
    # the eval callback wired in exactly as train_l2.py's main() does it. Verifies the
    # callback fires at least once (eval_freq=2, well within a short run) and that
    # self.model.predict() (via _l2_policy_action, VecNormalize-aware) works without
    # error against a real (tiny) SAC policy -- not just the TWAP-passthrough path
    # exercised by the tests above.
    env = _build_env(tmp_path)
    l3_model = _build_tiny_recurrent_ppo(env)
    vecnorm_path = _write_fake_vecnormalize(tmp_path, env)
    train_env = make_l2_env(_date_range(tmp_path), 20, 2, l3_model, vecnorm_path, 4, False, data_dir=_data_dir(tmp_path))

    sac_model = SAC(
        "MlpPolicy", train_env,
        buffer_size=1000, batch_size=8, learning_starts=0,
        device="cpu", seed=0, verbose=0,
    )
    eval_cb = ValISEvalCallback(
        val_date_range=_date_range(tmp_path),
        horizon_ticks=20, lookback_ticks=2, ticks_per_l2_decision=4,
        l3_model=l3_model, l3_vecnormalize_path=vecnorm_path,
        l2_include_prev_action=False, eval_freq=2, n_eval_episodes=2, verbose=0,
        data_dir=_data_dir(tmp_path),
    )

    sac_model.learn(total_timesteps=8, callback=eval_cb, progress_bar=False)

    assert eval_cb._twap_passthrough_is_bps is not None  # _on_training_start() ran
    assert eval_cb._last_eval_step > 0  # _on_step() fired at least once past eval_freq=2


def test_get_vec_normalize_env_is_none_when_l2_obs_not_normalized(tmp_path):
    # Documents current, deliberate state (docs/reports/phase4_l2_reconciliation_and_plan.md's
    # Decision (a): VecNormalize for L2's own obs is recommended but not yet implemented)
    # -- _l2_policy_action's defensive `if vec_normalize is not None` branch is currently
    # always the None path. This test exists so that decision's implementation status is
    # pinned by a test, not just a doc claim -- it should start failing (and be updated,
    # not silently left) the day someone adds VecNormalize around the L2 training env.
    # Still true after this round's vectorization work -- out of that round's own stated
    # scope (see train_l2.py's module docstring).
    env = _build_env(tmp_path)
    l3_model = _build_tiny_recurrent_ppo(env)
    vecnorm_path = _write_fake_vecnormalize(tmp_path, env)
    train_env = make_l2_env(_date_range(tmp_path), 20, 2, l3_model, vecnorm_path, 4, False, data_dir=_data_dir(tmp_path))
    sac_model = SAC("MlpPolicy", train_env, buffer_size=100, batch_size=8, device="cpu", verbose=0)
    assert sac_model.get_vec_normalize_env() is None


# --------------------------------------------------------------------------------------
# _resolve_gradient_steps -- pure function, no env/model needed. See its own docstring
# in train_l2.py for the SB3-source-confirmed mechanics this corrects for.
# --------------------------------------------------------------------------------------

def test_resolve_gradient_steps_defaults_to_n_envs():
    assert _resolve_gradient_steps(n_envs=4, override=None) == 4
    assert _resolve_gradient_steps(n_envs=1, override=None) == 1
    assert _resolve_gradient_steps(n_envs=8, override=None) == 8


def test_resolve_gradient_steps_explicit_override_wins():
    assert _resolve_gradient_steps(n_envs=8, override=1) == 1
    assert _resolve_gradient_steps(n_envs=1, override=16) == 16


# --------------------------------------------------------------------------------------
# resolve_l2_final_save_paths -- same guard/rationale as train_l3.py's
# resolve_final_save_paths (see tests/test_train_l3.py), adapted for L2's single-path
# (no VecNormalize) save.
# --------------------------------------------------------------------------------------

def test_resolve_l2_final_save_paths_fresh_dir_uses_canonical(tmp_path):
    model_stem = resolve_l2_final_save_paths(
        run_name="20260101_000000", overwrite_canonical=False, models_dir=tmp_path,
    )
    assert model_stem == str(tmp_path / "l2_strategist_v1")


def test_resolve_l2_final_save_paths_existing_canonical_redirects(tmp_path):
    (tmp_path / "l2_strategist_v1.zip").write_bytes(b"existing checkpoint")
    model_stem = resolve_l2_final_save_paths(
        run_name="probe_20260101", overwrite_canonical=False, models_dir=tmp_path,
    )
    assert model_stem == str(tmp_path / "l2_strategist_v1_probe_20260101")
    # And nothing was actually touched -- resolve_l2_final_save_paths only decides, it
    # does not write.
    assert (tmp_path / "l2_strategist_v1.zip").read_bytes() == b"existing checkpoint"


def test_resolve_l2_final_save_paths_overwrite_canonical_flag_forces_canonical(tmp_path):
    (tmp_path / "l2_strategist_v1.zip").write_bytes(b"existing checkpoint")
    model_stem = resolve_l2_final_save_paths(
        run_name="20260101_000000", overwrite_canonical=True, models_dir=tmp_path,
    )
    assert model_stem == str(tmp_path / "l2_strategist_v1")


# --------------------------------------------------------------------------------------
# CLI defaults -- locks in the eval_freq/n_eval_episodes/--eval defaults this round
# derived from the measured throughput, so they don't silently drift.
# --------------------------------------------------------------------------------------

def test_cli_eval_defaults_match_documented_values():
    # Uses the REAL parser from train_l2.build_parser() -- not a re-declared duplicate --
    # so this actually pins the shipped defaults; it fails if someone changes them in
    # train_l2.py without updating this test.
    from src.train.train_l2 import build_parser

    args = build_parser().parse_args([
        "--l3-checkpoint", "unused.zip", "--l3-vecnormalize", "unused.pkl",
        "--total-timesteps", "1",
    ])
    assert args.eval is True
    assert args.eval_freq == 10_000
    assert args.n_eval_episodes == 10


def test_cli_no_eval_flag_disables_it():
    from src.train.train_l2 import build_parser

    args = build_parser().parse_args([
        "--l3-checkpoint", "unused.zip", "--l3-vecnormalize", "unused.pkl",
        "--total-timesteps", "1", "--no-eval",
    ])
    assert args.eval is False


def test_cli_n_envs_and_seed_defaults():
    # n_envs=4 (not 8) is a deliberate memory-safety choice, not an arbitrary default --
    # see --n-envs's own CLI help in train_l2.py for the RSS/OOM-history rationale.
    # seed=42 did not exist on this script at all before this round.
    from src.train.train_l2 import build_parser

    args = build_parser().parse_args([
        "--l3-checkpoint", "unused.zip", "--l3-vecnormalize", "unused.pkl",
        "--total-timesteps", "1",
    ])
    assert args.n_envs == 4
    assert args.seed == 42
    assert args.gradient_steps is None
    assert args.use_numeric_format is True


def test_cli_resume_replay_buffer_requires_resume_from():
    # main()'s own validation (not argparse's) -- build_parser() alone doesn't enforce
    # this cross-flag dependency, matching train_l3.py's own --resume-from/
    # --resume-vecnormalize pairing check style (a plain ValueError in main(), not
    # something argparse's mutually_exclusive_group covers, since one flag is required
    # only conditionally on the other being present, not mutually exclusive).
    from src.train.train_l2 import build_parser

    args = build_parser().parse_args([
        "--l3-checkpoint", "unused.zip", "--l3-vecnormalize", "unused.pkl",
        "--total-timesteps", "1", "--resume-replay-buffer", "unused_buffer.pkl",
    ])
    assert args.resume_from is None
    assert args.resume_replay_buffer == "unused_buffer.pkl"
    # main() itself raises on this combination -- checked directly in the shakedown/
    # integration round rather than re-invoking main()'s full argv/GPU/data-file path
    # here; this test only pins that the CLI still parses the (invalid) combination
    # through to main() rather than argparse silently rejecting or coercing it.
