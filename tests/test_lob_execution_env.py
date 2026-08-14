"""Regression coverage for LOBExecutionEnv's file-selection / seed reproducibility.

Uses small synthetic parquet days (not the real Bybit archive) so this test is
self-contained and fast -- it only needs to exercise reset()'s file-selection
and window-sampling logic, not real market data.
"""
import json

import pandas as pd
import pytest

from src.envs.lob_execution_env import LOBExecutionEnv


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


def _reset_fingerprint(env: LOBExecutionEnv, seed: int):
    env.reset(seed=seed)
    start_tick = env._ticks[env._episode_start]
    return (start_tick.ts, start_tick.mid_price, env.side, env.qty_total)


def test_without_date_range_file_list_drifts_as_new_files_appear(tmp_path):
    # Reproduces the Check-1 bug directly: self._files is a fresh glob per
    # instantiation, so a file the backfill job adds between two separate runs
    # changes len(self._files), and therefore what a fixed seed resolves to.
    data_dir = tmp_path / "BTCUSDT"
    data_dir.mkdir()
    _write_synthetic_day(data_dir / "l2-BTCUSDT-2024-01-01.parquet", n_rows=20, base_price=100.0, ts_start=1_000_000)

    kwargs = dict(data_dir=data_dir, horizon_ticks=5, lookback_ticks=2)
    env1 = LOBExecutionEnv(**kwargs)
    assert len(env1._files) == 1

    # simulate the backfill job (which walks backward, prepending older days)
    # adding a new day to disk in between two separate script runs
    _write_synthetic_day(data_dir / "l2-BTCUSDT-2023-12-31.parquet", n_rows=20, base_price=999.0, ts_start=1)

    env2 = LOBExecutionEnv(**kwargs)
    assert len(env2._files) == 2, "file list should have grown -- this is the bug this test documents"


def test_date_range_pins_identical_window_across_separate_instantiations(tmp_path):
    data_dir = tmp_path / "BTCUSDT"
    data_dir.mkdir()
    _write_synthetic_day(data_dir / "l2-BTCUSDT-2024-01-01.parquet", n_rows=20, base_price=100.0, ts_start=1_000_000)

    kwargs = dict(
        data_dir=data_dir,
        horizon_ticks=5,
        lookback_ticks=2,
        date_range=("2024-01-01", "2024-01-01"),
    )
    env1 = LOBExecutionEnv(**kwargs)
    fingerprint1 = _reset_fingerprint(env1, seed=42)
    assert len(env1._files) == 1

    # same simulated backfill event as the drift test above -- an older day
    # lands on disk in between the two "runs" (env instantiations)
    _write_synthetic_day(data_dir / "l2-BTCUSDT-2023-12-31.parquet", n_rows=20, base_price=999.0, ts_start=1)

    env2 = LOBExecutionEnv(**kwargs)
    fingerprint2 = _reset_fingerprint(env2, seed=42)

    # date_range excludes the newly-added day entirely -- file list length is
    # pinned, unlike the undated case above
    assert len(env2._files) == 1
    assert fingerprint1 == fingerprint2, (
        f"seed=42 resolved to a different window despite date_range pinning: "
        f"{fingerprint1} != {fingerprint2}"
    )


def test_date_range_excludes_files_outside_the_window(tmp_path):
    data_dir = tmp_path / "BTCUSDT"
    data_dir.mkdir()
    _write_synthetic_day(data_dir / "l2-BTCUSDT-2024-01-01.parquet", n_rows=20, base_price=100.0, ts_start=1)
    _write_synthetic_day(data_dir / "l2-BTCUSDT-2024-01-05.parquet", n_rows=20, base_price=200.0, ts_start=1)
    _write_synthetic_day(data_dir / "l2-BTCUSDT-2024-01-10.parquet", n_rows=20, base_price=300.0, ts_start=1)

    env = LOBExecutionEnv(
        data_dir=data_dir, horizon_ticks=5, lookback_ticks=2,
        date_range=("2024-01-02", "2024-01-09"),
    )
    assert [p.name for p in env._files] == ["l2-BTCUSDT-2024-01-05.parquet"]


def test_date_range_with_no_matching_files_raises(tmp_path):
    data_dir = tmp_path / "BTCUSDT"
    data_dir.mkdir()
    _write_synthetic_day(data_dir / "l2-BTCUSDT-2024-01-01.parquet", n_rows=20, base_price=100.0, ts_start=1)

    with pytest.raises(FileNotFoundError):
        LOBExecutionEnv(
            data_dir=data_dir, horizon_ticks=5, lookback_ticks=2,
            date_range=("2020-01-01", "2020-01-02"),
        )
