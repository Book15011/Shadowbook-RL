import pandas as pd

from src.data.download_manager import bulk_download, download_and_verify


def test_bulk_download_concatenates_frames(tmp_path):
    out_dir = tmp_path / "raw"
    out_dir.mkdir()
    result = bulk_download("BTCUSDT", "bookDepth", "2024-01-01", "2024-01-02", out_dir)
    assert isinstance(result, pd.DataFrame)
    assert result.empty
