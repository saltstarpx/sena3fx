"""
data_pipeline/storage.py テスト。

# テスト用合成データ — 実市場データではありません
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from quant_bot.tests.conftest import make_synthetic_ohlcv
from quant_bot.data_pipeline.storage import ParquetStore


class TestParquetStore:

    @pytest.fixture
    def tmp_store(self, tmp_path):
        """一時ディレクトリを使った ParquetStore。"""
        return ParquetStore(root=tmp_path)

    def test_save_and_load_roundtrip(self, tmp_store):
        """保存したデータをロードすると同じデータが返ることを確認。"""
        # テスト用合成データ — 実市場データではありません
        df = make_synthetic_ohlcv(n=100)
        tmp_store.save(df, instrument="XAU_USD", granularity="H4")

        loaded = tmp_store.load(instrument="XAU_USD", granularity="H4")
        assert len(loaded) == len(df)
        assert list(loaded.columns) == list(df.columns)

    def test_index_name_preserved(self, tmp_store):
        """Parquet ラウンドトリップで index.name='datetime' が保持されることを確認。"""
        # テスト用合成データ — 実市場データではありません
        df = make_synthetic_ohlcv(n=100)
        assert df.index.name == "datetime"

        tmp_store.save(df, instrument="XAU_USD", granularity="H4")
        loaded = tmp_store.load(instrument="XAU_USD", granularity="H4")

        assert loaded.index.name == "datetime"

    def test_parquet_path_structure(self, tmp_store):
        """Parquet ファイルが正しいパス構造で保存されることを確認。"""
        # テスト用合成データ — 実市場データではありません
        df = make_synthetic_ohlcv(n=100, start="2024-03-01")
        paths = tmp_store.save(df, instrument="XAU_USD", granularity="H4")

        assert len(paths) > 0
        for p in paths:
            parts = p.parts
            assert "XAU_USD" in parts
            assert "H4" in parts
            assert p.suffix == ".parquet"

    def test_load_empty_when_no_data(self, tmp_store):
        """データがない場合は空のDataFrameを返すことを確認。"""
        loaded = tmp_store.load(instrument="XAU_USD", granularity="H4")
        assert len(loaded) == 0

    def test_date_filter_on_load(self, tmp_store):
        """ロード時に日付フィルタが機能することを確認。"""
        # テスト用合成データ — 実市場データではありません
        df = make_synthetic_ohlcv(n=200, start="2024-01-01")
        tmp_store.save(df, instrument="XAU_USD", granularity="H4")

        start_filter = pd.Timestamp("2024-06-01")
        loaded = tmp_store.load(
            instrument="XAU_USD",
            granularity="H4",
            start=start_filter,
        )
        if len(loaded) > 0:
            assert loaded.index[0] >= start_filter

    def test_no_timezone_in_index(self, tmp_store):
        """ロードしたデータのインデックスがタイムゾーンなしであることを確認。"""
        # テスト用合成データ — 実市場データではありません
        df = make_synthetic_ohlcv(n=50)
        tmp_store.save(df, instrument="XAG_USD", granularity="M15")
        loaded = tmp_store.load(instrument="XAG_USD", granularity="M15")
        if len(loaded) > 0:
            assert loaded.index.tz is None
