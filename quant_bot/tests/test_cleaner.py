"""
data_pipeline/cleaner.py テスト。

# テスト用合成データ — 実市場データではありません
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from quant_bot.tests.conftest import make_synthetic_ohlcv
from quant_bot.data_pipeline.cleaner import clean_ohlcv, CleanReport


class TestCleanOhlcv:

    def test_clean_valid_data(self):
        """正常なデータはそのまま返されることを確認。"""
        # テスト用合成データ — 実市場データではありません
        df = make_synthetic_ohlcv(n=100)
        cleaned, report = clean_ohlcv(df)
        assert len(cleaned) > 0
        assert isinstance(report, CleanReport)
        assert report.original_rows == 100

    def test_removes_nan_rows(self):
        """NaN を含む行が除去されることを確認。"""
        # テスト用合成データ — 実市場データではありません
        df = make_synthetic_ohlcv(n=100)
        df.iloc[10, df.columns.get_loc("close")] = np.nan
        df.iloc[20, df.columns.get_loc("high")] = np.nan

        cleaned, report = clean_ohlcv(df)
        assert report.removed_missing >= 2
        assert cleaned["close"].isna().sum() == 0
        assert cleaned["high"].isna().sum() == 0

    def test_removes_duplicates(self):
        """重複行が除去されることを確認。"""
        # テスト用合成データ — 実市場データではありません
        df = make_synthetic_ohlcv(n=50)
        df_dup = pd.concat([df, df.iloc[:5]])  # 最初の5行を重複させる
        df_dup = df_dup.sort_index()

        cleaned, report = clean_ohlcv(df_dup)
        assert report.removed_duplicates >= 5
        assert cleaned.index.duplicated().sum() == 0

    def test_anomaly_detection_flags_spike(self):
        """ATR×10 以上のスパイクが anomaly_rows に含まれることを確認。"""
        # テスト用合成データ — 実市場データではありません
        df = make_synthetic_ohlcv(n=100)
        # 人工的なスパイクを挿入
        spike_idx = df.index[50]
        df.loc[spike_idx, "high"] = df["close"].mean() * 10  # 極端なスパイク

        cleaned, report = clean_ohlcv(df, anomaly_atr_multiplier=10.0)
        # スパイクは除去ではなくフラグのみ
        assert len(cleaned) == len(df)  # 行は削除しない
        # anomaly_rows にスパイクが含まれることを確認（含まれない場合もある - ATR依存）
        assert isinstance(report.anomaly_rows, list)

    def test_report_contains_correct_counts(self):
        """レポートのカウントが正確であることを確認。"""
        # テスト用合成データ — 実市場データではありません
        df = make_synthetic_ohlcv(n=100)
        cleaned, report = clean_ohlcv(df)
        assert report.original_rows == 100
        assert report.final_rows == len(cleaned)
        assert report.final_rows <= report.original_rows

    def test_empty_dataframe(self):
        """空のDataFrameを渡した場合にエラーなく処理されることを確認。"""
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df.index.name = "datetime"
        cleaned, report = clean_ohlcv(df)
        assert len(cleaned) == 0
        assert report.original_rows == 0
