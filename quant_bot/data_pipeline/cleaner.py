"""
OHLCV データ品質チェックモジュール。

純粋関数 (pure functions) のみ — 入力DataFrameは変更しない。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd


@dataclass
class CleanReport:
    original_rows: int
    final_rows: int
    removed_duplicates: int
    removed_missing: int
    anomaly_rows: List[pd.Timestamp] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"元データ: {self.original_rows}行 → 最終: {self.final_rows}行",
            f"  NaN除去: {self.removed_missing}行, 重複除去: {self.removed_duplicates}行",
        ]
        if self.anomaly_rows:
            lines.append(f"  異常値フラグ: {len(self.anomaly_rows)}行")
        for w in self.warnings:
            lines.append(f"  [警告] {w}")
        return "\n".join(lines)


def clean_ohlcv(
    df: pd.DataFrame,
    anomaly_atr_multiplier: float = 10.0,
    atr_period: int = 14,
) -> tuple[pd.DataFrame, CleanReport]:
    """
    OHLCVデータの品質チェックと整形。

    実施内容:
      1. open/high/low/close の NaN 行を削除
      2. インデックスを昇順ソート + 重複除去（後の値を保持）
      3. OHLC制約チェック: high >= max(open,close), low <= min(open,close)
         → 違反行は削除せずに警告のみ
      4. バー幅が ATR × anomaly_atr_multiplier を超える行を異常値フラグ
         → 削除しない（ギャップオープンは実際の市場イベント）

    Args:
        df: 元のOHLCV DataFrame (DatetimeIndex必須)
        anomaly_atr_multiplier: 異常値判定のATR倍率
        atr_period: ATR計算期間

    Returns:
        (整形済みDataFrame, CleanReport)
    """
    report = CleanReport(
        original_rows=len(df),
        final_rows=0,
        removed_duplicates=0,
        removed_missing=0,
    )

    # Step 1: NaN除去
    before = len(df)
    df = df.dropna(subset=["open", "high", "low", "close"])
    report.removed_missing = before - len(df)

    # Step 2: ソート + 重複除去
    df = df.sort_index()
    before = len(df)
    df = df[~df.index.duplicated(keep="last")]
    report.removed_duplicates = before - len(df)

    if df.empty:
        report.final_rows = 0
        return df, report

    # Step 3: OHLC制約チェック（警告のみ）
    body_high = df[["open", "close"]].max(axis=1)
    body_low = df[["open", "close"]].min(axis=1)
    invalid_mask = (df["high"] < body_high) | (df["low"] > body_low)
    if invalid_mask.any():
        n = int(invalid_mask.sum())
        report.warnings.append(
            f"{n}行でOHLC制約違反 (high<実体上端 or low>実体下端)"
        )

    # Step 4: ATRベース異常値検出
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(atr_period).mean()
    bar_range = df["high"] - df["low"]

    with np.errstate(invalid="ignore"):
        anomaly_mask = (bar_range > atr * anomaly_atr_multiplier) & atr.notna()

    if anomaly_mask.any():
        anomaly_ts = list(df.index[anomaly_mask])
        report.anomaly_rows = anomaly_ts
        report.warnings.append(
            f"{len(anomaly_ts)}行が異常値フラグ "
            f"(バー幅 > ATR × {anomaly_atr_multiplier}倍)"
        )

    report.final_rows = len(df)
    return df, report
