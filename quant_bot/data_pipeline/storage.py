"""
Parquet 分割ストレージ。

パーティション構成: {root}/{instrument}/{granularity}/{year}/{month:02d}/data.parquet
例: data/parquet/XAU_USD/H4/2025/03/data.parquet

既存 data/ohlc/ CSV を Parquet に移行する ingest_csv() も提供。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from .cleaner import clean_ohlcv

log = logging.getLogger(__name__)


class ParquetStore:
    """
    OHLCV データを Parquet 形式で year/month パーティションに保存・読み込みする。

    書き込み:  save() → 既存ファイルがあればマージ（重複排除 + ソート）して上書き
    読み込み:  load() → 該当パーティションを全部読んで連結・フィルタリング
    CSV移行:   ingest_csv() → data/ohlc/ の CSV を読んで Parquet に保存
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)

    # ------------------------------------------------------------------ #
    #  書き込み                                                            #
    # ------------------------------------------------------------------ #

    def save(
        self,
        df: pd.DataFrame,
        instrument: str,
        granularity: str,
    ) -> list[Path]:
        """
        DataFrame を year/month パーティションに Parquet 保存。

        Args:
            df:          OHLCVデータ (DatetimeIndex, UTC-naive)
            instrument:  'XAU_USD' 等
            granularity: 'H4' 等

        Returns:
            書き込んだ Path のリスト
        """
        if df.empty:
            return []

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("df は DatetimeIndex を持つ必要があります")

        written: list[Path] = []

        for (year, month), group in df.groupby(
            [df.index.year, df.index.month]
        ):
            part_path = (
                self.root
                / instrument
                / granularity
                / str(year)
                / f"{month:02d}"
                / "data.parquet"
            )
            part_path.parent.mkdir(parents=True, exist_ok=True)

            # 既存ファイルがあればマージ
            if part_path.exists():
                try:
                    existing = pd.read_parquet(part_path)
                    group = pd.concat([existing, group])
                except Exception as e:
                    log.warning(f"既存Parquet読み込み失敗 {part_path}: {e}")

            group = group[~group.index.duplicated(keep="last")].sort_index()
            group.to_parquet(part_path, engine="pyarrow", compression="snappy")
            written.append(part_path)
            log.debug(f"保存: {len(group)}行 → {part_path}")

        return written

    # ------------------------------------------------------------------ #
    #  読み込み                                                            #
    # ------------------------------------------------------------------ #

    def load(
        self,
        instrument: str,
        granularity: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Parquet パーティションを全部読んで連結・フィルタリング。

        Args:
            instrument:  'XAU_USD' 等
            granularity: 'H4' 等
            start:       開始日時 '2025-01-01' (inclusive)
            end:         終了日時 '2025-12-31' (inclusive)

        Returns:
            ソート・重複排除済み DataFrame。データがない場合は空DataFrame。
        """
        base = self.root / instrument / granularity
        if not base.exists():
            return pd.DataFrame()

        frames: list[pd.DataFrame] = []
        for parquet_file in sorted(base.rglob("data.parquet")):
            try:
                part_df = pd.read_parquet(parquet_file)
                frames.append(part_df)
            except Exception as e:
                log.warning(f"Parquet読み込みスキップ {parquet_file}: {e}")

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames)
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()

        # インデックス名を保持（data/ohlc/ CSV との互換性）
        if combined.index.name is None:
            combined.index.name = "datetime"

        if start:
            combined = combined[combined.index >= pd.Timestamp(start)]
        if end:
            combined = combined[combined.index <= pd.Timestamp(end)]

        return combined

    # ------------------------------------------------------------------ #
    #  CSV一括移行                                                         #
    # ------------------------------------------------------------------ #

    def ingest_csv(
        self,
        csv_path: str | Path,
        instrument: str,
        granularity: str,
    ) -> pd.DataFrame:
        """
        data/ohlc/ の既存CSV を Parquet に移行。

        CSV フォーマット (data/ohlc/ 実績):
          インデックス列: datetime (ISO 8601, UTC-naive)
          カラム: open, high, low, close, volume

        Args:
            csv_path:    CSV ファイルパス
            instrument:  'XAU_USD' 等
            granularity: 'H4' 等

        Returns:
            整形・保存後の DataFrame
        """
        csv_path = Path(csv_path)
        log.info(f"CSV取り込み開始: {csv_path}")

        df = pd.read_csv(
            str(csv_path),
            index_col="datetime",
            parse_dates=True,
        )
        df.index.name = "datetime"

        df_clean, report = clean_ohlcv(df)
        for w in report.warnings:
            log.warning(f"[ingest {csv_path.name}] {w}")
        log.info(f"整形完了: {report.summary()}")

        self.save(df_clean, instrument, granularity)
        log.info(f"Parquet保存完了: {instrument}/{granularity}")
        return df_clean
