"""
やがみ5条件 基底クラスとデータクラス定義。

============================================================
リーケージ防止の唯一の砦 (THE LEAKAGE CONTRACT)
============================================================

ohlcv_df.iloc[-1] は現在形成中の「ライブバー」です。
このバーの値を条件判定に使うことは「未来情報の使用」にあたり禁止です。

全 evaluate() 実装は先頭で self._confirmed(ohlcv_df) を呼び出し、
その戻り値の確定バーのみを使って判定を行います。

    # 正しい実装
    confirmed = self._confirmed(ohlcv_df)   # iloc[:-1] を返す
    last_bar = confirmed.iloc[-1]            # 最後の確定バー

    # 禁止
    last_bar = ohlcv_df.iloc[-1]            # ← ライブバー = 未来情報

テストファイル test_base_leakage.py がセンチネル値注入でこれを検証します。
============================================================
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class ConditionResult:
    """1つの条件の評価結果。"""

    condition_id: str
    """条件ID: 'C1' 〜 'C5'"""

    satisfied: bool
    """条件が充足されているか"""

    score: float
    """スコア 0.0 〜 1.0 (3桁丸め)"""

    reason: str
    """人間が読める判定理由（日本語推奨）"""

    details: dict
    """デバッグ用詳細情報"""

    non_textbook: bool = False
    """教材外ルールかどうか（デフォルト False = 教材準拠）"""


class ConditionBase(ABC):
    """
    やがみ5条件の抽象基底クラス。

    サブクラスは evaluate() を実装し、先頭で _confirmed() を呼び出すこと。

    OANDA granularity → lib/timing.py freq 文字列の変換は GRANULARITY_MAP で行う。
    detect_bar_update_timing() は '15min', '15T', '1h', '1H', '4h', '4H' のみ受け付け、
    それ以外は全バーを更新タイミングとして扱う（サイレントエラー）。
    """

    CONDITION_ID: str = "C?"

    # OANDA granularity コード → lib/timing.py の freq 文字列
    GRANULARITY_MAP: dict[str, str] = {
        "M15": "15min",
        "H1": "1h",
        "H4": "4h",
        "H8": "4h",   # 8Hはtiming.pyにネイティブサポートなし → 4h扱い
        "D": "1D",
        # パススルー（既にマッピング済みの値）
        "15min": "15min",
        "15T": "15min",
        "1h": "1h",
        "1H": "1h",
        "4h": "4h",
        "4H": "4h",
        "1D": "1D",
    }

    @staticmethod
    def _confirmed(ohlcv_df: pd.DataFrame) -> pd.DataFrame:
        """
        確定バーのみを返す (ライブバーを除外)。

        ohlcv_df.iloc[-1] がライブバー（形成中）であり、
        このメソッドはそれを除いた iloc[:-1] を返します。

        Returns:
            確定バーのみの DataFrame。
            len(ohlcv_df) <= 1 の場合は空DataFrame（同じ列構成）。
        """
        if len(ohlcv_df) <= 1:
            return ohlcv_df.iloc[:0]
        return ohlcv_df.iloc[:-1]

    @staticmethod
    def _calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        True Range の rolling mean を計算。

        lib/candle.py, lib/backtest.py と同じ計算式を使用。
        確定バーのスライスに対して安全に呼び出せる。

        Args:
            df:     OHLCV DataFrame
            period: ATR 計算期間（デフォルト 14）

        Returns:
            ATR Series (df.index と同じインデックス)
            iloc[-1] が最後の確定バーの ATR 値。
        """
        h = df["high"].values
        l = df["low"].values
        c = df["close"].values

        prev_c = pd.Series(c).shift(1).values.copy()
        prev_c[0] = c[0]

        tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
        return pd.Series(tr, index=df.index).rolling(period).mean()

    def _not_enough_data(self, reason: str = "データ不足") -> ConditionResult:
        """データ不足の場合に返す標準的な ConditionResult。"""
        return ConditionResult(
            condition_id=self.CONDITION_ID,
            satisfied=False,
            score=0.0,
            reason=reason,
            details={"error": "insufficient_data"},
        )

    @abstractmethod
    def evaluate(
        self,
        ohlcv_df: pd.DataFrame,
        instrument: str,
        timeframe: str,
        timestamp: pd.Timestamp,
    ) -> ConditionResult:
        """
        指定バーの条件を評価する。

        Args:
            ohlcv_df:   ライブバー（iloc[-1]）を含む完全な OHLCV DataFrame。
                        先頭で必ず self._confirmed(ohlcv_df) を呼び出すこと。
            instrument: 銘柄コード 'XAU_USD' 等
            timeframe:  OANDA granularity 'M15', 'H1', 'H4', 'D'
            timestamp:  評価するバーのタイムスタンプ
                        （scanner が渡す: ライブバーの index 値）

        Returns:
            ConditionResult
        """
        raise NotImplementedError
