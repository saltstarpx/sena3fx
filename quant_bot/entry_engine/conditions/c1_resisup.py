"""
C1: レジサポの位置 — Support/Resistance Level Proximity

教材準拠:
  「ローソク足の髭先、髭と実体の間に水平線を引く」
  「反発回数が多い程、強固なレジサポ」
  「自分がトレードしている時間軸を大体12倍すると丁度いい」

ラップ元: lib/levels.extract_levels(), lib/levels.is_at_level()

重要な注意:
  config の atr_multiplier は is_at_level() の proximity_mult パラメータに対応する。
  extract_levels() の tolerance_atr_mult (クラスタリング精度) とは別物。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from lib.levels import extract_levels, is_at_level  # noqa: E402

from .base import ConditionBase, ConditionResult


class C1ResistanceSupport(ConditionBase):
    """C1: 現在価格がレジサポレベルの近傍にあるか判定。"""

    CONDITION_ID = "C1"

    def __init__(self, config: dict):
        """
        Args:
            config: entry_engine/config.yaml の 'c1' セクション dict

        config キー:
            atr_multiplier:   is_at_level() の proximity_mult に対応 (デフォルト 1.5)
                              「現在価格がレベルから ATR × atr_multiplier 以内」が充足条件
            min_touch_count:  extract_levels() の min_touches (デフォルト 2)
            level_lookback:   レベル抽出に使う確定バー数 (デフォルト 100)
        """
        # config キー名 'atr_multiplier' → 実際の関数引数名 is_at_level(..., proximity_mult)
        self._proximity_mult: float = float(config.get("atr_multiplier", 1.5))
        self._min_touches: int = int(config.get("min_touch_count", 2))
        self._lookback: int = int(config.get("level_lookback", 100))

    def evaluate(
        self,
        ohlcv_df: pd.DataFrame,
        instrument: str,
        timeframe: str,
        timestamp: pd.Timestamp,
    ) -> ConditionResult:
        # ライブバーを除外（リーケージ防止）
        confirmed = self._confirmed(ohlcv_df)
        if len(confirmed) < 20:
            return self._not_enough_data("C1: 確認バー不足（最低20本必要）")

        # 直近 lookback 本でレベルを抽出
        # 注意: extract_levels() は内部で c[-1] を現在価格として使う。
        # 確定バーのスライスを渡すことで正しい現在価格になる。
        bars_for_levels = confirmed.iloc[-self._lookback:]
        last_bar = confirmed.iloc[-1]
        current_price = float(last_bar["close"])

        # ATR 計算（確定バーのスライスで）
        atr_series = self._calc_atr(bars_for_levels)
        current_atr = float(atr_series.iloc[-1])
        if np.isnan(current_atr) or current_atr == 0:
            return self._not_enough_data("C1: ATR計算失敗")

        # レジサポレベル抽出
        # tolerance_atr_mult: クラスタリング許容誤差（固定 0.3）
        # min_touches: クラスタの最低構成数 = タッチ回数
        levels = extract_levels(
            bars_for_levels,
            tolerance_atr_mult=0.3,
            min_touches=self._min_touches,
            atr_period=14,
        )

        if not levels:
            return ConditionResult(
                condition_id=self.CONDITION_ID,
                satisfied=False,
                score=0.0,
                reason="C1: レジサポ未検出",
                details={
                    "levels_found": 0,
                    "current_price": current_price,
                    "direction": "NONE",
                },
            )

        # 近傍チェック
        # is_at_level(price, levels, atr, proximity_mult)
        at_level, level_type = is_at_level(
            current_price, levels, current_atr, self._proximity_mult
        )

        # 最近傍レベルを取得
        nearest = min(levels, key=lambda lv: abs(lv["level"] - current_price))
        dist = abs(current_price - nearest["level"])

        # スコア計算
        if at_level:
            score = round(float(nearest["strength"]), 3)
        else:
            # 部分スコア: 3ATR離れで 0 に減衰
            score = round(max(0.0, 1.0 - dist / (current_atr * 3.0)), 3)
        score = min(1.0, score)

        direction = (
            "BULL" if level_type == "support"
            else ("BEAR" if level_type == "resistance" else "NONE")
        )

        return ConditionResult(
            condition_id=self.CONDITION_ID,
            satisfied=at_level,
            score=score,
            reason=(
                f"C1: {level_type or '未検出'}レベル ${nearest['level']:.2f}, "
                f"距離={dist:.2f} (ATR比{dist / current_atr:.2f}倍)"
            ),
            details={
                "level": nearest["level"],
                "level_type": level_type,
                "touches": nearest["touches"],
                "distance_atr": nearest.get("distance_atr"),
                "strength": nearest["strength"],
                "current_price": current_price,
                "atr": round(current_atr, 4),
                "levels_found": len(levels),
                "direction": direction,
            },
        )
