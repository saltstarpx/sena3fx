"""
C3: プライスアクション — Price Action Pattern

教材準拠:
  「二番底/二番天井しか触るな。横軸を待て」
  「リバーサルロー/ハイ: 急落後にV字回復」
  「実体揃いでサポート/レジスタンス確認」

ラップ元: lib/candle.detect_price_action()

重要な注意:
  detect_price_action() の内部 (lib/candle.py line 344):
    result = bars.copy() if 'candle_type' not in bars.columns else bars
  C2 が enriched した DataFrame を C3 に渡すと candle_type 列が存在するため
  bars を copy せず参照が共有されてしまう。
  C3 は必ず self._confirmed(ohlcv_df) の新鮮なスライスを渡すこと。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from lib.candle import detect_price_action  # noqa: E402

from .base import ConditionBase, ConditionResult

# ロング根拠になるプライスアクションシグナル
BULL_PA = frozenset(
    {"reversal_low", "double_bottom", "body_align_support", "wick_fill_bull"}
)
# ショート根拠になるプライスアクションシグナル
BEAR_PA = frozenset(
    {"reversal_high", "double_top", "body_align_resist", "wick_fill_bear"}
)
# スラストは方向あり but lib/yagami.py では C3 スコアに弱い寄与
BULL_WEAK_PA = frozenset({"thrust_up"})
BEAR_WEAK_PA = frozenset({"thrust_down"})


class C3PriceAction(ConditionBase):
    """C3: 複数足にまたがるプライスアクションパターンを判定。"""

    CONDITION_ID = "C3"

    def __init__(self, config: dict):
        """
        Args:
            config: entry_engine/config.yaml の 'c3' セクション dict

        注意: double_bottom_tolerance_pct や require_horizontal_time は
             lib/candle.py にはパラメータとして公開されていない。
             将来の lib 拡張用にドキュメント目的で保持。
        """
        self._dbl_tolerance = float(config.get("double_bottom_tolerance_pct", 0.3))
        self._require_hor_time = bool(config.get("require_horizontal_time", True))

    def evaluate(
        self,
        ohlcv_df: pd.DataFrame,
        instrument: str,
        timeframe: str,
        timestamp: pd.Timestamp,
    ) -> ConditionResult:
        # ライブバーを除外
        confirmed = self._confirmed(ohlcv_df)
        if len(confirmed) < 5:  # detect_price_action は i=4 から開始
            return self._not_enough_data("C3: 確認バー不足（最低5本必要）")

        # 注意: C2 の enriched DataFrame を渡してはいけない（上記ドキュメント参照）
        # self._confirmed(ohlcv_df) から新鮮なスライスを使う
        enriched = detect_price_action(confirmed)
        last = enriched.iloc[-1]

        pa_signal = last.get("pa_signal")  # str or None
        pa_strength = float(last.get("pa_strength", 0.0))

        is_bull = pa_signal in BULL_PA or pa_signal in BULL_WEAK_PA
        is_bear = pa_signal in BEAR_PA or pa_signal in BEAR_WEAK_PA
        satisfied = pa_signal in BULL_PA or pa_signal in BEAR_PA  # 弱シグナルは充足とみなさない
        direction = "BULL" if is_bull else ("BEAR" if is_bear else "NONE")

        return ConditionResult(
            condition_id=self.CONDITION_ID,
            satisfied=satisfied,
            score=round(abs(pa_strength), 3),
            reason=f"C3: {pa_signal or 'パターン未検出'}, 強度={pa_strength:+.2f} ({direction})",
            details={
                "pa_signal": pa_signal,
                "pa_strength": pa_strength,
                "direction": direction,
                "bar_timestamp": str(last.name),
            },
        )
