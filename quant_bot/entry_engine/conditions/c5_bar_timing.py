"""
C5: 足更新タイミング — Bar Update Timing + HTF Direction

教材準拠:
  「足更新でポジションを取る（中途半端な時間にポジションを取らない）」
  「上位足の色に順張り」
  「4Hをお勧め: 同時に更新される足が多い」
  「アジア時間のブレイクアウトは危険」
  「Doji4本以上連続 → トレンドレス → ノーポジ」

ラップ元:
  lib/timing.detect_bar_update_timing()
  lib/timing.session_filter()
  lib/timing.higher_tf_direction()
  lib/candle.detect_trendless()

重要な注意:
  detect_bar_update_timing() は '15min', '15T', '1h', '1H', '4h', '4H' のみ認識する。
  それ以外の文字列はサイレントに「全バー更新」扱いになる（例外なし）。
  GRANULARITY_MAP で変換してから渡すこと。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from lib.timing import detect_bar_update_timing, higher_tf_direction, session_filter  # noqa: E402
from lib.candle import detect_trendless  # noqa: E402

from .base import ConditionBase, ConditionResult


class C5BarTiming(ConditionBase):
    """C5: 上位足更新タイミング + HTFトレンド方向の一致を判定。"""

    CONDITION_ID = "C5"

    def __init__(self, config: dict):
        """
        Args:
            config: entry_engine/config.yaml の 'c5' セクション dict

        config キー:
            require_trend: HTF 方向が不明（0.0）の場合に C5 不充足とするか (デフォルト True)
        """
        self._require_trend = bool(config.get("require_trend", True))

    def evaluate(
        self,
        ohlcv_df: pd.DataFrame,
        instrument: str,
        timeframe: str,
        timestamp: pd.Timestamp,
    ) -> ConditionResult:
        # ライブバーを除外
        confirmed = self._confirmed(ohlcv_df)
        if len(confirmed) < 5:
            return self._not_enough_data("C5: 確認バー不足（最低5本必要）")

        # OANDA granularity → timing.py freq 文字列に変換（必須）
        freq = self.GRANULARITY_MAP.get(timeframe, "1h")

        # 確定バーに対して各判定を実行
        bar_update_series = detect_bar_update_timing(confirmed, freq)
        is_update = bool(bar_update_series.iloc[-1])

        sessions = session_filter(confirmed)
        current_session = str(sessions.iloc[-1])

        htf_dir_series = higher_tf_direction(confirmed, freq)
        htf_direction_val = float(htf_dir_series.iloc[-1])

        trendless_series = detect_trendless(confirmed)
        is_trendless = bool(trendless_series.iloc[-1])

        # 充足条件: 足更新あり + トレンドレスでない
        satisfied = is_update and not is_trendless
        if self._require_trend:
            satisfied = satisfied and (htf_direction_val != 0.0)

        # アジア時間ブレイクアウトリスクフラグ（R8ルール）
        asia_breakout_risk = current_session == "asia" and is_update

        # スコア計算
        score = 0.0
        if is_update:
            score += 0.50
        if not is_trendless:
            score += 0.25
        if htf_direction_val != 0.0:
            score += 0.25

        trend_str = (
            "上昇"
            if htf_direction_val > 0
            else ("下降" if htf_direction_val < 0 else "不明")
        )
        direction = (
            "BULL"
            if htf_direction_val > 0
            else ("BEAR" if htf_direction_val < 0 else "NONE")
        )

        return ConditionResult(
            condition_id=self.CONDITION_ID,
            satisfied=satisfied,
            score=round(score, 3),
            reason=(
                f"C5: 足更新={'あり' if is_update else 'なし'}, "
                f"セッション={current_session}, HTFトレンド={trend_str}, "
                f"トレンドレス={'あり' if is_trendless else 'なし'}"
            ),
            details={
                "is_bar_update": is_update,
                "session": current_session,
                "htf_direction": htf_direction_val,
                "is_trendless": is_trendless,
                "asia_breakout_risk": asia_breakout_risk,
                "freq_mapped": freq,
                "direction": direction,
            },
        )
