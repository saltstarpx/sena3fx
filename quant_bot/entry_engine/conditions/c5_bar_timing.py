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
        # パフォーマンス最適化: HTF方向判定に50バー以上あれば十分 (4H×50=8日分)
        confirmed = confirmed.iloc[-50:]

        # OANDA granularity → timing.py freq 文字列に変換（必須）
        freq = self.GRANULARITY_MAP.get(timeframe, "1h")

        # 確定バーに対して各判定を実行
        bar_update_series = detect_bar_update_timing(confirmed, freq)
        is_midnight_update = bool(bar_update_series.iloc[-1])  # 日足更新 (0:00 UTC)

        sessions = session_filter(confirmed)
        current_session = str(sessions.iloc[-1])

        # ロンドン/NYオープンの足更新判定 (XAUUSD向け追加)
        # 「4Hをお勧め: 同時に更新される足が多い」+ アジア時間除外
        # - 0:00 UTC (東京): detect_bar_update_timing が検出、アジア session → ブロック
        # - 8:00 UTC (ロンドンオープン): 流動性大、XAUUSDに最重要
        # - 12:00 UTC (NY重複): 最もボラ高いセッション開始
        last_bar_hour = int(confirmed.index[-1].hour)
        is_london_ny_update = (freq in ("4h", "4H")) and (last_bar_hour in (8, 12))
        is_update = is_midnight_update or is_london_ny_update

        htf_dir_series = higher_tf_direction(confirmed, freq)

        # 日足更新の瞬間 (0:00 UTC) のみ: 新しい日足の最初の1本のため
        # higher_tf_direction が open==close → 0.0 を返す。
        # ロンドン/NYオープンは複数バーが蓄積されているため iloc[-1] を使用。
        if is_midnight_update and not is_london_ny_update and len(htf_dir_series) >= 2:
            htf_direction_val = float(htf_dir_series.iloc[-2])
        else:
            htf_direction_val = float(htf_dir_series.iloc[-1])

        trendless_series = detect_trendless(confirmed)
        is_trendless = bool(trendless_series.iloc[-1])

        # アジア時間ブレイクアウトリスクフラグ（R8ルール）
        # 「アジア時間のブレイクアウトは危険」→ アジアセッション更新は除外
        asia_breakout_risk = current_session == "asia" and is_update

        # 充足条件: 足更新あり + アジアセッション除外 + トレンドレスでない
        satisfied = is_update and not asia_breakout_risk and not is_trendless
        if self._require_trend:
            satisfied = satisfied and (htf_direction_val != 0.0)

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
