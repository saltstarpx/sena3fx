"""
エントリースキャナー — リーケージフリーウォークフォワードスキャン。

scan_dataframe() は ohlcv_df を 1 本ずつ前進させながら
5条件を評価し、シグナルを JSONL 形式で yield する。

設計上の保証:
  - bar[i] をライブバー（未確定）として扱う
  - 各条件は内部で _confirmed() を呼び出し iloc[-1] をドロップ
  - よって sentinel を iloc[-1] に注入してもスコアに影響しない
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional

import pandas as pd

from .conditions.base import ConditionResult
from .conditions.c1_resisup import C1ResistanceSupport
from .conditions.c2_candle import C2CandleStrength
from .conditions.c3_price_action import C3PriceAction
from .conditions.c4_chart_pattern import C4ChartPattern
from .conditions.c5_bar_timing import C5BarTiming
from .scorer import ConditionScore, ConditionScorer

log = logging.getLogger("quant_bot.entry_engine")


class EntryScanner:
    """5条件ウォークフォワードスキャナー。"""

    def __init__(self, config: dict):
        """
        Args:
            config: entry_engine/config.yaml の全 dict
        """
        cond_cfg = config.get("conditions", {})
        scorer_cfg = config.get("scorer", {})
        scan_cfg = config.get("scanner", {})

        self._c1 = C1ResistanceSupport(cond_cfg.get("c1", {}))
        self._c2 = C2CandleStrength(cond_cfg.get("c2", {}))
        self._c3 = C3PriceAction(cond_cfg.get("c3", {}))
        self._c4 = C4ChartPattern(cond_cfg.get("c4", {}))
        self._c5 = C5BarTiming(cond_cfg.get("c5", {}))
        self._scorer = ConditionScorer(scorer_cfg)

        self._warmup = int(scan_cfg.get("warmup_bars", 60))
        self._sl_atr_mult = float(scan_cfg.get("sl_atr_mult", 2.0))
        self._tp_atr_mult = float(scan_cfg.get("tp_atr_mult", 4.0))
        self._signal_only = bool(scan_cfg.get("signal_only", True))

    # ------------------------------------------------------------------ #
    #  メインAPI                                                           #
    # ------------------------------------------------------------------ #

    def evaluate_bar(
        self,
        slice_df: pd.DataFrame,
        instrument: str,
        timeframe: str,
        timestamp,
    ) -> Optional[dict]:
        """
        単一バーを評価してシグナルレコードを返す。

        Args:
            slice_df:   ohlcv_df.iloc[:i+1] — bar[i] がライブバー
            instrument: 'XAU_USD' 等
            timeframe:  'H4' / 'M15' 等
            timestamp:  bar[i] のタイムスタンプ

        Returns:
            JSONL レコード dict または None (NO_SIGNAL かつ signal_only=True)
        """
        results: list[ConditionResult] = [
            self._c1.evaluate(slice_df, instrument, timeframe, timestamp),
            self._c2.evaluate(slice_df, instrument, timeframe, timestamp),
            self._c3.evaluate(slice_df, instrument, timeframe, timestamp),
            self._c4.evaluate(slice_df, instrument, timeframe, timestamp),
            self._c5.evaluate(slice_df, instrument, timeframe, timestamp),
        ]

        # confirmed_df は scorer の SL/TP 計算用 (ライブバーを除いた末尾スライス)
        confirmed_df = slice_df.iloc[:-1] if len(slice_df) >= 2 else slice_df.iloc[:0]

        score: ConditionScore = self._scorer.score(
            results,
            confirmed_df,
            sl_atr_mult=self._sl_atr_mult,
            tp_atr_mult=self._tp_atr_mult,
        )

        if self._signal_only and score.signal is None:
            return None

        return self._to_record(score, instrument, timeframe, timestamp)

    def scan_dataframe(
        self,
        ohlcv_df: pd.DataFrame,
        instrument: str,
        timeframe: str,
    ) -> Generator[dict, None, None]:
        """
        DataFrame 全体をウォークフォワードスキャン。

        Args:
            ohlcv_df:   OHLCV DataFrame (index=datetime, columns=[open,high,low,close,volume])
            instrument: 'XAU_USD' 等
            timeframe:  'H4' / 'M15' 等

        Yields:
            JSONL 形式のシグナルレコード dict
        """
        n = len(ohlcv_df)
        warmup = max(self._warmup, 25)  # c4 は 25 本以上必要

        if n <= warmup:
            log.warning(
                f"データ不足: n={n} <= warmup={warmup}。スキャンをスキップ。"
            )
            return

        log.info(
            f"スキャン開始: {instrument} {timeframe}, "
            f"bars={n}, warmup={warmup}, scan_bars={n - warmup}"
        )

        for i in range(warmup, n):
            slice_df = ohlcv_df.iloc[: i + 1]  # bar[i] = ライブバー
            timestamp = ohlcv_df.index[i]

            record = self.evaluate_bar(slice_df, instrument, timeframe, timestamp)
            if record is not None:
                yield record

    def scan_to_jsonl(
        self,
        ohlcv_df: pd.DataFrame,
        instrument: str,
        timeframe: str,
        output_path: str | Path,
    ) -> int:
        """
        scan_dataframe() の結果を JSONL ファイルに書き出す。

        Returns:
            書き出したレコード数
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        count = 0
        with output_path.open("w", encoding="utf-8") as f:
            for record in self.scan_dataframe(ohlcv_df, instrument, timeframe):
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

        log.info(f"JSONL 書き出し完了: {output_path} ({count} レコード)")
        return count

    # ------------------------------------------------------------------ #
    #  内部ヘルパー                                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _to_record(
        score: ConditionScore,
        instrument: str,
        timeframe: str,
        timestamp,
    ) -> dict:
        """ConditionScore → JSONL レコード dict 変換。"""
        if isinstance(timestamp, (pd.Timestamp, datetime)):
            ts_str = timestamp.isoformat()
        else:
            ts_str = str(timestamp)

        conditions_dict = {}
        for cid, cr in score.conditions.items():
            conditions_dict[cid] = {
                "satisfied": cr.satisfied,
                "score": round(cr.score, 4),
                "reason": cr.reason,
                "non_textbook": cr.non_textbook,
                "details": {
                    k: v
                    for k, v in cr.details.items()
                    if isinstance(v, (str, int, float, bool, type(None)))
                },
            }

        return {
            "timestamp": ts_str,
            "instrument": instrument,
            "timeframe": timeframe,
            "total_score": score.total_score,
            "grade": score.grade,
            "signal": score.signal,
            "strength": score.strength,
            "bull_score": score.bull_score,
            "bear_score": score.bear_score,
            "conditions": conditions_dict,
            "suggested_sl": score.suggested_sl,
            "suggested_tp": score.suggested_tp,
            "risk_reward": score.risk_reward,
        }
