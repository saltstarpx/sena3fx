"""
リスク管理モジュール — ポジションサイジング v1.0
=================================================
VolatilityAdjustedSizer: ATRに基づく動的サイジング（DD抑制）
KellyCriterionSizer    : フラクショナル・ケリー基準（リターン最大化）

使い方:
    sizer = VolatilityAdjustedSizer(atr_lookback=100)
    engine.run(data=df, signal_func=sig, sizer=sizer, ...)
"""

import numpy as np
import pandas as pd
import os
import csv


class VolatilityAdjustedSizer:
    """
    ATR正規化によるボラティリティ調整サイジング。

    高ボラ時にポジションを縮小、低ボラ時に拡大することで
    リスクを安定化しMDDを抑制する。

    式: multiplier = avg_atr / current_atr
        (= base_size / (current_atr / avg_atr))

    Args:
        atr_lookback: avg_atr 計算に使う期間（デフォルト100）
        clip_min: 乗数下限（デフォルト0.25 — 最大4倍縮小）
        clip_max: 乗数上限（デフォルト2.0  — 最大2倍拡大）
    """

    def __init__(self, atr_lookback: int = 100,
                 clip_min: float = 0.25, clip_max: float = 2.0):
        self.atr_lookback = atr_lookback
        self.clip_min = clip_min
        self.clip_max = clip_max
        self._atr_series: pd.Series | None = None

    def fit(self, atr_series: pd.Series) -> None:
        """バックテスト開始前にATR系列をセット（BacktestEngineから自動呼び出し）。"""
        self._atr_series = atr_series

    def get_multiplier(self, i: int) -> float:
        """
        インデックス i のバーに対するリスク乗数を返す。

        Returns:
            float: risk_pct に掛ける乗数（1.0 = 変更なし）
        """
        if self._atr_series is None:
            return 1.0
        current_atr = self._atr_series.iloc[i]
        if np.isnan(current_atr) or current_atr <= 0:
            return 1.0

        start = max(0, i - self.atr_lookback)
        window = self._atr_series.iloc[start:i + 1].dropna()
        if len(window) < 2:
            return 1.0

        avg_atr = window.mean()
        if avg_atr <= 0:
            return 1.0

        mult = avg_atr / current_atr
        return float(np.clip(mult, self.clip_min, self.clip_max))

    def __repr__(self):
        return (f"VolatilityAdjustedSizer("
                f"lookback={self.atr_lookback}, "
                f"clip=[{self.clip_min}, {self.clip_max}])")


class KellyCriterionSizer:
    """
    フラクショナル・ケリー基準によるサイジング。

    バックテスト実績から最適投資比率を計算し、
    安全係数 (kelly_fraction) で保守化する。

    式:
        b = PF * (1-p) / p      # ペイオフレシオ
        f* = p - (1-p) / b      # = p * (1 - 1/PF)
        final = f* × kelly_fraction

    Args:
        win_rate  : 勝率 (0〜1)。Noneの場合はperformance_log.csvから自動読込。
        profit_factor: プロフィットファクター。Noneの場合は同上。
        kelly_fraction: フラクショナル係数（デフォルト0.25）
        strategy_name : performance_log.csv の参照戦略名
        perf_log_path : performance_log.csv のパス
        base_risk_pct : ベースrisk_pct（この値にKelly乗数をかけた値が最終値）
    """

    def __init__(self,
                 win_rate: float | None = None,
                 profit_factor: float | None = None,
                 kelly_fraction: float = 0.25,
                 strategy_name: str = 'Union_4H',
                 perf_log_path: str | None = None,
                 base_risk_pct: float = 0.05):
        self.kelly_fraction = kelly_fraction
        self.strategy_name = strategy_name
        self.base_risk_pct = base_risk_pct

        # win_rate / profit_factor が未指定なら performance_log.csv から読込
        if win_rate is None or profit_factor is None:
            wr, pf = self._load_from_log(perf_log_path, strategy_name)
            self._win_rate = wr if win_rate is None else win_rate
            self._profit_factor = pf if profit_factor is None else profit_factor
        else:
            self._win_rate = win_rate
            self._profit_factor = profit_factor

        self._kelly_f = self._calc_kelly()

    def _load_from_log(self, path: str | None, name: str):
        """performance_log.csv から最新の win_rate, profit_factor を取得。"""
        if path is None:
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            path = os.path.join(root, 'results', 'performance_log.csv')

        if not os.path.exists(path):
            return 0.6, 2.0  # デフォルト値

        best = None
        try:
            with open(path, newline='', encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    if row.get('strategy_name', '') == name:
                        best = row  # 最終行が最新
        except Exception:
            pass

        if best is None:
            # 部分一致フォールバック
            try:
                with open(path, newline='', encoding='utf-8') as f:
                    for row in csv.DictReader(f):
                        sn = row.get('strategy_name', '')
                        if name.split('_')[0] in sn:
                            best = row
            except Exception:
                pass

        if best is None:
            return 0.6, 2.0

        try:
            wr = float(best['win_rate']) / 100.0  # CSV は % 表記
            pf = float(best['profit_factor'])
            return wr, pf
        except (KeyError, ValueError):
            return 0.6, 2.0

    def _calc_kelly(self) -> float:
        """ケリー比率を計算して返す。"""
        p = self._win_rate
        pf = self._profit_factor
        if p <= 0 or p >= 1 or pf <= 0:
            return 0.0
        # b = PF * (1-p) / p
        # f* = p - (1-p)/b = p - p/PF = p*(1 - 1/PF)
        f_star = p * (1.0 - 1.0 / pf)
        f_star = max(0.0, f_star)
        return round(f_star * self.kelly_fraction, 4)

    @property
    def kelly_f(self) -> float:
        return self._kelly_f

    def get_multiplier(self, i: int = 0) -> float:
        """
        Kelly乗数を risk_pct の乗数として返す。

        Returns:
            float: final_ratio / base_risk_pct
        """
        if self._kelly_f <= 0 or self.base_risk_pct <= 0:
            return 1.0
        return float(np.clip(self._kelly_f / self.base_risk_pct, 0.1, 3.0))

    def fit(self, atr_series: pd.Series) -> None:
        """VolatilityAdjustedSizerとのAPI互換用（何もしない）。"""
        pass

    def __repr__(self):
        return (f"KellyCriterionSizer("
                f"WR={self._win_rate:.1%}, PF={self._profit_factor:.3f}, "
                f"f*={self._kelly_f:.4f}, fraction={self.kelly_fraction})")
