"""
リスク管理モジュール — ポジションサイジング v3.0
=================================================
VolatilityAdjustedSizer : ATRに基づく動的サイジング（DD抑制）
KellyCriterionSizer     : フラクショナル・ケリー基準（リターン最大化）
HybridKellySizer        : Kelly × VolSizer ハイブリッド（v13新設）
LivermorePyramidingSizer: リバモア式価格ベースピラミッティング（v16新設）

使い方:
    kelly  = KellyCriterionSizer(kelly_fraction=0.25)
    sizer  = LivermorePyramidingSizer(base_sizer=kelly, step_pct=0.01,
                 pyramid_ratios=[0.5, 0.3, 0.2], max_pyramids=3)
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


class HybridKellySizer:
    """
    Kelly × VolatilityAdjusted ハイブリッドサイジング (v13)。

    Kellyの爆発力（高リターン）とVolSizerのDD抑制能力を両立。
    final_multiplier = kelly_multiplier × volatility_multiplier

    Args:
        kelly_fraction : フラクショナル係数（デフォルト0.5 — v12の0.25から引上げ）
        vol_lookback   : VolSizer の avg_atr 計算期間
        vol_clip_min   : VolSizer 乗数下限
        vol_clip_max   : VolSizer 乗数上限（高ボラ時の暴走防止）
        strategy_name  : performance_log.csv の参照戦略名
        base_risk_pct  : ベースrisk_pct
        global_clip_max: 最終乗数の上限（安全装置）
    """

    def __init__(self,
                 kelly_fraction: float = 0.5,
                 vol_lookback: int = 100,
                 vol_clip_min: float = 0.3,
                 vol_clip_max: float = 1.5,
                 strategy_name: str = 'Union_4H',
                 base_risk_pct: float = 0.05,
                 global_clip_max: float = 5.0):
        self.global_clip_max = global_clip_max

        self._kelly = KellyCriterionSizer(
            kelly_fraction=kelly_fraction,
            strategy_name=strategy_name,
            base_risk_pct=base_risk_pct,
        )
        self._vol = VolatilityAdjustedSizer(
            atr_lookback=vol_lookback,
            clip_min=vol_clip_min,
            clip_max=vol_clip_max,
        )

    def fit(self, atr_series: pd.Series) -> None:
        """BacktestEngine から呼び出し: VolSizer に ATR 系列を渡す。"""
        self._vol.fit(atr_series)

    def get_multiplier(self, i: int) -> float:
        """
        Kelly乗数 × VolSizer乗数 の積を返す。

        Returns:
            float: risk_pct に掛ける最終乗数
        """
        k_mult = self._kelly.get_multiplier(i)
        v_mult = self._vol.get_multiplier(i)
        return float(np.clip(k_mult * v_mult, 0.1, self.global_clip_max))

    def __repr__(self):
        return (f"HybridKellySizer("
                f"kelly={self._kelly}, "
                f"vol={self._vol}, "
                f"global_clip_max={self.global_clip_max})")


class LivermorePyramidingSizer:
    """
    リバモア式ピラミッティング・サイジング (v16)

    ジェシー・リバモアの手法を忠実に再現:
    「勝っているポジションを機械的に増やす。負けているポジションには追加しない。」

    ロジック:
      - 初期エントリー: base_sizer (例: KellyCriterionSizer) の乗数をそのまま使う
      - 追加エントリー: 価格がエントリー価格から step_pct × 累計回数 上昇するたびに追加
      - 追加ロット: pyramid_ratios に従い徐々に減少 (1st=0.5x, 2nd=0.3x, 3rd=0.2x)
      - 最大追加回数: max_pyramids で制限

    BacktestEngine との連携:
      - `fit(atr)`: base_sizer に ATR を渡す (自動呼び出し)
      - `get_multiplier(i)`: 初期エントリー時の乗数 (base_sizer に委譲)
      - `reset(entry_price, initial_size)`: 新規ポジション時に呼び出す
      - `on_bar(direction, current_price)`: 毎バーチェック → 追加ロット数を返す

    Args:
        base_sizer     : 初期エントリーのサイジングを担当するSizer
                         (KellyCriterionSizer 等, get_multiplier(i) を持つ)
        step_pct       : ピラミッドトリガー価格変動率 (デフォルト 1%)
                         例: 0.01 → エントリー価格の+1%ごとに追加
        pyramid_ratios : 各追加ロットの初期ロットに対する比率リスト
                         例: [0.5, 0.3, 0.2] → 1st=50%, 2nd=30%, 3rd=20%
        max_pyramids   : 最大追加回数 (pyramid_ratios の長さで自動制限)
    """

    def __init__(self,
                 base_sizer,
                 step_pct: float = 0.01,
                 pyramid_ratios: list | None = None,
                 max_pyramids: int = 3):
        self.base_sizer = base_sizer
        self.step_pct = step_pct
        self.pyramid_ratios = pyramid_ratios or [0.5, 0.3, 0.2]
        self.max_pyramids = min(max_pyramids, len(self.pyramid_ratios))

        # ポジション追跡 (reset() でリセット)
        self._entry_price: float | None = None
        self._initial_size: float = 0.0
        self._pyramid_count: int = 0

    # ── BacktestEngine API ──────────────────────────

    def fit(self, atr_series: pd.Series) -> None:
        """base_sizer に ATR 系列を渡す (BacktestEngine から自動呼び出し)。"""
        if hasattr(self.base_sizer, 'fit'):
            self.base_sizer.fit(atr_series)

    def get_multiplier(self, i: int) -> float:
        """初期エントリー時: base_sizer の乗数をそのまま返す。"""
        return self.base_sizer.get_multiplier(i)

    # ── 追加エントリー管理 ──────────────────────────

    def reset(self, entry_price: float, initial_size: float) -> None:
        """
        新規ポジションオープン時に呼び出す。
        BacktestEngine の新規エントリー直後に呼ばれる。

        Args:
            entry_price  : 初回エントリー価格
            initial_size : 初回ポジションサイズ (ロット)
        """
        self._entry_price = entry_price
        self._initial_size = initial_size
        self._pyramid_count = 0

    def on_bar(self, direction: str, current_price: float) -> float:
        """
        毎バー呼び出し。追加ピラミッドが必要なら追加ロット数を返す。

        価格が次のトリガーレベルに到達した場合のみ > 0 を返す。
        到達条件:
          long : current_price >= entry * (1 + step_pct * (count + 1))
          short: current_price <= entry * (1 - step_pct * (count + 1))

        Returns:
            float: 追加ロット数 (0.0 = 追加なし)
        """
        if (self._entry_price is None
                or self._pyramid_count >= self.max_pyramids
                or self._initial_size <= 0):
            return 0.0

        # 次のトリガー価格 (累積ステップ)
        trigger_move = self.step_pct * (self._pyramid_count + 1)

        if direction == 'long':
            trigger_price = self._entry_price * (1.0 + trigger_move)
            if current_price < trigger_price:
                return 0.0
        else:  # short
            trigger_price = self._entry_price * (1.0 - trigger_move)
            if current_price > trigger_price:
                return 0.0

        # 追加ロット = 初期サイズ × 対応比率
        ratio = self.pyramid_ratios[self._pyramid_count]
        add_size = max(0.01, round(self._initial_size * ratio, 2))
        self._pyramid_count += 1
        return add_size

    @property
    def pyramid_count(self) -> int:
        """現在の追加回数。"""
        return self._pyramid_count

    def __repr__(self):
        return (f"LivermorePyramidingSizer("
                f"base={self.base_sizer.__class__.__name__}, "
                f"step_pct={self.step_pct:.1%}, "
                f"ratios={self.pyramid_ratios}, "
                f"max_pyramids={self.max_pyramids})")
