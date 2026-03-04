"""
ピラミッティングロジック — リバモア式ポジション追加サイジング
=============================================================
LivermorePyramidingSizer : 価格ベースの段階的ポジション追加（v16実装）

ジェシー・リバモアの格言:
  「勝っているポジションを機械的に増やす。負けているポジションには追加しない。」

使い方:
    from lib.pyramiding_logic import LivermorePyramidingSizer
    from lib.risk_manager import KellyCriterionSizer

    kelly  = KellyCriterionSizer(kelly_fraction=0.25)
    sizer  = LivermorePyramidingSizer(base_sizer=kelly, step_pct=0.01,
                 pyramid_ratios=[0.5, 0.3, 0.2], max_pyramids=3)
    engine.run(data=df, signal_func=sig, sizer=sizer, ...)
"""

import pandas as pd
import numpy as np


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
