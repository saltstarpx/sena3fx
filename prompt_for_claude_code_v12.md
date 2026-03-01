# Claude Code 指示書 v12: 1000万→1億を目指すための3つの新エンジン実装

## Task 1: VolatilityAdjustedSizer
lib/risk_manager.py に VolatilityAdjustedSizer クラスを新設。
ATR(14)を正規化し、position_size = base_size / (current_atr / avg_atr) で動的調整。
strategies/union_strategy.py のバックテストに適用。

## Task 2: KellyCriterionSizer
lib/risk_manager.py に KellyCriterionSizer クラスを新設。
f* = (b*p - (1-p)) / b、フラクショナル係数0.25を採用。
performance_log.csv から最新Union_4Hの勝率/PFを自動読込。

## Task 3: レジーム転換モデル (HMM)
lib/regime.py → HiddenMarkovRegimeDetector (2状態: レンジ/トレンド)
strategies/meta_strategy.py → MetaStrategy (レンジ→YagamiA, トレンド→Union)
requirements.txt に hmmlearn を追加。

## Task 4: ダッシュボード拡張
dashboard.html に資産曲線比較 + Sharpe/Calmar比較グラフを追加。

## 完了後
全変更を claude/add-trading-backtest-ePJat にプッシュ。
