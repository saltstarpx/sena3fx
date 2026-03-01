# Claude Code 指示書 v13: Kellyの最適化とMetaStrategyの再構築

## Task 1: Kellyサイザーの最適化 (HybridKellySizer)
lib/risk_manager.py に HybridKellySizer クラスを新設。
KellyCriterionSizer(f=0.5) × VolatilityAdjustedSizer の積算。

## Task 2: MetaStrategy再構築 (3状態HMM)
lib/regime.py を n_states=3 対応に改造。
range(低ボラ)/low_trend(中ボラ+正リターン)/high_trend(高ボラ) の3状態。
strategies/meta_strategy.py を3状態+グリッドサーチ対応に改造。

## Task 3: VolatilityAdjustedSizerの再設計
lib/backtest.py run() に disable_atr_sl=True オプション追加。
ATR-SLとVolSizerの重複解消。

## Task 4: ダッシュボード更新
dashboard.html を更新し:
- Kelly系列資産曲線比較
- v12/v13 Sharpe/Calmar棒グラフ更新
- 3状態HMMレジーム円グラフ追加

## 完了後
claude/add-trading-backtest-ePJat へプッシュ。
