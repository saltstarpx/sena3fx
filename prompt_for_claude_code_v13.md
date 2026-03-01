## Claude Code指示書v13：Kellyの最適化とMetaStrategyの再構築

### 背景

v12の結果、3つの重要な発見がありました。

1.  **Kellyは最強:** ポジションサイズを2.4倍にするだけで最終資産が3倍になった。Calmar 24.5は驚異的。
2.  **VolSizerは重複:** 既存のATRベースSLと機能が重複し、効果が限定的だった。
3.  **MetaStrategyは要改善:** HMMがトレンド相場をほとんど検出できず、弱い戦略を使い続けた。

これらの知見に基づき、v13では「Kellyの最適化」と「MetaStrategyの再構築」に集中します。

### Task 1: Kellyサイザーの最適化（Kelly-Volハイブリッド）

**目的:** Kellyの爆発力とVolSizerのDD抑制能力を両立させる。

1.  `lib/risk_manager.py`に`HybridKellySizer`クラスを新設。
2.  このクラスは、`KellyCriterionSizer`と`VolatilityAdjustedSizer`の両方を内部で呼び出し、2つの乗数を**掛け合わせる**ことで最終的なポジションサイズを決定する。
    - `final_multiplier = kelly_multiplier * volatility_multiplier`
3.  `KellyCriterionSizer`のフラクショナル係数を、`0.25`から`0.5`に引き上げて、より積極的なリターンを狙う。
4.  `strategies/union_strategy.py`のバックテストに`HybridKellySizer`を適用し、結果を`performance_log.csv`に記録。

### Task 2: MetaStrategyの再構築（3状態HMM + レジーム別パラメータ）

**目的:** HMMの精度を向上させ、MetaStrategyを実用的なレベルに引き上げる。

1.  `lib/regime.py`の`HiddenMarkovRegimeDetector`を以下のように改造。
    - 隠れ状態数を`n_states=3`に変更。「高ボラトレンド」「低ボラトレンド」「レンジ」の3状態を定義。
    - ボラティリティとリターンの両方を考慮してレジームをラベリングするロジックに改良。
2.  `strategies/meta_strategy.py`を以下のように改造。
    - 3つのレジームに応じて、実行する戦略と**そのパラメータ**を動的に変更する。
        - **高ボラトレンド:** `Union_4H`（パラメータはv12のまま）
        - **低ボラトレンド:** `Maedai_breakout`（DC期間を短くするなど、より敏感な設定）
        - **レンジ:** `YagamiA_4H`（RSIの閾値を厳しくするなど、より逆張り的な設定）
    - 各レジームに最適なパラメータを探索するため、簡単なグリッドサーチを実行する。
3.  この新しい`MetaStrategy`のバックテストを実行し、結果を`performance_log.csv`に記録。

### Task 3: VolatilityAdjustedSizerの再設計

**目的:** ATR-SLとの重複を解消し、VolSizer本来の価値を発揮させる。

1.  `lib/backtest.py`の`run`メソッドを改造し、`sizer`が指定された場合は、ATR-SLを無効化するオプションを追加。
2.  `VolatilityAdjustedSizer`単体でUnion戦略をバックテストし、ATR-SL無効化時のパフォーマンスを`performance_log.csv`に記録。

### Task 4: ダッシュボードの更新

**目的:** v13の成果を可視化する。

1.  `dashboard.html`を更新し、以下の比較グラフを追加。
    - Union + Kelly vs Union + HybridKelly の資産曲線比較
    - 新旧MetaStrategyの資産曲線比較
    - 3状態HMMのレジーム分布（円グラフ）

### 完了報告

全タスク完了後、`performance_log.csv`と`dashboard.html`の更新を含む全変更を`claude/add-trading-backtest-ePJat`ブランチにプッシュし、完了報告を行ってください。
