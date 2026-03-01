## Claude Code指示書v12：1000万→1億を目指すための3つの新エンジン実装

### 背景

水原さんの「俺を超えるbot」という目標と、DDを抑制しつつリターンを最大化したいという要望に基づき、Manus AIがクオンツ文献を調査しました。その結果、以下の3つの理論を我々のプロジェクトに統合します。

1.  **ボラティリティ調整サイジング**（DD抑制）
2.  **フラクショナル・ケリー基準**（リターン最大化）
3.  **レジーム転換モデル**（相場適応）

これらを実装し、**「守りを固めながら、勝機に賭ける」** 合理的なbotへと進化させます。

### Task 1: ボラティリティ調整サイジングの実装

**目的:** ATRに基づきポジションサイズを動的に調整し、DDを抑制する。

1.  `lib/risk_manager.py`に`VolatilityAdjustedSizer`クラスを新設。
2.  ATR（期間14）を計算するヘルパー関数を`lib/indicators.py`に追加。
3.  `VolatilityAdjustedSizer`は、基準ポジションサイズを現在のATRで割ることで、最終的なポジションサイズを決定する。
    - `position_size = base_size / (current_atr / avg_atr)`
    - `avg_atr`は過去100期間のATRの平均とする。
4.  `strategies/union_strategy.py`のバックテストにこのサイジングを適用し、MDDがどの程度改善されるかを確認。結果を`performance_log.csv`に記録。

### Task 2: フラクショナル・ケリー基準の実装

**目的:** Union戦略のバックテスト結果から最適投資比率を計算し、リターンを最大化する。

1.  `lib/risk_manager.py`に`KellyCriterionSizer`クラスを新設。
2.  `KellyCriterionSizer`は、バックテスト結果（`performance_log.csv`から読み込み）の勝率(p)とペイオフレシオ(b)を基に、ケリー比率`f*`を計算する。
    - `f* = (b * p - (1 - p)) / b`
3.  安全のため、**フラクショナル・ケリー（係数0.25）** を採用する。
    - `final_position_ratio = f * 0.25`
4.  `strategies/union_strategy.py`のバックテストにこのサイジングを適用し、Sharpe RatioとCalmar Ratioがどう変化するかを確認。結果を`performance_log.csv`に記録。

### Task 3: レジーム転換モデル（HMM）の実装

**目的:** 相場環境を「トレンド」「レンジ」に分類し、最適な戦略を自動で切り替える。

1.  `hmmlearn`ライブラリを`requirements.txt`に追加。
2.  `lib/regime.py`を新設し、`HiddenMarkovModelDetector`クラスを作成。
3.  `HiddenMarkovModelDetector`は、価格データ（日足終値）を学習し、2つの隠れ状態（レジーム0: レンジ、レジーム1: トレンド）を定義する。
    - 各レジームの平均リターンとボラティリティを計算し、ボラティリティが高い方を「トレンド」と定義する。
4.  `strategies/meta_strategy.py`を新設。
5.  `MetaStrategy`は、`HiddenMarkovModelDetector`を使って現在のレジームを判定し、以下のように戦略を切り替える。
    - **レジーム0（レンジ）の場合:** `YagamiA_4H`戦略を実行
    - **レジーム1（トレンド）の場合:** `Union_4H`戦略を実行
6.  この`MetaStrategy`のバックテストを実行し、結果を`performance_log.csv`に記録。

### Task 4: ダッシュボードの拡張

**目的:** 新しいサイジング手法とMetaStrategyのパフォーマンスを可視化する。

1.  `dashboard.html`を更新し、以下の比較グラフを追加。
    - Union戦略 vs Union+VolatilitySizing vs Union+KellySizing の資産曲線比較
    - `MetaStrategy`の資産曲線
    - 各戦略のレジーム別パフォーマンス（トレンド時/レンジ時のPF, WR）

### 完了報告

全タスク完了後、`performance_log.csv`と`dashboard.html`の更新を含む全変更を`claude/add-trading-backtest-ePJat`ブランチにプッシュし、完了報告を行ってください。
