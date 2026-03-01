## Claude Code指示書v14：MetaStrategyの最終進化とUnion+Kellyの実運用準備

### 背景

v13の結果、3つの重要な結論が出ました。

1.  **Union+Kelly(f=0.25)が最高効率:** Calmar 24.5という驚異的な効率を達成。Sharpe 2.798もほぼ最高値。
2.  **MetaStrategy v2.0は大幅改善:** Sharpe 0.581→1.366と改善したが、まだUnion単体に及ばない。
3.  **HMMの課題:** high_trendの検出率が1.7%と低すぎ、Union戦略の出番がほとんどない。

これを踏まえ、v14では「MetaStrategyの最終進化」と「Union+Kellyの実運用準備」の2つに集中します。

### Task 1: MetaStrategyの最終進化（特徴量エンジニアリング）

**目的:** HMMのトレンド検出精度を向上させ、MetaStrategyをUnion単体を超えるレベルに引き上げる。

1.  `lib/regime.py`の`HiddenMarkovRegimeDetector`を以下のように改造。
    - **特徴量エンジニアリング:** HMMの観測変数に、`log_return`と`abs_return`に加え、以下の3つを追加する。
        - **ATR (14):** ボラティリティの指標
        - **ADX (14):** トレンドの強さの指標
        - **RSI (14):** 相場の過熱感の指標
    - これら5次元の観測値でHMMを学習させることで、より高精度なレジーム判定を目指す。
2.  `strategies/meta_strategy.py`で、この新しい5次元HMMを使ったバックテストを実行。high_trendの検出率と、MetaStrategy全体のSharpe/Calmarが改善するかを確認し、結果を`performance_log.csv`に記録。

### Task 2: Union+Kelly(f=0.25)の実運用準備

**目的:** 現時点で最高効率の戦略である`Union+Kelly(f=0.25)`を、いつでも実運用に投入できる状態にする。

1.  `monitors/`ディレクトリに`monitor_union_kelly.py`を新設。
2.  このスクリプトは、OANDA APIに接続し、**リアルタイムでUnion戦略のシグナルを監視**する。
3.  シグナルが発生したら、**取引はせず**、以下の情報をログに出力する。
    - `[Signal] JST 2026-03-03 10:00:00 | XAGUSD | LONG | Price: 5300.0`
    - `[Sizing] Kelly Multiplier: 2.4x | Final Position: 11% of Equity`
4.  この監視スクリプトをデーモンとして実行できるように、`systemd`のサービスファイル（`union_kelly_monitor.service`）のテンプレートを`deploy/`ディレクトリに作成する。

### Task 3: ダッシュボードの最終化

**目的:** v14の成果を可視化し、意思決定をサポートする。

1.  `dashboard.html`を更新し、以下の比較グラフを追加。
    - MetaStrategy v2.0 vs v3.0 (5次元HMM) の資産曲線比較
    - 5次元HMMのレジーム分布（円グラフ）と、各レジームの主要な特徴量（ATR, ADX, RSI）の平均値を示すテーブル。

### 完了報告

全タスク完了後、`performance_log.csv`と`dashboard.html`の更新、および`monitors/`と`deploy/`の新設を含む全変更を`claude/add-trading-backtest-ePJat`ブランチにプッシュし、完了報告を行ってください。
