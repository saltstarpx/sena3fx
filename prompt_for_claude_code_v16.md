## Claude Code指示書v16：XAUUSDメイン化とリバモア式ピラミッティング

### 背景

v15でXAUUSDがSharpe 1.718, Calmar 5.971, トレード数70件という圧倒的な結果を出しました。v16ではXAUUSDをメイン戦略に昇格させ、リバモア式ピラミッティングを導入して利益の最大化を目指します。

### Task 1: XAUUSD + KellyCriterionSizer バックテスト

**目的:** XAUUSDにKellyサイジングを適用した場合のパフォーマンスを確認する。

1.  `scripts/`に`backtest_xau_kelly.py`を新設。
2.  このスクリプトは、`XAUUSD`に対し、`Union_4H_Base`戦略と`KellyCriterionSizer(f=0.25)`を組み合わせたバックテストを実行する。
3.  結果を`performance_log.csv`に`XAUUSD+Kelly(f=0.25)`という名前で記録する。

### Task 2: リバモア式ピラミッティングの実装

**目的:** 含み益が出ているポジションに機械的に追加投資する`LivermorePyramidingSizer`を実装する。

1.  `lib/risk_manager.py`に`LivermorePyramidingSizer`クラスを新設。
2.  このSizerは、以下のロジックでポジションサイズを決定する。
    - **初期ポジション:** 最初のシグナルでは、接続された他のSizer（例: Kelly）の計算結果をそのまま使う。
    - **追加ポジション:** ポジション保有中に、価格がエントリー価格から`step_pct`（例: 1%）上昇するごとに、追加のポジションを取る。
    - **追加ロット:** 追加ロットは`pyramid_ratios`（例: `[0.5, 0.3, 0.2]`）に従って減少させる。最初の追加が0.5倍、次が0.3倍…となる。
    - **最大追加回数:** `max_pyramids`（例: 3回）で制限する。
3.  `BacktestEngine`を改造し、`on_bar`で`LivermorePyramidingSizer`がポジションを監視し、追加エントリーできるようにする。

### Task 3: XAUUSD + Kelly + ピラミッティング統合バックテスト

**目的:** 全てを統合した最終形態のパフォーマンスを検証する。

1.  `scripts/`に`backtest_xau_final.py`を新設。
2.  このスクリプトは、`XAUUSD`に対し、`Union_4H_Base`戦略に`LivermorePyramidingSizer`を接続し、そのSizerが内部で`KellyCriterionSizer(f=0.25)`を使う構成でバックテストを実行する。
    - `LivermorePyramidingSizer(step_pct=0.01, pyramid_ratios=[0.5, 0.3, 0.2], max_pyramids=3)`
3.  結果を`performance_log.csv`に`XAUUSD+Kelly+Pyramid`という名前で記録する。

### 完了報告

全タスク完了後、`performance_log.csv`への追記、および新設・変更した全ファイルを`claude/add-trading-backtest-ePJat`ブランチにプッシュし、完了報告を行ってください。
