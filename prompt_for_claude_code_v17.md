## Claude Code指示書v17：リバモア式ピラミッティングの最適化とADXフィルター検証

### 背景

v16では、XAUUSD+Kelly(f=0.25)が最終資産+18%という素晴らしい結果を出した一方、リバモア式ピラミッティングは`step_pct=1%`が敏感すぎて機能しませんでした。v17では、このピラミッティングのパラメータを最適化し、XAUUSDにおけるADXフィルターの有効性を最終検証します。

### Task 1: ピラミッティング・パラメータのグリッドサーチ

**目的:** `LivermorePyramidingSizer`の最適なパラメータ（`step_pct`, `max_pyramids`）を発見する。

1.  `scripts/`に`grid_search_pyramid.py`を新設。
2.  このスクリプトは、`XAUUSD`に対し、`Union_4H_Base`戦略と`KellyCriterionSizer(f=0.25)`をベースに、`LivermorePyramidingSizer`のパラメータを変えながらグリッドサーチを実行する。
3.  探索するパラメータの範囲は以下の通りです。
    - `step_pct`: `[0.02, 0.03, 0.04]` (2%, 3%, 4%)
    - `max_pyramids`: `[1, 2]` (最大1回、最大2回)
4.  `pyramid_ratios`はデフォルトの`[0.5, 0.3, 0.2]`を維持してください。
5.  全6パターン（3×2）のバックテスト結果（Sharpe, Calmar, MDD, Trades, Final Equity, 各パラメータ）を`results/pyramid_grid_search.csv`に記録する。

### Task 2: XAUUSDにおけるADXフィルターの有効性検証

**目的:** XAGUSDでは逆効果だったADXフィルター(>25)が、XAUUSDで有効か否かを最終判断する。

1.  `scripts/`に`backtest_xau_adx_filter.py`を新設。
2.  このスクリプトは、`XAUUSD+Kelly(f=0.25)`に対し、ADXフィルターを適用した場合としない場合の2パターンのバックテストを実行する。
    - パターンA: フィルターなし（v16 Task1と同じ）
    - パターンB: `adx_threshold=25` を適用
3.  両方の結果を`performance_log.csv`に`XAUUSD+Kelly(f=0.25)+ADX`のような名前で記録し、比較できるようにする。

### Task 3: `approved_list.py`の更新

**目的:** v16の結果に基づき、承認リスト生成ロジックを更新する。

1.  `lib/approved_list.py`を開き、`generate_approved_list()`関数を修正する。
2.  現在のロジック（Sharpe > 1.0, Trades > 20）を、**Sharpe > 1.5 かつ Calmar > 5.0** という、より厳しい基準に変更する。
3.  これは、XAUUSD（Sharpe 1.7, Calmar 6.5）のような卓越したパフォーマンスを持つ戦略のみを自動承認するための変更です。

### 完了報告

全タスク完了後、`performance_log.csv`と`pyramid_grid_search.csv`への追記、および新設・変更した全ファイルを`claude/add-trading-backtest-ePJat`ブランチにプッシュし、完了報告を行ってください。特にグリッドサーチの結果とADXフィルターの比較結果を明確に報告してください。
