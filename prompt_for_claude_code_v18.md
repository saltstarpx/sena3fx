## Claude Code指示書v18：最終戦略の確立とフォワードテスト準備

### 背景

v17で**極めて重要な発見**がありました。XAGUSDでは逆効果だったADXフィルター(>25)が、XAUUSDではSharpeを+31%、Calmarを+75%向上させ、MDDを16.1%まで半減させるという驚異的な効果を発揮しました。リバモア式ピラミッティングは全滅したため、これを完全に棄却します。

v18では、この発見に基づき**`XAUUSD+Kelly(f=0.25)+ADX(>25)`**を最終的なメイン戦略として確立し、フォワードテスト（本番監視）の準備を整えます。

### Task 1: 最終戦略 `main_strategy.py` の作成

**目的:** 最終的に確定した戦略ロジックを、単一の実行可能なファイルにまとめる。

1.  `strategies/`ディレクトリに`main_strategy.py`を新設します。
2.  このスクリプトは、以下の確定ロジックを実装します。
    - **銘柄:** `XAUUSD`
    - **時間足:** 4時間足
    - **戦略:** `Union_4H_Base` (DCブレイク or RSI反発 + EMA200フィルター)
    - **フィルター:** `ADX(14) > 25`
    - **サイジング:** `KellyCriterionSizer(f=0.25)`
3.  スクリプトはコマンドラインから実行可能とし、バックテスト結果（Sharpe, Calmar, MDD, 最終資産など）を標準出力するようにしてください。これは将来的な本番環境での動作確認を容易にするためです。

### Task 2: フォワードテスト監視スクリプトの更新

**目的:** リアルタイム監視システムを、v18で確定した最終戦略に対応させる。

1.  `monitors/monitor_union_kelly.py`を`monitors/forward_main_strategy.py`にリネームします。
2.  新しい監視スクリプトは、Task 1で作成した`strategies/main_strategy.py`のロジックを呼び出すように変更します。
3.  OANDA APIから`XAUUSD`の4H足データを取得し、`XAUUSD+Kelly+ADX`のシグナルが発生した際に、`trade_logs/forward_signals.csv`にシグナル内容（日時, 銘柄, 方向, 価格）を記録するようにしてください。

### Task 3: 不要な実験ファイルの削除

**目的:** 役目を終えた実験用スクリプトを削除し、リポジトリをクリーンに保つ。

1.  以下の不要になったスクリプトをリポジトリから削除してください。
    - `scripts/grid_search_pyramid.py`
    - `scripts/backtest_xau_final.py`
    - `scripts/backtest_xau_adx_filter.py`

### Task 4: ダッシュボードの更新

**目的:** 最新の最良戦略の結果をダッシュボードに反映させる。

1.  `dashboard.html`を更新し、`XAUUSD+Kelly(f=0.25)+ADX(>25)`のバックテスト結果（Sharpe 2.250, Calmar 11.534, MDD 16.1%）を最も目立つ場所に「**Main Strategy**」として表示してください。
2.  過去の戦略（Union Base, MetaStrategyなど）は比較対象として残しつつも、新しいメイン戦略が最高性能であることが一目でわかるようにレイアウトを調整してください。

### 完了報告

全タスク完了後、変更した全ファイルを`claude/add-trading-backtest-ePJat`ブランチにプッシュし、完了報告を行ってください。特に、`main_strategy.py`が単体で動作すること、フォワードテスト監視が新ロジックに対応したことを報告してください。
