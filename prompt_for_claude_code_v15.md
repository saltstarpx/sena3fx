## Claude Code指示書v15：Union戦略のユニバース拡張とADXフィルター検証

### 背景

v14の結果、MetaStrategy v3.0はMDDとCalmarを改善したものの、依然としてUnion_4H_BaseのSharpe(2.817)を超えられませんでした。HMMの複雑なアプローチは一旦保留し、v15では最もシンプルで強力なUnion戦略を横展開するアプローチに切り替えます。

**v15の目的：**
1.  Union戦略がどの商品で有効か（あるいは無効か）を網羅的に検証する。
2.  統計的信頼性を高めるため、トレードサンプル数を増やす。
3.  HMMに代わるシンプルなレジームフィルター（ADX）をテストする。

### Task 1: Union戦略のユニバース拡張バックテスト

**目的:** Union戦略を複数商品に適用し、有効な商品を特定する。

1.  `scripts/`に`backtest_universe.py`を新設。
2.  このスクリプトは、`Union_4H_Base`戦略を以下の商品リストに対して一括でバックテストする。
    - `['XAGUSD', 'XAUUSD', 'NAS100USD', 'US30USD', 'EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD']`
    - 必要なOHLCデータは`data/ohlc/`から読み込む。データがない商品はスキップする。
3.  各商品のバックテスト結果（Sharpe, Calmar, MDD, PF, WR, Trades）を、**新しいCSVファイル`results/universe_performance.csv`** に保存する。

### Task 2: 「承認リスト」の自動生成

**目的:** バックテスト結果に基づき、botが取引すべき「承認済み商品リスト」を動的に作成する仕組みを構築する。

1.  `lib/`に`approved_list.py`を新設。
2.  このスクリプトは`results/universe_performance.csv`を読み込み、以下の条件を満たす商品名のリストを生成して標準出力する。
    - **承認条件:** `Sharpe > 1.0` AND `Trades > 20`
3.  このスクリプトは、将来的に実運用botが起動時に呼び出し、取引対象を決定するために使用する。

### Task 3: ADXフィルターの有効性検証

**目的:** HMMの代替として、シンプルなADXフィルターが「トレンド時のみトレードする」ランチェスター戦略として機能するかを検証する。

1.  `scripts/`に`backtest_adx_filter.py`を新設。
2.  このスクリプトは、`XAGUSD`に対し、`Union_4H_Base`戦略に**追加条件として`ADX(14) > 25`** を加えたバックテストを実行する。
3.  結果を`performance_log.csv`に`Union+ADXFilter`という名前で記録し、ADXフィルターなしの`Union_4H_Base`と比較して、Sharpe/MDD/トレード数がどう変化するかを評価する。

### 完了報告

全タスク完了後、`results/universe_performance.csv`の新設、`performance_log.csv`への追記、および新設スクリプト3点を含む全変更を`claude/add-trading-backtest-ePJat`ブランチにプッシュし、完了報告を行ってください。
