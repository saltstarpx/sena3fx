<<<<<<< HEAD
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
=======
# Claude Code v15 指示書: ユニバース拡張とADXフィルター検証

## ブランチ運用ルール（必須）
- **Claude Code の作業ブランチ**: `claude/add-trading-backtest-ePJat`
- **Manus AI のブランチ**: `main`
- Claude Code は `main` に直接 push できない（403エラー）

---

## プロジェクト背景

v14の結果:
- MetaStrategy v3.0 (5D HMM): MDD14.4%, Calmar3.158 — 目標達成
- Union_4H_Base: Sharpe=2.817, Calmar=13.7 — 依然として最強
- HMMアプローチはリスク削減に有効だが、Sharpeではベースラインに届かない

v15の方針: HMMの複雑さを保留し、Union戦略の「ユニバース拡張」と「シンプルフィルター」を検証。

---

## v15 タスク一覧

### Task 1: Union戦略のユニバース拡張バックテスト

**対象商品:**
`['XAGUSD', 'XAUUSD', 'NAS100USD', 'US30USD', 'EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD']`

**新設:** `scripts/backtest_universe.py`
- `data/ohlc/` に4Hデータがない商品はスキップ
- 結果を `results/universe_performance.csv` に保存
- `results/performance_log.csv` に追記

### Task 2: 承認リストの自動生成

**新設:** `lib/approved_list.py`
- `results/universe_performance.csv` を読み込み
- 承認条件: `Sharpe > 1.0` AND `Trades > 20`
- 標準出力 / `--save` でJSON保存
- `load_approved()` 関数でモジュールとして使用可能

### Task 3: ADXフィルター有効性検証

**新設:** `scripts/backtest_adx_filter.py`
- 対象: XAGUSD
- Union_4H_Base vs Union+ADX(14)>25 比較
- `performance_log.csv` に `Union+ADXFilter` として追記

---

## 実行結果 (達成済み)

### ユニバースバックテスト結果
| 商品 | Sharpe | Calmar | MDD% | PF | WR% | Trades |
|---|---|---|---|---|---|---|
| XAUUSD | 1.718 | 5.971 | 23.5 | 1.828 | 50.0 | 70 |
| XAGUSD | 1.189 | 2.111 | 25.3 | 1.611 | 57.6 | 33 |
| NAS100USD〜AUDUSD | — | — | — | — | — | データなし |

### 承認商品
- XAUUSD ✅ (Sharpe=1.718 > 1.0, Trades=70 > 20)
- XAGUSD ✅ (Sharpe=1.189 > 1.0, Trades=33 > 20)

### ADXフィルター結果 (XAGUSD)
| 戦略 | Sharpe | MDD% | Trades |
|---|---|---|---|
| Union_XAGUSD_Base | 1.189 | 25.3 | 33 |
| Union+ADX(>25) | 1.008 | 21.8 | 21 |
- 判定: 非推奨 (Sharpe低下 -0.181, Trades -36%)

---

## 完了後のコミット・プッシュ

```bash
git add -A
git commit -m "v15: ユニバース拡張 + ADXフィルター検証"
git push -u origin claude/add-trading-backtest-ePJat
```
>>>>>>> origin/claude/add-trading-backtest-ePJat
