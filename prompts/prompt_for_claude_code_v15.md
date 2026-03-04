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
