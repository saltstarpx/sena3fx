# Claude Code v18 指示書: メイン戦略確定・フォワードモニター刷新

## ブランチ運用ルール（必須）
- **Claude Code の作業ブランチ**: `claude/add-trading-backtest-ePJat`
- **Manus AI のブランチ**: `main`

---

## プロジェクト背景

v17でXAUUSD+Kelly(f=0.25)+ADX(14)>25がSharpe=2.250, Calmar=11.534, MDD=16.1%という
圧倒的な結果を達成。v18ではこれをメイン戦略として確定し、
本番運用体制（フォワードモニター）を整備する。

---

## v18 タスク一覧

### Task 1: 最終戦略 `strategies/main_strategy.py` の作成 ✅
- 銘柄: XAUUSD / 時間足: 4時間足
- 戦略: Union_4H_Base + ADX(14)>25 + KellyCriterionSizer(f=0.25)
- `make_signal_func()`, `run_backtest()`, `get_latest_signal()` を提供
- CLIから直接実行可能 (`python strategies/main_strategy.py`)

### Task 2: フォワードモニター刷新 ✅
- `monitors/monitor_union_kelly.py` を `monitors/forward_main_strategy.py` に置き換え
- `trade_logs/forward_signals.csv` にシグナルを記録 (datetime_utc, symbol, direction, price, adx)
- main_strategy.py の `make_signal_func()` を使用

### Task 3: 実験スクリプト削除 ✅
- `scripts/grid_search_pyramid.py` — 削除 (v17完了)
- `scripts/backtest_xau_final.py` — 削除 (v16完了)
- `scripts/backtest_xau_adx_filter.py` — 削除 (本番昇格済)

### Task 4: ダッシュボード更新 ✅
- `XAUUSD+Kelly+ADX(>25)` を最上部に最優秀戦略として表示
- KPIカード: Sharpe=2.250, Calmar=11.534, MDD=16.1%, WR=61.0%, ¥27,175,560
- 4戦略比較チャート (Union_Base / +Kelly / +Pyramid / +ADX)

---

## 実行結果 (達成済み)

### メイン戦略 確定パラメータ

| 項目 | 値 |
|---|---|
| 銘柄 | XAUUSD |
| 時間足 | 4H |
| シグナル | Union_4H_Base (sig_maedai_yagami_union) |
| フィルター | ADX(14) > 25 |
| サイジング | KellyCriterionSizer(f=0.25) |
| Kelly乗数 | 1.13x |

### 確定バックテスト結果 (2023-10-06 〜 2026-02-27)

| 指標 | 値 | 判定 |
|---|---|---|
| Sharpe Ratio | 2.250 | ✅ |
| Calmar Ratio | 11.534 | ✅ |
| Max Drawdown | 16.1% | ✅ |
| Win Rate | 61.0% | ✅ |
| Trades | 41 | ✅ |
| 最終資産 | ¥27,175,560 (+443.5%) | ✅ |

### 戦略進化サマリー (XAUUSD 4H)

| 戦略 | Sharpe | Calmar | MDD% | 採否 |
|---|---|---|---|---|
| Union_XAUUSD_Base | 1.718 | 5.971 | 23.5 | 参考 |
| +Kelly(f=0.25) | 1.717 | 6.574 | 26.3 | 採用 (v16) |
| +Kelly+Pyramid(LV) | 0.030 | -0.178 | 42.9 | ❌ 不採用 |
| +Kelly+ADX(>25) | **2.250** | **11.534** | **16.1** | ✅ **メイン戦略** |
