---
name: "backtest"
description: "YAGAMI改バックテストを実行する。銘柄・期間・ロジックを指定して結果をresults/に保存。"
tags: [backtest, trading, yagami]
trigger: "バックテスト"
---

# YAGAMI改 バックテスト実行スキル

## 概要
バックテストを実行し、OOS期間のPF/WR/MDD/Kellyを報告する。
過学習チェック（IS/OOS比較）も自動で行う。

## データ仕様
- IS期間: 2025-01-01 〜 2025-06-30
- OOS期間: 2025-07-01 〜 2026-02-28
- データ場所: `data/ohlc/{SYMBOL}_{tf}.csv`（大文字銘柄名）
- 利用可能TF: 1m / 15m / 4h / 1d

## 実行手順

### 1. 全銘柄Goldロジックバックテスト
```bash
cd /home/user/sena3fx
python scripts/backtest_all_1m_adaptive.py
```
結果: `results/backtest_all_*.png` / `results/backtest_all_*.csv`

### 2. 新4銘柄バックテスト（NZDUSD/USDJPY/USDCAD/USDCHF）
```bash
python scripts/backtest_new4_1m_adaptive.py
```

### 3. v77ロジックで再テスト（Goldロジック不合格銘柄向け）
```bash
python scripts/backtest_v77_rejects.py
```

### 4. 月次レビュー（本番データと比較）
```bash
python scripts/monthly_review.py --month YYYY-MM --csv production/trade_logs/paper_trades.csv
```
結果: `results/monthly_review_YYYY-MM.png`

## 採用基準（YAGAMI改）
| 指標 | 基準 |
|---|---|
| PF（OOS） | ≥ 2.0 |
| 勝率 | ≥ 65% |
| MDD | ≤ 20% |
| Kelly基準 | ≥ 0.45 |
| OOS PF / IS PF | ≥ 0.7（過学習チェック） |

## ロールバック手順（失敗時）
- データ不足エラー → `data/ohlc/` に対象ファイルが存在するか確認
- KeyError 'open' → `df.columns = [c.lower() for c in df.columns]` を load_csv に追加済み
- DataFrame truth value error → `d4h = df if df is not None else resample(...)` パターンを使用
- 結果が前回と大きく乖離 → IS/OOS分割期間・スプレッド設定を確認

## 注意事項
- USDJPYに1m足データは存在しない（15mで代用）
- `data/` の銘柄名は小文字、`data/ohlc/` は大文字
